"""Audio sources (mic, file) and sinks (speakers, file).

The pipeline consumes 16 kHz mono int16 frames of 80 ms (1280 samples): the
native frame size of openWakeWord, and an integer number of Silero chunks once
buffered. Sources and sinks share small protocols so the same ``Assistant`` loop
serves live use, file replay, tests and benchmarks.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 1280  # 80 ms
FRAME_S = FRAME_SAMPLES / SAMPLE_RATE

Int16 = npt.NDArray[np.int16]
Float32 = npt.NDArray[np.float32]


def resample(audio: Float32, src_rate: int, dst_rate: int) -> Float32:
    if src_rate == dst_rate:
        return audio.astype(np.float32, copy=False)
    g = np.gcd(src_rate, dst_rate)
    out = resample_poly(audio, dst_rate // g, src_rate // g)
    return np.asarray(out, dtype=np.float32)


def load_audio(path: Path, rate: int = SAMPLE_RATE) -> Float32:
    """Read any soundfile-supported file as mono float32 in [-1, 1] at ``rate``."""
    data, src_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1).astype(np.float32)
    return resample(mono, int(src_rate), rate)


def to_int16(audio: Float32) -> Int16:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)


def to_float32(pcm: Int16) -> Float32:
    return (pcm.astype(np.float32) / 32768.0).astype(np.float32)


# --------------------------------------------------------------------------- sources


class AudioSource(Protocol):
    def frames(self) -> Iterator[Int16]:
        """Yield FRAME_SAMPLES-long int16 frames until the source ends."""
        ...

    def close(self) -> None: ...


class FileSource:
    """Replays a file as if it were a microphone.

    ``realtime=True`` releases frames at the audio rate, so wall-clock timestamps
    (end of speech, endpointing wait) mean the same thing as in live use.
    ``realtime=False`` is for tests: as fast as possible, timings meaningless.
    Trailing silence is appended so the endpointer can close the last utterance.
    """

    def __init__(
        self, audio: Float32 | Path, *, realtime: bool = True, tail_silence_s: float = 2.0
    ) -> None:
        samples = load_audio(audio) if isinstance(audio, Path) else audio
        pcm = to_int16(samples)
        tail = np.zeros(int(tail_silence_s * SAMPLE_RATE), dtype=np.int16)
        pcm = np.concatenate([pcm, tail])
        pad = (-len(pcm)) % FRAME_SAMPLES
        self._pcm = np.concatenate([pcm, np.zeros(pad, dtype=np.int16)])
        self.realtime = realtime
        self.duration_s = len(self._pcm) / SAMPLE_RATE

    def frames(self) -> Iterator[Int16]:
        start = time.perf_counter()
        for i, offset in enumerate(range(0, len(self._pcm), FRAME_SAMPLES)):
            if self.realtime:
                # Frame i is "captured" once its last sample has been spoken.
                due = start + (i + 1) * FRAME_S
                delay = due - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            yield self._pcm[offset : offset + FRAME_SAMPLES]

    def close(self) -> None:
        return None


class MicSource:
    """Default input device via PortAudio, 16 kHz mono int16."""

    def __init__(self, device: int | str | None = None) -> None:
        import sounddevice as sd

        self._queue: queue.Queue[Int16] = queue.Queue()
        self._closed = threading.Event()
        self.overflows = 0

        def callback(indata: Int16, _frames: int, _time: object, status: object) -> None:
            if status:
                self.overflows += 1
            self._queue.put(indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
            device=device,
            callback=callback,
        )
        self._stream.start()

    def frames(self) -> Iterator[Int16]:
        while not self._closed.is_set():
            try:
                yield self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

    def close(self) -> None:
        self._closed.set()
        self._stream.stop()
        self._stream.close()


# ----------------------------------------------------------------------------- sinks


class AudioSink(Protocol):
    def begin(self, on_first_audio: Callable[[float], None]) -> None:
        """Start a reply; ``on_first_audio(t)`` fires when its first sample is emitted."""
        ...

    def play(self, audio: Float32, sample_rate: int) -> None:
        """Queue a chunk (non-blocking)."""
        ...

    def wait_done(self, cancelled: threading.Event) -> None:
        """Block until queued audio has finished playing or ``cancelled`` is set."""
        ...

    def stop(self) -> None:
        """Drop everything queued and silence output immediately."""
        ...

    def started_chunks(self) -> int:
        """How many chunks of the current reply have started playing (were heard)."""
        ...

    @property
    def playing(self) -> bool: ...

    def close(self) -> None: ...


class FileSink:
    """Collects reply audio (optionally to a wav) and simulates a playback clock.

    Playback is modelled, not performed: chunk *k* starts at
    ``max(enqueue_time, end of chunk k-1)``. That gives the same first-audio
    time a real device would (minus device buffer latency) and exposes stalls,
    where the TTS fell behind playback, as ``stall_s``.
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        realtime: bool = False,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.path = path
        self.realtime = realtime
        self.clock = clock
        self.sample_rate: int | None = None
        self._chunks: list[Float32] = []
        self._on_first: Callable[[float], None] | None = None
        self._sim_end = 0.0
        self._started = False
        self._starts: list[float] = []  # simulated start time of each chunk this reply
        self.stall_s = 0.0
        self._lock = threading.Lock()

    def begin(self, on_first_audio: Callable[[float], None]) -> None:
        with self._lock:
            if self._chunks and self.sample_rate is not None:  # pause between replies
                self._chunks.append(np.zeros(self.sample_rate // 2, dtype=np.float32))
            self._on_first = on_first_audio
            self._started = False
            self._starts = []
            self.stall_s = 0.0

    def play(self, audio: Float32, sample_rate: int) -> None:
        now = self.clock()
        with self._lock:
            if self.sample_rate is None:
                self.sample_rate = sample_rate
            elif sample_rate != self.sample_rate:
                audio = resample(audio, sample_rate, self.sample_rate)
            if not self._started:
                self._started = True
                self._sim_end = now
                if self._on_first is not None:
                    self._on_first(now)
            elif now > self._sim_end:
                self.stall_s += now - self._sim_end
            start = max(now, self._sim_end)
            self._sim_end = start + len(audio) / sample_rate
            self._chunks.append(audio)
            self._starts.append(start)

    @property
    def playback_end(self) -> float:
        return self._sim_end

    def wait_done(self, cancelled: threading.Event) -> None:
        if self.realtime:
            while not cancelled.is_set():
                remaining = self._sim_end - self.clock()
                if remaining <= 0:
                    break
                cancelled.wait(min(remaining, 0.05))

    def stop(self) -> None:
        """Cut the simulated playback now and drop the unplayed tail from the recording."""
        with self._lock:
            now = self.clock()
            if self.sample_rate is not None and self._sim_end > now:
                excess = int((self._sim_end - now) * self.sample_rate)
                while excess > 0 and self._chunks:
                    last = self._chunks[-1]
                    if len(last) <= excess:
                        excess -= len(last)
                        self._chunks.pop()
                    else:
                        self._chunks[-1] = last[: len(last) - excess]
                        excess = 0
            self._sim_end = min(self._sim_end, now)
            self._starts = [s for s in self._starts if s <= now]

    def started_chunks(self) -> int:
        now = self.clock()
        return sum(1 for s in self._starts if s <= now)

    @property
    def playing(self) -> bool:
        return self._started and self.clock() < self._sim_end

    def close(self) -> None:
        if self.path is None or not self._chunks or self.sample_rate is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(self.path), np.concatenate(self._chunks), self.sample_rate)


class SpeakerSink:
    """Plays through the default output device using a callback-driven stream.

    ``stop()`` empties the queue under the lock the audio callback uses, so
    barge-in silences the speaker within one block (~20 ms).
    """

    BLOCK = 480  # 20 ms at 24 kHz

    def __init__(self, sample_rate: int = 24_000, device: int | str | None = None) -> None:
        import sounddevice as sd

        self.sample_rate = sample_rate
        self._chunks: deque[Float32] = deque()
        self._pos = 0
        self._lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self._on_first: Callable[[float], None] | None = None
        self._fired = True
        self._started_chunks = 0

        def callback(outdata: Float32, frames: int, _time: object, _status: object) -> None:
            written = 0
            fire = False
            with self._lock:
                while written < frames and self._chunks:
                    chunk = self._chunks[0]
                    if self._pos == 0:
                        self._started_chunks += 1
                    take = min(frames - written, len(chunk) - self._pos)
                    outdata[written : written + take, 0] = chunk[self._pos : self._pos + take]
                    written += take
                    self._pos += take
                    if self._pos >= len(chunk):
                        self._chunks.popleft()
                        self._pos = 0
                if written and not self._fired:
                    self._fired = True
                    fire = True
                if not self._chunks:
                    self._idle.set()
            outdata[written:, 0] = 0.0
            if fire and self._on_first is not None:
                self._on_first(time.perf_counter())

        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.BLOCK,
            device=device,
            callback=callback,
        )
        self.output_latency_s = float(self._stream.latency)
        self._stream.start()

    def begin(self, on_first_audio: Callable[[float], None]) -> None:
        with self._lock:
            self._on_first = on_first_audio
            self._fired = False
            self._started_chunks = 0

    def play(self, audio: Float32, sample_rate: int) -> None:
        if sample_rate != self.sample_rate:
            audio = resample(audio, sample_rate, self.sample_rate)
        with self._lock:
            self._chunks.append(audio.astype(np.float32, copy=False))
            self._idle.clear()

    def wait_done(self, cancelled: threading.Event) -> None:
        while not cancelled.is_set():
            if self._idle.wait(0.05):
                return

    def stop(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._pos = 0
            self._idle.set()

    def started_chunks(self) -> int:
        return self._started_chunks

    @property
    def playing(self) -> bool:
        return not self._idle.is_set()

    def close(self) -> None:
        self.stop()
        self._stream.stop()
        self._stream.close()
