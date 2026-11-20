"""Deterministic stand-ins for every engine: no models, mic, network or Ollama."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator

import numpy as np

from jarvis.audio import FRAME_SAMPLES, SAMPLE_RATE, Float32, Int16
from jarvis.llm import LlmError, Message


class FakeLlm:
    name = "fake-llm"

    def __init__(
        self, tokens: list[str], delay_s: float = 0.0, fail_after: int | None = None
    ) -> None:
        self.tokens = tokens
        self.delay_s = delay_s
        self.fail_after = fail_after
        self.calls: list[list[Message]] = []
        self.yielded = 0
        self.closed = threading.Event()
        self.last_stats: dict[str, object] = {}

    def stream(self, messages: list[Message], cancel: threading.Event) -> Iterator[str]:
        self.calls.append(messages)
        try:
            for i, token in enumerate(self.tokens):
                if self.fail_after is not None and i >= self.fail_after:
                    raise LlmError("backend went away")
                if cancel.is_set():
                    return
                time.sleep(self.delay_s)
                self.yielded += 1
                yield token
        finally:
            self.closed.set()


class FakeTts:
    name = "fake-tts"
    rate = 1000

    def __init__(self, delay_s: float = 0.0, seconds_per_char: float = 0.01) -> None:
        self.delay_s = delay_s
        self.seconds_per_char = seconds_per_char
        self.texts: list[str] = []

    def synthesize(self, text: str) -> tuple[Float32, int]:
        self.texts.append(text)
        time.sleep(self.delay_s)
        n = max(1, int(len(text) * self.seconds_per_char * self.rate))
        return np.full(n, 0.1, dtype=np.float32), self.rate


class RecordingSink:
    """Plays instantly; records chunks and stop() calls."""

    def __init__(self) -> None:
        self.chunks: list[Float32] = []
        self.stops = 0
        self._on_first: Callable[[float], None] | None = None
        self._fired = False

    def begin(self, on_first_audio: Callable[[float], None]) -> None:
        self._on_first = on_first_audio
        self._fired = False

    def play(self, audio: Float32, sample_rate: int) -> None:
        self.chunks.append(audio)
        if not self._fired and self._on_first is not None:
            self._fired = True
            self._on_first(time.perf_counter())

    def wait_done(self, cancelled: threading.Event) -> None:
        return None

    def stop(self) -> None:
        self.stops += 1

    def started_chunks(self) -> int:
        return len(self.chunks)

    @property
    def playing(self) -> bool:
        return False

    def close(self) -> None:
        return None


class ScriptedWake:
    """Fires (score 1.0) on the given frame indices."""

    name = "scripted-wake"

    def __init__(self, fire_on: set[int]) -> None:
        self.fire_on = fire_on
        self.frame = -1

    def score(self, frame: Int16) -> float:
        self.frame += 1
        return 1.0 if self.frame in self.fire_on else 0.0

    def reset(self) -> None:
        return None


class EnergyVad:
    """Speech iff the frame is not silent."""

    def probability(self, frame: Int16) -> float:
        return 1.0 if np.abs(frame).max() > 100 else 0.0

    def reset(self) -> None:
        return None


class FixedStt:
    name = "fixed-stt"

    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.calls = 0

    def transcribe(self, audio: Float32) -> str:
        text = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return text


def frames_audio(pattern: str) -> Float32:
    """Build audio from a frame pattern: '.' = 80 ms silence, 's' = 80 ms "speech"."""
    parts = [np.full(FRAME_SAMPLES, 0.3 if ch == "s" else 0.0, dtype=np.float32) for ch in pattern]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


SECONDS_PER_FRAME = FRAME_SAMPLES / SAMPLE_RATE
