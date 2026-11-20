"""Endpointing rules, wake-phrase stripping and the file source/sink."""

import threading

import numpy as np
import pytest

from jarvis.audio import FRAME_SAMPLES, FileSink, FileSource
from jarvis.config import VadConfig
from jarvis.vad import Endpointer, Phase
from jarvis.wake import strip_wake_phrase


def phases(probs: list[float], cfg: VadConfig | None = None) -> list[Phase]:
    ep = Endpointer(cfg or VadConfig())
    return [ep.update(p).phase for p in probs]


def test_speech_start_needs_min_speech_frames() -> None:
    # 160 ms min speech = 2 frames of 80 ms; a single blip does not start speech.
    got = phases([0.9, 0.1, 0.9, 0.9])
    assert got == [Phase.WAITING, Phase.WAITING, Phase.WAITING, Phase.SPEECH_START]


def test_endpoint_after_end_silence() -> None:
    # 600 ms of silence = ceil(7.5) = 8 frames.
    got = phases([0.9, 0.9] + [0.1] * 8)
    assert got[1] is Phase.SPEECH_START
    assert got[2:9] == [Phase.SPEAKING] * 7
    assert got[9] is Phase.ENDPOINT


def test_pause_shorter_than_end_silence_does_not_end_utterance() -> None:
    got = phases([0.9, 0.9] + [0.1] * 5 + [0.9] + [0.1] * 8)
    assert Phase.ENDPOINT not in got[:-1]
    assert got[-1] is Phase.ENDPOINT


def test_timeout_without_speech_and_max_length() -> None:
    cfg = VadConfig(no_speech_timeout_s=0.4, max_utterance_s=0.8)
    assert phases([0.0] * 5, cfg)[-1] is Phase.TIMEOUT
    assert phases([0.9] * 10, cfg)[-1] is Phase.ENDPOINT


def test_threshold_is_inclusive() -> None:
    cfg = VadConfig(threshold=0.5)
    assert phases([0.5, 0.5], cfg)[-1] is Phase.SPEECH_START


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("Hey Jarvis, what's the weather?", "what's the weather?"),
        ("hey jarvis what time is it", "what time is it"),
        ("Jarvis. Turn it up.", "Turn it up."),
        ("Hey, Jervis! Tell me a joke.", "Tell me a joke."),
        ("Hey Jarvis.", ""),
        ("What does Jarvis stand for?", "What does Jarvis stand for?"),
    ],
)
def test_strip_wake_phrase(raw: str, clean: str) -> None:
    assert strip_wake_phrase(raw) == clean


def test_file_source_frames_and_tail() -> None:
    audio = np.zeros(FRAME_SAMPLES * 3 + 10, dtype=np.float32)
    source = FileSource(audio, realtime=False, tail_silence_s=0.16)
    frames = list(source.frames())
    assert all(len(f) == FRAME_SAMPLES and f.dtype == np.int16 for f in frames)
    # 3 full frames + 10 samples + 2 frames of tail, padded to whole frames.
    assert len(frames) == 6


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_file_sink_simulates_playback_and_stalls() -> None:
    clock = Clock()
    sink = FileSink(None, clock=clock)
    first: list[float] = []
    sink.begin(first.append)
    clock.t = 1.0
    sink.play(np.zeros(1000, dtype=np.float32), 1000)  # 1 s chunk, plays 1.0 -> 2.0
    clock.t = 1.5
    sink.play(np.zeros(500, dtype=np.float32), 1000)  # queued: plays 2.0 -> 2.5
    clock.t = 3.0
    sink.play(np.zeros(500, dtype=np.float32), 1000)  # TTS was late: 0.5 s stall
    assert first == [1.0]
    assert sink.stall_s == pytest.approx(0.5)
    assert sink.playback_end == pytest.approx(3.5)
    assert sink.playing
    clock.t = 4.0
    assert not sink.playing


def test_file_sink_stop_truncates_unplayed_audio() -> None:
    clock = Clock()
    sink = FileSink(None, clock=clock)
    sink.begin(lambda _t: None)
    sink.play(np.ones(1000, dtype=np.float32), 1000)
    sink.play(np.ones(1000, dtype=np.float32), 1000)
    clock.t = 0.25
    sink.stop()
    assert sum(len(c) for c in sink._chunks) == 250
    assert sink.playback_end == 0.25
    cancelled = threading.Event()
    sink.wait_done(cancelled)  # non-realtime: returns immediately


def test_file_sink_counts_chunks_that_started_playing() -> None:
    clock = Clock()
    sink = FileSink(None, clock=clock)
    sink.begin(lambda _t: None)
    sink.play(np.ones(1000, dtype=np.float32), 1000)  # 0 -> 1 s
    sink.play(np.ones(1000, dtype=np.float32), 1000)  # 1 -> 2 s
    sink.play(np.ones(1000, dtype=np.float32), 1000)  # 2 -> 3 s
    clock.t = 1.5
    assert sink.started_chunks() == 2
    sink.stop()
    clock.t = 10.0
    assert sink.started_chunks() == 2  # stopped chunks never start
