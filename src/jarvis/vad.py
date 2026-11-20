"""Voice activity detection (Silero VAD v5, ONNX) and utterance endpointing.

``SileroVad`` turns 80 ms frames into a speech probability. ``Endpointer`` is
a pure state machine over those probabilities, so its timing rules are unit
tested without any model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

import numpy as np

from jarvis.audio import FRAME_S, SAMPLE_RATE, Float32, Int16, to_float32
from jarvis.config import VadConfig


class Vad(Protocol):
    def probability(self, frame: Int16) -> float: ...

    def reset(self) -> None: ...


class SileroVad:
    """Silero VAD v5 via onnxruntime: 512-sample chunks plus 64 samples of context."""

    CHUNK = 512
    CONTEXT = 64

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.CONTEXT), dtype=np.float32)
        self._pending: Float32 = np.zeros(0, dtype=np.float32)

    def probability(self, frame: Int16) -> float:
        """Max speech probability over the Silero chunks completed by this frame."""
        audio = np.concatenate([self._pending, to_float32(frame)])
        best = 0.0
        n_full = len(audio) // self.CHUNK
        for i in range(n_full):
            chunk = audio[i * self.CHUNK : (i + 1) * self.CHUNK][None, :]
            x = np.concatenate([self._context, chunk], axis=1)
            out, self._state = self._session.run(
                None, {"input": x, "state": self._state, "sr": self._sr}
            )
            self._context = x[:, -self.CONTEXT :]
            best = max(best, float(out[0, 0]))
        self._pending = audio[n_full * self.CHUNK :]
        return best


class Phase(Enum):
    WAITING = "waiting"  # no speech yet
    SPEECH_START = "speech_start"  # this frame confirmed the start of speech
    SPEAKING = "speaking"
    ENDPOINT = "endpoint"  # enough trailing silence: utterance complete
    TIMEOUT = "timeout"  # nobody spoke


@dataclass
class EndpointResult:
    phase: Phase
    is_speech: bool


class Endpointer:
    """Frame-count state machine over VAD probabilities.

    * speech starts after ``min_speech_ms`` of consecutive speech frames;
    * the utterance ends after ``end_silence_ms`` of consecutive non-speech
      frames following speech (or at ``max_utterance_s``);
    * ``no_speech_timeout_s`` without speech gives up.
    """

    def __init__(self, cfg: VadConfig, frame_s: float = FRAME_S) -> None:
        self.threshold = cfg.threshold
        self.start_frames = max(1, math.ceil(cfg.min_speech_ms / 1000.0 / frame_s - 1e-9))
        self.end_frames = max(1, math.ceil(cfg.end_silence_ms / 1000.0 / frame_s - 1e-9))
        self.max_frames = math.ceil(cfg.max_utterance_s / frame_s)
        self.timeout_frames = math.ceil(cfg.no_speech_timeout_s / frame_s)
        self.reset()

    def reset(self) -> None:
        self.frames = 0
        self.started = False
        self._run = 0  # consecutive speech frames (before start)
        self._silence = 0  # consecutive silence frames (after start)

    def update(self, prob: float) -> EndpointResult:
        self.frames += 1
        speech = prob >= self.threshold
        if not self.started:
            self._run = self._run + 1 if speech else 0
            if self._run >= self.start_frames:
                self.started = True
                return EndpointResult(Phase.SPEECH_START, speech)
            if self.frames >= self.timeout_frames:
                return EndpointResult(Phase.TIMEOUT, speech)
            return EndpointResult(Phase.WAITING, speech)
        self._silence = 0 if speech else self._silence + 1
        if self._silence >= self.end_frames or self.frames >= self.max_frames:
            return EndpointResult(Phase.ENDPOINT, speech)
        return EndpointResult(Phase.SPEAKING, speech)
