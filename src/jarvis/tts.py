"""Text-to-speech engines behind one ``synthesize`` interface."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

import numpy as np

from jarvis.audio import Float32


class Tts(Protocol):
    name: str

    def synthesize(self, text: str) -> tuple[Float32, int]:
        """Return (mono float32 audio, sample rate)."""
        ...


class KokoroTts:
    """Kokoro-82M v1.0, int8 ONNX, via kokoro-onnx (espeak-ng phonemizer bundled).

    All synthesis runs on one long-lived worker thread. The phonemizer sets up
    espeak-ng per calling thread, which costs ~500 ms on an M2; since every reply
    runs in a fresh thread, calling Kokoro directly would add that to every
    reply's first sentence (measured; see README "Design decisions").
    """

    def __init__(
        self, models_dir: Path, voice: str = "am_michael", speed: float = 1.0, lang: str = "en-us"
    ) -> None:
        from kokoro_onnx import Kokoro

        self._kokoro = Kokoro(
            str(models_dir / "kokoro" / "kokoro-v1.0.int8.onnx"),
            str(models_dir / "kokoro" / "voices-v1.0.bin"),
        )
        voices = set(self._kokoro.get_voices())
        if voice not in voices:
            raise ValueError(f"unknown Kokoro voice {voice!r}; available: {sorted(voices)}")
        self.voice = voice
        self.speed = speed
        self.lang = lang
        self.name = f"kokoro-82m-v1.0-int8:{voice}"
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kokoro")

    @property
    def voices(self) -> list[str]:
        return sorted(self._kokoro.get_voices())

    def synthesize(
        self, text: str, voice: str | None = None, speed: float | None = None
    ) -> tuple[Float32, int]:
        return self._worker.submit(
            self._create, text, voice or self.voice, speed or self.speed
        ).result()

    def _create(self, text: str, voice: str, speed: float) -> tuple[Float32, int]:
        audio, rate = self._kokoro.create(text, voice=voice, speed=speed, lang=self.lang)
        return np.asarray(audio, dtype=np.float32), int(rate)
