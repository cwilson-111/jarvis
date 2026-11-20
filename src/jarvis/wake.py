"""Wake-word detection.

openWakeWord's pre-trained ``hey_jarvis_v0.1`` model is used as-is: a free
(non-commercial, CC BY-NC-SA 4.0), key-less, pre-trained "hey jarvis" model that
runs on CPU in ~2 ms per 80 ms frame. It was trained largely on synthetic speech,
which is also how this project evaluates it (see docs/RESULTS.md for what that
does and does not show).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from jarvis.audio import Int16

# Transcripts of a run-on "Hey Jarvis, what's ..." start with the wake phrase;
# STT sometimes hears "Jarvis" as a near-homophone.
_WAKE_PREFIX = re.compile(
    r"^\W*(?:(?:hey|hay|hi|a|okay|ok)\W+)?(?:jarvis|jarvi|jervis|javis|travis|garvis)\b\W*",
    re.IGNORECASE,
)


def strip_wake_phrase(text: str) -> str:
    return _WAKE_PREFIX.sub("", text, count=1).strip()


class WakeWord(Protocol):
    name: str

    def score(self, frame: Int16) -> float:
        """Probability-like score in [0, 1] for this 80 ms frame."""
        ...

    def reset(self) -> None: ...


class OpenWakeWord:
    def __init__(self, models_dir: Path, model: str) -> None:
        from openwakeword.model import Model

        model_path = models_dir / model
        self.name = model_path.stem
        self._model = Model(
            wakeword_models=[str(model_path)],
            inference_framework="onnx",
            melspec_model_path=str(models_dir / "openwakeword" / "melspectrogram.onnx"),
            embedding_model_path=str(models_dir / "openwakeword" / "embedding_model.onnx"),
        )

    def score(self, frame: Int16) -> float:
        scores = self._model.predict(frame)
        return float(scores[self.name])

    def reset(self) -> None:
        self._model.reset()


class NoWakeWord:
    """Used with push-to-talk: never fires."""

    name = "none"

    def score(self, frame: Int16) -> float:
        return 0.0

    def reset(self) -> None:
        return None
