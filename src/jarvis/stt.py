"""Speech-to-text engines behind one ``transcribe`` interface."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from jarvis.audio import SAMPLE_RATE, Float32
from jarvis.models import PARAKEET_DIR


class Stt(Protocol):
    name: str

    def transcribe(self, audio: Float32) -> str:
        """``audio``: mono float32 at 16 kHz."""
        ...


class ParakeetStt:
    """NVIDIA Parakeet TDT 0.6B v3 (int8) through sherpa-onnx's offline transducer."""

    name = "parakeet-tdt-0.6b-v3-int8"

    def __init__(self, models_dir: Path, threads: int = 4) -> None:
        import sherpa_onnx

        d = models_dir / PARAKEET_DIR
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(d / "encoder.int8.onnx"),
            decoder=str(d / "decoder.int8.onnx"),
            joiner=str(d / "joiner.int8.onnx"),
            tokens=str(d / "tokens.txt"),
            num_threads=threads,
            model_type="nemo_transducer",
            decoding_method="greedy_search",
        )

    def transcribe(self, audio: Float32) -> str:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        self._recognizer.decode_stream(stream)
        return str(stream.result.text).strip()


class WhisperStt:
    """faster-whisper (CTranslate2, int8 on CPU). Comparator only; weights fetched from
    Hugging Face into ``models/whisper`` on first use."""

    def __init__(self, models_dir: Path, size: str = "small", threads: int = 4) -> None:
        from faster_whisper import WhisperModel

        self.name = f"faster-whisper-{size}-int8"
        self._model = WhisperModel(
            size,
            device="cpu",
            compute_type="int8",
            cpu_threads=threads,
            download_root=str(models_dir / "whisper"),
        )

    def transcribe(self, audio: Float32) -> str:
        segments, _info = self._model.transcribe(
            audio, language="en", beam_size=1, vad_filter=False, condition_on_previous_text=False
        )
        return " ".join(s.text.strip() for s in segments).strip()
