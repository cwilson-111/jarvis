"""Construct concrete engines from config. The only module that knows which
implementation backs each interface."""

from __future__ import annotations

from jarvis.config import Config
from jarvis.llm import OllamaLlm
from jarvis.stt import ParakeetStt, Stt, WhisperStt
from jarvis.tts import KokoroTts
from jarvis.vad import SileroVad
from jarvis.wake import NoWakeWord, OpenWakeWord, WakeWord


def build_wake(cfg: Config) -> WakeWord:
    if cfg.wake.engine == "none":
        return NoWakeWord()
    return OpenWakeWord(cfg.models_dir, cfg.wake.model)


def build_vad(cfg: Config) -> SileroVad:
    return SileroVad(cfg.models_dir / "silero_vad.onnx")


def build_stt(cfg: Config) -> Stt:
    if cfg.stt.engine == "whisper":
        return WhisperStt(cfg.models_dir, cfg.stt.whisper_model, cfg.stt.threads)
    return ParakeetStt(cfg.models_dir, cfg.stt.threads)


def build_tts(cfg: Config) -> KokoroTts:
    return KokoroTts(cfg.models_dir, cfg.tts.voice, cfg.tts.speed, cfg.tts.lang)


def build_llm(cfg: Config) -> OllamaLlm:
    return OllamaLlm(cfg.llm)
