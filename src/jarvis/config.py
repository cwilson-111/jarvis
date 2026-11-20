"""Typed configuration loaded from an optional TOML file plus CLI overrides.

Unknown keys are an error rather than silently ignored: a typo such as
``modle = "qwen2.5:7b"`` must not leave the assistant running a different model
than the user believes it is.
"""

from __future__ import annotations

import dataclasses
import tomllib
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

DEFAULT_SYSTEM_PROMPT = (
    "You are Jarvis, a helpful voice assistant running entirely on the user's computer. "
    "Your replies are converted to speech, so answer in one to three short, natural "
    "sentences of plain spoken English. Never use markdown, bullet points, emoji, code "
    "or URLs. Say numbers and symbols the way a person would say them out loud. If you "
    "do not know something, say so briefly."
)


class ConfigError(ValueError):
    """Raised for malformed configuration."""


@dataclass
class WakeConfig:
    engine: Literal["openwakeword", "none"] = "openwakeword"
    model: str = "openwakeword/hey_jarvis_v0.1.onnx"
    threshold: float = 0.5
    # Audio kept from before the wake trigger so a run-on "hey jarvis what's ..." is
    # transcribed whole; the wake phrase is stripped from the transcript afterwards.
    preroll_s: float = 2.0


@dataclass
class VadConfig:
    threshold: float = 0.5
    end_silence_ms: int = 600
    min_speech_ms: int = 160
    max_utterance_s: float = 15.0
    no_speech_timeout_s: float = 5.0


@dataclass
class SttConfig:
    engine: Literal["parakeet", "whisper"] = "parakeet"
    threads: int = 4
    whisper_model: str = "small"


@dataclass
class LlmConfig:
    model: str = "llama3.2"
    host: str = "http://127.0.0.1:11434"
    temperature: float = 0.6
    seed: int | None = None
    max_tokens: int = 160
    history_turns: int = 8
    connect_timeout_s: float = 5.0
    read_timeout_s: float = 60.0
    keep_alive: str = "30m"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


@dataclass
class TtsConfig:
    engine: Literal["kokoro"] = "kokoro"
    voice: str = "am_michael"
    speed: float = 1.0
    lang: str = "en-us"


@dataclass
class PipelineConfig:
    streaming: bool = True
    # "clause": the first chunk of a reply may end at a comma so speech starts sooner;
    # "sentence": every chunk is a whole sentence.
    first_chunk: Literal["clause", "sentence"] = "clause"
    barge_in: Literal["wake", "vad", "off"] = "wake"
    # VAD barge-in guard: speech must be this confident and this long, and the
    # first part of playback is ignored entirely (speaker onset leaks into the mic).
    barge_in_vad_threshold: float = 0.85
    barge_in_min_speech_ms: int = 400
    barge_in_ignore_ms: int = 600
    chunker_min_chars: int = 10
    chunker_max_chars: int = 220


@dataclass
class PathsConfig:
    models_dir: str = "models"
    traces_dir: str = "traces"
    data_dir: str = "data"


@dataclass
class Config:
    wake: WakeConfig = field(default_factory=WakeConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @property
    def models_dir(self) -> Path:
        return Path(self.paths.models_dir).expanduser()

    @property
    def traces_dir(self) -> Path:
        return Path(self.paths.traces_dir).expanduser()

    @property
    def data_dir(self) -> Path:
        return Path(self.paths.data_dir).expanduser()

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _coerce(value: Any, annotation: Any, where: str) -> Any:
    """Validate ``value`` against a (simple) type annotation, allowing int -> float."""
    origin = typing.get_origin(annotation)
    if origin is Literal:
        allowed = typing.get_args(annotation)
        if value not in allowed:
            raise ConfigError(f"{where}: {value!r} is not one of {list(allowed)}")
        return value
    if origin in (typing.Union, types.UnionType):
        options = typing.get_args(annotation)
        if value is None and type(None) in options:
            return None
        errors = []
        for option in options:
            if option is type(None):
                continue
            try:
                return _coerce(value, option, where)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError("; ".join(errors))
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if annotation is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false, got {value!r}")
        return value
    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return value
    raise ConfigError(f"{where}: unsupported annotation {annotation!r}")


def _apply_section(section: object, values: dict[str, Any], name: str) -> None:
    hints = typing.get_type_hints(type(section))
    known = {f.name for f in dataclasses.fields(section)}  # type: ignore[arg-type]
    for key, value in values.items():
        if key not in known:
            raise ConfigError(f"unknown key [{name}].{key}; expected one of {sorted(known)}")
        setattr(section, key, _coerce(value, hints[key], f"[{name}].{key}"))


def _validate(cfg: Config) -> None:
    checks: list[tuple[bool, str]] = [
        (0.0 < cfg.wake.threshold < 1.0, "[wake].threshold must be in (0, 1)"),
        (0.0 < cfg.vad.threshold < 1.0, "[vad].threshold must be in (0, 1)"),
        (cfg.vad.end_silence_ms >= 80, "[vad].end_silence_ms must be >= 80"),
        (cfg.wake.preroll_s >= 0.0, "[wake].preroll_s must be >= 0"),
        (cfg.stt.threads >= 1, "[stt].threads must be >= 1"),
        (cfg.llm.max_tokens >= 1, "[llm].max_tokens must be >= 1"),
        (cfg.llm.history_turns >= 0, "[llm].history_turns must be >= 0"),
        (0.0 <= cfg.llm.temperature <= 2.0, "[llm].temperature must be in [0, 2]"),
        (cfg.llm.host.startswith(("http://", "https://")), "[llm].host must be an http(s) URL"),
        (0.5 <= cfg.tts.speed <= 2.0, "[tts].speed must be in [0.5, 2]"),
        (
            cfg.pipeline.chunker_max_chars > cfg.pipeline.chunker_min_chars >= 1,
            "[pipeline] needs chunker_max_chars > chunker_min_chars >= 1",
        ),
    ]
    for ok, message in checks:
        if not ok:
            raise ConfigError(message)


def from_dict(data: dict[str, Any]) -> Config:
    cfg = Config()
    for name, values in data.items():
        if not hasattr(cfg, name) or not dataclasses.is_dataclass(getattr(cfg, name)):
            raise ConfigError(f"unknown section [{name}]")
        if not isinstance(values, dict):
            raise ConfigError(f"[{name}] must be a table")
        _apply_section(getattr(cfg, name), values, name)
    _validate(cfg)
    return cfg


def load(path: Path | None) -> Config:
    """Load ``path`` (TOML) or, when ``None``, ``./jarvis.toml`` if present, else defaults."""
    if path is None:
        candidate = Path("jarvis.toml")
        if not candidate.is_file():
            return from_dict({})
        path = candidate
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    return from_dict(data)


def override(cfg: Config, dotted: dict[str, Any]) -> Config:
    """Apply ``{"llm.model": "llama3"}``-style overrides (from CLI flags) and revalidate."""
    data: dict[str, dict[str, Any]] = {}
    for key, value in dotted.items():
        if value is None:
            continue
        section, _, name = key.partition(".")
        data.setdefault(section, {})[name] = value
    merged = cfg.to_dict()
    for section, values in data.items():
        if section not in merged:
            raise ConfigError(f"unknown section [{section}]")
        merged[section].update(values)
    return from_dict(merged)
