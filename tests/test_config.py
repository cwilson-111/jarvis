from pathlib import Path

import pytest

from jarvis.config import Config, ConfigError, from_dict, load, override


def test_defaults_are_the_documented_engines() -> None:
    cfg = Config()
    assert cfg.llm.model == "llama3.2"
    assert cfg.tts.voice == "am_michael"
    assert cfg.stt.engine == "parakeet"
    assert cfg.pipeline.streaming is True
    assert cfg.pipeline.barge_in == "wake"


def test_load_toml(tmp_path: Path) -> None:
    path = tmp_path / "jarvis.toml"
    path.write_text(
        """
[llm]
model = "llama3"
temperature = 0   # int is accepted for a float field
seed = 7

[vad]
end_silence_ms = 500

[pipeline]
streaming = false
barge_in = "vad"
"""
    )
    cfg = load(path)
    assert cfg.llm.model == "llama3"
    assert cfg.llm.temperature == 0.0 and isinstance(cfg.llm.temperature, float)
    assert cfg.llm.seed == 7
    assert cfg.vad.end_silence_ms == 500
    assert cfg.pipeline.streaming is False
    assert cfg.pipeline.barge_in == "vad"
    assert cfg.tts.voice == "am_michael"  # untouched sections keep defaults


def test_missing_default_file_gives_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert load(None) == Config()


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"llm": {"modle": "x"}}, "unknown key [llm].modle"),
        ({"lm": {"model": "x"}}, "unknown section [lm]"),
        ({"llm": "llama3"}, "[llm] must be a table"),
        ({"llm": {"max_tokens": "many"}}, "expected an integer"),
        ({"llm": {"max_tokens": 1.5}}, "expected an integer"),
        ({"pipeline": {"streaming": "yes"}}, "expected true/false"),
        ({"vad": {"threshold": True}}, "expected a number"),
        ({"pipeline": {"barge_in": "always"}}, "not one of"),
        ({"stt": {"engine": "vosk"}}, "not one of"),
        ({"wake": {"threshold": 1.5}}, "[wake].threshold"),
        ({"llm": {"host": "localhost:11434"}}, "http(s) URL"),
        ({"tts": {"speed": 3}}, "[tts].speed"),
        ({"pipeline": {"chunker_min_chars": 50, "chunker_max_chars": 40}}, "chunker"),
    ],
)
def test_invalid_config_is_rejected(data: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError) as info:
        from_dict(data)
    assert message in str(info.value)


def test_optional_seed_accepts_null_like_absence_and_ints() -> None:
    assert from_dict({"llm": {"seed": 3}}).llm.seed == 3
    assert from_dict({}).llm.seed is None


def test_override_applies_dotted_keys_and_ignores_none() -> None:
    cfg = override(Config(), {"llm.model": "qwen2.5:7b", "tts.voice": None})
    assert cfg.llm.model == "qwen2.5:7b"
    assert cfg.tts.voice == "am_michael"
    with pytest.raises(ConfigError):
        override(Config(), {"llm.temperature": 9.0})
    with pytest.raises(ConfigError):
        override(Config(), {"nope.key": 1})


def test_bad_toml_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("[llm\nmodel=")
    with pytest.raises(ConfigError):
        load(path)
    with pytest.raises(ConfigError):
        load(tmp_path / "missing.toml")
