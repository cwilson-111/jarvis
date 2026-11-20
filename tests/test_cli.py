from pathlib import Path

from jarvis.cli import _safe, build_parser
from jarvis.config import load

ROOT = Path(__file__).resolve().parents[1]


def test_example_config_is_valid() -> None:
    cfg = load(ROOT / "jarvis.example.toml")
    assert cfg.llm.model == "llama3.2"
    assert cfg.tts.voice == "am_michael"


def test_parser_knows_every_mode() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "--input", "q.wav", "--no-streaming", "--model", "llama3"])
    assert args.command == "run" and args.no_streaming and args.model == "llama3"
    assert parser.parse_args(["listen", "--ptt"]).ptt
    assert parser.parse_args(["bench", "latency", "--turns", "5"]).turns == 5
    assert parser.parse_args(["doctor", "--no-audio"]).no_audio


def test_terminal_output_strips_escape_sequences() -> None:
    # Model output is untrusted: an ESC sequence must not reach the terminal.
    assert "\x1b" not in _safe("hello \x1b]0;pwned\x07 world")
