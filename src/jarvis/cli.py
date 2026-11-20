"""Command-line entry point: `jarvis {listen,run,say,bench,doctor,download}`."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from jarvis import config as config_mod
from jarvis.chunker import speakable
from jarvis.config import Config, ConfigError
from jarvis.trace import TraceWriter, TurnTrace


def _safe(text: str) -> str:
    """Model/STT output is untrusted: never let it put escape sequences on a terminal."""
    return "".join(ch if ch.isprintable() else " " for ch in text)


def _printer(kind: str, text: str) -> None:
    labels = {
        "user": "you",
        "reply_done": "jarvis",
        "wake": "wake word",
        "ptt": "push-to-talk",
        "barge_in": "barge-in: stopped speaking",
        "timeout": "no speech heard, going idle",
        "wake_only": "heard the wake phrase, listening for your request",
        "empty": "heard nothing intelligible",
        "error": "error",
        "stop": "stopped",
    }
    if kind == "state":
        if text == "idle":
            print("  [waiting for wake word]", file=sys.stderr)
        return
    label = labels.get(kind, kind)
    line = f"  {label}: {_safe(text)}" if text else f"  [{label}]"
    print(line, file=sys.stderr if kind not in ("user", "reply_done") else sys.stdout)


def _load_config(args: argparse.Namespace) -> Config:
    cfg = config_mod.load(args.config)
    overrides: dict[str, Any] = {
        "llm.model": getattr(args, "model", None),
        "tts.voice": getattr(args, "voice", None),
        "stt.engine": getattr(args, "stt", None),
        "pipeline.barge_in": getattr(args, "barge_in", None),
    }
    if getattr(args, "no_streaming", False):
        overrides["pipeline.streaming"] = False
    return config_mod.override(cfg, overrides)


def _turn_summary(trace: TurnTrace) -> str:
    stages = trace.stages_ms()
    parts = []
    for name in ("endpointing", "stt", "llm_ttft", "first_sentence", "tts_first", "e2e"):
        value = stages.get(name)
        if value is not None:
            parts.append(f"{name} {value:.0f}")
    return "  latency ms: " + ", ".join(parts)


def _build_assistant(cfg: Config, sink: Any, writer: TraceWriter, meta: dict[str, Any]) -> Any:
    from jarvis.engines import build_llm, build_stt, build_tts, build_vad, build_wake
    from jarvis.pipeline import Assistant

    t0 = time.perf_counter()
    wake = build_wake(cfg)
    vad = build_vad(cfg)
    stt = build_stt(cfg)
    tts = build_tts(cfg)
    llm = build_llm(cfg)
    load_s = llm.warmup()
    # First ONNX runs allocate buffers; keep that out of the first turn.
    tts.synthesize("Ready.")
    stt.transcribe(np.zeros(16_000, dtype=np.float32))
    print(
        f"  engines ready in {time.perf_counter() - t0:.1f} s "
        f"(stt={stt.name}, llm={llm.name}, tts={tts.name}, wake={wake.name}; "
        f"ollama load {load_s:.1f} s)",
        file=sys.stderr,
    )
    return Assistant(
        cfg,
        wake=wake,
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
        sink=sink,
        writer=writer,
        on_event=_printer,
        meta=meta,
    )


# --------------------------------------------------------------------------- commands


def cmd_run(args: argparse.Namespace) -> int:
    from jarvis.audio import FileSink, FileSource, SpeakerSink

    cfg = _load_config(args)
    if not args.input.is_file():
        print(f"no such file: {args.input}", file=sys.stderr)
        return 2
    sink: Any = SpeakerSink() if args.speak else FileSink(args.output, realtime=not args.fast)
    writer = TraceWriter(cfg.traces_dir)
    meta = {"mode": "run", "input": str(args.input), "realtime": not args.fast}
    assistant = _build_assistant(cfg, sink, writer, meta)
    source = FileSource(args.input, realtime=not args.fast)
    trigger = "always" if args.no_wake else "wake"
    try:
        traces = assistant.run(source, trigger=trigger, max_turns=args.max_turns)
    finally:
        sink.close()
    for trace in traces:
        print(_turn_summary(trace), file=sys.stderr)
    if args.output and not args.speak:
        print(f"  reply audio: {args.output}", file=sys.stderr)
    print(f"  trace: {writer.path}", file=sys.stderr)
    return 0 if traces else 1


def cmd_listen(args: argparse.Namespace) -> int:
    from jarvis.audio import MicSource, SpeakerSink

    cfg = _load_config(args)
    if args.ptt:
        cfg = config_mod.override(cfg, {"wake.engine": "none"})
    sink = SpeakerSink()
    writer = TraceWriter(cfg.traces_dir)
    assistant = _build_assistant(cfg, sink, writer, {"mode": "listen"})
    source = MicSource()
    if args.ptt:

        def read_keys() -> None:
            for _line in sys.stdin:
                assistant.trigger.set()

        threading.Thread(target=read_keys, daemon=True).start()
        print("  push-to-talk: press Enter, then speak (Enter while it talks interrupts)")
    else:
        print('  say "Hey Jarvis" ... (Ctrl-C to quit)')
    # Ctrl-C asks the loop to stop at the next frame, so an in-flight reply is cancelled
    # and its trace still written, instead of unwinding from wherever the signal lands.
    signal.signal(signal.SIGINT, lambda *_: assistant.stop_requested.set())
    try:
        assistant.run(source, trigger="ptt" if args.ptt else "wake")
    finally:
        source.close()
        sink.close()
    print(f"  trace: {writer.path}", file=sys.stderr)
    return 0


def cmd_say(args: argparse.Namespace) -> int:
    import soundfile as sf

    from jarvis.engines import build_tts

    cfg = _load_config(args)
    tts = build_tts(cfg)
    audio, rate = tts.synthesize(speakable(args.text))
    sf.write(str(args.out), audio, rate)
    print(f"  wrote {args.out} ({len(audio) / rate:.2f} s, {tts.name})", file=sys.stderr)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from jarvis.doctor import run_checks

    cfg = _load_config(args)
    checks = run_checks(cfg, audio=not args.no_audio)
    width = max(len(c.name) for c in checks)
    for c in checks:
        mark = "ok  " if c.ok else ("FAIL" if c.critical else "warn")
        print(f"  {mark}  {c.name:<{width}}  {c.detail}")
    print(
        f"\n  config: llm={cfg.llm.model} voice={cfg.tts.voice} stt={cfg.stt.engine} "
        f"streaming={cfg.pipeline.streaming} barge_in={cfg.pipeline.barge_in}"
    )
    return 0 if all(c.ok or not c.critical for c in checks) else 1


def cmd_download(args: argparse.Namespace) -> int:
    from jarvis.models import download

    cfg = _load_config(args)
    fetched = download(cfg.models_dir, force=args.force)
    print(f"  {len(fetched)} downloaded into {cfg.models_dir}", file=sys.stderr)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from jarvis.bench import run_bench

    cfg = _load_config(args)
    return run_bench(cfg, args)


# ----------------------------------------------------------------------------- parser


def _engine_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, help="TOML config (default: ./jarvis.toml if present)")
    p.add_argument("--model", help="Ollama model, e.g. llama3.2 or llama3")
    p.add_argument("--voice", help="Kokoro voice (default am_michael)")
    p.add_argument("--stt", choices=["parakeet", "whisper"], help="STT engine")
    p.add_argument("--no-streaming", action="store_true", help="wait for the whole reply")
    p.add_argument("--barge-in", choices=["wake", "vad", "off"], help="interrupt mode")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jarvis", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("listen", help="live microphone assistant")
    _engine_flags(p)
    p.add_argument("--ptt", action="store_true", help="push-to-talk (Enter) instead of wake word")
    p.set_defaults(func=cmd_listen)

    p = sub.add_parser("run", help="run the same pipeline on an audio file")
    _engine_flags(p)
    p.add_argument("--input", type=Path, required=True, help="wav/flac containing the request")
    p.add_argument("--output", type=Path, help="write the spoken reply to this wav")
    p.add_argument("--speak", action="store_true", help="play the reply through the speakers")
    p.add_argument("--fast", action="store_true", help="feed audio faster than real time")
    p.add_argument("--no-wake", action="store_true", help="input has no wake phrase")
    p.add_argument("--max-turns", type=int, help="stop after N turns")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("say", help="synthesise text to a wav (handy for making test inputs)")
    _engine_flags(p)
    p.add_argument("text")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_say)

    p = sub.add_parser("doctor", help="check models, microphone and Ollama")
    _engine_flags(p)
    p.add_argument("--no-audio", action="store_true", help="skip the microphone check")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("download", help="fetch all models into models/")
    _engine_flags(p)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("bench", help="latency / STT / wake-word / memory benchmarks")
    _engine_flags(p)
    from jarvis.bench import add_arguments

    add_arguments(p)
    p.set_defaults(func=cmd_bench)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2


__all__ = ["build_parser", "main"]
