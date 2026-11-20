"""`jarvis bench {latency,stt,wake,report}`.

Each benchmark writes ``docs/results/<name>.json`` (the source of truth, with
hardware, versions and the exact command) and then re-renders
``docs/RESULTS.md`` from whatever result files exist.
"""

from __future__ import annotations

import argparse
import sys
import time

from jarvis.config import Config


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("which", choices=["latency", "stt", "wake", "tts", "report"])
    p.add_argument("--turns", type=int, default=30, help="latency: requests per mode")
    p.add_argument("--modes", default="clause,sentence,batch", help="latency: comma list")
    p.add_argument("--no-barge-in-trials", action="store_true", help="latency: skip barge-in")
    p.add_argument("--n", type=int, default=200, help="stt: LibriSpeech utterances")
    p.add_argument("--engines", default="parakeet,whisper", help="stt: comma list")
    p.add_argument("--workers", type=int, default=4, help="wake: processes for negatives")
    p.add_argument("--negatives-limit", type=int, help="wake: only the first N chapters")


def run_bench(cfg: Config, args: argparse.Namespace) -> int:
    from jarvis.bench import report
    from jarvis.bench.common import save_result

    command = "jarvis " + " ".join(sys.argv[1:])
    started = time.time()
    if args.which == "latency":
        from jarvis.bench import latency

        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        unknown = set(modes) - set(latency.MODES)
        if unknown:
            print(f"unknown modes: {sorted(unknown)}", file=sys.stderr)
            return 2
        data = latency.run(cfg, args.turns, modes, barge_in=not args.no_barge_in_trials)
    elif args.which == "stt":
        from jarvis.bench import stt

        engines = [e.strip() for e in args.engines.split(",") if e.strip()]
        data = stt.run(cfg, args.n, engines)
    elif args.which == "wake":
        from jarvis.bench import wake

        data = wake.run(cfg, args.workers, args.negatives_limit)
    elif args.which == "tts":
        from jarvis.bench import tts

        data = tts.run(cfg)
    else:
        path = report.render()
        print(f"  wrote {path}", file=sys.stderr)
        return 0
    data["command"] = command
    data["wall_s"] = round(time.time() - started, 1)
    saved = save_result(args.which, data)
    print(f"  wrote {saved}", file=sys.stderr)
    print(f"  wrote {report.render()}", file=sys.stderr)
    return 0
