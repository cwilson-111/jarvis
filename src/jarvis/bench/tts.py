"""Kokoro synthesis speed per model precision (why the int8 export is the default).

The fp32 and fp16 exports are only needed for this comparison and are fetched
into ``models/kokoro/`` on demand.
"""

from __future__ import annotations

import gc
import sys
import time
from typing import Any

from jarvis.bench.common import hardware
from jarvis.config import Config
from jarvis.models import fetch

BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
VARIANTS = ("kokoro-v1.0.int8.onnx", "kokoro-v1.0.fp16.onnx", "kokoro-v1.0.onnx")
SENTENCES = (
    "Sure.",
    "Paris is the capital of France.",
    "The tallest mountain in the world is Mount Everest, at about twenty nine thousand feet.",
    "Photosynthesis is how plants turn sunlight, water and carbon dioxide into sugar and "
    "oxygen, and it happens mostly in their leaves.",
)
REPEATS = 3


def run(cfg: Config) -> dict[str, Any]:
    from kokoro_onnx import Kokoro

    kokoro_dir = cfg.models_dir / "kokoro"
    voices = str(kokoro_dir / "voices-v1.0.bin")
    results: dict[str, Any] = {}
    for variant in VARIANTS:
        path = kokoro_dir / variant
        if not path.is_file():
            print(f"  fetching {variant}", file=sys.stderr)
            fetch(BASE + variant, path)
        kokoro = Kokoro(str(path), voices)
        kokoro.create("Warm up.", voice=cfg.tts.voice)
        rows = []
        for text in SENTENCES:
            best = float("inf")
            seconds = 0.0
            for _ in range(REPEATS):
                t = time.perf_counter()
                audio, rate = kokoro.create(text, voice=cfg.tts.voice)
                best = min(best, time.perf_counter() - t)
                seconds = len(audio) / rate
            rows.append(
                {
                    "chars": len(text),
                    "audio_s": round(seconds, 2),
                    "synth_ms": round(best * 1000),
                    "rtf": round(best / seconds, 3),
                }
            )
        total_synth = sum(r["synth_ms"] for r in rows) / 1000
        total_audio = sum(r["audio_s"] for r in rows)
        results[variant] = {
            "size_mb": round(path.stat().st_size / 2**20),
            "rows": rows,
            "rtf": round(total_synth / total_audio, 3),
        }
        print(f"  [tts] {variant}: RTF {results[variant]['rtf']}", file=sys.stderr)
        del kokoro
        gc.collect()
    return {
        "voice": cfg.tts.voice,
        "repeats": REPEATS,
        "metric": "best of 3 wall-clock synth time per sentence; RTF = synth time / audio length",
        "variants": results,
        "hardware": hardware(),
    }
