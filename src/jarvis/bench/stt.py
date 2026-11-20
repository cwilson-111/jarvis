"""STT accuracy (WER) and speed (real-time factor) on LibriSpeech test-clean."""

from __future__ import annotations

import gc
import sys
import time
from typing import Any

from jarvis import config as config_mod
from jarvis.audio import SAMPLE_RATE, load_audio
from jarvis.bench import librispeech
from jarvis.bench.common import hardware
from jarvis.config import Config
from jarvis.engines import build_stt
from jarvis.wer import WerCounts, wer_counts


def run(cfg: Config, n: int, engines: list[str]) -> dict[str, Any]:
    root = librispeech.ensure(cfg.data_dir)
    subset = librispeech.spread_subset(librispeech.utterances(root), n)
    audio = {u.id: load_audio(u.path) for u in subset}
    total_audio_s = sum(len(a) for a in audio.values()) / SAMPLE_RATE
    speakers = sorted({u.id.split("-")[0] for u in subset})
    result: dict[str, Any] = {
        "dataset": "LibriSpeech test-clean",
        "utterances": len(subset),
        "speakers": len(speakers),
        "audio_s": round(total_audio_s, 1),
        "selection": "every k-th utterance of the id-sorted list (k = 2620 / n)",
        "engines": {},
    }
    for engine in engines:
        engine_cfg = config_mod.override(cfg, {"stt.engine": engine})
        t0 = time.perf_counter()
        stt = build_stt(engine_cfg)
        load_s = time.perf_counter() - t0
        stt.transcribe(audio[subset[0].id])  # warm-up, untimed
        totals = WerCounts(0, 0, 0, 0)
        compute_s = 0.0
        rows = []
        for i, utt in enumerate(subset):
            t = time.perf_counter()
            hyp = stt.transcribe(audio[utt.id])
            elapsed = time.perf_counter() - t
            compute_s += elapsed
            counts = wer_counts(utt.text, hyp)
            totals = totals + counts
            rows.append({"id": utt.id, "ref": utt.text, "hyp": hyp, "errors": counts.errors})
            if (i + 1) % 25 == 0:
                done = f"{i + 1}/{len(subset)}"
                print(f"  [{stt.name}] {done} WER so far {totals.wer:.4f}", file=sys.stderr)
        result["engines"][stt.name] = {
            "wer": round(totals.wer, 4),
            "substitutions": totals.substitutions,
            "deletions": totals.deletions,
            "insertions": totals.insertions,
            "ref_words": totals.ref_words,
            "rtf": round(compute_s / total_audio_s, 4),
            "compute_s": round(compute_s, 2),
            "load_s": round(load_s, 2),
            "threads": engine_cfg.stt.threads,
            "utterances_with_errors": sum(1 for r in rows if r["errors"]),
            "rows": rows,
        }
        del stt
        gc.collect()
    result["hardware"] = hardware()
    return result
