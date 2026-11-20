"""End-to-end latency (streaming vs. whole-reply), barge-in reaction, and peak RAM.

Every turn goes through the real ``Assistant`` loop: a synthesised
"Hey Jarvis, <question>" wav is replayed in real time as if it were the
microphone, so the wake word, Silero endpointing (including its silence
wait), Parakeet, Ollama and Kokoro all run exactly as in live use. The only
difference from live use is the sink: playback is simulated (``FileSink``)
instead of sent to a device, so "first audio out" excludes the output
device's buffer latency (CoreAudio reports ~10-20 ms on this machine).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from jarvis import config as config_mod
from jarvis.audio import FileSink, FileSource
from jarvis.bench.common import MemorySampler, hardware, ollama_info
from jarvis.bench.questions import QUESTIONS, SPEAKER_VOICES, WAKE_PREFIX
from jarvis.config import Config
from jarvis.engines import build_llm, build_stt, build_tts, build_vad, build_wake
from jarvis.pipeline import Assistant
from jarvis.trace import STAGES, TraceWriter, percentile
from jarvis.tts import KokoroTts
from jarvis.wer import wer

LEAD_SILENCE_S = 1.5
BARGE_IN_PROMPT = "Tell me a story about a lighthouse keeper and a lost whale."
# Interrupt this long after the request ends: after first audio (p95 < 3 s) but while
# the TTS is usually still synthesising later sentences of the story.
BARGE_IN_AFTER_S = 3.0
BARGE_IN_TRIALS = 10

# Benchmark variants -> config overrides.
MODES: dict[str, dict[str, object]] = {
    "clause": {"pipeline.streaming": True, "pipeline.first_chunk": "clause"},
    "sentence": {"pipeline.streaming": True, "pipeline.first_chunk": "sentence"},
    "batch": {"pipeline.streaming": False},
}


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "p50": round(percentile(values, 50), 1),
        "p95": round(percentile(values, 95), 1),
        "mean": round(float(np.mean(values)), 1),
        "min": round(min(values), 1),
        "max": round(max(values), 1),
    }


def _write_wav(path: Path, parts: list[np.ndarray], rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.concatenate(parts).astype(np.float32), rate)


def _silence(seconds: float, rate: int) -> np.ndarray:
    return np.zeros(int(seconds * rate), dtype=np.float32)


def make_inputs(tts: KokoroTts, out_dir: Path, n: int) -> list[tuple[Path, str]]:
    """Synthesise (once, cached) the spoken requests."""
    inputs = []
    for i, question in enumerate(QUESTIONS[:n]):
        voice = SPEAKER_VOICES[i % len(SPEAKER_VOICES)]
        path = out_dir / f"q{i:02d}-{voice}.wav"
        if not path.is_file():
            audio, rate = tts.synthesize(WAKE_PREFIX + question, voice=voice)
            _write_wav(path, [_silence(LEAD_SILENCE_S, rate), audio], rate)
        inputs.append((path, question))
    return inputs


def _voiced_samples(audio: np.ndarray, floor: float = 0.01) -> int:
    """Length up to the last sample above ``floor`` x peak (Kokoro leaves a quiet tail)."""
    loud = np.flatnonzero(np.abs(audio) >= floor * float(np.max(np.abs(audio))))
    return int(loud[-1]) + 1 if len(loud) else 0


def make_barge_in_inputs(tts: KokoroTts, out_dir: Path) -> list[tuple[Path, float]]:
    """Request, BARGE_IN_AFTER_S of silence, then "Hey Jarvis." and "Stop."

    Returns (path, seconds from file start to the last voiced sample of "Hey Jarvis").
    """
    inputs = []
    for i in range(BARGE_IN_TRIALS):
        voice = SPEAKER_VOICES[i % len(SPEAKER_VOICES)]
        path = out_dir / f"barge-{BARGE_IN_AFTER_S:.1f}s-{i:02d}-{voice}.wav"
        request, rate = tts.synthesize(WAKE_PREFIX + BARGE_IN_PROMPT, voice=voice)
        wake, _ = tts.synthesize("Hey Jarvis.", voice=voice)
        stop, _ = tts.synthesize("Stop.", voice=voice)
        lead = _silence(LEAD_SILENCE_S, rate)
        gap = _silence(BARGE_IN_AFTER_S, rate)
        if not path.is_file():
            _write_wav(path, [lead, request, gap, wake, _silence(0.2, rate), stop], rate)
        wake_end = (len(lead) + len(request) + len(gap) + _voiced_samples(wake)) / rate
        inputs.append((path, wake_end))
    return inputs


def _stage_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for stage in STAGES:
        values = [r["stages_ms"][stage] for r in records if r["stages_ms"].get(stage) is not None]
        out[stage] = summarize(values)
    out["stall_ms"] = summarize([r["stall_s"] * 1000 for r in records if r.get("stall_s")])
    out["reply_chars"] = summarize([float(len(r["reply"])) for r in records])
    out["playback_total_ms"] = summarize(
        [
            r["marks_ms"]["done"] - r["marks_ms"]["first_audio_out"]
            for r in records
            if "done" in r["marks_ms"] and "first_audio_out" in r["marks_ms"]
        ]
    )
    return out


def run(cfg: Config, turns: int, modes: list[str], barge_in: bool) -> dict[str, Any]:
    # Deterministic replies so both modes speak the same text for the same question.
    cfg = config_mod.override(cfg, {"llm.temperature": 0.0, "llm.seed": 42})
    out_dir = cfg.data_dir / "bench" / "latency"
    stamp = time.strftime("%Y%m%d-%H%M%S")

    with MemorySampler() as mem:
        wake = build_wake(cfg)
        vad = build_vad(cfg)
        stt = build_stt(cfg)
        tts = build_tts(cfg)
        llm = build_llm(cfg)
        cold_load_s = llm.warmup()
        inputs = make_inputs(tts, out_dir, turns)

        result: dict[str, Any] = {"modes": {}}
        for mode in modes:
            mode_cfg = config_mod.override(cfg, MODES[mode])
            sink = FileSink(None, realtime=False)
            writer = TraceWriter(cfg.traces_dir / "bench", f"latency-{mode}-{stamp}")
            assistant = Assistant(
                mode_cfg,
                wake=wake,
                vad=vad,
                stt=stt,
                llm=llm,
                tts=tts,
                sink=sink,
                writer=TraceWriter(None),
                meta={"mode": "bench", "variant": mode},
            )
            assistant.run(FileSource(inputs[0][0]), max_turns=1)  # untimed warm-up
            assistant.writer = writer
            records: list[dict[str, Any]] = []
            failures: list[str] = []
            for i, (path, question) in enumerate(inputs):
                assistant.conversation.clear()
                traces = assistant.run(FileSource(path), max_turns=1)
                if not traces:
                    failures.append(path.name)
                    print(f"  [{mode}] {i + 1}/{len(inputs)} no turn (wake/endpoint missed)")
                    continue
                record = traces[0].to_record()
                record["question"] = question
                record["input"] = path.name
                record["transcript_wer"] = round(wer(question, record["user_text"]), 3)
                records.append(record)
                e2e = record["stages_ms"]["e2e"]
                print(f"  [{mode}] {i + 1}/{len(inputs)} e2e {e2e} ms", file=sys.stderr)
            result["modes"][mode] = {
                "turns": len(records),
                "failures": failures,
                "stages_ms": _stage_summary(records),
                "transcript_wer_mean": round(
                    float(np.mean([r["transcript_wer"] for r in records])), 4
                )
                if records
                else None,
                "trace_file": str(writer.path),
                "records": records,
            }

        if barge_in:
            result["barge_in"] = _run_barge_in(
                cfg, wake=wake, vad=vad, stt=stt, llm=llm, tts=tts, out_dir=out_dir
            )
        result["ollama"] = ollama_info(cfg.llm.host, cfg.llm.model)

    result["memory"] = mem.summary()
    result["ollama_cold_load_s"] = round(cold_load_s, 2)
    result["config"] = {
        "llm_model": cfg.llm.model,
        "llm_temperature": cfg.llm.temperature,
        "llm_seed": cfg.llm.seed,
        "llm_max_tokens": cfg.llm.max_tokens,
        "stt": stt.name,
        "tts": tts.name,
        "wake": wake.name,
        "wake_threshold": cfg.wake.threshold,
        "vad_end_silence_ms": cfg.vad.end_silence_ms,
        "speaker_voices": list(SPEAKER_VOICES),
    }
    result["hardware"] = hardware()
    return result


def _run_barge_in(
    cfg: Config, *, wake: Any, vad: Any, stt: Any, llm: Any, tts: KokoroTts, out_dir: Path
) -> dict[str, Any]:
    inputs = make_barge_in_inputs(tts, out_dir)
    sink = FileSink(None, realtime=True)  # playback must still be "running" when interrupted
    assistant = Assistant(
        cfg,
        wake=wake,
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
        sink=sink,
        writer=TraceWriter(None),
        meta={"mode": "bench", "variant": "barge_in"},
    )
    trials: list[dict[str, Any]] = []
    for path, wake_end_s in inputs:
        assistant.conversation.clear()
        source = FileSource(path)
        started = time.perf_counter()
        traces = assistant.run(source, max_turns=1)
        if not traces:
            trials.append({"input": path.name, "interrupted": False, "reason": "no turn"})
            continue
        trace = traces[0]
        rec: dict[str, Any] = {"input": path.name, "interrupted": trace.meta["interrupted"]}
        if trace.has("barge_in"):
            barge = trace.marks["barge_in"]
            # The source releases audio in real time, so the wake phrase's last voiced sample
            # reached the pipeline at started + wake_end_s (frames are 80 ms, so a detection
            # can land up to one frame either side of that instant).
            rec["wake_detect_ms"] = round((barge - started - wake_end_s) * 1e3)
            rec["stop_audio_ms"] = trace.duration_ms("barge_in", "cancelled")
            rec["wind_down_ms"] = trace.duration_ms("barge_in", "done")
            rec["audio_had_started"] = trace.marks.get("first_audio_out", barge + 1) < barge
            rec["llm_still_generating"] = trace.marks.get("llm_done", barge + 1) > barge
            rec["sentences_sent"] = len(trace.meta["spoken"])
            rec["sentences_heard"] = len(trace.meta["heard"])
        trials.append(rec)
        print(f"  [barge-in] {path.name}: {rec}", file=sys.stderr)
    done = [t for t in trials if "wake_detect_ms" in t]
    return {
        "trials": trials,
        "interrupted": len(done),
        "total": len(trials),
        "audio_had_started": sum(1 for t in done if t["audio_had_started"]),
        "llm_still_generating": sum(1 for t in done if t["llm_still_generating"]),
        "wake_detect_ms": summarize([float(t["wake_detect_ms"]) for t in done]),
        "stop_audio_ms": summarize([float(t["stop_audio_ms"]) for t in done]),
        "wind_down_ms": summarize([float(t["wind_down_ms"]) for t in done]),
        "interrupt_after_s": BARGE_IN_AFTER_S,
    }
