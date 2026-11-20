"""Wake-word false rejects (synthetic positives) and false accepts (LibriSpeech).

Positives: "hey jarvis" phrases synthesised by every English Kokoro voice at
three speaking rates, under several noise conditions. A clip counts as
detected if the score crosses the threshold anywhere in it.

Negatives: all of LibriSpeech test-clean, each chapter concatenated into one
continuous stream. A false accept is a rising edge across the threshold, with
a 2 s refractory period (the assistant ignores re-triggers for 1.5 s anyway).

Limits: Kokoro voices are clean, close-mic, synthetic speech, and openWakeWord
was itself trained on synthetic speech, so the false-reject numbers are an
optimistic, best-case view. They say nothing about accents, distance, room
reverb or children. LibriSpeech is read audiobook speech; TV, music and
conversation are not covered.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from jarvis.audio import (
    FRAME_S,
    FRAME_SAMPLES,
    SAMPLE_RATE,
    Float32,
    load_audio,
    resample,
    to_int16,
)
from jarvis.bench import librispeech
from jarvis.bench.common import hardware
from jarvis.config import Config
from jarvis.engines import build_tts
from jarvis.wake import OpenWakeWord

THRESHOLDS = (0.3, 0.5, 0.7, 0.9)
PHRASES = ("Hey Jarvis.", "Hey Jarvis, what's the weather like today?")
SPEEDS = (0.85, 1.0, 1.2)
CONDITIONS = (
    "clean",
    "quiet -20 dB",
    "white noise 10 dB SNR",
    "speech babble 10 dB SNR",
    "speech babble 0 dB SNR",
)
REFRACTORY_S = 2.0
LEAD_S = 1.5
TAIL_S = 1.0


def _scores(detector: OpenWakeWord, audio: Float32) -> Float32:
    pcm = to_int16(audio)
    pad = (-len(pcm)) % FRAME_SAMPLES
    pcm = np.concatenate([pcm, np.zeros(pad, dtype=np.int16)])
    out = np.empty(len(pcm) // FRAME_SAMPLES, dtype=np.float32)
    for i in range(len(out)):
        out[i] = detector.score(pcm[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES])
    return out


def _reset(detector: OpenWakeWord) -> None:
    np.random.seed(0)  # openWakeWord seeds its feature buffer with random noise on reset
    detector.reset()


def count_events(scores: Float32, threshold: float, refractory_frames: int) -> int:
    events = 0
    last = -refractory_frames - 1
    above = False
    for i, s in enumerate(scores):
        now_above = s >= threshold
        if now_above and not above and i - last > refractory_frames:
            events += 1
            last = i
        above = now_above
    return events


def _mix(speech: Float32, noise: Float32, snr_db: float) -> Float32:
    p_speech = float(np.mean(np.square(speech[np.abs(speech) > 1e-4]))) or 1e-8
    p_noise = float(np.mean(np.square(noise))) or 1e-8
    gain = np.sqrt(p_speech / (p_noise * 10 ** (snr_db / 10)))
    return np.asarray(speech + gain * noise, dtype=np.float32)


def _condition(clip: Float32, name: str, rng: np.random.Generator, babble: Float32) -> Float32:
    lead = np.zeros(int(LEAD_S * SAMPLE_RATE), dtype=np.float32)
    tail = np.zeros(int(TAIL_S * SAMPLE_RATE), dtype=np.float32)
    x = np.concatenate([lead, clip, tail])
    if name == "clean":
        return x
    if name.startswith("quiet"):
        return (x * 0.1).astype(np.float32)
    if name.startswith("white"):
        noise = rng.standard_normal(len(x)).astype(np.float32)
        return _mix(x, noise, 10.0)
    start = int(rng.integers(0, len(babble) - len(x)))
    snr = 10.0 if "10 dB" in name else 0.0
    return _mix(x, babble[start : start + len(x)], snr)


def _negatives_worker(args: tuple[str, str, list[str]]) -> dict[str, Any]:
    models_dir, model, paths = args
    detector = OpenWakeWord(Path(models_dir), model)
    _reset(detector)
    audio = np.concatenate([load_audio(Path(p)) for p in paths])
    scores = _scores(detector, audio)
    refractory = round(REFRACTORY_S / FRAME_S)
    peak = int(np.argmax(scores))
    return {
        "seconds": len(audio) / SAMPLE_RATE,
        "events": {str(t): count_events(scores, t, refractory) for t in THRESHOLDS},
        "max_score": float(scores[peak]),
        "max_at_s": round(peak * FRAME_S, 2),
        "first_file": Path(paths[0]).stem,
    }


def run(cfg: Config, workers: int = 4, negatives_limit: int | None = None) -> dict[str, Any]:
    root = librispeech.ensure(cfg.data_dir)
    utts = librispeech.utterances(root)
    tts = build_tts(cfg)
    detector = OpenWakeWord(cfg.models_dir, cfg.wake.model)
    rng = np.random.default_rng(1234)
    babble = np.concatenate([load_audio(u.path) for u in utts[::40][:40]])

    # ---- positives
    voices = [v for v in tts.voices if v[:3] in ("af_", "am_", "bf_", "bm_")]
    per_condition: dict[str, list[float]] = {c: [] for c in CONDITIONS}
    per_voice: dict[str, list[float]] = {}
    for vi, voice in enumerate(voices):
        for phrase in PHRASES:
            for speed in SPEEDS:
                audio, rate = tts.synthesize(phrase, voice=voice, speed=speed)
                clip = resample(audio, rate, SAMPLE_RATE)
                for condition in CONDITIONS:
                    _reset(detector)
                    scores = _scores(detector, _condition(clip, condition, rng, babble))
                    peak = float(np.max(scores))
                    per_condition[condition].append(peak)
                    per_voice.setdefault(voice, []).append(peak)
        print(f"  [wake] positives: voice {vi + 1}/{len(voices)} {voice}", file=sys.stderr)

    def frr(peaks: list[float], t: float) -> float:
        return round(sum(1 for p in peaks if p < t) / len(peaks), 4)

    all_peaks = [p for peaks in per_condition.values() for p in peaks]
    positives = {
        "voices": voices,
        "phrases": list(PHRASES),
        "speeds": list(SPEEDS),
        "clips_per_condition": len(per_condition[CONDITIONS[0]]),
        "frr": {
            c: {str(t): frr(peaks, t) for t in THRESHOLDS} for c, peaks in per_condition.items()
        },
        "frr_all": {str(t): frr(all_peaks, t) for t in THRESHOLDS},
        "frr_by_voice_at_0.5": {v: frr(p, 0.5) for v, p in per_voice.items()},
    }

    # ---- negatives
    chapters: dict[str, list[str]] = {}
    for u in utts:
        chapters.setdefault(u.chapter, []).append(str(u.path))
    jobs = [(str(cfg.models_dir), cfg.wake.model, paths) for paths in chapters.values()]
    if negatives_limit:
        jobs = jobs[:negatives_limit]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, r in enumerate(pool.map(_negatives_worker, jobs)):
            results.append(r)
            if (i + 1) % 10 == 0:
                print(f"  [wake] negatives: chapter {i + 1}/{len(jobs)}", file=sys.stderr)
    hours = sum(r["seconds"] for r in results) / 3600
    events = {str(t): sum(r["events"][str(t)] for r in results) for t in THRESHOLDS}
    worst = sorted(results, key=lambda r: r["max_score"], reverse=True)[:5]
    negatives = {
        "dataset": "LibriSpeech test-clean (all chapters, each concatenated)",
        "chapters": len(results),
        "hours": round(hours, 2),
        "false_accepts": events,
        "false_accepts_per_hour": {t: round(n / hours, 3) for t, n in events.items()},
        "highest_scores": [
            {
                "chapter_first_file": r["first_file"],
                "score": round(r["max_score"], 4),
                "at_s": r["max_at_s"],
            }
            for r in worst
        ],
    }
    return {
        "model": cfg.wake.model,
        "thresholds": list(THRESHOLDS),
        "positives": positives,
        "negatives": negatives,
        "hardware": hardware(),
    }
