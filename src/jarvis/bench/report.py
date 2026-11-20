"""Render docs/RESULTS.md from docs/results/*.json.

Only numbers present in the JSON files are printed; nothing here is typed in by
hand. Model-generated text is never copied into the Markdown.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jarvis.bench.common import RESULTS_DIR, load_result

OUT = Path("docs/RESULTS.md")

NA = "n/a"
STAGE_ROWS = (
    ("endpointing", "End of speech -> endpoint (VAD silence wait)"),
    ("stt", "Endpoint -> transcript (Parakeet)"),
    ("llm_ttft", "Transcript -> first LLM token"),
    ("first_sentence", "First token -> first speakable chunk"),
    ("tts_first", "Chunk -> first audio synthesised (Kokoro)"),
    ("playback_start", "Audio ready -> playback starts"),
    ("compute", "**Endpoint -> first audio out**"),
    ("e2e", "**End of speech -> first audio out**"),
)
MODE_LABELS = {
    "clause": "streaming, first chunk at a clause (default)",
    "sentence": "streaming, whole sentences",
    "batch": "no streaming (whole reply, then TTS)",
}


def _ms(stat: dict[str, Any], key: str) -> str:
    value = stat.get(key)
    return NA if value is None else f"{value:,.0f}"


def _hardware_line(hw: dict[str, Any]) -> str:
    chip = hw.get("chip") or hw.get("machine")
    os_name = f"macOS {hw['macos']}" if hw.get("macos") else hw.get("platform")
    return (
        f"{chip} ({hw.get('model', '?')}), {hw.get('ram_gb')} GB RAM, {os_name}, "
        f"Python {hw.get('python')}"
    )


def _run_block(data: dict[str, Any]) -> list[str]:
    return [
        f"- Command: `{data.get('command', '?')}`",
        f"- Run: {data.get('generated_at', '?')} (wall time {data.get('wall_s', '?')} s)",
        f"- Hardware: {_hardware_line(data.get('hardware', {}))}",
    ]


def _latency(data: dict[str, Any]) -> list[str]:
    modes: dict[str, Any] = data["modes"]
    cfg = data.get("config", {})
    ollama = data.get("ollama", {})
    lines = ["## End-to-end latency", ""]
    lines += _run_block(data)
    turns = ", ".join(f"{m}: {v['turns']}" for m, v in modes.items())
    lines += [
        f"- Turns per mode: {turns} (plus one untimed warm-up turn each)",
        f"- LLM: `{cfg.get('llm_model')}` via Ollama {ollama.get('version', '?')} "
        f"({ollama.get('parameter_size', '?')}, {ollama.get('quantization', '?')}), "
        f"temperature {cfg.get('llm_temperature')}, seed {cfg.get('llm_seed')}, "
        f"max {cfg.get('llm_max_tokens')} tokens",
        f"- STT `{cfg.get('stt')}`, TTS `{cfg.get('tts')}`, wake `{cfg.get('wake')}` "
        f"(threshold {cfg.get('wake_threshold')}), VAD end-of-speech silence "
        f"{cfg.get('vad_end_silence_ms')} ms",
        '- Input: 30 fixed requests, each spoken as "Hey Jarvis, <request>" by one of '
        f"{len(cfg.get('speaker_voices', []))} Kokoro voices "
        f"({', '.join(cfg.get('speaker_voices', []))}), replayed in real time through the "
        "full pipeline. Playback is simulated, so *first audio out* excludes the output "
        "device buffer (CoreAudio reports ~10-20 ms).",
        "",
        "Milliseconds, p50 / p95 over all turns:",
        "",
    ]
    header = "| Stage | " + " | ".join(MODE_LABELS.get(m, m) for m in modes) + " |"
    lines += [header, "|---|" + "---|" * len(modes)]
    for key, label in STAGE_ROWS:
        cells = []
        for m in modes:
            stat = modes[m]["stages_ms"].get(key, {})
            cells.append(f"{_ms(stat, 'p50')} / {_ms(stat, 'p95')}" if stat.get("n") else NA)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    extra = (
        ("llm_total", "Transcript -> LLM finished (whole reply)"),
        ("playback_total_ms", "Reply audio length (first audio -> end)"),
        ("stall_ms", "Playback stalls, turns with any (sum per turn)"),
    )
    for key, label in extra:
        cells = []
        for m in modes:
            stat = modes[m]["stages_ms"].get(key, {})
            if not stat.get("n"):
                cells.append("none" if key == "stall_ms" else NA)
                continue
            cell = f"{_ms(stat, 'p50')} / {_ms(stat, 'p95')}"
            if key == "stall_ms":
                cell += f" ({stat['n']} of {modes[m]['turns']} turns)"
            cells.append(cell)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    for m, v in modes.items():
        if v.get("failures"):
            failed = ", ".join(v["failures"])
            lines.append(f"- {m}: {len(v['failures'])} input(s) produced no turn: {failed}")
        if v.get("transcript_wer_mean") is not None:
            lines.append(
                f"- {MODE_LABELS.get(m, m)}: mean WER of the pipeline's own transcripts vs. the "
                f"request text {v['transcript_wer_mean']:.3f}"
            )
    lines.append(
        f"- Ollama cold model load before the run: {data.get('ollama_cold_load_s')} s "
        "(not included above; the assistant loads the model at startup)"
    )
    lines.append("")
    barge = data.get("barge_in")
    if barge:
        lines += [
            "### Barge-in",
            "",
            f'{barge["total"]} trials: a request for a story, then "Hey Jarvis. Stop." '
            f"{barge['interrupt_after_s']} s after the request ended. Interrupted: "
            f"{barge['interrupted']} / {barge['total']}; reply audio had already started in "
            f"{barge.get('audio_had_started', NA)}, the LLM was still generating in "
            f"{barge.get('llm_still_generating', NA)}. Detection time is measured from the "
            "last voiced sample of the wake phrase, so it can be negative when the model fires "
            'on "Jarv-"; frames are 80 ms.',
            "",
            "| Measure | p50 | p95 | max |",
            "|---|---|---|---|",
        ]
        for key, label in (
            ("wake_detect_ms", 'End of "Hey Jarvis" -> barge-in detected'),
            ("stop_audio_ms", "Barge-in detected -> playback stopped"),
            ("wind_down_ms", "Barge-in detected -> LLM/TTS threads finished"),
        ):
            stat = barge.get(key, {})
            lines.append(
                f"| {label} | {_ms(stat, 'p50')} | {_ms(stat, 'p95')} | {_ms(stat, 'max')} |"
            )
        lines.append("")
    mem = data.get("memory")
    if mem:
        lines += [
            "### Peak memory (during the latency run above)",
            "",
            "| Measure | MB |",
            "|---|---|",
            f"| Peak RSS, jarvis process (wake, VAD, Parakeet, Kokoro, pipeline) | "
            f"{mem['peak_jarvis_rss_mb']:,} |",
            f"| Peak RSS, Ollama processes | {mem['peak_ollama_rss_mb']:,} |",
            f"| Peak RSS, combined | {mem['peak_combined_rss_mb']:,} |",
            f"| Ollama's own report of the loaded model (`/api/ps` size) | "
            f"{ollama.get('loaded_size_mb') or NA} |",
            f"| Lowest system 'available' memory seen | {mem['min_system_available_mb']:,} |",
            f"| Swap used at start -> peak | {mem['swap_used_start_mb']:,} -> "
            f"{mem['swap_used_peak_mb']:,} |",
            "",
            f"Sampled every {mem['interval_s']} s ({mem['samples']} samples) with psutil while "
            "the benchmark ran. The Ollama figure is its server plus the `llama-server` runner "
            "holding the model. 'Available' memory and swap are machine-wide and include "
            "every other app open at the time (editor, browser, terminal agents).",
            "",
        ]
    return lines


def _stt(data: dict[str, Any]) -> list[str]:
    lines = ["## Speech-to-text accuracy", ""]
    lines += _run_block(data)
    lines += [
        f"- Data: {data['dataset']}, {data['utterances']} utterances from "
        f"{data['speakers']} speakers, {data['audio_s'] / 60:.1f} min of audio; "
        f"selection: {data['selection']}",
        "- WER normalisation (both sides): lowercase, digits spelled out, punctuation "
        "removed; corpus WER = total edits / total reference words. Greedy decoding for all "
        "engines. RTF = transcription time / audio duration (lower is faster), one "
        "utterance at a time after one untimed warm-up.",
        "",
        "| Engine | WER | Sub / Del / Ins | Ref words | RTF | Threads | Load time |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, e in data["engines"].items():
        lines.append(
            f"| `{name}` | {e['wer'] * 100:.2f}% | {e['substitutions']} / {e['deletions']} / "
            f"{e['insertions']} | {e['ref_words']:,} | {e['rtf']:.3f} | {e['threads']} | "
            f"{e['load_s']:.1f} s |"
        )
    lines.append("")
    return lines


def _wake(data: dict[str, Any]) -> list[str]:
    pos = data["positives"]
    neg = data["negatives"]
    thresholds = [str(t) for t in data["thresholds"]]
    lines = ["## Wake word (`hey_jarvis_v0.1`)", ""]
    lines += _run_block(data)
    lines += [
        f"- Positives: {len(pos['voices'])} English Kokoro voices x {len(pos['phrases'])} "
        f"phrases ({' / '.join(repr(p) for p in pos['phrases'])}) x speeds "
        f"{', '.join(str(s) for s in pos['speeds'])} = {pos['clips_per_condition']} clips per "
        "condition. Babble noise is other LibriSpeech speech; SNR is relative to the wake "
        "clip's speech power.",
        f"- Negatives: {neg['dataset']}: {neg['chapters']} chapters, {neg['hours']} h. A false "
        "accept is a threshold crossing, with a 2 s refractory period.",
        "",
        "False-reject rate (lower is better):",
        "",
        "| Condition | " + " | ".join(f"t = {t}" for t in thresholds) + " |",
        "|---|" + "---|" * len(thresholds),
    ]
    for cond, rates in pos["frr"].items():
        cells = " | ".join(f"{rates[t] * 100:.1f}%" for t in thresholds)
        lines.append(f"| {cond} | {cells} |")
    cells = " | ".join(f"**{pos['frr_all'][t] * 100:.1f}%**" for t in thresholds)
    lines.append(f"| **all conditions** | {cells} |")
    lines += [
        "",
        "False accepts on non-wake speech:",
        "",
        "| | " + " | ".join(f"t = {t}" for t in thresholds) + " |",
        "|---|" + "---|" * len(thresholds),
        "| False accepts | " + " | ".join(str(neg["false_accepts"][t]) for t in thresholds) + " |",
        "| Per hour | "
        + " | ".join(f"{neg['false_accepts_per_hour'][t]:.2f}" for t in thresholds)
        + " |",
        "",
    ]
    worst_voices = sorted(pos["frr_by_voice_at_0.5"].items(), key=lambda kv: -kv[1])[:5]
    lines.append(
        "Voices with the most misses at t = 0.5 (all conditions): "
        + ", ".join(f"`{v}` {r * 100:.0f}%" for v, r in worst_voices)
    )
    lines += [
        "",
        "**Limits.** The positives are clean, close-mic synthetic speech, and the wake model "
        "itself was trained on synthetic speech (from other TTS engines), so these "
        "false-reject rates are a best case: they say nothing about real accents, distance "
        f"from the mic, room echo or children, and {len(pos['voices'])} voices is a small "
        "speaker sample. "
        "LibriSpeech is read audiobook speech; TV, music and conversation will behave "
        "differently, so treat the false-accept rate as a lower bound for a noisy home.",
        "",
    ]
    return lines


def _tts(data: dict[str, Any]) -> list[str]:
    lines = ["## TTS precision (why int8 Kokoro)", ""]
    lines += _run_block(data)
    first = next(iter(data["variants"].values()))
    columns = " | ".join(f"{r['chars']}-char sentence" for r in first["rows"])
    lines += [
        f"- Voice `{data['voice']}`; {data['metric']}.",
        "",
        f"| Kokoro export | Size | RTF (all sentences) | {columns} |",
        "|---|---|---|" + "---|" * len(first["rows"]),
    ]
    for name, v in data["variants"].items():
        cells = " | ".join(f"{r['synth_ms']} ms for {r['audio_s']} s" for r in v["rows"])
        lines.append(f"| `{name}` | {v['size_mb']} MB | {v['rtf']:.3f} | {cells} |")
    lines.append("")
    return lines


def render(results_dir: Path = RESULTS_DIR, out: Path = OUT) -> Path:
    lines = [
        "# Results",
        "",
        "Generated by `jarvis bench report` from the JSON files in `docs/results/` "
        "(each holds per-turn or per-utterance data, versions and hardware). Every number "
        "below comes from a run of the command shown with it.",
        "",
    ]
    sections = (("latency", _latency), ("stt", _stt), ("wake", _wake), ("tts", _tts))
    for name, fn in sections:
        data = load_result(name, results_dir)
        if data is not None:
            lines += fn(data)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out
