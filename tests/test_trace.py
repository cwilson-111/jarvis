import json
from pathlib import Path

import numpy as np
import pytest

from jarvis.trace import STAGES, TraceWriter, TurnTrace, percentile


class StepClock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def full_trace() -> TurnTrace:
    clock = StepClock()
    trace = TurnTrace(3, "s1", clock=clock)
    timeline = {
        "wake_detected": 100.000,
        "speech_start": 100.400,
        "speech_end": 102.000,
        "endpoint": 102.640,
        "stt_done": 102.940,
        "llm_first_token": 103.090,
        "first_sentence_ready": 103.300,
        "first_audio_ready": 103.700,
        "first_audio_out": 103.701,
        "llm_done": 104.000,
        "done": 108.000,
    }
    for name, t in timeline.items():
        clock.t = t
        trace.mark(name)
    return trace


def test_stage_durations_come_from_marks() -> None:
    stages = full_trace().stages_ms()
    assert stages["endpointing"] == pytest.approx(640.0)
    assert stages["stt"] == pytest.approx(300.0)
    assert stages["llm_ttft"] == pytest.approx(150.0)
    assert stages["first_sentence"] == pytest.approx(210.0)
    assert stages["tts_first"] == pytest.approx(400.0)
    assert stages["playback_start"] == pytest.approx(1.0)
    assert stages["compute"] == pytest.approx(1061.0)
    assert stages["e2e"] == pytest.approx(1701.0)
    assert stages["llm_total"] == pytest.approx(1060.0)
    assert stages["barge_in_stop"] is None  # never interrupted


def test_pipeline_stages_sum_to_end_to_end() -> None:
    stages = full_trace().stages_ms()
    parts = ["endpointing", "stt", "llm_ttft", "first_sentence", "tts_first", "playback_start"]
    assert sum(stages[p] or 0.0 for p in parts) == pytest.approx(stages["e2e"])


def test_first_mark_wins_and_unknown_marks_are_rejected() -> None:
    clock = StepClock()
    trace = TurnTrace(1, "s", clock=clock)
    trace.mark("first_audio_out")
    clock.t = 200.0
    assert trace.mark("first_audio_out") == 100.0
    assert trace.mark("endpoint", at=150.0) == 150.0
    with pytest.raises(KeyError):
        trace.mark("first_audio")
    trace.unmark("endpoint")
    assert not trace.has("endpoint")


def test_missing_marks_give_none() -> None:
    trace = TurnTrace(1, "s", clock=StepClock())
    trace.mark("endpoint")
    assert trace.duration_ms("endpoint", "stt_done") is None
    assert set(trace.stages_ms()) == set(STAGES)


def test_record_is_relative_and_json_serialisable(tmp_path: Path) -> None:
    trace = full_trace()
    trace.meta["user_text"] = "hi"
    record = trace.to_record()
    assert record["marks_ms"]["wake_detected"] == 0.0
    assert record["marks_ms"]["done"] == 8000.0
    assert list(record["marks_ms"]) == [m for m in record["marks_ms"]]  # canonical order
    writer = TraceWriter(tmp_path, session="abc")
    writer.write(trace)
    writer.write(trace)
    lines = (tmp_path / "abc.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["user_text"] == "hi"


def test_writer_without_directory_is_silent(tmp_path: Path) -> None:
    writer = TraceWriter(None)
    writer.write(full_trace())
    assert writer.path is None


@pytest.mark.parametrize("pct", [0, 5, 50, 90, 95, 100])
def test_percentile_matches_numpy(pct: float) -> None:
    values = [3.0, 1.0, 7.5, 2.2, 9.9, 4.4, 4.4, 0.1]
    assert percentile(values, pct) == pytest.approx(float(np.percentile(values, pct)))


def test_percentile_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 101)
