"""Per-turn timing instrumentation.

Every stage of a turn records a monotonic timestamp (``time.perf_counter`` by
default, injectable for tests). Marks are first-write-wins so concurrent stages
can call ``mark("first_audio_out")`` freely; only the earliest sticks. Derived
durations are computed from the marks, never measured separately, so the
breakdown always sums consistently.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

Clock = Callable[[], float]

# Canonical order of marks within a turn. Not every turn has every mark (e.g. a
# push-to-talk turn has no wake_detected; a cancelled turn may stop early).
MARKS: tuple[str, ...] = (
    "wake_detected",
    "speech_start",
    "speech_end",
    "endpoint",
    "stt_done",
    "llm_first_token",
    "first_sentence_ready",
    "first_audio_ready",
    "first_audio_out",
    "llm_done",
    "barge_in",
    "cancelled",
    "done",
)

# name -> (from mark, to mark). These are the numbers the benchmarks report.
STAGES: dict[str, tuple[str, str]] = {
    "endpointing": ("speech_end", "endpoint"),
    "stt": ("endpoint", "stt_done"),
    "llm_ttft": ("stt_done", "llm_first_token"),
    "first_sentence": ("llm_first_token", "first_sentence_ready"),
    "tts_first": ("first_sentence_ready", "first_audio_ready"),
    "playback_start": ("first_audio_ready", "first_audio_out"),
    "compute": ("endpoint", "first_audio_out"),
    "e2e": ("speech_end", "first_audio_out"),
    "llm_total": ("stt_done", "llm_done"),
    "barge_in_stop": ("barge_in", "cancelled"),
}


@dataclass
class TurnTrace:
    turn: int
    session: str
    clock: Clock = field(default=time.perf_counter, repr=False)
    marks: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def mark(self, name: str, at: float | None = None) -> float:
        """Record ``name`` at ``at`` (default: now). The first write wins."""
        if name not in MARKS:
            raise KeyError(f"unknown trace mark {name!r}")
        t = self.clock() if at is None else at
        with self._lock:
            return self.marks.setdefault(name, t)

    def has(self, name: str) -> bool:
        return name in self.marks

    def unmark(self, *names: str) -> None:
        """Forget marks (used when a turn restarts listening after a bare wake phrase)."""
        with self._lock:
            for name in names:
                self.marks.pop(name, None)

    def duration_ms(self, start: str, end: str) -> float | None:
        if start not in self.marks or end not in self.marks:
            return None
        return (self.marks[end] - self.marks[start]) * 1000.0

    def stages_ms(self) -> dict[str, float | None]:
        return {name: self.duration_ms(a, b) for name, (a, b) in STAGES.items()}

    def relative_ms(self) -> dict[str, float]:
        """Marks in canonical order, in ms relative to the earliest mark."""
        if not self.marks:
            return {}
        origin = min(self.marks.values())
        return {
            name: round((self.marks[name] - origin) * 1000.0, 2)
            for name in MARKS
            if name in self.marks
        }

    def to_record(self) -> dict[str, Any]:
        stages = {k: (None if v is None else round(v, 2)) for k, v in self.stages_ms().items()}
        return {
            "session": self.session,
            "turn": self.turn,
            "started_at": self.started_at,
            "marks_ms": self.relative_ms(),
            "stages_ms": stages,
            **self.meta,
        }


class TraceWriter:
    """Appends one JSON object per turn to ``<dir>/<session>.jsonl``."""

    def __init__(self, directory: Path | None, session: str | None = None) -> None:
        self.session = session or datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.path: Path | None = None
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            self.path = directory / f"{self.session}.jsonl"
        self._lock = threading.Lock()

    def write(self, trace: TurnTrace) -> None:
        if self.path is None:
            return
        line = json.dumps(trace.to_record(), ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile (same definition as numpy's default)."""
    if not values:
        raise ValueError("percentile of empty list")
    if not 0.0 <= pct <= 100.0:
        raise ValueError("pct must be in [0, 100]")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)
