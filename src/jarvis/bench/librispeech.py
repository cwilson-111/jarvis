"""LibriSpeech test-clean (CC BY 4.0), fetched on demand into ``data/``."""

from __future__ import annotations

import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

from jarvis.models import fetch

URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
SIZE_MB = 331


@dataclass(frozen=True)
class Utterance:
    id: str
    path: Path
    text: str

    @property
    def chapter(self) -> str:
        speaker, chapter, _ = self.id.split("-")
        return f"{speaker}-{chapter}"


def ensure(data_dir: Path) -> Path:
    root = data_dir / "librispeech" / "LibriSpeech" / "test-clean"
    if root.is_dir() and any(root.glob("*/*/*.trans.txt")):
        return root
    archive = data_dir / "librispeech" / "test-clean.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        print(f"  downloading LibriSpeech test-clean (~{SIZE_MB} MB)", file=sys.stderr)
        partial = archive.with_suffix(".part")
        fetch(URL, partial)
        partial.replace(archive)
    print("  extracting", file=sys.stderr)
    with tarfile.open(archive) as tar:
        tar.extractall(archive.parent, filter="data")
    return root


def utterances(root: Path) -> list[Utterance]:
    out = []
    for trans in sorted(root.glob("*/*/*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            utt_id, _, text = line.partition(" ")
            out.append(Utterance(utt_id, trans.parent / f"{utt_id}.flac", text))
    return sorted(out, key=lambda u: u.id)


def spread_subset(items: list[Utterance], n: int) -> list[Utterance]:
    """Every k-th utterance, so the subset spans all speakers deterministically."""
    if n >= len(items):
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]
