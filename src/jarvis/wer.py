"""Word error rate with a small, documented text normaliser.

Normalisation (applied to reference and hypothesis alike): lowercase, digits
spelled out as words (LibriSpeech references spell numbers; STT engines often
write digits), hyphens to spaces, all punctuation except in-word apostrophes
removed, whitespace collapsed. Corpus WER is total edits over total reference
words, not the mean of per-utterance WERs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen",
]  # fmt: skip
_TENS = ["_", "_", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_NUMBER = re.compile(r"\d[\d,]*")
_NON_WORD = re.compile(r"[^a-z0-9' ]+")
_EDGE_APOSTROPHE = re.compile(r"(^|\s)'+|'+(\s|$)")


def number_to_words(n: int) -> str:
    if n < 0:
        raise ValueError("negative numbers are not supported")
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + ("" if ones == 0 else " " + _ONES[ones])
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return _ONES[hundreds] + " hundred" + ("" if rest == 0 else " " + number_to_words(rest))
    for scale, word in ((10**9, "billion"), (10**6, "million"), (10**3, "thousand")):
        if n >= scale:
            head, rest = divmod(n, scale)
            tail = "" if rest == 0 else " " + number_to_words(rest)
            return number_to_words(head) + " " + word + tail
    raise AssertionError("unreachable")


def normalize(text: str) -> list[str]:
    text = text.lower().replace("-", " ").replace("\u2019", "'")

    def spell(match: re.Match[str]) -> str:
        digits = match.group(0).replace(",", "")
        return " " + number_to_words(int(digits)) + " " if digits.isdigit() else " "

    text = _NUMBER.sub(spell, text)
    text = _NON_WORD.sub(" ", text)
    text = _EDGE_APOSTROPHE.sub(" ", text)
    return text.split()


@dataclass(frozen=True)
class WerCounts:
    substitutions: int
    deletions: int
    insertions: int
    ref_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        if self.ref_words == 0:
            return 0.0 if self.errors == 0 else float("inf")
        return self.errors / self.ref_words

    def __add__(self, other: WerCounts) -> WerCounts:
        return WerCounts(
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
            self.ref_words + other.ref_words,
        )


def align(ref: list[str], hyp: list[str]) -> WerCounts:
    """Levenshtein alignment over words; ties prefer substitution, then deletion."""
    # cost[j] holds (edits, subs, dels, ins) for ref[:i] vs hyp[:j]
    prev = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i in range(1, len(ref) + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur.append(prev[j - 1])
                continue
            s, d, n = prev[j - 1], prev[j], cur[j - 1]
            best = min(
                (s[0] + 1, 0, (s[1] + 1, s[2], s[3])),
                (d[0] + 1, 1, (d[1], d[2] + 1, d[3])),
                (n[0] + 1, 2, (n[1], n[2], n[3] + 1)),
            )
            cur.append((best[0], *best[2]))
        prev = cur
    _, subs, dels, ins = prev[-1]
    return WerCounts(subs, dels, ins, len(ref))


def wer_counts(reference: str, hypothesis: str) -> WerCounts:
    return align(normalize(reference), normalize(hypothesis))


def wer(reference: str, hypothesis: str) -> float:
    return wer_counts(reference, hypothesis).wer
