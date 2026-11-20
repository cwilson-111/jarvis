"""Turn a stream of LLM tokens into speakable sentences.

The chunker only commits to a boundary once it has seen the character *after*
the punctuation, so ``3.14``, ``e.g.`` and ``Dr. Smith`` are not split, and a
sentence is released the moment the next token starts it. Fragments shorter
than ``min_chars`` are held back and merged with what follows so the TTS is not
asked to say "Sure." on its own; overly long run-ons are split at the last
comma or space before ``max_chars``.
"""

from __future__ import annotations

import re

_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
        "approx", "fig", "inc", "ltd", "mt", "ft", "a.m", "p.m", "u.s",
    }
)  # fmt: skip
_TERMINATORS = ".!?"
_CLOSERS = "\"')]\u201d\u2019"
_CLAUSE = ",;:"
_LIST_MARKER = re.compile(r"^\s*(\d+|[a-zA-Z])[.)]$")
_MARKDOWN = re.compile(r"[*_#`>|~]+")
_SPACES = re.compile(r"\s+")


def speakable(text: str) -> str:
    """Strip markdown/control characters an LLM may emit despite the prompt."""
    text = "".join(ch if ch.isprintable() else " " for ch in text)
    text = _MARKDOWN.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


class SentenceChunker:
    def __init__(
        self, min_chars: int = 10, max_chars: int = 220, eager_first: bool = False
    ) -> None:
        """``eager_first``: release the *first* chunk of a reply at a clause boundary
        (", " "; " ": ") instead of waiting for the end of the sentence, so speech can
        start sooner. Later chunks are always whole sentences."""
        if not max_chars > min_chars >= 1:
            raise ValueError("need max_chars > min_chars >= 1")
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.eager_first = eager_first
        self._emitted = 0
        self._buf = ""

    def feed(self, token: str) -> list[str]:
        """Add a token; return any sentences that are now complete."""
        self._buf += token
        out: list[str] = []
        while True:
            cut = self._find_boundary()
            if cut is None:
                break
            sentence = speakable(self._buf[:cut])
            self._buf = self._buf[cut:].lstrip()
            if sentence:
                out.append(sentence)
                self._emitted += 1
        return out

    def flush(self) -> list[str]:
        """End of stream: return whatever is left."""
        rest = speakable(self._buf)
        self._buf = ""
        self._emitted = 0
        return [rest] if rest else []

    def _find_boundary(self) -> int | None:
        buf = self._buf
        # Newlines are hard boundaries (LLMs use them between list items).
        search_from = 0
        while True:
            nl = buf.find("\n", search_from)
            if nl == -1:
                break
            if len(speakable(buf[:nl])) >= self.min_chars:
                return nl + 1
            search_from = nl + 1
        i = 0
        while i < len(buf):
            ch = buf[i]
            if ch in _TERMINATORS:
                end = i + 1
                while end < len(buf) and (buf[end] in _TERMINATORS or buf[end] in _CLOSERS):
                    end += 1
                if end >= len(buf):
                    break  # need to see the next character before committing
                if buf[end].isspace() and self._is_sentence_end(buf[:end]):
                    return end
                i = end
                continue
            if (
                self.eager_first
                and self._emitted == 0
                and ch in _CLAUSE
                and i + 1 < len(buf)
                and buf[i + 1] == " "
                and len(speakable(buf[: i + 1])) >= self.min_chars
            ):
                return i + 1
            i += 1
        if len(buf) > self.max_chars:
            return self._soft_split(buf)
        return None

    def _is_sentence_end(self, candidate: str) -> bool:
        text = candidate.strip()
        if len(speakable(text)) < self.min_chars:
            return False
        if _LIST_MARKER.match(text.rsplit("\n", 1)[-1]):
            return False
        if text.endswith("."):
            words = text[:-1].split()
            last = words[-1].lower().strip("\"'([") if words else ""
            if last in _ABBREVIATIONS:
                return False
            if len(last) == 1 and last.isalpha():  # initials: "J. R. R. Tolkien"
                return False
        return True

    def _soft_split(self, buf: str) -> int:
        window = buf[: self.max_chars]
        for sep in (", ", "; ", ": ", " - ", " "):
            idx = window.rfind(sep)
            if idx >= self.min_chars:
                return idx + len(sep)
        return self.max_chars
