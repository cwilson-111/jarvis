"""LLM engines (Ollama) and session conversation memory."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx

from jarvis.config import LlmConfig

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


class LlmError(RuntimeError):
    """The LLM backend failed or returned something we cannot parse."""


class Llm(Protocol):
    name: str

    def stream(self, messages: list[Message], cancel: threading.Event) -> Iterator[str]:
        """Yield text deltas; stop promptly (and release the request) once ``cancel`` is set."""
        ...


@dataclass
class Conversation:
    """System prompt plus the last ``history_turns`` user/assistant exchanges."""

    system_prompt: str
    history_turns: int = 8
    _turns: list[tuple[str, str]] = field(default_factory=list)

    def messages(self, user_text: str) -> list[Message]:
        msgs = [Message("system", self.system_prompt)]
        for user, assistant in self._turns:
            msgs += [Message("user", user), Message("assistant", assistant)]
        msgs.append(Message("user", user_text))
        return msgs

    def record(self, user_text: str, assistant_text: str) -> None:
        if self.history_turns == 0:
            return
        self._turns.append((user_text, assistant_text))
        del self._turns[: -self.history_turns]

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)


class OllamaLlm:
    """Streams from Ollama's ``/api/chat``. The model name is whatever the config says,
    and the model Ollama reports back is recorded in ``last_stats`` for the trace."""

    def __init__(self, cfg: LlmConfig, transport: httpx.BaseTransport | None = None) -> None:
        self.cfg = cfg
        self.name = f"ollama:{cfg.model}"
        self._client = httpx.Client(
            base_url=cfg.host,
            timeout=httpx.Timeout(cfg.read_timeout_s, connect=cfg.connect_timeout_s),
            transport=transport,
        )
        self.last_stats: dict[str, Any] = {}

    def _payload(self, messages: list[Message], max_tokens: int) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": self.cfg.temperature,
            "num_predict": max_tokens,
        }
        if self.cfg.seed is not None:
            options["seed"] = self.cfg.seed
        return {
            "model": self.cfg.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "keep_alive": self.cfg.keep_alive,
            "options": options,
        }

    def stream(self, messages: list[Message], cancel: threading.Event) -> Iterator[str]:
        return self._stream(messages, cancel, self.cfg.max_tokens)

    def _stream(
        self, messages: list[Message], cancel: threading.Event, max_tokens: int
    ) -> Iterator[str]:
        self.last_stats = {}
        payload = self._payload(messages, max_tokens)
        try:
            with self._client.stream("POST", "/api/chat", json=payload) as resp:
                if resp.status_code != httpx.codes.OK:
                    resp.read()
                    raise LlmError(f"Ollama HTTP {resp.status_code}: {resp.text[:300]}")
                for line in resp.iter_lines():
                    if cancel.is_set():
                        return  # leaving the context manager closes the connection
                    if not line.strip():
                        continue
                    chunk = parse_chunk(line)
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield str(content)
                    if chunk.get("done"):
                        self.last_stats = {
                            k: chunk.get(k)
                            for k in (
                                "model",
                                "done_reason",
                                "prompt_eval_count",
                                "eval_count",
                                "load_duration",
                                "prompt_eval_duration",
                                "eval_duration",
                            )
                        }
                        return
        except httpx.HTTPError as exc:
            raise LlmError(f"Ollama request to {self.cfg.host} failed: {exc}") from exc
        raise LlmError("Ollama stream ended without a final 'done' message")

    def warmup(self) -> float:
        """Load the model into memory; returns Ollama's reported load time in seconds."""
        for _ in self._stream([Message("user", "Hi")], threading.Event(), 1):
            pass
        return float(self.last_stats.get("load_duration") or 0) / 1e9

    def close(self) -> None:
        self._client.close()


def parse_chunk(line: str) -> dict[str, Any]:
    """Parse one NDJSON line from Ollama. Malformed output is an error, never a default."""
    try:
        chunk = json.loads(line)
    except json.JSONDecodeError as exc:
        raise LlmError(f"malformed chunk from Ollama: {line[:200]!r}") from exc
    if not isinstance(chunk, dict):
        raise LlmError(f"unexpected chunk from Ollama: {line[:200]!r}")
    if "error" in chunk:
        raise LlmError(f"Ollama error: {chunk['error']}")
    return chunk
