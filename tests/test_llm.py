"""Ollama client against a mocked transport (no network, no Ollama)."""

import json
import threading

import httpx
import pytest

from jarvis.config import LlmConfig
from jarvis.llm import LlmError, Message, OllamaLlm, parse_chunk


def ndjson(*objs: object) -> bytes:
    return b"".join(json.dumps(o).encode() + b"\n" for o in objs)


def chunk(text: str) -> dict[str, object]:
    return {"model": "llama3.2", "message": {"role": "assistant", "content": text}, "done": False}


FINAL = {"model": "llama3.2", "done": True, "done_reason": "stop", "eval_count": 3}


def client(handler: httpx.MockTransport, **cfg: object) -> OllamaLlm:
    return OllamaLlm(LlmConfig(**cfg), transport=handler)  # type: ignore[arg-type]


def test_streams_deltas_and_records_stats() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, content=ndjson(chunk("Hello"), chunk(" there."), FINAL))

    llm = client(httpx.MockTransport(handler), seed=5, max_tokens=50)
    out = list(llm.stream([Message("user", "hi")], threading.Event()))
    assert out == ["Hello", " there."]
    assert llm.last_stats["model"] == "llama3.2"
    assert llm.last_stats["eval_count"] == 3
    assert seen["model"] == "llama3.2"
    assert seen["stream"] is True
    assert seen["options"] == {"temperature": 0.6, "num_predict": 50, "seed": 5}


def test_cancel_stops_iteration() -> None:
    body = ndjson(*(chunk(f"t{i} ") for i in range(50)), FINAL)
    llm = client(httpx.MockTransport(lambda _r: httpx.Response(200, content=body)))
    cancel = threading.Event()
    got = []
    for delta in llm.stream([Message("user", "hi")], cancel):
        got.append(delta)
        if len(got) == 3:
            cancel.set()
    assert len(got) == 3


def test_http_error_status_is_an_llm_error() -> None:
    llm = client(
        httpx.MockTransport(lambda _r: httpx.Response(404, json={"error": "model not found"}))
    )
    with pytest.raises(LlmError, match="404"):
        list(llm.stream([Message("user", "hi")], threading.Event()))


def test_connection_failure_is_an_llm_error() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    llm = client(httpx.MockTransport(refuse))
    with pytest.raises(LlmError, match="failed"):
        list(llm.stream([Message("user", "hi")], threading.Event()))


def test_stream_without_done_is_an_error() -> None:
    llm = client(httpx.MockTransport(lambda _r: httpx.Response(200, content=ndjson(chunk("a")))))
    with pytest.raises(LlmError, match="without a final"):
        list(llm.stream([Message("user", "hi")], threading.Event()))


def test_parse_chunk_never_invents_a_default() -> None:
    assert parse_chunk('{"message": {"content": "x"}}')["message"]["content"] == "x"
    with pytest.raises(LlmError, match="malformed"):
        parse_chunk("{not json")
    with pytest.raises(LlmError, match="unexpected"):
        parse_chunk("[1, 2]")
    with pytest.raises(LlmError, match="model is loading"):
        parse_chunk('{"error": "model is loading"}')
