"""Reply streaming, barge-in cancellation and the assistant loop, with fake engines."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

from jarvis.audio import FileSource, Int16
from jarvis.config import Config
from jarvis.llm import Conversation
from jarvis.pipeline import LLM_FAILURE_REPLY, Assistant, BargeInGuard, Reply, is_stop_command
from jarvis.trace import MARKS, TraceWriter, TurnTrace
from tests.fakes import (
    EnergyVad,
    FakeLlm,
    FakeTts,
    FixedStt,
    RecordingSink,
    ScriptedWake,
    frames_audio,
)

SENTENCES = ["Hello there, my friend. ", "This is the ", "second sentence. ", "And a third."]


def make_reply(llm: FakeLlm, tts: FakeTts, sink: RecordingSink, *, streaming: bool = True) -> Reply:
    return Reply(
        "hi",
        llm=llm,
        tts=tts,
        sink=sink,
        conversation=Conversation("system"),
        trace=TurnTrace(1, "test"),
        streaming=streaming,
    )


def wait_for(predicate: object, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if callable(predicate) and predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition not met in time")


# ------------------------------------------------------------------------------ reply


def test_streaming_speaks_first_sentence_before_llm_finishes() -> None:
    llm, tts, sink = FakeLlm(SENTENCES, delay_s=0.05), FakeTts(), RecordingSink()
    reply = make_reply(llm, tts, sink).start()
    result = reply.wait(5)
    marks = reply.trace.marks
    assert tts.texts == ["Hello there, my friend.", "This is the second sentence.", "And a third."]
    assert marks["first_audio_out"] < marks["llm_done"]
    assert result.text == "".join(SENTENCES).strip()
    assert not result.interrupted
    assert len(sink.chunks) == 3


def test_batch_mode_synthesises_whole_reply_after_llm_finishes() -> None:
    llm, tts, sink = FakeLlm(SENTENCES, delay_s=0.01), FakeTts(), RecordingSink()
    reply = make_reply(llm, tts, sink, streaming=False).start()
    reply.wait(5)
    marks = reply.trace.marks
    assert tts.texts == ["".join(SENTENCES).strip()]
    assert marks["llm_done"] <= marks["first_sentence_ready"] <= marks["first_audio_out"]


def test_cancel_stops_llm_tts_and_playback_promptly() -> None:
    tokens = [f"This is sentence number {i}. " for i in range(200)]
    llm, tts, sink = FakeLlm(tokens, delay_s=0.01), FakeTts(delay_s=0.01), RecordingSink()
    conversation = Conversation("system")
    reply = Reply(
        "tell me a story",
        llm=llm,
        tts=tts,
        sink=sink,
        conversation=conversation,
        trace=TurnTrace(1, "test"),
    ).start()
    wait_for(lambda: len(sink.chunks) >= 2)
    reply.cancel()
    result = reply.wait(2)

    assert reply.finished.is_set()
    assert result.interrupted
    assert llm.closed.is_set(), "LLM stream must be closed so Ollama stops generating"
    assert llm.yielded < len(tokens)
    assert sink.stops == 1
    assert reply.trace.has("cancelled")
    assert not reply.trace.has("llm_done")
    synthesized = len(tts.texts)
    time.sleep(0.1)
    assert len(tts.texts) == synthesized, "no TTS work after cancellation"
    # Nothing reaches the sink after cancel, and memory keeps what was heard.
    assert len(result.spoken) == len(sink.chunks)
    assert result.heard == result.spoken  # this fake sink "plays" chunks instantly
    history = conversation.messages("next")
    assert history[2].content == " ".join(result.heard)


class SlowSpeaker(RecordingSink):
    """Only the first queued chunk has started playing when we are interrupted."""

    def started_chunks(self) -> int:
        return min(1, len(self.chunks))

    def wait_done(self, cancelled: threading.Event) -> None:
        cancelled.wait(5)  # still "playing" until interrupted


def test_memory_keeps_only_sentences_that_started_playing() -> None:
    llm, tts, sink = FakeLlm(SENTENCES), FakeTts(), SlowSpeaker()
    conversation = Conversation("system")
    reply = Reply(
        "hi", llm=llm, tts=tts, sink=sink, conversation=conversation, trace=TurnTrace(1, "t")
    ).start()
    wait_for(lambda: len(sink.chunks) >= 3)
    reply.cancel()
    result = reply.wait(2)
    assert len(result.spoken) == 3
    assert result.heard == ["Hello there, my friend."]
    assert conversation.messages("next")[2].content == "Hello there, my friend."


def test_cancel_after_finish_is_a_no_op() -> None:
    llm, tts, sink = FakeLlm(["Short reply here."]), FakeTts(), RecordingSink()
    reply = make_reply(llm, tts, sink).start()
    reply.wait(2)
    reply.cancel()
    assert sink.stops == 0
    assert not reply.trace.has("cancelled")


def test_llm_failure_speaks_an_apology_and_is_not_remembered() -> None:
    llm, tts, sink = FakeLlm(SENTENCES, fail_after=0), FakeTts(), RecordingSink()
    reply = make_reply(llm, tts, sink).start()
    result = reply.wait(2)
    assert result.error is not None and "backend went away" in result.error
    assert tts.texts == [LLM_FAILURE_REPLY]
    assert len(reply.conversation) == 0


def test_conversation_keeps_last_n_turns() -> None:
    conv = Conversation("sys", history_turns=2)
    for i in range(4):
        conv.record(f"q{i}", f"a{i}")
    msgs = conv.messages("now")
    assert [m.content for m in msgs] == ["sys", "q2", "a2", "q3", "a3", "now"]
    assert [m.role for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]


# -------------------------------------------------------------------------- assistant


def make_assistant(
    wake_frames: set[int], stt: FixedStt, llm: FakeLlm, tts: FakeTts | None = None
) -> tuple[Assistant, RecordingSink]:
    sink = RecordingSink()
    assistant = Assistant(
        Config(),
        wake=ScriptedWake(wake_frames),
        vad=EnergyVad(),
        stt=stt,
        llm=llm,
        tts=tts or FakeTts(),
        sink=sink,
        writer=TraceWriter(None, session="test"),
    )
    return assistant, sink


def test_full_turn_marks_every_stage_in_order() -> None:
    # 5 silent frames, wake fires on frame 5, then 10 frames of speech, then silence.
    audio = frames_audio("." * 5 + "s" * 10 + "." * 20)
    stt = FixedStt("Hey Jarvis, what time is it?")
    llm = FakeLlm(["It is ", "noon. ", "Have a nice day."])
    assistant, sink = make_assistant({5}, stt, llm)

    traces = assistant.run(FileSource(audio, realtime=False))

    assert len(traces) == 1
    trace = traces[0]
    assert trace.meta["user_text"] == "what time is it?"
    assert trace.meta["outcome"] == "replied"
    assert llm.calls[0][-1].content == "what time is it?"
    expected = [
        "wake_detected", "speech_start", "speech_end", "endpoint", "stt_done",
        "llm_first_token", "first_sentence_ready", "first_audio_ready", "first_audio_out", "done",
    ]  # fmt: skip
    times = [trace.marks[m] for m in expected]
    assert times == sorted(times)
    assert set(trace.marks) <= set(MARKS)
    assert len(sink.chunks) == 2


def test_wake_word_during_reply_barges_in() -> None:
    audio = frames_audio("." * 5 + "s" * 10 + "." * 40)
    stt = FixedStt("Hey Jarvis, tell me a story.")
    llm = FakeLlm([f"Sentence {i} of a story. " for i in range(100)], delay_s=0.02)
    # Endpoint happens at frame 23; frame 30 is during the reply.
    assistant, sink = make_assistant({5, 30}, stt, llm, FakeTts(delay_s=0.01))

    traces = assistant.run(FileSource(audio, realtime=False))

    trace = traces[0]
    assert trace.meta["outcome"] == "interrupted"
    assert trace.meta["interrupted"] is True
    assert trace.has("barge_in") and trace.has("cancelled")
    assert trace.duration_ms("barge_in", "cancelled") is not None
    assert llm.closed.is_set()
    assert llm.yielded < 100
    assert sink.stops >= 1


def test_vad_barge_in_needs_sustained_speech() -> None:
    # After the endpoint: a 2-frame blip (too short to count), then sustained speech.
    audio = frames_audio("." * 5 + "s" * 10 + "." * 12 + "ss" + "." * 2 + "s" * 12 + "." * 30)
    llm = FakeLlm([f"Sentence {i} here. " for i in range(100)], delay_s=0.02)
    assistant, _sink = make_assistant({5}, FixedStt("Hey Jarvis, talk"), llm)
    assistant.cfg.pipeline.barge_in = "vad"
    assistant.guard.ignore_frames = 0
    traces = assistant.run(FileSource(audio, realtime=False), max_turns=1)
    assert len(traces) == 1
    assert traces[0].meta["outcome"] == "interrupted"
    assert llm.yielded < 100


class CountingWriter(TraceWriter):
    def __init__(self) -> None:
        super().__init__(None, session="count")
        self.writes = 0

    def write(self, trace: TurnTrace) -> None:
        self.writes += 1


def test_max_turns_writes_each_trace_once() -> None:
    """Regression: stopping at max_turns used to finish (and write) the last turn twice."""
    # Long trailing silence: the reply finishes while frames are still arriving.
    source = PausingSource("." * 5 + "s" * 10 + "." * 12, "." * 40, pause_s=0.2)
    assistant, _sink = make_assistant({5}, FixedStt("Hey Jarvis, hi"), FakeLlm(["Hello there."]))
    writer = CountingWriter()
    assistant.writer = writer
    traces = assistant.run(source, max_turns=1)
    assert len(traces) == 1
    assert writer.writes == 1


def test_barge_in_off_lets_reply_finish() -> None:
    audio = frames_audio("." * 5 + "s" * 10 + "." * 40)
    llm = FakeLlm(["One sentence here. ", "Two sentences here."], delay_s=0.02)
    assistant, _sink = make_assistant({5, 30}, FixedStt("Hey Jarvis, hi there"), llm)
    assistant.cfg.pipeline.barge_in = "off"
    traces = assistant.run(FileSource(audio, realtime=False))
    assert traces[0].meta["outcome"] == "replied"
    assert not traces[0].has("barge_in")


def test_stop_command_ends_turn_without_llm() -> None:
    audio = frames_audio("." * 5 + "s" * 6 + "." * 20)
    llm = FakeLlm(["never used"])
    assistant, sink = make_assistant({5}, FixedStt("Hey Jarvis, stop."), llm)
    traces = assistant.run(FileSource(audio, realtime=False))
    assert traces[0].meta["outcome"] == "stop_command"
    assert llm.calls == []
    assert sink.chunks == []


def test_bare_wake_phrase_keeps_listening_for_the_request() -> None:
    audio = frames_audio("." * 5 + "s" * 3 + "." * 12 + "s" * 8 + "." * 20)
    stt = FixedStt("Hey Jarvis.", "What's the weather?")
    llm = FakeLlm(["Sunny all day."])
    assistant, _sink = make_assistant({5}, stt, llm)
    traces = assistant.run(FileSource(audio, realtime=False))
    assert stt.calls == 2
    assert traces[0].meta["user_text"] == "What's the weather?"


class PausingSource:
    """Yields the frames of ``first``, waits (as a live mic would while the assistant
    talks), then yields ``second``."""

    def __init__(self, first: str, second: str, pause_s: float) -> None:
        self.parts = [
            FileSource(frames_audio(p), realtime=False, tail_silence_s=0) for p in (first, second)
        ]
        self.pause_s = pause_s

    def frames(self) -> Iterator[Int16]:
        yield from self.parts[0].frames()
        time.sleep(self.pause_s)
        yield from self.parts[1].frames()

    def close(self) -> None:
        return None


def test_no_wake_mode_answers_every_utterance() -> None:
    source = PausingSource("s" * 6 + "." * 15, "." * 2 + "s" * 6 + "." * 15, pause_s=0.3)
    stt = FixedStt("first question", "second question")
    llm = FakeLlm(["An answer."])
    assistant, _sink = make_assistant(set(), stt, llm)
    traces = assistant.run(source, trigger="always")
    assert [t.meta["user_text"] for t in traces] == ["first question", "second question"]
    assert not any(t.has("wake_detected") for t in traces)


def test_speech_while_replying_is_ignored_without_barge_in() -> None:
    """Without a wake word or VAD barge-in, talking over the assistant is not a new turn."""
    audio = frames_audio("s" * 6 + "." * 15 + "s" * 6 + "." * 15)
    llm = FakeLlm(["A slow answer. ", "With two sentences."], delay_s=0.1)
    assistant, _sink = make_assistant(set(), FixedStt("first", "second"), llm)
    traces = assistant.run(FileSource(audio, realtime=False), trigger="always")
    assert [t.meta["user_text"] for t in traces] == ["first"]


def test_no_wake_word_means_no_turn() -> None:
    audio = frames_audio("s" * 10 + "." * 20)
    assistant, _sink = make_assistant(set(), FixedStt("hello"), FakeLlm(["Hi."]))
    assert assistant.run(FileSource(audio, realtime=False)) == []


# --------------------------------------------------------------------------- helpers


def test_barge_in_guard_ignores_onset_and_needs_sustained_speech() -> None:
    guard = BargeInGuard(threshold=0.8, min_frames=3, ignore_frames=2)
    # Loud speech during the ignore window does not count.
    assert [guard.update(1.0) for _ in range(2)] == [False, False]
    # A dip resets the run.
    assert [guard.update(p) for p in (0.9, 0.9, 0.5, 0.9, 0.9)] == [False] * 5
    assert guard.update(0.95) is True
    guard.reset()
    assert guard.update(1.0) is False


def test_stop_commands() -> None:
    for text in ("stop", "Stop.", "okay stop please", "never mind", "Nevermind!", "that's enough"):
        assert is_stop_command(text), text
    for text in ("stop the timer", "how do I stop hiccups?", "cancel my meeting", ""):
        assert not is_stop_command(text), text
