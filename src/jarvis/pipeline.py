"""The assistant: a frame loop (wake -> endpoint -> STT) and a streamed reply.

Threads per reply:

* producer: LLM token stream -> ``SentenceChunker`` -> sentence queue
* coordinator: sentence queue -> TTS -> sink (playback queue)

so the first sentence is synthesised and playing while the LLM is still
writing the second. Barge-in sets one ``cancelled`` event that every stage
checks; ``AudioSink.stop()`` silences output immediately, the LLM request is
closed (Ollama stops generating when the client disconnects) and queued
sentences are dropped. A TTS call already in flight cannot be interrupted, but
its output is discarded.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np

from jarvis.audio import FRAME_S, AudioSink, AudioSource, Float32, Int16, to_float32
from jarvis.chunker import SentenceChunker, speakable
from jarvis.config import Config
from jarvis.llm import Conversation, Llm, LlmError
from jarvis.stt import Stt
from jarvis.trace import Clock, TraceWriter, TurnTrace
from jarvis.tts import Tts
from jarvis.vad import Endpointer, Phase, Vad
from jarvis.wake import WakeWord, strip_wake_phrase

LLM_FAILURE_REPLY = "Sorry, I can't reach my language model right now."
EventHook = Callable[[str, str], None]
_STOP = re.compile(
    r"(?:ok(?:ay)?\W+)?(?:stop(?: talking)?|cancel|never ?mind|be quiet|quiet|shut up|"
    r"that'?s enough|enough)(?:\W+(?:please|thanks|thank you))?",
    re.IGNORECASE,
)


def _no_events(_kind: str, _text: str) -> None:
    return None


def is_stop_command(text: str) -> bool:
    """A bare "stop" / "never mind" ends the turn without asking the LLM anything."""
    return _STOP.fullmatch(text.strip(" .!?,")) is not None


# ----------------------------------------------------------------------------- reply


@dataclass
class ReplyResult:
    text: str
    spoken: list[str]  # sentences sent to the audio sink
    heard: list[str]  # of those, the ones that had started playing (== spoken unless cut off)
    interrupted: bool
    error: str | None = None
    llm_stats: dict[str, Any] = field(default_factory=dict)


class Reply:
    """One assistant reply, produced and played in background threads."""

    def __init__(
        self,
        user_text: str,
        *,
        llm: Llm,
        tts: Tts,
        sink: AudioSink,
        conversation: Conversation,
        trace: TurnTrace,
        streaming: bool = True,
        chunker: Callable[[], SentenceChunker] = SentenceChunker,
        sentence_pause_s: float = 0.12,
    ) -> None:
        self.user_text = user_text
        self.llm = llm
        self.tts = tts
        self.sink = sink
        self.conversation = conversation
        self.trace = trace
        self.streaming = streaming
        self._chunker = chunker
        self._pause_s = sentence_pause_s
        self.cancelled = threading.Event()
        self.finished = threading.Event()
        self._parts: list[str] = []
        self._spoken: list[str] = []
        self._heard: int | None = None  # set on cancel
        self._error: str | None = None
        # Serialises "check cancelled, then queue audio" against cancel(), so no chunk
        # can slip into the sink after it has been stopped.
        self._play_lock = threading.Lock()
        self._thread = threading.Thread(target=self._coordinate, name="reply", daemon=True)

    def start(self) -> Reply:
        self._thread.start()
        return self

    def cancel(self) -> None:
        """Barge-in: silence now, then let every stage wind down."""
        if self.finished.is_set():
            return
        with self._play_lock:
            self.cancelled.set()
            self._heard = self.sink.started_chunks()
            self.sink.stop()
        self.trace.mark("cancelled")

    def wait(self, timeout: float | None = None) -> ReplyResult:
        self._thread.join(timeout)
        heard = self._spoken if self._heard is None else self._spoken[: self._heard]
        return ReplyResult(
            text="".join(self._parts).strip(),
            spoken=list(self._spoken),
            heard=list(heard),
            interrupted=self.cancelled.is_set(),
            error=self._error,
            llm_stats=dict(getattr(self.llm, "last_stats", {}) or {}),
        )

    # -- threads ---------------------------------------------------------------

    def _produce(self, sentences: queue.Queue[str | None]) -> None:
        chunker = self._chunker()
        messages = self.conversation.messages(self.user_text)

        def emit(sentence: str) -> None:
            self.trace.mark("first_sentence_ready")
            sentences.put(sentence)

        try:
            stream = self.llm.stream(messages, self.cancelled)
            try:
                for delta in stream:
                    self.trace.mark("llm_first_token")
                    self._parts.append(delta)
                    if self.streaming:
                        for sentence in chunker.feed(delta):
                            emit(sentence)
                    if self.cancelled.is_set():
                        break
            finally:
                close = getattr(stream, "close", None)
                if close is not None:
                    close()
            if not self.cancelled.is_set():
                self.trace.mark("llm_done")
                if self.streaming:
                    for sentence in chunker.flush():
                        emit(sentence)
                else:
                    whole = speakable("".join(self._parts))
                    if whole:
                        emit(whole)
        except LlmError as exc:
            self._error = str(exc)
            if not self._spoken and not self.cancelled.is_set():
                emit(LLM_FAILURE_REPLY)
        finally:
            sentences.put(None)

    def _coordinate(self) -> None:
        sentences: queue.Queue[str | None] = queue.Queue()

        def on_first_audio(t: float) -> None:
            self.trace.mark("first_audio_out", t)

        self.sink.begin(on_first_audio)
        producer = threading.Thread(target=self._produce, args=(sentences,), name="llm")
        producer.start()
        try:
            while True:
                sentence = sentences.get()
                if sentence is None or self.cancelled.is_set():
                    break
                audio, rate = self.tts.synthesize(sentence)
                pause = np.zeros(int(self._pause_s * rate), dtype=np.float32)
                with self._play_lock:
                    if self.cancelled.is_set():
                        break
                    self.trace.mark("first_audio_ready")
                    self.sink.play(np.concatenate([audio, pause]), rate)
                    self._spoken.append(sentence)
            if not self.cancelled.is_set():
                self.sink.wait_done(self.cancelled)
        except Exception as exc:  # TTS/sink failure must not leave the loop hanging
            self._error = f"{type(exc).__name__}: {exc}"
            with self._play_lock:
                self.cancelled.set()
                self._heard = self.sink.started_chunks()
                self.sink.stop()
        finally:
            producer.join()
            done_at = self.trace.clock()
            end = getattr(self.sink, "playback_end", None)
            if isinstance(end, float) and not self.cancelled.is_set():
                done_at = max(done_at, end)  # simulated playback (FileSink) ends later
            self.trace.mark("done", done_at)
            self._remember()
            self.finished.set()

    def _remember(self) -> None:
        if self.cancelled.is_set():
            # Remember only what the user actually heard.
            heard = " ".join(self._spoken[: self._heard or 0])
            if heard:
                self.conversation.record(self.user_text, heard)
        elif self._parts and self._error is None:
            self.conversation.record(self.user_text, "".join(self._parts).strip())


# ------------------------------------------------------------------------- barge-in


class BargeInGuard:
    """Decides whether VAD speech during playback is the user, not our own echo.

    Without acoustic echo cancellation the mic hears the speaker, so this is a
    heuristic: ignore the first ``ignore_frames`` of playback, then require
    ``min_frames`` consecutive frames above a strict threshold. With
    headphones it works; with speakers expect self-interruptions on loud
    playback, which is why ``barge_in = "wake"`` is the default.
    """

    def __init__(self, threshold: float, min_frames: int, ignore_frames: int) -> None:
        self.threshold = threshold
        self.min_frames = min_frames
        self.ignore_frames = ignore_frames
        self.reset()

    def reset(self) -> None:
        self._frames = 0
        self._run = 0

    def update(self, prob: float) -> bool:
        self._frames += 1
        if self._frames <= self.ignore_frames:
            return False
        self._run = self._run + 1 if prob >= self.threshold else 0
        return self._run >= self.min_frames


# ------------------------------------------------------------------------ assistant


class State(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    SPEAKING = "speaking"


TriggerMode = Literal["wake", "ptt", "always"]


@dataclass
class _Turn:
    trace: TurnTrace
    utterance: list[Int16]
    triggered_by_wake: bool
    last_speech_t: float | None = None
    retried: bool = False
    transcript: str = ""
    user_text: str = ""
    reply: Reply | None = None


class Assistant:
    WAKE_REFRACTORY_S = 1.5

    def __init__(
        self,
        cfg: Config,
        *,
        wake: WakeWord,
        vad: Vad,
        stt: Stt,
        llm: Llm,
        tts: Tts,
        sink: AudioSink,
        writer: TraceWriter,
        on_event: EventHook = _no_events,
        clock: Clock = time.perf_counter,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.wake = wake
        self.vad = vad
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.sink = sink
        self.writer = writer
        self.on_event = on_event
        self.clock = clock
        self.meta = meta or {}
        self.conversation = Conversation(cfg.llm.system_prompt, cfg.llm.history_turns)
        self.endpointer = Endpointer(cfg.vad)
        p = cfg.pipeline
        self.guard = BargeInGuard(
            p.barge_in_vad_threshold,
            max(1, round(p.barge_in_min_speech_ms / 1000 / FRAME_S)),
            round(p.barge_in_ignore_ms / 1000 / FRAME_S),
        )
        self.trigger = threading.Event()  # push-to-talk key press
        self.stop_requested = threading.Event()
        self._turn_no = 0

    # -- helpers ---------------------------------------------------------------

    def _new_turn(self, utterance: list[Int16], by_wake: bool, at: float | None) -> _Turn:
        self._turn_no += 1
        trace = TurnTrace(self._turn_no, self.writer.session, clock=self.clock)
        if at is not None and by_wake:
            trace.mark("wake_detected", at)
        self.endpointer.reset()
        self.vad.reset()
        return _Turn(trace, utterance, by_wake)

    def _finish(self, turn: _Turn, traces: list[TurnTrace], outcome: str) -> None:
        """Wait for the turn's reply (if any), then write and keep its trace."""
        meta: dict[str, Any] = {
            "outcome": outcome,
            "transcript": turn.transcript,
            "user_text": turn.user_text,
            "engines": {
                "wake": self.wake.name,
                "stt": self.stt.name,
                "llm": self.llm.name,
                "tts": self.tts.name,
            },
        }
        result: ReplyResult | None = None
        if turn.reply is not None:
            result = turn.reply.wait()
            sink_stall = getattr(self.sink, "stall_s", None)
            meta.update(
                {
                    "reply": result.text,
                    "spoken": result.spoken,
                    "heard": result.heard,
                    "interrupted": result.interrupted,
                    "error": result.error,
                    "streaming": turn.reply.streaming,
                    "stall_s": None if sink_stall is None else round(float(sink_stall), 3),
                    "llm_stats": result.llm_stats,
                }
            )
        turn.trace.meta.update({**meta, **self.meta})
        self.writer.write(turn.trace)
        traces.append(turn.trace)
        if result is not None:
            if result.error:
                self.on_event("error", result.error)
            said = " ".join(result.heard) if result.interrupted else result.text
            self.on_event("reply_done", said)

    def _respond(self, turn: _Turn) -> Literal["reply", "stop", "empty"]:
        audio: Float32 = to_float32(np.concatenate(turn.utterance))
        turn.transcript = self.stt.transcribe(audio)
        turn.trace.mark("stt_done")
        text = strip_wake_phrase(turn.transcript) if turn.triggered_by_wake else turn.transcript
        turn.user_text = text.strip()
        if not turn.user_text:
            return "empty"
        self.on_event("user", turn.user_text)
        if is_stop_command(turn.user_text):
            return "stop"
        p = self.cfg.pipeline
        turn.reply = Reply(
            turn.user_text,
            llm=self.llm,
            tts=self.tts,
            sink=self.sink,
            conversation=self.conversation,
            trace=turn.trace,
            streaming=p.streaming,
            chunker=lambda: SentenceChunker(
                p.chunker_min_chars, p.chunker_max_chars, eager_first=p.first_chunk == "clause"
            ),
        ).start()
        self.guard.reset()
        return "reply"

    # -- main loop -------------------------------------------------------------

    def run(
        self, source: AudioSource, *, trigger: TriggerMode = "wake", max_turns: int | None = None
    ) -> list[TurnTrace]:
        traces: list[TurnTrace] = []
        preroll: deque[Int16] = deque(maxlen=max(1, round(self.cfg.wake.preroll_s / FRAME_S)))
        refractory = round(self.WAKE_REFRACTORY_S / FRAME_S)
        since_wake = refractory + 1
        barge_mode = self.cfg.pipeline.barge_in
        state = State.IDLE
        turn: _Turn | None = None
        if trigger == "always":
            turn = self._new_turn([], by_wake=False, at=None)
            state = State.LISTENING
        self.on_event("state", state.value)

        for frame in source.frames():
            if self.stop_requested.is_set():
                break
            now = self.clock()
            since_wake += 1
            # The wake model is fed every frame so its context is always current;
            # only its decisions are gated by state and a refractory period.
            woke = self.wake.score(frame) >= self.cfg.wake.threshold and since_wake > refractory
            if woke:
                since_wake = 0
            pressed = self.trigger.is_set()
            if pressed:
                self.trigger.clear()

            if state is State.SPEAKING:
                assert turn is not None and turn.reply is not None
                reply = turn.reply
                if not reply.finished.is_set():
                    barge = pressed or (barge_mode != "off" and woke)
                    if barge_mode == "vad" and not barge:
                        barge = self.guard.update(self.vad.probability(frame))
                    preroll.append(frame)
                    if barge:
                        turn.trace.mark("barge_in", now)
                        reply.cancel()
                        self.on_event("barge_in", "")
                        self._finish(turn, traces, "interrupted")
                        turn = None
                        if max_turns is not None and len(traces) >= max_turns:
                            break
                        utterance = list(preroll) if woke else []
                        turn = self._new_turn(utterance, by_wake=woke, at=now)
                        state = State.LISTENING
                    continue
                self._finish(turn, traces, "replied")
                turn = None
                if max_turns is not None and len(traces) >= max_turns:
                    break
                turn, state = self._rest(trigger)
                if state is State.LISTENING:
                    continue

            if state is State.IDLE:
                preroll.append(frame)
                if woke or pressed:
                    utterance = list(preroll) if woke else []
                    turn = self._new_turn(utterance, by_wake=woke, at=now)
                    state = State.LISTENING
                    self.on_event("wake" if woke else "ptt", "")
                continue

            # LISTENING
            assert turn is not None
            turn.utterance.append(frame)
            result = self.endpointer.update(self.vad.probability(frame))
            if result.is_speech:
                turn.last_speech_t = now
            if result.phase is Phase.SPEECH_START:
                turn.trace.mark("speech_start", now)
            elif result.phase is Phase.TIMEOUT:
                self.on_event("timeout", "")
                turn, state = self._rest(trigger)
            elif result.phase is Phase.ENDPOINT:
                turn.trace.mark("speech_end", turn.last_speech_t)
                turn.trace.mark("endpoint", now)
                outcome = self._respond(turn)
                if outcome == "reply":
                    state = State.SPEAKING
                elif outcome == "empty" and turn.triggered_by_wake and not turn.retried:
                    # A bare "hey jarvis" then a pause: keep listening for the request.
                    turn.retried = True
                    turn.utterance = []
                    turn.last_speech_t = None
                    turn.trace.unmark("speech_start", "speech_end", "endpoint", "stt_done")
                    self.endpointer.reset()
                    self.on_event("wake_only", turn.transcript)
                else:
                    if outcome == "stop":
                        self.on_event("stop", "")
                        self._finish(turn, traces, "stop_command")
                        if max_turns is not None and len(traces) >= max_turns:
                            break
                    else:
                        self.on_event("empty", turn.transcript)
                    turn, state = self._rest(trigger)

        if turn is not None and turn.reply is not None:
            if self.stop_requested.is_set():
                turn.reply.cancel()
            self._finish(
                turn, traces, "interrupted" if turn.reply.cancelled.is_set() else "replied"
            )
        return traces

    def _rest(self, trigger: TriggerMode) -> tuple[_Turn | None, State]:
        """Where to go after a turn: idle, or (no wake word) straight back to listening."""
        if trigger == "always":
            return self._new_turn([], by_wake=False, at=None), State.LISTENING
        self.on_event("state", State.IDLE.value)
        return None, State.IDLE
