"""Turning agent events into server-sent events, and back.

The web UI is a *subscriber* to the same stream the terminal renders, not a
second code path. That is the entire reason ``core/events.py`` was written in the
first milestone rather than this one: a loop that printed to stdout would have to
be rebuilt here, and every fix after that would have to be made twice.

SSE rather than websockets, deliberately. The traffic is one-directional --
events out, commands in over ordinary POSTs -- and SSE reconnects on its own,
survives proxies that mangle websocket upgrades, and needs no client library. The
one thing it cannot do is binary, and there is no binary here.

Two things are stripped on the way out. **Thinking is opt-in**, because it is
long, it is not the answer, and a UI that renders it by default buries the answer
under it. And **usage numbers carry no prices**, because a cost estimate that is
wrong is worse than no estimate at all -- the tokens are exact, the money is not.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

from harness_agentic.core.events import (
    AgentEvent,
    ApprovalRequested,
    ApprovalResolved,
    CompactionFinished,
    CompactionStarted,
    IterationStarted,
    Notice,
    ProviderFallback,
    RetryScheduled,
    SkillLoaded,
    TextChunk,
    ThinkingChunk,
    ToolCallFinished,
    ToolCallStarted,
    ToolProgress,
    TurnFinished,
    TurnStarted,
    UsageReported,
)

RETRY_MS = 3000
"""What the browser waits before reconnecting. Sent once per stream."""
HEARTBEAT_COMMENT = ": keep-alive\n\n"
"""A comment frame. Proxies close an idle connection at thirty or sixty
seconds, and a stream that dies mid-turn looks to the user like the agent
crashed."""


@dataclass(frozen=True, slots=True)
class Frame:
    """One SSE frame."""

    event: str
    data: dict[str, Any]
    id: str = ""

    def encode(self) -> str:
        """Serialize in SSE wire format.

        The payload is one line of JSON on purpose: a multi-line ``data:`` field
        is legal and every hand-rolled client gets it wrong.
        """
        lines = [f"event: {self.event}"]
        if self.id:
            lines.append(f"id: {self.id}")
        lines.append(f"data: {json.dumps(self.data, separators=(',', ':'))}")
        return "\n".join(lines) + "\n\n"


def to_frame(  # noqa: PLR0911, PLR0912
    event: AgentEvent, *, include_thinking: bool = False
) -> Frame | None:
    """Convert one agent event into a frame, or ``None`` to drop it."""
    match event:
        case TurnStarted(session_id=session, model=model):
            return Frame("turn_started", {"session": session, "model": model})
        case IterationStarted(index=index):
            return Frame("iteration", {"index": index})
        case TextChunk(text=text):
            return Frame("text", {"text": text})
        case ThinkingChunk(text=text):
            # Opt-in: long, not the answer, and it buries the answer.
            return Frame("thinking", {"text": text}) if include_thinking else None
        case ToolCallStarted(call_id=call, tool=tool, danger=danger, summary=summary):
            return Frame(
                "tool_started",
                {"id": call, "tool": tool, "danger": int(danger), "summary": summary},
            )
        case ToolProgress(call_id=call, message=message):
            return Frame("tool_progress", {"id": call, "message": message})
        case ToolCallFinished(call_id=call, tool=tool, is_error=failed, duration_s=took):
            return Frame(
                "tool_finished",
                {"id": call, "tool": tool, "error": failed, "seconds": round(took, 3)},
            )
        case ApprovalRequested(request_id=request, tool=tool, danger=danger, summary=summary):
            return Frame(
                "approval_requested",
                {"id": request, "tool": tool, "danger": int(danger), "summary": summary},
            )
        case ApprovalResolved(request_id=request, granted=granted, by=who):
            return Frame("approval_resolved", {"id": request, "granted": granted, "by": who})
        case ProviderFallback(from_model=old, to_model=new, reason=reason):
            return Frame("fallback", {"from": old, "to": new, "reason": reason})
        case RetryScheduled(attempt=attempt, delay_s=delay, reason=reason):
            return Frame("retry", {"attempt": attempt, "delay": delay, "reason": reason})
        case CompactionStarted(from_seq=low, to_seq=high):
            return Frame("compaction_started", {"from": low, "to": high})
        case CompactionFinished(replaced_messages=count, tokens_before=before, tokens_after=after):
            return Frame(
                "compaction_finished",
                {"replaced": count, "before": before, "after": after},
            )
        case UsageReported(usage=usage, cumulative=total):
            # Tokens, never money. A wrong cost estimate is worse than none.
            return Frame(
                "usage",
                {
                    "input": usage.input_tokens,
                    "output": usage.output_tokens,
                    "cache_read": usage.cache_read_tokens,
                    "total_input": total.input_tokens,
                    "total_output": total.output_tokens,
                },
            )
        case SkillLoaded(name=name, version=version, tokens=tokens):
            return Frame("skill", {"name": name, "version": version, "tokens": tokens})
        case Notice(level=level, message=message):
            return Frame("notice", {"level": level, "message": message})
        case TurnFinished(reason=reason, iterations=iterations, error=error):
            return Frame(
                "turn_finished",
                {"reason": reason, "iterations": iterations, "error": error},
            )
    # No fallback branch: the match is exhaustive over AgentEvent, and mypy
    # proves it. A new event type is then a type error here rather than one that
    # silently never reaches the browser.


@dataclass(eq=False)
class Stream:
    """A queue of frames for **one** subscriber.

    Bounded, and it drops the *oldest* frames when full while keeping a count.
    A browser tab that was backgrounded and stopped reading must not be able to
    hold the agent's memory hostage, and a subscriber that missed frames needs to
    know it did -- a silently truncated stream shows a half-finished answer and
    no indication that anything is missing.

    One subscriber is not a suggestion: :meth:`drain` empties the buffer, so two
    consumers sharing a stream each get an arbitrary half of the frames. Fanning
    one session out to several connections is :class:`Fanout`'s job.

    Compared by identity (``eq=False``), because two streams holding the same
    frames are still different subscribers -- and a value-equal stream would make
    ``Fanout.unsubscribe`` detach whichever one happened to match first.
    """

    maxlen: int = 2000
    frames: list[Frame] = field(default_factory=list)
    dropped: int = 0
    closed: bool = False

    def push(self, frame: Frame) -> None:
        """Add a frame, evicting the oldest if the buffer is full."""
        if self.closed:
            return
        self.frames.append(frame)
        while len(self.frames) > self.maxlen:
            self.frames.pop(0)
            self.dropped += 1

    def drain(self) -> list[str]:
        """Every buffered frame as wire text, clearing the buffer.

        A list rather than a generator: a generator that is never exhausted
        leaves the buffer half-cleared, and the caller here joins the result.
        """
        out: list[str] = []
        if self.dropped:
            out.append(Frame("gap", {"dropped": self.dropped}).encode())
            self.dropped = 0
        pending, self.frames = self.frames, []
        out.extend(frame.encode() for frame in pending)
        return out

    def close(self) -> None:
        """Stop accepting frames."""
        self.closed = True


class FrameTarget(Protocol):
    """Anything a sink can push frames at -- one :class:`Stream` or a fan-out."""

    def push(self, frame: Frame) -> None:
        """Accept one frame."""
        ...


@dataclass
class Fanout:
    """Every live subscriber for one session, plus a backlog for when there are none.

    A session is not one connection, and assuming it was cost real text. SSE
    reconnects **on its own** every ``RETRY_MS`` if anything interrupts the
    stream, and for a moment the old server task and the new one are both alive.
    With one shared queue between them each ``drain()`` takes whatever happens to
    be buffered, so an answer arrives split across two connections -- the reader
    sees a gap and there is nothing in the log. Two browser tabs on the same
    session did the same thing, permanently.

    So every subscriber gets its own :class:`Stream` and each frame is pushed to
    all of them. Slow subscribers are then each other's problem and nobody else's:
    a backgrounded tab fills and drops its own buffer.

    The backlog exists because a turn can start before any browser is listening --
    ``POST /turns`` returns immediately and the page subscribes after. Frames
    pushed with nobody attached go there and are handed to the first subscriber,
    so the opening of an answer is not lost. Later subscribers join live; that is
    ordinary SSE, and replaying a whole session to a second tab would be worse.
    """

    maxlen: int = 2000
    subscribers: list[Stream] = field(default_factory=list)
    backlog: Stream = field(default_factory=Stream)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def push(self, frame: Frame) -> None:
        """Give one frame to every subscriber, or to the backlog if there are none.

        Pushed from a worker thread while the event loop drains, so the
        subscriber list is copied under the lock and the pushes happen outside it
        -- holding a lock across ``Stream.push`` would let one subscriber's
        eviction loop stall the thread running the turn.
        """
        with self._lock:
            targets = list(self.subscribers) or [self.backlog]
        for target in targets:
            target.push(frame)

    def subscribe(self) -> Stream:
        """Attach a new subscriber, handing it anything buffered so far."""
        stream = Stream(maxlen=self.maxlen)
        with self._lock:
            if not self.subscribers:
                # The first subscriber inherits the backlog, so a turn that began
                # before the page connected still shows its opening frames.
                stream.frames = self.backlog.frames
                stream.dropped = self.backlog.dropped
                self.backlog.frames, self.backlog.dropped = [], 0
            self.subscribers.append(stream)
        return stream

    def unsubscribe(self, stream: Stream) -> None:
        """Detach one subscriber, closing its stream.

        Called from the connection's ``finally``. Without it a disconnected tab
        stays in the list forever and every frame is copied into a buffer nobody
        reads -- a leak that grows with each reconnect.
        """
        with self._lock:
            if stream in self.subscribers:
                self.subscribers.remove(stream)
        stream.close()

    def close(self) -> None:
        """Close every subscriber and stop accepting frames."""
        with self._lock:
            attached, self.subscribers = list(self.subscribers), []
        for stream in attached:
            stream.close()
        self.backlog.close()

    def subscriber_count(self) -> int:
        """How many connections are attached. For diagnostics and tests."""
        with self._lock:
            return len(self.subscribers)


def sink_for(target: FrameTarget, *, include_thinking: bool = False) -> Any:
    """An :class:`~harness_agentic.core.events.EventSink` that fills a stream.

    Takes a :class:`FrameTarget` rather than a :class:`Stream` so the same sink
    serves one subscriber or a whole :class:`Fanout`.
    """

    def sink(event: AgentEvent) -> None:
        frame = to_frame(event, include_thinking=include_thinking)
        if frame is not None:
            target.push(frame)

    return sink


def preamble() -> str:
    """What every stream sends first.

    The retry interval, and a comment. The comment matters: some proxies buffer
    a response until the first bytes arrive, and until then the browser has an
    open connection showing nothing.
    """
    return f"retry: {RETRY_MS}\n\n{HEARTBEAT_COMMENT}"
