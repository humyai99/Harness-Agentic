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
from dataclasses import dataclass, field
from typing import Any

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


@dataclass
class Stream:
    """A queue of frames for one subscriber.

    Bounded, and it drops the *oldest* frames when full while keeping a count.
    A browser tab that was backgrounded and stopped reading must not be able to
    hold the agent's memory hostage, and a subscriber that missed frames needs to
    know it did -- a silently truncated stream shows a half-finished answer and
    no indication that anything is missing.
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


def sink_for(stream: Stream, *, include_thinking: bool = False) -> Any:
    """An :class:`~harness_agentic.core.events.EventSink` that fills a stream."""

    def sink(event: AgentEvent) -> None:
        frame = to_frame(event, include_thinking=include_thinking)
        if frame is not None:
            stream.push(frame)

    return sink


def preamble() -> str:
    """What every stream sends first.

    The retry interval, and a comment. The comment matters: some proxies buffer
    a response until the first bytes arrive, and until then the browser has an
    open connection showing nothing.
    """
    return f"retry: {RETRY_MS}\n\n{HEARTBEAT_COMMENT}"
