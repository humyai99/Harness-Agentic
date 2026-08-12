"""Making a long conversation fit again.

The hardest correctness problem in the framework, and the one whose failures
are worst: a bad compaction does not throw, it produces a history the provider
rejects on *every subsequent turn*, so the session is dead and the error points
somewhere else entirely.

Four invariants, asserted before anything is returned:

1. every retained tool call still has its result;
2. the first user message survives -- it states the goal, and summarizing it
   away is how an agent confidently finishes the wrong task;
3. the last N turns survive verbatim, because that is the working context;
4. nothing unreplayable (unsigned thinking) is introduced.

Segmentation is what makes the first one hold. Messages are grouped so a tool
call and its result can never land on opposite sides of the boundary, which
means the summarizer is never able to split a pair no matter where it cuts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from harness_agentic.core.types import (
    Message,
    TextBlock,
    ThinkingBlock,
)
from harness_agentic.errors import ContextExhausted
from harness_agentic.memory.budget import estimate_message

MIN_SEGMENTS_TO_COMPACT = 3
"""Head, at least one middle segment to summarize, and a tail."""

SUMMARY_MARKER = "<compacted-history>"
"""Delimits the summary so the model can tell it from a real user turn."""

SUMMARY_PROMPT = """\
Summarize the conversation so far for an agent that will continue the work.

Write only these sections, and only what is actually established:

TASK: what the user is trying to achieve, in their terms.
DECISIONS: choices already made and why, so they are not relitigated.
FILES: paths touched, with what changed in each.
COMMANDS: what was run and what it returned.
OPEN: what is unresolved or still to do.

Facts only. No narration, no "the assistant then", no restating this format.
Anything omitted here is gone -- the original messages will not be visible.
"""


@dataclass(frozen=True, slots=True)
class CompactionNote:
    """What one compaction did."""

    replaced: tuple[int, int]
    """Inclusive index range in the input history."""
    messages_before: int
    messages_after: int
    tokens_before: int
    tokens_after: int
    summary_chars: int


@dataclass(frozen=True, slots=True)
class Segment:
    """An atomic run of messages that compaction may not split."""

    start: int
    end: int
    """Inclusive."""
    pinned: bool = False


Summarizer = Callable[[Sequence[Message]], str]
"""Turns a span of history into prose. Injected so tests need no model."""


def segment(messages: Sequence[Message]) -> list[Segment]:
    """Group messages so no tool call is ever separated from its result.

    A boundary may only fall where nothing is outstanding. Anything else lets a
    summary swallow a call while its result survives -- an instant rejection
    from every provider.
    """
    segments: list[Segment] = []
    start = 0
    outstanding: set[str] = set()

    for index, message in enumerate(messages):
        outstanding.update(call.id for call in message.tool_uses())
        outstanding.difference_update(block.tool_use_id for block in message.tool_results())
        if not outstanding:
            segments.append(Segment(start=start, end=index))
            start = index + 1

    if start < len(messages):
        # A trailing run with an unanswered call is still one atomic unit.
        segments.append(Segment(start=start, end=len(messages) - 1))
    return segments


class ContextCompactor:
    """Replaces a span of history with a summary of it."""

    def __init__(
        self,
        summarizer: Summarizer,
        *,
        keep_recent_turns: int = 4,
        min_messages_to_compact: int = 4,
    ) -> None:
        """Configure how much history is always kept verbatim."""
        self._summarize = summarizer
        self._keep_recent = keep_recent_turns
        self._min_messages = min_messages_to_compact

    def compact(
        self,
        messages: Sequence[Message],
        *,
        now: datetime,
        aggressiveness: int = 0,
    ) -> tuple[list[Message], CompactionNote]:
        """Return a shorter history plus a note describing the change.

        ``aggressiveness`` rises with each retry in a turn and shrinks the
        verbatim tail. A second attempt that kept exactly as much as the first
        would fail exactly the same way.
        """
        keep_recent = max(1, self._keep_recent - aggressiveness)
        segments = segment(messages)

        if len(messages) < self._min_messages or len(segments) < MIN_SEGMENTS_TO_COMPACT:
            msg = (
                "nothing left to compact: the conversation is already at its "
                "minimum shape and still does not fit"
            )
            raise ContextExhausted(msg)

        head = segments[0]
        tail_start = max(1, len(segments) - keep_recent)
        if tail_start <= 1:
            msg = "compaction would leave nothing but the pinned head and tail"
            raise ContextExhausted(msg)

        middle = segments[1:tail_start]
        low = middle[0].start
        high = middle[-1].end
        span = list(messages[low : high + 1])

        summary_text = self._summarize(span).strip()
        if not summary_text:
            msg = "the summarizer returned nothing"
            raise ContextExhausted(msg)

        summary = Message(
            role="user",
            content=(
                TextBlock(
                    f"{SUMMARY_MARKER}\n{summary_text}\n</compacted-history>\n\n"
                    f"That summary replaces {len(span)} earlier messages. "
                    f"Use session_search to recover any detail it omits."
                ),
            ),
            created_at=now,
        )

        rebuilt = [
            *messages[head.start : head.end + 1],
            summary,
            *messages[high + 1 :],
        ]

        before = sum(estimate_message(m) for m in messages)
        after = sum(estimate_message(m) for m in rebuilt)
        note = CompactionNote(
            replaced=(low, high),
            messages_before=len(messages),
            messages_after=len(rebuilt),
            tokens_before=before,
            tokens_after=after,
            summary_chars=len(summary_text),
        )

        # Verify before returning. A compactor that hands back a corrupt
        # history has turned a recoverable situation into a dead session.
        assert_invariants(rebuilt, original=messages, keep_recent=keep_recent)
        return rebuilt, note


def assert_invariants(
    rebuilt: Sequence[Message],
    *,
    original: Sequence[Message],
    keep_recent: int,
) -> None:
    """Raise if a compacted history violates any of the four invariants."""
    asked = {call.id for m in rebuilt for call in m.tool_uses()}
    answered = {block.tool_use_id for m in rebuilt for block in m.tool_results()}
    if unanswered := asked - answered:
        msg = f"compaction orphaned tool call(s): {sorted(unanswered)}"
        raise AssertionError(msg)
    if orphans := answered - asked:
        msg = f"compaction orphaned tool result(s): {sorted(orphans)}"
        raise AssertionError(msg)

    first_user = next((m for m in original if m.role == "user"), None)
    if first_user is not None and first_user not in rebuilt:
        msg = "compaction removed the first user message, which states the goal"
        raise AssertionError(msg)

    if keep_recent and original:
        last = original[-1]
        if last not in rebuilt:
            msg = "compaction removed the most recent message"
            raise AssertionError(msg)

    for message in rebuilt:
        for block in message.content:
            if isinstance(block, ThinkingBlock) and not block.replayable:
                msg = "compaction produced an unreplayable thinking block"
                raise AssertionError(msg)


def transcript_for_summary(messages: Sequence[Message]) -> str:
    """Render a span for the summarizer.

    Tool results are truncated hard. The model summarizing does not need forty
    thousand characters of file contents to write "read src/app.py"; feeding it
    the whole thing is how a compaction turn costs more than the compaction
    saves.
    """
    lines: list[str] = []
    for message in messages:
        for block in message.content:
            match block:
                case TextBlock(text=text) if text.strip():
                    lines.append(f"{message.role}: {text.strip()}")
                case ThinkingBlock():
                    continue
                case _ if hasattr(block, "name"):
                    lines.append(f"{message.role}: [called {block.name}]")
                case _ if hasattr(block, "tool_use_id"):
                    body = getattr(block, "text", "")[:400]
                    lines.append(f"tool result: {body}")
    return "\n".join(lines)
