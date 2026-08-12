"""Normalized streaming events and the accumulator that folds them.

Every transport emits this same event vocabulary, so the CLI renderer, the chat
gateway, and the tests all consume one shape regardless of which provider is
behind it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TypeAlias

from harness_agentic.core.types import (
    ContentBlock,
    FinishReason,
    Message,
    ModelResponse,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    Usage,
)


@dataclass(frozen=True, slots=True)
class TextDelta:
    """A fragment of visible text."""

    text: str


@dataclass(frozen=True, slots=True)
class ThinkingDelta:
    """A fragment of model reasoning."""

    text: str


@dataclass(frozen=True, slots=True)
class ThinkingSignature:
    """The signature closing a thinking block."""

    index: int
    signature: str


@dataclass(frozen=True, slots=True)
class ToolUseStart:
    """A tool call has begun; arguments arrive as deltas."""

    index: int
    id: str
    name: str


@dataclass(frozen=True, slots=True)
class ToolUseArgsDelta:
    """A fragment of a tool call's JSON arguments."""

    index: int
    fragment: str


@dataclass(frozen=True, slots=True)
class BlockStop:
    """A content block is complete."""

    index: int


@dataclass(frozen=True, slots=True)
class UsageUpdate:
    """Token accounting, which most providers send at the end."""

    usage: Usage


@dataclass(frozen=True, slots=True)
class StreamDone:
    """The provider finished. Carries the reason and the final model id."""

    finish_reason: FinishReason
    model: str = ""


StreamEvent: TypeAlias = (
    TextDelta
    | ThinkingDelta
    | ThinkingSignature
    | ToolUseStart
    | ToolUseArgsDelta
    | BlockStop
    | UsageUpdate
    | StreamDone
)


@dataclass
class _PartialToolUse:
    id: str
    name: str
    fragments: list[str] = field(default_factory=list)
    closed: bool = False

    def raw(self) -> str:
        return "".join(self.fragments)


class StreamAccumulator:
    """Folds a stream into a :class:`ModelResponse`.

    Safe to finalize mid-stream. That matters: when a user hits Ctrl-C while the
    model is talking, we still have to persist something that is a *legal*
    conversation prefix, or the next turn is rejected before it starts.
    """

    def __init__(self, *, provider: str, model: str, now: datetime) -> None:
        """Prepare an accumulator for one response."""
        self._provider = provider
        self._model = model
        self._now = now
        self._blocks: list[ContentBlock] = []
        self._text: list[str] = []
        self._thinking: list[str] = []
        self._thinking_signature: str | None = None
        self._tools: dict[int, _PartialToolUse] = {}
        self._tool_order: list[int] = []
        self._usage = Usage()
        self._finish: FinishReason = "stop"

    # -- feeding ------------------------------------------------------------

    def feed(self, event: StreamEvent) -> None:
        """Absorb one event."""
        match event:
            case TextDelta(text=text):
                self._flush_thinking()
                self._text.append(text)
            case ThinkingDelta(text=text):
                self._flush_text()
                self._thinking.append(text)
            case ThinkingSignature(signature=signature):
                self._thinking_signature = signature
            case ToolUseStart(index=index, id=call_id, name=name):
                self._flush_text()
                self._flush_thinking()
                self._tools[index] = _PartialToolUse(id=call_id, name=name)
                self._tool_order.append(index)
            case ToolUseArgsDelta(index=index, fragment=fragment):
                if partial := self._tools.get(index):
                    partial.fragments.append(fragment)
            case BlockStop(index=index):
                if partial := self._tools.get(index):
                    partial.closed = True
            case UsageUpdate(usage=usage):
                self._usage = usage
            case StreamDone(finish_reason=finish_reason, model=model):
                self._finish = finish_reason
                if model:
                    self._model = model

    def _flush_text(self) -> None:
        if self._text:
            self._blocks.append(TextBlock("".join(self._text)))
            self._text.clear()

    def _flush_thinking(self) -> None:
        if self._thinking:
            self._blocks.append(
                ThinkingBlock("".join(self._thinking), signature=self._thinking_signature)
            )
            self._thinking.clear()
            self._thinking_signature = None

    # -- finalizing ---------------------------------------------------------

    def finalize(self, *, interrupted: bool = False, latency_ms: int = 0) -> ModelResponse:
        """Produce the response.

        When ``interrupted``, two kinds of block are dropped: tool calls whose
        arguments never finished arriving, and thinking blocks that never
        received a signature. Both would be rejected on replay, and persisting
        them would poison every subsequent turn in the session.
        """
        self._flush_text()
        self._flush_thinking()

        blocks = list(self._blocks)
        if interrupted:
            blocks = [b for b in blocks if not (isinstance(b, ThinkingBlock) and not b.replayable)]

        for index in self._tool_order:
            partial = self._tools[index]
            if interrupted and not partial.closed:
                continue
            blocks.append(
                ToolUseBlock(
                    id=partial.id,
                    name=partial.name,
                    arguments=_parse_arguments(partial.raw()),
                    raw_arguments=partial.raw() or None,
                )
            )

        finish: FinishReason = "interrupted" if interrupted else self._finish
        return ModelResponse(
            message=Message(role="assistant", content=tuple(blocks), created_at=self._now),
            finish_reason=finish,
            usage=self._usage,
            model=self._model,
            provider=self._provider,
            latency_ms=latency_ms,
        )

    @property
    def visible_text(self) -> str:
        """Text emitted so far, for rendering a partial response."""
        return "".join([*(b.text for b in self._blocks if isinstance(b, TextBlock)), *self._text])


def _parse_arguments(raw: str) -> dict[str, object]:
    """Best-effort decode of a tool call's argument JSON.

    A model that emits malformed JSON is a recoverable situation, not a crash:
    the raw string is kept on the block, dispatch reports a schema error, and
    the model gets to correct itself next turn.
    """
    import json  # noqa: PLC0415  -- module import cost is paid per stream, not per import

    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def accumulate(
    events: Sequence[StreamEvent], *, provider: str, model: str, now: datetime
) -> ModelResponse:
    """Fold a complete event sequence in one call. Convenience for tests."""
    acc = StreamAccumulator(provider=provider, model=model, now=now)
    for event in events:
        acc.feed(event)
    return acc.finalize()
