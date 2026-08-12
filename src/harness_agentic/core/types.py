"""The canonical conversation format.

Everything in the framework speaks these types; provider transports convert to
and from their own wire shapes at the edge.

The format is **block-based** rather than OpenAI chat-completions shaped. That
is a deliberate departure from the obvious choice, and the reason is
directional: a block list is a strict superset that can hold interleaved
thinking and tool-use in their original order, thinking signatures, several
tool calls in one assistant message, and structured tool results. Converting
*down* to chat-completions is lossy but mechanical. Converting *up* from
chat-completions means reconstructing ordering that was already discarded --
and Anthropic rejects replayed thinking whose block order or signature no
longer matches.

Messages are frozen. A loop that clones history every turn is exactly where
aliasing bugs breed, and immutability makes structural cloning free.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Literal, TypeAlias

Role: TypeAlias = Literal["system", "user", "assistant", "tool"]

FinishReason: TypeAlias = Literal[
    "stop",
    "tool_calls",
    "length",
    "content_filter",
    "interrupted",
    "error",
]


# --------------------------------------------------------------- blocks -----


@dataclass(frozen=True, slots=True)
class TextBlock:
    """Ordinary visible text."""

    text: str
    kind: Literal["text"] = "text"


@dataclass(frozen=True, slots=True)
class ThinkingBlock:
    """Model reasoning.

    ``signature`` is not decoration. Anthropic issues one per thinking block and
    refuses the request if a replayed block's signature is missing or does not
    match. A block without one is unreplayable and must be dropped before the
    next request rather than sent and rejected.
    """

    text: str
    signature: str | None = None
    redacted: bool = False
    kind: Literal["thinking"] = "thinking"

    @property
    def replayable(self) -> bool:
        """Whether this block may be sent back to a provider."""
        return self.signature is not None


@dataclass(frozen=True, slots=True)
class ImageBlock:
    """An image, either inline or by reference."""

    media_type: str
    data_b64: str | None = None
    url: str | None = None
    kind: Literal["image"] = "image"


@dataclass(frozen=True, slots=True)
class ToolUseBlock:
    """A tool call requested by the model.

    ``raw_arguments`` keeps the model's original JSON string. Models do
    occasionally emit malformed argument JSON, and holding the raw text lets the
    sanitizer repair it and lets a failure be reported back as a tool error the
    model can correct, instead of an exception that ends the turn.
    """

    id: str
    name: str
    arguments: Mapping[str, object] = field(default_factory=dict)
    raw_arguments: str | None = None
    kind: Literal["tool_use"] = "tool_use"


@dataclass(frozen=True, slots=True)
class ToolResultBlock:
    """The outcome of one tool call, paired by ``tool_use_id``."""

    tool_use_id: str
    text: str
    is_error: bool = False
    truncated: bool = False
    kind: Literal["tool_result"] = "tool_result"


ContentBlock: TypeAlias = TextBlock | ThinkingBlock | ImageBlock | ToolUseBlock | ToolResultBlock


# -------------------------------------------------------------- message -----


@dataclass(frozen=True, slots=True)
class Message:
    """One conversation turn, as an ordered list of blocks."""

    role: Role
    content: tuple[ContentBlock, ...]
    created_at: datetime
    name: str | None = None
    provider_meta: Mapping[str, object] = field(default_factory=dict)

    def text(self) -> str:
        """Concatenate the visible text blocks, ignoring thinking and tools."""
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    def tool_uses(self) -> tuple[ToolUseBlock, ...]:
        """Return the tool calls in this message, in wire order."""
        return tuple(b for b in self.content if isinstance(b, ToolUseBlock))

    def tool_results(self) -> tuple[ToolResultBlock, ...]:
        """Return the tool results in this message, in wire order."""
        return tuple(b for b in self.content if isinstance(b, ToolResultBlock))

    def without_unsigned_thinking(self) -> Message:
        """Drop thinking blocks that cannot legally be replayed."""
        kept = tuple(
            b for b in self.content if not (isinstance(b, ThinkingBlock) and not b.replayable)
        )
        return self if len(kept) == len(self.content) else replace(self, content=kept)

    @property
    def is_empty(self) -> bool:
        """Whether this message would serialize to nothing useful."""
        return not self.content


# ---------------------------------------------------------------- usage -----


@dataclass(frozen=True, slots=True)
class Usage:
    """Token accounting for one or more requests.

    Cache reads and writes are tracked separately from ordinary input because
    they are priced differently, and because a sudden collapse in
    ``cache_read_tokens`` is the signal that something started mutating the
    stable part of the prompt.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        """Sum two usage records field by field."""
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    @property
    def total(self) -> int:
        """Every token the provider billed for."""
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )


# ------------------------------------------------------------- response -----


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """A normalized provider reply."""

    message: Message
    finish_reason: FinishReason
    usage: Usage
    model: str
    provider: str
    latency_ms: int = 0
    raw: Mapping[str, object] | None = None
    """Retained only when tracing is enabled; it is large and can hold secrets."""

    def tool_uses(self) -> tuple[ToolUseBlock, ...]:
        """Convenience accessor for the reply's tool calls."""
        return self.message.tool_uses()


# ------------------------------------------------- request-side contract -----


@dataclass(frozen=True, slots=True)
class ToolSchema:
    """A tool as the model sees it.

    Deliberately not the same object as the executable ``ToolSpec``: the
    transport layer has no business knowing about handlers, danger levels, or
    approval policy, and keeping the wire-facing shape separate is what stops
    that knowledge leaking into it.
    """

    name: str
    description: str
    parameters: Mapping[str, object]
    """JSON Schema for the arguments object."""


@dataclass(frozen=True, slots=True)
class SystemSegment:
    """One ordered piece of the system prompt.

    ``cache_breakpoint`` marks a place where a provider that supports prompt
    caching should be told to cut. Whether that hint is honoured is the
    transport's business; whether it is *placed* correctly is the prompt
    builder's, and putting one after volatile content is the classic way to
    triple a bill in silence.
    """

    text: str
    cache_breakpoint: bool = False


@dataclass(frozen=True, slots=True)
class SystemPrompt:
    """The assembled system prompt, still in segments."""

    segments: tuple[SystemSegment, ...] = ()

    def rendered(self) -> str:
        """Flatten to the single string a provider without caching wants."""
        return "\n\n".join(s.text for s in self.segments if s.text)

    def __bool__(self) -> bool:
        """Whether any segment carries text."""
        return any(s.text for s in self.segments)


# ------------------------------------------------------------ utilities -----


def user_message(text: str, *, now: datetime) -> Message:
    """Build a plain-text user message."""
    return Message(role="user", content=(TextBlock(text),), created_at=now)


def tool_result_message(results: Sequence[ToolResultBlock], *, now: datetime) -> Message:
    """Bundle tool results into the single message that answers a tool call.

    All results for one assistant turn belong in one message: providers pair
    them positionally against that turn's tool calls, and splitting them across
    messages is a reliable way to earn a 400.
    """
    return Message(role="tool", content=tuple(results), created_at=now)
