"""What the agent tells the outside world while it works.

The loop never prints. It emits these, and a surface subscribes. That is what
makes the CLI, the chat gateway, a web UI over SSE, and a voice front end four
subscribers rather than four forks of the loop -- and it is why this module
exists in the first milestone rather than the ninth. A loop that writes to
stdout has to be rebuilt the first time something other than a terminal needs
to watch it.

Events are also the audit trail: every tool call, approval, and fallback shows
up here, which is what "observable execution" means in practice.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeAlias, TypeVar

if TYPE_CHECKING:
    from harness_agentic.core.types import Usage

    # Annotation-only, so `core` keeps no runtime dependency on `tools`.
    from harness_agentic.tools.spec import Danger


@dataclass(frozen=True, slots=True)
class TurnStarted:
    """A user turn began."""

    session_id: str
    model: str


@dataclass(frozen=True, slots=True)
class IterationStarted:
    """One request/response cycle within a turn began."""

    index: int


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A fragment of the assistant's visible answer."""

    text: str


@dataclass(frozen=True, slots=True)
class ThinkingChunk:
    """A fragment of the assistant's reasoning.

    Surfaces may render, dim, or drop this. Voice drops it.
    """

    text: str


@dataclass(frozen=True, slots=True)
class ToolCallStarted:
    """The agent is about to run a tool."""

    call_id: str
    tool: str
    danger: Danger
    summary: str


@dataclass(frozen=True, slots=True)
class ToolProgress:
    """A running tool reporting on itself."""

    call_id: str
    message: str


@dataclass(frozen=True, slots=True)
class ToolCallFinished:
    """A tool call completed, successfully or not."""

    call_id: str
    tool: str
    is_error: bool
    duration_s: float
    display: str | None = None


@dataclass(frozen=True, slots=True)
class ApprovalRequested:
    """The agent is blocked waiting for permission.

    Carries an id so a surface that is not the one that asked -- a web UI, a
    chat message -- can answer it.
    """

    request_id: str
    tool: str
    danger: Danger
    summary: str
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class ApprovalResolved:
    """A pending approval was granted or refused."""

    request_id: str
    granted: bool
    by: str


@dataclass(frozen=True, slots=True)
class ProviderFallback:
    """The primary model failed and the loop moved down the chain.

    Surfaced rather than logged quietly: a silent downgrade to a weaker model
    looks like the agent suddenly got worse for no reason.
    """

    from_model: str
    to_model: str
    reason: str


@dataclass(frozen=True, slots=True)
class RetryScheduled:
    """A transient failure will be retried after a delay."""

    attempt: int
    delay_s: float
    reason: str


@dataclass(frozen=True, slots=True)
class CompactionStarted:
    """The conversation is being summarized to fit the context window."""

    from_seq: int
    to_seq: int


@dataclass(frozen=True, slots=True)
class CompactionFinished:
    """Compaction completed."""

    replaced_messages: int
    tokens_before: int
    tokens_after: int


@dataclass(frozen=True, slots=True)
class UsageReported:
    """Token accounting for one request."""

    usage: Usage
    cumulative: Usage


@dataclass(frozen=True, slots=True)
class SkillLoaded:
    """A skill's instructions entered the conversation."""

    name: str
    version: str
    tokens: int


@dataclass(frozen=True, slots=True)
class Notice:
    """Something the operator should see that is not part of the answer."""

    level: Literal["info", "warning", "error"]
    message: str
    detail: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TurnFinished:
    """The turn ended, for whatever reason."""

    reason: Literal[
        "completed",
        "max_iterations",
        "budget_exhausted",
        "interrupted",
        "content_filter",
        "error",
    ]
    iterations: int
    usage: Usage
    error: str | None = None


AgentEvent: TypeAlias = (
    TurnStarted
    | IterationStarted
    | TextChunk
    | ThinkingChunk
    | ToolCallStarted
    | ToolProgress
    | ToolCallFinished
    | ApprovalRequested
    | ApprovalResolved
    | ProviderFallback
    | RetryScheduled
    | CompactionStarted
    | CompactionFinished
    | UsageReported
    | SkillLoaded
    | Notice
    | TurnFinished
)

EventSink: TypeAlias = Callable[[AgentEvent], None]
"""Where a running agent sends its events."""

E = TypeVar("E", bound=AgentEvent)


def null_sink(_event: AgentEvent) -> None:
    """Discard every event. The default, so a caller need not supply one."""


class RecordingSink:
    """Collects events in order. Used by tests and by the trace writer."""

    def __init__(self) -> None:
        """Start with an empty log."""
        self.events: list[AgentEvent] = []

    def __call__(self, event: AgentEvent) -> None:
        """Record one event."""
        self.events.append(event)

    def of_type(self, kind: type[E]) -> list[E]:
        """Return every recorded event of one type, in order."""
        return [e for e in self.events if isinstance(e, kind)]

    def text(self) -> str:
        """Reassemble the visible answer from the recorded chunks."""
        return "".join(e.text for e in self.events if isinstance(e, TextChunk))


class FanOutSink:
    """Delivers each event to several sinks.

    A misbehaving subscriber must not be able to kill a turn, so exceptions
    from one sink are swallowed and reported to the others as a ``Notice``.
    """

    def __init__(self, *sinks: EventSink) -> None:
        """Wrap the given sinks."""
        self._sinks = list(sinks)

    def add(self, sink: EventSink) -> None:
        """Subscribe another sink."""
        self._sinks.append(sink)

    def __call__(self, event: AgentEvent) -> None:
        """Deliver to every subscriber."""
        for sink in self._sinks:
            try:
                sink(event)
            except Exception as exc:  # a broken sink must never break a turn
                for other in self._sinks:
                    if other is sink:
                        continue
                    with contextlib.suppress(Exception):
                        other(Notice("warning", f"event sink failed: {exc!r}"))
