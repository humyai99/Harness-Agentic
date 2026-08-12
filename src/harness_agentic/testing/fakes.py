"""A scripted provider.

This is the most load-bearing piece of test infrastructure in the project, and
it is written before any real transport on purpose. Registered as the ``fake``
provider, it lets the agent loop, the CLI, the gateway, and the skills curator
all be driven end to end by setting one config value -- no API key, no network,
no cost, and deterministic output.

It also records what it was sent. Most interesting assertions are about the
*request*: did the sanitizer pair that tool result, did volatile content stay
out of the cached prefix, did the loop actually stop offering a disabled tool.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from harness_agentic.core.cancel import NEVER_CANCELLED, CancelToken
from harness_agentic.core.clock import Clock, SystemClock
from harness_agentic.core.stream import (
    BlockStop,
    StreamAccumulator,
    StreamDone,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ThinkingSignature,
    ToolUseArgsDelta,
    ToolUseStart,
    UsageUpdate,
)
from harness_agentic.core.types import (
    FinishReason,
    Message,
    ModelResponse,
    SystemPrompt,
    ThinkingBlock,
    ToolSchema,
    Usage,
)
from harness_agentic.errors import ProviderError
from harness_agentic.providers.base import (
    CompletionRequest,
    Credentials,
    ProviderTransport,
    WireRequest,
)
from harness_agentic.providers.catalog import TransportFeature


@dataclass(frozen=True, slots=True)
class ScriptedTurn:
    """One response the fake provider will give, in order."""

    text: str | None = None
    thinking: str | None = None
    thinking_signature: str | None = "sig-fake"
    tool_calls: tuple[tuple[str, Mapping[str, Any]], ...] = ()
    finish_reason: FinishReason | None = None
    """Inferred from ``tool_calls`` when omitted."""
    usage: Usage = field(default_factory=lambda: Usage(input_tokens=100, output_tokens=20))
    raises: Exception | None = None
    """Raised instead of answering -- for exercising retry and fallback."""
    text_chunks: Sequence[str] | None = None
    """Override how ``text`` is split, to test partial output and interruption."""
    malformed_tool_args: str | None = None
    """Emit this verbatim as a tool call's arguments, valid JSON or not."""

    def resolved_finish(self) -> FinishReason:
        """The finish reason this turn reports."""
        if self.finish_reason is not None:
            return self.finish_reason
        return "tool_calls" if self.tool_calls else "stop"


def text_turn(text: str, **kwargs: Any) -> ScriptedTurn:
    """A turn that just answers."""
    return ScriptedTurn(text=text, **kwargs)


def tool_turn(name: str, arguments: Mapping[str, Any], **kwargs: Any) -> ScriptedTurn:
    """A turn that calls one tool."""
    return ScriptedTurn(tool_calls=((name, arguments),), **kwargs)


class ScriptExhausted(ProviderError):
    """The script ran out of turns.

    Loud on purpose: a loop that iterates more than the test expected is
    usually the bug the test was written to catch, and silently repeating the
    last turn would hide it.
    """


class FakeTransport(ProviderTransport):
    """A provider that replays a fixed script."""

    api_mode: ClassVar[str] = "fake"
    features: ClassVar[frozenset[TransportFeature]] = frozenset(
        {
            TransportFeature.TOOLS,
            TransportFeature.PARALLEL_TOOL_CALLS,
            TransportFeature.STREAMING,
            TransportFeature.PROMPT_CACHE,
        }
    )

    def __init__(
        self,
        script: Sequence[ScriptedTurn] = (),
        *,
        credentials: Credentials | None = None,
        clock: Clock | None = None,
        model: str = "fake/scripted",
    ) -> None:
        """Prepare a transport that will answer with ``script`` in order."""
        super().__init__(
            credentials=credentials or Credentials(base_url="", source="none"),
            timeout_s=1.0,
        )
        self._script = list(script)
        self._clock = clock or SystemClock()
        self._model = model
        self._index = 0

        # Inspection surface. Assertions belong against these, not the network.
        self.requests: list[CompletionRequest] = []
        self.wire_requests: list[WireRequest] = []
        self.seen_messages: list[tuple[Message, ...]] = []
        self.seen_systems: list[SystemPrompt] = []
        self.seen_tools: list[tuple[ToolSchema, ...]] = []

    # -- script control -----------------------------------------------------

    def push(self, *turns: ScriptedTurn) -> None:
        """Append more turns to the script."""
        self._script.extend(turns)

    @property
    def calls(self) -> int:
        """How many completions have been requested."""
        return self._index

    @property
    def remaining(self) -> int:
        """How many scripted turns are unused."""
        return max(0, len(self._script) - self._index)

    def _next(self, request: CompletionRequest) -> ScriptedTurn:
        self.requests.append(request)
        self.seen_messages.append(request.messages)
        self.seen_systems.append(request.system)
        self.seen_tools.append(request.tools)
        self.wire_requests.append(self.build_request(request))
        if self._index >= len(self._script):
            msg = (
                f"FakeTransport script exhausted after {self._index} call(s); "
                f"the loop asked for another completion"
            )
            raise ScriptExhausted(msg)
        turn = self._script[self._index]
        self._index += 1
        if turn.raises is not None:
            raise turn.raises
        return turn

    # -- conversion ---------------------------------------------------------

    def build_request(self, request: CompletionRequest) -> WireRequest:
        """Render a readable stand-in for a wire body.

        Not a real API shape, but a faithful reflection of what the loop
        decided -- which is what assertions actually care about.
        """
        return {
            "model": request.model,
            "system": [
                {"text": s.text, "cache_breakpoint": s.cache_breakpoint}
                for s in request.system.segments
            ],
            "messages": [
                {"role": m.role, "blocks": [b.kind for b in m.content]} for m in request.messages
            ],
            "tools": [t.name for t in request.tools],
            "tool_choice": request.tool_choice,
            "max_output_tokens": request.max_output_tokens,
            "stream": request.stream,
        }

    def normalize_response(self, raw: Mapping[str, Any]) -> ModelResponse:
        """Round-trip a recorded fake response."""
        accumulator = StreamAccumulator(provider="fake", model=self._model, now=self._clock.now())
        for event in self._events(_turn_from_mapping(raw)):
            accumulator.feed(event)
        return accumulator.finalize()

    def parse_stream(self, chunks: Iterator[Mapping[str, Any]]) -> Iterator[StreamEvent]:
        """Convert recorded frames back into events."""
        for chunk in chunks:
            yield from self._events(_turn_from_mapping(chunk))

    # -- I/O ----------------------------------------------------------------

    def send(
        self, request: CompletionRequest, *, cancel: CancelToken = NEVER_CANCELLED
    ) -> ModelResponse:
        """Answer without streaming."""
        turn = self._next(request)
        accumulator = StreamAccumulator(provider="fake", model=request.model, now=self._clock.now())
        for event in self._events(turn):
            accumulator.feed(event)
        return accumulator.finalize(interrupted=cancel.is_set())

    def stream(
        self, request: CompletionRequest, *, cancel: CancelToken = NEVER_CANCELLED
    ) -> Iterator[StreamEvent]:
        """Answer as a stream, stopping early if cancelled mid-flight."""
        turn = self._next(request)
        for event in self._events(turn):
            if cancel.is_set():
                return
            yield event

    def count_tokens(self, request: CompletionRequest) -> int | None:
        """Estimate at four characters per token, deterministically."""
        text = request.system.rendered() + "".join(m.text() for m in request.messages)
        return max(1, len(text) // 4)

    # -- event generation ---------------------------------------------------

    def _events(self, turn: ScriptedTurn) -> list[StreamEvent]:
        events: list[StreamEvent] = []

        if turn.thinking:
            events.append(ThinkingDelta(turn.thinking))
            if turn.thinking_signature:
                events.append(ThinkingSignature(index=0, signature=turn.thinking_signature))

        if turn.text:
            chunks = turn.text_chunks if turn.text_chunks is not None else _split(turn.text)
            events.extend(TextDelta(chunk) for chunk in chunks)

        for position, (name, arguments) in enumerate(turn.tool_calls):
            events.append(ToolUseStart(index=position, id=f"call_{position + 1}", name=name))
            payload = (
                turn.malformed_tool_args
                if turn.malformed_tool_args is not None
                else json.dumps(dict(arguments))
            )
            events.append(ToolUseArgsDelta(index=position, fragment=payload))
            events.append(BlockStop(index=position))

        events.append(UsageUpdate(turn.usage))
        events.append(StreamDone(finish_reason=turn.resolved_finish(), model=self._model))
        return events

    # -- assertions ---------------------------------------------------------

    def assert_no_unsigned_thinking_sent(self) -> None:
        """Fail if any request replayed a thinking block without a signature.

        Anthropic rejects those outright, so the sanitizer must strip them. The
        check lives here because the transport is the last place that sees a
        request before it would have gone out.
        """
        for turn_index, messages in enumerate(self.seen_messages):
            for message in messages:
                for block in message.content:
                    if isinstance(block, ThinkingBlock) and not block.replayable:
                        msg = (
                            f"request {turn_index} replayed an unsigned thinking "
                            f"block: {block.text[:60]!r}"
                        )
                        raise AssertionError(msg)

    def assert_tool_offered(self, name: str, *, at: int = -1) -> None:
        """Fail unless ``name`` was among the tools offered on that request."""
        offered = {tool.name for tool in self.seen_tools[at]}
        if name not in offered:
            msg = f"tool {name!r} was not offered; saw {sorted(offered)}"
            raise AssertionError(msg)

    def last_system_text(self) -> str:
        """The rendered system prompt from the most recent request."""
        return self.seen_systems[-1].rendered()


def _split(text: str, size: int = 12) -> list[str]:
    """Chop text into stream-sized pieces.

    Deliberately not on word boundaries: a renderer that only works when
    chunks are whole words is broken, and this is what finds that out.
    """
    return [text[i : i + size] for i in range(0, len(text), size)] or [text]


def _turn_from_mapping(raw: Mapping[str, Any]) -> ScriptedTurn:
    """Rebuild a turn from a recorded mapping."""
    usage = raw.get("usage") or {}
    return ScriptedTurn(
        text=raw.get("text"),
        thinking=raw.get("thinking"),
        tool_calls=tuple(
            (call["name"], call.get("arguments", {})) for call in raw.get("tool_calls", [])
        ),
        finish_reason=raw.get("finish_reason"),
        usage=Usage(**usage) if usage else Usage(),
    )
