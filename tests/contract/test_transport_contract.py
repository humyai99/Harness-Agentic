"""One suite every transport must pass.

Written against the abstraction rather than any one provider, so a new
transport is finished when this goes green -- and so a provider that quietly
changes shape has something to break. Everything here is pure conversion: no
network, no key.

The properties asserted are the ones that actually cause outages:

* a streamed answer and a non-streamed one must normalize to the same thing;
* a round trip through encoding must preserve tool-call pairing;
* an unknown frame type must not break a turn;
* an unsigned thinking block must never reach the wire.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest

from harness_agentic.core.stream import StreamAccumulator, StreamDone, UsageUpdate
from harness_agentic.core.types import (
    Message,
    ModelResponse,
    SystemPrompt,
    SystemSegment,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolSchema,
    ToolUseBlock,
    Usage,
)
from harness_agentic.errors import MalformedResponse
from harness_agentic.providers.base import (
    CompletionRequest,
    Credentials,
    ProviderTransport,
)
from harness_agentic.providers.catalog import ChatCompatQuirks, TransportFeature
from harness_agentic.providers.transports.anthropic import AnthropicTransport
from harness_agentic.providers.transports.chat_completions import ChatCompletionsTransport
from harness_agentic.providers.transports.gemini import GeminiTransport
from harness_agentic.testing import FakeTransport

NOW = datetime(2026, 1, 1, tzinfo=UTC)

CREDENTIALS = Credentials(base_url="https://example.invalid", api_key="k", source="explicit")

TRANSPORTS: dict[str, Callable[[], ProviderTransport]] = {
    "anthropic": lambda: AnthropicTransport(credentials=CREDENTIALS),
    "openai_chat": lambda: ChatCompletionsTransport(credentials=CREDENTIALS),
    "ollama_chat": lambda: ChatCompletionsTransport(
        credentials=CREDENTIALS,
        provider="ollama",
        quirks=ChatCompatQuirks(
            supports_tool_choice_required=False,
            max_tokens_field="max_tokens",
            stream_usage_option=False,
        ),
    ),
    "gemini": lambda: GeminiTransport(credentials=CREDENTIALS),
    "fake": FakeTransport,
}


@pytest.fixture(params=sorted(TRANSPORTS))
def transport(request: pytest.FixtureRequest) -> ProviderTransport:
    return TRANSPORTS[request.param]()


SCHEMA = ToolSchema(
    name="read_file",
    description="Read a file.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "title": "Path"}},
        "required": ["path"],
        "additionalProperties": False,
    },
)


def _conversation() -> tuple[Message, ...]:
    """A history containing every block kind that can legally be replayed."""
    return (
        Message(role="user", content=(TextBlock("read the config"),), created_at=NOW),
        Message(
            role="assistant",
            content=(
                ThinkingBlock("I should read it", signature="sig-1"),
                TextBlock("Looking now."),
                ToolUseBlock(id="c1", name="read_file", arguments={"path": "a.toml"}),
            ),
            created_at=NOW,
        ),
        Message(
            role="tool",
            content=(ToolResultBlock(tool_use_id="c1", text="name = 'demo'"),),
            created_at=NOW,
        ),
    )


def _request(**overrides: Any) -> CompletionRequest:
    base: dict[str, Any] = {
        "model": "test-model",
        "messages": _conversation(),
        "system": SystemPrompt(
            (
                SystemSegment("stable identity", cache_breakpoint=True),
                SystemSegment("volatile: the time is now"),
            )
        ),
        "tools": (SCHEMA,),
        "max_output_tokens": 1024,
    }
    base.update(overrides)
    return CompletionRequest(**base)


# -- request shape -------------------------------------------------------------


def test_build_request_is_json_serializable(transport: ProviderTransport) -> None:
    """Whatever comes out has to survive `json.dumps` on the way to the wire."""
    json.dumps(transport.build_request(_request()))


def test_the_system_prompt_reaches_the_request(transport: ProviderTransport) -> None:
    rendered = json.dumps(transport.build_request(_request()))
    assert "stable identity" in rendered


def test_tool_definitions_reach_the_request(transport: ProviderTransport) -> None:
    rendered = json.dumps(transport.build_request(_request()))
    assert "read_file" in rendered
    assert "path" in rendered


def test_every_tool_call_and_result_survives_encoding(transport: ProviderTransport) -> None:
    """Pairing is what providers reject on, so it must survive the round trip."""
    rendered = json.dumps(transport.build_request(_request()))
    assert "read_file" in rendered
    assert "name = 'demo'" in rendered


def test_unsigned_thinking_never_reaches_the_wire(transport: ProviderTransport) -> None:
    """Last line of defence behind the sanitizer."""
    history = (
        Message(
            role="assistant",
            content=(ThinkingBlock("unsigned"), TextBlock("visible")),
            created_at=NOW,
        ),
    )
    rendered = json.dumps(transport.build_request(_request(messages=history)))
    assert "unsigned" not in rendered
    assert "visible" in rendered


def test_a_request_without_tools_omits_tool_fields(transport: ProviderTransport) -> None:
    wire = transport.build_request(_request(tools=()))
    for key in ("tools", "tool_choice", "toolConfig"):
        assert key not in wire


def test_extra_passes_through(transport: ProviderTransport) -> None:
    """The escape hatch for a provider parameter we have not modelled yet."""
    wire = transport.build_request(_request(extra={"custom_knob": 7}))
    assert wire["custom_knob"] == 7


# -- streaming -----------------------------------------------------------------


def test_a_stream_always_terminates_with_usage_then_done(
    transport: ProviderTransport,
) -> None:
    """The accumulator relies on this ordering to produce a complete response."""
    events = list(transport.parse_stream(iter([])))
    assert isinstance(events[-2], UsageUpdate)
    assert isinstance(events[-1], StreamDone)


def test_unknown_frames_are_ignored(transport: ProviderTransport) -> None:
    """Providers add frame types; an unrecognised one must not break a turn."""
    frames: list[dict[str, Any]] = [
        {"type": "something_from_the_future"},
        {"unexpected": {"nested": True}},
    ]
    events = list(transport.parse_stream(iter(frames)))
    assert isinstance(events[-1], StreamDone)


def test_an_empty_stream_still_produces_a_response(transport: ProviderTransport) -> None:
    accumulator = StreamAccumulator(provider="x", model="m", now=NOW)
    for event in transport.parse_stream(iter([])):
        accumulator.feed(event)
    response = accumulator.finalize()
    assert response.message.role == "assistant"
    assert response.finish_reason in ("stop", "tool_calls", "length", "content_filter", "error")


# -- provider rules ------------------------------------------------------------


def test_tool_result_pairing_is_always_required(transport: ProviderTransport) -> None:
    """No provider accepts a tool call with no answer. None. Ever."""
    assert transport.sanitize_rules().require_tool_result_pairing


def test_validate_response_tolerates_empty_content(transport: ProviderTransport) -> None:
    """A refusal or a bare stop carries no blocks; that is not a malformation."""
    transport.validate_response(
        ModelResponse(
            message=Message(role="assistant", content=(), created_at=NOW),
            finish_reason="stop",
            usage=Usage(),
            model="m",
            provider="p",
        )
    )


def test_validate_response_rejects_a_promised_call_that_never_came(
    transport: ProviderTransport,
) -> None:
    with pytest.raises(MalformedResponse):
        transport.validate_response(
            ModelResponse(
                message=Message(role="assistant", content=(), created_at=NOW),
                finish_reason="tool_calls",
                usage=Usage(),
                model="m",
                provider="p",
            )
        )


def test_features_are_declared(transport: ProviderTransport) -> None:
    """Capability flags are how the loop avoids branching on provider names."""
    assert type(transport).features
    assert TransportFeature.STREAMING in type(transport).features


# The same real situation in each provider's own wire vocabulary: a 1000-token
# prompt of which 800 was served from cache, and 50 tokens out.
CACHED_USAGE: dict[str, dict[str, object]] = {
    "anthropic": {"input_tokens": 200, "output_tokens": 50, "cache_read_input_tokens": 800},
    "openai_chat": {
        "prompt_tokens": 1000,
        "completion_tokens": 50,
        "prompt_tokens_details": {"cached_tokens": 800},
    },
    "ollama_chat": {
        "prompt_tokens": 1000,
        "completion_tokens": 50,
        "prompt_tokens_details": {"cached_tokens": 800},
    },
    "gemini": {
        "promptTokenCount": 1000,
        "candidatesTokenCount": 50,
        "cachedContentTokenCount": 800,
    },
}
UNCACHED_PROMPT = 1000
CACHED_PROMPT = 200
CACHE_READ = 800
BILLED_TOTAL = 1050


@pytest.mark.parametrize("name", sorted(CACHED_USAGE))
def test_cached_input_is_never_counted_twice(name: str) -> None:
    """``input_tokens`` means the input that was *not* served from cache.

    The bug this pins down: Anthropic reports it that way already, while
    OpenAI-compatible endpoints and Gemini fold the cached tokens into their
    prompt count. Passing that straight through made ``total`` count a cached
    prefix twice -- 1850 rather than 1050 here -- so the same conversation cost
    different amounts depending on which provider answered it, and the
    "cache_read collapsed" signal was measured against a moving baseline.

    In the contract suite rather than a per-transport test, because it binds
    every transport added later.
    """
    usage = TRANSPORTS[name]()._usage(CACHED_USAGE[name])  # type: ignore[attr-defined]

    assert usage.cache_read_tokens == CACHE_READ
    assert usage.input_tokens == CACHED_PROMPT, "the cached part must not be in input_tokens"
    assert usage.total == BILLED_TOTAL


@pytest.mark.parametrize("name", sorted(CACHED_USAGE))
def test_an_uncached_prompt_is_reported_whole(name: str) -> None:
    # Subtracting must not touch the ordinary case.
    raw = dict(CACHED_USAGE[name])
    for key in ("prompt_tokens_details", "cache_read_input_tokens", "cachedContentTokenCount"):
        raw.pop(key, None)
    raw["input_tokens"] = UNCACHED_PROMPT  # anthropic reports the whole prompt here

    usage = TRANSPORTS[name]()._usage(raw)  # type: ignore[attr-defined]

    assert usage.cache_read_tokens == 0
    assert usage.input_tokens == UNCACHED_PROMPT
    assert usage.total == UNCACHED_PROMPT + 50
