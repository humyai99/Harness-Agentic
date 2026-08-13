"""Conversion tests for the Anthropic transport.

Conversion is pure, so all of this runs with no network. These assertions are
the start of the drift corpus: every 400 the transport earns in the field
should end up here as a case.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from harness_agentic.core.secrets import Secret
from harness_agentic.core.stream import (
    StreamAccumulator,
    StreamDone,
    TextDelta,
    ToolUseStart,
    UsageUpdate,
)
from harness_agentic.core.types import (
    Message,
    SystemPrompt,
    SystemSegment,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolSchema,
    ToolUseBlock,
)
from harness_agentic.errors import MalformedResponse
from harness_agentic.providers.base import (
    CompletionRequest,
    Credentials,
    ReasoningConfig,
)
from harness_agentic.providers.transports.anthropic import AnthropicTransport

NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def transport() -> AnthropicTransport:
    return AnthropicTransport(
        credentials=Credentials(
            base_url="https://api.anthropic.com",
            api_key=Secret("sk-test", source="test"),
            source="explicit",
        )
    )


def _msg(role: str, *blocks: object) -> Message:
    return Message(role=role, content=tuple(blocks), created_at=NOW)  # type: ignore[arg-type]


def _req(*messages: Message, **kwargs: object) -> CompletionRequest:
    return CompletionRequest(model="claude-sonnet-4-6", messages=messages, **kwargs)  # type: ignore[arg-type]


# -- system prompt and caching -----------------------------------------------


def test_cache_breakpoint_becomes_cache_control(transport: AnthropicTransport) -> None:
    system = SystemPrompt(
        (
            SystemSegment("stable identity", cache_breakpoint=True),
            SystemSegment("volatile: the time is now"),
        )
    )
    wire = transport.build_request(_req(_msg("user", TextBlock("hi")), system=system))
    assert wire["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in wire["system"][1]


def test_cache_breakpoints_are_capped_at_the_api_limit(
    transport: AnthropicTransport,
) -> None:
    """A fifth marker is an API error, not something the provider ignores."""
    system = SystemPrompt(
        tuple(SystemSegment(f"segment {i}", cache_breakpoint=True) for i in range(6))
    )
    wire = transport.build_request(_req(_msg("user", TextBlock("hi")), system=system))
    marked = [b for b in wire["system"] if "cache_control" in b]
    assert len(marked) == 4


# -- message encoding ---------------------------------------------------------


def test_tool_results_become_a_user_turn(transport: AnthropicTransport) -> None:
    """Anthropic has no `tool` role; results ride in a user message."""
    wire = transport.build_request(
        _req(
            _msg("user", TextBlock("read it")),
            _msg("assistant", ToolUseBlock(id="c1", name="read_file", arguments={"p": "a"})),
            _msg("tool", ToolResultBlock(tool_use_id="c1", text="contents")),
        )
    )
    assert [m["role"] for m in wire["messages"]] == ["user", "assistant", "user"]
    assert wire["messages"][2]["content"][0]["type"] == "tool_result"


def test_consecutive_same_role_turns_are_merged(transport: AnthropicTransport) -> None:
    """Two user turns in a row are rejected by the API, so they are joined."""
    wire = transport.build_request(
        _req(_msg("user", TextBlock("first")), _msg("user", TextBlock("second")))
    )
    assert len(wire["messages"]) == 1
    assert len(wire["messages"][0]["content"]) == 2


def test_signed_thinking_is_replayed_with_its_signature(
    transport: AnthropicTransport,
) -> None:
    wire = transport.build_request(
        _req(_msg("assistant", ThinkingBlock("reasoning", signature="sig-1")))
    )
    block = wire["messages"][0]["content"][0]
    assert block["type"] == "thinking"
    assert block["signature"] == "sig-1"


def test_unsigned_thinking_is_dropped(transport: AnthropicTransport) -> None:
    """Last line of defence behind the sanitizer -- the API would 400 on this."""
    wire = transport.build_request(
        _req(_msg("assistant", ThinkingBlock("no signature"), TextBlock("answer")))
    )
    kinds = [b["type"] for b in wire["messages"][0]["content"]]
    assert kinds == ["text"]


def test_empty_messages_are_omitted(transport: AnthropicTransport) -> None:
    wire = transport.build_request(
        _req(_msg("user", TextBlock("hi")), _msg("assistant", TextBlock("")))
    )
    assert len(wire["messages"]) == 1


# -- request options ----------------------------------------------------------


def test_tools_are_encoded_with_input_schema(transport: AnthropicTransport) -> None:
    schema = ToolSchema(
        name="read_file",
        description="Read a file.",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
    )
    wire = transport.build_request(_req(_msg("user", TextBlock("go")), tools=(schema,)))
    assert wire["tools"][0]["name"] == "read_file"
    assert wire["tools"][0]["input_schema"]["properties"]["path"]["type"] == "string"
    assert wire["tool_choice"] == {"type": "auto"}


def test_thinking_drops_temperature(transport: AnthropicTransport) -> None:
    """The API rejects both together, so one has to give -- and it is temperature."""
    wire = transport.build_request(
        _req(
            _msg("user", TextBlock("go")),
            temperature=0.7,
            reasoning=ReasoningConfig(effort="high", budget_tokens=8000),
        )
    )
    assert wire["thinking"] == {"type": "enabled", "budget_tokens": 8000}
    assert "temperature" not in wire


def test_extra_is_an_escape_hatch(transport: AnthropicTransport) -> None:
    wire = transport.build_request(
        _req(_msg("user", TextBlock("go")), extra={"metadata": {"user_id": "u1"}})
    )
    assert wire["metadata"] == {"user_id": "u1"}


# -- response normalization ---------------------------------------------------


def test_normalize_reads_blocks_usage_and_stop_reason(
    transport: AnthropicTransport,
) -> None:
    response = transport.normalize_response(
        {
            "model": "claude-sonnet-4-6",
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "let me look"},
                {"type": "tool_use", "id": "c1", "name": "read_file", "input": {"p": "a"}},
            ],
            "usage": {
                "input_tokens": 1200,
                "output_tokens": 45,
                "cache_read_input_tokens": 900,
                "cache_creation_input_tokens": 100,
            },
        }
    )
    assert response.finish_reason == "tool_calls"
    assert response.message.text() == "let me look"
    assert response.tool_uses()[0].arguments == {"p": "a"}
    assert response.usage.cache_read_tokens == 900
    assert response.usage.cache_write_tokens == 100


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("tool_use", "tool_calls"),
        ("max_tokens", "length"),
        ("refusal", "content_filter"),
        ("something_new", "stop"),
    ],
)
def test_stop_reason_mapping(raw: str, expected: str) -> None:
    assert AnthropicTransport.map_stop_reason(raw) == expected


# -- streaming ----------------------------------------------------------------


def test_stream_frames_reassemble_into_the_same_response(
    transport: AnthropicTransport,
) -> None:
    """A streamed answer and a non-streamed one must normalize identically."""
    frames = [
        {"type": "message_start", "message": {"model": "m", "usage": {"input_tokens": 10}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "he"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "llo"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 3},
        },
        {"type": "message_stop"},
    ]
    accumulator = StreamAccumulator(provider="anthropic", model="m", now=NOW)
    for event in transport.parse_stream(iter(frames)):
        accumulator.feed(event)
    response = accumulator.finalize()

    assert response.message.text() == "hello"
    assert response.finish_reason == "stop"
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 3


def test_stream_carries_thinking_signature(transport: AnthropicTransport) -> None:
    frames = [
        {"type": "message_start", "message": {"model": "m", "usage": {}}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "hmm"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig-9"},
        },
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        {"type": "message_stop"},
    ]
    accumulator = StreamAccumulator(provider="anthropic", model="m", now=NOW)
    for event in transport.parse_stream(iter(frames)):
        accumulator.feed(event)
    thinking = next(
        b for b in accumulator.finalize().message.content if isinstance(b, ThinkingBlock)
    )
    assert thinking.signature == "sig-9"


def test_stream_tool_arguments_arrive_as_partial_json(
    transport: AnthropicTransport,
) -> None:
    frames = [
        {"type": "message_start", "message": {"model": "m", "usage": {}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "c1", "name": "terminal"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"cmd":'},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '"ls"}'},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
        {"type": "message_stop"},
    ]
    accumulator = StreamAccumulator(provider="anthropic", model="m", now=NOW)
    for event in transport.parse_stream(iter(frames)):
        accumulator.feed(event)
    (call,) = accumulator.finalize().tool_uses()
    assert call.name == "terminal"
    assert call.arguments == {"cmd": "ls"}


def test_stream_always_ends_with_usage_then_done(transport: AnthropicTransport) -> None:
    events = list(transport.parse_stream(iter([{"type": "message_stop"}])))
    assert isinstance(events[-2], UsageUpdate)
    assert isinstance(events[-1], StreamDone)


def test_stream_error_frame_raises(transport: AnthropicTransport) -> None:
    frames = [{"type": "error", "error": {"message": "overloaded"}}]
    with pytest.raises(MalformedResponse, match="overloaded"):
        list(transport.parse_stream(iter(frames)))


def test_unknown_frames_are_ignored(transport: AnthropicTransport) -> None:
    """Providers add frame types; an unknown one must not break a turn."""
    frames = [
        {"type": "ping"},
        {"type": "some_future_thing", "payload": 1},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
        {"type": "message_stop"},
    ]
    events = list(transport.parse_stream(iter(frames)))
    assert any(isinstance(e, TextDelta) for e in events)
    assert not any(isinstance(e, ToolUseStart) for e in events)


# -- provider rules -----------------------------------------------------------


def test_sanitize_rules_describe_anthropics_constraints(
    transport: AnthropicTransport,
) -> None:
    rules = transport.sanitize_rules()
    assert rules.tool_results_in_user_message
    assert rules.require_alternating_roles
    assert rules.drop_unsigned_thinking
