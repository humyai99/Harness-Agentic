"""The fake provider has to be trustworthy before anything is tested with it."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from harness_agentic.core.cancel import CancelToken
from harness_agentic.core.clock import ManualClock
from harness_agentic.core.types import (
    Message,
    SystemPrompt,
    SystemSegment,
    TextBlock,
    ThinkingBlock,
)
from harness_agentic.errors import RateLimited
from harness_agentic.providers.base import CompletionRequest
from harness_agentic.testing import FakeTransport, ScriptedTurn, text_turn, tool_turn
from harness_agentic.testing.fakes import ScriptExhausted


def _request(text: str = "hi", **kwargs: object) -> CompletionRequest:
    clock = ManualClock()
    return CompletionRequest(
        model="fake/scripted",
        messages=(Message(role="user", content=(TextBlock(text),), created_at=clock.now()),),
        **kwargs,  # type: ignore[arg-type]
    )


def test_replays_the_script_in_order() -> None:
    transport = FakeTransport([text_turn("first"), text_turn("second")])
    assert transport.send(_request()).message.text() == "first"
    assert transport.send(_request()).message.text() == "second"


def test_exhausting_the_script_is_loud() -> None:
    """An extra iteration is usually the bug the test exists to catch."""
    transport = FakeTransport([text_turn("only one")])
    transport.send(_request())
    with pytest.raises(ScriptExhausted, match="exhausted after 1"):
        transport.send(_request())


def test_tool_turn_produces_a_parsed_call() -> None:
    transport = FakeTransport([tool_turn("read_file", {"path": "pyproject.toml"})])
    (call,) = transport.send(_request()).tool_uses()
    assert call.name == "read_file"
    assert call.arguments == {"path": "pyproject.toml"}


def test_scripted_exception_propagates_for_retry_tests() -> None:
    transport = FakeTransport([ScriptedTurn(raises=RateLimited("slow down", retry_after_s=2.0))])
    with pytest.raises(RateLimited) as caught:
        transport.send(_request())
    assert caught.value.retry_after_s == 2.0


def test_streaming_stops_when_cancelled() -> None:
    cancel = CancelToken()
    transport = FakeTransport([text_turn("a fairly long answer to stream")])
    seen = 0
    for _ in transport.stream(_request(), cancel=cancel):
        seen += 1
        cancel.cancel("user pressed ctrl-c")
    assert seen == 1


def test_requests_are_recorded_for_assertions() -> None:
    transport = FakeTransport([text_turn("ok")])
    system = SystemPrompt((SystemSegment("you are a test", cache_breakpoint=True),))
    transport.send(_request(system=system))
    assert transport.calls == 1
    assert transport.last_system_text() == "you are a test"
    assert transport.wire_requests[0]["system"] == [
        {"text": "you are a test", "cache_breakpoint": True}
    ]


def test_unsigned_thinking_assertion_catches_a_bad_replay() -> None:
    transport = FakeTransport([text_turn("ok")])
    bad = Message(
        role="assistant",
        content=(ThinkingBlock("no signature here"),),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    transport.send(CompletionRequest(model="fake/scripted", messages=(bad,), stream=False))
    with pytest.raises(AssertionError, match="unsigned thinking"):
        transport.assert_no_unsigned_thinking_sent()


def test_malformed_tool_args_can_be_forced() -> None:
    transport = FakeTransport(
        [ScriptedTurn(tool_calls=(("read_file", {}),), malformed_tool_args='{"path": ')]
    )
    (call,) = transport.send(_request()).tool_uses()
    assert call.arguments == {}
    assert call.raw_arguments == '{"path": '
