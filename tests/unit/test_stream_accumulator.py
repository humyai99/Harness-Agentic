"""The accumulator's interrupt behaviour is a correctness requirement.

When a user stops the model mid-sentence, whatever we persist becomes the
prefix of the *next* request. Persisting a tool call whose arguments never
arrived, or a thinking block that never got its signature, does not just lose
that turn -- it makes every later turn in the session illegal.
"""

from __future__ import annotations

from datetime import UTC, datetime

from harness_agentic.core.stream import (
    BlockStop,
    StreamAccumulator,
    StreamDone,
    TextDelta,
    ThinkingDelta,
    ThinkingSignature,
    ToolUseArgsDelta,
    ToolUseStart,
    UsageUpdate,
    accumulate,
)
from harness_agentic.core.types import (
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    Usage,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _acc() -> StreamAccumulator:
    return StreamAccumulator(provider="fake", model="fake/scripted", now=NOW)


def test_text_deltas_join_into_one_block() -> None:
    response = accumulate(
        [TextDelta("hel"), TextDelta("lo "), TextDelta("world"), StreamDone("stop")],
        provider="fake",
        model="m",
        now=NOW,
    )
    assert response.message.text() == "hello world"
    assert len(response.message.content) == 1


def test_tool_arguments_are_parsed_and_raw_is_kept() -> None:
    response = accumulate(
        [
            ToolUseStart(index=0, id="c1", name="read_file"),
            ToolUseArgsDelta(index=0, fragment='{"path": '),
            ToolUseArgsDelta(index=0, fragment='"a.py"}'),
            BlockStop(index=0),
            StreamDone("tool_calls"),
        ],
        provider="fake",
        model="m",
        now=NOW,
    )
    (call,) = response.tool_uses()
    assert call.arguments == {"path": "a.py"}
    assert call.raw_arguments == '{"path": "a.py"}'


def test_malformed_tool_arguments_do_not_raise() -> None:
    """A model emitting broken JSON is recoverable, not fatal.

    The raw text survives so dispatch can report a schema error the model can
    correct next turn.
    """
    response = accumulate(
        [
            ToolUseStart(index=0, id="c1", name="read_file"),
            ToolUseArgsDelta(index=0, fragment='{"path": '),
            BlockStop(index=0),
            StreamDone("tool_calls"),
        ],
        provider="fake",
        model="m",
        now=NOW,
    )
    (call,) = response.tool_uses()
    assert call.arguments == {}
    assert call.raw_arguments == '{"path": '


def test_thinking_keeps_its_signature() -> None:
    acc = _acc()
    acc.feed(ThinkingDelta("weighing options"))
    acc.feed(ThinkingSignature(index=0, signature="sig-1"))
    acc.feed(TextDelta("done"))
    acc.feed(StreamDone("stop"))
    thinking = next(b for b in acc.finalize().message.content if isinstance(b, ThinkingBlock))
    assert thinking.signature == "sig-1"
    assert thinking.replayable


def test_interrupt_drops_unsigned_thinking() -> None:
    acc = _acc()
    acc.feed(ThinkingDelta("half a thought"))
    response = acc.finalize(interrupted=True)
    assert not [b for b in response.message.content if isinstance(b, ThinkingBlock)]
    assert response.finish_reason == "interrupted"


def test_interrupt_drops_unterminated_tool_calls_but_keeps_closed_ones() -> None:
    acc = _acc()
    acc.feed(ToolUseStart(index=0, id="c1", name="read_file"))
    acc.feed(ToolUseArgsDelta(index=0, fragment='{"path":"a.py"}'))
    acc.feed(BlockStop(index=0))
    acc.feed(ToolUseStart(index=1, id="c2", name="write_file"))
    acc.feed(ToolUseArgsDelta(index=1, fragment='{"path":'))
    response = acc.finalize(interrupted=True)
    assert [c.id for c in response.tool_uses()] == ["c1"]


def test_interrupt_keeps_visible_text() -> None:
    acc = _acc()
    acc.feed(TextDelta("partial ans"))
    response = acc.finalize(interrupted=True)
    assert response.message.text() == "partial ans"


def test_block_order_is_preserved_across_kinds() -> None:
    """Ordering is the whole reason for the block-based canonical format."""
    acc = _acc()
    acc.feed(ThinkingDelta("first I think"))
    acc.feed(ThinkingSignature(index=0, signature="s"))
    acc.feed(TextDelta("then I speak"))
    acc.feed(ToolUseStart(index=0, id="c1", name="terminal"))
    acc.feed(ToolUseArgsDelta(index=0, fragment="{}"))
    acc.feed(BlockStop(index=0))
    acc.feed(StreamDone("tool_calls"))
    kinds = [type(b) for b in acc.finalize().message.content]
    assert kinds == [ThinkingBlock, TextBlock, ToolUseBlock]


def test_usage_is_carried_through() -> None:
    acc = _acc()
    acc.feed(UsageUpdate(Usage(input_tokens=1200, output_tokens=340, cache_read_tokens=900)))
    acc.feed(StreamDone("stop"))
    usage = acc.finalize().usage
    assert usage.input_tokens == 1200
    assert usage.total == 2440


def test_visible_text_is_available_mid_stream() -> None:
    acc = _acc()
    acc.feed(TextDelta("so far"))
    assert acc.visible_text == "so far"
