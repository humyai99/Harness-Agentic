"""Sanitizer tests, including a property test over pathological histories.

The property test is the important one. Handwritten cases cover the damage we
thought of; the generator covers the shapes a real session produces after an
interrupt lands in an awkward place. Every one of them must come out satisfying
the target provider's rules, because what does not is a 400 that blames the
provider for our bug.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from harness_agentic.agent.sanitize import (
    MISSING_RESULT_TEXT,
    assert_valid,
    sanitize,
)
from harness_agentic.core.types import (
    ContentBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from harness_agentic.providers.base import SanitizeRules

NOW = datetime(2026, 1, 1, tzinfo=UTC)

ANTHROPIC = SanitizeRules(
    require_alternating_roles=True,
    require_tool_result_pairing=True,
    tool_results_in_user_message=True,
    drop_unsigned_thinking=True,
)
CHAT = SanitizeRules(
    require_alternating_roles=False,
    require_tool_result_pairing=True,
    tool_results_in_user_message=False,
    drop_unsigned_thinking=True,
)


def _m(role: str, *blocks: ContentBlock) -> Message:
    return Message(role=role, content=tuple(blocks), created_at=NOW)  # type: ignore[arg-type]


# -- handwritten cases ---------------------------------------------------------


def test_a_clean_history_is_left_alone() -> None:
    history = [_m("user", TextBlock("hi")), _m("assistant", TextBlock("hello"))]
    out, report = sanitize(history, CHAT, now=NOW)
    assert report.clean
    assert out == history


def test_an_unanswered_tool_call_gets_a_synthetic_result() -> None:
    """The most common damage: the turn died between asking and answering."""
    history = [
        _m("user", TextBlock("read it")),
        _m("assistant", ToolUseBlock(id="c1", name="read_file")),
    ]
    out, report = sanitize(history, CHAT, now=NOW)
    assert report.synthesized_results == 1
    assert out[-1].role == "tool"
    result = out[-1].tool_results()[0]
    assert result.tool_use_id == "c1"
    assert result.is_error
    assert result.text == MISSING_RESULT_TEXT
    assert_valid(out, CHAT)


def test_an_orphan_tool_result_is_dropped() -> None:
    """Compaction can remove the assistant turn that asked."""
    history = [
        _m("user", TextBlock("hi")),
        _m("tool", ToolResultBlock(tool_use_id="ghost", text="stale")),
    ]
    out, report = sanitize(history, CHAT, now=NOW)
    assert report.dropped_orphan_results == 1
    assert not [b for m in out for b in m.tool_results()]


def test_unsigned_thinking_is_removed() -> None:
    history = [_m("assistant", ThinkingBlock("no signature"), TextBlock("answer"))]
    out, report = sanitize(history, CHAT, now=NOW)
    assert report.dropped_unsigned_thinking == 1
    assert out[0].text() == "answer"


def test_signed_thinking_survives() -> None:
    history = [_m("assistant", ThinkingBlock("kept", signature="s"), TextBlock("answer"))]
    out, report = sanitize(history, CHAT, now=NOW)
    assert report.dropped_unsigned_thinking == 0
    assert any(isinstance(b, ThinkingBlock) for b in out[0].content)


def test_empty_messages_are_dropped() -> None:
    history = [_m("user", TextBlock("hi")), _m("assistant", TextBlock("   "))]
    out, report = sanitize(history, CHAT, now=NOW)
    assert report.dropped_empty_messages == 1
    assert len(out) == 1


def test_malformed_arguments_are_repaired_to_an_empty_object() -> None:
    """Sending the broken string back would fail again; {} yields a clear error."""
    history = [
        _m("user", TextBlock("go")),
        _m(
            "assistant",
            ToolUseBlock(id="c1", name="read_file", arguments={}, raw_arguments='{"path": '),
        ),
        _m("tool", ToolResultBlock(tool_use_id="c1", text="ok")),
    ]
    out, report = sanitize(history, CHAT, now=NOW)
    assert report.repaired_arguments == 1
    call = out[1].tool_uses()[0]
    assert call.arguments == {}
    assert call.raw_arguments is None


def test_recoverable_arguments_are_reparsed() -> None:
    history = [
        _m("user", TextBlock("go")),
        _m(
            "assistant",
            ToolUseBlock(id="c1", name="read_file", arguments={}, raw_arguments='{"path":"a"}'),
        ),
        _m("tool", ToolResultBlock(tool_use_id="c1", text="ok")),
    ]
    out, _ = sanitize(history, CHAT, now=NOW)
    assert out[1].tool_uses()[0].arguments == {"path": "a"}


def test_tool_result_and_user_turn_merge_for_anthropic() -> None:
    """Under Anthropic's rules a tool turn *is* a user turn, so two would collide."""
    history = [
        _m("user", TextBlock("go")),
        _m("assistant", ToolUseBlock(id="c1", name="t")),
        _m("tool", ToolResultBlock(tool_use_id="c1", text="done")),
        _m("user", TextBlock("and now this")),
    ]
    out, report = sanitize(history, ANTHROPIC, now=NOW)
    assert report.merged_messages == 1
    assert_valid(out, ANTHROPIC)


def test_a_merged_user_turn_puts_its_tool_results_first() -> None:
    """Anthropic requires ``tool_result`` blocks to lead the content array.

    Merging preserves order, so text merged in ahead of a result would be a 400.
    Nothing in this codebase produces that order today -- a segment never starts
    with a tool result, and orphans are dropped before merging -- but that is a
    property of compaction and of the pairing pass rather than of the merge, so
    the merge imposes it instead of trusting it.
    """
    history = [
        _m("user", TextBlock("go")),
        _m("assistant", ToolUseBlock(id="c1", name="t")),
        # A user turn sitting between the call and its result: the shape the
        # merge must not turn into an invalid request.
        _m("user", TextBlock("actually, also do this")),
        _m("tool", ToolResultBlock(tool_use_id="c1", text="done")),
    ]
    out, _ = sanitize(history, ANTHROPIC, now=NOW)

    merged = next(m for m in out if m.tool_results())
    kinds = [block.kind for block in merged.content]
    assert kinds[0] == "tool_result", f"got {kinds}"
    assert "text" in kinds, "the user's text must survive, not be dropped"
    assert_valid(out, ANTHROPIC)


def test_assert_valid_rejects_content_before_a_tool_result() -> None:
    # The guard has to fail on the bad shape, or it is not a guard.
    broken = [
        _m("user", TextBlock("go")),
        _m("assistant", ToolUseBlock(id="c1", name="t")),
        _m("tool", TextBlock("chatter"), ToolResultBlock(tool_use_id="c1", text="done")),
    ]
    with pytest.raises(AssertionError, match="before its tool_result"):
        assert_valid(broken, ANTHROPIC)


def test_a_turn_that_arrives_with_its_result_behind_something_is_repaired() -> None:
    # The other way into the bad shape, and the one a merge cannot explain: a
    # single message simply holding a result behind another block.
    history = [
        _m("user", TextBlock("go")),
        _m("assistant", ToolUseBlock(id="c1", name="t")),
        _m("tool", TextBlock("chatter"), ToolResultBlock(tool_use_id="c1", text="done")),
    ]
    out, report = sanitize(history, ANTHROPIC, now=NOW)

    assert report.reordered_tool_results == 1
    assert not report.clean
    assert_valid(out, ANTHROPIC)


def test_chat_completions_keeps_the_tool_role_separate() -> None:
    history = [
        _m("user", TextBlock("go")),
        _m("assistant", ToolUseBlock(id="c1", name="t")),
        _m("tool", ToolResultBlock(tool_use_id="c1", text="done")),
        _m("user", TextBlock("next")),
    ]
    out, report = sanitize(history, CHAT, now=NOW)
    assert report.merged_messages == 0
    assert [m.role for m in out] == ["user", "assistant", "tool", "user"]


def test_the_input_is_never_mutated() -> None:
    """The store keeps the truth; only what goes on the wire is repaired."""
    original = [
        _m("user", TextBlock("go")),
        _m("assistant", ThinkingBlock("unsigned"), ToolUseBlock(id="c1", name="t")),
    ]
    snapshot = [m.content for m in original]
    sanitize(original, ANTHROPIC, now=NOW)
    assert [m.content for m in original] == snapshot


def test_partial_answers_still_get_the_missing_ones_filled() -> None:
    history = [
        _m("user", TextBlock("go")),
        _m(
            "assistant",
            ToolUseBlock(id="c1", name="a"),
            ToolUseBlock(id="c2", name="b"),
        ),
        _m("tool", ToolResultBlock(tool_use_id="c1", text="done")),
    ]
    out, report = sanitize(history, CHAT, now=NOW)
    assert report.synthesized_results == 1
    assert_valid(out, CHAT)


def test_assert_valid_catches_a_broken_history() -> None:
    broken = [_m("assistant", ToolUseBlock(id="c1", name="t"))]
    with pytest.raises(AssertionError, match="unanswered"):
        assert_valid(broken, CHAT)


# -- property test -------------------------------------------------------------


@st.composite
def pathological_history(draw: st.DrawFn) -> list[Message]:
    """Generate the message shapes a real session produces when things go wrong.

    Deliberately includes orphan results, unanswered calls, empty turns,
    consecutive same-role turns, unsigned thinking, and broken argument JSON --
    each of which a genuine interrupt or crash can leave behind.
    """
    call_ids = ["c1", "c2", "c3"]

    def block() -> st.SearchStrategy[ContentBlock]:
        return st.one_of(
            st.builds(TextBlock, st.sampled_from(["", "  ", "hello", "some text"])),
            st.builds(
                ThinkingBlock,
                st.just("reasoning"),
                st.sampled_from([None, "sig-1"]),
            ),
            st.builds(
                ToolUseBlock,
                st.sampled_from(call_ids),
                st.just("some_tool"),
                st.just({}),
                st.sampled_from([None, '{"broken": ', '{"ok": 1}']),
            ),
            st.builds(
                ToolResultBlock,
                st.sampled_from([*call_ids, "orphan"]),
                st.just("result text"),
            ),
        )

    count = draw(st.integers(min_value=0, max_value=8))
    return [
        _m(
            draw(st.sampled_from(["user", "assistant", "tool"])),
            *draw(st.lists(block(), min_size=0, max_size=3)),
        )
        for _ in range(count)
    ]


@settings(max_examples=250, deadline=None)
@given(history=pathological_history(), anthropic=st.booleans())
def test_sanitized_history_always_satisfies_the_rules(
    history: list[Message], anthropic: bool
) -> None:
    rules = ANTHROPIC if anthropic else CHAT
    out, _ = sanitize(history, rules, now=NOW)
    assert_valid(out, rules)


@settings(max_examples=100, deadline=None)
@given(history=pathological_history())
def test_sanitizing_is_idempotent(history: list[Message]) -> None:
    """A second pass must find nothing left to repair.

    If it does not, the repairs are fighting each other -- one pass creating
    the damage another pass removes -- and the history is unstable turn to turn.
    """
    once, _ = sanitize(history, ANTHROPIC, now=NOW)
    twice, report = sanitize(once, ANTHROPIC, now=NOW)
    assert report.clean, report.summary()
    assert once == twice
