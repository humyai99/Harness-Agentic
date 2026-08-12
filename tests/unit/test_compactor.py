"""Compaction: the four invariants, including under generated histories.

A compaction bug does not throw. It produces a history the provider rejects on
every *subsequent* turn, so the session is dead and the error points somewhere
else. That is why the invariants are asserted inside `compact()` and then
hammered here with generated input.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from harness_agentic.core.types import (
    ContentBlock,
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolSchema,
    ToolUseBlock,
)
from harness_agentic.errors import ContextExhausted
from harness_agentic.memory.budget import (
    TokenBudget,
    estimate_message,
    estimate_tokens,
    estimate_tools,
)
from harness_agentic.memory.compactor import (
    SUMMARY_MARKER,
    ContextCompactor,
    assert_invariants,
    segment,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _m(role: str, *blocks: ContentBlock) -> Message:
    return Message(role=role, content=tuple(blocks), created_at=NOW)  # type: ignore[arg-type]


def _fake_summary(messages: Sequence[Message]) -> str:
    return f"TASK: something\nOPEN: nothing\n({len(messages)} messages)"


def _conversation(turns: int) -> list[Message]:
    """A realistic history: user, assistant-with-tool, tool result, repeated."""
    history: list[Message] = [_m("user", TextBlock("the original goal"))]
    for index in range(turns):
        history.append(_m("assistant", ToolUseBlock(id=f"c{index}", name="read_file")))
        history.append(_m("tool", ToolResultBlock(tool_use_id=f"c{index}", text="x" * 500)))
        history.append(_m("assistant", TextBlock(f"step {index} done")))
    return history


# -- segmentation --------------------------------------------------------------


def test_a_call_and_its_result_share_a_segment() -> None:
    """The property that makes invariant 1 impossible to violate by cutting."""
    history = [
        _m("user", TextBlock("go")),
        _m("assistant", ToolUseBlock(id="c1", name="t")),
        _m("tool", ToolResultBlock(tool_use_id="c1", text="done")),
    ]
    segments = segment(history)
    assert [(s.start, s.end) for s in segments] == [(0, 0), (1, 2)]


def test_an_unanswered_trailing_call_is_still_one_segment() -> None:
    history = [
        _m("user", TextBlock("go")),
        _m("assistant", ToolUseBlock(id="c1", name="t")),
    ]
    assert [(s.start, s.end) for s in segment(history)] == [(0, 0), (1, 1)]


def test_parallel_calls_stay_together() -> None:
    history = [
        _m("user", TextBlock("go")),
        _m("assistant", ToolUseBlock(id="c1", name="t"), ToolUseBlock(id="c2", name="t")),
        _m("tool", ToolResultBlock(tool_use_id="c1", text="a")),
        _m("tool", ToolResultBlock(tool_use_id="c2", text="b")),
    ]
    assert [(s.start, s.end) for s in segment(history)] == [(0, 0), (1, 3)]


# -- compaction ----------------------------------------------------------------


def test_compaction_shortens_the_history() -> None:
    compactor = ContextCompactor(_fake_summary)
    history = _conversation(6)
    rebuilt, note = compactor.compact(history, now=NOW)

    assert note.messages_after < note.messages_before
    assert note.tokens_after < note.tokens_before
    assert any(SUMMARY_MARKER in m.text() for m in rebuilt)


def test_the_goal_survives() -> None:
    """Summarizing away the first user message is how an agent finishes the wrong task."""
    compactor = ContextCompactor(_fake_summary)
    history = _conversation(6)
    rebuilt, _ = compactor.compact(history, now=NOW)
    assert rebuilt[0].text() == "the original goal"


def test_the_recent_tail_survives_verbatim() -> None:
    compactor = ContextCompactor(_fake_summary, keep_recent_turns=2)
    history = _conversation(6)
    rebuilt, _ = compactor.compact(history, now=NOW)
    assert rebuilt[-1] == history[-1]


def test_tool_pairing_survives() -> None:
    compactor = ContextCompactor(_fake_summary)
    rebuilt, _ = compactor.compact(_conversation(6), now=NOW)
    asked = {c.id for m in rebuilt for c in m.tool_uses()}
    answered = {b.tool_use_id for m in rebuilt for b in m.tool_results()}
    assert asked == answered


def test_aggressiveness_keeps_less() -> None:
    """A retry that kept exactly as much would fail exactly the same way."""
    compactor = ContextCompactor(_fake_summary, keep_recent_turns=4)
    history = _conversation(8)
    gentle, _ = compactor.compact(history, now=NOW, aggressiveness=0)
    harsh, _ = compactor.compact(history, now=NOW, aggressiveness=3)
    assert len(harsh) < len(gentle)


def test_too_short_to_compact_raises_rather_than_corrupting() -> None:
    compactor = ContextCompactor(_fake_summary)
    with pytest.raises(ContextExhausted):
        compactor.compact([_m("user", TextBlock("only one"))], now=NOW)


def test_an_empty_summary_is_refused() -> None:
    compactor = ContextCompactor(lambda _: "   ")
    with pytest.raises(ContextExhausted, match="returned nothing"):
        compactor.compact(_conversation(6), now=NOW)


def test_the_input_is_not_mutated() -> None:
    compactor = ContextCompactor(_fake_summary)
    history = _conversation(6)
    snapshot = list(history)
    compactor.compact(history, now=NOW)
    assert history == snapshot


def test_invariant_check_catches_an_orphaned_call() -> None:
    """The guard the compactor runs on itself before returning."""
    broken = [_m("assistant", ToolUseBlock(id="c1", name="t"))]
    with pytest.raises(AssertionError, match="orphaned tool call"):
        assert_invariants(broken, original=broken, keep_recent=1)


def test_invariant_check_catches_a_lost_goal() -> None:
    original = [_m("user", TextBlock("goal")), _m("assistant", TextBlock("ok"))]
    with pytest.raises(AssertionError, match="first user message"):
        assert_invariants(original[1:], original=original, keep_recent=1)


def test_invariant_check_catches_unreplayable_thinking() -> None:
    history = [
        _m("user", TextBlock("goal")),
        _m("assistant", ThinkingBlock("no signature")),
    ]
    with pytest.raises(AssertionError, match="unreplayable"):
        assert_invariants(history, original=history, keep_recent=1)


@settings(max_examples=150, deadline=None)
@given(turns=st.integers(min_value=3, max_value=25), keep=st.integers(min_value=1, max_value=6))
def test_invariants_hold_for_any_conversation(turns: int, keep: int) -> None:
    compactor = ContextCompactor(_fake_summary, keep_recent_turns=keep)
    history = _conversation(turns)
    try:
        rebuilt, _ = compactor.compact(history, now=NOW)
    except ContextExhausted:
        return  # refusing is always a legal outcome
    assert_invariants(rebuilt, original=history, keep_recent=keep)


# -- budget --------------------------------------------------------------------


def test_the_threshold_accounts_for_output_and_headroom() -> None:
    budget = TokenBudget(window=100_000, reserve_output=8_000, headroom=0.1, compact_at=0.8)
    assert budget.usable == 82_000
    assert budget.compact_threshold == 65_600


def test_the_estimator_learns_from_real_usage() -> None:
    """Estimator error is systematic per model, which makes it learnable."""
    budget = TokenBudget(window=100_000)
    budget.observe(1200, raw=1000)
    assert budget.correction > 1.0
    for _ in range(10):
        budget.observe(1200, raw=1000)
    # Converges on the true ratio rather than compounding away from it.
    assert budget.correction == pytest.approx(1.2, abs=0.01)


def test_an_overflow_bumps_the_correction_hard() -> None:
    budget = TokenBudget(window=100_000)
    before = budget.correction
    budget.shrink_from_error()
    assert budget.correction > before * 1.2


def test_the_correction_is_bounded() -> None:
    """A runaway factor would make the agent compact a two-line conversation."""
    budget = TokenBudget(window=100_000)
    for _ in range(50):
        budget.shrink_from_error()
    assert budget.correction <= 3.0


def test_retargeting_resets_what_was_learned() -> None:
    """A factor learned for one tokenizer is worse than nothing for another."""
    budget = TokenBudget(window=100_000)
    budget.observe(1500, raw=1000)
    budget.retarget(window=200_000)
    assert budget.correction == 1.0
    assert budget.window == 200_000


def test_tool_definitions_are_counted() -> None:
    """A dozen tools is a few thousand tokens on every single request."""
    tools = tuple(
        ToolSchema(
            name=f"tool_{i}",
            description="A tool that does a thing with some described behaviour.",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        )
        for i in range(12)
    )
    assert estimate_tools(tools) > 200


def test_images_are_not_free() -> None:
    text_only = _m("user", TextBlock("hi"))
    with_image = _m("user", TextBlock("hi"), ImageBlock("image/png", data_b64="A"))
    assert estimate_message(with_image) > estimate_message(text_only) + 1000


def test_thai_text_is_estimated_not_ignored() -> None:
    assert estimate_tokens("ช่วยดูไฟล์นี้ให้หน่อยครับ") > 0
