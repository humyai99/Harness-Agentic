"""Compaction as the loop actually experiences it.

The unit tests prove the compactor produces a legal history. These prove the
loop reaches for it at the right moment, that what it produces is still
acceptable to a provider on the *next* turn, and that the text it summarized
away is still findable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness_agentic.agent.build import AgentBundle, build_agent
from harness_agentic.core.events import CompactionFinished, RecordingSink
from harness_agentic.core.types import Usage
from harness_agentic.errors import ContextOverflow
from harness_agentic.testing import FakeTransport, ScriptedTurn, text_turn
from harness_agentic.tools.approval import ApprovalPolicy, Mode


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root


def _agent(
    workspace: Path,
    tmp_path: Path,
    fake: FakeTransport,
    *,
    window: int | None = None,
) -> tuple[AgentBundle, RecordingSink]:
    events = RecordingSink()
    bundle = build_agent(
        model="fake/scripted",
        workspace=workspace,
        sessions_dir=tmp_path / "sessions",
        toolsets=["file"],
        surface="cli",
        emit=events,
        approval=ApprovalPolicy(surface="cli", modes={"cli": Mode.ALLOW}),
        transports={"fake": fake},
    )
    if window is not None and bundle.runner._budget is not None:
        bundle.runner._budget.retarget(window=window, reserve_output=256)
    return bundle, events


def test_a_long_conversation_triggers_compaction(workspace: Path, tmp_path: Path) -> None:
    """A tiny window forces the branch that a real 200k window rarely reaches.

    The script is generous because summarizing is itself a model call -- one
    per compaction, drawn from the same transport.
    """
    fake = FakeTransport([text_turn("x" * 4000) for _ in range(30)])
    bundle, events = _agent(workspace, tmp_path, fake, window=6_000)
    session = bundle.store.latest()
    assert session is not None

    for _ in range(5):
        bundle.runner.run_turn("keep going", session=session)

    assert events.of_type(CompactionFinished)


def test_the_history_stays_legal_after_compaction(workspace: Path, tmp_path: Path) -> None:
    """The failure mode compaction bugs actually produce: the *next* turn dies."""
    fake = FakeTransport([text_turn("y" * 4000) for _ in range(40)])
    bundle, _ = _agent(workspace, tmp_path, fake, window=6_000)
    session = bundle.store.latest()
    assert session is not None

    for _ in range(6):
        result = bundle.runner.run_turn("continue", session=session)
        assert result.exit_reason in ("completed", "max_iterations"), result.error

    fake.assert_no_unsigned_thinking_sent()
    # Every request sent after a compaction must still pair its tool calls.
    for messages in fake.seen_messages:
        asked = {c.id for m in messages for c in m.tool_uses()}
        answered = {b.tool_use_id for m in messages for b in m.tool_results()}
        assert asked == answered


def test_compacted_text_is_still_searchable(workspace: Path, tmp_path: Path) -> None:
    """Search over originals is what makes a lossy summary survivable."""
    fake = FakeTransport([text_turn("z" * 4000) for _ in range(30)])
    bundle, _ = _agent(workspace, tmp_path, fake, window=6_000)
    session = bundle.store.latest()
    assert session is not None

    bundle.runner.run_turn("the deploy key rotates on ZZUNIQUEDATE", session=session)
    for _ in range(4):
        bundle.runner.run_turn("keep going", session=session)

    assert bundle.store.search("ZZUNIQUEDATE")


def test_a_provider_overflow_compacts_rather_than_repeating(
    workspace: Path, tmp_path: Path
) -> None:
    """Retrying an identical over-long request just fails identically."""
    fake = FakeTransport(
        [
            text_turn("a" * 3000),
            text_turn("b" * 3000),
            text_turn("c" * 3000),
            ScriptedTurn(raises=ContextOverflow("prompt is too long")),
            text_turn("SUMMARY: the conversation so far"),
            text_turn("recovered after compaction"),
        ]
    )
    bundle, events = _agent(workspace, tmp_path, fake, window=200_000)
    session = bundle.store.latest()
    assert session is not None

    for _ in range(3):
        bundle.runner.run_turn("go", session=session)
    result = bundle.runner.run_turn("go", session=session)

    assert result.exit_reason == "completed"
    assert result.final_text == "recovered after compaction"
    assert events.of_type(CompactionFinished)


def test_the_budget_learns_from_reported_usage(workspace: Path, tmp_path: Path) -> None:
    """A provider that consistently bills more than we estimated is learned from."""
    fake = FakeTransport(
        [ScriptedTurn(text="ok", usage=Usage(input_tokens=5000, output_tokens=5))] * 4
    )
    bundle, _ = _agent(workspace, tmp_path, fake)
    session = bundle.store.latest()
    assert session is not None

    assert bundle.runner._budget.correction == 1.0
    for _ in range(4):
        bundle.runner.run_turn("hi", session=session)

    # Our estimate for a two-word exchange is far below 5000, so the factor
    # must have climbed toward reality rather than staying at its default.
    assert bundle.runner._budget.correction > 1.5


def test_session_search_is_always_available(workspace: Path, tmp_path: Path) -> None:
    """An agent that cannot reach its own history has to guess instead."""
    fake = FakeTransport([text_turn("ok")])
    bundle, _ = _agent(workspace, tmp_path, fake)
    session = bundle.store.latest()
    assert session is not None

    bundle.runner.run_turn("hi", session=session)

    offered = {tool.name for tool in fake.seen_tools[-1]}
    assert "session_search" in offered
    assert "terminal" not in offered
