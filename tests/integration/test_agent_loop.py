"""End-to-end turns through the real loop, with a scripted provider.

No network, no API key, no cost -- but every other layer is the production one:
the real registry, the real tools, the real filesystem environment, the real
sanitizer, the real session store. That is the point of writing FakeTransport
before any real transport.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness_agentic.agent.build import AgentBundle, build_agent
from harness_agentic.core.events import (
    FanOutSink,
    ProviderFallback,
    RecordingSink,
    TextChunk,
    ToolCallFinished,
    ToolCallStarted,
    TurnFinished,
)
from harness_agentic.errors import AuthError, RateLimited
from harness_agentic.testing import FakeTransport, ScriptedTurn, text_turn, tool_turn
from harness_agentic.tools.approval import ApprovalPolicy, Mode, always_deny


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nrequires-python = ">=3.11"\n', encoding="utf-8"
    )
    return root


def _agent(
    workspace: Path,
    tmp_path: Path,
    script: list[ScriptedTurn],
    *,
    approval: ApprovalPolicy | None = None,
    sink: RecordingSink | None = None,
    transport: FakeTransport | None = None,
) -> tuple[AgentBundle, RecordingSink, FakeTransport]:
    events = sink or RecordingSink()
    fake = transport or FakeTransport(script)
    bundle = build_agent(
        model="fake/scripted",
        workspace=workspace,
        sessions_dir=tmp_path / "sessions",
        toolsets=["file", "terminal"],
        surface="cli",
        emit=events,
        approval=approval or ApprovalPolicy(surface="cli", modes={"cli": Mode.ALLOW}),
        transports={"fake": fake},
    )
    return bundle, events, fake


def test_a_plain_answer_completes_in_one_iteration(workspace: Path, tmp_path: Path) -> None:
    bundle, events, _ = _agent(workspace, tmp_path, [text_turn("hello there")])
    session = bundle.store.latest()
    assert session is not None

    result = bundle.runner.run_turn("hi", session=session)

    assert result.exit_reason == "completed"
    assert result.final_text == "hello there"
    assert result.iterations == 1
    assert events.text() == "hello there"


def test_the_agent_reads_a_real_file_and_answers(workspace: Path, tmp_path: Path) -> None:
    """The full round trip: tool call, real filesystem, result, final answer."""
    bundle, events, _ = _agent(
        workspace,
        tmp_path,
        [
            tool_turn("read_file", {"path": "pyproject.toml"}),
            text_turn("It requires Python 3.11 or newer."),
        ],
    )
    session = bundle.store.latest()
    assert session is not None

    result = bundle.runner.run_turn("what python version does this need?", session=session)

    assert result.exit_reason == "completed"
    assert result.iterations == 2
    assert "3.11" in result.final_text

    started = events.of_type(ToolCallStarted)
    assert [e.tool for e in started] == ["read_file"]
    assert not events.of_type(ToolCallFinished)[0].is_error


def test_the_agent_writes_a_real_file(workspace: Path, tmp_path: Path) -> None:
    bundle, _, _ = _agent(
        workspace,
        tmp_path,
        [
            tool_turn("write_file", {"path": "hello.py", "content": "print('สวัสดี')\n"}),
            text_turn("Created hello.py."),
        ],
    )
    session = bundle.store.latest()
    assert session is not None

    bundle.runner.run_turn("create hello.py", session=session)

    written = workspace / "hello.py"
    assert written.exists()
    assert "สวัสดี" in written.read_text(encoding="utf-8")


def test_a_denied_tool_is_reported_and_the_turn_continues(workspace: Path, tmp_path: Path) -> None:
    """A refusal is an answer the model can act on, not a crash."""
    bundle, events, _ = _agent(
        workspace,
        tmp_path,
        [
            tool_turn("terminal", {"command": "rm -rf /"}),
            text_turn("I was not allowed to run that."),
        ],
        approval=always_deny(),
    )
    session = bundle.store.latest()
    assert session is not None

    result = bundle.runner.run_turn("delete everything", session=session)

    assert result.exit_reason == "completed"
    assert events.of_type(ToolCallFinished)[0].is_error


def test_parallel_reads_happen_in_one_iteration(workspace: Path, tmp_path: Path) -> None:
    for name in ("a.txt", "b.txt"):
        (workspace / name).write_text(name, encoding="utf-8")
    bundle, events, _ = _agent(
        workspace,
        tmp_path,
        [
            ScriptedTurn(
                tool_calls=(
                    ("read_file", {"path": "a.txt"}),
                    ("read_file", {"path": "b.txt"}),
                )
            ),
            text_turn("Both read."),
        ],
    )
    session = bundle.store.latest()
    assert session is not None

    result = bundle.runner.run_turn("read both", session=session)

    assert result.iterations == 2
    assert len(events.of_type(ToolCallFinished)) == 2


def test_history_survives_across_turns(workspace: Path, tmp_path: Path) -> None:
    bundle, _, fake = _agent(workspace, tmp_path, [text_turn("first"), text_turn("second")])
    session = bundle.store.latest()
    assert session is not None

    bundle.runner.run_turn("one", session=session)
    bundle.runner.run_turn("two", session=session)

    # The second request must carry the whole conversation, not just the new turn.
    sent = fake.seen_messages[-1]
    assert [m.role for m in sent] == ["user", "assistant", "user"]
    assert sent[0].text() == "one"


def test_the_transcript_is_durable(workspace: Path, tmp_path: Path) -> None:
    """`harn --continue` after a crash depends on this."""
    bundle, _, _ = _agent(workspace, tmp_path, [text_turn("answer")])
    session = bundle.store.latest()
    assert session is not None

    bundle.runner.run_turn("question", session=session)

    reloaded = bundle.store.history(session.id)
    assert [m.role for m in reloaded] == ["user", "assistant"]
    assert reloaded[1].text() == "answer"


def test_a_transient_failure_is_retried(workspace: Path, tmp_path: Path) -> None:
    bundle, _, _ = _agent(
        workspace,
        tmp_path,
        [
            ScriptedTurn(raises=RateLimited("slow down", retry_after_s=0.0)),
            text_turn("recovered"),
        ],
    )
    session = bundle.store.latest()
    assert session is not None

    result = bundle.runner.run_turn("go", session=session)

    assert result.exit_reason == "completed"
    assert result.final_text == "recovered"


def test_auth_failure_falls_back_to_the_next_model(workspace: Path, tmp_path: Path) -> None:
    """A silent downgrade is worse than a loud one, so the event is emitted."""
    primary = FakeTransport([ScriptedTurn(raises=AuthError("bad key"))])
    events = RecordingSink()
    bundle = build_agent(
        model="fake/primary",
        fallbacks=["fake/backup"],
        workspace=workspace,
        sessions_dir=tmp_path / "sessions",
        surface="cli",
        emit=events,
        approval=ApprovalPolicy(surface="cli", modes={"cli": Mode.ALLOW}),
        transports={"fake": primary},
    )
    primary.push(text_turn("answered by the backup"))
    session = bundle.store.latest()
    assert session is not None

    result = bundle.runner.run_turn("go", session=session)

    assert result.exit_reason == "completed"
    assert result.model_used == "backup"
    assert events.of_type(ProviderFallback)


def test_max_iterations_stops_a_runaway(workspace: Path, tmp_path: Path) -> None:
    """A model that only ever calls tools must not loop forever."""
    fake = FakeTransport([tool_turn("read_file", {"path": "pyproject.toml"}) for _ in range(10)])
    events = RecordingSink()
    bundle = build_agent(
        model="fake/scripted",
        workspace=workspace,
        sessions_dir=tmp_path / "sessions",
        surface="cli",
        emit=events,
        approval=ApprovalPolicy(surface="cli", modes={"cli": Mode.ALLOW}),
        transports={"fake": fake},
        max_iterations=3,
    )
    session = bundle.store.latest()
    assert session is not None

    result = bundle.runner.run_turn("loop", session=session)

    assert result.exit_reason == "max_iterations"
    assert result.iterations == 3


def test_interrupting_mid_stream_leaves_a_replayable_history(
    workspace: Path, tmp_path: Path
) -> None:
    """The whole point of the interrupt path: the *next* turn must still work.

    Ctrl-C during streaming is simulated the way it actually happens -- the
    cancel arrives while chunks are still arriving, not between turns.
    """
    fake = FakeTransport(
        [
            ScriptedTurn(
                thinking="a partial thought",
                thinking_signature=None,
                text="a long answer being streamed one chunk at a time",
            ),
            text_turn("carried on"),
        ]
    )
    events = RecordingSink()
    bundle = build_agent(
        model="fake/scripted",
        workspace=workspace,
        sessions_dir=tmp_path / "sessions",
        surface="cli",
        approval=ApprovalPolicy(surface="cli", modes={"cli": Mode.ALLOW}),
        transports={"fake": fake},
        emit=events,
    )
    session = bundle.store.latest()
    assert session is not None

    fired = False

    def stop_on_first_chunk(event: object) -> None:
        nonlocal fired
        if isinstance(event, TextChunk) and not fired:
            fired = True
            bundle.runner.interrupt("user pressed ctrl-c")

    bundle.runner._emit = FanOutSink(events, stop_on_first_chunk)

    first = bundle.runner.run_turn("go", session=session)
    assert first.exit_reason == "interrupted"

    # A new turn starts uncancelled; one Ctrl-C must not poison the session.
    second = bundle.runner.run_turn("carry on", session=session)
    assert second.exit_reason == "completed"

    # The replayed history must contain nothing the provider would reject:
    # no unsigned thinking, and no tool call left unanswered.
    fake.assert_no_unsigned_thinking_sent()


def test_the_tool_list_offered_matches_the_enabled_toolsets(
    workspace: Path, tmp_path: Path
) -> None:
    fake = FakeTransport([text_turn("ok")])
    bundle = build_agent(
        model="fake/scripted",
        workspace=workspace,
        sessions_dir=tmp_path / "sessions",
        toolsets=["file"],
        surface="cli",
        approval=ApprovalPolicy(surface="cli", modes={"cli": Mode.ALLOW}),
        transports={"fake": fake},
    )
    session = bundle.store.latest()
    assert session is not None

    bundle.runner.run_turn("hi", session=session)

    offered = {tool.name for tool in fake.seen_tools[-1]}
    assert "read_file" in offered
    assert "terminal" not in offered


def test_volatile_content_stays_out_of_the_cached_prefix(workspace: Path, tmp_path: Path) -> None:
    """The invariant that silently triples a bill when it breaks."""
    bundle, _, fake = _agent(workspace, tmp_path, [text_turn("ok")])
    session = bundle.store.latest()
    assert session is not None

    bundle.runner.run_turn("hi", session=session)

    segments = fake.seen_systems[-1].segments
    breakpoint_at = max((i for i, s in enumerate(segments) if s.cache_breakpoint), default=-1)
    assert breakpoint_at >= 0, "no cache breakpoint was placed"
    cached = "\n".join(s.text for s in segments[: breakpoint_at + 1])
    assert "Current time:" not in cached


def test_the_turn_finishes_with_a_usage_report(workspace: Path, tmp_path: Path) -> None:
    bundle, events, _ = _agent(workspace, tmp_path, [text_turn("ok")])
    session = bundle.store.latest()
    assert session is not None

    bundle.runner.run_turn("hi", session=session)

    finished = events.of_type(TurnFinished)[-1]
    assert finished.reason == "completed"
    assert finished.usage.output_tokens > 0
    assert events.of_type(TextChunk)
