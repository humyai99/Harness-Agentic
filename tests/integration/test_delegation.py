"""Delegation and cron, through the real loop.

The point of delegation is context, not parallelism: a search that costs forty
tool calls and yields one paragraph should leave the parent holding the
paragraph. So the assertion that matters is what the parent's history contains
afterwards -- and it is not the child's tool results.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from harness_agentic.agent.build import AgentBundle, build_agent
from harness_agentic.agent.delegate import DelegationLimits
from harness_agentic.core.clock import ManualClock
from harness_agentic.cron.runner import CronRunner, Job
from harness_agentic.net.fetch import RecordedFetcher
from harness_agentic.net.policy import UrlPolicy
from harness_agentic.testing import FakeTransport, ScriptedTurn, text_turn, tool_turn
from harness_agentic.tools.approval import ApprovalPolicy, Mode

pytestmark = pytest.mark.usefixtures("isolated_home")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "app.py").write_text("SETTING = read_config('retries')\n", encoding="utf-8")
    (root / "worker.py").write_text("x = read_config('retries')\n", encoding="utf-8")
    return root


def agent(
    workspace: Path,
    tmp_path: Path,
    script: list[ScriptedTurn],
    *,
    toolsets: list[str] | None = None,
    delegation: DelegationLimits | None = None,
    **kwargs: object,
) -> AgentBundle:
    return build_agent(
        model="fake/scripted",
        workspace=workspace,
        sessions_dir=tmp_path / "sessions",
        toolsets=toolsets or ["file"],
        surface="cli",
        approval=ApprovalPolicy(surface="cli", modes={"cli": Mode.ALLOW}),
        transports={"fake": FakeTransport(script)},
        delegation=delegation,
        **kwargs,  # type: ignore[arg-type]
    )


def run(bundle: AgentBundle, prompt: str) -> object:
    session = bundle.store.latest()
    assert session is not None
    return bundle.runner.run_turn(prompt, session=session)


def test_a_subagent_answers_and_only_its_answer_comes_back(workspace: Path, tmp_path: Path) -> None:
    # The child spends three turns; the parent ends up holding one paragraph.
    script = [
        tool_turn("delegate", {"task": "find every caller of read_config", "toolsets": ["file"]}),
        # -- the child's script, consumed by the child's own runner --
        tool_turn("grep_files", {"pattern": "read_config"}),
        text_turn("read_config is called in app.py and worker.py."),
        # -- back in the parent --
        text_turn("Two callers: app.py and worker.py."),
    ]
    bundle = agent(workspace, tmp_path, script)
    result = run(bundle, "where is read_config used?")

    assert "app.py" in result.final_text  # type: ignore[attr-defined]
    history = "\n".join(
        block.text
        for message in bundle.store.history(bundle.context.session_id)
        for block in message.content
        if getattr(block, "text", None)
    )
    # The child's conclusion is here...
    assert "app.py and worker.py" in history
    # ...and the raw grep output it drew that from is not.
    assert "SETTING = read_config" not in history


def test_a_subagent_cannot_reach_a_toolset_the_parent_lacks(
    workspace: Path, tmp_path: Path
) -> None:
    # The parent has `file` only. Asking for `terminal` on behalf of a child is
    # the privilege-escalation route, and it is closed.
    script = [
        tool_turn("delegate", {"task": "run the tests", "toolsets": ["terminal"]}),
        text_turn("I cannot run commands."),
        text_turn("Reported: the subagent had no terminal access."),
    ]
    bundle = agent(workspace, tmp_path, script, toolsets=["file"])
    run(bundle, "run the tests via a subagent")

    records = [r for r in bundle.executor.records if r.tool == "delegate"]
    assert len(records) == 1
    assert "not granted: terminal" in records[0].result.text
    assert records[0].result.data is not None
    assert "terminal" not in records[0].result.data["toolsets"]  # type: ignore[operator]


def test_delegation_disappears_at_the_depth_limit(workspace: Path, tmp_path: Path) -> None:
    # Not present-and-refusing: a tool the model can see and cannot use is a
    # tool it keeps trying.
    deep = agent(
        workspace,
        tmp_path,
        [text_turn("ok")],
        delegation=DelegationLimits(max_depth=0, allowed_toolsets=("file",)),
    )
    assert "delegate" not in deep.registry.all()

    shallow = agent(
        workspace,
        tmp_path / "b",
        [text_turn("ok")],
        delegation=DelegationLimits(max_depth=1, allowed_toolsets=("file",)),
    )
    assert "delegate" in shallow.registry.all()


def test_the_per_turn_child_budget_is_enforced(workspace: Path, tmp_path: Path) -> None:
    script = [
        tool_turn("delegate", {"task": "first", "toolsets": []}),
        text_turn("child one done"),
        tool_turn("delegate", {"task": "second", "toolsets": []}),
        text_turn("both attempted"),
    ]
    bundle = agent(
        workspace,
        tmp_path,
        script,
        delegation=DelegationLimits(max_depth=1, max_children=1, allowed_toolsets=("file",)),
    )
    run(bundle, "delegate twice")

    refusals = [r for r in bundle.executor.records if r.tool == "delegate" and r.result.is_error]
    assert len(refusals) == 1
    assert "budget" in refusals[0].result.text


def test_a_failing_subagent_is_a_tool_error_not_a_crash(workspace: Path, tmp_path: Path) -> None:
    from harness_agentic.errors import AuthError

    script = [
        tool_turn("delegate", {"task": "do something", "toolsets": []}),
        ScriptedTurn(raises=AuthError("the child's key is bad")),
        text_turn("The subagent could not run."),
    ]
    bundle = agent(workspace, tmp_path, script)
    result = run(bundle, "delegate it")

    assert result.exit_reason == "completed"  # type: ignore[attr-defined]
    assert any(r.tool == "delegate" for r in bundle.executor.records)


def test_taint_propagates_from_a_subagent_to_its_parent(workspace: Path, tmp_path: Path) -> None:
    # The conclusion drawn from untrusted content is in the parent now, so the
    # parent's session is tainted too.
    def resolve(host: str) -> list[object]:
        import ipaddress

        del host
        return [ipaddress.ip_address("93.184.216.34")]

    net = RecordedFetcher(policy=UrlPolicy(resolver=resolve))
    net.add("https://docs.test/a", "<p>the docs say retries default to 3</p>")

    script = [
        tool_turn("delegate", {"task": "read the docs page", "toolsets": ["web"]}),
        tool_turn("web_fetch", {"url": "https://docs.test/a"}),
        text_turn("Retries default to 3."),
        text_turn("The docs say retries default to 3."),
    ]
    bundle = agent(workspace, tmp_path, script, toolsets=["web"], fetcher=net)
    assert not bundle.executor.tainted
    run(bundle, "what is the retry default?")

    assert bundle.executor.tainted


# -- cron ---------------------------------------------------------------------------


def test_a_due_job_runs_and_is_recorded(workspace: Path, tmp_path: Path) -> None:
    clock = ManualClock(datetime(2026, 3, 1, 2, 30, tzinfo=UTC))
    made: list[AgentBundle] = []

    def factory(job: Job) -> AgentBundle:
        bundle = agent(
            workspace,
            tmp_path / job.name,
            [text_turn("the backups are fine")],
            toolsets=list(job.toolsets),
        )
        made.append(bundle)
        return bundle

    runner = CronRunner(bundle_factory=factory, clock=clock)
    runner.add(Job.create("nightly", "30 2 * * *", "check the backups"))
    runs = runner.tick()

    assert len(runs) == 1
    assert runs[0].outcome == "ok"
    assert runs[0].session_id == made[0].context.session_id
    assert "nightly" in "\n".join(runner.report())


def test_a_job_does_not_fire_twice_in_one_minute(workspace: Path, tmp_path: Path) -> None:
    clock = ManualClock(datetime(2026, 3, 1, 2, 30, tzinfo=UTC))
    runner = CronRunner(
        bundle_factory=lambda job: agent(workspace, tmp_path / job.name, [text_turn("done")] * 5),
        clock=clock,
    )
    runner.add(Job.create("nightly", "30 2 * * *", "check"))

    assert len(runner.tick()) == 1
    # Ticking again inside the same minute must not run it again.
    assert runner.tick() == []


def test_a_missed_window_is_not_backfilled(workspace: Path, tmp_path: Path) -> None:
    # A process down for six hours must not wake up and run the hourly report
    # six times.
    clock = ManualClock(datetime(2026, 3, 1, 8, 5, tzinfo=UTC))
    runner = CronRunner(
        bundle_factory=lambda job: agent(workspace, tmp_path / job.name, [text_turn("done")]),
        clock=clock,
    )
    runner.add(Job.create("hourly", "0 * * * *", "report"))

    # It is 08:05; the 02:00 through 08:00 firings are simply gone.
    assert runner.tick() == []


def test_a_disabled_job_never_runs(workspace: Path, tmp_path: Path) -> None:
    clock = ManualClock(datetime(2026, 3, 1, 2, 30, tzinfo=UTC))
    runner = CronRunner(bundle_factory=lambda job: agent(workspace, tmp_path, []), clock=clock)
    runner.add(Job.create("nightly", "30 2 * * *", "check", enabled=False))
    assert runner.tick() == []


def test_a_failing_job_is_recorded_rather_than_raised(workspace: Path, tmp_path: Path) -> None:
    clock = ManualClock(datetime(2026, 3, 1, 2, 30, tzinfo=UTC))

    def factory(job: Job) -> AgentBundle:
        detail = "the model is unreachable"
        raise RuntimeError(detail)

    runner = CronRunner(bundle_factory=factory, clock=clock)
    runner.add(Job.create("nightly", "30 2 * * *", "check"))
    runs = runner.tick()

    assert runs[0].outcome == "error"
    assert "unreachable" in runs[0].detail


def test_a_notifier_failure_does_not_fail_the_run(workspace: Path, tmp_path: Path) -> None:
    # The result is already in the store; a broken delivery must not rewrite
    # history as a failure.
    clock = ManualClock(datetime(2026, 3, 1, 2, 30, tzinfo=UTC))

    def notify(job: Job, run_record: object, text: str) -> None:
        detail = "the chat platform is down"
        raise RuntimeError(detail)

    runner = CronRunner(
        bundle_factory=lambda job: agent(workspace, tmp_path / job.name, [text_turn("all good")]),
        clock=clock,
        notifier=notify,  # type: ignore[arg-type]
    )
    runner.add(Job.create("nightly", "30 2 * * *", "check"))
    assert runner.tick()[0].outcome == "ok"


def test_upcoming_firings_are_listed_soonest_first(workspace: Path, tmp_path: Path) -> None:
    clock = ManualClock(datetime(2026, 3, 1, 1, 0, tzinfo=UTC))
    runner = CronRunner(clock=clock)
    runner.add(Job.create("late", "0 23 * * *", "a"))
    runner.add(Job.create("soon", "0 2 * * *", "b"))

    upcoming = runner.next_runs()
    assert [name for name, _ in upcoming] == ["soon", "late"]


def test_a_cron_job_cannot_be_asked_for_approval(workspace: Path, tmp_path: Path) -> None:
    # There is nobody to ask, so nothing consequential is permitted unless the
    # operator wrote it down in advance.
    job = Job.create("nightly", "@daily", "deploy staging")
    decision = job.policy().check(
        __import__("harness_agentic.tools.spec", fromlist=["ApprovalRequest"]).ApprovalRequest(
            tool="terminal", danger=3, summary="kubectl apply -f ."
        )
    )
    assert not decision.granted
    assert "cannot ask anyone" in decision.reason
