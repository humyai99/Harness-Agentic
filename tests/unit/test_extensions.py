"""Plugins, cron schedules, and delegation limits.

Each of these has one property that carries the design, and each is a property
about *refusal*: a plugin cannot declare its own tool safe, a cron job cannot
backfill a window it missed, and a subagent cannot be granted a toolset its
parent was denied.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from harness_agentic.agent.delegate import DelegationLimits, Report
from harness_agentic.core.types import Usage
from harness_agentic.cron.runner import Job, load_jobs
from harness_agentic.cron.schedule import BadSchedule, parse
from harness_agentic.plugins.loader import PLUGIN_FLOOR, discover, load
from harness_agentic.tools.approval import Mode
from harness_agentic.tools.registry import ToolRegistry
from harness_agentic.tools.spec import Danger

# -- plugins ---------------------------------------------------------------------

GOOD_PLUGIN = '''
"""A plugin that registers one tool."""
from pydantic import BaseModel

from harness_agentic.tools.spec import Danger, ToolContext, ToolResult


class Params(BaseModel):
    """Arguments."""

    text: str = "hi"


def setup(registry):
    """Register the tool."""

    @registry.tool(toolset="demo", danger=Danger.SAFE, name="demo_echo")
    def demo_echo(params: Params, ctx: ToolContext) -> ToolResult:
        """Echo the text back."""
        return ToolResult(text=params.text)
'''

BROKEN_PLUGIN = '''
"""A plugin that raises on import."""
raise RuntimeError("this plugin is broken")
'''

NO_SETUP_PLUGIN = '''
"""A plugin missing its entry point."""
VALUE = 1
'''


@pytest.fixture
def plugin_dir(tmp_path: Path) -> Path:
    root = tmp_path / "plugins"
    root.mkdir()
    (root / "good.py").write_text(GOOD_PLUGIN, encoding="utf-8")
    return root


def test_a_plugin_registers_its_tool(plugin_dir: Path) -> None:
    registry = ToolRegistry()
    result = load(registry, discover(home=plugin_dir, include_entry_points=False))

    assert result.tool_names() == ("demo_echo",)
    assert not result.failures
    assert "demo_echo" in registry.all()


def test_a_plugin_tool_cannot_claim_to_be_safe(plugin_dir: Path) -> None:
    # The plugin declared SAFE. A plugin that could route around the approval
    # policy by declaring its own danger level is a plugin that will.
    registry = ToolRegistry()
    load(registry, discover(home=plugin_dir, include_entry_points=False))

    tool = registry.get("demo_echo")
    assert tool.danger == PLUGIN_FLOOR
    assert tool.danger > Danger.SAFE
    assert tool.source == "plugin:good"


def test_a_broken_plugin_is_reported_and_does_not_stop_the_others(tmp_path: Path) -> None:
    root = tmp_path / "plugins"
    root.mkdir()
    (root / "good.py").write_text(GOOD_PLUGIN, encoding="utf-8")
    (root / "broken.py").write_text(BROKEN_PLUGIN, encoding="utf-8")
    (root / "nosetup.py").write_text(NO_SETUP_PLUGIN, encoding="utf-8")

    registry = ToolRegistry()
    result = load(registry, discover(home=root, include_entry_points=False))

    assert result.tool_names() == ("demo_echo",)
    assert {failure.plugin.name for failure in result.failures} == {"broken", "nosetup"}
    # Named in the report, so nobody spends an afternoon wondering where their
    # tool went.
    joined = "\n".join(result.report())
    assert "broken" in joined
    assert "this plugin is broken" in joined
    assert "no setup(registry) function" in joined


def test_a_directory_plugin_is_found(tmp_path: Path) -> None:
    root = tmp_path / "plugins"
    (root / "pack").mkdir(parents=True)
    (root / "pack" / "__init__.py").write_text(GOOD_PLUGIN, encoding="utf-8")

    found = discover(home=root, include_entry_points=False)
    assert [entry.name for entry in found] == ["pack"]


def test_the_project_overrides_the_user_directory(tmp_path: Path) -> None:
    # A repository's own extension is the most specific statement of intent, and
    # the one that went through code review.
    user = tmp_path / "user"
    project = tmp_path / "project"
    user.mkdir()
    project.mkdir()
    (user / "shared.py").write_text(GOOD_PLUGIN, encoding="utf-8")
    (project / "shared.py").write_text(GOOD_PLUGIN, encoding="utf-8")

    found = discover(home=user, project=project, include_entry_points=False)
    assert len(found) == 1
    assert found[0].origin == "project"


def test_an_allowlist_stops_a_dropped_in_plugin_from_running(plugin_dir: Path) -> None:
    # The control that matters for a deployment: naming what may load means a
    # file appearing in the directory later does not run on its own.
    registry = ToolRegistry()
    result = load(registry, discover(home=plugin_dir, include_entry_points=False), allow=[])

    assert result.tool_names() == ()
    assert [entry.name for entry in result.skipped] == ["good"]


def test_a_denied_plugin_is_skipped(plugin_dir: Path) -> None:
    registry = ToolRegistry()
    result = load(registry, discover(home=plugin_dir, include_entry_points=False), deny=["good"])
    assert result.tool_names() == ()


def test_dotfiles_and_private_modules_are_ignored(tmp_path: Path) -> None:
    root = tmp_path / "plugins"
    root.mkdir()
    (root / "_helper.py").write_text("x = 1", encoding="utf-8")
    (root / ".hidden.py").write_text("x = 1", encoding="utf-8")
    assert discover(home=root, include_entry_points=False) == []


# -- cron schedules ----------------------------------------------------------------


def at(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def test_a_plain_schedule_matches_its_minute() -> None:
    schedule = parse("30 2 * * *")
    assert schedule.matches(at("2026-03-01T02:30"))
    assert not schedule.matches(at("2026-03-01T02:31"))
    assert not schedule.matches(at("2026-03-01T03:30"))


def test_steps_ranges_and_lists() -> None:
    assert parse("*/15 * * * *").minutes == frozenset({0, 15, 30, 45})
    assert parse("0 9-17 * * *").hours == frozenset(range(9, 18))
    assert parse("0 0 1,15 * *").days == frozenset({1, 15})
    assert parse("0 0 * * mon-fri").weekdays == frozenset({1, 2, 3, 4, 5})


def test_named_months_and_days() -> None:
    assert parse("0 0 1 jan *").months == frozenset({1})
    assert parse("0 0 * * sun").weekdays == frozenset({0})
    # Both 0 and 7 spell Sunday, and people write both.
    assert parse("0 0 * * 7").weekdays == frozenset({0})


def test_a_wrapping_range_works() -> None:
    # `22-2` and `fri-mon` are both things people write, and rejecting them
    # would be a surprise with no upside.
    assert parse("0 22-2 * * *").hours == frozenset({22, 23, 0, 1, 2})


def test_aliases() -> None:
    assert parse("@daily").expression == "0 0 * * *"
    assert parse("@hourly").minutes == frozenset({0})


def test_day_of_month_and_day_of_week_are_ored() -> None:
    # Cron's rule, and it surprises everyone: `0 0 13 * 5` is the 13th AND
    # every Friday, not Friday the 13th. Getting it wrong means eleven firings
    # a year instead of sixty.
    schedule = parse("0 0 13 * 5")
    assert schedule.matches(at("2026-01-13T00:00"))  # a Tuesday, but the 13th
    assert schedule.matches(at("2026-01-16T00:00"))  # a Friday, not the 13th
    assert not schedule.matches(at("2026-01-14T00:00"))
    # And the surprise is stated out loud.
    assert "OR" in schedule.describe()


def test_one_restricted_field_still_ands_with_the_other() -> None:
    schedule = parse("0 0 15 * *")
    assert schedule.matches(at("2026-01-15T00:00"))
    assert not schedule.matches(at("2026-01-16T00:00"))


def test_the_next_firing_is_found() -> None:
    schedule = parse("0 3 * * *")
    assert schedule.next_after(at("2026-03-01T04:00")) == at("2026-03-02T03:00")
    assert schedule.next_after(at("2026-03-01T02:00")) == at("2026-03-01T03:00")


def test_the_next_firing_crosses_a_leap_day() -> None:
    schedule = parse("0 0 29 2 *")
    following = schedule.next_after(at("2026-03-01T00:00"))
    assert following is not None
    assert (following.year, following.month, following.day) == (2028, 2, 29)


def test_an_impossible_schedule_returns_none() -> None:
    assert parse("0 0 30 2 *").next_after(at("2026-01-01T00:00")) is None


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "* * *",
        "60 * * * *",
        "* 24 * * *",
        "0 0 0 * *",
        "0 0 * 13 *",
        "*/0 * * * *",
        "x * * * *",
    ],
)
def test_a_malformed_expression_is_refused(expression: str) -> None:
    # Refused, not silently accepted. A typo that means "never runs" is the one
    # failure a scheduler must not have.
    with pytest.raises(BadSchedule):
        parse(expression)


def test_loading_jobs_refuses_an_incomplete_entry() -> None:
    with pytest.raises(ValueError, match="needs name, schedule and prompt"):
        load_jobs([{"name": "x", "schedule": "@daily"}])


def test_loaded_jobs_default_to_the_strictest_settings() -> None:
    jobs = load_jobs([{"name": "report", "schedule": "@daily", "prompt": "summarize yesterday"}])
    job = jobs[0]

    # Nobody is there to be asked, so nothing consequential is permitted.
    assert job.approval is Mode.DENY
    assert job.toolsets == ("core",)
    assert job.policy().mode() is Mode.DENY
    assert job.timeout_s > 0


def test_a_bad_config_value_falls_back_rather_than_crashing() -> None:
    jobs = load_jobs(
        [
            {
                "name": "x",
                "schedule": "@daily",
                "prompt": "p",
                "max_iterations": "not a number",
                "timeout_s": None,
            }
        ]
    )
    assert jobs[0].max_iterations > 0
    assert jobs[0].timeout_s > 0


def test_a_job_describes_itself() -> None:
    job = Job.create("nightly", "0 2 * * *", "check the backups")
    assert "nightly" in job.describe()
    assert "0 2 * * *" in job.describe()


# -- delegation limits --------------------------------------------------------------


def test_a_child_cannot_be_granted_what_the_parent_lacks() -> None:
    # The whole attack: an agent denied `terminal` asks a child to run the
    # command instead.
    limits = DelegationLimits(allowed_toolsets=("file", "web"))
    granted = limits.narrow(["file", "terminal", "web"])

    assert "terminal" not in granted
    assert set(granted) == {"core", "file", "web"}


def test_core_is_always_granted() -> None:
    # A child that cannot search its own history has to guess at anything it
    # was not told.
    assert "core" in DelegationLimits(allowed_toolsets=()).narrow([])


def test_depth_decreases_one_level_at_a_time() -> None:
    parent = DelegationLimits(max_depth=2, allowed_toolsets=("file",))
    child = parent.child()
    grandchild = child.child()

    assert (parent.max_depth, child.max_depth, grandchild.max_depth) == (2, 1, 0)
    # At zero the tool is not registered at all, so there is no fourth level.
    assert grandchild.max_depth == 0
    assert grandchild.allowed_toolsets == ("file",)


def test_the_per_turn_child_budget_is_finite() -> None:
    limits = DelegationLimits(max_children=2)
    assert limits.remaining() == 2
    limits.spawned = 2
    assert limits.remaining() == 0


def test_a_report_states_when_a_subagent_stopped_early() -> None:
    # A parent that treats a capped answer as complete builds on something
    # unfinished.
    report = Report(
        task="find every caller",
        answer="I found three so far.",
        exit_reason="max_iterations",
        iterations=20,
        usage=Usage(input_tokens=100, output_tokens=20),
        tools_used=("grep_files", "read_file"),
    )
    rendered = report.render()

    assert "stopped early: max_iterations" in rendered
    assert "grep_files" in rendered
    assert "20 step(s)" in rendered


def test_an_empty_answer_says_so_rather_than_being_blank() -> None:
    report = Report(task="t", answer="   ", exit_reason="completed", iterations=1, usage=Usage())
    assert "produced no answer" in report.render()


def test_a_long_answer_is_bounded() -> None:
    report = Report(
        task="t", answer="x" * 50_000, exit_reason="completed", iterations=1, usage=Usage()
    )
    assert len(report.render()) < 10_000


def test_schedules_are_utc_only() -> None:
    # A naive local time in a DST zone either runs twice or not at all on two
    # days a year, silently. Everything here is UTC.
    schedule = parse("30 2 * * *")
    aware = datetime(2026, 3, 29, 2, 30, tzinfo=UTC)
    assert schedule.matches(aware)
    shifted = aware.astimezone(UTC) + timedelta(hours=1)
    assert not schedule.matches(shifted)
