"""Handing a sub-task to a fresh agent, and getting back only the answer.

The reason to delegate is context, not parallelism. A task like "find every
place this setting is read" costs forty tool calls and produces one paragraph of
useful conclusion; running it inline means the parent carries all forty results
for the rest of the session. A subagent runs them in its own window and returns
the paragraph.

Which makes the return value the whole design. A subagent that hands back its
transcript has saved nothing, so what crosses the boundary is its final text and
a usage count -- never its history, never its tool results. The parent gets a
report, the same way it would from a colleague.

Three limits, all of which exist because the failure they prevent is expensive
and silent:

* **Depth.** A subagent that can delegate can delegate to something that
  delegates, and the bill is exponential. Depth is carried in the context and
  the tool disappears at the limit.
* **Toolsets narrow, never widen.** A child's toolsets are intersected with its
  parent's. Otherwise delegation is a privilege-escalation primitive: an agent
  denied ``terminal`` asks a child to run the command.
* **Taint propagates upward.** If the child read the web, the parent's session is
  tainted too -- the conclusion drawn from that content is in the parent now.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from harness_agentic.core.events import Notice, SkillLoaded
from harness_agentic.core.types import Usage
from harness_agentic.tools.spec import Danger, ToolContext, ToolResult

if TYPE_CHECKING:
    from harness_agentic.agent.build import AgentBundle
    from harness_agentic.tools.registry import ToolRegistry

MAX_DEPTH = 2
"""A subagent may delegate once more, and that is all."""
MAX_CHILDREN_PER_TURN = 6
DEFAULT_MAX_ITERATIONS = 20
MAX_REPORT_CHARS = 8_000


class DelegateParams(BaseModel):
    """Arguments for ``delegate``."""

    task: str = Field(
        min_length=1,
        description=(
            "The complete task, written as if to someone with no context. The "
            "subagent cannot see this conversation."
        ),
    )
    toolsets: list[str] = Field(
        default_factory=list,
        description=(
            "Toolsets the subagent needs. Narrower is better, and anything not "
            "available here cannot be granted."
        ),
    )
    max_iterations: int = Field(
        DEFAULT_MAX_ITERATIONS, gt=0, le=60, description="Cap on the subagent's steps."
    )


@dataclass(frozen=True, slots=True)
class Report:
    """What a subagent hands back."""

    task: str
    answer: str
    exit_reason: str
    iterations: int
    usage: Usage
    tainted: bool = False
    tools_used: tuple[str, ...] = ()

    def render(self) -> str:
        """Format for the parent's tool result.

        The exit reason is included deliberately. A subagent that stopped at its
        iteration cap has produced a partial answer, and a parent that treats it
        as complete will build on something unfinished.
        """
        head = self.answer.strip()[:MAX_REPORT_CHARS] or "(the subagent produced no answer)"
        notes = [f"{self.iterations} step(s)"]
        if self.tools_used:
            notes.append("used " + ", ".join(dict.fromkeys(self.tools_used)))
        if self.exit_reason != "completed":
            notes.append(f"stopped early: {self.exit_reason}")
        return f"{head}\n\n[subagent: {'; '.join(notes)}]"


@dataclass
class DelegationLimits:
    """What a parent may spend on children."""

    max_depth: int = MAX_DEPTH
    max_children: int = MAX_CHILDREN_PER_TURN
    allowed_toolsets: tuple[str, ...] = ()
    """The parent's own toolsets. A child's request is intersected with this."""
    spawned: int = 0

    def remaining(self) -> int:
        """How many more children this turn may spawn."""
        return max(0, self.max_children - self.spawned)

    def narrow(self, requested: Sequence[str]) -> list[str]:
        """The toolsets a child may actually have.

        Intersection, always. Delegation must not be a way to reach a toolset
        the parent was denied -- an agent without ``terminal`` asking a child to
        run the command is the whole attack.
        """
        permitted = set(self.allowed_toolsets)
        asked = [name for name in requested if name in permitted]
        # `core` is always on: a child that cannot search its own history has to
        # guess at anything it was not told.
        return list(dict.fromkeys(["core", *asked]))

    def child(self) -> DelegationLimits:
        """The limits for one level down."""
        return DelegationLimits(
            max_depth=self.max_depth - 1,
            max_children=self.max_children,
            allowed_toolsets=self.allowed_toolsets,
        )


ChildFactory = Callable[[str, Sequence[str], int], "AgentBundle"]
"""Builds a subagent for a task, toolsets and iteration cap."""


@dataclass
class Delegator:
    """Runs subagents on a parent's behalf, within limits."""

    factory: ChildFactory
    limits: DelegationLimits = field(default_factory=DelegationLimits)
    reports: list[Report] = field(default_factory=list)

    def run(self, task: str, toolsets: Sequence[str], max_iterations: int) -> Report:
        """Run one subagent to completion and return its report."""
        granted = self.limits.narrow(toolsets)
        child = self.factory(task, granted, max_iterations)
        session = child.store.get(child.context.session_id)
        if session is None:  # pragma: no cover - just created
            detail = "the subagent's session is missing from the store"
            raise RuntimeError(detail)

        result = child.runner.run_turn(task, session=session)
        report = Report(
            task=task,
            answer=result.final_text,
            exit_reason=result.exit_reason,
            iterations=result.iterations,
            usage=result.usage,
            # Propagated upward: the conclusion drawn from untrusted content is
            # in the parent now, so the parent's session is tainted too.
            tainted=child.executor.tainted,
            tools_used=tuple(record.tool for record in child.executor.records),
        )
        self.reports.append(report)
        self.limits.spawned += 1
        return report

    def total_usage(self) -> Usage:
        """What every child has cost, so far."""
        total = Usage()
        for report in self.reports:
            total = Usage(
                input_tokens=total.input_tokens + report.usage.input_tokens,
                output_tokens=total.output_tokens + report.usage.output_tokens,
                cache_read_tokens=total.cache_read_tokens + report.usage.cache_read_tokens,
                cache_write_tokens=total.cache_write_tokens + report.usage.cache_write_tokens,
            )
        return total


def install_delegate_tool(target: ToolRegistry, delegator: Delegator) -> ToolRegistry:
    """Register ``delegate``, unless the depth limit has been reached.

    Not registered at all at the limit, rather than registered and refusing. A
    tool the model can see and cannot use is a tool it keeps trying.
    """
    if delegator.limits.max_depth <= 0:
        return target

    @target.tool(
        toolset="core",
        danger=Danger.SAFE,
        name="delegate",
        max_result_chars=MAX_REPORT_CHARS + 500,
        override=True,
    )
    def delegate(params: DelegateParams, ctx: ToolContext) -> ToolResult:
        """Hand a self-contained sub-task to a fresh agent and get its answer.

        Use for work that needs many steps but yields a short conclusion --
        searching a large codebase, reading several documents. The subagent
        cannot see this conversation, so state the task completely.
        """
        if delegator.limits.remaining() <= 0:
            return ToolResult.error(
                f"the delegation budget for this turn is spent "
                f"({delegator.limits.max_children} subagent(s)); do the rest here"
            )
        ctx.emit(f"delegating: {params.task[:80]}")
        granted = delegator.limits.narrow(params.toolsets)
        dropped = sorted(set(params.toolsets) - set(granted))

        try:
            report = delegator.run(params.task, params.toolsets, params.max_iterations)
        except Exception as exc:  # a failed child is a tool error, not a crash
            return ToolResult.error(f"the subagent failed: {type(exc).__name__}: {exc}")

        text = report.render()
        if dropped:
            # Said out loud: a child silently denied a toolset produces a
            # confusing partial answer, and the parent cannot tell why.
            text += f"\n[not granted: {', '.join(dropped)} -- unavailable to this agent]"
        return ToolResult(
            text=text,
            display=f"subagent: {params.task[:60]} ({report.iterations} step(s))",
            tainted=report.tainted,
            data={"exit_reason": report.exit_reason, "toolsets": granted},
        )

    return target


def announce(report: Report) -> Notice:
    """An event describing a finished subagent, for surfaces that show progress."""
    return Notice(
        "info",
        f"subagent finished in {report.iterations} step(s): {report.exit_reason}",
        detail={"tainted": report.tainted, "tools": list(report.tools_used)},
    )


def skill_event(name: str, version: str, tokens: int) -> SkillLoaded:
    """Re-exported so a subagent's skill loads surface on the parent's stream."""
    return SkillLoaded(name=name, version=version, tokens=tokens)
