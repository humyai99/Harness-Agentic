"""``harn skills`` -- the human half of the self-improvement loop.

Autonomy defaults to ``propose``, which means nothing reaches the library without
a person approving it. That default is only meaningful if there is a way for the
person to *see* what is waiting, so these commands are not a convenience: without
them the default setting is a queue that fills up and is never read, and the
operator's only options are to leave every proposal pending forever or to switch
the gate off.

``diff`` before ``approve`` is the intended order, and ``approve`` re-validates
rather than trusting what staging recorded -- the library may have moved since,
and a patch that no longer applies must be refused rather than written as an
unchanged file.

``audit`` is the command worth running in CI. It replays every skill's routing
cases against the whole catalog and fails when one skill has started capturing
another's requests, which is the specific failure that turns a large library from
an asset into a liability.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.syntax import Syntax
from rich.table import Table

from harness_agentic.agent.build import default_library
from harness_agentic.cli.render import console
from harness_agentic.config import load_settings
from harness_agentic.constants import harness_home
from harness_agentic.skills.model import Lifecycle
from harness_agentic.skills.proposals import Autonomy, ProposalStore, Quotas
from harness_agentic.skills.testing import SkillTestRunner, keyword_router
from harness_agentic.tools.builtin import builtin_registry

if TYPE_CHECKING:
    from harness_agentic.skills.registry import SkillRegistry

TESTS_FILE = "tests/cases.yaml"


def register(app: typer.Typer) -> None:
    """Attach the ``skills`` command group."""
    group = typer.Typer(
        help="Inspect the skill library and review what the agent proposes.",
        no_args_is_help=True,
    )
    app.add_typer(group, name="skills")

    @group.command("list")
    def list_skills(
        workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w"),
        all_lifecycles: bool = typer.Option(
            False, "--all", help="Include archived and quarantined skills."
        ),
    ) -> None:
        """Show every skill the agent would be told about."""
        library = _library(workspace)
        skills = library.all()
        if not all_lifecycles:
            skills = [s for s in skills if s.lifecycle is Lifecycle.ACTIVE]
        if not skills:
            console.print("No skills found.")
            console.print(
                f"[dim]Looked in {harness_home() / 'skills'} and {_project(workspace)}.[/]"
            )
            return

        table = Table("skill", "version", "trust", "state", "uses", "wins", "description")
        for meta in skills:
            stats = library.stats(meta.name)
            table.add_row(
                meta.name,
                meta.version,
                meta.trust.name.lower(),
                meta.lifecycle.value,
                str(stats.uses),
                f"{stats.win_rate:.0%}" if stats.uses else "-",
                meta.description.split("\n")[0][:60],
            )
        console.print(table)
        catalog = library.catalog(_active_tools(workspace))
        console.print(
            f"[dim]{len(skills)} skill(s); the catalog costs about "
            f"{catalog.tokens:,} tokens per session[/]"
        )
        if catalog.omitted:
            console.print(
                f"[yellow]over budget:[/] {', '.join(catalog.omitted)} are not in the "
                f"catalog and can only be found with skill_search"
            )

    @group.command("doctor")
    def doctor(workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w")) -> None:
        """Report skills that failed to load, and which shadow which."""
        library = _library(workspace)
        problems = library.problems()
        for problem in problems:
            console.print(f"[red]{problem.path}[/]")
            for finding in problem.findings:
                console.print(f"  [{finding.severity}] {finding.code}: {finding.message}")

        shadowed = [meta for meta in library.all() if meta.shadows]
        for meta in shadowed:
            console.print(
                f"[yellow]{meta.name}[/] at {meta.path} shadows "
                f"{', '.join(str(p) for p in meta.shadows)}"
            )
        if not problems and not shadowed:
            console.print("[green]every skill loaded, nothing shadowed[/]")
        if problems:
            raise typer.Exit(code=1)

    @group.command("audit")
    def audit(workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w")) -> None:
        """Replay every skill's routing cases against the whole catalog.

        The one check that notices a new skill quietly capturing an existing
        skill's requests. Worth running in CI: a library only stays useful while
        each skill is still chosen for the work it was written for.
        """
        library = _library(workspace)
        # The keyword router, not a model: this has to be runnable in CI without a
        # key, and it catches the blunt collisions, which are most of them. A
        # deployment that wants the real thing passes its own router.
        runner = SkillTestRunner(router=keyword_router)
        registered = 0
        for meta in library.all():
            cases = meta.path / TESTS_FILE
            if cases.is_file():
                runner.register(meta.name, cases.read_text(encoding="utf-8"))
                registered += 1

        if not registered:
            console.print("No skill ships routing cases, so there is nothing to audit.")
            console.print("[dim]Every agent-proposed skill is required to have them.[/]")
            return

        report = runner.audit(library.all())
        console.print(report.summary())
        if not report.clean:
            raise typer.Exit(code=1)

    # -- review ----------------------------------------------------------------

    @group.command("pending")
    def pending() -> None:
        """List proposals waiting for review."""
        store = _proposals()
        waiting = store.pending()
        if not waiting:
            console.print("Nothing is waiting for review.")
            return
        table = Table("id", "kind", "target", "tainted", "rationale")
        for proposal in waiting:
            table.add_row(
                proposal.proposal_id,
                proposal.kind,
                proposal.target_name,
                "[red]yes[/]" if proposal.tainted else "no",
                proposal.rationale[:60],
            )
        console.print(table)
        console.print(f"[dim]{len(waiting)} proposal(s). `harn skills diff <id>` to read one.[/]")

    @group.command("diff")
    def diff(
        proposal_id: str = typer.Argument(..., help="A proposal id from `skills pending`."),
    ) -> None:
        """Show what approving a proposal would change."""
        store = _proposals()
        proposal = store.get(proposal_id)
        if proposal is None:
            console.print(f"[red]no proposal {proposal_id!r}[/]")
            raise typer.Exit(code=1)

        console.print(f"[bold]{proposal.kind} {proposal.target_name}[/] ({proposal.proposal_id})")
        console.print(f"rationale: {proposal.rationale}")
        for line in proposal.evidence:
            console.print(f"  evidence: {line}")
        if proposal.tainted:
            console.print(
                "[red]tainted:[/] this session read untrusted content, so it needs a "
                "human whatever the autonomy setting says. Read the diff with that in mind."
            )
        console.print()
        console.print(Syntax(store.diff(proposal_id), "diff", theme="ansi_dark"))

    @group.command("approve")
    def approve(
        proposal_id: str = typer.Argument(...),
        actor: str = typer.Option("operator", "--actor", help="Recorded in the provenance."),
    ) -> None:
        """Write a proposal into the library. The only write path there is."""
        store = _proposals()
        result = store.approve(proposal_id, actor=actor)
        if not result.applied:
            console.print(f"[red]not applied:[/] {result.reason}")
            raise typer.Exit(code=1)
        console.print(f"[green]applied[/] to {result.path}")
        console.print(
            "[dim]It joins the catalog for sessions started from now on -- the catalog "
            "is frozen per session so it stays inside the cached prompt prefix.[/]"
        )

    @group.command("reject")
    def reject(
        proposal_id: str = typer.Argument(...),
        reason: str = typer.Option(..., "--reason", "-r", help="Why. Recorded before deletion."),
    ) -> None:
        """Discard a proposal, recording why."""
        if not _proposals().reject(proposal_id, reason=reason):
            console.print(f"[red]no proposal {proposal_id!r}[/]")
            raise typer.Exit(code=1)
        console.print(f"[green]rejected[/] {proposal_id}")


def _active_tools(workspace: Path) -> list[str]:
    """The tool names an agent in this workspace would actually be offered.

    Passing no tools drops every skill that declares ``requires_tools``, which
    is most of the useful ones -- so the cost line reported zero tokens for a
    catalog that is not empty, and reported it most confidently about exactly
    the skills that cost something.
    """
    settings = load_settings(workspace=workspace).settings
    offered = builtin_registry().resolve(enabled_toolsets=settings.enabled_toolsets())
    return [tool.name for tool in offered]


def _project(workspace: Path) -> Path:
    """Where a project's own skills live."""
    return workspace / ".harness" / "skills"


def _library(workspace: Path) -> SkillRegistry:
    """The library as an agent in this workspace would see it."""
    library = default_library(workspace)
    library.refresh()
    return library


def _proposals() -> ProposalStore:
    """The staging area, at the default autonomy.

    ``propose`` regardless of what a session was configured with: this is the
    review surface, and a review command that could auto-apply what it was
    shown would not be one.
    """
    return ProposalStore(
        pending_dir=harness_home() / "skills-pending",
        skills_dir=harness_home() / "skills",
        autonomy=Autonomy.PROPOSE,
        quotas=Quotas(),
    )
