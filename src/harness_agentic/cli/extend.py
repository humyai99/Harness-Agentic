"""``harn plugins`` and ``harn cron`` -- inspecting what extends the agent.

Both commands exist because both subsystems fail *quietly* by nature. A plugin
that raised on import is a tool that is simply not there, and a cron job with a
typo in its expression is a job that never runs. Neither announces itself, so
each gets a command that says plainly what loaded, what did not, and why.

``cron list`` prints the next firing for every job rather than the expression
alone. An expression is a claim about intent; the next firing is what will
actually happen, and the gap between them is where the day-of-week surprise
lives.
"""

from __future__ import annotations

import tomllib
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.markup import escape
from rich.table import Table

from harness_agentic.agent.build import build_agent
from harness_agentic.cli.events import ConsoleRenderer
from harness_agentic.cli.render import console
from harness_agentic.constants import harness_home
from harness_agentic.cron.runner import CronRunner, Job, load_jobs
from harness_agentic.cron.schedule import BadSchedule, parse
from harness_agentic.mcp.bridge import McpBridge
from harness_agentic.mcp.stdio import ServerConfig, load_servers
from harness_agentic.plugins.loader import discover, load
from harness_agentic.tools.builtin import install_builtins
from harness_agentic.tools.registry import ToolRegistry

JOBS_FILENAME = "cron.toml"


def register(app: typer.Typer) -> None:
    """Attach the ``plugins``, ``cron`` and ``mcp`` command groups."""
    _register_plugins(app)
    _register_cron(app)
    register_mcp(app)


def _register_plugins(app: typer.Typer) -> None:
    group = typer.Typer(help="Inspect third-party extensions.", no_args_is_help=True)
    app.add_typer(group, name="plugins")

    @group.command("list")
    def list_plugins(
        project: Path = typer.Option(Path.cwd(), "--project", "-p"),
        entry_points: bool = typer.Option(True, "--entry-points/--no-entry-points"),
    ) -> None:
        """Show every plugin that would be discovered, without loading any."""
        found = discover(
            home=harness_home() / "plugins",
            project=project / ".harness" / "plugins",
            include_entry_points=entry_points,
        )
        if not found:
            console.print("No plugins found.")
            console.print(
                f"[dim]Looked in {harness_home() / 'plugins'}, "
                f"{project / '.harness' / 'plugins'}, and pip entry points.[/]"
            )
            return
        table = Table("plugin", "origin", "location")
        for entry in found:
            table.add_row(entry.name, entry.origin, str(entry.path or entry.module))
        console.print(table)

    @group.command("check")
    def check_plugins(
        project: Path = typer.Option(Path.cwd(), "--project", "-p"),
        allow: str = typer.Option("", "--allow", help="Comma-separated names to load."),
    ) -> None:
        """Load every plugin into a throwaway registry and report the result.

        A throwaway registry on purpose: this is a diagnostic, and a diagnostic
        that mutates the thing it is diagnosing is a diagnostic nobody trusts.
        """
        registry = install_builtins(ToolRegistry())
        found = discover(home=harness_home() / "plugins", project=project / ".harness" / "plugins")
        result = load(
            registry,
            found,
            allow=[name.strip() for name in allow.split(",") if name.strip()] or None,
        )
        for line in result.report():
            style = "red" if "FAILED" in line else "dim" if "skipped" in line else "green"
            console.print(f"[{style}]{line}[/]")
        if result.failures:
            raise typer.Exit(code=1)
        if not result.report():
            console.print("No plugins found.")


def _register_cron(app: typer.Typer) -> None:
    group = typer.Typer(help="Scheduled, unattended runs.", no_args_is_help=True)
    app.add_typer(group, name="cron")

    @group.command("list")
    def list_jobs(
        config: Path = typer.Option(None, "--config", "-c", help="Path to cron.toml."),
        count: int = typer.Option(10, "--count", "-n"),
    ) -> None:
        """Show every job and when it will next actually fire."""
        jobs = _load(config)
        if not jobs:
            console.print(f"No jobs configured. Create {_config_path(config)}.")
            return
        runner = CronRunner(jobs=jobs)
        upcoming = dict(runner.next_runs(limit=len(jobs) * 2))

        table = Table("job", "schedule", "next run (UTC)", "toolsets", "approval")
        for job in jobs:
            when = upcoming.get(job.name)
            table.add_row(
                job.name if job.enabled else f"[dim]{job.name} (disabled)[/]",
                job.schedule.expression,
                f"{when:%Y-%m-%d %H:%M}" if when else "[red]never[/]",
                ", ".join(job.toolsets),
                job.approval.value,
            )
        console.print(table)
        for job in jobs:
            if job.schedule.day_restricted and job.schedule.weekday_restricted:
                console.print(f"[yellow]note:[/] {job.name} -- {job.schedule.describe()}")
        console.print(f"[dim]Showing the next firing for each of {len(jobs)} job(s).[/]")
        del count

    @group.command("check")
    def check_expression(expression: str = typer.Argument(..., help="A cron expression.")) -> None:
        """Explain an expression, and show its next few firings."""
        try:
            schedule = parse(expression)
        except BadSchedule as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=1) from exc

        console.print(schedule.describe())
        cursor = datetime.now(UTC)
        for _ in range(5):
            following = schedule.next_after(cursor)
            if following is None:
                console.print("[red]no further firings[/]")
                break
            console.print(f"  {following:%Y-%m-%d %H:%M} UTC ({following:%A})")
            cursor = following

    @group.command("run")
    def run_job(
        name: str = typer.Argument(..., help="Which job to run."),
        config: Path = typer.Option(None, "--config", "-c"),
        model: str = typer.Option("anthropic/claude-sonnet-5", "--model", "-m"),
        workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w"),
    ) -> None:
        """Run one job immediately, ignoring its schedule.

        For testing a job before trusting it to run unattended. It runs under
        the job's real approval policy, so a job that will be refused at 3am is
        refused here too rather than appearing to work.
        """
        jobs = {job.name: job for job in _load(config)}
        job = jobs.get(name)
        if job is None:
            console.print(f"[red]no job named {name!r}[/]; known: {', '.join(sorted(jobs))}")
            raise typer.Exit(code=1)

        console.print(
            f"[yellow]Running {job.name} for real[/] under approval mode "
            f"{job.approval.value}. Nothing is simulated: tools the policy "
            f"permits will execute."
        )
        runner = CronRunner(
            jobs=[job],
            bundle_factory=lambda due: build_agent(
                model=model,
                workspace=workspace,
                sessions_dir=harness_home() / "cron" / due.name,
                toolsets=list(due.toolsets),
                surface="cron",
                emit=ConsoleRenderer(),
                # The job's own policy, not a relaxed one. A job that will be
                # refused at 3am has to be refused here too, or this command
                # tells the operator something that is not true.
                approval=due.policy(),
                max_iterations=due.max_iterations,
            ),
        )
        record = runner.run(job)
        console.print(f"\n[dim]{record.describe()}[/]")
        if record.outcome != "ok":
            raise typer.Exit(code=1)


def _config_path(explicit: Path | None) -> Path:
    """Where cron jobs are read from."""
    return explicit or (harness_home() / JOBS_FILENAME)


def _load(explicit: Path | None) -> list[Job]:
    """Read jobs from TOML, refusing anything malformed."""
    path = _config_path(explicit)
    if not path.exists():
        return []
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("jobs") or raw.get("job") or []
    if not isinstance(entries, list):
        console.print(f"[red]{path} should contain a list of {escape('[[jobs]]')} tables[/]")
        raise typer.Exit(code=1)
    try:
        return load_jobs(entries)
    except (ValueError, BadSchedule) as exc:
        # Refused rather than skipped: a job that silently never runs is the one
        # failure a scheduler must not have.
        console.print(f"[red]{path}: {exc}[/]")
        raise typer.Exit(code=1) from exc


def register_mcp(app: typer.Typer) -> None:
    """Attach the ``mcp`` command group."""
    group = typer.Typer(help="Model Context Protocol servers.", no_args_is_help=True)
    app.add_typer(group, name="mcp")

    @group.command("list")
    def list_servers(config: Path = typer.Option(None, "--config", "-c")) -> None:
        """Show the configured servers, without starting any."""
        servers = _servers(config)
        if not servers:
            console.print(
                f"No MCP servers configured. Add {escape('[[mcp.servers]]')} "
                f"to {_mcp_path(config)}."
            )
            return
        table = Table("server", "command", "env passthrough", "enabled")
        for server in servers:
            table.add_row(
                server.name,
                " ".join(server.command),
                ", ".join(server.env_passthrough) or "-",
                "yes" if server.enabled else "no",
            )
        console.print(table)

    @group.command("check")
    def check_servers(config: Path = typer.Option(None, "--config", "-c")) -> None:
        """Start each server, list its tools, and stop it again.

        The diagnostic that matters before trusting a server: it reports what
        each one actually offers and at what danger level, rather than what its
        documentation claims.
        """
        servers = _servers(config)
        if not servers:
            console.print("No MCP servers configured.")
            return
        registry = install_builtins(ToolRegistry())
        bridge = McpBridge(registry=registry)
        try:
            bridge.connect_all(servers)
            for line in bridge.report():
                console.print(f"[{'red' if 'FAILED' in line else 'green'}]{line}[/]")
            if bridge.tool_names():
                table = Table("tool", "danger", "server")
                for name in bridge.tool_names():
                    tool = registry.get(name)
                    table.add_row(name, tool.danger.name.lower(), tool.source)
                console.print(table)
            for bridged in bridge.servers.values():
                if diagnostics := bridged.server.diagnostics():
                    console.print(f"[dim]{bridged.name} stderr:\n{diagnostics}[/]")
        finally:
            # Every server is a child process; leaving one running is a leak the
            # operator finds later with `ps`.
            bridge.close()
        if bridge.failures:
            raise typer.Exit(code=1)


def _mcp_path(explicit: Path | None) -> Path:
    """Where MCP server configuration is read from."""
    return explicit or (harness_home() / "mcp.toml")


def _servers(explicit: Path | None) -> list[ServerConfig]:
    """Read server configs from TOML, refusing anything malformed."""
    path = _mcp_path(explicit)
    if not path.exists():
        return []
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    section = raw.get("mcp") or raw
    entries = section.get("servers") or []
    if not isinstance(entries, list):
        console.print(f"[red]{path} should contain a list of {escape('[[mcp.servers]]')} tables[/]")
        raise typer.Exit(code=1)
    try:
        return load_servers(entries)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{path}: {exc}[/]")
        raise typer.Exit(code=1) from exc
