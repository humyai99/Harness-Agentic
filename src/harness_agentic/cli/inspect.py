"""``harn models``, ``harn tools``, ``harn sessions``, and a real ``harn doctor``.

``doctor`` is the command people run when something is wrong, so it answers the
questions that are actually being asked: where is state, which credentials
resolved, and *from where*. It never prints a value -- only a source. A
diagnostic that leaks the thing it is diagnosing is worse than no diagnostic.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from harness_agentic.cli.render import console
from harness_agentic.constants import active_profile, harness_home, profile_dir
from harness_agentic.errors import CredentialError, HarnessError
from harness_agentic.providers.catalog import PROVIDERS, known_models, model_info
from harness_agentic.providers.credentials import SecretResolver, resolve_credentials
from harness_agentic.session.store import JsonlSessionStore, workspace_key_for
from harness_agentic.tools.builtin import install_builtins
from harness_agentic.tools.registry import registry
from harness_agentic.version import __version__

_DANGER_LABEL = {0: "safe", 1: "network", 2: "writes", 3: "destructive"}


def register(app: typer.Typer) -> None:
    """Attach the inspection commands."""

    @app.command()
    def models() -> None:
        """List providers, whether they are configured, and known models."""
        resolver = SecretResolver()
        table = Table("provider", "api mode", "credential", "base url")
        for name, info in sorted(PROVIDERS.items()):
            if name == "fake":
                continue
            try:
                credentials = resolve_credentials(name, resolver)
                status = f"[green]{credentials.source}[/]"
            except CredentialError:
                wanted = " or ".join(info.api_key_env) or "-"
                status = f"[dim]missing ({wanted})[/]"
            table.add_row(name, info.api_mode, status, info.base_url or "-")
        console.print(table)

        console.print("\n[bold]catalogued models[/]")
        for reference in known_models():
            if reference.startswith("fake/"):
                continue
            provider, model = reference.split("/", 1)
            facts = model_info(provider, model)
            console.print(
                f"  {reference}  [dim]{facts.context_window:,} ctx  "
                f"{facts.max_output_tokens:,} out[/]"
            )
        console.print(
            "\n[dim]An uncatalogued model still works; it gets conservative "
            "defaults. Self-hosted deployments name models freely.[/]"
        )

    @app.command()
    def tools(
        toolsets: str = typer.Option("", "--toolsets", help="Comma-separated filter."),
        surface: str = typer.Option("cli", "--surface"),
    ) -> None:
        """List the tools an agent would be offered, and their danger level."""
        install_builtins(registry)
        wanted = [t.strip() for t in toolsets.split(",") if t.strip()] or None
        resolved = registry.resolve(enabled_toolsets=wanted, surface=surface)
        table = Table("tool", "toolset", "danger", "description")
        for tool in resolved:
            table.add_row(
                tool.name,
                tool.toolset,
                _DANGER_LABEL.get(int(tool.danger), str(tool.danger)),
                tool.description.split("\n")[0][:70],
            )
        console.print(table)
        console.print(f"[dim]{len(resolved)} tool(s) on the {surface} surface[/]")

    @app.command()
    def sessions(limit: int = typer.Option(20, "--limit", "-n")) -> None:
        """List recent sessions in this workspace."""
        store = JsonlSessionStore(profile_dir() / "sessions")
        key = workspace_key_for(Path.cwd())
        records = store.recent(limit=limit, workspace_key=key)
        if not records:
            console.print("[dim]no sessions for this workspace yet[/]")
            return
        table = Table("id", "updated", "source", "model", "tokens")
        for record in records:
            table.add_row(
                record.id,
                record.updated_at.strftime("%Y-%m-%d %H:%M"),
                record.source,
                record.model,
                f"{record.total_usage.total:,}",
            )
        console.print(table)


def register_doctor(app: typer.Typer) -> None:
    """Replace the placeholder ``doctor`` with the real one."""

    @app.command()
    def doctor() -> None:
        """Report configuration, credentials, and state locations.

        Prints where each credential was resolved from, never its value.
        """
        console.print(f"[bold]harness-agentic[/bold] {__version__}")
        console.print(f"home     {harness_home()}")
        console.print(f"profile  {active_profile()}  ({profile_dir()})")

        console.print("\n[bold]credentials[/]")
        resolver = SecretResolver()
        found = False
        for name, info in sorted(PROVIDERS.items()):
            if name == "fake" or not info.api_key_env:
                continue
            try:
                credentials = resolve_credentials(name, resolver)
            except HarnessError as exc:
                console.print(f"  [red]{name}[/]: {exc}")
                continue
            if credentials.api_key:
                found = True
                console.print(f"  [green]{name}[/]: resolved from {credentials.source}")
        if not found:
            console.print(
                f"  [yellow]none resolved.[/] Put keys in {harness_home() / '.env'} "
                f"and chmod 600 it, or export them."
            )

        console.print("\n[bold]tools[/]")
        install_builtins(registry)
        for name, group in sorted(registry.toolsets().items()):
            count = len(registry.resolve(enabled_toolsets=[name]))
            console.print(f"  {name}: {count} tool(s)  [dim]{group.description}[/]")
