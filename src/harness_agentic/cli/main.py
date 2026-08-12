"""The ``harn`` entry point.

Subcommands are added milestone by milestone. Everything printed here goes
through :mod:`harness_agentic.cli.render` so the gateway can reuse the same
formatting -- ``ruff``'s ``T20`` rule keeps stray ``print`` calls out.
"""

from __future__ import annotations

import typer

from harness_agentic.cli import chat as chat_commands
from harness_agentic.cli.render import console
from harness_agentic.constants import active_profile, harness_home
from harness_agentic.version import __version__

app = typer.Typer(
    name="harn",
    help="Harness-Agentic: a self-improving agent framework.",
    no_args_is_help=True,
    add_completion=True,
)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"harness-agentic {__version__}")
        raise typer.Exit


@app.callback()
def _root(
    _version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Print the version and exit.",
    ),
) -> None:
    """Harness-Agentic command line."""


@app.command()
def doctor() -> None:
    """Report where state lives and which profile is active.

    Grows into the full environment check (credential sources, docker
    reachability, database integrity) as those subsystems land. It never prints
    a secret value -- only where one was resolved from.
    """
    console.print(f"[bold]harness-agentic[/bold] {__version__}")
    console.print(f"home    {harness_home()}")
    console.print(f"profile {active_profile()}")


chat_commands.register(app)


def main() -> None:
    """Console-script entry point."""
    app()
