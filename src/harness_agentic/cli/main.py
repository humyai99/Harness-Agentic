"""The ``harn`` entry point.

Subcommands are added milestone by milestone. Everything printed here goes
through :mod:`harness_agentic.cli.render` so the gateway can reuse the same
formatting -- ``ruff``'s ``T20`` rule keeps stray ``print`` calls out.
"""

from __future__ import annotations

import typer

from harness_agentic.cli import chat as chat_commands
from harness_agentic.cli import config as config_commands
from harness_agentic.cli import extend as extend_commands
from harness_agentic.cli import gateway as gateway_commands
from harness_agentic.cli import inspect as inspect_commands
from harness_agentic.cli import kb as kb_commands
from harness_agentic.cli import memory as memory_commands
from harness_agentic.cli import skills as skills_commands
from harness_agentic.cli import surfaces as surface_commands
from harness_agentic.cli.render import console
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


chat_commands.register(app)
config_commands.register(app)
inspect_commands.register(app)
inspect_commands.register_doctor(app)
gateway_commands.register(app)
extend_commands.register(app)
skills_commands.register(app)
memory_commands.register(app)
kb_commands.register(app)
surface_commands.register(app)


def main() -> None:
    """Console-script entry point."""
    app()
