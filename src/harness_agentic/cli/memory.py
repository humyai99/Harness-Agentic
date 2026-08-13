"""``harn memory`` -- reading and pruning what the agent carries between sessions.

Memory is the one piece of state that is in the prompt on every turn of every
future session, which makes it the one piece an operator should be able to read
in full without starting a conversation. It is also, being facts about a person
and their work, the state most worth being able to delete.

``show`` prints both files with how much room is left, because the number is the
part that drives a decision: a file at 90% means the next useful fact will be
refused, and the fix is to prune now rather than to discover it mid-task.

``forget`` exists because the agent's own ``memory_remove`` requires a session
and a model. Removing something wrong should not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer
from rich.table import Table

from harness_agentic.cli.render import console
from harness_agentic.config import load_settings
from harness_agentic.constants import memories_dir
from harness_agentic.memory.manager import MemoryStore

if TYPE_CHECKING:
    from harness_agentic.config import Settings

FULL_ENOUGH_TO_MENTION = 0.8


def memory_store(settings: Settings | None = None) -> MemoryStore:
    """The store the agent in this profile would use.

    One function so ``harn memory`` and a running agent cannot end up on
    different directories, each reporting the other's file as empty.
    """
    loaded = settings if settings is not None else load_settings().settings
    return MemoryStore(
        root=memories_dir(),
        memory_limit=loaded.memory.memory_limit,
        user_limit=loaded.memory.user_limit,
    )


def register(app: typer.Typer) -> None:
    """Attach the ``memory`` command group."""
    group = typer.Typer(
        help="Read and prune what the agent remembers between sessions.",
        no_args_is_help=True,
    )
    app.add_typer(group, name="memory")

    @group.command("show")
    def show(
        kind: str = typer.Argument("", help="'memory', 'user', or omit for both."),
    ) -> None:
        """Print what is remembered, and how much room is left."""
        store = memory_store()
        kinds = [kind] if kind else list(store.KINDS)
        try:
            states = [(name, store.load(name)) for name in kinds]
        except ValueError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=1) from exc

        for name, state in states:
            console.print(f"[bold]{state.path.name}[/] [dim]{state.path}[/]")
            if state.entries:
                console.print(state.render().strip())
            else:
                console.print("[dim](nothing recorded)[/]")
            console.print(f"[dim]{state.used}/{state.limit} characters[/]")
            if state.used > state.limit * FULL_ENOUGH_TO_MENTION:
                console.print(
                    f"[yellow]nearly full:[/] the next fact the agent tries to record "
                    f"will likely be refused. `harn memory forget {name} <text>` to "
                    f"make room."
                )
            console.print()

    @group.command("add")
    def add(
        kind: str = typer.Argument(..., help="'memory' for the work, 'user' for the person."),
        text: str = typer.Argument(..., help="One fact, stated so it is useful months from now."),
        section: str = typer.Option("", "--section", "-s", help="Heading to group it under."),
    ) -> None:
        """Record a fact yourself, without going through the agent."""
        store = memory_store()
        try:
            entry = store.add(kind, text, section=section)
        except ValueError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=1) from exc

        console.print(f"[green]recorded[/] {entry.text}")
        console.print(
            f"[dim]{store.load(kind).remaining} characters left. It joins the prompt "
            f"for sessions started from now on.[/]"
        )

    @group.command("forget")
    def forget(
        kind: str = typer.Argument(..., help="'memory' or 'user'."),
        text: str = typer.Argument(..., help="Text identifying the entry to delete."),
    ) -> None:
        """Delete one remembered fact."""
        store = memory_store()
        try:
            removed = store.remove(kind, text)
        except ValueError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=1) from exc
        if not removed:
            console.print(f"[red]no entry in {kind} matching[/] {text!r}")
            raise typer.Exit(code=1)
        console.print("[green]forgotten[/]")

    @group.command("usage")
    def usage() -> None:
        """Show what memory costs, per file.

        Worth a look before raising a limit: this is spent on every turn of every
        session, so it is the one number that compounds.
        """
        store = memory_store()
        table = Table("file", "used", "limit", "full")
        total = 0
        for name, used, limit in store.usage():
            total += used
            table.add_row(name, str(used), str(limit), f"{used / limit:.0%}")
        console.print(table)
        console.print(f"[dim]about {total // 4:,} tokens in every prompt of every session[/]")
