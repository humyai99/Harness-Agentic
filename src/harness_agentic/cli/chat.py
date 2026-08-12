"""The ``harn chat`` and ``harn run`` commands.

Ctrl-C is handled rather than allowed to propagate. A ``KeyboardInterrupt``
through a streaming response leaves the transcript with a half-finished
assistant turn, and the next request is then rejected before it starts. The
handler asks the runner to stop at its next checkpoint, which persists a legal
prefix; a second Ctrl-C within a moment is treated as "I meant it" and exits.
"""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from harness_agentic.agent.build import AgentBundle, build_agent
from harness_agentic.cli.events import ConsoleRenderer
from harness_agentic.cli.render import console, err_console
from harness_agentic.constants import ensure_dirs
from harness_agentic.errors import CredentialError, HarnessError
from harness_agentic.tools.approval import ApprovalPolicy, Mode
from harness_agentic.tools.paths import looks_like_secret

if TYPE_CHECKING:
    from types import FrameType

    from harness_agentic.agent.runner import AgentRunner
    from harness_agentic.tools.spec import ApprovalRequest

DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"
DOUBLE_INTERRUPT_WINDOW_S = 1.5


def _prompter(request: ApprovalRequest) -> bool:
    """Ask the operator to approve one call."""
    console.print(f"\n[yellow]{request.tool}[/] wants to run:", markup=True)
    console.print(f"  {request.summary}", highlight=False)
    return typer.confirm("allow?", default=False)


def _install_interrupt_handler(runner: AgentRunner) -> None:
    """Turn Ctrl-C into a cooperative stop, twice-means-exit."""
    state = {"last": 0.0}

    def handler(_signum: int, _frame: FrameType | None) -> None:
        now = time.monotonic()
        if now - state["last"] < DOUBLE_INTERRUPT_WINDOW_S:
            err_console.print("\n[red]exiting[/]")
            raise KeyboardInterrupt
        state["last"] = now
        err_console.print("\n[yellow]stopping after the current step (Ctrl-C again to exit)[/]")
        runner.interrupt("interrupted by user")

    signal.signal(signal.SIGINT, handler)


def _warn_if_secret(text: str) -> None:
    """Tell the operator when they have pasted something key-shaped.

    Their message is about to be sent to a provider and written to a
    transcript, so the useful advice is to rotate it, not to be quiet.
    """
    if looks_like_secret(text):
        err_console.print(
            "[red]that looks like a credential.[/] It will be sent to the model "
            "provider and stored in the transcript. Rotate it if it is real."
        )


def register(app: typer.Typer) -> None:
    """Attach the chat commands to the CLI."""

    @app.command()
    def run(
        prompt: str = typer.Argument(..., help="What the agent should do."),
        model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
        toolsets: str = typer.Option("file,terminal", "--toolsets"),
        workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w"),
        stream: bool = typer.Option(default=True, help="Stream the answer as it arrives."),
        thinking: bool = typer.Option(default=False, help="Show the model's reasoning."),
        yes: bool = typer.Option(
            default=False, help="Approve every tool call. Use only in a sandbox."
        ),
    ) -> None:
        """Run one turn non-interactively and exit.

        Exit code 0 when the turn completed, 1 otherwise -- so this composes in
        a shell pipeline or a CI step.
        """
        bundle = _build(model, workspace, toolsets, stream, thinking, approve_all=yes)
        try:
            session = bundle.store.latest()
            if session is None:
                err_console.print("[red]could not create a session[/]")
                raise typer.Exit(1)
            _warn_if_secret(prompt)
            _install_interrupt_handler(bundle.runner)
            result = bundle.runner.run_turn(prompt, session=session)
        except HarnessError as exc:
            err_console.print(f"[red]{exc}[/]")
            raise typer.Exit(1) from exc
        console.print()
        raise typer.Exit(0 if result.exit_reason == "completed" else 1)

    @app.command()
    def chat(
        model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
        toolsets: str = typer.Option("file,terminal", "--toolsets"),
        workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w"),
        stream: bool = typer.Option(default=True, help="Stream answers as they arrive."),
        thinking: bool = typer.Option(default=False, help="Show the model's reasoning."),
        yes: bool = typer.Option(
            default=False, help="Approve every tool call. Use only in a sandbox."
        ),
    ) -> None:
        """Start an interactive session. Ctrl-D or /exit to leave."""
        bundle = _build(model, workspace, toolsets, stream, thinking, approve_all=yes)
        session = bundle.store.latest()
        if session is None:
            err_console.print("[red]could not create a session[/]")
            raise typer.Exit(1)

        console.print(f"[dim]{model} | {workspace} | session {session.id}[/]", markup=True)
        console.print("[dim]/exit to quit, Ctrl-C to stop a running turn[/]", markup=True)
        _install_interrupt_handler(bundle.runner)

        while True:
            try:
                line = typer.prompt("\n>", prompt_suffix=" ")
            except (EOFError, KeyboardInterrupt):
                console.print()
                break
            text = line.strip()
            if not text:
                continue
            if text in ("/exit", "/quit"):
                break
            _warn_if_secret(text)
            try:
                bundle.runner.run_turn(text, session=session)
            except KeyboardInterrupt:
                break
            except HarnessError as exc:
                err_console.print(f"[red]{exc}[/]")
            console.print()


def _build(
    model: str,
    workspace: Path,
    toolsets: str,
    stream: bool,
    thinking: bool,
    *,
    approve_all: bool,
) -> AgentBundle:
    """Assemble the agent, reporting a missing credential as advice not a stack."""
    profile = ensure_dirs()
    policy = (
        ApprovalPolicy(surface="cli", modes={"cli": Mode.ALLOW})
        if approve_all
        else ApprovalPolicy(surface="cli", prompter=_prompter)
    )
    if approve_all:
        err_console.print(
            "[yellow]--yes is on: every tool call runs without asking.[/] "
            "Only do this in a sandbox."
        )
    try:
        return build_agent(
            model=model,
            workspace=workspace.resolve(),
            sessions_dir=profile / "sessions",
            toolsets=[t.strip() for t in toolsets.split(",") if t.strip()],
            surface="cli",
            emit=ConsoleRenderer(show_thinking=thinking),
            approval=policy,
            stream=stream,
        )
    except CredentialError as exc:
        err_console.print(f"[red]{exc}[/]")
        sys.exit(1)
