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
from harness_agentic.cli.memory import memory_store
from harness_agentic.cli.render import console, err_console
from harness_agentic.config import ConfigError, Settings, load_settings
from harness_agentic.constants import ensure_dirs
from harness_agentic.errors import CredentialError, HarnessError
from harness_agentic.mcp.config import McpConfigError, configured_servers
from harness_agentic.skills.proposals import autonomy_from, default_store
from harness_agentic.tools.approval import ApprovalPolicy, Mode
from harness_agentic.tools.paths import looks_like_secret

if TYPE_CHECKING:
    from harness_agentic.envs.base import ExecEnvironment
    from harness_agentic.mcp.stdio import ServerConfig
    from harness_agentic.skills.proposals import ProposalStore

if TYPE_CHECKING:
    from types import FrameType

    from harness_agentic.agent.runner import AgentRunner
    from harness_agentic.tools.spec import ApprovalRequest

DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"
DOUBLE_INTERRUPT_WINDOW_S = 1.5


def _prompter(request: ApprovalRequest) -> bool:
    """Ask the operator to approve one call, if there is one to ask.

    Without a terminal there is nobody to answer, and asking anyway is worse
    than useless: ``typer.confirm`` blocks on stdin, so ``harn run`` in a
    pipeline or a CI step -- the use its own docstring recommends -- hangs until
    something kills it. Refusing immediately turns that into a tool error the
    loop reports and the model can respond to, which is what a refusal is
    supposed to be.
    """
    if not sys.stdin.isatty():
        console.print(
            f"[yellow]{request.tool}[/] needs approval and there is no terminal to ask. "
            f"Pass --yes in a sandbox, or add a pattern to tools.approval.allowlist.",
            markup=True,
        )
        return False
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
        model: str = typer.Option("", "--model", "-m", help="Overrides model.default."),
        toolsets: str = typer.Option("", "--toolsets", help="Overrides tools.toolsets."),
        workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w"),
        env: str = typer.Option(
            "", "--env", help="Where tools run: 'local' or 'docker'. Overrides tools.env."
        ),
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
        bundle, renderer = _build(
            model, workspace, toolsets, stream, thinking, approve_all=yes, backend=env
        )
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
        if not renderer.wrote_answer and result.final_text.strip():
            # Nothing streamed it. With --no-stream there are no text chunks at
            # all, so this command printed a usage line and no answer -- in the
            # mode its own docstring recommends for a pipeline or a CI step.
            console.print(result.final_text, markup=False, highlight=False)
        console.print()
        raise typer.Exit(0 if result.exit_reason == "completed" else 1)

    @app.command()
    def chat(
        model: str = typer.Option("", "--model", "-m", help="Overrides model.default."),
        toolsets: str = typer.Option("", "--toolsets", help="Overrides tools.toolsets."),
        workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w"),
        env: str = typer.Option(
            "", "--env", help="Where tools run: 'local' or 'docker'. Overrides tools.env."
        ),
        stream: bool = typer.Option(default=True, help="Stream answers as they arrive."),
        thinking: bool = typer.Option(default=False, help="Show the model's reasoning."),
        yes: bool = typer.Option(
            default=False, help="Approve every tool call. Use only in a sandbox."
        ),
    ) -> None:
        """Start an interactive session. Ctrl-D or /exit to leave."""
        bundle, renderer = _build(
            model, workspace, toolsets, stream, thinking, approve_all=yes, backend=env
        )
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
            renderer.wrote_answer = False
            try:
                turn = bundle.runner.run_turn(text, session=session)
            except KeyboardInterrupt:
                break
            except HarnessError as exc:
                err_console.print(f"[red]{exc}[/]")
            else:
                if not renderer.wrote_answer and turn.final_text.strip():
                    console.print(turn.final_text, markup=False, highlight=False)
            console.print()


def _build(
    model: str,
    workspace: Path,
    toolsets: str,
    stream: bool,
    thinking: bool,
    *,
    approve_all: bool,
    backend: str = "",
) -> tuple[AgentBundle, ConsoleRenderer]:
    """Assemble the agent, reporting a missing credential as advice not a stack.

    Flags override configuration rather than replacing it. An empty flag means
    "not given", which is why the defaults are empty strings and not the schema's
    values -- otherwise a config file could never change a default, because the
    flag would always be sitting on top of it saying the same thing.
    """
    profile = ensure_dirs()
    try:
        loaded = load_settings(
            workspace=workspace.resolve(),
            overrides={
                "model.default": model or None,
                "tools.toolsets": _split(toolsets) or None,
                "tools.env": backend or None,
            },
        )
    except ConfigError as exc:
        err_console.print(f"[red]{exc}[/]")
        sys.exit(1)
    for problem in loaded.problems:
        err_console.print(f"[yellow]unreadable config:[/] {problem}")
    settings = loaded.settings
    environment = _environment(settings.tools.env, workspace.resolve(), settings)
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
    renderer = ConsoleRenderer(show_thinking=thinking)
    try:
        bundle = build_agent(
            model=settings.model.default,
            workspace=workspace.resolve(),
            sessions_dir=profile / "sessions",
            toolsets=settings.enabled_toolsets(),
            fallbacks=settings.model.fallbacks,
            surface="cli",
            emit=renderer,
            approval=policy,
            stream=stream and settings.model.stream,
            env=environment,
            memory=memory_store(settings),
            mcp_servers=_mcp_servers(),
            proposal_store=_proposal_store(settings),
            max_iterations=settings.model.max_iterations,
        )
    except CredentialError as exc:
        err_console.print(f"[red]{exc}[/]")
        sys.exit(1)
    return bundle, renderer


def _proposal_store(settings: Settings) -> ProposalStore | None:
    """Where the agent stages a skill it wants to propose.

    Nothing supplied this, so ``skill_propose`` was never registered and the
    agent had no way to offer a skill at all -- while ``harn skills pending``
    stood ready to review a queue that nothing could add to. The autonomy
    setting decides whether an approval still needs a person; ``propose``, the
    default, means it does.
    """
    if not settings.skills.enabled:
        return None
    return default_store(autonomy=autonomy_from(settings.skills.autonomy))


def _mcp_servers() -> list[ServerConfig]:
    """The MCP servers an agent started here should connect to.

    Nothing passed these before, so `harn mcp check` could start a server and
    list its tools while every agent ran without them -- the tools existed and
    were unreachable. A broken entry is reported and skipped rather than
    stopping the session: one bad server should not cost the operator their
    agent, and `harn mcp check` is where the details belong.
    """
    try:
        return configured_servers()
    except McpConfigError as exc:
        err_console.print(f"[yellow]ignoring MCP configuration:[/] {exc}")
        return []


def _split(raw: str) -> list[str]:
    """A comma-separated flag as a list, or empty when the flag was not given."""
    return [part.strip() for part in raw.split(",") if part.strip()]


def _environment(backend: str, workspace: Path, settings: Settings) -> ExecEnvironment | None:
    """Build the execution environment named by ``--env``.

    ``None`` means "let build_agent use the local one", which keeps the default
    path free of any Docker import at all -- the module shells out and should not
    be loaded for a run that will never use it.

    A missing daemon is reported here rather than on the first tool call, because
    "the sandbox you asked for is not available" is something to learn before the
    agent has started doing work you believed was contained.
    """
    if backend == "local":
        return None
    if backend != "docker":
        err_console.print(f"[red]unknown --env {backend!r}[/]; use 'local' or 'docker'")
        sys.exit(1)

    from harness_agentic.envs.docker import (
        DockerEnvironment,
        DockerLimits,
        DockerUnavailable,
        probe,
    )

    try:
        version = probe()
    except DockerUnavailable as exc:
        err_console.print(f"[red]{exc}[/]")
        err_console.print(
            "[dim]Run with --env local to use this host instead, knowing that "
            "`terminal` is then unsandboxed.[/]"
        )
        sys.exit(1)
    sandbox = DockerEnvironment(
        workspace=workspace,
        image=settings.docker.image,
        network=settings.docker.network,
        read_only_root=settings.docker.read_only_root,
        limits=DockerLimits(
            memory=settings.docker.memory,
            cpus=settings.docker.cpus,
            pids=settings.docker.pids,
        ),
    )
    err_console.print(f"[green]docker {version}[/]: {sandbox.describe()}")
    return sandbox
