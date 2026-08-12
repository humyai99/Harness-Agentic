"""``harn web`` and ``harn voice``.

Both commands build an agent the same way ``harn chat`` does and then attach a
different subscriber to its event stream. That is the visible payoff of having
written ``core/events.py`` in the first milestone: neither of these needed the
loop changed.

``harn web token`` exists separately from ``harn web serve`` because a token
should be minted once and stored, not regenerated on every start -- a token that
changes when the process restarts is a token nobody can bookmark past, so people
turn auth off instead.
"""

from __future__ import annotations

import os
from pathlib import Path

import typer

from harness_agentic.agent.build import AgentBundle, build_agent
from harness_agentic.api.server import DEFAULT_HOST, DEFAULT_PORT, Api, new_token, serve
from harness_agentic.cli.render import console
from harness_agentic.constants import harness_home
from harness_agentic.tools.approval import ApprovalPolicy, Mode
from harness_agentic.voice.session import VoiceSession, voice_surfaces

TOKEN_ENV = "HARNESS_WEB_TOKEN"  # noqa: S105 - a variable name, not a secret
TOKEN_FILE = "web-token"  # noqa: S105 - a filename, not a secret


def register(app: typer.Typer) -> None:
    """Attach the ``web`` and ``voice`` command groups."""
    _register_web(app)
    _register_voice(app)


def _register_web(app: typer.Typer) -> None:
    group = typer.Typer(help="A browser UI over the agent's event stream.", no_args_is_help=True)
    app.add_typer(group, name="web")

    @group.command("token")
    def token(
        rotate: bool = typer.Option(False, "--rotate", help="Replace the stored token."),
    ) -> None:
        """Show the API token, minting one on first use.

        Stored at mode 0600 rather than printed and forgotten. A token that
        changes on every restart is one nobody can keep, so people disable auth
        instead -- which is the outcome this command exists to prevent.
        """
        path = harness_home() / TOKEN_FILE
        if rotate or not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(new_token(), encoding="utf-8")
            path.chmod(0o600)
            console.print("[green]minted a new token[/]" if rotate else "[green]minted a token[/]")
        console.print(path.read_text(encoding="utf-8").strip())
        console.print(f"[dim]stored at {path} (mode 0600)[/]")

    @group.command("serve")
    def serve_web(
        model: str = typer.Option("anthropic/claude-sonnet-5", "--model", "-m"),
        toolsets: str = typer.Option("file", "--toolsets"),
        workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w"),
        host: str = typer.Option(DEFAULT_HOST, "--host"),
        port: int = typer.Option(DEFAULT_PORT, "--port"),
        thinking: bool = typer.Option(False, "--thinking/--no-thinking"),
    ) -> None:
        """Serve the browser UI.

        Approvals are answered from the browser through the same
        :class:`~harness_agentic.tools.approval.ApprovalPolicy` the terminal
        uses, so a destructive command still has to be approved -- it is just a
        dialog rather than a prompt.
        """
        secret = _read_token()
        if not secret:
            console.print("[red]no token[/]: run `harn web token` first")
            raise typer.Exit(code=1)

        bundles: dict[str, AgentBundle] = {}

        def not_yet(session: str, prompt: str) -> None:
            """Replaced below, once `bundle_for` exists to close over."""
            del session, prompt

        api = Api(token=secret, run_turn=not_yet, include_thinking=thinking)

        def bundle_for(session: str) -> AgentBundle:
            if session not in bundles:
                bundles[session] = build_agent(
                    model=model,
                    workspace=workspace,
                    sessions_dir=harness_home() / "web" / session,
                    toolsets=[part.strip() for part in toolsets.split(",") if part.strip()],
                    surface="web",
                    emit=api.sink(session),
                    approval=ApprovalPolicy(
                        surface="web", modes={"web": Mode.PROMPT}, prompter=api.prompter(session)
                    ),
                )
            return bundles[session]

        def run(session: str, prompt: str) -> None:
            bundle = bundle_for(session)
            record = bundle.store.get(bundle.context.session_id)
            if record is None:  # pragma: no cover - created moments earlier
                detail = "the session vanished"
                raise RuntimeError(detail)
            bundle.runner.run_turn(prompt, session=record)

        api.run_turn = run
        console.print(f"[green]web UI[/] on http://{host}:{port}")
        console.print("[dim]paste the token from `harn web token` into the page[/]")
        serve(api, host=host, port=port)


def _register_voice(app: typer.Typer) -> None:
    group = typer.Typer(help="Speak to the agent.", no_args_is_help=True)
    app.add_typer(group, name="voice")

    @group.command("check")
    def check_voice(toolsets: str = typer.Option("file", "--toolsets")) -> None:
        """Report what a voice session would and would not be able to do.

        Worth its own command because the surprising part of voice is which
        tools are *withheld*: reading a file listing aloud is not a degraded
        experience, it is an unusable one.
        """
        from harness_agentic.tools.builtin import builtin_registry
        from harness_agentic.voice.session import VOICE_UNFRIENDLY

        registry = builtin_registry()
        wanted = ["core", *[part.strip() for part in toolsets.split(",") if part.strip()]]
        offered = [tool.name for tool in registry.resolve(enabled_toolsets=wanted, surface="voice")]
        spoken = voice_surfaces(offered)

        console.print(f"[green]spoken:[/] {', '.join(spoken) or 'nothing'}")
        withheld = sorted(set(offered) - set(spoken))
        if withheld:
            console.print(f"[yellow]withheld (no spoken form):[/] {', '.join(withheld)}")
        console.print(f"[dim]always withheld: {', '.join(sorted(VOICE_UNFRIENDLY))}[/]")

    @group.command("demo")
    def demo(
        text: str = typer.Argument(..., help="What to pretend was said."),
        model: str = typer.Option("anthropic/claude-sonnet-5", "--model", "-m"),
        workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w"),
    ) -> None:
        """Run one turn and print what would have been spoken.

        No microphone and no synthesizer: this shows the *text* a voice surface
        would produce, which is the part worth checking before wiring audio to
        it. STT and TTS are deployment choices, and a Thai deployment needs
        models tested against Thai speech rather than whatever is default.
        """
        spoken: list[str] = []

        class Printer:
            """A speaker that writes instead of speaking."""

            speaking = False

            def speak(self, utterance: object) -> None:
                spoken.append(getattr(utterance, "text", ""))

            def stop(self) -> None:
                return None

        class Silent:
            """A listener that hears nothing; the text comes from the argument."""

            def listen(self) -> object:
                return iter(())

            def stop(self) -> None:
                return None

        def ignore(_prompt: str) -> None:
            """A voice demo runs one turn; it does not start another."""

        session = VoiceSession(
            speaker=Printer(),
            listener=Silent(),  # type: ignore[arg-type]
            submit=ignore,
        )
        bundle = build_agent(
            model=model,
            workspace=workspace,
            sessions_dir=harness_home() / "voice",
            toolsets=["core", "file"],
            surface="voice",
            emit=session.sink(),
        )
        record = bundle.store.get(bundle.context.session_id)
        if record is None:  # pragma: no cover - created moments earlier
            console.print("[red]the session vanished[/]")
            raise typer.Exit(code=1)
        bundle.runner.run_turn(text, session=record)

        for line in spoken:
            console.print(f"[cyan]🔊[/] {line}")
        if not spoken:
            console.print("[dim]nothing would have been spoken[/]")


def _read_token() -> str:
    """The API token, from the environment or the stored file."""
    from_env = os.environ.get(TOKEN_ENV, "").strip()
    if from_env:
        return from_env
    path = harness_home() / TOKEN_FILE
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""
