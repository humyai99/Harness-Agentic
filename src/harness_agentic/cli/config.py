"""``harn config`` and ``harn auth``.

Two commands and one rule between them: **settings go in ``config.toml``, secrets
go in ``.env``**, and ``config set`` refuses to blur that. A key that looks like a
credential is redirected to ``auth set`` rather than written, because a config
file is the thing people commit, paste into an issue, and share on a call --
and a token in one is a token that has already leaked by the time anyone notices.

``config show --origin`` is the command worth having. "Why is it using that
model?" has six possible answers -- a default, one of three files, an environment
variable, a flag -- and answering it by reading files in precedence order is
minutes of work every time.

``auth set`` reads from a hidden prompt. It never takes the value as an argument,
because an argument is in the shell history, in ``ps`` output while it runs, and
in any terminal recording.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import tomli_w
import typer
from rich.syntax import Syntax
from rich.table import Table

from harness_agentic.cli.render import console, err_console
from harness_agentic.config import (
    CONFIG_FILENAME,
    ConfigError,
    dotted_keys,
    find_project_config,
    load_settings,
)
from harness_agentic.constants import (
    active_profile,
    ensure_dirs,
    harness_home,
    profile_dir,
)
from harness_agentic.core.secrets import looks_like_credential
from harness_agentic.providers.catalog import PROVIDERS
from harness_agentic.providers.credentials import SecretResolver

SECRET_HINTS = ("key", "token", "secret", "password", "credential")
"""Key fragments that mean "this belongs in .env". Matched on the *last* segment
so ``gateway.platforms`` is not mistaken for one."""


def register(app: typer.Typer) -> None:
    """Attach the ``config`` and ``auth`` command groups."""
    _register_config(app)
    _register_auth(app)


def _register_config(app: typer.Typer) -> None:
    group = typer.Typer(help="Read and write settings.", no_args_is_help=True)
    app.add_typer(group, name="config")

    @group.command("show")
    def show(
        origin: bool = typer.Option(False, "--origin", help="Say which layer set each value."),
        workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w"),
    ) -> None:
        """Print the effective settings.

        With ``--origin``, every value names the layer and the file that set it.
        That is the whole reason this exists: the answer to "why is it using that
        model" is otherwise six files deep.
        """
        loaded = load_settings(workspace=workspace)
        for problem in loaded.problems:
            err_console.print(f"[yellow]unreadable:[/] {problem}")

        if origin:
            table = Table("setting", "value", "layer", "source")
            for line in loaded.describe():
                setting, _, rest = line.partition(" = ")
                value, _, tail = rest.rpartition("  [")
                layer, _, source = tail.rstrip("]").partition(": ")
                table.add_row(setting, value, layer, _shorten(source))
            console.print(table)
        else:
            # TOML has no null, so an unset optional is omitted rather than
            # rendered -- which is also what a config file should look like: the
            # absence of a key *is* how "unset" is spelled.
            # As syntax rather than through `console.print`: rich reads `[model]`
            # as a style tag and silently eats every section header, which makes
            # the output look like one flat namespace.
            rendered = tomli_w.dumps(_without_nulls(loaded.settings.model_dump(mode="json")))
            console.print(Syntax(rendered, "toml", theme="ansi_dark"))

        console.print(f"[dim]profile: {active_profile()}[/]")
        project = find_project_config(workspace)
        if project is not None:
            console.print(f"[dim]project config: {project}[/]")

    @group.command("path")
    def paths(workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w")) -> None:
        """Show every file settings are read from, in precedence order."""
        table = Table("layer", "path", "exists")
        rows = [
            ("home", harness_home() / CONFIG_FILENAME),
            ("profile", profile_dir() / CONFIG_FILENAME),
        ]
        project = find_project_config(workspace)
        if project is not None:
            rows.append(("project", project))
        for layer, path in rows:
            table.add_row(layer, str(path), "yes" if path.is_file() else "no")
        console.print(table)
        console.print("[dim]Later layers win. Environment (HARNESS__…) beats all files.[/]")

    @group.command("set")
    def set_value(
        key: str = typer.Argument(..., help="A dotted key, e.g. model.default."),
        value: str = typer.Argument(..., help="The value. Lists are comma-separated."),
        scope: str = typer.Option(
            "profile", "--scope", help="Where to write: 'home', 'profile', or 'project'."
        ),
        workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w"),
    ) -> None:
        """Write one setting, refusing to put a credential in a config file."""
        if _is_secretish(key) or looks_like_credential(value):
            err_console.print(
                f"[red]{key} looks like a credential.[/] Config files get committed, "
                f"pasted into issues, and shared on calls. Use `harn auth set "
                f"<PROVIDER>` -- it writes to .env at mode 0600 and prompts without "
                f"echoing."
            )
            raise typer.Exit(code=1)

        known = dotted_keys()
        if key not in known:
            err_console.print(f"[red]unknown setting {key!r}[/]")
            near = [name for name in known if key.rsplit(".", maxsplit=1)[-1] in name]
            if near:
                err_console.print(f"[dim]did you mean: {', '.join(near[:5])}[/]")
            raise typer.Exit(code=1)

        target = _target_for(scope, workspace)
        raw = tomllib.loads(target.read_text(encoding="utf-8")) if target.is_file() else {}
        _assign(raw, key, _parse(value))

        # Validated before it is written: a file that does not load is worse than
        # a rejected command, because the next run is the one that finds out.
        try:
            load_settings(workspace=workspace, overrides=_dotted(raw))
        except ConfigError as exc:
            err_console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=1) from exc

        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        target.write_text(tomli_w.dumps(raw), encoding="utf-8")
        console.print(f"[green]{key}[/] = {_parse(value)!r} in {target}")


def _register_auth(app: typer.Typer) -> None:
    group = typer.Typer(help="Store provider credentials.", no_args_is_help=True)
    app.add_typer(group, name="auth")

    @group.command("set")
    def set_credential(
        provider: str = typer.Argument(..., help="A provider name, e.g. anthropic."),
        name: str = typer.Option("", "--name", help="Override the variable name."),
    ) -> None:
        """Store a provider credential in the profile's ``.env``.

        Prompts without echoing, and never accepts the value as an argument: an
        argument is in the shell history, in ``ps`` while it runs, and in any
        recording of the terminal.
        """
        variable = name or _variable_for(provider)
        if not variable:
            err_console.print(f"[red]unknown provider {provider!r}[/]")
            console.print(f"[dim]known: {', '.join(sorted(_provider_names()))}[/]")
            raise typer.Exit(code=1)

        secret = typer.prompt(f"{variable}", hide_input=True).strip()
        if not secret:
            err_console.print("[red]nothing entered[/]")
            raise typer.Exit(code=1)
        if not looks_like_credential(secret) and len(secret) < 16:  # noqa: PLR2004
            # A warning, not a refusal: a self-hosted endpoint may take anything,
            # and refusing would be wrong for it.
            err_console.print(
                "[yellow]that does not look like a provider key.[/] Storing it anyway."
            )

        path = _write_env(variable, secret)
        console.print(f"[green]stored[/] {variable} in {path} (mode 0600)")
        console.print("[dim]Rotate it if it has ever been in a shell history or a chat.[/]")

    @group.command("list")
    def list_credentials() -> None:
        """Show which providers are configured. Never any values."""
        resolver = SecretResolver()
        table = Table("provider", "variable", "status")
        for name, info in sorted(PROVIDERS.items()):
            if not info.requires_key:
                table.add_row(name, "-", "[dim]no key needed[/]")
                continue
            found = next((var for var in info.api_key_env if resolver.get(var) is not None), "")
            secret = resolver.get(found) if found else None
            table.add_row(
                name,
                " or ".join(info.api_key_env) or "-",
                # The *source* is safe to print and is the useful part: "which of
                # my three .env files did that come from" is the real question.
                f"[green]set[/] ({secret.source})" if secret else "[dim]missing[/]",
            )
        console.print(table)
        console.print(
            "[dim]Values are never printed. Use `harn auth set <provider>` to store one.[/]"
        )


def _provider_names() -> list[str]:
    """Every provider the catalog knows."""
    return sorted(PROVIDERS)


def _variable_for(provider: str) -> str:
    """The environment variable a provider's credential goes in."""
    info = PROVIDERS.get(provider.lower())
    return info.api_key_env[0] if info and info.api_key_env else ""


def _write_env(variable: str, secret: str) -> Path:
    """Upsert one variable into the profile's ``.env``, at mode 0600.

    Rewritten line by line rather than appended, so setting a key twice does not
    leave the old value further up the file -- where it is both confusing and
    still a leaked secret.
    """
    path = ensure_dirs() / ".env"
    lines: list[str] = []
    replaced = False
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(f"{variable}="):
                lines.append(f"{variable}={secret}")
                replaced = True
                continue
            lines.append(line)
    if not replaced:
        lines.append(f"{variable}={secret}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def _target_for(scope: str, workspace: Path) -> Path:
    """The file a ``config set`` should write to."""
    if scope == "home":
        return harness_home() / CONFIG_FILENAME
    if scope == "profile":
        return profile_dir() / CONFIG_FILENAME
    if scope == "project":
        return workspace / ".harness" / CONFIG_FILENAME
    err_console.print(f"[red]unknown scope {scope!r}[/]; use home, profile, or project")
    raise typer.Exit(code=1)


def _is_secretish(key: str) -> bool:
    """Whether a key's *last* segment names a credential."""
    last = key.rsplit(".", maxsplit=1)[-1].lower()
    return any(hint in last for hint in SECRET_HINTS)


def _parse(value: str) -> object:
    """Interpret a command-line value as a bool, int, list, or string."""
    lowered = value.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if value.strip().lstrip("-").isdigit():
        return int(value)
    if "," in value:
        return [part.strip() for part in value.split(",") if part.strip()]
    return value


def _assign(into: dict[str, object], dotted: str, value: object) -> None:
    """Set a dotted key in a nested mapping."""
    parts = dotted.split(".")
    cursor = into
    for part in parts[:-1]:
        nested = cursor.setdefault(part, {})
        if not isinstance(nested, dict):
            nested = {}
            cursor[part] = nested
        cursor = nested
    cursor[parts[-1]] = value


def _dotted(payload: dict[str, object], prefix: str = "") -> dict[str, object]:
    """Flatten a nested mapping to dotted keys."""
    flat: dict[str, object] = {}
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_dotted(value, prefix=f"{name}."))
            continue
        flat[name] = value
    return flat


def _without_nulls(payload: dict[str, object]) -> dict[str, object]:
    """Drop keys whose value is ``None``, recursively.

    TOML cannot represent null, and in a config file the absence of a key is
    exactly how "unset" is spelled -- so omitting is both necessary and correct.
    """
    cleaned: dict[str, object] = {}
    for key, value in payload.items():
        if value is None:
            continue
        cleaned[key] = _without_nulls(value) if isinstance(value, dict) else value
    return cleaned


def _shorten(path: str) -> str:
    """Abbreviate a path against the home directory, for a readable table."""
    home = str(Path.home())
    return path.replace(home, "~") if path.startswith(home) else path
