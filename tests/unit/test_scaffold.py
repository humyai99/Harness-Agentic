"""M0 checks: the package imports cheaply and paths respect the environment."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from harness_agentic import __version__
from harness_agentic.cli.main import app
from harness_agentic.constants import (
    active_profile,
    ensure_dirs,
    harness_home,
    profile_dir,
)


def test_version_is_importable() -> None:
    assert __version__
    assert __version__.count(".") >= 2


def test_home_follows_the_environment(isolated_home: Path) -> None:
    assert harness_home() == isolated_home
    assert active_profile() == "default"
    assert profile_dir() == isolated_home / "profiles" / "default"


def test_ensure_dirs_creates_a_private_tree(isolated_home: Path) -> None:
    pdir = ensure_dirs()
    for sub in ("skills", "logs", "traces", "cache"):
        assert (pdir / sub).is_dir()
    if sys.platform != "win32":
        # These directories hold .env files, transcripts, and agent-authored
        # skills; group- and world-readable would be a real leak.
        assert pdir.stat().st_mode & 0o077 == 0


def test_cli_reports_its_version() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_cli_doctor_reports_the_active_home(isolated_home: Path) -> None:
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "profile" in result.stdout


@pytest.mark.parametrize("banned", ["httpx", "typer", "rich"])
def test_importing_the_package_stays_cheap(banned: str) -> None:
    """``import harness_agentic`` must not drag in HTTP or CLI machinery.

    Startup cost compounds: the gateway imports this on every worker, and a
    cheap root import is what lets provider SDKs and platform adapters stay
    lazily loaded.
    """
    code = f"import harness_agentic, sys; sys.exit(1 if {banned!r} in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_builtin_registry_actually_contains_the_builtins() -> None:
    """The bug this guards: ``install_builtins(ToolRegistry())`` returns nothing.

    The tools register into the process-wide registry at import time and that
    call only adds the *toolsets*, so passing a fresh registry produces something
    that looks right and offers no tools -- which is how three CLI commands came
    to build empty registries.
    """
    from harness_agentic.tools.builtin import builtin_registry, install_builtins
    from harness_agentic.tools.registry import ToolRegistry

    populated = builtin_registry()
    assert "read_file" in populated.all()
    assert "file" in populated.toolsets()

    # The misuse, demonstrated so nobody reintroduces it.
    assert not install_builtins(ToolRegistry()).all()


def test_every_builtin_is_offered_on_every_surface_by_default() -> None:
    """A surface's restrictions belong to the surface, not to each tool.

    Missing this made the web UI and the voice surface offer zero tools.
    """
    from harness_agentic.tools.builtin import builtin_registry

    registry = builtin_registry()
    for surface in ("cli", "gateway", "cron", "web", "voice"):
        offered = registry.resolve(enabled_toolsets=["core", "file"], surface=surface)
        assert offered, f"no tools reachable on the {surface} surface"


def test_harn_sessions_reads_the_store_the_agent_writes(
    isolated_home: Path, tmp_path: Path
) -> None:
    """It read a JsonlSessionStore; every agent has written SQLite for milestones.

    So the one command whose whole job is listing sessions found none -- and
    said so in exactly the words it uses when there genuinely are none, which is
    why it could look like working software. Written against the real store
    rather than a mock, because a mock of the wrong store is what the bug was.
    """
    from harness_agentic.constants import profile_dir
    from harness_agentic.session.sqlite_store import SqliteSessionStore

    workspace = tmp_path / "ws"
    workspace.mkdir()
    store = SqliteSessionStore(profile_dir() / "sessions" / "state.db")
    record = store.create(source="cli", cwd=workspace, model="fake/scripted")

    result = CliRunner().invoke(app, ["sessions", "--workspace", str(workspace)])

    assert result.exit_code == 0
    assert record.id in result.stdout
