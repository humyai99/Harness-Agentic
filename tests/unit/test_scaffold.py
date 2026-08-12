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
