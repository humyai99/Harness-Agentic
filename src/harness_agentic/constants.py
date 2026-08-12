"""Filesystem locations and process-wide constants.

Every path the framework touches is derived here. Nothing else in the codebase
may hardcode ``~/.harness`` -- profile isolation and the test suite's
``isolated_home`` fixture both depend on that rule holding.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

APP_NAME: Final = "harness"
"""Directory name used under the user's home."""

ENV_HOME: Final = "HARNESS_HOME"
ENV_PROFILE: Final = "HARNESS_PROFILE"
ENV_PREFIX: Final = "HARNESS__"
"""Prefix for settings overrides. A double underscore separates nesting."""

DEFAULT_PROFILE: Final = "default"

PLUGIN_ENTRY_POINT_GROUP: Final = "harness_agentic.plugins"


def harness_home() -> Path:
    """Return the root state directory, honouring ``HARNESS_HOME``.

    The directory is not created as a side effect of asking for it; call
    :func:`ensure_dirs` when you actually intend to write.
    """
    override = os.environ.get(ENV_HOME)
    if override:
        return Path(override).expanduser()
    return Path.home() / f".{APP_NAME}"


def active_profile() -> str:
    """Return the active profile name, honouring ``HARNESS_PROFILE``."""
    return os.environ.get(ENV_PROFILE) or DEFAULT_PROFILE


def profile_dir(profile: str | None = None) -> Path:
    """Return the directory owning one profile's config, state, and logs."""
    return harness_home() / "profiles" / (profile or active_profile())


def ensure_dirs(profile: str | None = None) -> Path:
    """Create the profile directory tree if missing and return the profile dir.

    Directories are created with mode ``0o700``: they hold ``.env`` files,
    session transcripts, and agent-authored skills.
    """
    home = harness_home()
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    pdir = profile_dir(profile)
    for sub in ("", "skills", "logs", "traces", "cache"):
        (pdir / sub if sub else pdir).mkdir(mode=0o700, parents=True, exist_ok=True)
    return pdir
