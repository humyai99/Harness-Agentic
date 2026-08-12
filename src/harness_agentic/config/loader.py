"""Finding, merging, and explaining configuration.

TOML rather than YAML, and not only out of taste. ``tomllib`` is in the standard
library, so reading config costs no dependency; TOML has no ``yes``/``no`` to
coerce wrongly, no anchors, no tags, and no way to smuggle behaviour into a data
file. A config format that can execute something is the wrong shape for a file
an agent may one day be asked to edit.

Six layers, lowest precedence first:

1. the schema's own defaults
2. ``$HARNESS_HOME/config.toml``
3. ``$HARNESS_HOME/profiles/<profile>/config.toml``
4. ``<project>/.harness/config.toml``, walking up from the workspace
5. environment variables, ``HARNESS__SECTION__KEY``
6. whatever the caller passes on the command line

Every value remembers which layer set it, because "why is it using that model?"
is the question this module exists to answer and guessing at it costs real time.
``harn config show --origin`` prints the answer.

Lists **replace** rather than concatenate. A project that sets
``tools.toolsets`` means *that list*; if merging appended, there would be no way
to express "only these", and the inherited entries would be invisible in the file
that appears to define them. ``disabled_append`` exists for the additive case, so
the choice is stated rather than inferred.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness_agentic.config.schema import Settings
from harness_agentic.constants import ENV_PREFIX, active_profile, harness_home, profile_dir
from harness_agentic.errors import HarnessError

CONFIG_FILENAME = "config.toml"
PROJECT_DIR = ".harness"
MAX_PARENTS = 5
"""How far up to look for a project config. Bounded so a workspace inside a home
directory does not accidentally inherit from every ancestor."""


class ConfigError(HarnessError):
    """A configuration file could not be read or does not validate."""


@dataclass(frozen=True, slots=True)
class Origin:
    """Where one setting's effective value came from."""

    layer: str
    """``defaults``, ``home``, ``profile``, ``project``, ``env`` or ``cli``."""
    source: str
    """A path, an environment variable name, or ``--flag``."""


@dataclass
class LoadedSettings:
    """Validated settings, plus where each value came from."""

    settings: Settings
    origins: dict[str, Origin] = field(default_factory=dict)
    problems: tuple[str, ...] = ()
    """Files that could not be read. Reported rather than fatal: one unreadable
    project file should not stop the agent, but it must not be silent either."""

    def origin_of(self, dotted: str) -> Origin:
        """Where ``section.key`` was set, defaulting to the schema."""
        return self.origins.get(dotted, Origin(layer="defaults", source="schema"))

    def describe(self) -> list[str]:
        """One line per setting, naming its value and its source."""
        lines: list[str] = []
        for dotted, value in sorted(_flatten(self.settings.model_dump()).items()):
            origin = self.origin_of(dotted)
            lines.append(f"{dotted} = {value!r}  [{origin.layer}: {origin.source}]")
        return lines


def load_settings(
    *,
    workspace: Path | None = None,
    profile: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> LoadedSettings:
    """Assemble the effective settings from every layer.

    ``overrides`` is the CLI layer, expressed in dotted keys
    (``{"model.default": "openai/gpt-5"}``) so a flag and a file name the same
    setting the same way.
    """
    merged: dict[str, Any] = {}
    origins: dict[str, Origin] = {}
    problems: list[str] = []

    for layer, path in _files(workspace=workspace, profile=profile):
        if not path.is_file():
            continue
        try:
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            problems.append(f"{path}: {exc}")
            continue
        _merge(merged, raw, origins=origins, layer=layer, source=str(path))

    env_values, env_sources = _from_environment(environ if environ is not None else os.environ)
    for dotted, value in env_values.items():
        _assign(merged, dotted, value)
        origins[dotted] = Origin(layer="env", source=env_sources[dotted])

    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        _assign(merged, dotted, value)
        origins[dotted] = Origin(layer="cli", source=f"--{dotted.split('.')[-1]}")

    try:
        settings = Settings.model_validate(merged)
    except Exception as exc:
        # Named, with the offending key, because "extra inputs are not permitted"
        # against an unnamed file is the least useful error in configuration.
        detail = f"configuration is not valid: {exc}"
        raise ConfigError(detail) from exc

    return LoadedSettings(settings=settings, origins=origins, problems=tuple(problems))


def _files(*, workspace: Path | None, profile: str | None) -> list[tuple[str, Path]]:
    """Every config file to read, lowest precedence first."""
    layers = [
        ("home", harness_home() / CONFIG_FILENAME),
        ("profile", profile_dir(profile) / CONFIG_FILENAME),
    ]
    project = find_project_config(workspace or Path.cwd())
    if project is not None:
        layers.append(("project", project))
    return layers


def find_project_config(start: Path) -> Path | None:
    """The nearest ``<dir>/.harness/config.toml`` at or above ``start``.

    Stops at a git root, because that is where a project's boundary is in
    practice, and bounded at :data:`MAX_PARENTS` so a workspace nested in a home
    directory does not inherit from unrelated ancestors.
    """
    current = start.resolve()
    for _ in range(MAX_PARENTS + 1):
        candidate = current / PROJECT_DIR / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
        if (current / ".git").exists() or current.parent == current:
            return None
        current = current.parent
    return None


def _merge(
    into: dict[str, Any],
    incoming: Mapping[str, Any],
    *,
    origins: dict[str, Origin],
    layer: str,
    source: str,
    prefix: str = "",
) -> None:
    """Deep-merge one layer, recording provenance per leaf.

    Mappings merge; everything else -- lists included -- replaces. See the module
    docstring for why lists do not concatenate.
    """
    for key, value in incoming.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            nested = into.setdefault(key, {})
            if not isinstance(nested, dict):
                nested = {}
                into[key] = nested
            _merge(nested, value, origins=origins, layer=layer, source=source, prefix=f"{dotted}.")
            continue
        into[key] = value
        origins[dotted] = Origin(layer=layer, source=source)


def _from_environment(environ: Mapping[str, str]) -> tuple[dict[str, Any], dict[str, str]]:
    """Read ``HARNESS__SECTION__KEY`` variables into dotted keys."""
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for name, raw in environ.items():
        if not name.startswith(ENV_PREFIX):
            continue
        path = name.removeprefix(ENV_PREFIX).lower().split("__")
        if not all(path):
            continue
        dotted = ".".join(path)
        values[dotted] = _coerce(raw)
        sources[dotted] = name
    return values, sources


def _coerce(raw: str) -> Any:
    """Turn an environment string into a bool, int, list, or string.

    Environment variables are always strings, and pydantic will coerce most of
    them -- but a comma-separated list has to be split here, because ``"a,b"`` is
    a perfectly valid single string and nothing downstream can tell the
    difference.
    """
    text = raw.strip()
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if "," in text:
        return tuple(part.strip() for part in text.split(",") if part.strip())
    return text


def _assign(into: dict[str, Any], dotted: str, value: Any) -> None:
    """Set a dotted key, creating intermediate mappings."""
    parts = dotted.split(".")
    cursor = into
    for part in parts[:-1]:
        nested = cursor.setdefault(part, {})
        if not isinstance(nested, dict):
            nested = {}
            cursor[part] = nested
        cursor = nested
    cursor[parts[-1]] = value


def _flatten(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested settings into dotted keys, for display."""
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict) and value and all(isinstance(k, str) for k in value):
            flat.update(_flatten(value, prefix=f"{dotted}."))
            continue
        flat[dotted] = value
    return flat


def dotted_keys() -> Sequence[str]:
    """Every settable key, for validating ``harn config set``."""
    return sorted(_flatten(Settings().model_dump()))


def active_profile_name() -> str:
    """The profile these settings were read for."""
    return active_profile()
