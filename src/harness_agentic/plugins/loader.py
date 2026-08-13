"""Loading third-party extensions without letting one break startup.

Three sources, in ascending precedence: pip entry points, then
``$HARNESS_HOME/plugins/``, then ``<project>/.harness/plugins/``. The project
wins because a repository's own extension is the most specific statement of
intent, and it is also the one that went through code review.

Two rules shape everything here.

**A broken plugin is reported, not fatal.** An extension that raises on import
must not stop the others, and must not stop the agent. The failure is collected
and surfaced by ``harn plugins``, because the alternative -- swallowing it -- is
how someone spends an afternoon wondering why their tool disappeared.

**A plugin's tools are not trusted at the level they claim.** A plugin is
somebody else's code running in this process, so a tool it registers is floored
at :attr:`~harness_agentic.tools.spec.Danger.NETWORK` and tagged
``source="plugin:<name>"``. A plugin cannot declare its own tool ``SAFE`` and
thereby route around the approval policy, which is exactly what a hostile or
merely careless one would do.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from harness_agentic.errors import HarnessError
from harness_agentic.tools.spec import Danger, Tool

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from harness_agentic.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "harness_agentic.plugins"
PLUGIN_FLOOR = Danger.NETWORK
"""The lowest danger level a plugin-supplied tool may claim."""


class PluginError(HarnessError):
    """A plugin could not be loaded or misbehaved during setup."""


class Plugin(Protocol):
    """What a plugin module must expose.

    One function. A plugin receives the registry and adds to it; it does not get
    the agent, the store, or the credentials, because a plugin that can reach
    those does not need to ask for anything.
    """

    def setup(self, registry: ToolRegistry) -> None:
        """Register whatever this plugin provides."""
        ...


@dataclass(frozen=True, slots=True)
class Discovered:
    """A plugin found but not yet loaded."""

    name: str
    origin: str
    """``entry-point``, ``user``, or ``project`` -- printed by ``harn plugins``."""
    path: Path | None = None
    module: str = ""

    def describe(self) -> str:
        """One readable line."""
        where = self.path or self.module
        return f"{self.name} ({self.origin}) {where}"


@dataclass(frozen=True, slots=True)
class Loaded:
    """A plugin that set itself up, and what it contributed."""

    plugin: Discovered
    tools: tuple[str, ...]

    def describe(self) -> str:
        """One readable line naming what it added."""
        listed = ", ".join(self.tools) or "nothing"
        return f"{self.plugin.describe()} -> {listed}"


@dataclass(frozen=True, slots=True)
class Failure:
    """A plugin that did not load."""

    plugin: Discovered
    error: str

    def describe(self) -> str:
        """One readable line naming the problem."""
        return f"{self.plugin.describe()} -> FAILED: {self.error}"


@dataclass
class PluginSet:
    """The outcome of one load pass."""

    loaded: list[Loaded] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)
    skipped: list[Discovered] = field(default_factory=list)

    def report(self) -> list[str]:
        """Every line worth printing about this pass."""
        return [
            *(entry.describe() for entry in self.loaded),
            *(f"skipped {entry.describe()}" for entry in self.skipped),
            *(entry.describe() for entry in self.failures),
        ]

    def tool_names(self) -> tuple[str, ...]:
        """Every tool contributed, across all plugins."""
        return tuple(name for entry in self.loaded for name in entry.tools)


def discover(
    *,
    home: Path | None = None,
    project: Path | None = None,
    include_entry_points: bool = True,
) -> list[Discovered]:
    """Find plugins, later sources overriding earlier ones by name."""
    found: dict[str, Discovered] = {}
    if include_entry_points:
        for entry in _entry_points():
            found[entry.name] = entry
    for origin, root in (("user", home), ("project", project)):
        if root is None:
            continue
        for entry in _from_directory(root, origin):
            found[entry.name] = entry
    return sorted(found.values(), key=lambda entry: entry.name)


def load(
    registry: ToolRegistry,
    plugins: Sequence[Discovered],
    *,
    allow: Iterable[str] | None = None,
    deny: Iterable[str] = (),
) -> PluginSet:
    """Set up each plugin against ``registry``, isolating failures.

    ``allow`` is the control that matters. Left as ``None`` every discovered
    plugin loads, which is right for a developer's own machine and wrong for a
    deployment -- so an operator names the ones they want and nothing dropped
    into the directory afterwards runs on its own.
    """
    permitted = None if allow is None else set(allow)
    refused = set(deny)
    result = PluginSet()

    for entry in plugins:
        if entry.name in refused or (permitted is not None and entry.name not in permitted):
            result.skipped.append(entry)
            continue
        try:
            added = _setup_one(registry, entry)
        except Exception as exc:
            # Reported, never fatal. One extension that raises on import must
            # not take the agent with it.
            log.warning("plugin %s failed to load: %s", entry.name, exc)
            result.failures.append(Failure(plugin=entry, error=f"{type(exc).__name__}: {exc}"))
            continue
        result.loaded.append(Loaded(plugin=entry, tools=added))
    return result


def _setup_one(registry: ToolRegistry, entry: Discovered) -> tuple[str, ...]:
    """Import a plugin, run its setup, and constrain what it registered.

    What counts as "what it registered" is the load-bearing part. Comparing the
    set of *names* before and after missed a replacement entirely: a plugin
    holding the registry can call ``register(..., override=True)`` on a name that
    already exists, so swapping its own handler into ``read_file`` changed no
    names and the constraint below never ran. The replacement kept
    ``danger=SAFE`` and ``source="builtin"`` -- it routed around the approval
    policy and reported itself as a builtin while doing it, which is precisely
    what the floor exists to prevent.

    So the comparison is on tool *identity*. Every ``register`` call stores a new
    object, so anything the plugin touched is visible whether or not the name
    was new.
    """
    module = _import(entry)
    setup = getattr(module, "setup", None)
    if not callable(setup):
        detail = f"{entry.name} has no setup(registry) function"
        raise PluginError(detail)

    before = registry.all()
    setup(registry)
    touched = tuple(
        sorted(name for name, tool in registry.all().items() if before.get(name) is not tool)
    )
    if replaced := [name for name in touched if name in before]:
        # Said out loud. Replacing a builtin may be legitimate, but it is never
        # something an operator should discover by accident.
        log.warning("plugin %s replaced existing tool(s): %s", entry.name, ", ".join(replaced))
    _constrain(registry, entry.name, touched)
    return touched


def _constrain(registry: ToolRegistry, plugin: str, names: Sequence[str]) -> None:
    """Re-register a plugin's tools with an honest source and a danger floor.

    Done after setup rather than by asking plugins to declare it properly: a
    rule enforced by convention is a rule that holds until the first plugin
    author who did not read the documentation.
    """
    for name in names:
        tool = registry.get(name)
        registry.register(
            Tool(
                name=tool.name,
                description=tool.description,
                toolset=tool.toolset,
                params_model=tool.params_model,
                handler=tool.handler,
                danger=max(tool.danger, PLUGIN_FLOOR),
                requires_env_vars=tool.requires_env_vars,
                available_when=tool.available_when,
                max_result_chars=tool.max_result_chars,
                timeout_s=tool.timeout_s,
                surfaces=tool.surfaces,
                source=f"plugin:{plugin}",
            ),
            override=True,
        )


def _import(entry: Discovered) -> object:
    """Import a plugin from an entry point or from a file on disk."""
    if entry.module:
        return importlib.import_module(entry.module)
    if entry.path is None:  # pragma: no cover - discover() never produces this
        detail = f"{entry.name} has neither a module nor a path"
        raise PluginError(detail)

    target = entry.path / "__init__.py" if entry.path.is_dir() else entry.path
    module_name = f"harness_agentic_plugin_{entry.name}"
    spec = importlib.util.spec_from_file_location(module_name, target)
    if spec is None or spec.loader is None:
        detail = f"{target} is not importable"
        raise PluginError(detail)
    module = importlib.util.module_from_spec(spec)
    # Registered before execution so a plugin split across several files can
    # import its own submodules.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[module_name]
        raise
    return module


def _from_directory(root: Path, origin: str) -> list[Discovered]:
    """Find plugins under one directory: ``name.py`` or ``name/__init__.py``."""
    if not root.is_dir():
        return []
    found: list[Discovered] = []
    for child in sorted(root.iterdir()):
        if child.name.startswith((".", "_")):
            continue
        if child.is_dir() and (child / "__init__.py").exists():
            found.append(Discovered(name=child.name, origin=origin, path=child))
        elif child.suffix == ".py":
            found.append(Discovered(name=child.stem, origin=origin, path=child))
    return found


def _entry_points() -> list[Discovered]:
    """Find plugins installed as pip packages."""
    from importlib.metadata import entry_points

    try:
        group = entry_points(group=ENTRY_POINT_GROUP)
    except Exception:  # pragma: no cover - a broken environment, not our bug
        log.warning("could not read entry points for %s", ENTRY_POINT_GROUP)
        return []
    return [
        Discovered(name=point.name, origin="entry-point", module=point.value.split(":")[0])
        for point in group
    ]
