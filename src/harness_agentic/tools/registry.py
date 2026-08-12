"""The tool registry.

Builtin tools are registered by explicit import in ``tools/builtin/__init__.py``
rather than by scanning the filesystem. Scanning is how a tool silently
disappears because its module raised on import, and it hides every tool from
mypy. Autodiscovery is kept for *plugins*, where it is genuinely needed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from harness_agentic.errors import ToolNotFound
from harness_agentic.tools.spec import Danger, Tool, ToolContext, ToolResult, Toolset

if TYPE_CHECKING:
    from harness_agentic.core.types import ToolSchema

P = TypeVar("P", bound=BaseModel)


class ToolRegistry:
    """Holds every known tool and resolves which ones apply to a given run."""

    def __init__(self) -> None:
        """Start empty."""
        self._tools: dict[str, Tool] = {}
        self._toolsets: dict[str, Toolset] = {}

    # -- registration -------------------------------------------------------

    def register(self, tool: Tool, *, override: bool = False) -> None:
        """Add a tool, refusing to shadow an existing name unless told to."""
        if tool.name in self._tools and not override:
            msg = f"tool {tool.name!r} is already registered by {self._tools[tool.name].source!r}"
            raise ValueError(msg)
        self._tools[tool.name] = tool
        self._toolsets.setdefault(
            tool.toolset, Toolset(name=tool.toolset, description=tool.toolset)
        )

    def define_toolset(self, toolset: Toolset) -> None:
        """Declare or replace a toolset's description and membership."""
        self._toolsets[toolset.name] = toolset

    def tool(
        self,
        *,
        toolset: str,
        danger: Danger = Danger.SAFE,
        name: str | None = None,
        description: str | None = None,
        requires_env_vars: Sequence[str] = (),
        available_when: Callable[[], bool] | None = None,
        max_result_chars: int = 40_000,
        timeout_s: float = 120.0,
        surfaces: Sequence[str] = ("cli", "gateway", "cron"),
        source: str = "builtin",
        override: bool = False,
    ) -> Callable[[Callable[[P, ToolContext], ToolResult]], Callable[[P, ToolContext], ToolResult]]:
        """Register the decorated function as a tool.

        The parameter model is read from the first argument's annotation and
        the description from the docstring, so a tool definition stays one
        function rather than a function plus a schema plus a registration call
        that can fall out of step with each other.

        ``override`` is for tools closed over a live dependency -- a store, an
        HTTP client -- which are installed once per agent rather than once per
        process. The gateway builds an agent per conversation, so refusing the
        second registration would take the second conversation with it.
        """

        def decorate(
            fn: Callable[[P, ToolContext], ToolResult],
        ) -> Callable[[P, ToolContext], ToolResult]:
            params_model = _first_param_model(fn)
            doc = (fn.__doc__ or "").strip()
            if not (description or doc):
                msg = f"tool {fn.__name__!r} needs a docstring or an explicit description"
                raise ValueError(msg)
            self.register(
                Tool(
                    name=name or fn.__name__,
                    description=description or doc,
                    toolset=toolset,
                    params_model=params_model,
                    handler=fn,  # type: ignore[arg-type]
                    danger=danger,
                    requires_env_vars=tuple(requires_env_vars),
                    available_when=available_when,
                    max_result_chars=max_result_chars,
                    timeout_s=timeout_s,
                    surfaces=frozenset(surfaces),
                    source=source,
                ),
                override=override,
            )
            return fn

        return decorate

    # -- lookup -------------------------------------------------------------

    def get(self, name: str) -> Tool:
        """Return one tool by name."""
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFound(name) from None

    def maybe_get(self, name: str) -> Tool | None:
        """Return one tool by name, or ``None``."""
        return self._tools.get(name)

    def all(self) -> Mapping[str, Tool]:
        """Every registered tool, keyed by name."""
        return dict(self._tools)

    def toolsets(self) -> Mapping[str, Toolset]:
        """Every known toolset."""
        return dict(self._toolsets)

    def resolve(
        self,
        *,
        enabled_toolsets: Sequence[str] | None = None,
        disabled: Sequence[str] = (),
        surface: str = "cli",
        max_danger: Danger = Danger.DESTRUCTIVE,
    ) -> tuple[Tool, ...]:
        """Decide which tools this run may use.

        The single place that answer is computed. Everything downstream -- the
        prompt builder listing tools, the transport serializing schemas,
        dispatch checking a call is legal -- reads from this one list, so they
        cannot disagree about what was on offer.
        """
        wanted = set(self._expand(enabled_toolsets)) if enabled_toolsets is not None else None
        blocked = set(disabled)
        chosen = [
            tool
            for tool in self._tools.values()
            if tool.name not in blocked
            and (wanted is None or tool.toolset in wanted)
            and surface in tool.surfaces
            and tool.danger <= max_danger
            and tool.is_available()
        ]
        return tuple(sorted(chosen, key=lambda t: t.name))

    def schemas(self, tools: Sequence[Tool]) -> tuple[ToolSchema, ...]:
        """Render wire schemas for a resolved tool list."""
        return tuple(tool.schema() for tool in tools)

    def _expand(self, names: Sequence[str]) -> set[str]:
        """Resolve toolset ``includes`` transitively."""
        seen: set[str] = set()
        stack = list(names)
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            if group := self._toolsets.get(current):
                stack.extend(group.includes)
        return seen

    def unregister(self, name: str) -> bool:
        """Remove a tool. Returns whether it was there.

        Needed because a tool's provider can go away while the process runs: an
        MCP server that exits leaves a tool that fails in a way looking like the
        tool is broken, and the agent will keep retrying it.
        """
        return self._tools.pop(name, None) is not None

    def fork(self) -> ToolRegistry:
        """A copy that shares nothing mutable with this one.

        Import-time registration lands in one process-wide registry, which is
        fine for tools that are the same everywhere. Tools closed over a live
        dependency are not: two conversations in one gateway can be configured
        differently, and the second must not inherit whichever fetcher or store
        the first happened to be built with. So every agent gets a fork.
        """
        copy = ToolRegistry()
        copy._tools = dict(self._tools)
        copy._toolsets = dict(self._toolsets)
        return copy

    # -- testing ------------------------------------------------------------

    @contextmanager
    def isolated(self) -> Iterator[ToolRegistry]:
        """Snapshot and restore, so a test's registrations do not leak.

        Import-time registration into a module-level registry is convenient and
        is also a testing hazard; this makes it tractable rather than pretending
        the hazard is not there.
        """
        tools, sets_ = dict(self._tools), dict(self._toolsets)
        try:
            yield self
        finally:
            self._tools, self._toolsets = tools, sets_


def _first_param_model(fn: Callable[..., object]) -> type[BaseModel]:
    """Read the pydantic model from a handler's first parameter annotation."""
    import inspect  # noqa: PLC0415  -- paid once per tool at import time
    import typing  # noqa: PLC0415

    signature = inspect.signature(fn)
    params = list(signature.parameters.values())
    if not params:
        msg = f"tool {fn.__name__!r} must take (params, ctx)"
        raise ValueError(msg)
    try:
        hints = typing.get_type_hints(fn)
    except NameError as exc:
        # With PEP 563 the annotation is a string resolved against module
        # globals, so a model defined inside a function is invisible here.
        msg = (
            f"tool {fn.__name__!r} annotates its parameters with a type that is "
            f"not resolvable at module level ({exc}); define the parameter model "
            f"at module scope"
        )
        raise TypeError(msg) from exc
    annotation = hints.get(params[0].name)
    if not (isinstance(annotation, type) and issubclass(annotation, BaseModel)):
        msg = (
            f"tool {fn.__name__!r} must annotate its first parameter with a "
            f"pydantic BaseModel subclass; got {annotation!r}"
        )
        raise TypeError(msg)
    return annotation


registry = ToolRegistry()
"""The process-wide registry builtin tools register into."""
