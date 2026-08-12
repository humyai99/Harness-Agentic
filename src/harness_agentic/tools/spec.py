"""What a tool is.

Tools declare their arguments as a pydantic model rather than hand-written JSON
Schema. The schema is then generated, which means the validation the handler
relies on and the schema the model is shown can never drift apart -- and a
handler receives a typed object instead of an untyped dict.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, Protocol

from harness_agentic.core.types import ImageBlock, TextBlock, ToolSchema

if TYPE_CHECKING:
    from pydantic import BaseModel

    from harness_agentic.core.cancel import CancelToken
    from harness_agentic.envs.base import ExecEnvironment


class Danger(IntEnum):
    """How much damage a tool can do, which drives the approval policy.

    Ordered so policies can be written as thresholds rather than as sets.
    """

    SAFE = 0
    """Read-only, no network."""
    NETWORK = 1
    """Reaches out; can leak what it is given."""
    WRITES = 2
    """Mutates the workspace."""
    DESTRUCTIVE = 3
    """Deletes, mutates outside the workspace, installs, or runs a shell."""


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The outcome of one tool call.

    ``text`` is what the model sees and ``display`` is what the human sees;
    they differ more often than expected. A file write should tell the model
    "wrote 42 lines to src/app.py" and show the operator a diff.
    """

    text: str
    is_error: bool = False
    display: str | None = None
    images: tuple[ImageBlock, ...] = ()
    truncated: bool = False
    data: Mapping[str, object] | None = None
    """Structured detail for hooks and traces. Never sent to the model."""
    tainted: bool = False
    """This result carried content from outside the trust boundary.

    Set by anything that reads the network or a third-party file. It propagates
    to the session, and a tainted session's skill proposals require human
    review whatever the autonomy setting says -- because a poisoned memory
    ruins one session and a poisoned skill ruins every future one whose
    description matches. See :mod:`harness_agentic.skills.reflection`.
    """

    @classmethod
    def error(cls, message: str) -> ToolResult:
        """Build a failure result.

        Tool failures are results, not exceptions: a model that is told what
        went wrong usually fixes it on the next turn, whereas an exception ends
        the turn and loses the work.
        """
        return cls(text=message, is_error=True)

    def blocks(self) -> tuple[TextBlock | ImageBlock, ...]:
        """Render as content blocks for a tool-result message."""
        return (TextBlock(self.text), *self.images)


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """A request for permission to run something consequential."""

    tool: str
    danger: Danger
    summary: str
    detail: str | None = None


class ToolContext(Protocol):
    """Everything a handler is allowed to reach.

    Notably absent: the filesystem and the shell. Those arrive via ``env``, and
    a pre-commit hook keeps direct access out of ``tools/``.
    """

    session_id: str
    workspace_root: Path
    cwd: PurePath
    env: ExecEnvironment
    cancel: CancelToken

    def emit(self, message: str) -> None:
        """Report progress to whatever surface is watching."""
        ...

    def approve(self, request: ApprovalRequest) -> bool:
        """Ask the active policy whether this may proceed."""
        ...


class _Handler(Protocol):
    def __call__(self, params: Any, ctx: ToolContext) -> ToolResult: ...


@dataclass(frozen=True, slots=True)
class Tool:
    """A registered, callable tool."""

    name: str
    description: str
    toolset: str
    params_model: type[BaseModel] | None
    """``None`` for a tool whose schema comes from elsewhere -- an MCP server
    publishes JSON Schema already, and round-tripping it through a generated
    pydantic model would lose the parts pydantic cannot express."""
    handler: _Handler
    danger: Danger = Danger.SAFE
    requires_env_vars: tuple[str, ...] = ()
    available_when: Callable[[], bool] | None = None
    max_result_chars: int = 40_000
    timeout_s: float = 120.0
    surfaces: frozenset[str] = frozenset({"cli", "gateway", "cron"})
    source: str = "builtin"
    raw_schema: Mapping[str, Any] | None = None
    """A JSON Schema supplied directly, used when ``params_model`` is ``None``."""

    def __post_init__(self) -> None:
        """Refuse a tool with no way to describe its arguments."""
        if self.params_model is None and self.raw_schema is None:
            detail = f"tool {self.name!r} needs either a params_model or a raw_schema"
            raise ValueError(detail)

    def schema(self) -> ToolSchema:
        """Render the wire-facing schema the model is shown."""
        if self.params_model is None:
            assert self.raw_schema is not None  # noqa: S101 - __post_init__ guarantees it
            return ToolSchema(
                name=self.name,
                description=self.description,
                parameters=_inline_refs(dict(self.raw_schema), self.raw_schema.get("$defs", {})),
            )
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=_json_schema(self.params_model),
        )

    def validate_arguments(self, arguments: Mapping[str, Any]) -> Any:
        """Coerce arguments for the handler, raising ``ValidationError``.

        A schema-only tool gets the raw mapping: the schema belongs to a remote
        server, and re-validating against a locally reconstructed model would
        reject arguments the server would have accepted.
        """
        if self.params_model is None:
            return dict(arguments)
        return self.params_model.model_validate(dict(arguments))

    def is_available(self) -> bool:
        """Whether this tool should be offered at all right now."""
        import os  # noqa: PLC0415  -- checked per resolve, not per import

        if any(var not in os.environ for var in self.requires_env_vars):
            return False
        return self.available_when() if self.available_when else True


def _json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Produce a JSON Schema providers will accept.

    pydantic emits ``$defs`` and ``$ref`` for nested models, which several
    provider endpoints reject outright. Inlining them here keeps tool authors
    free to use nested models without having to know that.
    """
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})
    if defs:
        schema = _inline_refs(schema, defs)
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        if isinstance(prop, dict):
            prop.pop("title", None)
    return schema


def _inline_refs(node: Any, defs: Mapping[str, Any]) -> Any:
    """Recursively replace ``$ref`` pointers with their definitions."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            target = defs.get(ref.removeprefix("#/$defs/"), {})
            merged = {**_inline_refs(target, defs)}
            merged.update({k: v for k, v in node.items() if k != "$ref"})
            return merged
        return {k: _inline_refs(v, defs) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline_refs(item, defs) for item in node]
    return node


@dataclass(frozen=True, slots=True)
class Toolset:
    """A named group of tools that can be enabled or disabled together.

    Grouping matters for more than convenience: an agent answering strangers on
    a public LINE account must not have ``terminal`` in reach, and per-surface
    toolset gating is how that is expressed.
    """

    name: str
    description: str
    tools: tuple[str, ...] = ()
    includes: tuple[str, ...] = field(default_factory=tuple)
