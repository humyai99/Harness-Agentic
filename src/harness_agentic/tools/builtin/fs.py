"""Filesystem tools.

Every path here goes through ``ctx.env``. Nothing in this module opens a file
directly, and ``scripts/check_io_boundary.py`` fails the commit if that
changes -- which is what lets the same code run unmodified inside Docker.
"""

from __future__ import annotations

from pathlib import PurePath

from pydantic import BaseModel, Field

from harness_agentic.errors import PathOutsideWorkspace
from harness_agentic.tools.paths import PathPolicy
from harness_agentic.tools.registry import registry
from harness_agentic.tools.spec import Danger, ToolContext, ToolResult

_policy = PathPolicy()

MAX_LINE_CHARS = 2000
"""One minified bundle line should not consume the whole context window."""


class ReadFileParams(BaseModel):
    """Arguments for :func:`read_file`."""

    path: str = Field(description="File path, absolute or relative to the workspace root.")
    offset: int = Field(0, ge=0, description="First line to return, zero-based.")
    limit: int = Field(2000, gt=0, le=10_000, description="How many lines to return.")


class WriteFileParams(BaseModel):
    """Arguments for :func:`write_file`."""

    path: str = Field(description="File path, absolute or relative to the workspace root.")
    content: str = Field(description="The complete new contents of the file.")


class ListDirParams(BaseModel):
    """Arguments for :func:`list_dir`."""

    path: str = Field(".", description="Directory to list.")


class GlobParams(BaseModel):
    """Arguments for :func:`glob_files`."""

    pattern: str = Field(description="Glob pattern, for example 'src/**/*.py'.")
    limit: int = Field(200, gt=0, le=2000, description="Maximum paths to return.")


class GrepParams(BaseModel):
    """Arguments for :func:`grep_files`."""

    pattern: str = Field(description="Regular expression to search for.")
    glob: str = Field("**/*", description="Restrict the search to files matching this glob.")
    limit: int = Field(100, gt=0, le=1000, description="Maximum matching lines to return.")
    ignore_case: bool = Field(default=False, description="Match case-insensitively.")


@registry.tool(toolset="file", danger=Danger.SAFE, max_result_chars=60_000)
def read_file(params: ReadFileParams, ctx: ToolContext) -> ToolResult:
    """Read a text file with line numbers. Use offset and limit for large files."""
    target = PurePath(params.path)
    try:
        _policy.check(target if target.is_absolute() else ctx.env.root / target, root=ctx.env.root)
        raw = ctx.env.read_bytes(target)
    except PathOutsideWorkspace as exc:
        return ToolResult.error(str(exc))
    except FileNotFoundError:
        return ToolResult.error(f"{params.path} does not exist")
    except OSError as exc:
        return ToolResult.error(f"could not read {params.path}: {exc}")

    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    window = lines[params.offset : params.offset + params.limit]
    if not window:
        return ToolResult(text=f"{params.path} has {len(lines)} lines; offset is past the end.")

    width = len(str(params.offset + len(window)))
    body = "\n".join(
        f"{params.offset + i + 1:>{width}}\t{line[:MAX_LINE_CHARS]}"
        for i, line in enumerate(window)
    )
    footer = ""
    if params.offset + len(window) < len(lines):
        footer = f"\n\n[showing {len(window)} of {len(lines)} lines]"
    return ToolResult(text=body + footer, display=f"read {params.path} ({len(window)} lines)")


@registry.tool(toolset="file", danger=Danger.WRITES)
def write_file(params: WriteFileParams, ctx: ToolContext) -> ToolResult:
    """Write a file, replacing it entirely. Creates parent directories as needed."""
    target = PurePath(params.path)
    try:
        _policy.check(target if target.is_absolute() else ctx.env.root / target, root=ctx.env.root)
        existing = ctx.env.stat(target)
        ctx.env.write_bytes(target, params.content.encode("utf-8"))
    except PathOutsideWorkspace as exc:
        return ToolResult.error(str(exc))
    except OSError as exc:
        return ToolResult.error(f"could not write {params.path}: {exc}")

    verb = "updated" if existing else "created"
    lines = params.content.count("\n") + 1
    return ToolResult(
        text=f"{verb} {params.path} ({lines} lines)",
        display=f"{verb} {params.path}",
        data={"path": params.path, "lines": lines, "created": existing is None},
    )


@registry.tool(toolset="file", danger=Danger.SAFE)
def list_dir(params: ListDirParams, ctx: ToolContext) -> ToolResult:
    """List one directory. Directories are marked with a trailing slash."""
    try:
        entries = ctx.env.list_dir(PurePath(params.path))
    except PathOutsideWorkspace as exc:
        return ToolResult.error(str(exc))
    except (FileNotFoundError, NotADirectoryError):
        return ToolResult.error(f"{params.path} is not a directory")
    except OSError as exc:
        return ToolResult.error(f"could not list {params.path}: {exc}")

    if not entries:
        return ToolResult(text=f"{params.path} is empty")
    body = "\n".join(f"{e.name}/" if e.is_dir else f"{e.name}\t{e.size}" for e in entries)
    return ToolResult(text=body, display=f"listed {params.path} ({len(entries)} entries)")


@registry.tool(toolset="file", danger=Danger.SAFE)
def glob_files(params: GlobParams, ctx: ToolContext) -> ToolResult:
    """Find files by glob pattern, relative to the workspace root."""
    try:
        found = ctx.env.glob(params.pattern)
    except OSError as exc:
        return ToolResult.error(f"glob failed: {exc}")

    visible = [p for p in found if not _policy.is_denied(p, root=ctx.env.root)]
    if not visible:
        return ToolResult(text=f"no files match {params.pattern}")
    shown = visible[: params.limit]
    body = "\n".join(str(p) for p in shown)
    if len(visible) > len(shown):
        body += f"\n\n[{len(visible) - len(shown)} more matches not shown]"
    return ToolResult(text=body, display=f"{len(visible)} match(es) for {params.pattern}")


@registry.tool(toolset="file", danger=Danger.SAFE, max_result_chars=30_000)
def grep_files(params: GrepParams, ctx: ToolContext) -> ToolResult:
    """Search file contents with a regular expression. Returns path:line:text."""
    import re  # noqa: PLC0415  -- only paid when the tool actually runs

    try:
        matcher = re.compile(params.pattern, re.IGNORECASE if params.ignore_case else 0)
    except re.error as exc:
        return ToolResult.error(f"invalid pattern: {exc}")

    hits: list[str] = []
    scanned = 0
    for path in ctx.env.glob(params.glob):
        if len(hits) >= params.limit:
            break
        if _policy.is_denied(path, root=ctx.env.root):
            continue
        info = ctx.env.stat(path)
        if info is None or info.is_dir:
            continue
        try:
            content = ctx.env.read_bytes(path).decode("utf-8", errors="replace")
        except (OSError, ValueError):
            continue
        scanned += 1
        for number, line in enumerate(content.splitlines(), start=1):
            if matcher.search(line):
                hits.append(f"{path}:{number}:{line[:MAX_LINE_CHARS]}")
                if len(hits) >= params.limit:
                    break

    if not hits:
        return ToolResult(text=f"no matches for {params.pattern!r} in {scanned} file(s)")
    return ToolResult(
        text="\n".join(hits),
        display=f"{len(hits)} match(es) for {params.pattern!r}",
    )
