"""Running commands.

Classified ``DESTRUCTIVE`` unconditionally. Not because every command is
dangerous -- most are ``ls`` -- but because the danger level is what routes a
call to the approval policy, and there is no reliable way to tell in advance
which shell string is the harmless one. The allowlist is where "this particular
command is fine unattended" gets expressed, and it is a list a human wrote.
"""

from __future__ import annotations

from pathlib import PurePath

from pydantic import BaseModel, Field

from harness_agentic.envs.base import DEFAULT_MAX_OUTPUT_BYTES
from harness_agentic.errors import PathOutsideWorkspace
from harness_agentic.tools.registry import registry
from harness_agentic.tools.spec import Danger, ToolContext, ToolResult

MAX_TIMEOUT_S = 600.0


class TerminalParams(BaseModel):
    """Arguments for :func:`terminal`."""

    command: str = Field(description="The shell command to run.")
    cwd: str | None = Field(None, description="Working directory, relative to the workspace root.")
    timeout_s: float = Field(
        120.0, gt=0, le=MAX_TIMEOUT_S, description="Kill the command after this long."
    )


@registry.tool(
    toolset="terminal",
    danger=Danger.DESTRUCTIVE,
    max_result_chars=30_000,
    timeout_s=MAX_TIMEOUT_S,
)
def terminal(params: TerminalParams, ctx: ToolContext) -> ToolResult:
    """Run a shell command in the workspace and return its output.

    Prefer specific tools when one exists -- read_file over cat, glob_files
    over find. They are cheaper, they are not gated behind approval, and their
    output is shaped for reading.
    """
    cwd = PurePath(params.cwd) if params.cwd else None
    ctx.emit(f"$ {params.command}")
    try:
        outcome = ctx.env.run_shell(
            params.command,
            cwd=cwd,
            timeout_s=params.timeout_s,
            cancel=ctx.cancel,
            max_output_bytes=DEFAULT_MAX_OUTPUT_BYTES,
        )
    except PathOutsideWorkspace as exc:
        return ToolResult.error(str(exc))
    except OSError as exc:
        return ToolResult.error(f"could not run command: {exc}")

    body = _render(outcome.stdout, outcome.stderr)

    if outcome.timed_out:
        return ToolResult(
            text=f"timed out after {params.timeout_s:g}s\n\n{body}".rstrip(),
            is_error=True,
            display=f"$ {params.command} (timed out)",
        )

    if outcome.exit_code != 0:
        # A non-zero exit is reported as an error so the model reacts to it,
        # but the output still goes back -- the failure message is the point.
        return ToolResult(
            text=f"exit code {outcome.exit_code}\n\n{body}".rstrip(),
            is_error=True,
            display=f"$ {params.command} (exit {outcome.exit_code})",
            truncated=outcome.truncated,
        )

    return ToolResult(
        text=body or "(no output)",
        display=f"$ {params.command}",
        truncated=outcome.truncated,
        data={"exit_code": 0, "duration_s": round(outcome.duration_s, 3)},
    )


def _render(stdout: str, stderr: str) -> str:
    """Combine the two streams, labelling stderr only when both are present."""
    out, err = stdout.strip(), stderr.strip()
    if out and err:
        return f"{out}\n\n--- stderr ---\n{err}"
    return out or err
