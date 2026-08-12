"""Running tool calls.

One rule governs this module: **a tool can never kill a turn.** Unknown name,
malformed arguments, a handler that raises, a timeout, a refused approval --
every one of them becomes a ``ToolResultBlock`` with ``is_error=True`` and goes
back to the model, which usually fixes itself on the next iteration. An
exception escaping here would throw away everything the turn had achieved.

Ordering is the other invariant. Results come back in the order the calls were
made regardless of how they were scheduled, because providers pair tool results
against tool calls positionally.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from harness_agentic.core.events import (
    EventSink,
    ToolCallFinished,
    ToolCallStarted,
    null_sink,
)
from harness_agentic.core.types import ToolResultBlock, ToolUseBlock
from harness_agentic.errors import ToolNotFound
from harness_agentic.tools.spec import ApprovalRequest, Danger, Tool, ToolContext, ToolResult

if TYPE_CHECKING:
    from harness_agentic.tools.approval import ApprovalPolicy
    from harness_agentic.tools.registry import ToolRegistry

_TRUNCATION_NOTE = "\n\n[... output truncated at {limit} characters ...]"


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    """What happened when one call ran. Feeds traces and the session store."""

    call_id: str
    tool: str
    arguments: dict[str, Any]
    result: ToolResult
    duration_s: float


class ToolExecutor:
    """Validates, authorizes, runs, and bounds every tool call."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        approval: ApprovalPolicy,
        emit: EventSink = null_sink,
        max_parallel: int = 4,
    ) -> None:
        """Bind an executor to a registry and an approval policy."""
        self._registry = registry
        self._approval = approval
        self._emit = emit
        self._max_parallel = max(1, max_parallel)
        self.records: list[ToolCallRecord] = []

    # -- single call --------------------------------------------------------

    def execute(self, call: ToolUseBlock, ctx: ToolContext) -> ToolResultBlock:
        """Run one call and return a result block, never an exception."""
        started = time.monotonic()
        try:
            tool = self._registry.get(call.name)
        except ToolNotFound:
            known = ", ".join(sorted(self._registry.all())) or "none"
            return self._finish(
                call,
                ToolResult.error(f"No tool named {call.name!r}. Available: {known}"),
                started,
                arguments={},
            )

        self._emit(
            ToolCallStarted(
                call_id=call.id,
                tool=tool.name,
                danger=tool.danger,
                summary=_summarize(tool, call),
            )
        )

        parsed = self._parse_arguments(tool, call)
        if isinstance(parsed, ToolResult):
            return self._finish(call, parsed, started, arguments={})

        if not tool.is_available():
            missing = ", ".join(tool.requires_env_vars) or "unmet preconditions"
            return self._finish(
                call,
                ToolResult.error(f"{tool.name} is unavailable ({missing})"),
                started,
                arguments=parsed.model_dump(),
            )

        decision = self._approval.check(
            ApprovalRequest(
                tool=tool.name,
                danger=tool.danger,
                summary=_summarize(tool, call),
                detail=str(dict(call.arguments)),
            )
        )
        if not decision.granted:
            return self._finish(
                call,
                ToolResult.error(f"Not permitted: {decision.reason}"),
                started,
                arguments=parsed.model_dump(),
            )

        try:
            result = tool.handler(parsed, ctx)
        except Exception as exc:  # a handler must never end the turn
            result = ToolResult.error(f"{tool.name} failed: {type(exc).__name__}: {exc}")

        return self._finish(call, self._bound(result, tool), started, arguments=parsed.model_dump())

    def _parse_arguments(self, tool: Tool, call: ToolUseBlock) -> Any:
        """Validate arguments, or return the error the model should see."""
        try:
            return tool.params_model.model_validate(dict(call.arguments))
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}"
                for e in exc.errors()
            )
            hint = ""
            if call.raw_arguments and not call.arguments:
                hint = f" The arguments were not valid JSON: {call.raw_arguments[:200]!r}."
            return ToolResult.error(f"Invalid arguments for {tool.name}: {problems}.{hint}")

    def _bound(self, result: ToolResult, tool: Tool) -> ToolResult:
        """Cap result size so one call cannot swallow the context window.

        A single ``grep`` across a large repository can add tens of thousands
        of tokens between two budget checks, which is how a turn overflows
        without any individual step looking unreasonable.
        """
        limit = tool.max_result_chars
        if len(result.text) <= limit:
            return result
        note = _TRUNCATION_NOTE.format(limit=limit)
        return ToolResult(
            text=result.text[: limit - len(note)] + note,
            is_error=result.is_error,
            display=result.display,
            images=result.images,
            truncated=True,
            data=result.data,
        )

    def _finish(
        self,
        call: ToolUseBlock,
        result: ToolResult,
        started: float,
        *,
        arguments: dict[str, Any],
    ) -> ToolResultBlock:
        """Record, emit, and convert a result into a wire block."""
        duration = time.monotonic() - started
        self.records.append(
            ToolCallRecord(
                call_id=call.id,
                tool=call.name,
                arguments=arguments,
                result=result,
                duration_s=duration,
            )
        )
        self._emit(
            ToolCallFinished(
                call_id=call.id,
                tool=call.name,
                is_error=result.is_error,
                duration_s=duration,
                display=result.display,
            )
        )
        return ToolResultBlock(
            tool_use_id=call.id,
            text=result.text,
            is_error=result.is_error,
            truncated=result.truncated,
        )

    # -- batches ------------------------------------------------------------

    def execute_batch(
        self, calls: Sequence[ToolUseBlock], ctx: ToolContext
    ) -> list[ToolResultBlock]:
        """Run several calls, returning results in the original order.

        Read-only and network calls run concurrently on a thread pool -- they
        are blocking I/O, which is what threads are for. Anything that writes
        runs serially in the order the model asked for, because two writes to
        the same file racing is not a performance win.
        """
        if not calls:
            return []
        if len(calls) == 1:
            return [self.execute(calls[0], ctx)]

        results: list[ToolResultBlock | None] = [None] * len(calls)
        parallel: list[int] = []

        for index, call in enumerate(calls):
            tool = self._registry.maybe_get(call.name)
            if tool is not None and tool.danger <= Danger.NETWORK:
                parallel.append(index)
            else:
                results[index] = self.execute(call, ctx)

        if parallel:
            workers = min(self._max_parallel, len(parallel))
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="harness-tool") as pool:
                futures = {pool.submit(self.execute, calls[i], ctx): i for i in parallel}
                for future, index in futures.items():
                    results[index] = future.result()

        return [r for r in results if r is not None]


def _summarize(tool: Tool, call: ToolUseBlock) -> str:
    """A one-line description of a call, for approval prompts and events.

    For shell tools the command itself is the only summary worth showing -- an
    operator approving ``terminal`` needs to see what it will run, not the word
    "terminal".
    """
    arguments = dict(call.arguments)
    for key in ("command", "cmd", "path", "query", "url"):
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return f"{tool.name}: {value[:200]}"
    return tool.name
