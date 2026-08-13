"""Rendering agent events to a terminal.

A subscriber, not a special case. The gateway, the web UI, and the voice front
end will each write one of these against the same event stream -- which is only
possible because the loop emits instead of printing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from harness_agentic.cli.render import console, err_console
from harness_agentic.core.events import (
    AgentEvent,
    ApprovalRequested,
    Notice,
    ProviderFallback,
    RetryScheduled,
    TextChunk,
    ThinkingChunk,
    ToolCallFinished,
    ToolCallStarted,
    ToolProgress,
    TurnFinished,
    UsageReported,
)
from harness_agentic.tools.spec import Danger

if TYPE_CHECKING:
    from harness_agentic.core.types import Usage

_DANGER_STYLE = {
    Danger.SAFE: "dim",
    Danger.NETWORK: "cyan",
    Danger.WRITES: "yellow",
    Danger.DESTRUCTIVE: "bold red",
}


class ConsoleRenderer:
    """Prints agent events as they happen."""

    def __init__(self, *, show_thinking: bool = False, show_usage: bool = True) -> None:
        """Configure what to display."""
        self._show_thinking = show_thinking
        self._show_usage = show_usage
        self._streaming = False
        self.wrote_answer = False
        """Whether any answer text has been printed.

        Read by ``harn run``, which otherwise cannot tell "the model said
        nothing" from "the answer never came through this renderer". With
        ``--no-stream`` there are no text chunks at all, and the command printed
        a usage line and nothing else.
        """

    def __call__(self, event: AgentEvent) -> None:  # noqa: PLR0912
        """Render one event. One branch per event kind."""
        match event:
            case TextChunk(text=text):
                self._streaming = True
                self.wrote_answer = True
                console.print(text, end="", markup=False, highlight=False)

            case ThinkingChunk(text=text) if self._show_thinking:
                console.print(f"[dim italic]{text}[/]", end="", markup=True)

            case ToolCallStarted(tool=tool, danger=danger, summary=summary):
                self._break_stream()
                style = _DANGER_STYLE.get(danger, "")
                console.print(f"  [{style}]>[/] {summary}", markup=True, highlight=False)

            case ToolProgress(message=message):
                console.print(f"    [dim]{message}[/]", markup=True, highlight=False)

            case ToolCallFinished(tool=tool, is_error=is_error, duration_s=duration):
                mark = "[red]✗[/]" if is_error else "[green]✓[/]"
                console.print(f"  {mark} [dim]{tool} ({duration:.1f}s)[/]", markup=True)

            case ApprovalRequested(summary=summary):
                self._break_stream()
                console.print(f"[yellow]approval needed:[/] {summary}", markup=True)

            case ProviderFallback(from_model=old, to_model=new, reason=reason):
                self._break_stream()
                err_console.print(f"[yellow]falling back {old} → {new}[/]: {reason}")

            case RetryScheduled(attempt=attempt, delay_s=delay, reason=reason):
                err_console.print(f"[dim]retry {attempt} in {delay:.1f}s: {reason}[/]")

            case Notice(level=level, message=message):
                self._break_stream()
                colour = {"info": "dim", "warning": "yellow", "error": "red"}[level]
                err_console.print(f"[{colour}]{message}[/]")

            case UsageReported(cumulative=cumulative) if self._show_usage:
                self._last_usage = cumulative

            case TurnFinished(reason=reason, iterations=iterations, usage=usage):
                self._break_stream()
                if reason != "completed":
                    err_console.print(f"[yellow]turn ended: {reason}[/]")
                if self._show_usage:
                    console.print(f"[dim]{_usage_line(usage, iterations)}[/]", markup=True)

            case _:
                pass

    def _break_stream(self) -> None:
        """End the current streamed line before printing structured output."""
        if self._streaming:
            console.print()
            self._streaming = False


def _usage_line(usage: Usage, iterations: int) -> str:
    """Summarize a turn's cost.

    Cache reads are shown separately because a number that suddenly collapses
    is the visible symptom of a broken cached prefix -- and that is otherwise
    an invisible, expensive regression.
    """
    parts = [
        f"{iterations} step(s)",
        f"in {usage.input_tokens}",
        f"out {usage.output_tokens}",
    ]
    if usage.cache_read_tokens:
        parts.append(f"cached {usage.cache_read_tokens}")
    return "  ".join(parts)
