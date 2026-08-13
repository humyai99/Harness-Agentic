"""The one place the CLI writes to a terminal.

Routing all output through a single console keeps ``ruff``'s ``T20`` rule
enforceable and means the gateway can render the same events to a chat
platform by swapping the sink rather than by rewriting call sites.
"""

from __future__ import annotations

from rich.console import Console

console: Console = Console()
"""Standard output."""

err_console: Console = Console(stderr=True)
"""Standard error. Diagnostics go here so stdout stays pipeable."""
