"""Cooperative cancellation.

Cancellation has to be cooperative because the alternative does not exist:
``asyncio.CancelledError`` cannot interrupt a running ``subprocess.run``, and
threads cannot be killed. What actually stops work is a flag checked at loop
boundaries, an aborted HTTP stream, and a killed process *group* -- and all
three work identically whether the caller is sync or async.
"""

from __future__ import annotations

import threading


class CancelToken:
    """A thread-safe stop flag carrying the reason it was set."""

    __slots__ = ("_event", "_reason")

    def __init__(self) -> None:
        """Create an unset token."""
        self._event = threading.Event()
        self._reason: str = ""

    def cancel(self, reason: str = "cancelled") -> None:
        """Request cancellation. Idempotent; the first reason wins."""
        if not self._event.is_set():
            self._reason = reason
        self._event.set()

    def is_set(self) -> bool:
        """Whether cancellation has been requested."""
        return self._event.is_set()

    @property
    def reason(self) -> str:
        """Why cancellation was requested, or the empty string."""
        return self._reason

    def wait(self, timeout: float | None = None) -> bool:
        """Block until cancelled or ``timeout`` elapses."""
        return self._event.wait(timeout)

    def raise_if_cancelled(self) -> None:
        """Raise :class:`~harness_agentic.errors.Interrupted` if cancelled."""
        if self._event.is_set():
            from harness_agentic.errors import Interrupted  # noqa: PLC0415  -- avoids a cycle

            raise Interrupted(self._reason or "cancelled")


NEVER_CANCELLED = CancelToken()
"""A token that is never set. Use as a default rather than ``None``."""
