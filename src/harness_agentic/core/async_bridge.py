"""The one place where the synchronous core meets asyncio.

The core -- agent loop, tools, transports, session store -- is synchronous. The
chat gateway is asyncio, because every platform SDK is. This module is the only
sanctioned crossing, and ``scripts/check_async_boundary.py`` enforces that in
pre-commit and in CI.

Why this way round. Tool handlers are overwhelmingly blocking work: subprocess,
file reads, sqlite, ripgrep. An async core would force every tool author and
plugin author to write ``async def`` and then immediately ``await
asyncio.to_thread(...)`` inside it -- no benefit, and one forgotten ``await``
stalls every chat platform at once. Keeping the core synchronous concentrates
that hazard in this file, where it can be reviewed once.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from concurrent.futures import Future
from typing import Any, TypeVar

T = TypeVar("T")


class AsyncBridge:
    """Runs coroutines on a private event loop from synchronous code.

    The loop lives on a daemon thread and is started lazily, so a purely
    synchronous session never pays for it.
    """

    def __init__(self) -> None:
        """Create an idle bridge; the loop thread starts on first use."""
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is not None:
                return self._loop
            ready = threading.Event()
            loop_box: list[asyncio.AbstractEventLoop] = []

            def _run() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop_box.append(loop)
                ready.set()
                loop.run_forever()

            thread = threading.Thread(
                target=_run, name="harness-async-bridge", daemon=True
            )
            thread.start()
            ready.wait()
            self._loop = loop_box[0]
            self._thread = thread
            return self._loop

    def run(self, coro: Coroutine[Any, Any, T], *, timeout: float | None = None) -> T:
        """Run ``coro`` to completion and return its result.

        Raises whatever the coroutine raises. On timeout the coroutine is
        cancelled and :class:`concurrent.futures.TimeoutError` propagates.
        """
        loop = self._ensure_loop()
        future: Future[T] = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return future.result(timeout)
        except TimeoutError:
            future.cancel()
            raise

    def close(self) -> None:
        """Stop the loop thread. Safe to call when it was never started."""
        with self._lock:
            loop, thread = self._loop, self._thread
            self._loop, self._thread = None, None
        if loop is None:
            return
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5.0)
        loop.close()

    def __enter__(self) -> AsyncBridge:
        """Enter a context that closes the loop on exit."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Stop the loop thread."""
        self.close()
