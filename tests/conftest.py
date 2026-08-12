"""Fixtures that make the suite hermetic by default.

Two of these are autouse and load-bearing. A unit test that quietly reaches a
real provider becomes flaky and expensive rather than failing, and a test that
writes to the developer's real ``~/.harness`` corrupts their state. Both are
closed off here rather than left to each test's discipline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from harness_agentic.constants import ENV_HOME, ENV_PREFIX, ENV_PROFILE

_REFUSAL = (
    "network access from a unit test -- use FakeTransport or httpx.MockTransport, "
    "or mark the test with @pytest.mark.integration"
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any outbound HTTP call fail loudly.

    Tests that legitimately need the network carry ``@pytest.mark.integration``
    and are deselected by default; they re-enable access by overriding this
    fixture.
    """
    import httpx  # noqa: PLC0415  -- imported here so collection stays cheap

    real_sync = httpx.Client.send
    real_async = httpx.AsyncClient.send

    def _mocked(client: Any) -> bool:
        """Whether this client is wired to an in-memory transport.

        A client built on ``httpx.MockTransport`` never opens a socket, so
        letting it through keeps the guard's actual promise -- no real network
        -- while allowing adapters to be tested against recorded wire traffic.
        The alternative, overriding this fixture per test, switches the guard
        off entirely for those tests, which is how a live call sneaks back in.
        """
        return isinstance(getattr(client, "_transport", None), httpx.MockTransport)

    def _blocked_sync(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _mocked(self):
            return real_sync(self, *args, **kwargs)
        raise RuntimeError(_REFUSAL)

    async def _blocked_async(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _mocked(self):
            return await real_async(self, *args, **kwargs)
        raise RuntimeError(_REFUSAL)

    monkeypatch.setattr(httpx.Client, "send", _blocked_sync)
    monkeypatch.setattr(httpx.AsyncClient, "send", _blocked_async)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every path lookup at a throwaway directory.

    Also clears inherited ``HARNESS__*`` overrides so a developer's shell
    environment cannot change what the suite asserts.
    """
    home = tmp_path / "harness-home"
    home.mkdir()
    monkeypatch.setenv(ENV_HOME, str(home))
    monkeypatch.setenv(ENV_PROFILE, "default")
    for key in [k for k in os.environ if k.startswith(ENV_PREFIX)]:
        monkeypatch.delenv(key, raising=False)
    return home
