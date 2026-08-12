"""Platform adapters.

Imported lazily by name so that installing the framework does not require every
platform's dependencies -- and so that a broken or unconfigured adapter cannot
stop the others from starting.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from harness_agentic.errors import AdapterError

if TYPE_CHECKING:
    from harness_agentic.gateway.adapter import PlatformAdapter

KNOWN: dict[str, str] = {
    "fake": "harness_agentic.gateway.platforms.fake:FakeAdapter",
    "telegram": "harness_agentic.gateway.platforms.telegram:TelegramAdapter",
    "line": "harness_agentic.gateway.platforms.line:LineAdapter",
}
"""Platform name to ``module:attribute``. The single place a name resolves."""


def load(platform: str) -> type[PlatformAdapter]:
    """Import one adapter class by platform name."""
    target = KNOWN.get(platform)
    if target is None:
        known = ", ".join(sorted(KNOWN))
        detail = f"unknown platform {platform!r}; known platforms are {known}"
        raise AdapterError(detail)
    module_name, _, attribute = target.partition(":")
    module = importlib.import_module(module_name)
    loaded: type[PlatformAdapter] = getattr(module, attribute)
    return loaded
