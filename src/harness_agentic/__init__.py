"""Harness-Agentic: a self-improving agent framework.

Importing this package must stay cheap -- no provider SDK, no HTTP client, no
filesystem scan happens here. Subsystems are imported by the CLI and the
gateway when they are actually needed.
"""

from __future__ import annotations

from harness_agentic.version import __version__

__all__ = ["__version__"]
