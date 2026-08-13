"""Reading the configured MCP servers.

Its own module because two callers need it and they must agree: ``harn mcp``
inspects the servers, and ``harn run`` / ``harn chat`` hand them to an agent. The
loader living inside the inspection command is how an agent came to have no MCP
tools at all while ``harn mcp check`` reported them working.

Its own file rather than a section of ``config.toml`` because a server entry is a
command line to execute. Settings are values; this is a list of subprocesses, and
keeping it separate makes "what will this start" one file to read.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

from harness_agentic.constants import harness_home
from harness_agentic.errors import HarnessError
from harness_agentic.mcp.stdio import load_servers

if TYPE_CHECKING:
    from harness_agentic.mcp.stdio import ServerConfig

CONFIG_FILENAME = "mcp.toml"


class McpConfigError(HarnessError):
    """The MCP configuration file could not be used."""


def config_path(explicit: Path | None = None) -> Path:
    """Where MCP server configuration is read from."""
    return explicit or (harness_home() / CONFIG_FILENAME)


def configured_servers(explicit: Path | None = None) -> list[ServerConfig]:
    """The servers to start, or an empty list when none are configured.

    Raises rather than returning nothing when the file is present and wrong. A
    malformed entry silently becoming "no servers" is how an operator ends up
    debugging a missing tool instead of reading a typo.
    """
    path = config_path(explicit)
    if not path.exists():
        return []
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        msg = f"{path}: {exc}"
        raise McpConfigError(msg) from exc
    section = raw.get("mcp") or raw
    entries = section.get("servers") or []
    if not isinstance(entries, list):
        msg = f"{path} should contain a list of [[mcp.servers]] tables"
        raise McpConfigError(msg)
    try:
        loaded: list[ServerConfig] = load_servers(entries)
    except (TypeError, ValueError) as exc:
        msg = f"{path}: {exc}"
        raise McpConfigError(msg) from exc
    return loaded
