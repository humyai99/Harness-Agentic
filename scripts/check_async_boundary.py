#!/usr/bin/env python3
"""Fail if event-loop plumbing leaks outside the two modules allowed to have it.

The core -- agent loop, tools, transports, session store -- is synchronous. The
gateway is asyncio. Exactly one module bridges them. That split is deliberate:
a blocking call inside an async core stalls every chat platform at once, and
the mistake would be spread across every tool author. Centralising it means the
hazard lives in one reviewed file, and this check is what keeps it there.

Usage: check_async_boundary.py FILE [FILE ...]
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

BANNED: frozenset[str] = frozenset(
    {
        "asyncio.run",
        "asyncio.get_event_loop",
        "asyncio.new_event_loop",
        "asyncio.set_event_loop",
        "asyncio.run_coroutine_threadsafe",
    }
)

ALLOWED_SUFFIXES: tuple[tuple[str, ...], ...] = (
    ("harness_agentic", "core", "async_bridge.py"),
    ("harness_agentic", "cli", "main.py"),
)
ALLOWED_PACKAGES: frozenset[str] = frozenset({"gateway", "testing"})


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _exempt(path: Path) -> bool:
    parts = path.parts
    if any(pkg in parts for pkg in ALLOWED_PACKAGES):
        return True
    return any(parts[-len(suffix) :] == suffix for suffix in ALLOWED_SUFFIXES)


def check(path: Path) -> list[str]:
    """Return one message per violation found in ``path``."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [f"{path}: could not parse ({exc})"]

    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name in BANNED:
                problems.append(
                    f"{path}:{node.lineno}: {name}() outside the async bridge; "
                    f"the core is synchronous -- route through "
                    f"harness_agentic.core.async_bridge"
                )
    return problems


def main(argv: list[str]) -> int:
    """Check every file named on the command line."""
    problems: list[str] = []
    for arg in argv:
        path = Path(arg)
        if _exempt(path):
            continue
        problems.extend(check(path))
    for line in problems:
        print(line, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
