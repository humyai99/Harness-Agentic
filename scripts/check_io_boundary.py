#!/usr/bin/env python3
"""Fail if a tool module performs raw I/O instead of going through an environment.

``ExecEnvironment`` is what makes the docker and ssh backends free: every
builtin tool reaches the filesystem and the shell through it, so swapping the
backend swaps where the tools run without touching a single tool. That property
only holds while the rule holds, so it is checked rather than documented.

Usage: check_io_boundary.py FILE [FILE ...]
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

BANNED_CALLS: dict[str, str] = {
    "open": "use ctx.env.read_bytes() / ctx.env.write_bytes()",
    "subprocess": "use ctx.env.run() or ctx.env.run_shell()",
    "os.system": "use ctx.env.run_shell()",
    "os.popen": "use ctx.env.run()",
}
BANNED_PATH_METHODS: frozenset[str] = frozenset(
    {"read_text", "read_bytes", "write_text", "write_bytes", "unlink", "mkdir", "rmdir"}
)
BANNED_IMPORTS: frozenset[str] = frozenset({"subprocess", "shutil"})


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def check(path: Path) -> list[str]:
    """Return one message per violation found in ``path``."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:  # let ruff report the syntax error itself
        return [f"{path}: could not parse ({exc})"]

    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in BANNED_IMPORTS:
                    problems.append(
                        f"{path}:{node.lineno}: imports {alias.name!r}; "
                        f"tools must go through ctx.env"
                    )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in BANNED_IMPORTS:
                problems.append(
                    f"{path}:{node.lineno}: imports from {node.module!r}; "
                    f"tools must go through ctx.env"
                )
        elif isinstance(node, ast.Call):
            name = _dotted(node.func)
            for banned, hint in BANNED_CALLS.items():
                if name == banned or name.startswith(f"{banned}."):
                    problems.append(f"{path}:{node.lineno}: calls {name}(); {hint}")
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in BANNED_PATH_METHODS
                and "path" in _dotted(node.func).lower()
            ):
                problems.append(f"{path}:{node.lineno}: filesystem call {name}(); use ctx.env")
    return problems


def main(argv: list[str]) -> int:
    """Check every file named on the command line."""
    problems: list[str] = []
    for arg in argv:
        path = Path(arg)
        # The environments themselves are the sanctioned place for real I/O.
        if "envs" in path.parts:
            continue
        problems.extend(check(path))
    for line in problems:
        print(line, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
