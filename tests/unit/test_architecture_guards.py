"""The architectural guards are themselves tested.

A guard that silently stops catching things is worse than no guard, because the
invariant it protects keeps being cited in review as though it were enforced.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_io_guard_flags_subprocess_in_a_tool(tmp_path: Path) -> None:
    guard = _load("check_io_boundary")
    offender = tmp_path / "tools" / "bad.py"
    offender.parent.mkdir()
    offender.write_text("import subprocess\nsubprocess.run(['ls'])\n", encoding="utf-8")
    assert guard.check(offender)


def test_io_guard_allows_the_environment_layer(tmp_path: Path) -> None:
    guard = _load("check_io_boundary")
    ok = tmp_path / "envs" / "local.py"
    ok.parent.mkdir()
    ok.write_text("import subprocess\nsubprocess.run(['ls'])\n", encoding="utf-8")
    assert guard.main([str(ok)]) == 0


def test_async_guard_flags_a_loop_in_the_core(tmp_path: Path) -> None:
    guard = _load("check_async_boundary")
    offender = tmp_path / "agent" / "loop.py"
    offender.parent.mkdir()
    offender.write_text("import asyncio\nasyncio.run(main())\n", encoding="utf-8")
    assert guard.check(offender)


def test_async_guard_allows_the_gateway(tmp_path: Path) -> None:
    guard = _load("check_async_boundary")
    ok = tmp_path / "gateway" / "runner.py"
    ok.parent.mkdir()
    ok.write_text("import asyncio\nasyncio.run(main())\n", encoding="utf-8")
    assert guard.main([str(ok)]) == 0


def test_guards_pass_against_the_real_tree() -> None:
    io_guard = _load("check_io_boundary")
    async_guard = _load("check_async_boundary")
    sources = [str(p) for p in (REPO_ROOT / "src").rglob("*.py")]
    assert async_guard.main(sources) == 0
    tools = [str(p) for p in (REPO_ROOT / "src").rglob("tools/**/*.py")]
    assert io_guard.main(tools) == 0
