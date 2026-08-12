"""Test doubles, shipped with the package rather than kept in ``tests/``.

Plugin authors need the same fakes we do. A tool, a platform adapter, or a
memory provider written outside this repository should be testable without
network access or an API key, and that only works if the fakes are importable.
"""

from harness_agentic.testing.fakes import (
    FakeTransport,
    ScriptedTurn,
    text_turn,
    tool_turn,
)

__all__ = ["FakeTransport", "ScriptedTurn", "text_turn", "tool_turn"]
