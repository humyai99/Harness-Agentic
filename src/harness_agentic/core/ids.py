"""Identifier generation, injected so tests can make it deterministic.

Tool-use ids appear in golden prompt fixtures and in recorded wire payloads; a
random id would make both unstable.
"""

from __future__ import annotations

import uuid
from typing import Protocol


class IdGen(Protocol):
    """Source of unique identifiers."""

    def new(self, prefix: str = "") -> str:
        """Return a fresh identifier, optionally prefixed."""
        ...


class UuidIdGen:
    """Production identifiers."""

    def new(self, prefix: str = "") -> str:
        """Return a fresh hex identifier."""
        value = uuid.uuid4().hex[:16]
        return f"{prefix}_{value}" if prefix else value


class SequentialIdGen:
    """Deterministic identifiers for tests and fixtures."""

    def __init__(self) -> None:
        """Start the counter at zero."""
        self._n = 0

    def new(self, prefix: str = "") -> str:
        """Return the next identifier in sequence."""
        self._n += 1
        return f"{prefix}_{self._n:04d}" if prefix else f"{self._n:04d}"
