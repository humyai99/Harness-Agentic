"""Letting an agent read a database without letting it change one.

Two separate problems, and conflating them is how this goes wrong.

**Read-only is enforced twice.** The statement is gated here, *and* the
connection is opened read-only where the driver supports it. Either alone is
insufficient: a parser can be fooled by syntax it does not know, and a read-only
connection still lets a hostile query lock a table or scan a billion rows. Both
together mean a bypass of the gate hits a connection that cannot write anyway.

**Results are masked before they reach the model.** A ``SELECT * FROM users``
returns password hashes, API tokens and personal data, and every one of those
lands in a transcript that gets stored, compacted, summarized, and possibly
distilled into a skill. Columns whose names look like credentials are replaced
with a marker naming the column, so the agent can still reason about the shape
of the data without the values ever existing outside the database.

What this module deliberately does *not* do is make writes possible behind a
flag. A migration is a human's decision with a human's review, and an agent that
can run one is an agent that eventually does at 3am.
"""

from __future__ import annotations

import itertools
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

MAX_ROWS = 200
MAX_CELL_CHARS = 500
"""One base64 blob in one cell should not consume the context window."""

READ_ONLY_STARTS = frozenset({"select", "with", "explain", "values", "table", "pragma"})
"""Statements that may only read. ``table`` covers SQLite's ``TABLE t``
shorthand; ``pragma`` is admitted here so it reaches the narrower introspection
check below and gets a message that says what is actually wrong."""

FORBIDDEN = frozenset(
    {
        "insert", "update", "delete", "drop", "create", "alter", "truncate",
        "replace", "attach", "detach", "vacuum", "reindex", "grant", "revoke",
        "commit", "rollback", "savepoint", "begin", "call", "do", "copy",
        "merge", "upsert", "load_extension", "writefile", "readfile", "edit",
    }
)  # fmt: skip
"""Keywords that may not appear anywhere, not merely at the start.

A ``WITH x AS (DELETE ... RETURNING *) SELECT * FROM x`` reads as a SELECT to
anything that only inspects the first token, and on PostgreSQL it deletes rows."""

SENSITIVE_TOKENS = frozenset(
    {
        "password", "passwd", "passphrase", "pass", "secret", "token", "apikey",
        "privatekey", "credential", "credentials", "cvv", "ssn", "pin", "otp",
        "salt", "hash", "auth", "authorization", "bearer", "cookie", "session",
    }
)  # fmt: skip
"""Name parts whose values must not reach the model or the transcript.

Matched against the *parts* of a column name rather than as substrings, which is
what makes `password_hash` sensitive and `author` not. An anchored regex misses
the first; a substring search wrongly catches the second, and a masked byline
is the kind of false positive that gets masking switched off entirely."""

SENSITIVE_PAIRS = frozenset({("api", "key"), ("private", "key"), ("secret", "key")})
"""Adjacent parts that are sensitive together but innocuous apart: `key` alone
is usually a primary key."""

_SPLIT = re.compile(r"[^A-Za-z0-9]+|(?<=[a-z0-9])(?=[A-Z])")

MASK = "<redacted:{column}>"
_COMMENTS = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


class NotReadOnly(ValueError):
    """A statement was refused before it reached a database."""


@dataclass(frozen=True, slots=True)
class Rows:
    """A result set, already bounded and masked."""

    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    truncated: bool = False
    masked: tuple[str, ...] = ()
    """Columns whose values were replaced. Reported, never hidden -- an agent
    that does not know a column was masked will conclude the field is empty."""

    def render(self) -> str:
        """Format as a text table for a tool result."""
        if not self.columns:
            return "(no columns)"
        if not self.rows:
            return f"{' | '.join(self.columns)}\n(no rows)"
        widths = [
            max(len(self.columns[i]), *(len(row[i]) for row in self.rows))
            for i in range(len(self.columns))
        ]
        header = " | ".join(name.ljust(widths[i]) for i, name in enumerate(self.columns))
        divider = "-+-".join("-" * width for width in widths)
        body = "\n".join(
            " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in self.rows
        )
        notes = [f"{len(self.rows)} row(s)"]
        if self.truncated:
            notes.append(f"truncated at {len(self.rows)}")
        if self.masked:
            notes.append(f"masked columns: {', '.join(self.masked)}")
        return f"{header}\n{divider}\n{body}\n\n[{'; '.join(notes)}]"


class SqlSource(Protocol):
    """One queryable database. Kept narrow: describe, and read."""

    name: str

    def describe(self) -> str:
        """The schema, as text the model can reason about."""
        ...

    def query(self, sql: str, *, limit: int = MAX_ROWS) -> Rows:
        """Run a read-only statement."""
        ...


def assert_read_only(sql: str) -> str:
    """Return the statement if it only reads, else raise :class:`NotReadOnly`.

    Rejects multiple statements outright. ``SELECT 1; DROP TABLE users`` is the
    oldest trick there is, and no legitimate tool call needs a semicolon in the
    middle.
    """
    stripped = _COMMENTS.sub(" ", sql).strip().rstrip(";").strip()
    if not stripped:
        detail = "the statement is empty"
        raise NotReadOnly(detail)
    if ";" in stripped:
        detail = "only one statement may be run at a time"
        raise NotReadOnly(detail)

    words = [word.lower() for word in _WORD.findall(stripped)]
    if not words or words[0] not in READ_ONLY_STARTS:
        detail = (
            f"{words[0] if words else 'that'!r} is not a read-only statement; "
            f"this tool runs {', '.join(sorted(READ_ONLY_STARTS))} only"
        )
        raise NotReadOnly(detail)

    if offenders := sorted(set(words) & FORBIDDEN):
        # Checked across the whole statement, not just the first token: a CTE
        # can hide a DELETE inside something that starts with WITH.
        detail = f"{', '.join(offenders)} may not appear in a read-only query"
        raise NotReadOnly(detail)

    if "pragma" in words and not _is_introspection_pragma(stripped):
        detail = "only introspection pragmas are allowed"
        raise NotReadOnly(detail)
    return stripped


def is_sensitive(column: str) -> bool:
    """Whether a column's values must be withheld from the model."""
    parts = [part.lower() for part in _SPLIT.split(column) if part]
    if any(part in SENSITIVE_TOKENS for part in parts):
        return True
    return any(pair in SENSITIVE_PAIRS for pair in itertools.pairwise(parts))


def mask_row(
    columns: Sequence[str], values: Sequence[Any], *, max_chars: int = MAX_CELL_CHARS
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Render one row as text, masking columns that look like credentials."""
    cells: list[str] = []
    masked: list[str] = []
    for name, value in zip(columns, values, strict=False):
        if is_sensitive(name):
            cells.append(MASK.format(column=name))
            masked.append(name)
            continue
        cells.append(_cell(value, max_chars))
    return tuple(cells), tuple(masked)


@dataclass
class SqliteSource:
    """A SQLite file, opened read-only."""

    path: Path
    name: str = "sqlite"
    max_rows: int = MAX_ROWS

    def _connect(self) -> sqlite3.Connection:
        """Open the file read-only.

        The URI form is the second half of the enforcement: even if a statement
        slipped past :func:`assert_read_only`, the connection refuses to write.
        """
        uri = f"file:{self.path}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=5.0)
        connection.row_factory = sqlite3.Row
        return connection

    def describe(self) -> str:
        """List tables and their columns."""
        with self._connect() as connection:
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            lines: list[str] = []
            for table in tables:
                columns = connection.execute(f'PRAGMA table_info("{table["name"]}")').fetchall()
                described = ", ".join(f"{c['name']} {c['type']}" for c in columns)
                lines.append(f"{table['name']}({described})")
        return "\n".join(lines) or "(no tables)"

    def query(self, sql: str, *, limit: int = MAX_ROWS) -> Rows:
        """Run a read-only statement and return bounded, masked rows."""
        statement = assert_read_only(sql)
        cap = min(limit, self.max_rows)
        with self._connect() as connection:
            cursor = connection.execute(statement)
            columns = tuple(str(d[0]) for d in cursor.description or ())
            fetched = cursor.fetchmany(cap + 1)

        truncated = len(fetched) > cap
        masked: set[str] = set()
        rows: list[tuple[str, ...]] = []
        for raw in fetched[:cap]:
            cells, hidden = mask_row(columns, tuple(raw))
            rows.append(cells)
            masked.update(hidden)
        return Rows(
            columns=columns,
            rows=tuple(rows),
            truncated=truncated,
            masked=tuple(sorted(masked)),
        )


@dataclass
class StaticSource:
    """An in-memory source. Ships for tests and for demos."""

    tables: dict[str, tuple[tuple[str, ...], tuple[tuple[Any, ...], ...]]] = field(
        default_factory=dict
    )
    name: str = "static"
    queries: list[str] = field(default_factory=list)

    def describe(self) -> str:
        """List the registered tables."""
        return "\n".join(f"{name}({', '.join(cols)})" for name, (cols, _) in self.tables.items())

    def query(self, sql: str, *, limit: int = MAX_ROWS) -> Rows:
        """Return whichever table the statement names, masked."""
        statement = assert_read_only(sql)
        self.queries.append(statement)
        for name, (columns, values) in self.tables.items():
            if name.lower() in statement.lower():
                masked: set[str] = set()
                rows = []
                for raw in values[:limit]:
                    cells, hidden = mask_row(columns, raw)
                    rows.append(cells)
                    masked.update(hidden)
                return Rows(
                    columns=columns,
                    rows=tuple(rows),
                    truncated=len(values) > limit,
                    masked=tuple(sorted(masked)),
                )
        return Rows(columns=(), rows=())


def _is_introspection_pragma(statement: str) -> bool:
    """Whether a pragma only reads schema metadata."""
    lowered = statement.lower()
    return any(
        f"pragma {name}" in lowered
        for name in ("table_info", "table_list", "index_list", "index_info", "foreign_key_list")
    )


def _cell(value: Any, max_chars: int) -> str:
    """Render one value as bounded text."""
    if value is None:
        return "NULL"
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    text = str(value)
    if len(text) > max_chars:
        return text[:max_chars] + f"…(+{len(text) - max_chars})"
    return text
