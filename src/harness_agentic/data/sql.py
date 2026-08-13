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

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9.$]*|,|\(|\)|\*")
_RENAMES_A_TABLE = frozenset({"from", "join"})
"""``FROM audit_tokens AS t`` renames a table, not a column, and its values still
arrive under their own names."""
_BEFORE_THE_SOURCE = 2
"""``FROM``/``JOIN`` sits two tokens before the ``AS`` that renames a table:
``from``, the table, then ``as``."""


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

    if renamed := aliased_secret(stripped):
        source, alias = renamed
        detail = (
            f"{source} may not be renamed to {alias}: masking works on the name a "
            f"column comes back under, so an alias would return the value in full. "
            f"Select it as {source} and it will come back masked."
        )
        raise NotReadOnly(detail)
    return stripped


def aliased_secret(statement: str) -> tuple[str, str] | None:
    """A sensitive column renamed to something innocuous, if the statement does that.

    Masking keys on the name a column *comes back* under, and an alias is
    exactly the thing that controls that name -- so ``SELECT password_hash AS
    notes`` returns the hash in full while ``SELECT password_hash`` returns a
    marker. The values are what must never reach a transcript, so a rename has
    to be refused rather than trusted.

    Works on select *items* rather than on the ``AS`` keyword, because the
    keyword is optional in every dialect this targets. Checking for ``as``
    caught ``SELECT password_hash AS notes`` and missed ``SELECT password_hash
    notes``, which is the same rename with one word deleted and returned the
    hash in full -- into the model, the transcript, the session store, and
    anything later distilled from them. A quoted alias, ``password_hash
    "notes"``, went the same way.

    Refused rather than masked-anyway: which output column a renamed expression
    corresponds to is exactly what cannot be determined without parsing the
    dialect, and a query answered with everything masked teaches the model that
    the data is empty. Refusing says what to do instead.
    """
    for item in _select_items(_TOKEN.findall(statement)):
        alias = _alias_of(item)
        if alias is None or is_sensitive(alias):
            # No rename, or renamed to something ordinary masking still catches.
            continue
        for word in item[:-1]:
            column = word.rsplit(".", 1)[-1]
            if _is_name(word) and is_sensitive(column):
                return column, alias
    return None


def _select_items(tokens: Sequence[str]) -> list[list[str]]:
    """Every comma-separated item of every select list in the statement.

    Scoped to select lists so that ``FROM users u`` -- a table alias, whose
    columns still arrive under their own names -- is never mistaken for a
    column rename. Subqueries and CTEs get their own lists, which is where a
    rename would otherwise hide.
    """
    items: list[list[str]] = []
    index = 0
    while index < len(tokens):
        if tokens[index].lower() != "select":
            index += 1
            continue
        index += 1
        depth = 0
        current: list[str] = []
        while index < len(tokens):
            word = tokens[index]
            lowered = word.lower()
            if word == "(":
                depth += 1
            elif word == ")":
                if depth == 0:
                    break  # the subquery this select sits in has closed
                depth -= 1
            elif depth == 0 and lowered in _ENDS_A_SELECT_LIST:
                break
            elif depth == 0 and lowered == "select":
                break  # a nested select; the outer loop will pick it up
            elif depth == 0 and word == ",":
                items.append(current)
                current = []
                index += 1
                continue
            current.append(word)
            index += 1
        items.append(current)
    return [item for item in items if item]


_ENDS_A_SELECT_LIST = frozenset(
    {"from", "where", "group", "order", "having", "limit", "offset", "union", "intersect", "except"}
)
"""Words that close a select list. Reached only at paren depth zero, so a
``CAST(x AS text)`` or a scalar subquery inside an item does not end it."""


def _is_name(word: str) -> bool:
    """Whether a word is an identifier rather than punctuation."""
    return bool(word) and (word[0].isalpha() or word[0] == "_")


_ALIASABLE_TAIL = 2
"""An implicit alias needs the aliased thing in front of it."""


def _alias_of(item: Sequence[str]) -> str | None:
    """The name one select item comes back under, when it renames itself.

    ``None`` when the item is a bare column or an unaliased expression: the
    driver names those after the expression text, so the sensitive part appears
    in the name and ordinary masking already covers them.
    """
    lowered = [word.lower() for word in item]
    depth = 0
    for position, word in enumerate(item):
        if word == "(":
            depth += 1
        elif word == ")":
            depth -= 1
        elif depth == 0 and lowered[position] == "as" and position + 1 < len(item):
            # Not a CAST's `as`, which sits inside parentheses.
            return item[position + 1]
    if len(item) < _ALIASABLE_TAIL or not _is_name(item[-1]):
        return None
    # `password_hash notes` and `substr(password_hash,1,3) frag`: the keyword is
    # optional, and leaving it out renames the column just the same.
    if _is_name(item[-2]) or item[-2] == ")":
        return item[-1]
    return None


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
