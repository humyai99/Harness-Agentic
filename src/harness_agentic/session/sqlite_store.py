"""SQLite-backed sessions with full-text search.

Replaces the JSONL store behind the same Protocol, which is the whole reason
that Protocol exists -- the loop never learns which one it has.

Two design points are load-bearing:

**Compaction hides, it does not delete.** A superseded message keeps its row
with ``visible = 0`` and the summary records the range it replaced. Resume
stays one query, ``undo_compaction`` is real, and search runs over the original
text rather than over a summary of it. The alternative -- forking a child
session per compaction -- makes resume a graph walk and undo a surgery.

**Search skips tool results.** They are the overwhelming majority of the bytes
and almost none of the value: nobody searches their history for the contents of
a file they had the agent read.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness_agentic.core.types import (
    Message,
    TextBlock,
    ToolUseBlock,
    Usage,
)
from harness_agentic.errors import CompactionDeferred
from harness_agentic.session.serde import (
    block_from_json,
    block_to_json,
    usage_from_json,
    usage_to_json,
)
from harness_agentic.session.store import SearchHit, SessionRecord, workspace_key_for

SCHEMA_VERSION = 1
_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


@dataclass(frozen=True, slots=True)
class StoredMessage:
    """A message plus the storage metadata the loop does not see."""

    message: Message
    seq: int
    visible: bool
    supersedes: tuple[int, int] | None = None


class SqliteSessionStore:
    """Sessions in SQLite, with FTS5 search and reversible compaction."""

    def __init__(self, path: Path) -> None:
        """Open or create the database at ``path``."""
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        self._migrate()

    # -- schema -------------------------------------------------------------

    def _migrate(self) -> None:
        """Create or verify the schema.

        Forward-only. A database written by a newer build is refused rather
        than opened optimistically -- silently reading a schema you do not
        understand is how a transcript gets corrupted.
        """
        with self._lock:
            self._conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
            row = self._conn.execute(
                "SELECT value FROM state_meta WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO state_meta(key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
                self._conn.commit()
                return
            found = int(row["value"])
            if found > SCHEMA_VERSION:
                msg = (
                    f"{self._path} was written by a newer version "
                    f"(schema {found}, this build understands {SCHEMA_VERSION})"
                )
                raise RuntimeError(msg)

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """Run one short write transaction."""
        with self._lock, self._conn:
            yield self._conn

    # -- sessions -----------------------------------------------------------

    def create(
        self,
        *,
        source: str,
        cwd: Path,
        model: str,
        workspace_key: str | None = None,
        parent_id: str | None = None,
    ) -> SessionRecord:
        """Start a new session."""
        now = datetime.now(UTC)
        record = SessionRecord(
            id=uuid.uuid4().hex[:16],
            source=source,
            created_at=now,
            updated_at=now,
            cwd=str(cwd),
            workspace_key=workspace_key or workspace_key_for(cwd),
            model=model,
            parent_id=parent_id,
        )
        with self._write() as conn:
            conn.execute(
                """
                INSERT INTO sessions
                    (id, source, created_at, updated_at, cwd, workspace_key,
                     model, parent_id, total_usage)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.source,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    record.cwd,
                    record.workspace_key,
                    record.model,
                    record.parent_id,
                    json.dumps(usage_to_json(record.total_usage)),
                ),
            )
        return record

    def get(self, session_id: str) -> SessionRecord | None:
        """Look up one session."""
        row = self._conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return _record(row) if row else None

    def recent(self, *, limit: int = 50, workspace_key: str | None = None) -> list[SessionRecord]:
        """Recent sessions, newest first."""
        if workspace_key is None:
            rows = self._conn.execute(
                "SELECT * FROM sessions WHERE archived = 0 ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM sessions WHERE archived = 0 AND workspace_key = ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (workspace_key, limit),
            ).fetchall()
        return [_record(row) for row in rows]

    def latest(
        self, *, workspace_key: str | None = None, source: str | None = None
    ) -> SessionRecord | None:
        """The session ``--continue`` should resume."""
        for record in self.recent(limit=200, workspace_key=workspace_key):
            if source is None or record.source == source:
                return record
        return None

    def update(self, session_id: str, **fields: Any) -> None:
        """Patch session metadata."""
        allowed = {
            "model",
            "title",
            "compaction_count",
            "total_usage",
            "archived",
            "updated_at",
        }
        assignments: list[str] = []
        values: list[Any] = []
        for key, value in fields.items():
            if key not in allowed:
                continue
            assignments.append(f"{key} = ?")
            values.append(_encode_field(key, value))
        assignments.append("updated_at = ?")
        values.append(
            fields.get("updated_at", datetime.now(UTC)).isoformat()
            if isinstance(fields.get("updated_at"), datetime)
            else datetime.now(UTC).isoformat()
        )
        values.append(session_id)
        with self._write() as conn:
            conn.execute(
                f"UPDATE sessions SET {', '.join(assignments)} WHERE id = ?",  # noqa: S608
                values,
            )

    # -- messages -----------------------------------------------------------

    def append(
        self, session_id: str, messages: Sequence[Message], *, usage: Usage | None = None
    ) -> None:
        """Persist messages inside one transaction."""
        if not messages:
            return
        with self._write() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), -1) AS s FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            seq = int(row["s"]) + 1
            for index, message in enumerate(messages):
                conn.execute(
                    """
                    INSERT INTO messages
                        (session_id, seq, role, blocks, created_at, usage, search_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        seq + index,
                        message.role,
                        json.dumps([block_to_json(b) for b in message.content]),
                        message.created_at.isoformat(),
                        json.dumps(usage_to_json(usage)) if usage and index == 0 else None,
                        _searchable(message),
                    ),
                )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (datetime.now(UTC).isoformat(), session_id),
            )

    def history(self, session_id: str, *, include_hidden: bool = False) -> list[Message]:
        """The conversation as the model should see it."""
        return [s.message for s in self.stored(session_id, include_hidden=include_hidden)]

    def stored(self, session_id: str, *, include_hidden: bool = False) -> list[StoredMessage]:
        """The conversation with its storage metadata."""
        clause = "" if include_hidden else "AND visible = 1"
        # A summary is appended at the end of the table but belongs where the
        # range it replaced used to be. Ordering by supersedes_lo when present
        # puts it back in chronological position without renumbering anything
        # -- renumbering would invalidate every recorded compaction range.
        rows = self._conn.execute(
            f"SELECT * FROM messages WHERE session_id = ? {clause} "  # noqa: S608
            "ORDER BY COALESCE(supersedes_lo, seq), seq",
            (session_id,),
        ).fetchall()
        return [_stored(row) for row in rows]

    def usage_total(self, session_id: str) -> Usage:
        """Sum every recorded usage row for a session."""
        rows = self._conn.execute(
            "SELECT usage FROM messages WHERE session_id = ? AND usage IS NOT NULL",
            (session_id,),
        ).fetchall()
        total = Usage()
        for row in rows:
            total = total + usage_from_json(json.loads(row["usage"]))
        return total

    # -- compaction ---------------------------------------------------------

    def record_compaction(
        self, session_id: str, *, replaced: tuple[int, int], summary: Message
    ) -> int:
        """Hide a range and insert the summary that stands in for it."""
        low, high = replaced
        with self._write() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), -1) AS s FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            summary_seq = int(row["s"]) + 1
            conn.execute(
                """
                INSERT INTO messages
                    (session_id, seq, role, blocks, created_at, search_text,
                     supersedes_lo, supersedes_hi)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    summary_seq,
                    summary.role,
                    json.dumps([block_to_json(b) for b in summary.content]),
                    summary.created_at.isoformat(),
                    _searchable(summary),
                    low,
                    high,
                ),
            )
            conn.execute(
                "UPDATE messages SET visible = 0 WHERE session_id = ? AND seq BETWEEN ? AND ?",
                (session_id, low, high),
            )
            conn.execute(
                """
                INSERT INTO compactions
                    (session_id, from_seq, to_seq, summary_seq, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, low, high, summary_seq, datetime.now(UTC).isoformat()),
            )
            conn.execute(
                "UPDATE sessions SET compaction_count = compaction_count + 1 WHERE id = ?",
                (session_id,),
            )
        return summary_seq

    def undo_compaction(self, session_id: str) -> bool:
        """Reverse the most recent compaction. False when there is none."""
        with self._write() as conn:
            row = conn.execute(
                "SELECT * FROM compactions WHERE session_id = ? AND undone = 0 "
                "ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            if row is None:
                return False
            conn.execute(
                "UPDATE messages SET visible = 1 WHERE session_id = ? AND seq BETWEEN ? AND ?",
                (session_id, row["from_seq"], row["to_seq"]),
            )
            conn.execute(
                "UPDATE messages SET visible = 0 WHERE session_id = ? AND seq = ?",
                (session_id, row["summary_seq"]),
            )
            conn.execute("UPDATE compactions SET undone = 1 WHERE id = ?", (row["id"],))
            conn.execute(
                "UPDATE sessions SET compaction_count = MAX(0, compaction_count - 1) WHERE id = ?",
                (session_id,),
            )
        return True

    # -- search -------------------------------------------------------------

    def search(
        self, query: str, *, session_id: str | None = None, limit: int = 20
    ) -> list[SearchHit]:
        """Full-text search across transcripts.

        User input is passed to FTS5 as a phrase rather than as a query
        expression: a stray quote or ``NEAR`` in someone's question should be
        text to look for, not syntax that raises.
        """
        phrase = '"' + query.replace('"', '""') + '"'
        sql = """
            SELECT m.session_id, m.seq, m.role, m.created_at,
                   snippet(messages_fts, 0, '[', ']', '…', 12) AS snippet
            FROM messages_fts f
            JOIN messages m ON m.rowid = f.rowid
            WHERE messages_fts MATCH ?
        """
        params: list[Any] = [phrase]
        if session_id is not None:
            sql += " AND m.session_id = ?"
            params.append(session_id)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []
        return [
            SearchHit(
                session_id=row["session_id"],
                seq=int(row["seq"]),
                role=row["role"],
                snippet=row["snippet"] or "",
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    # -- concurrency --------------------------------------------------------

    @contextmanager
    def turn_lease(self, session_id: str, *, timeout_s: float = 30.0) -> Iterator[None]:
        """Serialize writers for one session.

        Contention raises :class:`~harness_agentic.errors.CompactionDeferred`,
        which is soft. The CLI, the gateway and cron all write to the same
        database, and treating a busy session as a failure would mean throwing
        away a conversation because two surfaces spoke at once.
        """
        key = f"lease:{session_id}"
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                with self._write() as conn:
                    conn.execute(
                        "INSERT INTO state_meta(key, value) VALUES (?, ?)",
                        (key, datetime.now(UTC).isoformat()),
                    )
                break
            except sqlite3.IntegrityError:
                if time.monotonic() >= deadline:
                    msg = f"another writer holds session {session_id}"
                    raise CompactionDeferred(msg) from None
                time.sleep(0.05)
        try:
            yield
        finally:
            with suppress(sqlite3.Error), self._write() as conn:
                conn.execute("DELETE FROM state_meta WHERE key = ?", (key,))

    def close(self) -> None:
        """Close the connection."""
        with self._lock:
            self._conn.close()


# -- helpers -------------------------------------------------------------------


def _searchable(message: Message) -> str:
    """The text FTS should index for one message.

    Text blocks and tool *names* only. Tool results are excluded on purpose --
    they dominate the bytes and contribute almost nothing anyone searches for.
    """
    parts: list[str] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            parts.append(block.text)
        elif isinstance(block, ToolUseBlock):
            parts.append(block.name)
    return "\n".join(p for p in parts if p)


def _record(row: sqlite3.Row) -> SessionRecord:
    """Rebuild a session record from a row."""
    return SessionRecord(
        id=row["id"],
        source=row["source"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        cwd=row["cwd"],
        workspace_key=row["workspace_key"],
        model=row["model"],
        title=row["title"],
        parent_id=row["parent_id"],
        compaction_count=int(row["compaction_count"]),
        total_usage=usage_from_json(json.loads(row["total_usage"] or "{}")),
        archived=bool(row["archived"]),
        meta=json.loads(row["meta"] or "{}"),
    )


def _stored(row: sqlite3.Row) -> StoredMessage:
    """Rebuild a message and its storage metadata from a row."""
    blocks = [b for b in (block_from_json(x) for x in json.loads(row["blocks"])) if b]
    supersedes = None
    if row["supersedes_lo"] is not None:
        supersedes = (int(row["supersedes_lo"]), int(row["supersedes_hi"]))
    return StoredMessage(
        message=Message(
            role=row["role"],
            content=tuple(blocks),
            created_at=datetime.fromisoformat(row["created_at"]),
        ),
        seq=int(row["seq"]),
        visible=bool(row["visible"]),
        supersedes=supersedes,
    )


def _encode_field(key: str, value: Any) -> Any:
    """Serialize one updatable session field."""
    if key == "total_usage" and isinstance(value, Usage):
        return json.dumps(usage_to_json(value))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bool):
        return int(value)
    return value
