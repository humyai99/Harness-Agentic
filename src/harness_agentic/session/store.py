"""Session persistence.

``SessionStore`` is the Protocol; ``JsonlSessionStore`` is the M1 implementation
and SQLite with full-text search replaces it later. Getting the Protocol right
now is what makes that swap uneventful -- the loop is written against the
interface and never learns which one it has.

Append-only, one JSON object per line, flushed after every message. That is not
a placeholder detail: the agent has to be resumable after a ``kill -9``, and a
transcript written only at the end of a turn loses exactly the long, expensive
turn you most wanted back.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from harness_agentic.core.types import Message, Usage
from harness_agentic.session.serde import (
    message_from_json,
    message_to_json,
    usage_from_json,
    usage_to_json,
)


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One match from a history search.

    Richer than returning whole messages: a hit is shown as a snippet with its
    position, and loading the full message only matters once someone asks for
    it.
    """

    session_id: str
    seq: int
    role: str
    snippet: str
    created_at: datetime


@dataclass(slots=True)
class SessionRecord:
    """Metadata for one conversation."""

    id: str
    source: str
    """Which surface started it: cli, telegram, cron, web, voice."""
    created_at: datetime
    updated_at: datetime
    cwd: str
    workspace_key: str
    """Git root or resolved cwd. Used to find the right session to resume."""
    model: str
    title: str | None = None
    parent_id: str | None = None
    """Subagent lineage only -- compaction stays inside one session."""
    compaction_count: int = 0
    total_usage: Usage = field(default_factory=Usage)
    archived: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


class SessionStore(Protocol):
    """Where conversations live."""

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
        ...

    def get(self, session_id: str) -> SessionRecord | None:
        """Look up one session."""
        ...

    def latest(
        self, *, workspace_key: str | None = None, source: str | None = None
    ) -> SessionRecord | None:
        """The most recently updated session matching the filters."""
        ...

    def recent(self, *, limit: int = 50, workspace_key: str | None = None) -> list[SessionRecord]:
        """Recent sessions, newest first."""
        ...

    def update(self, session_id: str, **fields: Any) -> None:
        """Patch session metadata."""
        ...

    def append(
        self, session_id: str, messages: Sequence[Message], *, usage: Usage | None = None
    ) -> None:
        """Persist messages. Must be durable before returning."""
        ...

    def history(self, session_id: str) -> list[Message]:
        """Every visible message, in order."""
        ...

    def search(
        self, query: str, *, session_id: str | None = None, limit: int = 20
    ) -> list[SearchHit]:
        """Find past messages."""
        ...

    @contextmanager
    def turn_lease(self, session_id: str, *, timeout_s: float = 30.0) -> Iterator[None]:
        """Hold the single-writer lease for one session."""
        ...


class JsonlSessionStore:
    """Sessions as append-only JSONL files under the profile directory."""

    def __init__(self, root: Path) -> None:
        """Store sessions beneath ``root``."""
        self._root = root
        self._root.mkdir(mode=0o700, parents=True, exist_ok=True)

    # -- layout -------------------------------------------------------------

    def _dir(self, session_id: str) -> Path:
        return self._root / session_id

    def _meta_path(self, session_id: str) -> Path:
        return self._dir(session_id) / "session.json"

    def _log_path(self, session_id: str) -> Path:
        return self._dir(session_id) / "messages.jsonl"

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
        """Start a new session and write its metadata immediately."""
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
        self._dir(record.id).mkdir(mode=0o700, parents=True, exist_ok=True)
        self._write_meta(record)
        return record

    def get(self, session_id: str) -> SessionRecord | None:
        """Load one session's metadata."""
        path = self._meta_path(session_id)
        if not path.exists():
            return None
        return _record_from_json(json.loads(path.read_text(encoding="utf-8")))

    def recent(self, *, limit: int = 50, workspace_key: str | None = None) -> list[SessionRecord]:
        """Recent sessions, newest first."""
        records: list[SessionRecord] = []
        for child in self._root.iterdir():
            if not child.is_dir():
                continue
            record = self.get(child.name)
            if record is None or record.archived:
                continue
            if workspace_key is not None and record.workspace_key != workspace_key:
                continue
            records.append(record)
        records.sort(key=lambda r: r.updated_at, reverse=True)
        return records[:limit]

    def latest(
        self, *, workspace_key: str | None = None, source: str | None = None
    ) -> SessionRecord | None:
        """The session ``--continue`` should resume."""
        for record in self.recent(limit=200, workspace_key=workspace_key):
            if source is None or record.source == source:
                return record
        return None

    def update(self, session_id: str, **fields: Any) -> None:
        """Patch metadata, always refreshing ``updated_at``."""
        record = self.get(session_id)
        if record is None:
            return
        fields.setdefault("updated_at", datetime.now(UTC))
        self._write_meta(replace(record, **fields))

    def _write_meta(self, record: SessionRecord) -> None:
        payload = {
            "id": record.id,
            "source": record.source,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
            "cwd": record.cwd,
            "workspace_key": record.workspace_key,
            "model": record.model,
            "title": record.title,
            "parent_id": record.parent_id,
            "compaction_count": record.compaction_count,
            "total_usage": usage_to_json(record.total_usage),
            "archived": record.archived,
            "meta": record.meta,
        }
        _atomic_write(self._meta_path(record.id), json.dumps(payload, ensure_ascii=False, indent=2))

    # -- messages -----------------------------------------------------------

    def append(
        self, session_id: str, messages: Sequence[Message], *, usage: Usage | None = None
    ) -> None:
        """Append messages, fsyncing before returning.

        The fsync is the point. Without it a crash loses the tail of the
        transcript, and the tail is where the tool results the model was about
        to act on live -- resuming without them produces a history with
        unanswered calls, which the sanitizer then has to paper over.
        """
        if not messages:
            return
        self._dir(session_id).mkdir(mode=0o700, parents=True, exist_ok=True)
        lines = [
            json.dumps(
                message_to_json(message, usage=usage if index == 0 else None),
                ensure_ascii=False,
            )
            for index, message in enumerate(messages)
        ]
        with self._log_path(session_id).open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.update(session_id)

    def history(self, session_id: str) -> list[Message]:
        """Read the whole transcript.

        A malformed trailing line is skipped rather than fatal: a crash during
        a write leaves a partial line, and refusing to resume over one truncated
        record would throw away the entire conversation.
        """
        path = self._log_path(session_id)
        if not path.exists():
            return []
        messages: list[Message] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                messages.append(message_from_json(json.loads(line)))
            except (ValueError, TypeError, KeyError):
                continue
        return messages

    def usage_total(self, session_id: str) -> Usage:
        """Sum the usage records in a transcript."""
        path = self._log_path(session_id)
        if not path.exists():
            return Usage()
        total = Usage()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if "usage" in payload:
                total = total + usage_from_json(payload["usage"])
        return total

    def search(
        self, query: str, *, session_id: str | None = None, limit: int = 20
    ) -> list[SearchHit]:
        """Substring search across transcripts.

        Deliberately naive; the SQLite store does this properly with FTS5. It
        exists so the Protocol has two implementations from the start, which is
        what keeps the Protocol honest.
        """
        needle = query.lower()
        hits: list[SearchHit] = []
        for record in self.recent(limit=200):
            if session_id is not None and record.id != session_id:
                continue
            for seq, message in enumerate(self.history(record.id)):
                text = message.text()
                if needle in text.lower():
                    hits.append(
                        SearchHit(
                            session_id=record.id,
                            seq=seq,
                            role=message.role,
                            snippet=text[:200],
                            created_at=message.created_at,
                        )
                    )
                    if len(hits) >= limit:
                        return hits
        return hits

    # -- concurrency --------------------------------------------------------

    @contextmanager
    def turn_lease(self, session_id: str, *, timeout_s: float = 30.0) -> Iterator[None]:
        """Serialize writers for one session.

        A best-effort lock file here; the SQLite store makes it real. It exists
        now so the loop is written against it from the start -- retrofitting a
        lease into a loop that assumed it was alone is much harder than having
        one that turns out to be stronger later.
        """
        lock = self._dir(session_id) / ".lease"
        lock.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        deadline = time.monotonic() + timeout_s
        acquired = False
        while time.monotonic() < deadline:
            try:
                handle = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                time.sleep(0.05)
                continue
            os.close(handle)
            acquired = True
            break
        try:
            yield
        finally:
            if acquired:
                lock.unlink(missing_ok=True)


def workspace_key_for(cwd: Path) -> str:
    """Identify the workspace a session belongs to.

    The git root when there is one, so sessions started from a subdirectory
    still resume together -- running ``harn --continue`` from ``src/`` should
    find the conversation you started at the repository root.
    """
    current = cwd.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return str(candidate)
    return str(current)


def _record_from_json(raw: Mapping[str, Any]) -> SessionRecord:
    """Rebuild a session record from stored metadata."""
    return SessionRecord(
        id=str(raw["id"]),
        source=str(raw.get("source", "cli")),
        created_at=datetime.fromisoformat(str(raw["created_at"])),
        updated_at=datetime.fromisoformat(str(raw["updated_at"])),
        cwd=str(raw.get("cwd", "")),
        workspace_key=str(raw.get("workspace_key", "")),
        model=str(raw.get("model", "")),
        title=raw.get("title"),
        parent_id=raw.get("parent_id"),
        compaction_count=int(raw.get("compaction_count", 0)),
        total_usage=usage_from_json(raw.get("total_usage")),
        archived=bool(raw.get("archived", False)),
        meta=dict(raw.get("meta") or {}),
    )


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temporary file and rename, so readers never see a partial."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
