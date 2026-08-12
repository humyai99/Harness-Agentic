"""A searchable local corpus, for answering from documents instead of memory.

This is the retrieval half of the framework, and it is deliberately FTS5 rather
than embeddings. The reasoning is not that vectors are bad -- it is that the
session store already has FTS5 with external-content tables, so full-text search
costs one more table and no new dependency, no model to call, no index to
rebuild, and no dimension to migrate. For a support bot answering from a few
hundred internal documents it is also, in practice, *better*: an exact match on a
product name or an error code beats cosine similarity, and it never returns three
confidently wrong neighbours.

The seam for embeddings is :class:`Retriever`. When a corpus grows past the point
where keyword search stops finding things, a second implementation goes behind
that protocol and nothing above it changes.

Chunking matters more than the search algorithm. A whole document returned for a
one-line question wastes context; a fragment cut mid-sentence loses the meaning
that made it a match. So chunks are paragraph-aligned with a small overlap, and
every one carries the heading it fell under -- because a chunk saying "set it to
false" is useless without the section that says what "it" is.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

TARGET_CHUNK_CHARS = 1200
OVERLAP_CHARS = 150
"""Enough that a sentence spanning a boundary survives in one of the halves."""
MIN_CHUNK_CHARS = 80
MAX_SNIPPET_CHARS = 600

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FTS_SPECIAL = re.compile(r'["*()^:]')


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable passage."""

    doc_id: str
    ordinal: int
    heading: str
    text: str

    @property
    def id(self) -> str:
        """A stable identifier, so re-indexing does not duplicate."""
        return f"{self.doc_id}#{self.ordinal}"

    def render(self) -> str:
        """The passage with the context needed to read it."""
        where = f"{self.doc_id}" + (f" -- {self.heading}" if self.heading else "")
        return f"[{where}]\n{self.text}"


@dataclass(frozen=True, slots=True)
class Passage:
    """A chunk that matched, with its score."""

    chunk: Chunk
    score: float
    snippet: str = ""

    def render(self) -> str:
        """Format for a tool result."""
        body = self.snippet or self.chunk.text
        where = self.chunk.doc_id + (f" -- {self.chunk.heading}" if self.chunk.heading else "")
        return f"[{where}]\n{body[:MAX_SNIPPET_CHARS]}"


class Retriever(Protocol):
    """What ``kb_search`` needs. FTS5 today, embeddings later if warranted."""

    def search(self, query: str, *, limit: int = 5) -> list[Passage]:
        """Find passages relevant to a query."""
        ...

    def count(self) -> int:
        """How many passages are indexed."""
        ...


def chunk_document(text: str, *, doc_id: str, target: int = TARGET_CHUNK_CHARS) -> list[Chunk]:
    """Split a document into paragraph-aligned, heading-tagged chunks."""
    chunks: list[Chunk] = []
    heading = ""
    buffer: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal buffer, size
        body = "\n\n".join(buffer).strip()
        if not body:
            buffer, size = [], 0
            return
        # A short fragment is folded into the previous chunk, but only when that
        # chunk is under the same heading. Folding across a heading boundary
        # would merge two sections into one passage, and then a search for a
        # term in the second returns a passage labelled with the first -- which
        # is worse than a short chunk, because the label is now a lie.
        previous = chunks[-1] if chunks else None
        if len(body) < MIN_CHUNK_CHARS and previous is not None and previous.heading == heading:
            chunks[-1] = Chunk(
                doc_id=previous.doc_id,
                ordinal=previous.ordinal,
                heading=previous.heading,
                text=f"{previous.text}\n\n{body}",
            )
        else:
            chunks.append(Chunk(doc_id=doc_id, ordinal=len(chunks), heading=heading, text=body))
        buffer, size = [], 0

    for block in _paragraphs(text):
        match = _HEADING.match(block)
        if match:
            # A heading starts a new chunk: the section it names is the unit a
            # reader would quote, and carrying it into every chunk beneath is
            # what makes those chunks answerable on their own.
            flush()
            heading = match.group(2)
            buffer.append(block)
            size += len(block)
            continue
        if size + len(block) > target and buffer:
            tail = _tail(buffer, OVERLAP_CHARS)
            flush()
            if tail:
                buffer, size = [tail], len(tail)
        buffer.append(block)
        size += len(block)
    flush()
    return chunks


@dataclass
class SqliteKnowledgeBase:
    """An FTS5 index over local documents."""

    path: Path
    name: str = "kb"
    _connection: sqlite3.Connection | None = field(default=None, init=False)

    def connect(self) -> sqlite3.Connection:
        """Open the index, creating the schema on first use."""
        if self._connection is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=5.0)
            connection.row_factory = sqlite3.Row
            connection.executescript(SCHEMA)
            self._connection = connection
        return self._connection

    def close(self) -> None:
        """Close the index."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def index(self, chunks: Iterable[Chunk]) -> int:
        """Add or replace passages. Returns how many were written."""
        connection = self.connect()
        written = 0
        with connection:
            for chunk in chunks:
                connection.execute(
                    "INSERT INTO chunks(id, doc_id, ordinal, heading, body, checksum) "
                    "VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                    "heading=excluded.heading, body=excluded.body, checksum=excluded.checksum",
                    (
                        chunk.id,
                        chunk.doc_id,
                        chunk.ordinal,
                        chunk.heading,
                        chunk.text,
                        _checksum(chunk.text),
                    ),
                )
                written += 1
        return written

    def index_file(self, path: Path, *, doc_id: str | None = None) -> int:
        """Index one text file."""
        text = path.read_text(encoding="utf-8", errors="replace")
        return self.index(chunk_document(text, doc_id=doc_id or path.name))

    def index_directory(self, root: Path, *, pattern: str = "**/*.md") -> int:
        """Index every matching file under a directory."""
        return sum(
            self.index_file(path, doc_id=str(path.relative_to(root)))
            for path in sorted(root.glob(pattern))
            if path.is_file()
        )

    def forget(self, doc_id: str) -> int:
        """Drop every passage from one document."""
        connection = self.connect()
        with connection:
            cursor = connection.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        return cursor.rowcount

    def search(self, query: str, *, limit: int = 5) -> list[Passage]:
        """Find passages matching a query."""
        prepared = prepare_query(query)
        if not prepared:
            return []
        connection = self.connect()
        rows = connection.execute(
            "SELECT c.doc_id, c.ordinal, c.heading, c.body, "
            "       snippet(chunks_fts, 1, '', '', '…', 24) AS excerpt, "
            "       bm25(chunks_fts) AS rank "
            "FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            (prepared, limit),
        ).fetchall()
        return [
            Passage(
                chunk=Chunk(
                    doc_id=row["doc_id"],
                    ordinal=row["ordinal"],
                    heading=row["heading"],
                    text=row["body"],
                ),
                # bm25 returns smaller-is-better; flipped so callers can sort
                # descending like every other scoring API.
                score=-float(row["rank"]),
                snippet=row["excerpt"],
            )
            for row in rows
        ]

    def count(self) -> int:
        """How many passages are indexed."""
        row = self.connect().execute("SELECT count(*) AS n FROM chunks").fetchone()
        return int(row["n"])

    def documents(self) -> list[tuple[str, int]]:
        """Each document and how many passages it contributed."""
        rows = (
            self.connect()
            .execute("SELECT doc_id, count(*) AS n FROM chunks GROUP BY doc_id ORDER BY doc_id")
            .fetchall()
        )
        return [(row["doc_id"], int(row["n"])) for row in rows]


@dataclass
class StaticKnowledgeBase:
    """An in-memory retriever. Ships for tests and small corpora."""

    passages: list[Chunk] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)

    def search(self, query: str, *, limit: int = 5) -> list[Passage]:
        """Score by how many query words a passage contains."""
        self.queries.append(query)
        words = {w for w in re.findall(r"[\w]+", query.lower()) if len(w) > 2}  # noqa: PLR2004
        scored: list[Passage] = []
        for chunk in self.passages:
            haystack = f"{chunk.heading} {chunk.text}".lower()
            overlap = sum(1 for word in words if word in haystack)
            if overlap:
                scored.append(Passage(chunk=chunk, score=float(overlap)))
        return sorted(scored, key=lambda p: -p.score)[:limit]

    def count(self) -> int:
        """How many passages are held."""
        return len(self.passages)


def prepare_query(query: str) -> str:
    """Turn a natural-language question into an FTS5 match expression.

    FTS5's query syntax is not free text: an unquoted ``"`` or ``*`` is a syntax
    error, and a question mark makes it complain. A model asking "what's the
    retry policy?" should get results, not an exception -- so the punctuation is
    stripped and the terms are OR-ed, since requiring every word of a question
    matches nothing.
    """
    cleaned = _FTS_SPECIAL.sub(" ", query)
    terms = [term for term in re.findall(r"[\w']+", cleaned) if len(term) > 1]
    if not terms:
        return ""
    return " OR ".join(f'"{term}"' for term in terms)


SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    rowid    INTEGER PRIMARY KEY,
    id       TEXT NOT NULL UNIQUE,
    doc_id   TEXT NOT NULL,
    ordinal  INTEGER NOT NULL,
    heading  TEXT NOT NULL DEFAULT '',
    body     TEXT NOT NULL,
    checksum TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_by_doc ON chunks(doc_id, ordinal);

-- External content: the text lives in `chunks` and the index only points at
-- it, so a corpus is not stored twice.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    heading, body, content='chunks', content_rowid='rowid', tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, heading, body) VALUES (new.rowid, new.heading, new.body);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, heading, body)
        VALUES ('delete', old.rowid, old.heading, old.body);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, heading, body)
        VALUES ('delete', old.rowid, old.heading, old.body);
    INSERT INTO chunks_fts(rowid, heading, body) VALUES (new.rowid, new.heading, new.body);
END;
"""


def _paragraphs(text: str) -> Iterator[str]:
    """Split on blank lines, keeping headings as their own blocks."""
    for raw in re.split(r"\n\s*\n", text):
        block = raw.strip()
        if not block:
            continue
        lines = block.splitlines()
        if len(lines) > 1 and _HEADING.match(lines[0]):
            yield lines[0].strip()
            rest = "\n".join(lines[1:]).strip()
            if rest:
                yield rest
            continue
        yield block


def _tail(blocks: Sequence[str], chars: int) -> str:
    """The last ``chars`` characters of a chunk, for overlap."""
    joined = "\n\n".join(blocks)
    if len(joined) <= chars:
        return ""
    return joined[-chars:].lstrip()


def _checksum(text: str) -> str:
    """A content hash, so re-indexing an unchanged file is detectable."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]
