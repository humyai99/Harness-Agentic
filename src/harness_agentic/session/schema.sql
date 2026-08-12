-- Session storage.
--
-- One session holds its whole history, including the parts compaction has
-- hidden. Hermes forks a child session per compaction and threads a lineage
-- pointer; we set `visible = 0` instead. That buys three things: resume is a
-- single query rather than a walk, compaction is reversible, and full-text
-- search runs over the *original* text rather than over summaries of it.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS state_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id               TEXT PRIMARY KEY,
    source           TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    cwd              TEXT NOT NULL,
    workspace_key    TEXT NOT NULL,
    model            TEXT NOT NULL,
    title            TEXT,
    -- Subagent lineage only. Compaction never forks.
    parent_id        TEXT REFERENCES sessions(id) ON DELETE SET NULL,
    compaction_count INTEGER NOT NULL DEFAULT 0,
    total_usage      TEXT NOT NULL DEFAULT '{}',
    archived         INTEGER NOT NULL DEFAULT 0,
    meta             TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS sessions_by_workspace
    ON sessions(workspace_key, updated_at DESC);
CREATE INDEX IF NOT EXISTS sessions_by_source
    ON sessions(source, updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    session_id    TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seq           INTEGER NOT NULL,
    role          TEXT NOT NULL,
    blocks        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    usage         TEXT,
    -- 0 once compaction has replaced this message with a summary. The row
    -- stays, so `sessions show --raw` and `undo_compaction` both work.
    visible       INTEGER NOT NULL DEFAULT 1,
    -- On a summary row, the inclusive seq range it stands in for.
    supersedes_lo INTEGER,
    supersedes_hi INTEGER,
    -- Denormalized for FTS: the searchable text of this message.
    search_text   TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (session_id, seq)
);

CREATE INDEX IF NOT EXISTS messages_visible
    ON messages(session_id, visible, seq);

-- External-content FTS5 over `messages`. Only text blocks and tool *names*
-- are indexed -- tool results are ~90% of the bytes and ~5% of the search
-- value, and indexing them makes the database several times larger for
-- worse results.
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    search_text,
    session_id UNINDEXED,
    seq        UNINDEXED,
    content    = 'messages',
    content_rowid = 'rowid',
    tokenize   = 'unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, search_text, session_id, seq)
    VALUES (new.rowid, new.search_text, new.session_id, new.seq);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, search_text, session_id, seq)
    VALUES ('delete', old.rowid, old.search_text, old.session_id, old.seq);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, search_text, session_id, seq)
    VALUES ('delete', old.rowid, old.search_text, old.session_id, old.seq);
    INSERT INTO messages_fts(rowid, search_text, session_id, seq)
    VALUES (new.rowid, new.search_text, new.session_id, new.seq);
END;

CREATE TABLE IF NOT EXISTS compactions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    from_seq     INTEGER NOT NULL,
    to_seq       INTEGER NOT NULL,
    summary_seq  INTEGER NOT NULL,
    created_at   TEXT NOT NULL,
    undone       INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS compactions_by_session
    ON compactions(session_id, id DESC);
