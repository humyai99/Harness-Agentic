"""Read-only SQL, credential masking, and the knowledge base.

The SQL tests are about what must be *refused*. A read-only gate that only
inspects the first token is not a gate -- `WITH x AS (DELETE ... RETURNING *)
SELECT * FROM x` reads as a SELECT to one and deletes rows on PostgreSQL -- and
a query tool that returns password hashes puts them in a transcript that gets
stored, compacted, and possibly distilled into a skill.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from harness_agentic.data.kb import (
    Chunk,
    SqliteKnowledgeBase,
    StaticKnowledgeBase,
    chunk_document,
    prepare_query,
)
from harness_agentic.data.sql import NotReadOnly, SqliteSource, StaticSource, assert_read_only

# -- the read-only gate ----------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "select id, name from users where active = 1",
        "WITH recent AS (SELECT * FROM orders) SELECT count(*) FROM recent",
        "EXPLAIN SELECT * FROM t",
        "VALUES (1, 2)",
        "SELECT 1;",
        "-- a comment\nSELECT 1",
    ],
)
def test_read_only_statements_pass(sql: str) -> None:
    assert assert_read_only(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM users",
        "UPDATE users SET admin = 1",
        "INSERT INTO users VALUES (1)",
        "DROP TABLE users",
        "ALTER TABLE users ADD COLUMN x INT",
        "CREATE TABLE t (id INT)",
        "ATTACH DATABASE '/tmp/x.db' AS other",
        "VACUUM",
        "TRUNCATE users",
    ],
)
def test_writing_statements_are_refused(sql: str) -> None:
    with pytest.raises(NotReadOnly):
        assert_read_only(sql)


def test_a_second_statement_is_refused() -> None:
    # The oldest trick there is, and no legitimate tool call needs it.
    with pytest.raises(NotReadOnly, match="one statement"):
        assert_read_only("SELECT 1; DROP TABLE users")


def test_a_write_hidden_in_a_cte_is_refused() -> None:
    # Reads as a SELECT to anything that only checks the first token, and on
    # PostgreSQL it deletes rows.
    with pytest.raises(NotReadOnly, match="delete"):
        assert_read_only("WITH gone AS (DELETE FROM users RETURNING *) SELECT * FROM gone")


def test_a_write_hidden_in_a_comment_boundary_is_refused() -> None:
    with pytest.raises(NotReadOnly):
        assert_read_only("SELECT 1 /* harmless */ ; DELETE FROM users")


def test_only_introspection_pragmas_are_allowed() -> None:
    with pytest.raises(NotReadOnly, match="introspection"):
        assert_read_only("PRAGMA journal_mode = WAL")


def test_an_empty_statement_is_refused() -> None:
    with pytest.raises(NotReadOnly, match="empty"):
        assert_read_only("   ;  ")


# -- masking ----------------------------------------------------------------------


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "app.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY, email TEXT, password_hash TEXT,
            api_key TEXT, display_name TEXT
        );
        INSERT INTO users VALUES
            (1, 'a@example.test', '$2b$12$verysecrethash', 'sk-live-abc123', 'Anong'),
            (2, 'b@example.test', '$2b$12$anotherhash', 'sk-live-def456', 'Somchai');
        CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT);
        INSERT INTO notes VALUES (1, 'ordinary text');
        """
    )
    connection.commit()
    connection.close()
    return path


def test_credential_columns_never_reach_the_caller(database: Path) -> None:
    source = SqliteSource(path=database)
    rows = source.query("SELECT id, email, password_hash, api_key, display_name FROM users")
    flattened = " ".join(cell for row in rows.rows for cell in row)

    assert "verysecrethash" not in flattened
    assert "sk-live-abc123" not in flattened
    # But the shape survives, so the agent can still reason about the table.
    assert "a@example.test" in flattened
    assert "Anong" in flattened
    assert rows.masked == ("api_key", "password_hash")


def test_masking_is_reported_not_hidden(database: Path) -> None:
    # An agent that does not know a column was masked concludes it is empty.
    rendered = SqliteSource(path=database).query("SELECT * FROM users").render()
    assert "masked columns" in rendered
    assert "<redacted:password_hash>" in rendered


def test_a_select_star_on_an_ordinary_table_is_untouched(database: Path) -> None:
    rows = SqliteSource(path=database).query("SELECT * FROM notes")
    assert rows.masked == ()
    assert rows.rows[0][1] == "ordinary text"


def test_results_are_capped_and_say_so(database: Path) -> None:
    source = SqliteSource(path=database, max_rows=1)
    rows = source.query("SELECT * FROM users", limit=50)
    assert len(rows.rows) == 1
    assert rows.truncated
    assert "truncated" in rows.render()


def test_renaming_a_credential_column_is_refused(database: Path) -> None:
    """The bug: masking keys on the name a column comes back under.

    An alias is exactly what controls that name, so ``SELECT password_hash AS
    notes`` returned the hash in full while ``SELECT password_hash`` returned a
    marker -- and the value then lives in a transcript that gets stored,
    compacted and possibly distilled into a skill. The model can reach this
    itself, and so can an instruction in a page it was asked to read.
    """
    source = SqliteSource(path=database)

    with pytest.raises(NotReadOnly, match="may not be renamed"):
        source.query("SELECT id, password_hash AS notes FROM users")
    # Wrapping it in an expression is the same bypass.
    with pytest.raises(NotReadOnly, match="may not be renamed"):
        source.query("SELECT substr(api_key, 1, 8) AS prefix FROM users")
    with pytest.raises(NotReadOnly, match="may not be renamed"):
        source.query("SELECT group_concat(password_hash) AS blob FROM users")


def test_renaming_to_another_sensitive_name_is_allowed_and_still_masked(
    database: Path,
) -> None:
    # The alias decides the output name, so a sensitive alias is still caught by
    # the ordinary masking. Refusing it would be a false positive.
    rows = SqliteSource(path=database).query("SELECT id, password_hash AS user_password FROM users")
    assert rows.masked == ("user_password",)
    assert "verysecrethash" not in " ".join(cell for row in rows.rows for cell in row)


@pytest.mark.parametrize(
    "sql",
    [
        # A table alias renames a table, not a column; the values still arrive
        # under their own names and are masked there.
        "SELECT id AS n FROM auth_tokens AS t",
        "SELECT t.id FROM audit_tokens AS t JOIN sessions AS s ON s.id = t.id",
        "SELECT count(*) AS total FROM users",
        "SELECT id, display_name AS label FROM users",
        # Referring to a credential column without returning it is fine.
        "SELECT id FROM users WHERE password_hash IS NOT NULL",
    ],
)
def test_ordinary_aliases_are_not_refused(sql: str) -> None:
    # A control that fires on legitimate queries is one somebody switches off.
    assert assert_read_only(sql)


def test_the_connection_itself_is_read_only(database: Path) -> None:
    # The second half of the enforcement: even if a statement slipped past the
    # gate, the connection refuses to write.
    source = SqliteSource(path=database)
    with source._connect() as connection, pytest.raises(sqlite3.OperationalError):
        connection.execute("DELETE FROM users")


def test_the_schema_lists_tables_and_columns(database: Path) -> None:
    described = SqliteSource(path=database).describe()
    assert "users(" in described
    assert "password_hash" in described
    assert "notes(" in described


def test_a_blob_is_summarized_rather_than_dumped(tmp_path: Path) -> None:
    path = tmp_path / "blobs.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE files (id INTEGER, data BLOB)")
    connection.execute("INSERT INTO files VALUES (1, ?)", (b"\x00" * 5000,))
    connection.commit()
    connection.close()

    rows = SqliteSource(path=path).query("SELECT * FROM files")
    assert rows.rows[0][1] == "<5000 bytes>"


def test_the_static_source_shares_the_gate() -> None:
    source = StaticSource(tables={"users": (("id", "token"), ((1, "sk-secret"),))})
    rows = source.query("SELECT * FROM users")
    assert rows.rows[0][1] == "<redacted:token>"
    with pytest.raises(NotReadOnly):
        source.query("DELETE FROM users")


# -- chunking ---------------------------------------------------------------------

DOC = """
# Deploying

Push to the staging branch and the pipeline runs automatically.

## Rollbacks

Run the rollback script with the previous tag.
Set force to false unless the cluster is wedged.

## Troubleshooting

An ImagePullBackOff usually means the pull secret expired.
"""


def test_chunks_carry_the_heading_they_fell_under() -> None:
    # A chunk saying "set force to false" is useless without the section that
    # says what is being forced.
    chunks = chunk_document(DOC, doc_id="deploy.md")
    headings = {chunk.heading for chunk in chunks}
    assert "Rollbacks" in headings
    rollback = next(c for c in chunks if c.heading == "Rollbacks")
    assert "force to false" in rollback.text
    assert "## Rollbacks" in rollback.text


def test_a_heading_starts_a_new_chunk() -> None:
    chunks = chunk_document(DOC, doc_id="deploy.md")
    assert len(chunks) >= 3
    assert all(chunk.doc_id == "deploy.md" for chunk in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_long_prose_is_split_with_overlap() -> None:
    body = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(20))
    chunks = chunk_document(body, doc_id="long.md", target=800)
    assert len(chunks) > 1
    assert all(len(chunk.text) < 2000 for chunk in chunks)
    # Every paragraph survives somewhere.
    joined = " ".join(chunk.text for chunk in chunks)
    for index in range(20):
        assert f"Paragraph {index}." in joined


def test_a_short_document_is_one_chunk() -> None:
    chunks = chunk_document("Just a line.", doc_id="tiny.md")
    assert len(chunks) == 1
    assert chunks[0].text == "Just a line."


def test_chunk_ids_are_stable() -> None:
    first = chunk_document(DOC, doc_id="deploy.md")
    second = chunk_document(DOC, doc_id="deploy.md")
    assert [c.id for c in first] == [c.id for c in second]


# -- the knowledge base ------------------------------------------------------------


def test_indexing_then_searching_finds_the_right_section(tmp_path: Path) -> None:
    kb = SqliteKnowledgeBase(path=tmp_path / "kb.db")
    kb.index(chunk_document(DOC, doc_id="deploy.md"))

    hits = kb.search("ImagePullBackOff")
    assert hits
    assert hits[0].chunk.heading == "Troubleshooting"
    assert "pull secret" in hits[0].chunk.text
    kb.close()


def test_a_natural_language_question_does_not_break_fts(tmp_path: Path) -> None:
    # FTS5's syntax is not free text: an unquoted quote or star is a syntax
    # error, and requiring every word of a question matches nothing.
    kb = SqliteKnowledgeBase(path=tmp_path / "kb.db")
    kb.index(chunk_document(DOC, doc_id="deploy.md"))

    for question in (
        "how do I roll back?",
        'what is the "force" flag for?',
        "rollback * script",
        "ต้อง deploy ยังไง",
    ):
        kb.search(question)  # must not raise
    assert kb.search("rollback script")
    kb.close()


def test_re_indexing_replaces_rather_than_duplicates(tmp_path: Path) -> None:
    kb = SqliteKnowledgeBase(path=tmp_path / "kb.db")
    kb.index(chunk_document(DOC, doc_id="deploy.md"))
    first = kb.count()
    kb.index(chunk_document(DOC, doc_id="deploy.md"))
    assert kb.count() == first
    kb.close()


def test_forgetting_a_document_removes_it_from_search(tmp_path: Path) -> None:
    kb = SqliteKnowledgeBase(path=tmp_path / "kb.db")
    kb.index(chunk_document(DOC, doc_id="deploy.md"))
    assert kb.search("rollback")

    kb.forget("deploy.md")
    assert kb.search("rollback") == []
    assert kb.count() == 0
    kb.close()


def test_a_directory_is_indexed_by_relative_path(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    (root / "guides").mkdir(parents=True)
    (root / "guides" / "deploy.md").write_text(DOC, encoding="utf-8")
    (root / "readme.md").write_text("# Readme\n\nStart here for setup.", encoding="utf-8")

    kb = SqliteKnowledgeBase(path=tmp_path / "kb.db")
    kb.index_directory(root)
    documents = dict(kb.documents())

    assert "readme.md" in documents
    assert any(name.endswith("deploy.md") for name in documents)
    kb.close()


def test_an_empty_query_returns_nothing_rather_than_everything(tmp_path: Path) -> None:
    kb = SqliteKnowledgeBase(path=tmp_path / "kb.db")
    kb.index(chunk_document(DOC, doc_id="deploy.md"))
    assert kb.search("?? **") == []
    kb.close()


def test_prepare_query_ors_the_terms() -> None:
    assert prepare_query("retry policy") == '"retry" OR "policy"'
    assert prepare_query("!!") == ""


def test_the_static_knowledge_base_scores_by_overlap() -> None:
    kb = StaticKnowledgeBase(
        passages=[
            Chunk("a.md", 0, "Rollbacks", "run the rollback script"),
            Chunk("a.md", 1, "Deploying", "push to staging"),
        ]
    )
    hits = kb.search("rollback script")
    assert hits[0].chunk.heading == "Rollbacks"
    assert kb.count() == 2
