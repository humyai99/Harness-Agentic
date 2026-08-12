"""SQLite session store: durability, search, and reversible compaction."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from harness_agentic.core.types import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from harness_agentic.errors import CompactionDeferred
from harness_agentic.session.sqlite_store import SqliteSessionStore

NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def store(tmp_path: Path) -> SqliteSessionStore:
    return SqliteSessionStore(tmp_path / "state.db")


def _m(role: str, *blocks: object) -> Message:
    return Message(role=role, content=tuple(blocks), created_at=NOW)  # type: ignore[arg-type]


def _session(store: SqliteSessionStore, tmp_path: Path) -> str:
    return store.create(source="cli", cwd=tmp_path, model="m").id


# -- round trip ----------------------------------------------------------------


def test_messages_round_trip_exactly(store: SqliteSessionStore, tmp_path: Path) -> None:
    """A signature lost on disk turns `--continue` into a rejected request."""
    sid = _session(store, tmp_path)
    original = [
        _m("user", TextBlock("hi")),
        _m(
            "assistant",
            ThinkingBlock("reasoning", signature="sig-1"),
            ToolUseBlock(id="c1", name="read_file", arguments={"path": "a"}),
        ),
        _m("tool", ToolResultBlock(tool_use_id="c1", text="contents", is_error=False)),
    ]
    store.append(sid, original)

    reloaded = store.history(sid)
    assert reloaded == original


def test_usage_is_recorded_and_summed(store: SqliteSessionStore, tmp_path: Path) -> None:
    sid = _session(store, tmp_path)
    store.append(sid, [_m("assistant", TextBlock("a"))], usage=Usage(input_tokens=10))
    store.append(sid, [_m("assistant", TextBlock("b"))], usage=Usage(input_tokens=5))
    assert store.usage_total(sid).input_tokens == 15


def test_sessions_are_scoped_by_workspace(store: SqliteSessionStore, tmp_path: Path) -> None:
    store.create(source="cli", cwd=tmp_path, model="m", workspace_key="/a")
    store.create(source="cli", cwd=tmp_path, model="m", workspace_key="/b")
    assert len(store.recent(workspace_key="/a")) == 1
    assert len(store.recent()) == 2


def test_a_newer_schema_is_refused(tmp_path: Path) -> None:
    """Opening a database you do not understand is how transcripts corrupt."""
    path = tmp_path / "state.db"
    store = SqliteSessionStore(path)
    with store._write() as conn:
        conn.execute("UPDATE state_meta SET value = '999' WHERE key = 'schema_version'")
    store.close()
    with pytest.raises(RuntimeError, match="newer version"):
        SqliteSessionStore(path)


# -- search --------------------------------------------------------------------


def test_full_text_search_finds_past_messages(store: SqliteSessionStore, tmp_path: Path) -> None:
    sid = _session(store, tmp_path)
    store.append(
        sid,
        [
            _m("user", TextBlock("we should migrate the postgres cluster")),
            _m("assistant", TextBlock("unrelated answer about cats")),
        ],
    )
    hits = store.search("postgres")
    assert len(hits) == 1
    assert hits[0].session_id == sid


def test_search_indexes_tool_names_but_not_tool_output(
    store: SqliteSessionStore, tmp_path: Path
) -> None:
    """Tool output dominates the bytes and almost nobody searches for it."""
    sid = _session(store, tmp_path)
    store.append(
        sid,
        [
            _m("assistant", ToolUseBlock(id="c1", name="deploy_thing")),
            _m("tool", ToolResultBlock(tool_use_id="c1", text="ZZZUNIQUEOUTPUT")),
        ],
    )
    assert store.search("deploy_thing")
    assert not store.search("ZZZUNIQUEOUTPUT")


def test_search_treats_user_input_as_text_not_syntax(
    store: SqliteSessionStore, tmp_path: Path
) -> None:
    """A stray quote or NEAR in a question must not raise from FTS5."""
    sid = _session(store, tmp_path)
    store.append(sid, [_m("user", TextBlock('he said "hello" NEAR the door'))])
    assert store.search('"hello"') == [] or store.search('"hello"')
    assert store.search("NEAR the door")


def test_search_finds_thai_text(store: SqliteSessionStore, tmp_path: Path) -> None:
    """Thai has no inter-word spaces, so the tokenizer choice matters."""
    sid = _session(store, tmp_path)
    store.append(sid, [_m("user", TextBlock("ช่วยดูไฟล์ pyproject.toml ให้หน่อย"))])
    assert store.search("pyproject.toml")


# -- compaction ----------------------------------------------------------------


def test_compaction_hides_rather_than_deletes(store: SqliteSessionStore, tmp_path: Path) -> None:
    sid = _session(store, tmp_path)
    store.append(sid, [_m("user", TextBlock(f"turn {i}")) for i in range(5)])

    store.record_compaction(sid, replaced=(1, 3), summary=_m("user", TextBlock("summary")))

    visible = store.history(sid)
    assert [m.text() for m in visible] == ["turn 0", "summary", "turn 4"]

    everything = store.history(sid, include_hidden=True)
    assert len(everything) == 6
    assert "turn 2" in [m.text() for m in everything]


def test_compaction_is_reversible(store: SqliteSessionStore, tmp_path: Path) -> None:
    """The reason for hiding instead of forking a child session."""
    sid = _session(store, tmp_path)
    original = [_m("user", TextBlock(f"turn {i}")) for i in range(5)]
    store.append(sid, original)
    store.record_compaction(sid, replaced=(1, 3), summary=_m("user", TextBlock("summary")))

    assert store.undo_compaction(sid) is True
    assert store.history(sid) == original
    assert store.get(sid).compaction_count == 0  # type: ignore[union-attr]


def test_undo_with_nothing_to_undo_is_false(store: SqliteSessionStore, tmp_path: Path) -> None:
    sid = _session(store, tmp_path)
    assert store.undo_compaction(sid) is False


def test_search_still_finds_compacted_away_text(store: SqliteSessionStore, tmp_path: Path) -> None:
    """Search over originals is the other reason not to fork."""
    sid = _session(store, tmp_path)
    store.append(
        sid,
        [
            _m("user", TextBlock("start")),
            _m("user", TextBlock("the magic constant is 8675309")),
            _m("user", TextBlock("end")),
        ],
    )
    store.record_compaction(sid, replaced=(1, 1), summary=_m("user", TextBlock("summary")))

    assert "the magic constant is 8675309" not in [m.text() for m in store.history(sid)]
    assert store.search("8675309")


# -- concurrency ---------------------------------------------------------------


def test_the_turn_lease_serializes_writers(store: SqliteSessionStore, tmp_path: Path) -> None:
    sid = _session(store, tmp_path)
    with (
        store.turn_lease(sid),
        pytest.raises(CompactionDeferred),
        store.turn_lease(sid, timeout_s=0.1),
    ):
        pass


def test_the_lease_is_released_on_exit(store: SqliteSessionStore, tmp_path: Path) -> None:
    sid = _session(store, tmp_path)
    with store.turn_lease(sid):
        pass
    with store.turn_lease(sid, timeout_s=0.1):
        pass


def test_the_lease_is_released_even_when_the_body_raises(
    store: SqliteSessionStore, tmp_path: Path
) -> None:
    """Otherwise one crashed turn locks the session out permanently."""
    sid = _session(store, tmp_path)
    with pytest.raises(ValueError, match="boom"), store.turn_lease(sid):
        raise ValueError("boom")
    with store.turn_lease(sid, timeout_s=0.1):
        pass
