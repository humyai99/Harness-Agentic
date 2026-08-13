"""Durable memory: the bound, the split, and the frozen snapshot.

Two behaviours carry this subsystem and both are about refusing to do something.

The **hard limit** is what stops memory becoming a landfill. Memory is in the
prompt on every turn of every future session, so an unbounded one costs forever
and nobody prunes it, because nothing ever forces the question. `add` refusing is
what turns "remember this too" into "which of these matters more".

The **frozen snapshot** is what stops memory destroying the prompt cache. The
block has to be byte-identical every turn, so a write lands on disk now and joins
the prompt next session -- and the tool has to say so, or the model plans its
next turn around a fact it cannot see.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness_agentic.memory.manager import (
    MAX_ENTRY_CHARS,
    MemoryFull,
    MemoryStore,
    parse_entries,
)


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(root=tmp_path / "memories")


# -- the two files ---------------------------------------------------------------


def test_facts_about_the_work_and_the_person_are_kept_apart(store: MemoryStore) -> None:
    """A new project should not lose what was learned about the person.

    And a new colleague should not inherit somebody else's preferences.
    """
    store.add("memory", "This project builds with uv, not pip.")
    store.add("user", "Prefers Thai for explanations.")

    assert "uv" in store.load("memory").render()
    assert "uv" not in store.load("user").render()
    assert "Thai" in store.load("user").render()
    assert "Thai" not in store.load("memory").render()


def test_an_unknown_kind_is_refused(store: MemoryStore) -> None:
    with pytest.raises(ValueError, match="unknown memory"):
        store.add("everything", "something")


def test_entries_can_be_grouped_under_headings(store: MemoryStore) -> None:
    store.add("memory", "Runs on Python 3.12.", section="Build")
    store.add("memory", "Staging is staging-1.", section="Deploy")
    rendered = store.load("memory").render()

    assert "## Build" in rendered
    assert "## Deploy" in rendered
    assert rendered.index("## Build") < rendered.index("## Deploy"), "first-seen order"


# -- the bound -------------------------------------------------------------------


def test_memory_refuses_to_grow_past_its_limit(tmp_path: Path) -> None:
    """The refusal is the feature.

    A warning would be ignored, and memory that grows without bound costs tokens
    on every turn forever while nobody prunes it.
    """
    store = MemoryStore(root=tmp_path / "m", memory_limit=300)
    for index in range(20):
        try:
            store.add("memory", f"Fact number {index} about this project.")
        except MemoryFull:
            break
    else:
        pytest.fail("the limit was never enforced")

    assert store.load("memory").used <= 300


def _fill_until_refused(store: MemoryStore) -> None:
    """Add filler until the file says no. Raising is the expected outcome."""
    for index in range(20):
        store.add("memory", f"Filler fact number {index}, taking up room.")


def test_the_refusal_names_what_could_go(tmp_path: Path) -> None:
    # "Make room" is not actionable without knowing what is in there, and the
    # model cannot see the file -- only this message.
    store = MemoryStore(root=tmp_path / "m", memory_limit=200)
    store.add("memory", "The very first thing that was ever recorded here.")
    with pytest.raises(MemoryFull, match="The very first thing"):
        _fill_until_refused(store)


def test_a_full_file_still_accepts_a_replacement(tmp_path: Path) -> None:
    """Otherwise the only way out of a full file is to delete something first.

    Which is a worse experience than it sounds: the obvious action fails, and the
    recovery is two steps that can half-succeed.
    """
    store = MemoryStore(root=tmp_path / "m", memory_limit=200)
    store.add("memory", "Staging is called staging-1.")
    with pytest.raises(MemoryFull):
        _fill_until_refused(store)

    assert store.replace("memory", "staging-1", "Staging is called staging-2.")
    assert "staging-2" in store.load("memory").render()


def test_an_entry_longer_than_a_fact_is_refused(store: MemoryStore) -> None:
    # A long entry is a procedure wearing a fact's clothes, and it belongs in a
    # skill where it costs nothing until loaded.
    with pytest.raises(ValueError, match="belongs in a skill"):
        store.add("memory", "x" * (MAX_ENTRY_CHARS + 1))


def test_the_same_fact_is_not_recorded_twice(store: MemoryStore) -> None:
    """Otherwise re-adding it every session crowds out everything else."""
    store.add("memory", "This project builds with uv.")
    store.add("memory", "This project builds with uv.")
    assert store.load("memory").render().count("builds with uv") == 1


def test_an_empty_entry_is_refused(store: MemoryStore) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        store.add("memory", "   ")


# -- editing ---------------------------------------------------------------------


def test_a_fact_that_changed_can_be_updated_in_place(store: MemoryStore) -> None:
    store.add("memory", "The staging cluster is staging-1.", section="Deploy")
    assert store.replace("memory", "staging-1", "The staging cluster is staging-2.")

    rendered = store.load("memory").render()
    assert "staging-2" in rendered
    assert "staging-1" not in rendered
    assert "## Deploy" in rendered, "the entry keeps its section"


def test_a_stale_fact_can_be_forgotten(store: MemoryStore) -> None:
    # A file of stale facts costs the same as a file of good ones and is worse
    # than empty, because it is believed.
    store.add("memory", "We deploy on Fridays.")
    assert store.remove("memory", "Fridays")
    assert not store.load("memory").entries


def test_editing_something_absent_says_so(store: MemoryStore) -> None:
    assert not store.replace("memory", "nothing like this", "new")
    assert not store.remove("memory", "nothing like this")


def test_an_empty_search_does_not_match_the_first_entry(store: MemoryStore) -> None:
    """``"" in text`` is always true, which would delete something unrelated.

    And report success while doing it, so nothing downstream would notice. The
    tool schema demands a non-empty string, but a string of spaces satisfies
    that and cleans to nothing, so the guard belongs here rather than there.
    """
    store.add("memory", "Staging is staging-1.")
    store.add("memory", "Builds with uv.")

    with pytest.raises(ValueError, match="cannot be empty"):
        store.remove("memory", "   ")
    with pytest.raises(ValueError, match="cannot be empty"):
        store.replace("memory", " ", "something else")

    assert len(store.load("memory").entries) == 2


def test_a_replacement_is_bounded_like_an_addition(store: MemoryStore) -> None:
    # Otherwise the per-entry limit is enforced on one write path and not the
    # other, and a procedure can enter memory by being an edit rather than an add.
    store.add("memory", "A short fact.")
    with pytest.raises(ValueError, match="belongs in a skill"):
        store.replace("memory", "short fact", "x" * (MAX_ENTRY_CHARS + 1))

    assert "A short fact." in store.load("memory").render()


def test_a_replacement_cannot_empty_an_entry(store: MemoryStore) -> None:
    # An entry rendered as a bare "- " does not parse back on the next read, so
    # the entry would vanish one session later rather than at the point of edit.
    store.add("memory", "A fact worth keeping.")
    with pytest.raises(ValueError, match="cannot be empty"):
        store.replace("memory", "worth keeping", "   ")

    assert store.load("memory").entries


# -- the snapshot ----------------------------------------------------------------


def test_the_snapshot_is_empty_when_nothing_is_remembered(store: MemoryStore) -> None:
    """A fresh install should not spend a cache-breaking block on two headings."""
    assert store.snapshot() == ""


def test_the_snapshot_carries_both_files_and_says_they_may_be_stale(
    store: MemoryStore,
) -> None:
    store.add("memory", "Builds with uv.")
    store.add("user", "Prefers Thai.")
    snapshot = store.snapshot()

    assert "Builds with uv" in snapshot
    assert "Prefers Thai" in snapshot
    # Presented as recollection rather than as fact, because it may be months old.
    assert "may be out of date" in snapshot


def test_a_write_does_not_change_a_snapshot_already_taken(store: MemoryStore) -> None:
    """The frozen-snapshot rule, which is what protects the cached prefix.

    Re-reading per turn would change the block mid-session and throw away the
    cache for every remaining turn -- a cost with no error and no warning.
    """
    store.add("memory", "Known at session start.")
    snapshot = store.snapshot()

    store.add("memory", "Learned mid-session.")

    assert "Learned mid-session" not in snapshot, "the snapshot is a value, not a view"
    # It is on disk, though, and the next session's snapshot has it.
    assert "Learned mid-session" in store.snapshot()


# -- the files on disk -----------------------------------------------------------


def test_memory_files_are_not_world_readable(store: MemoryStore) -> None:
    # Not secret in the credential sense, and nobody else's business either.
    store.add("memory", "Something about this project.")
    assert store.path_for("memory").stat().st_mode & 0o077 == 0


def test_a_hand_edited_file_survives_being_read(tmp_path: Path) -> None:
    """These files are meant to be edited by a person as well as by the agent.

    Someone who adds a paragraph of prose should not have their file silently
    emptied, so anything that is not a bullet is ignored rather than rejected.
    """
    root = tmp_path / "memories"
    root.mkdir(parents=True)
    (root / "MEMORY.md").write_text(
        "# Project memory\n\n"
        "Some prose a person wrote to explain the sections below.\n\n"
        "## Build\n"
        "- Builds with uv.\n"
        "* Tests need Postgres.\n",
        encoding="utf-8",
    )
    store = MemoryStore(root=root)
    entries = store.load("memory").entries

    assert [e.text for e in entries] == ["Builds with uv.", "Tests need Postgres."]
    assert all(e.section == "Build" for e in entries)


def test_a_multi_line_entry_is_collapsed_rather_than_rejected() -> None:
    # Pasting something multi-line is a normal thing to do, and the file format
    # is one fact per bullet.
    entries = parse_entries("- one fact\n- another fact\n")
    assert len(entries) == 2


def test_an_empty_file_costs_nothing(store: MemoryStore) -> None:
    # It is left out of the snapshot entirely, so charging for its heading would
    # make a fresh install look like it had already spent part of its budget.
    assert store.load("memory").used == 0
    assert store.load("memory").remaining == store.memory_limit


def test_usage_reports_room_left_per_file(store: MemoryStore) -> None:
    store.add("memory", "A fact.")
    usage = {kind: (used, limit) for kind, used, limit in store.usage()}

    assert usage["memory"][0] > 0
    assert usage["memory"][1] == store.memory_limit
    assert usage["user"][1] == store.user_limit
