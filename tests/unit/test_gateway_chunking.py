"""Splitting answers into messages a platform will accept."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from harness_agentic.gateway.chunking import clip, split_for_platform


def test_short_text_is_one_chunk() -> None:
    assert split_for_platform("hello", 100) == ["hello"]


def test_splits_on_a_paragraph_break() -> None:
    text = "First paragraph here.\n\n" + "Second paragraph here." * 3
    chunks = split_for_platform(text, 60)
    assert chunks[0] == "First paragraph here."
    assert len(chunks) > 1


def test_prefers_a_line_break_over_a_word_break() -> None:
    text = "alpha beta gamma\ndelta epsilon zeta eta theta"
    chunks = split_for_platform(text, 24)
    assert chunks[0] == "alpha beta gamma"


def test_thai_without_spaces_still_splits() -> None:
    # No spaces to break on. A hard cut is correct: the alternative is a
    # message the platform rejects outright.
    text = "สวัสดีครับผมคือผู้ช่วยของคุณ" * 6
    chunks = split_for_platform(text, 40)
    assert all(len(c) <= 40 for c in chunks)
    assert "".join(chunks) == text


def test_code_fence_is_closed_and_reopened() -> None:
    body = "\n".join(f"line_{i} = {i}" for i in range(40))
    text = f"Here is the file:\n\n```python\n{body}\n```\n\nThat is all."
    chunks = split_for_platform(text, 200)

    assert len(chunks) > 1
    for chunk in chunks:
        # Every chunk must have balanced fences, or the second one renders as
        # prose and the code is unreadable.
        assert chunk.count("```") % 2 == 0, chunk
    assert "```python" in chunks[0]
    assert any(c.startswith("```") for c in chunks[1:])


def test_reopened_fence_keeps_the_language() -> None:
    body = "\n".join(f"x{i} = {i}" for i in range(30))
    chunks = split_for_platform(f"```rust\n{body}\n```", 150)
    assert chunks[1].startswith("```rust")


def test_line_sized_answer_fits_in_one_message() -> None:
    # 4500 characters is over Telegram's limit and under LINE's; the same text
    # must produce different numbers of messages per platform.
    text = "ก" * 4500
    assert len(split_for_platform(text, 5000)) == 1
    assert len(split_for_platform(text, 4096)) == 2


@given(
    text=st.text(min_size=0, max_size=800),
    limit=st.integers(min_value=8, max_value=120),
)
@settings(max_examples=300, deadline=None)
def test_no_chunk_ever_exceeds_the_limit(text: str, limit: int) -> None:
    # The property that matters: an over-length message is rejected by the
    # platform, and the user simply never receives that part of the answer.
    for chunk in split_for_platform(text, limit):
        assert len(chunk) <= limit


@given(text=st.text(min_size=1, max_size=600), limit=st.integers(min_value=16, max_value=120))
@settings(max_examples=200, deadline=None)
def test_nothing_is_silently_dropped(text: str, limit: int) -> None:
    # Fences and whitespace may be added or trimmed, but no non-whitespace
    # character of the original may vanish.
    chunks = split_for_platform(text, limit)
    joined = "".join(chunks).replace("`", "").replace("~", "")
    original = "".join(text.split()).replace("`", "").replace("~", "")
    assert "".join(joined.split()) == original


def test_clip_marks_what_it_cut() -> None:
    assert clip("abcdefghij", 5) == "abcd…"
    assert clip("abc", 5) == "abc"
