"""Streaming an answer onto a platform, both ways.

The interesting assertions here are about *cost and shape*, not content: how
many API calls a streamed answer makes, whether debouncing actually coalesced
anything, and whether a platform that cannot edit still gets the whole answer.
Getting these wrong is invisible in a manual test with one short reply and very
visible on a rate-limited production account.
"""

from __future__ import annotations

import asyncio

import pytest

from harness_agentic.core.events import (
    ApprovalRequested,
    CompactionStarted,
    Notice,
    TextChunk,
    ToolCallFinished,
    ToolCallStarted,
    TurnFinished,
)
from harness_agentic.core.types import Usage
from harness_agentic.gateway.clock import ManualAsyncClock
from harness_agentic.gateway.delivery import Delivery
from harness_agentic.gateway.platforms.fake import FakeAdapter, line_like, telegram_like
from harness_agentic.gateway.types import Capabilities, DeliveryTarget
from harness_agentic.tools.spec import Danger

TARGET = DeliveryTarget(platform="fake", chat_id="c1")


def make(adapter: FakeAdapter, clock: ManualAsyncClock) -> Delivery:
    return Delivery(adapter=adapter, target=TARGET, clock=clock)


# -- edit-in-place --------------------------------------------------------------


async def test_streaming_posts_once_and_edits() -> None:
    adapter = FakeAdapter(capabilities=telegram_like())
    clock = ManualAsyncClock()
    delivery = make(adapter, clock)

    await delivery.handle(TextChunk("Hello"))
    await clock.advance(2.0)
    await delivery.handle(TextChunk(", world"))
    await clock.advance(2.0)
    await delivery.finish()

    assert len(adapter.posted) == 1
    assert adapter.posted[0].final_text == "Hello, world"


async def test_rapid_chunks_coalesce_into_one_edit() -> None:
    # Nine chunks arriving inside one debounce window must not become nine
    # edits; that is how a bot gets rate-limited off a platform.
    adapter = FakeAdapter(capabilities=telegram_like())
    clock = ManualAsyncClock()
    delivery = make(adapter, clock)

    for index in range(9):
        await delivery.handle(TextChunk(f"chunk{index} "))
    await clock.advance(2.0)
    await delivery.finish()

    assert delivery.stats.edits_skipped >= 7
    assert adapter.posted[0].final_text.startswith("chunk0")
    assert "chunk8" in adapter.posted[0].final_text


async def test_the_final_text_is_never_debounced_away() -> None:
    # The last chunk arriving inside the debounce window is the one most likely
    # to be lost, and losing it truncates the answer.
    adapter = FakeAdapter(capabilities=telegram_like())
    clock = ManualAsyncClock()
    delivery = make(adapter, clock)

    await delivery.handle(TextChunk("start "))
    await clock.advance(2.0)
    await delivery.handle(TextChunk("and the important ending"))
    await delivery.finish()

    assert adapter.posted[-1].final_text.endswith("important ending")


async def test_an_overlong_stream_seals_and_starts_a_new_message() -> None:
    adapter = FakeAdapter(capabilities=Capabilities(max_chars=60, edit_messages=True))
    clock = ManualAsyncClock()
    delivery = make(adapter, clock)

    for _ in range(6):
        await delivery.handle(TextChunk("0123456789 "))
        await clock.advance(2.0)
    await delivery.finish()

    assert len(adapter.posted) >= 2
    adapter.assert_within_limits()


async def test_finish_is_idempotent() -> None:
    adapter = FakeAdapter(capabilities=telegram_like())
    delivery = make(adapter, ManualAsyncClock())
    await delivery.handle(TextChunk("done"))
    await delivery.finish()
    await delivery.finish()
    assert len(adapter.posted) == 1


# -- chunked (LINE) -------------------------------------------------------------


async def test_a_platform_without_edits_never_calls_edit() -> None:
    adapter = FakeAdapter(capabilities=line_like())
    delivery = make(adapter, ManualAsyncClock())

    await delivery.handle(TextChunk("part one. "))
    await delivery.handle(TextChunk("part two."))
    await delivery.finish()

    assert delivery.stats.edits == 0
    assert all(not p.edits for p in adapter.posted)
    assert "part one. part two." in adapter.transcript()


async def test_a_long_answer_is_split_into_whole_messages() -> None:
    adapter = FakeAdapter(capabilities=Capabilities(max_chars=80, edit_messages=False))
    delivery = make(adapter, ManualAsyncClock())

    for index in range(8):
        await delivery.handle(TextChunk(f"Paragraph number {index}.\n\n"))
    await delivery.finish()

    assert len(adapter.posted) > 1
    adapter.assert_within_limits()
    joined = " ".join(adapter.texts())
    for index in range(8):
        assert f"Paragraph number {index}." in joined


async def test_progress_is_suppressed_where_it_would_cost_a_push() -> None:
    # Every status line on LINE is a metered push message and an unwanted
    # notification, so the chunked path drops them rather than narrating.
    adapter = FakeAdapter(capabilities=line_like())
    delivery = make(adapter, ManualAsyncClock())

    await delivery.handle(ToolCallStarted("t1", "read_file", Danger.SAFE, "pyproject.toml"))
    await delivery.finish()
    assert adapter.posted == []


async def test_progress_is_shown_where_it_is_cheap() -> None:
    adapter = FakeAdapter(capabilities=telegram_like())
    delivery = make(adapter, ManualAsyncClock())

    await delivery.handle(ToolCallStarted("t1", "read_file", Danger.SAFE, "pyproject.toml"))
    await delivery.handle(ToolCallFinished("t1", "read_file", is_error=True, duration_s=0.1))
    assert len(adapter.of_kind("status")) == 2


# -- out-of-band messages -------------------------------------------------------


async def test_an_approval_request_arrives_immediately_and_says_how_to_answer() -> None:
    adapter = FakeAdapter(capabilities=telegram_like())
    delivery = make(adapter, ManualAsyncClock())

    await delivery.handle(TextChunk("about to run something"))
    await delivery.handle(ApprovalRequested("a1", "terminal", Danger.DESTRUCTIVE, "rm -rf build"))

    approvals = adapter.of_kind("approval")
    assert len(approvals) == 1
    assert "rm -rf build" in approvals[0].message.text
    assert "/approve" in approvals[0].message.text


async def test_a_fallback_is_announced_rather_than_hidden() -> None:
    from harness_agentic.core.events import ProviderFallback

    adapter = FakeAdapter(capabilities=telegram_like())
    delivery = make(adapter, ManualAsyncClock())
    await delivery.handle(ProviderFallback("big/model", "small/model", "rate limited"))
    assert "small/model" in adapter.of_kind("status")[0].message.text


async def test_errors_and_compaction_reach_the_user() -> None:
    adapter = FakeAdapter(capabilities=telegram_like())
    delivery = make(adapter, ManualAsyncClock())
    await delivery.handle(CompactionStarted(0, 10))
    await delivery.handle(Notice("error", "the tool crashed"))
    assert adapter.of_kind("status")
    assert "the tool crashed" in adapter.of_kind("error")[0].message.text


async def test_turn_finished_flushes_the_buffer() -> None:
    adapter = FakeAdapter(capabilities=line_like())
    delivery = make(adapter, ManualAsyncClock())
    await delivery.handle(TextChunk("the answer"))
    await delivery.handle(TurnFinished("completed", iterations=1, usage=Usage()))
    assert adapter.transcript() == "the answer"


async def test_a_typing_indicator_is_only_used_where_it_exists() -> None:
    with_typing = FakeAdapter(capabilities=telegram_like())
    await make(with_typing, ManualAsyncClock()).typing()
    assert with_typing.typing_calls == 1

    without = FakeAdapter(capabilities=line_like())
    await make(without, ManualAsyncClock()).typing()
    assert without.typing_calls == 0


# -- the manual clock itself -----------------------------------------------------


async def test_the_manual_clock_releases_sleepers_in_order() -> None:
    clock = ManualAsyncClock()
    woken: list[str] = []

    async def sleeper(name: str, seconds: float) -> None:
        await clock.sleep(seconds)
        woken.append(name)

    tasks = [
        asyncio.create_task(sleeper("late", 5.0)),
        asyncio.create_task(sleeper("early", 1.0)),
    ]
    await asyncio.sleep(0)
    await clock.advance(2.0)
    assert woken == ["early"]
    await clock.advance(5.0)
    assert woken == ["early", "late"]
    await asyncio.gather(*tasks)


async def test_the_manual_clock_advances_wall_time_too() -> None:
    clock = ManualAsyncClock()
    before = clock.now()
    await clock.advance(60.0)
    assert (clock.now() - before).total_seconds() == pytest.approx(60.0)
