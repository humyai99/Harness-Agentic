"""Repairing a message history before it goes on the wire.

This is where provider 400s live. The failure is rarely the request the model
just produced -- it is some earlier turn that ended badly and left the history
in a shape the API refuses, so every subsequent turn fails too. Interruption,
a crashed tool, a timeout mid-stream, a model emitting malformed JSON: each
leaves a specific kind of damage, and each has to be repaired before the next
request rather than diagnosed after the rejection.

Repair is driven by a :class:`~harness_agentic.providers.base.SanitizeRules`
value object rather than by provider names, so adding a provider means
describing its rules, not editing this module.

The one invariant worth stating plainly: **every tool call must be answered.**
A ``tool_use`` block with no matching ``tool_result`` in the next turn is an
immediate rejection from every provider, and it is the single most common way
a history goes bad.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING

from harness_agentic.core.types import (
    ContentBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)

if TYPE_CHECKING:
    from harness_agentic.providers.base import SanitizeRules

MISSING_RESULT_TEXT = (
    "No result was recorded for this call. It may have been interrupted or the "
    "process may have restarted. Assume it did not run and decide what to do next."
)


@dataclass(frozen=True, slots=True)
class SanitizeReport:
    """What repair had to do. Surfaced in traces, not to the model."""

    synthesized_results: int = 0
    dropped_orphan_results: int = 0
    dropped_unsigned_thinking: int = 0
    dropped_empty_messages: int = 0
    merged_messages: int = 0
    repaired_arguments: int = 0
    reordered_tool_results: int = 0

    @property
    def clean(self) -> bool:
        """Whether the history needed no repair at all."""
        return not any(
            (
                self.synthesized_results,
                self.dropped_orphan_results,
                self.dropped_unsigned_thinking,
                self.dropped_empty_messages,
                self.merged_messages,
                self.repaired_arguments,
                self.reordered_tool_results,
            )
        )

    def summary(self) -> str:
        """A one-line description, or the empty string when nothing changed."""
        parts = [
            f"{self.synthesized_results} synthesized result(s)" if self.synthesized_results else "",
            f"{self.dropped_orphan_results} orphan result(s)"
            if self.dropped_orphan_results
            else "",
            f"{self.dropped_unsigned_thinking} unsigned thinking block(s)"
            if self.dropped_unsigned_thinking
            else "",
            f"{self.dropped_empty_messages} empty message(s)"
            if self.dropped_empty_messages
            else "",
            f"{self.merged_messages} merged message(s)" if self.merged_messages else "",
            f"{self.repaired_arguments} repaired argument(s)" if self.repaired_arguments else "",
            f"{self.reordered_tool_results} reordered turn(s)"
            if self.reordered_tool_results
            else "",
        ]
        return ", ".join(p for p in parts if p)


def sanitize(
    messages: Sequence[Message], rules: SanitizeRules, *, now: datetime
) -> tuple[list[Message], SanitizeReport]:
    """Return a history the provider will accept, plus what had to change.

    Pure: the input is never mutated, which matters because the caller keeps
    the original for the session store. What gets persisted is the truth; what
    gets sent is the repaired version.
    """
    counts = {
        "synthesized_results": 0,
        "dropped_orphan_results": 0,
        "dropped_unsigned_thinking": 0,
        "dropped_empty_messages": 0,
        "merged_messages": 0,
        "repaired_arguments": 0,
        "reordered_tool_results": 0,
    }

    working = list(messages)
    working = _repair_arguments(working, counts)
    if rules.drop_unsigned_thinking:
        working = _drop_unsigned_thinking(working, counts)
    if rules.require_tool_result_pairing:
        working = _pair_tool_calls(working, counts, now=now)
    working = _drop_empty(working, rules, counts)
    if rules.require_alternating_roles:
        working = _merge_adjacent(working, rules, counts)
    if rules.tool_results_in_user_message:
        # Last, so it sees the merged shape as well as the arriving one.
        working = _order_tool_results(working, rules, counts)

    return working, SanitizeReport(**counts)


# -- individual repairs --------------------------------------------------------


def _repair_arguments(messages: list[Message], counts: dict[str, int]) -> list[Message]:
    """Re-encode tool arguments that arrived as unparsable JSON.

    The model's raw string is kept on the block, so a truncated stream can be
    salvaged: an empty ``arguments`` with a non-empty ``raw_arguments`` means
    parsing failed, and sending the broken string back would fail again. An
    empty object at least lets dispatch report a clear schema error the model
    can correct.
    """
    out: list[Message] = []
    for message in messages:
        blocks: list[ContentBlock] = []
        changed = False
        for block in message.content:
            if isinstance(block, ToolUseBlock) and block.raw_arguments and not block.arguments:
                try:
                    parsed = json.loads(block.raw_arguments)
                except ValueError:
                    parsed = None
                recovered = parsed if isinstance(parsed, dict) else {}
                blocks.append(replace(block, arguments=recovered, raw_arguments=None))
                # Counted only when something was actually salvaged or lost. A
                # tool that takes no arguments arrives as the string "{}", which
                # parses to exactly the empty dict already there -- so every
                # no-argument call was reported as a repaired history, on every
                # turn, for the life of the session. "history repaired" is the
                # line that matters when something is genuinely damaged, and a
                # version of it that fires constantly is one nobody reads.
                if not isinstance(parsed, dict) or parsed != block.arguments:
                    # Reported unless the raw text parsed to a mapping that
                    # agrees with what was already there. Anything else lost or
                    # changed what the model meant to send -- including raw text
                    # that parsed to a list, where the arguments are silently
                    # discarded and the result looks identical to the benign case.
                    counts["repaired_arguments"] += 1
                changed = True
                continue
            blocks.append(block)
        out.append(replace(message, content=tuple(blocks)) if changed else message)
    return out


def _drop_unsigned_thinking(messages: list[Message], counts: dict[str, int]) -> list[Message]:
    """Remove thinking blocks that cannot legally be replayed."""
    out: list[Message] = []
    for message in messages:
        cleaned = message.without_unsigned_thinking()
        if len(cleaned.content) != len(message.content):
            counts["dropped_unsigned_thinking"] += len(message.content) - len(cleaned.content)
        out.append(cleaned)
    return out


def _pair_tool_calls(
    messages: list[Message], counts: dict[str, int], *, now: datetime
) -> list[Message]:
    """Guarantee every tool call is answered and every answer has a call.

    Two directions of damage, both fatal to the next request:

    * a call with no result -- the turn died between asking and answering, so a
      synthetic error result is inserted saying the outcome is unknown;
    * a result with no call -- compaction or an edit removed the assistant turn
      that asked, so the orphan is dropped.
    """
    answered: set[str] = set()
    for message in messages:
        answered.update(block.tool_use_id for block in message.tool_results())

    out: list[Message] = []
    for index, message in enumerate(messages):
        kept: list[ContentBlock] = []
        for block in message.content:
            if isinstance(block, ToolResultBlock) and not _has_call(messages, block.tool_use_id):
                counts["dropped_orphan_results"] += 1
                continue
            kept.append(block)
        current = (
            replace(message, content=tuple(kept)) if len(kept) != len(message.content) else message
        )
        out.append(current)

        unanswered = [c for c in current.tool_uses() if c.id not in answered]
        if unanswered and not _next_answers(messages, index, unanswered):
            out.append(
                Message(
                    role="tool",
                    content=tuple(
                        ToolResultBlock(
                            tool_use_id=call.id, text=MISSING_RESULT_TEXT, is_error=True
                        )
                        for call in unanswered
                    ),
                    created_at=now,
                )
            )
            counts["synthesized_results"] += len(unanswered)
            answered.update(call.id for call in unanswered)

    return out


def _has_call(messages: Sequence[Message], call_id: str) -> bool:
    """Whether any assistant turn asked for ``call_id``."""
    return any(call.id == call_id for message in messages for call in message.tool_uses())


def _next_answers(messages: Sequence[Message], index: int, calls: Sequence[ToolUseBlock]) -> bool:
    """Whether the following message already answers all of ``calls``."""
    if index + 1 >= len(messages):
        return False
    answered = {block.tool_use_id for block in messages[index + 1].tool_results()}
    return all(call.id in answered for call in calls)


def _drop_empty(
    messages: list[Message], rules: SanitizeRules, counts: dict[str, int]
) -> list[Message]:
    """Remove messages that would serialize to nothing.

    An assistant turn with no content is rejected outright by several
    providers, and it carries no information anyway.
    """
    out: list[Message] = []
    for message in messages:
        meaningful = [
            b for b in message.content if not (isinstance(b, TextBlock) and not b.text.strip())
        ]
        if not meaningful:
            if message.role == "assistant" and rules.allow_empty_assistant:
                out.append(message)
                continue
            counts["dropped_empty_messages"] += 1
            continue
        out.append(
            replace(message, content=tuple(meaningful))
            if len(meaningful) != len(message.content)
            else message
        )
    return out


def _merge_adjacent(
    messages: list[Message], rules: SanitizeRules, counts: dict[str, int]
) -> list[Message]:
    """Join consecutive turns that map to the same wire role.

    Anthropic rejects two user turns in a row, and our ``tool`` role becomes a
    user turn there -- so a tool result followed by the user's next message is
    two user turns unless they are merged here.

    Merging preserves order, which for a user turn carrying tool results is not
    quite enough on its own -- see :func:`_order_tool_results`, which runs after
    this and fixes up the order however it arose.
    """
    out: list[Message] = []
    for message in messages:
        wire_role = _wire_role(message.role, rules)
        if out and _wire_role(out[-1].role, rules) == wire_role:
            previous = out[-1]
            out[-1] = replace(previous, content=previous.content + message.content)
            counts["merged_messages"] += 1
            continue
        out.append(message)
    return out


def _order_tool_results(
    messages: list[Message], rules: SanitizeRules, counts: dict[str, int]
) -> list[Message]:
    """Put ``tool_result`` blocks first in every turn that becomes a user turn.

    Anthropic requires them to lead the content array, so anything ahead of one
    is a 400 -- and there are two ways to get there. Merging preserves order, so
    a user turn joined in front of a tool turn puts its text first; and a single
    message can simply arrive holding a result behind something else.

    Repaired rather than asserted against, because this module's job is to
    produce a history the provider accepts. Ordering is stable within each group,
    so results keep their order relative to each other and so does everything
    else; only the two groups move.
    """
    out: list[Message] = []
    for message in messages:
        blocks = message.content
        if _wire_role(message.role, rules) != "user" or not message.tool_results():
            out.append(message)
            continue
        results = [b for b in blocks if isinstance(b, ToolResultBlock)]
        rest = [b for b in blocks if not isinstance(b, ToolResultBlock)]
        reordered: tuple[ContentBlock, ...] = (*results, *rest)
        if reordered == blocks:
            out.append(message)
            continue
        counts["reordered_tool_results"] += 1
        out.append(replace(message, content=reordered))
    return out


def _wire_role(role: str, rules: SanitizeRules) -> str:
    """The role a message will actually carry on the wire."""
    if role == "tool" and rules.tool_results_in_user_message:
        return "user"
    return role


# -- verification --------------------------------------------------------------


def assert_valid(messages: Sequence[Message], rules: SanitizeRules) -> None:
    """Raise if a history still violates the rules after sanitizing.

    Called by the loop in debug builds and by the property tests. A sanitizer
    that silently returns something illegal is worse than no sanitizer, because
    the resulting 400 points at the provider rather than at here.
    """
    if rules.require_tool_result_pairing:
        asked = {call.id for m in messages for call in m.tool_uses()}
        answered = {block.tool_use_id for m in messages for block in m.tool_results()}
        if unanswered := asked - answered:
            msg = f"tool calls left unanswered after sanitizing: {sorted(unanswered)}"
            raise AssertionError(msg)
        if orphans := answered - asked:
            msg = f"tool results with no matching call: {sorted(orphans)}"
            raise AssertionError(msg)

    if rules.drop_unsigned_thinking:
        for message in messages:
            for block in message.content:
                if isinstance(block, ThinkingBlock) and not block.replayable:
                    msg = "an unsigned thinking block survived sanitizing"
                    raise AssertionError(msg)

    if rules.require_alternating_roles:
        previous = ""
        for message in messages:
            current = _wire_role(message.role, rules)
            if current == previous:
                msg = f"consecutive {current!r} turns survived sanitizing"
                raise AssertionError(msg)
            previous = current

    if rules.tool_results_in_user_message:
        _assert_results_lead(messages, rules)


def _assert_results_lead(messages: Sequence[Message], rules: SanitizeRules) -> None:
    """Raise if any user-role turn carries content ahead of its tool results."""
    for message in messages:
        if _wire_role(message.role, rules) != "user":
            continue
        kinds = [block.kind for block in message.content]
        if "tool_result" not in kinds:
            continue
        if any(kind != "tool_result" for kind in kinds[: kinds.index("tool_result")]):
            msg = (
                "a user turn carries content before its tool_result blocks, which Anthropic rejects"
            )
            raise AssertionError(msg)
