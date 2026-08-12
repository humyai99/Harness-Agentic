"""Cutting a long answer into messages a platform will accept.

Every chat platform caps message length -- LINE at 5000 characters, Telegram at
4096, Discord at 2000 -- and the naive fix, slicing every N characters, breaks
code fences in half and leaves the second message rendering as prose. Worse, it
splits mid-word in Thai, which has no spaces to fall back on.

So the split point is chosen, in order of preference: a blank line, then a line
break, then a sentence end, then a space, then wherever it has to be. Fenced
code blocks are tracked across the whole text, and a cut inside one closes the
fence and reopens it with the same info string on the other side, so both halves
render as code.
"""

from __future__ import annotations

import re

_FENCE = re.compile(r"^(`{3,}|~{3,})(.*)$")
SENTENCE_ENDS = (". ", "! ", "? ", "。", "! ", "? ", "ๆ ")
MIN_SPLIT_RATIO = 0.3
"""How far into the window a break point has to be to be worth taking.

A floor is needed -- a paragraph break at character 5 of a 4096-character
budget would produce a five-character message -- but it has to stay low.
Set it near the middle and a perfectly good paragraph break at 40% gets
skipped in favour of a mid-sentence cut at 99%, which is the outcome the
whole function exists to avoid.
"""


def split_for_platform(text: str, limit: int) -> list[str]:
    """Split ``text`` into chunks of at most ``limit`` characters.

    Returns a single-element list when it already fits, so the common case
    allocates nothing interesting. Never returns an empty list for empty input
    -- callers rely on ``len(chunks)`` matching the number of messages sent, and
    a zero-length answer still deserves one delivery attempt.
    """
    if limit <= 0:
        message = f"limit must be positive, got {limit}"
        raise ValueError(message)
    if len(text) <= limit:
        return [text]

    # Room to close a fence at the end of a chunk. The reopening fence is
    # charged separately, as a prefix on the following chunk.
    reserve = _fence_reserve(text)
    chunks: list[str] = []
    remaining = text
    open_fence = ""

    while True:
        prefix = f"{open_fence}\n" if open_fence else ""
        if len(prefix) + len(remaining) <= limit:
            chunks.append((prefix + remaining).rstrip())
            break

        budget = max(1, limit - len(prefix) - reserve)
        cut = _break_point(remaining, budget)
        piece = remaining[:cut]
        remaining = remaining[cut:].lstrip("\n")

        open_fence = _fence_after(open_fence, piece)
        body = (prefix + piece).rstrip()
        if open_fence:
            body = f"{body}\n{_bare(open_fence)}"
        chunks.append(body)
        if not remaining:
            break

    return [c for c in chunks if c] or [""]


def _fence_reserve(text: str) -> int:
    """Characters to hold back for a closing fence, if the text has any."""
    widest = 0
    for line in text.split("\n"):
        match = _FENCE.match(line.strip())
        if match:
            widest = max(widest, len(match.group(1)))
    return widest + 1 if widest else 0


def _break_point(text: str, limit: int) -> int:
    """Choose where to cut, preferring the largest structural boundary."""
    window = text[:limit]
    floor = int(limit * MIN_SPLIT_RATIO)

    para = window.rfind("\n\n")
    if para >= floor:
        return para + 2
    line = window.rfind("\n")
    if line >= floor:
        return line + 1

    best = max((window.rfind(end) for end in SENTENCE_ENDS), default=-1)
    if best >= floor:
        return best + 1
    space = window.rfind(" ")
    if space >= floor:
        return space + 1
    # Thai, Chinese and Japanese have no spaces to find. A hard cut is correct
    # here rather than a failure -- the alternative is a message over the limit,
    # which the platform rejects outright.
    return limit


def _fence_after(open_fence: str, piece: str) -> str:
    """The fence left open at the end of ``piece``, given what was open before."""
    current = open_fence
    for line in piece.split("\n"):
        match = _FENCE.match(line.strip())
        if not match:
            continue
        marker, info = match.group(1), match.group(2).strip()
        if current:
            # A closing fence must be at least as long as the opening one.
            if marker[0] == current[0] and len(marker) >= len(_bare(current)):
                current = ""
        else:
            current = f"{marker}{info}"
    return current


def _bare(fence: str) -> str:
    """The fence marker without its info string, for use as a closer."""
    match = _FENCE.match(fence)
    return match.group(1) if match else fence


def clip(text: str, limit: int, *, suffix: str = "…") -> str:
    """Truncate to ``limit`` characters, marking that something was cut.

    For status lines and previews, where splitting into several messages would
    be noise. Answers go through :func:`split_for_platform` instead -- silently
    dropping the end of an answer is how a user comes to distrust the agent.
    """
    if len(text) <= limit:
        return text
    if limit <= len(suffix):
        return text[:limit]
    return text[: limit - len(suffix)].rstrip() + suffix
