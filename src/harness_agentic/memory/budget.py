"""Deciding when the conversation no longer fits.

Estimation is always approximate, so the design treats it that way. A budget
that assumed its own arithmetic would produce a hard overflow at 200k tokens
from a five-percent error -- and a five-percent error is normal. So the
estimate is advisory, ``ContextOverflow`` is a first-class recoverable
outcome, and every real response teaches the estimator something.

The self-correction matters more than the initial accuracy. Estimator error is
*systematic* per model -- a given tokenizer is consistently high or low for a
given kind of content -- which makes it learnable. After a handful of turns the
correction factor is worth more than any hand-tuned constant.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

from harness_agentic.core.types import (
    ImageBlock,
    Message,
    SystemPrompt,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolSchema,
    ToolUseBlock,
)

CHARS_PER_TOKEN = 3.6
"""Empirical average across English, code, and Thai. Calibrated away at runtime."""

IMAGE_TOKENS = 1_600
"""A rough cost for one inline image; providers differ by a lot."""

PER_MESSAGE_OVERHEAD = 4
"""Role markers and delimiters that every provider adds per message."""


def estimate_tokens(text: str) -> int:
    """Estimate tokens for a string."""
    return max(1, int(len(text) / CHARS_PER_TOKEN)) if text else 0


def estimate_message(message: Message) -> int:
    """Estimate tokens for one message, including its non-text blocks."""
    total = PER_MESSAGE_OVERHEAD
    for block in message.content:
        match block:
            case TextBlock(text=text) | ThinkingBlock(text=text):
                total += estimate_tokens(text)
            case ToolUseBlock(name=name, arguments=arguments):
                total += estimate_tokens(name) + estimate_tokens(json.dumps(dict(arguments)))
            case ToolResultBlock(text=text):
                total += estimate_tokens(text)
            case ImageBlock():
                total += IMAGE_TOKENS
    return total


def estimate_tools(tools: Sequence[ToolSchema]) -> int:
    """Estimate tokens for the tool definitions.

    Easy to forget and surprisingly large: a dozen tools with real descriptions
    is a few thousand tokens on every single request.
    """
    return sum(
        estimate_tokens(tool.name)
        + estimate_tokens(tool.description)
        + estimate_tokens(json.dumps(dict(tool.parameters)))
        for tool in tools
    )


@dataclass
class TokenBudget:
    """Tracks how close a conversation is to its window, and calibrates itself."""

    window: int
    reserve_output: int = 8_192
    headroom: float = 0.10
    """Structural slack for what the estimator cannot see."""
    compact_at: float = 0.80
    """Fraction of the usable window that triggers compaction."""
    correction: float = 1.0
    """Learned multiplier on raw estimates, per model."""

    _samples: int = field(default=0, repr=False)
    last_estimate: int = field(default=0, repr=False)
    """The corrected estimate, which is what the loop compares to the threshold."""
    last_raw: int = field(default=0, repr=False)
    """The uncorrected estimate. Calibration is measured against this one."""

    @property
    def usable(self) -> int:
        """Tokens available for input after output and headroom are reserved."""
        return max(1, int(self.window * (1.0 - self.headroom)) - self.reserve_output)

    @property
    def compact_threshold(self) -> int:
        """The estimate at which compaction should run."""
        return int(self.usable * self.compact_at)

    def estimate(
        self,
        system: SystemPrompt,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema] = (),
    ) -> int:
        """Estimate the input size of a request, with the learned correction."""
        raw = (
            estimate_tokens(system.rendered())
            + estimate_tools(tools)
            + sum(estimate_message(m) for m in messages)
        )
        self.last_raw = raw
        self.last_estimate = int(raw * self.correction)
        return self.last_estimate

    def observe(self, actual: int, *, raw: int | None = None) -> None:
        """Learn from a real response's reported input tokens.

        The correction is an exponential moving average of ``actual / raw``,
        measured against the **uncorrected** estimate. Measuring against the
        corrected one would fold the correction into its own input and the
        factor would compound away from reality instead of converging on it.

        ``raw`` defaults to the last estimate this budget produced, so the
        caller cannot accidentally pass the wrong one.
        """
        baseline = raw if raw is not None else self.last_raw
        if baseline <= 0 or actual <= 0:
            return
        ratio = actual / baseline
        weight = 0.3 if self._samples else 1.0
        self.correction = (1 - weight) * self.correction + weight * ratio
        self.correction = min(max(self.correction, 0.5), 3.0)
        self._samples += 1

    def shrink_from_error(self) -> None:
        """React to a real overflow.

        A hard bump rather than a nudge: the estimator was wrong in the
        direction that costs a round trip, and being conservative afterwards is
        much cheaper than overflowing again.
        """
        self.correction = min(self.correction * 1.25, 3.0)

    def retarget(self, *, window: int, reserve_output: int | None = None) -> None:
        """Point the budget at a different model.

        The correction factor is reset: it was learned for the previous
        tokenizer and carrying it across would be worse than starting over.
        """
        self.window = window
        if reserve_output is not None:
            self.reserve_output = reserve_output
        self.correction = 1.0
        self._samples = 0

    def over_threshold(self, estimate: int | None = None) -> bool:
        """Whether the given (or last) estimate calls for compaction."""
        return (estimate if estimate is not None else self.last_estimate) > self.compact_threshold

    def usage_ratio(self, estimate: int | None = None) -> float:
        """How full the usable window is, as a fraction."""
        value = estimate if estimate is not None else self.last_estimate
        return value / self.usable
