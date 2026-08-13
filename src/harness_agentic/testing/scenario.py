"""Driving the whole stack from a file, with no provider and no key.

``fake/scripted`` is a real entry in the provider catalog, so every surface --
``harn run``, ``harn chat``, the gateway, the web UI, a cron job -- can be
exercised end to end by pointing ``HARNESS_FAKE_SCRIPT`` at a scenario. That is
the point: a demo, a reproduction attached to a bug report, or a CI check of the
gateway should not need a credential, a network, or a bill.

The scenario is the model's half of the conversation, written down in advance::

    turns:
      - tool: read_file
        arguments: {path: pyproject.toml}
      - text: It depends on fastapi and httpx.

Everything else is real. Real tool dispatch against the real filesystem, real
approval policy, real session store, real prompt assembly with real cache
breakpoints. Only the model is replaced, which is what makes a failure here a
failure of the harness rather than of the weather.

Parsed with the frontmatter reader the skills library already uses, so a
scenario file costs no dependency. That reader takes a small, inert subset of
YAML -- no anchors, no tags, nothing that executes -- which is the right amount
of YAML for a file the agent may be pointed at by someone else.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from harness_agentic.errors import HarnessError, SkillValidationError
from harness_agentic.skills.frontmatter import parse_frontmatter
from harness_agentic.testing.fakes import FakeTransport, ScriptedTurn

if TYPE_CHECKING:
    from harness_agentic.core.clock import Clock
    from harness_agentic.providers.base import Credentials

SCRIPT_ENV = "HARNESS_FAKE_SCRIPT"
"""Where to find the scenario. Named in the error when it is not set, because
"no script" is otherwise indistinguishable from "the model said nothing"."""


class ScenarioError(HarnessError):
    """A scenario file could not be read."""


def parse_scenario(raw: str) -> list[ScriptedTurn]:
    """Read a scenario into the turns the fake provider will replay.

    A malformed turn raises rather than being skipped. A scenario that silently
    drops half its turns produces a run that ends early and looks like the agent
    simply decided to stop -- which is a deeply confusing thing to debug when the
    whole point of the file was to make the run predictable.
    """
    if not raw.strip():
        msg = "the scenario is empty; it needs a 'turns:' list"
        raise ScenarioError(msg)
    try:
        parsed = parse_frontmatter(raw)
    except SkillValidationError as exc:
        # Reported as a scenario problem rather than as a skill one. The reader
        # is shared, and its error names a line number in a file the person is
        # holding -- but "SkillValidationError" sends them looking at skills.
        msg = f"the scenario is not readable -- {exc}"
        raise ScenarioError(msg) from exc
    entries = parsed.get("turns")
    if entries is None:
        msg = "no 'turns:' key -- a scenario is a list of turns under it"
        raise ScenarioError(msg)
    if not isinstance(entries, list):
        msg = f"'turns:' should be a list, got {type(entries).__name__}"
        raise ScenarioError(msg)

    turns: list[ScriptedTurn] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            msg = f"turn {index + 1} is not a mapping: {entry!r}"
            raise ScenarioError(msg)
        turns.append(_turn(entry, index))
    if not turns:
        msg = "the scenario lists no turns"
        raise ScenarioError(msg)
    return turns


def _turn(entry: dict[str, Any], index: int) -> ScriptedTurn:
    """One scenario entry as a scripted turn."""
    tool = entry.get("tool")
    text = entry.get("text")
    thinking = entry.get("thinking")
    if tool:
        arguments = entry.get("arguments") or {}
        if not isinstance(arguments, dict):
            msg = f"turn {index + 1}: 'arguments' should be a mapping, got {arguments!r}"
            raise ScenarioError(msg)
        return ScriptedTurn(
            tool_calls=((str(tool), arguments),),
            text=str(text) if text else None,
            thinking=str(thinking) if thinking else None,
        )
    if text is None:
        msg = f"turn {index + 1} has neither 'text' nor 'tool': {entry!r}"
        raise ScenarioError(msg)
    return ScriptedTurn(text=str(text), thinking=str(thinking) if thinking else None)


def load_scenario(path: Path) -> list[ScriptedTurn]:
    """Read a scenario file, naming the file when it cannot be used."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"cannot read the scenario at {path}: {exc}"
        raise ScenarioError(msg) from exc
    try:
        return parse_scenario(raw)
    except ScenarioError as exc:
        msg = f"{path}: {exc}"
        raise ScenarioError(msg) from exc


def scripted_transport(
    credentials: Credentials, *, clock: Clock | None = None, model: str = "fake/scripted"
) -> FakeTransport:
    """Build the fake provider from ``HARNESS_FAKE_SCRIPT``.

    Refuses to start without one. A fake that answers something plausible when
    it has no script is worse than a fake that stops: the run appears to work,
    and whatever it proves is unrelated to what anyone wrote down.
    """
    location = os.environ.get(SCRIPT_ENV, "").strip()
    if not location:
        msg = (
            f"the fake provider needs a scenario: set {SCRIPT_ENV} to a YAML file "
            f"listing the turns to replay, or choose a real model with --model"
        )
        raise ScenarioError(msg)
    return FakeTransport(
        load_scenario(Path(location).expanduser()),
        credentials=credentials,
        clock=clock,
        model=model,
    )
