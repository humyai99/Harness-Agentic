"""The ``memory`` toolset: recording facts that outlive the session.

Four narrow tools rather than one ``memory_manage`` taking a verb. The model
picks the right one more reliably, and an audit log of ``memory_remove`` reads
as something that happened rather than as a parameter somebody has to decode.

The tools describe the *limit* rather than hiding it. Every answer says how much
room is left, because the model is the one that has to decide what earns a place
and it cannot see the file. When the file is full, ``memory_add`` fails with the
oldest entries named -- which is what makes "remove something first" an
instruction rather than a riddle.

What each tool cannot do is as deliberate: there is no way to read another
session's memory, no way to write outside the two files, and nothing here runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from harness_agentic.memory.manager import MAX_ENTRY_CHARS, MemoryFull
from harness_agentic.tools.spec import Danger, ToolContext, ToolResult

if TYPE_CHECKING:
    from harness_agentic.memory.manager import MemoryStore
    from harness_agentic.tools.registry import ToolRegistry

KIND_HELP = "'memory' for facts about the work, 'user' for facts about the person."
FROZEN_NOTE = (
    "Written to disk. It will not appear in your context until the next session -- "
    "what you remember is a snapshot taken at session start so it stays inside the "
    "cached part of the prompt."
)


class AddParams(BaseModel):
    """Arguments for ``memory_add``."""

    kind: str = Field(description=KIND_HELP, pattern="^(memory|user)$")
    text: str = Field(
        min_length=1,
        max_length=MAX_ENTRY_CHARS,
        description=(
            "One fact, stated so it is still useful months from now. Not a "
            "procedure -- those belong in a skill, where they cost nothing until "
            "loaded."
        ),
    )
    section: str = Field(
        "", description="Optional heading to group it under, e.g. 'Build' or 'Style'."
    )


class ReplaceParams(BaseModel):
    """Arguments for ``memory_replace``."""

    kind: str = Field(description=KIND_HELP, pattern="^(memory|user)$")
    old: str = Field(min_length=1, description="Text identifying the entry to change.")
    new: str = Field(min_length=1, max_length=MAX_ENTRY_CHARS, description="What it becomes.")


class RemoveParams(BaseModel):
    """Arguments for ``memory_remove``."""

    kind: str = Field(description=KIND_HELP, pattern="^(memory|user)$")
    text: str = Field(min_length=1, description="Text identifying the entry to delete.")


class ListParams(BaseModel):
    """Arguments for ``memory_list``."""

    kind: str = Field("", description=f"{KIND_HELP} Omit for both.")


def install_memory_tools(target: ToolRegistry, store: MemoryStore) -> ToolRegistry:
    """Register the memory tools against one store."""

    @target.tool(toolset="memory", danger=Danger.WRITES, override=True)
    def memory_add(params: AddParams, ctx: ToolContext) -> ToolResult:
        """Remember one fact for future sessions.

        Record what would change your answer to a question you have not been
        asked yet: how this project builds, what the user prefers, a constraint
        that is not written down anywhere. Do not record what you can look up in
        a second, and do not record a procedure -- that is a skill.

        Space is limited and shared with every future session, so adding
        something means deciding it matters more than what is already there.
        """
        try:
            entry = store.add(params.kind, params.text, section=params.section)
        except MemoryFull as exc:
            return ToolResult.error(str(exc))
        except ValueError as exc:
            return ToolResult.error(str(exc))

        state = store.load(params.kind)
        ctx.emit(f"remembered: {entry.text[:60]}")
        return ToolResult(
            text=f"Recorded in {params.kind}. {state.remaining} characters left.\n{FROZEN_NOTE}",
            display=f"memory_add: {entry.text[:50]}",
        )

    @target.tool(toolset="memory", danger=Danger.WRITES, override=True)
    def memory_replace(params: ReplaceParams, ctx: ToolContext) -> ToolResult:
        """Update a fact that has changed.

        Preferred over remove-then-add: it is one operation, it keeps the entry
        where it was, and it cannot half-succeed and lose the original.
        """
        try:
            found = store.replace(params.kind, params.old, params.new)
        except (MemoryFull, ValueError) as exc:
            return ToolResult.error(str(exc))
        if not found:
            return ToolResult.error(
                f"no entry in {params.kind} matching {params.old!r}; "
                f"use memory_list to see what is there"
            )
        ctx.emit(f"updated memory: {params.new[:60]}")
        state = store.load(params.kind)
        return ToolResult(
            text=f"Updated. {state.remaining} characters left.\n{FROZEN_NOTE}",
            display=f"memory_replace: {params.new[:50]}",
        )

    @target.tool(toolset="memory", danger=Danger.WRITES, override=True)
    def memory_remove(params: RemoveParams, ctx: ToolContext) -> ToolResult:
        """Forget a fact that is wrong or no longer relevant.

        Removing is as much a part of keeping memory useful as adding: a file of
        stale facts costs the same tokens as a file of good ones and is worse
        than empty, because it is believed.
        """
        try:
            found = store.remove(params.kind, params.text)
        except ValueError as exc:
            return ToolResult.error(str(exc))
        if not found:
            return ToolResult.error(
                f"no entry in {params.kind} matching {params.text!r}; "
                f"use memory_list to see what is there"
            )
        ctx.emit(f"forgot: {params.text[:60]}")
        state = store.load(params.kind)
        return ToolResult(
            text=f"Removed. {state.remaining} characters left.\n{FROZEN_NOTE}",
            display=f"memory_remove: {params.text[:50]}",
        )

    @target.tool(toolset="memory", danger=Danger.SAFE, override=True)
    def memory_list(params: ListParams, ctx: ToolContext) -> ToolResult:
        """Show what is currently remembered, with how much room is left.

        Your context holds a snapshot from session start. Use this when you are
        about to change memory, so you are deciding against what is on disk now
        rather than against what was there when the session began.
        """
        del ctx
        kinds = [params.kind] if params.kind else list(store.KINDS)
        blocks: list[str] = []
        for kind in kinds:
            state = store.load(kind)
            body = state.render().strip() if state.entries else "(nothing recorded)"
            blocks.append(f"{body}\n[{state.used}/{state.limit} characters used]")
        return ToolResult(
            text="\n\n".join(blocks),
            display=f"memory_list: {', '.join(kinds)}",
        )

    return target
