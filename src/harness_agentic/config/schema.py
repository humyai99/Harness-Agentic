"""The settings schema.

Every section forbids unknown keys. A typo in a config file is then an error
naming the key rather than a setting that silently does nothing -- which is the
failure that costs an afternoon, because the file looks right and the behaviour
is not.

Only settings something actually reads are modelled. A schema full of knobs
nothing consults is worse than no schema: it documents behaviour the code does
not have, and the first person to set one of them will believe it worked.

Secrets are deliberately absent. They live in ``.env`` or the keyring and are
referred to by name, so a config file can be committed, pasted into an issue, or
read out over a call without anybody having to think about it first.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Not deferred: pydantic resolves field annotations at runtime, so a
# TYPE_CHECKING import would make this model fail to build.
from harness_agentic.memory.manager import MEMORY_LIMIT, USER_LIMIT
from harness_agentic.tools.approval import Mode  # noqa: TC001

STRICT = ConfigDict(extra="forbid")


class ModelSettings(BaseModel):
    """Which model to use, and what to fall back to."""

    model_config = STRICT

    default: str = "anthropic/claude-sonnet-5"
    fallbacks: tuple[str, ...] = ()
    """Tried in order when the primary is exhausted. Logged when it happens, so
    a quietly degraded run is visible rather than merely cheaper."""
    stream: bool = True
    max_iterations: int = Field(40, gt=0, le=500)
    max_output_tokens: int | None = None


class ToolSettings(BaseModel):
    """What the agent may do, and where."""

    model_config = STRICT

    toolsets: tuple[str, ...] = ("file", "terminal")
    disabled: tuple[str, ...] = ()
    """Replaces rather than extends. A project that sets this means *this list*;
    ``disabled_append`` is the separate spelling for adding to what is inherited,
    because guessing which one somebody meant gets it wrong half the time."""
    disabled_append: tuple[str, ...] = ()
    env: str = "local"
    """``local`` or ``docker``."""


class ApprovalSettings(BaseModel):
    """Who may say yes, per surface."""

    model_config = STRICT

    modes: dict[str, Mode] = Field(default_factory=dict)
    """Merged over the defaults, so a file naming one surface does not silently
    reset the others to permissive."""
    allowlist: tuple[str, ...] = ()


class DockerSettings(BaseModel):
    """The sandbox, when ``tools.env`` is ``docker``."""

    model_config = STRICT

    image: str = "python:3.12-slim-bookworm"
    network: str = "none"
    memory: str = "2g"
    cpus: str = "2.0"
    pids: int = Field(512, gt=0)
    read_only_root: bool = True


class SkillSettings(BaseModel):
    """The skill library and how fast it may change."""

    model_config = STRICT

    enabled: bool = True
    autonomy: str = "propose"
    """``propose``, ``auto-safe`` or ``auto``. The default is a security
    decision: a poisoned skill costs every future session whose request matches
    its description."""
    catalog_budget: int = Field(2500, gt=0)
    external_dirs: tuple[str, ...] = ()


class MemorySettings(BaseModel):
    """Facts carried between sessions, and what they are allowed to cost."""

    model_config = STRICT

    enabled: bool = True
    memory_limit: int = Field(MEMORY_LIMIT, gt=0)
    """Characters in ``MEMORY.md``. Configurable, but raising it is not free: this
    text is in the prompt on every turn of every session from now on."""
    user_limit: int = Field(USER_LIMIT, gt=0)


class GatewaySettings(BaseModel):
    """Chat platforms, and who is allowed to talk to them."""

    model_config = STRICT

    platforms: tuple[str, ...] = ()
    allow_all: bool = False
    """Warned about loudly at startup. Anyone who finds the bot can spend money."""
    host: str = "127.0.0.1"
    port: int = Field(8787, gt=0, lt=65536)
    max_concurrent_runs: int = Field(4, gt=0)


class Settings(BaseModel):
    """Everything configurable, in one object."""

    model_config = STRICT

    model: ModelSettings = Field(default_factory=ModelSettings)
    tools: ToolSettings = Field(default_factory=ToolSettings)
    approval: ApprovalSettings = Field(default_factory=ApprovalSettings)
    docker: DockerSettings = Field(default_factory=DockerSettings)
    skills: SkillSettings = Field(default_factory=SkillSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    gateway: GatewaySettings = Field(default_factory=GatewaySettings)

    def enabled_toolsets(self) -> list[str]:
        """The toolsets to offer, with the disabled ones removed."""
        removed = {*self.tools.disabled, *self.tools.disabled_append}
        return [name for name in self.tools.toolsets if name not in removed]
