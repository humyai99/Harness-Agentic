"""The filesystem-and-shell abstraction that makes remote backends free.

Every builtin tool reaches the outside world through an ``ExecEnvironment``.
Nothing under ``tools/`` calls ``open()``, ``Path.read_text()``, or
``subprocess`` directly, and ``scripts/check_io_boundary.py`` enforces that in
pre-commit and CI.

The payoff is concrete: switching ``env.backend`` from ``local`` to ``docker``
runs the identical, unmodified tools inside a container. If even one tool
reached the disk directly, that switch would be a lie -- half the work would
happen on the host, and the sandbox would be decorative.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import ClassVar, Self

from harness_agentic.core.cancel import NEVER_CANCELLED, CancelToken

DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_OUTPUT_BYTES = 1 << 20
DEFAULT_MAX_READ_BYTES = 1 << 22


@dataclass(frozen=True, slots=True)
class CommandResult:
    """The outcome of running one command."""

    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    truncated: bool = False
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        """Whether the command exited cleanly."""
        return self.exit_code == 0 and not self.timed_out


@dataclass(frozen=True, slots=True)
class FileStat:
    """Metadata for one filesystem entry."""

    path: PurePath
    size: int
    is_dir: bool
    is_symlink: bool
    mtime: float
    mode: int = 0


@dataclass(frozen=True, slots=True)
class DirEntry:
    """One entry from a directory listing."""

    name: str
    is_dir: bool
    size: int


class ExecEnvironment(ABC):
    """Where a tool's filesystem and process operations actually happen."""

    name: ClassVar[str]

    @property
    @abstractmethod
    def root(self) -> PurePath:
        """The workspace root inside this environment."""

    # -- processes ----------------------------------------------------------

    @abstractmethod
    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: PurePath | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        stdin: str | None = None,
        cancel: CancelToken = NEVER_CANCELLED,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    ) -> CommandResult:
        """Run a command from an argument vector.

        Never uses a shell. Passing a list means no quoting rules to get wrong
        and no injection surface from model-generated strings.
        """

    @abstractmethod
    def run_shell(
        self,
        command: str,
        *,
        cwd: PurePath | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        cancel: CancelToken = NEVER_CANCELLED,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    ) -> CommandResult:
        """Run a command through a shell.

        Separate from :meth:`run` and separately audited, because pipes and
        globs are genuinely useful and pretending otherwise just pushes people
        into worse workarounds. Tools exposing this are classified destructive.
        """

    # -- filesystem ---------------------------------------------------------

    @abstractmethod
    def read_bytes(self, path: PurePath, *, max_bytes: int = DEFAULT_MAX_READ_BYTES) -> bytes:
        """Read a file, refusing anything larger than ``max_bytes``."""

    @abstractmethod
    def write_bytes(self, path: PurePath, data: bytes, *, mkdirs: bool = True) -> None:
        """Write a file, creating parent directories when asked."""

    @abstractmethod
    def list_dir(self, path: PurePath) -> list[DirEntry]:
        """List one directory, without recursing."""

    @abstractmethod
    def stat(self, path: PurePath) -> FileStat | None:
        """Return metadata, or ``None`` when the path does not exist."""

    @abstractmethod
    def realpath(self, path: PurePath) -> PurePath:
        """Resolve symlinks.

        Access-control checks must run against the *resolved* path. Checking
        the path as written lets a symlink inside the workspace point anywhere
        on the host.
        """

    @abstractmethod
    def glob(self, pattern: str, *, cwd: PurePath | None = None) -> list[PurePath]:
        """Expand a glob relative to ``cwd`` (default: the workspace root)."""

    # -- transfer -----------------------------------------------------------

    def upload(self, local: Path, remote: PurePath) -> None:
        """Copy a host file into the environment."""
        self.write_bytes(remote, local.read_bytes())

    def download(self, remote: PurePath, local: Path) -> None:
        """Copy a file out of the environment onto the host."""
        local.write_bytes(self.read_bytes(remote))

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:  # noqa: B027  -- optional hook; local needs nothing
        """Release any container or connection this environment holds."""

    def __enter__(self) -> Self:
        """Enter a context that closes the environment on exit."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the environment."""
        self.close()


def truncate_output(text: str, max_bytes: int) -> tuple[str, bool]:
    """Trim ``text`` to ``max_bytes``, marking where content was removed.

    Cuts from the middle: the head of a command's output says what it was doing
    and the tail says how it ended, while the middle is usually the repetitive
    part. Truncation is always announced -- silently shortened output is how a
    model concludes a build succeeded because it never saw the error.
    """
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= max_bytes:
        return text, False
    keep = max_bytes // 2
    head = raw[:keep].decode("utf-8", errors="ignore")
    tail = raw[-keep:].decode("utf-8", errors="ignore")
    elided = len(raw) - (2 * keep)
    return f"{head}\n\n[... {elided} bytes elided ...]\n\n{tail}", True
