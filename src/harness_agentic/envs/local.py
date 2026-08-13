"""Run tools directly on the host.

This module is the one sanctioned place in the codebase for ``subprocess``, and
``pyproject.toml`` grants it the corresponding ``ruff`` exemptions by path.
Everything else goes through :class:`~harness_agentic.envs.base.ExecEnvironment`.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePath
from typing import IO, ClassVar

from harness_agentic.core.cancel import NEVER_CANCELLED, CancelToken
from harness_agentic.envs.base import (
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_MAX_READ_BYTES,
    DEFAULT_TIMEOUT_S,
    CommandResult,
    DirEntry,
    ExecEnvironment,
    FileStat,
    truncate_output,
)
from harness_agentic.errors import PathOutsideWorkspace

_POLL_INTERVAL_S = 0.05
_READ_CHUNK = 65_536
_DRAIN_JOIN_S = 5.0
_KEEP_MARGIN = 4_096
"""Read a little past the output cap so truncation is detectable rather than
landing exactly on the limit and looking complete."""


def _drain(stream: IO[str], into: dict[str, str], key: str, keep: int) -> None:
    """Read one pipe to EOF on a thread, keeping at most ``keep`` characters.

    Reading continues past the cap rather than stopping, because closing the pipe
    early sends the child SIGPIPE partway through its own output -- turning a
    verbose command into a failed one. What is dropped is the excess, not the
    child. Bounded at all because a command printing a gigabyte should not be
    held in memory in full before being truncated for display.
    """
    parts: list[str] = []
    kept = 0
    try:
        while chunk := stream.read(_READ_CHUNK):
            if kept < keep:
                wanted = chunk[: keep - kept]
                parts.append(wanted)
                kept += len(wanted)
    except (OSError, ValueError):
        # The pipe was closed under us by a kill. Whatever arrived is what there
        # is, and it is more useful than nothing.
        pass
    finally:
        into[key] = "".join(parts)
        with contextlib.suppress(OSError):
            stream.close()


class LocalEnvironment(ExecEnvironment):
    """Execute on the host filesystem, confined to a workspace root."""

    name: ClassVar[str] = "local"

    def __init__(self, root: Path, *, allow_outside_root: bool = False) -> None:
        """Confine operations to ``root`` unless explicitly told otherwise."""
        self._root = root.resolve()
        self._allow_outside = allow_outside_root

    @property
    def root(self) -> PurePath:
        """The workspace root."""
        return self._root

    # -- path handling ------------------------------------------------------

    def _resolve(self, path: PurePath) -> Path:
        """Resolve ``path`` and confirm it stays inside the workspace.

        Resolution happens *before* the containment check, so a symlink placed
        inside the workspace cannot be used to reach outside it.
        """
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self._root / candidate
        resolved = candidate.resolve()
        if self._allow_outside:
            return resolved
        if resolved != self._root and self._root not in resolved.parents:
            msg = f"{resolved} is outside the workspace root {self._root}"
            raise PathOutsideWorkspace(msg)
        return resolved

    def realpath(self, path: PurePath) -> PurePath:
        """Resolve symlinks without enforcing containment."""
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self._root / candidate
        return candidate.resolve()

    # -- processes ----------------------------------------------------------

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
        """Run an argument vector without a shell."""
        return self._spawn(
            list(argv),
            shell=False,
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
            stdin=stdin,
            cancel=cancel,
            max_output_bytes=max_output_bytes,
        )

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
        """Run a command line through the system shell."""
        return self._spawn(
            command,
            shell=True,
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
            stdin=None,
            cancel=cancel,
            max_output_bytes=max_output_bytes,
        )

    def _spawn(
        self,
        command: list[str] | str,
        *,
        shell: bool,
        cwd: PurePath | None,
        env: Mapping[str, str] | None,
        timeout_s: float,
        stdin: str | None,
        cancel: CancelToken,
        max_output_bytes: int,
    ) -> CommandResult:
        workdir = self._resolve(cwd) if cwd is not None else self._root
        environment = {**os.environ, **(env or {})}
        started = time.monotonic()

        # A new session so that cancelling kills the whole process tree.
        # Killing only the direct child leaves `uv run pytest`-style wrappers
        # behind, still holding the terminal and still burning CPU. POSIX only;
        # Windows ignores the flag and falls back to terminating the child.
        proc = subprocess.Popen(
            command,
            shell=shell,
            cwd=workdir,
            env=environment,
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=hasattr(os, "setsid"),
        )

        try:
            if stdin is not None and proc.stdin is not None:
                # A child that exited before reading its input is ordinary; its
                # exit code is the story, not a broken pipe here.
                with contextlib.suppress(OSError):
                    proc.stdin.write(stdin)
                    proc.stdin.close()
            stdout, stderr, timed_out = self._wait(
                proc, timeout_s=timeout_s, cancel=cancel, keep=max_output_bytes + _KEEP_MARGIN
            )
        except BaseException:
            # Including KeyboardInterrupt: leaving a process group running after
            # Ctrl-C is how a killed `harn` leaves a build burning CPU.
            self._kill_tree(proc)
            raise

        out, out_trunc = truncate_output(stdout or "", max_output_bytes)
        err, err_trunc = truncate_output(stderr or "", max_output_bytes)
        return CommandResult(
            exit_code=proc.returncode if proc.returncode is not None else -1,
            stdout=out,
            stderr=err,
            duration_s=time.monotonic() - started,
            truncated=out_trunc or err_trunc,
            timed_out=timed_out,
        )

    def _wait(
        self,
        proc: subprocess.Popen[str],
        *,
        timeout_s: float,
        cancel: CancelToken,
        keep: int,
    ) -> tuple[str, str, bool]:
        """Wait for the process. Returns ``(stdout, stderr, timed_out)``.

        Both pipes are drained on threads for the whole wait. Polling for exit
        *without* reading them deadlocks the moment the child writes more than
        the pipe buffer holds -- 64 KiB on Linux -- and the symptom is worse than
        a hang: the command is killed at its timeout and reported as having timed
        out, with its output cut off at exactly the buffer size. ``pytest -v``,
        ``git diff`` and any verbose build cross that line routinely, so the
        common case was a command that worked being reported as one that hung.

        One path for both cancellable and not, deliberately. The old fast path
        used ``communicate(timeout=…)``, which drains correctly, so the drainless
        branch was the one nothing exercised -- and the only one the agent ever
        took, since every tool call carries the loop's token.
        """
        collected: dict[str, str] = {}
        readers = [
            threading.Thread(
                target=_drain, args=(stream, collected, key, keep), name=f"harn-{key}", daemon=True
            )
            for key, stream in (("stdout", proc.stdout), ("stderr", proc.stderr))
            if stream is not None
        ]
        for reader in readers:
            reader.start()

        timed_out = False
        deadline = time.monotonic() + timeout_s
        while proc.poll() is None:
            if cancel.is_set():
                self._kill_tree(proc)
                break
            if time.monotonic() >= deadline:
                timed_out = True
                self._kill_tree(proc)
                break
            time.sleep(_POLL_INTERVAL_S)

        # The child is gone either way, so both pipes reach EOF and the readers
        # finish on their own. Joined so their output is complete before it is
        # read, and bounded so a grandchild still holding the pipe open -- a
        # backgrounded process inheriting stdout -- cannot wedge the turn.
        for reader in readers:
            reader.join(timeout=_DRAIN_JOIN_S)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=_DRAIN_JOIN_S)
        return collected.get("stdout", ""), collected.get("stderr", ""), timed_out

    @staticmethod
    def _kill_tree(proc: subprocess.Popen[str]) -> None:
        """Terminate the whole process group, escalating if it lingers."""
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
        except (ProcessLookupError, PermissionError, OSError):
            proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
            except (ProcessLookupError, PermissionError, OSError):
                proc.kill()

    # -- filesystem ---------------------------------------------------------

    def read_bytes(self, path: PurePath, *, max_bytes: int = DEFAULT_MAX_READ_BYTES) -> bytes:
        """Read a file, refusing anything over ``max_bytes``."""
        target = self._resolve(path)
        size = target.stat().st_size
        if size > max_bytes:
            msg = f"{target} is {size} bytes, over the {max_bytes} byte limit"
            raise ValueError(msg)
        return target.read_bytes()

    def write_bytes(self, path: PurePath, data: bytes, *, mkdirs: bool = True) -> None:
        """Write a file atomically, preserving its permissions.

        Write-then-rename, so an interrupted write does not leave the model
        looking at a half-truncated source file next turn. The rename carries the
        temporary file's mode with it, though, which silently widened every
        existing file it replaced: editing a ``0600`` file -- a ``.env``, a
        private key, an SSH config -- handed it back as ``0644``, readable by
        every local user. Nothing failed and nothing said so.

        So an existing file's mode is read first and restored before the rename,
        and the temporary file is created private while it holds the data: it sits
        in the workspace under a predictable name, and its contents may be exactly
        what the mode was protecting.
        """
        target = self._resolve(path)
        if mkdirs:
            target.parent.mkdir(parents=True, exist_ok=True)

        previous: int | None = None
        with contextlib.suppress(OSError):
            previous = target.stat().st_mode & 0o7777

        tmp = target.with_name(f".{target.name}.harness-tmp")
        try:
            if previous is None:
                # A new file: the usual default, as an ordinary create would give.
                tmp.write_bytes(data)
            else:
                descriptor = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(data)
                tmp.chmod(previous)
            tmp.replace(target)
        except BaseException:
            # A failed write must not leave `.name.harness-tmp` littering the
            # user's repository, where it shows up in the next `git status`.
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise

    def list_dir(self, path: PurePath) -> list[DirEntry]:
        """List a directory, directories first then names."""
        target = self._resolve(path)
        entries = [
            DirEntry(
                name=child.name,
                is_dir=child.is_dir(),
                size=child.stat().st_size if child.is_file() else 0,
            )
            for child in target.iterdir()
        ]
        return sorted(entries, key=lambda e: (not e.is_dir, e.name))

    def stat(self, path: PurePath) -> FileStat | None:
        """Return metadata, or ``None`` when the path is absent."""
        target = self._resolve(path)
        if not target.exists():
            return None
        info = target.stat()
        return FileStat(
            path=target,
            size=info.st_size,
            is_dir=target.is_dir(),
            is_symlink=target.is_symlink(),
            mtime=info.st_mtime,
            mode=info.st_mode,
        )

    def glob(self, pattern: str, *, cwd: PurePath | None = None) -> list[PurePath]:
        """Expand a glob, skipping anything that escapes the workspace."""
        base = self._resolve(cwd) if cwd is not None else self._root
        found: list[PurePath] = []
        for match in sorted(base.glob(pattern)):
            try:
                found.append(self._resolve(match))
            except PathOutsideWorkspace:
                continue
        return found
