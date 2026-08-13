"""Run tools inside a hardened container.

The claim :mod:`harness_agentic.envs.base` makes is that switching the backend
runs the *identical, unmodified* tools inside a container -- and that only holds
if every operation goes through the container. So the filesystem methods here
are ``docker exec`` calls too, not host reads against the bind mount. That is
slower, and it is the difference between a sandbox and a decoration: a symlink
inside the workspace pointing at ``/etc/passwd`` has to resolve to the
*container's* ``/etc/passwd``, and a host read would follow it out.

**One container per environment, kept alive.** A container per command would
throw away the working directory, anything installed, and every background
process between one call and the next -- so `cd build && make` would not work,
and neither would half of what a developer does in a shell. The container is
created on first use and removed on :meth:`close`.

The hardening is the point of the exercise, so it is spelled out rather than
left to defaults:

* no network unless the operator asks for one, because a sandbox with egress is
  a sandbox an exfiltration payload does not notice;
* a read-only root filesystem with ``tmpfs`` where writes are legitimately
  needed, so a compromised process cannot leave anything behind outside the
  workspace;
* every capability dropped, ``no-new-privileges`` set, so a setuid binary in the
  image is not a way out;
* memory, CPU and pid limits, because a fork bomb inside a container is still a
  fork bomb on the host's scheduler;
* a non-root user, since ``root`` in a container is ``root`` on the host the
  moment anything else goes wrong.
"""

from __future__ import annotations

import base64
import json
import shlex
import subprocess
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path, PurePath, PurePosixPath
from typing import ClassVar

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
from harness_agentic.errors import HarnessError, PathOutsideWorkspace

DEFAULT_IMAGE = "python:3.12-slim-bookworm"
WORKSPACE_MOUNT = PurePosixPath("/workspace")
DEFAULT_MEMORY = "2g"
DEFAULT_CPUS = "2.0"
DEFAULT_PIDS = 512
DEFAULT_TMPFS_SIZE = "256m"
STARTUP_TIMEOUT_S = 120.0
DOCKER_OVERHEAD_S = 15.0
"""Added to a command's own timeout when waiting on ``docker exec``, so a slow
daemon is not reported as the command timing out."""


class DockerUnavailable(HarnessError):
    """Docker is not installed, not running, or refused the container."""


@dataclass
class DockerLimits:
    """Resource ceilings for the container.

    Defaults are deliberately modest. A sandbox exists to bound damage, and an
    unbounded one bounds nothing -- a fork bomb inside a container still competes
    for the host's scheduler.
    """

    memory: str = DEFAULT_MEMORY
    cpus: str = DEFAULT_CPUS
    pids: int = DEFAULT_PIDS
    tmpfs_size: str = DEFAULT_TMPFS_SIZE


@dataclass
class DockerEnvironment(ExecEnvironment):
    """Execute in a container, with the workspace bind-mounted.

    ``network`` is off by default. Turning it on is a deliberate act, because a
    container that can reach the internet can also reach whatever is on the
    operator's private network -- and the URL policy that guards ``web_fetch``
    does not apply to a command the model wrote.
    """

    workspace: Path
    image: str = DEFAULT_IMAGE
    network: str = "none"
    user: str = "1000:1000"
    limits: DockerLimits = field(default_factory=DockerLimits)
    docker: str = "docker"
    read_only_root: bool = True
    extra_env: Mapping[str, str] = field(default_factory=dict)
    container: str = ""
    """Set once started. Exposed so an operator can `docker exec` into it."""
    label: str = ""
    """The container's name. Generated once per environment; a fresh one per call
    would make `create_argv` unreproducible and leave `exec_argv` naming a
    container that was never created."""

    name: ClassVar[str] = "docker"

    def __post_init__(self) -> None:
        """Fix the container's name for this environment's lifetime."""
        if not self.label:
            # Prefixed so a stray container is identifiable as ours in `docker ps`.
            self.label = f"harn-{uuid.uuid4().hex[:10]}"

    # -- lifecycle -------------------------------------------------------------

    def create_argv(self) -> list[str]:
        """The full ``docker run`` command line.

        A method rather than an inline string so the hardening can be asserted in
        a test without a daemon. Every flag here is load-bearing; a silently
        dropped one turns the sandbox into a wrapper.
        """
        argv = [
            self.docker,
            "run",
            "--detach",
            "--name",
            self.label,
            # Sleeps forever so the container outlives each command: `cd build &&
            # make` has to still be in `build` on the next call.
            "--entrypoint",
            "sh",
            "--network",
            self.network,
            "--user",
            self.user,
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--memory",
            self.limits.memory,
            "--cpus",
            self.limits.cpus,
            "--pids-limit",
            str(self.limits.pids),
            "--workdir",
            str(WORKSPACE_MOUNT),
            "--volume",
            f"{self.workspace.resolve()}:{WORKSPACE_MOUNT}:rw",
        ]
        if self.read_only_root:
            # Writable only where a toolchain genuinely needs it. Without the
            # tmpfs mounts a read-only root breaks pip, git and most compilers,
            # and the usual response to that is to turn the whole thing off.
            argv += [
                "--read-only",
                "--tmpfs",
                # container, not a path on the host.
                f"/tmp:rw,size={self.limits.tmpfs_size},mode=1777",  # noqa: S108
                "--tmpfs",
                "/run:rw,size=16m",
                "--tmpfs",
                f"/home/user:rw,size={self.limits.tmpfs_size}",
            ]
        for key, value in self.extra_env.items():
            argv += ["--env", f"{key}={value}"]
        argv += [self.image, "-c", "sleep infinity"]
        return argv

    def start(self) -> None:
        """Create the container, or explain what is wrong."""
        if self.container:
            return
        if not self.workspace.is_dir():
            detail = f"the workspace {self.workspace} does not exist"
            raise DockerUnavailable(detail)
        try:
            completed = subprocess.run(  # noqa: S603 - an argv we built, no shell
                self.create_argv(),
                capture_output=True,
                text=True,
                timeout=STARTUP_TIMEOUT_S,
                check=False,
            )
        except FileNotFoundError as exc:
            detail = (
                f"{self.docker!r} is not installed. Install Docker, or run with "
                f"the local environment and accept that tools run unsandboxed."
            )
            raise DockerUnavailable(detail) from exc
        except subprocess.TimeoutExpired as exc:
            detail = f"docker did not start a container within {STARTUP_TIMEOUT_S:.0f}s"
            raise DockerUnavailable(detail) from exc

        if completed.returncode != 0:
            detail = f"could not start a container: {completed.stderr.strip() or 'unknown error'}"
            raise DockerUnavailable(detail)
        self.container = completed.stdout.strip()[:12] or self.label

    def close(self) -> None:
        """Remove the container.

        Forced, and failures are swallowed: this runs on the way out, and a
        container that will not die politely still has to be attempted -- while a
        raise here would mask whatever the caller was actually doing.
        """
        if not self.container:
            return
        name, self.container = self.container, ""
        with suppress(Exception):
            subprocess.run(  # noqa: S603 - an argv we built, no shell
                [self.docker, "rm", "--force", name],
                capture_output=True,
                timeout=STARTUP_TIMEOUT_S,
                check=False,
            )

    @property
    def root(self) -> PurePath:
        """The workspace root *inside* the container."""
        return WORKSPACE_MOUNT

    # -- paths -----------------------------------------------------------------

    def _resolve(self, path: PurePath) -> PurePosixPath:
        """Map a path into the container and confirm it stays in the workspace.

        Lexical, and then :meth:`realpath` is what handles symlinks -- the same
        split as the local environment, and for the same reason: containment has
        to be decided on the resolved path, and resolving happens in the
        container.
        """
        candidate = PurePosixPath(str(path).replace("\\", "/"))
        if not candidate.is_absolute():
            candidate = WORKSPACE_MOUNT / candidate
        # PurePosixPath does not normalize `..`, so it is done here rather than
        # trusted: `/workspace/../etc/passwd` is inside the root by string
        # prefix and outside it in fact.
        parts: list[str] = []
        for part in candidate.parts:
            if part == "..":
                if parts[1:]:
                    parts.pop()
                continue
            if part != ".":
                parts.append(part)
        normalized = PurePosixPath(*parts)
        if normalized != WORKSPACE_MOUNT and WORKSPACE_MOUNT not in normalized.parents:
            detail = f"{normalized} is outside the workspace root {WORKSPACE_MOUNT}"
            raise PathOutsideWorkspace(detail)
        return normalized

    def realpath(self, path: PurePath) -> PurePath:
        """Resolve symlinks inside the container."""
        target = PurePosixPath(str(path))
        if not target.is_absolute():
            target = WORKSPACE_MOUNT / target
        result = self._exec(["readlink", "-f", str(target)], timeout_s=30.0)
        resolved = result.stdout.strip()
        return PurePosixPath(resolved) if resolved else target

    # -- processes -------------------------------------------------------------

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
        """Run an argument vector in the container, without a shell."""
        return self._exec(
            list(argv),
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
        """Run a command line through the container's shell."""
        return self._exec(
            ["sh", "-c", command],
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
            cancel=cancel,
            max_output_bytes=max_output_bytes,
        )

    def exec_argv(
        self,
        inner: Sequence[str],
        *,
        cwd: PurePath | None = None,
        env: Mapping[str, str] | None = None,
        with_stdin: bool = False,
    ) -> list[str]:
        """The ``docker exec`` command line for one inner command.

        Separated out so it can be asserted without a daemon, which is most of
        what is worth checking here: that the working directory is inside the
        mount and that nothing is passed through a host shell.
        """
        argv = [self.docker, "exec"]
        if with_stdin:
            argv.append("--interactive")
        argv += ["--workdir", str(self._resolve(cwd) if cwd is not None else WORKSPACE_MOUNT)]
        for key, value in (env or {}).items():
            argv += ["--env", f"{key}={value}"]
        # The id once started, the name before: `docker exec` accepts either, and
        # this way `exec_argv` is inspectable in a test without a daemon.
        argv.append(self.container or self.label)
        argv += list(inner)
        return argv

    def _exec(
        self,
        inner: Sequence[str],
        *,
        cwd: PurePath | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        stdin: str | None = None,
        cancel: CancelToken = NEVER_CANCELLED,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    ) -> CommandResult:
        """Run one command in the container and normalize the outcome."""
        self.start()
        argv = self.exec_argv(inner, cwd=cwd, env=env, with_stdin=stdin is not None)
        started = time.monotonic()
        timed_out = False
        try:
            completed = subprocess.run(  # noqa: S603 - an argv we built, no shell
                argv,
                input=stdin,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                # The daemon's own latency is not the command's, so a slow
                # docker must not be reported as the command timing out.
                timeout=timeout_s + DOCKER_OVERHEAD_S,
                check=False,
            )
            code, out, err = completed.returncode, completed.stdout, completed.stderr
        except subprocess.TimeoutExpired as expired:
            timed_out = True
            code = -1
            out = _as_text(expired.stdout)
            err = _as_text(expired.stderr)
        if cancel.is_set() and not timed_out:
            # Cancellation cannot interrupt a `docker exec` already in flight, so
            # it is reported rather than pretended away.
            err = f"{err}\n(the run was cancelled)".strip()

        kept_out, out_trunc = truncate_output(out or "", max_output_bytes)
        kept_err, err_trunc = truncate_output(err or "", max_output_bytes)
        return CommandResult(
            exit_code=code,
            stdout=kept_out,
            stderr=kept_err,
            duration_s=time.monotonic() - started,
            truncated=out_trunc or err_trunc,
            timed_out=timed_out,
        )

    # -- filesystem ------------------------------------------------------------

    def read_bytes(self, path: PurePath, *, max_bytes: int = DEFAULT_MAX_READ_BYTES) -> bytes:
        """Read a file from inside the container.

        Through the container rather than through the bind mount, so a symlink in
        the workspace resolves against the *container's* filesystem. Base64 on the
        wire because the content is bytes and ``docker exec`` gives back text.
        """
        target = self._resolve(path)
        size = self._size_of(target)
        if size > max_bytes:
            detail = f"{target} is {size} bytes, over the {max_bytes} byte limit"
            raise ValueError(detail)
        result = self._exec(["base64", str(target)], timeout_s=60.0)
        if result.exit_code != 0:
            detail = f"could not read {target}: {result.stderr.strip()}"
            raise FileNotFoundError(detail)
        return base64.b64decode(result.stdout)

    def write_bytes(self, path: PurePath, data: bytes, *, mkdirs: bool = True) -> None:
        """Write a file inside the container, atomically."""
        target = self._resolve(path)
        quoted = shlex.quote(str(target))
        temporary = shlex.quote(f"{target}.harness-tmp")
        script = (
            (f"mkdir -p {shlex.quote(str(target.parent))} && " if mkdirs else "")
            # Write-then-rename here as well: a half-written file is worse than a
            # failed write, because the model reads it next turn and believes it.
            + f"base64 -d > {temporary} && "
            # The destination's mode is preserved when it exists, for the same
            # reason as on the host: an edit must not widen a 0600 file.
            + f"if [ -f {quoted} ]; then chmod --reference={quoted} {temporary} "
            + "2>/dev/null || true; fi && "
            + f"mv {temporary} {quoted}"
        )
        result = self._exec(
            ["sh", "-c", script],
            stdin=base64.b64encode(data).decode("ascii"),
            timeout_s=60.0,
        )
        if result.exit_code != 0:
            detail = f"could not write {target}: {result.stderr.strip()}"
            raise OSError(detail)

    def list_dir(self, path: PurePath) -> list[DirEntry]:
        """List one directory inside the container."""
        target = self._resolve(path)
        # `-printf` over parsing `ls`: ls output is for humans and its columns
        # shift with locale, filenames containing spaces, and platform.
        result = self._exec(
            ["find", str(target), "-mindepth", "1", "-maxdepth", "1", "-printf", "%y\\t%s\\t%f\\n"],
            timeout_s=60.0,
        )
        if result.exit_code != 0:
            detail = f"could not list {target}: {result.stderr.strip()}"
            raise FileNotFoundError(detail)
        entries: list[DirEntry] = []
        for line in result.stdout.splitlines():
            kind, _, rest = line.partition("\t")
            size_text, _, name = rest.partition("\t")
            if not name:
                continue
            entries.append(
                DirEntry(
                    name=name,
                    is_dir=kind == "d",
                    size=int(size_text) if size_text.isdigit() and kind != "d" else 0,
                )
            )
        return sorted(entries, key=lambda entry: (not entry.is_dir, entry.name))

    def stat(self, path: PurePath) -> FileStat | None:
        """Return metadata from inside the container, or ``None`` if absent."""
        target = self._resolve(path)
        result = self._exec(
            ["find", str(target), "-maxdepth", "0", "-printf", "%y\\t%s\\t%T@\\t%m\\n"],
            timeout_s=30.0,
        )
        if result.exit_code != 0 or not result.stdout.strip():
            return None
        fields = [*result.stdout.strip().split("\t"), "", "", ""]
        kind, size_text, mtime_text, mode_text = fields[:4]
        return FileStat(
            path=target,
            size=int(size_text) if size_text.isdigit() else 0,
            is_dir=kind == "d",
            is_symlink=kind == "l",
            mtime=float(mtime_text) if mtime_text.replace(".", "", 1).isdigit() else 0.0,
            mode=int(mode_text, 8) if mode_text.isdigit() else 0,
        )

    def glob(self, pattern: str, *, cwd: PurePath | None = None) -> list[PurePath]:
        """Expand a glob inside the container, dropping anything that escapes."""
        base = self._resolve(cwd) if cwd is not None else WORKSPACE_MOUNT
        # `find` rather than the shell's own globbing: a pattern the model wrote
        # must not reach a shell, and `**` is not portable across shells anyway.
        depth = ["-maxdepth", "1"] if "**" not in pattern else []
        name = pattern.replace("**/", "")
        result = self._exec(
            ["find", str(base), *depth, "-name", name, "-print"],
            timeout_s=60.0,
        )
        found: list[PurePath] = []
        for line in result.stdout.splitlines():
            candidate = line.strip()
            if not candidate:
                continue
            with suppress(PathOutsideWorkspace):
                found.append(self._resolve(PurePosixPath(candidate)))
        return sorted(found)

    def _size_of(self, target: PurePosixPath) -> int:
        """The size of one file, or zero when it cannot be determined."""
        info = self.stat(target)
        return info.size if info else 0

    def describe(self) -> str:
        """One line for ``harn doctor``."""
        return (
            f"docker {self.image} network={self.network} user={self.user} "
            f"memory={self.limits.memory} cpus={self.limits.cpus} "
            f"read_only_root={self.read_only_root}"
        )


def probe(docker: str = "docker") -> str:
    """Return the Docker server version, or raise :class:`DockerUnavailable`.

    Used by ``harn doctor`` and before a run that asked for the sandbox, so the
    failure is "the daemon is not running" up front rather than a confusing
    error on the first tool call.
    """
    try:
        completed = subprocess.run(  # noqa: S603 - an argv we built, no shell
            [docker, "version", "--format", "{{json .Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=20.0,
            check=False,
        )
    except FileNotFoundError as exc:
        detail = f"{docker!r} is not installed"
        raise DockerUnavailable(detail) from exc
    except subprocess.TimeoutExpired as exc:
        detail = "docker did not answer within 20s"
        raise DockerUnavailable(detail) from exc
    if completed.returncode != 0:
        detail = f"the docker daemon is not reachable: {completed.stderr.strip()}"
        raise DockerUnavailable(detail)
    try:
        return str(json.loads(completed.stdout.strip()))
    except ValueError:
        return completed.stdout.strip()


def _as_text(raw: object) -> str:
    """Decode whatever ``TimeoutExpired`` captured, which may be bytes or None."""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw) if raw else ""
