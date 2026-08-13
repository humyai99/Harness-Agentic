"""The Docker backend: hardening, path containment, and output parsing.

No daemon is available in CI, so what runs by default is everything that can be
decided without one -- which is most of what matters here. The hardening flags
are the security control, and a control is only real if a silently dropped flag
fails a test rather than merely making the sandbox weaker.

The one test that needs a real container is marked ``integration`` and is
deselected by default. It is not skipped silently: a sandbox nobody has ever run
is a sandbox nobody should trust, so it exists to be run where a daemon does.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path, PurePosixPath

import pytest

from harness_agentic.envs.docker import (
    WORKSPACE_MOUNT,
    DockerEnvironment,
    DockerLimits,
    DockerUnavailable,
)
from harness_agentic.errors import PathOutsideWorkspace


@pytest.fixture
def env(tmp_path: Path) -> DockerEnvironment:
    return DockerEnvironment(workspace=tmp_path)


# -- hardening -----------------------------------------------------------------


def test_the_container_has_no_network_by_default(env: DockerEnvironment) -> None:
    """A sandbox with egress is a sandbox an exfiltration payload does not notice."""
    argv = env.create_argv()
    assert "--network" in argv
    assert argv[argv.index("--network") + 1] == "none"


def test_every_capability_is_dropped_and_privileges_cannot_grow(
    env: DockerEnvironment,
) -> None:
    argv = env.create_argv()
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    assert argv[argv.index("--security-opt") + 1] == "no-new-privileges"


def test_the_container_does_not_run_as_root(env: DockerEnvironment) -> None:
    # root in a container is root on the host the moment anything else goes wrong.
    argv = env.create_argv()
    assert argv[argv.index("--user") + 1] == "1000:1000"


def test_the_root_filesystem_is_read_only_with_tmpfs_where_needed(
    env: DockerEnvironment,
) -> None:
    """Without the tmpfs mounts a read-only root breaks pip, git and compilers.

    And the usual response to that breakage is to turn the whole thing off, which
    is why they are part of the hardening rather than a concession to it.
    """
    argv = env.create_argv()
    assert "--read-only" in argv
    mounted = {argv[i + 1].split(":")[0] for i, flag in enumerate(argv) if flag == "--tmpfs"}
    assert mounted == {"/tmp", "/run", "/home/user"}  # noqa: S108 - container paths


def test_resource_limits_are_always_set(env: DockerEnvironment) -> None:
    # A fork bomb inside a container is still a fork bomb on the host's scheduler.
    argv = env.create_argv()
    assert argv[argv.index("--memory") + 1] == "2g"
    assert argv[argv.index("--cpus") + 1] == "2.0"
    assert argv[argv.index("--pids-limit") + 1] == "512"


def test_limits_are_configurable(tmp_path: Path) -> None:
    env = DockerEnvironment(
        workspace=tmp_path, limits=DockerLimits(memory="512m", cpus="0.5", pids=64)
    )
    argv = env.create_argv()
    assert argv[argv.index("--memory") + 1] == "512m"
    assert argv[argv.index("--pids-limit") + 1] == "64"


def test_the_workspace_is_the_only_thing_mounted(env: DockerEnvironment, tmp_path: Path) -> None:
    argv = env.create_argv()
    volumes = [argv[i + 1] for i, flag in enumerate(argv) if flag == "--volume"]
    assert volumes == [f"{tmp_path.resolve()}:{WORKSPACE_MOUNT}:rw"]


def test_the_container_name_is_stable_across_calls(env: DockerEnvironment) -> None:
    """A fresh name per call would leave `exec_argv` naming a container that never existed."""
    first = env.create_argv()
    second = env.create_argv()
    assert first == second
    assert env.label in env.exec_argv(["true"])


def test_nothing_reaches_a_host_shell(env: DockerEnvironment) -> None:
    # The inner command may be shell-interpreted *inside* the container; the argv
    # `docker` itself is given must never be.
    argv = env.exec_argv(["sh", "-c", "rm -rf / # not on the host"])
    assert argv[0] == "docker"
    assert argv[1] == "exec"
    assert argv[-3:] == ["sh", "-c", "rm -rf / # not on the host"]


# -- paths ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("file.txt", "/workspace/file.txt"),
        ("sub/dir/file.txt", "/workspace/sub/dir/file.txt"),
        ("/workspace/file.txt", "/workspace/file.txt"),
        ("./file.txt", "/workspace/file.txt"),
        ("sub/../file.txt", "/workspace/file.txt"),
    ],
)
def test_paths_map_into_the_mount(env: DockerEnvironment, given: str, expected: str) -> None:
    assert env.exec_argv(["true"], cwd=PurePosixPath(given))[3] == expected


@pytest.mark.parametrize(
    "escape",
    [
        "/etc/passwd",
        "/workspace/../etc/passwd",
        "../../../etc/passwd",
        "/workspace/sub/../../etc",
    ],
)
def test_a_path_outside_the_workspace_is_refused(env: DockerEnvironment, escape: str) -> None:
    """``..`` has to be normalized rather than trusted.

    ``/workspace/../etc/passwd`` is inside the root by string prefix and outside
    it in fact, and ``PurePosixPath`` does not collapse ``..`` on its own.
    """
    with pytest.raises(PathOutsideWorkspace):
        env.exec_argv(["true"], cwd=PurePosixPath(escape))


def test_the_root_is_the_mount_not_the_host_directory(
    env: DockerEnvironment, tmp_path: Path
) -> None:
    # Tools resolve paths against this, so it has to be the container's view.
    assert env.root == WORKSPACE_MOUNT
    assert str(tmp_path) not in str(env.root)


# -- failure modes -------------------------------------------------------------


def test_a_missing_workspace_is_reported_before_anything_starts(tmp_path: Path) -> None:
    env = DockerEnvironment(workspace=tmp_path / "not-there")
    with pytest.raises(DockerUnavailable, match="does not exist"):
        env.start()


def test_a_missing_docker_binary_says_what_to_do(tmp_path: Path) -> None:
    """The message has to name the alternative, or the operator is stuck."""
    env = DockerEnvironment(workspace=tmp_path, docker="definitely-not-a-real-binary")
    with pytest.raises(DockerUnavailable, match="not installed"):
        env.start()


def test_closing_without_starting_does_nothing(env: DockerEnvironment) -> None:
    env.close()  # must not raise, and must not shell out
    assert not env.container


def test_describe_names_the_settings_that_matter(env: DockerEnvironment) -> None:
    # `harn doctor` prints this; an operator should be able to see at a glance
    # whether the sandbox is actually sandboxing.
    described = env.describe()
    assert "network=none" in described
    assert "read_only_root=True" in described
    assert "user=1000:1000" in described


# -- with a real daemon --------------------------------------------------------


def _daemon_is_running() -> bool:
    """Whether a real Docker daemon will answer.

    Checked rather than assumed, so the live test is deselected for a clear
    reason instead of failing with a connection error.
    """
    found = shutil.which("docker")
    if not found:
        return False
    probe = subprocess.run([found, "info"], capture_output=True, check=False, timeout=20)
    return probe.returncode == 0


_DAEMON = _daemon_is_running()


@pytest.mark.integration
@pytest.mark.skipif(not _DAEMON, reason="needs a running Docker daemon")
@pytest.mark.slow
def test_a_real_container_runs_commands_and_keeps_its_state(tmp_path: Path) -> None:
    """The test that proves the sandbox is not decorative.

    Asserts the three things that cannot be checked from an argv: a command runs
    at all, state survives between commands (so `cd build && make` works), and
    the workspace is genuinely shared with the host.
    """
    (tmp_path / "hello.txt").write_text("from the host\n", encoding="utf-8")
    with DockerEnvironment(workspace=tmp_path) as env:
        assert env.run(["echo", "hi"]).stdout.strip() == "hi"
        # The host's file is visible inside.
        assert env.read_bytes(PurePosixPath("hello.txt")) == b"from the host\n"
        # A write inside is visible on the host.
        env.write_bytes(PurePosixPath("written.txt"), b"from the container\n")
        assert (tmp_path / "written.txt").read_text(encoding="utf-8") == "from the container\n"
        # State persists: the second command sees the first one's work.
        env.run_shell("mkdir -p build && echo made > build/marker")
        assert env.run_shell("cat build/marker").stdout.strip() == "made"
        # And there is no network.
        assert env.run_shell("getent hosts example.com").exit_code != 0
