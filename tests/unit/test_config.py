"""Layered settings: precedence, provenance, and the secret boundary.

The precedence tests are the point. Six layers can each set the same key, and
"why is it using that model?" is only answerable if the answer is recorded --
which is what `origins` is for, and what `harn config show --origin` prints.

The list-replace tests matter almost as much. Appending would be the obvious
choice and it is the wrong one: a project that sets `tools.toolsets` means *that
list*, and if merging concatenated there would be no way to express "only these"
and the inherited entries would be invisible in the file that appears to define
them.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from harness_agentic.config import ConfigError, Settings, dotted_keys, load_settings
from harness_agentic.config.loader import CONFIG_FILENAME, find_project_config


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated HARNESS_HOME, so no test reads the real one."""
    root = tmp_path / "home"
    root.mkdir()
    monkeypatch.setenv("HARNESS_HOME", str(root))
    monkeypatch.delenv("HARNESS_PROFILE", raising=False)
    # A HARNESS__ variable in the developer's own shell would otherwise win
    # over every file and make these tests pass or fail by accident.
    for name in [key for key in os.environ if key.startswith("HARNESS__")]:
        monkeypatch.delenv(name, raising=False)
    return root


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


# -- precedence ----------------------------------------------------------------


def test_defaults_apply_when_nothing_is_configured(home: Path, tmp_path: Path) -> None:
    loaded = load_settings(workspace=tmp_path, environ={})
    assert loaded.settings.model.default == Settings().model.default
    assert loaded.origin_of("model.default").layer == "defaults"


def test_the_home_file_beats_the_defaults(home: Path, tmp_path: Path) -> None:
    _write(home / CONFIG_FILENAME, '[model]\ndefault = "openai/gpt-5"\n')
    loaded = load_settings(workspace=tmp_path, environ={})
    assert loaded.settings.model.default == "openai/gpt-5"
    assert loaded.origin_of("model.default").layer == "home"


def test_the_profile_file_beats_the_home_file(home: Path, tmp_path: Path) -> None:
    _write(home / CONFIG_FILENAME, '[model]\ndefault = "openai/gpt-5"\n')
    _write(home / "profiles" / "default" / CONFIG_FILENAME, '[model]\ndefault = "ollama/qwen3"\n')
    loaded = load_settings(workspace=tmp_path, environ={})
    assert loaded.settings.model.default == "ollama/qwen3"
    assert loaded.origin_of("model.default").layer == "profile"


def test_the_project_file_beats_the_profile(home: Path, tmp_path: Path) -> None:
    """A repository's own config is the most specific statement of intent.

    It is also the one that went through code review, which is the better reason.
    """
    _write(home / "profiles" / "default" / CONFIG_FILENAME, '[model]\ndefault = "ollama/qwen3"\n')
    project = tmp_path / "project"
    _write(project / ".harness" / CONFIG_FILENAME, '[model]\ndefault = "gemini/gemini-2.5-pro"\n')
    loaded = load_settings(workspace=project, environ={})
    assert loaded.settings.model.default == "gemini/gemini-2.5-pro"
    assert loaded.origin_of("model.default").layer == "project"


def test_the_environment_beats_every_file(home: Path, tmp_path: Path) -> None:
    _write(home / CONFIG_FILENAME, '[model]\ndefault = "openai/gpt-5"\n')
    loaded = load_settings(
        workspace=tmp_path, environ={"HARNESS__MODEL__DEFAULT": "anthropic/claude-opus-5"}
    )
    assert loaded.settings.model.default == "anthropic/claude-opus-5"
    origin = loaded.origin_of("model.default")
    assert origin.layer == "env"
    assert origin.source == "HARNESS__MODEL__DEFAULT"


def test_a_flag_beats_the_environment(home: Path, tmp_path: Path) -> None:
    loaded = load_settings(
        workspace=tmp_path,
        environ={"HARNESS__MODEL__DEFAULT": "anthropic/claude-opus-5"},
        overrides={"model.default": "openai/gpt-5"},
    )
    assert loaded.settings.model.default == "openai/gpt-5"
    assert loaded.origin_of("model.default").layer == "cli"


def test_an_override_of_none_means_the_flag_was_not_given(home: Path, tmp_path: Path) -> None:
    """Otherwise a flag's default would sit permanently on top of every file.

    This is the mechanism that lets `--model` be optional while still winning
    when it is passed.
    """
    _write(home / CONFIG_FILENAME, '[model]\ndefault = "openai/gpt-5"\n')
    loaded = load_settings(workspace=tmp_path, environ={}, overrides={"model.default": None})
    assert loaded.settings.model.default == "openai/gpt-5"


def test_layers_merge_per_key_rather_than_wholesale(home: Path, tmp_path: Path) -> None:
    # A file naming one setting must not reset the section's other settings.
    _write(home / CONFIG_FILENAME, '[model]\ndefault = "openai/gpt-5"\nmax_iterations = 7\n')
    _write(home / "profiles" / "default" / CONFIG_FILENAME, "[model]\nstream = false\n")
    loaded = load_settings(workspace=tmp_path, environ={})
    assert loaded.settings.model.default == "openai/gpt-5"
    assert loaded.settings.model.max_iterations == 7
    assert loaded.settings.model.stream is False


# -- lists ---------------------------------------------------------------------


def test_a_list_replaces_rather_than_appends(home: Path, tmp_path: Path) -> None:
    """Appending would leave no way to say "only these".

    And the inherited entries would be invisible in the file that appears to
    define the list, which is the part that wastes an afternoon.
    """
    _write(home / CONFIG_FILENAME, '[tools]\ntoolsets = ["file", "terminal", "web"]\n')
    project = tmp_path / "project"
    _write(project / ".harness" / CONFIG_FILENAME, '[tools]\ntoolsets = ["file"]\n')
    loaded = load_settings(workspace=project, environ={})
    assert loaded.settings.tools.toolsets == ("file",)


def test_the_additive_case_has_its_own_spelling(home: Path, tmp_path: Path) -> None:
    _write(
        home / CONFIG_FILENAME,
        '[tools]\ntoolsets = ["file", "terminal"]\ndisabled = ["terminal"]\n',
    )
    project = tmp_path / "project"
    _write(project / ".harness" / CONFIG_FILENAME, '[tools]\ndisabled_append = ["file"]\n')
    loaded = load_settings(workspace=project, environ={})
    # `disabled` survives from the home layer and the project adds to it.
    assert loaded.settings.enabled_toolsets() == []


def test_a_comma_separated_environment_value_becomes_a_list(home: Path, tmp_path: Path) -> None:
    # An environment variable is always a string, and nothing downstream can tell
    # "a,b" from a single string that happens to contain a comma.
    loaded = load_settings(workspace=tmp_path, environ={"HARNESS__TOOLS__TOOLSETS": "file, skill"})
    assert loaded.settings.tools.toolsets == ("file", "skill")


@pytest.mark.parametrize(("raw", "expected"), [("true", True), ("false", False)])
def test_a_boolean_environment_value_is_not_a_string(
    home: Path, tmp_path: Path, raw: str, expected: bool
) -> None:
    loaded = load_settings(workspace=tmp_path, environ={"HARNESS__MODEL__STREAM": raw})
    assert loaded.settings.model.stream is expected


# -- failure modes -------------------------------------------------------------


def test_an_unknown_key_is_an_error_not_a_no_op(home: Path, tmp_path: Path) -> None:
    """A typo that silently does nothing is the expensive kind.

    The file looks right, the behaviour is not, and nothing points at the file.
    """
    _write(home / CONFIG_FILENAME, '[model]\ndefualt = "openai/gpt-5"\n')
    with pytest.raises(ConfigError, match="not valid"):
        load_settings(workspace=tmp_path, environ={})


def test_an_out_of_range_value_is_refused(home: Path, tmp_path: Path) -> None:
    _write(home / CONFIG_FILENAME, "[model]\nmax_iterations = 100000\n")
    with pytest.raises(ConfigError):
        load_settings(workspace=tmp_path, environ={})


def test_an_unreadable_file_is_reported_but_not_fatal(home: Path, tmp_path: Path) -> None:
    """One broken project file must not stop the agent, and must not be silent."""
    _write(home / CONFIG_FILENAME, "this is not toml [[[\n")
    loaded = load_settings(workspace=tmp_path, environ={})
    assert loaded.problems
    assert "config.toml" in loaded.problems[0]
    assert loaded.settings.model.default == Settings().model.default


# -- project discovery ---------------------------------------------------------


def test_a_project_config_is_found_from_a_subdirectory(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    _write(project / ".harness" / CONFIG_FILENAME, "[model]\nstream = false\n")
    (project / ".git").mkdir(parents=True)
    deep = project / "src" / "pkg" / "sub"
    deep.mkdir(parents=True)

    assert find_project_config(deep) == project / ".harness" / CONFIG_FILENAME


def test_the_search_stops_at_a_git_root(tmp_path: Path) -> None:
    """A repository is where a project's boundary is in practice.

    Without this, a checkout inside a directory that happens to have its own
    `.harness` would silently inherit settings from an unrelated project.
    """
    outer = tmp_path / "outer"
    _write(outer / ".harness" / CONFIG_FILENAME, '[model]\ndefault = "wrong/one"\n')
    inner = outer / "repo"
    (inner / ".git").mkdir(parents=True)

    assert find_project_config(inner) is None


def test_every_settable_key_is_discoverable(home: Path) -> None:
    # `config set` validates against this, so a section missing from it would be
    # unsettable with a confusing "unknown setting".
    keys = dotted_keys()
    assert "model.default" in keys
    assert "tools.env" in keys
    assert "docker.network" in keys
    assert "skills.autonomy" in keys
    assert "gateway.allow_all" in keys
