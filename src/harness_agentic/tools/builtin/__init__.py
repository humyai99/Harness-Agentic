"""Builtin tools, registered by explicit import.

Not a filesystem scan. Scanning is how a tool silently vanishes because its
module raised on import -- the agent simply stops being able to read files and
nothing says why. It also hides every tool from mypy. Autodiscovery stays for
*plugins*, where the set genuinely is not known ahead of time.

Importing this module registers everything into the process-wide registry.
"""

from harness_agentic.tools.builtin import fs, session, shell
from harness_agentic.tools.registry import ToolRegistry, registry
from harness_agentic.tools.spec import Toolset

__all__ = ["builtin_registry", "fs", "install_builtins", "session", "shell"]


def builtin_registry() -> ToolRegistry:
    """A private registry holding every builtin tool and toolset.

    The one correct way to get a registry with the builtins in it.
    ``install_builtins(ToolRegistry())`` looks like it should work and returns
    something empty, because the tools register into the process-wide registry at
    import time and only the *toolsets* are added by that call. Forking after is
    what makes both halves present.
    """
    return install_builtins(registry).fork()


def install_builtins(target: ToolRegistry | None = None) -> ToolRegistry:
    """Declare the builtin toolsets on ``target`` (default: the global registry).

    The tools themselves register at import time; this adds the groupings that
    per-surface gating relies on. An agent answering strangers on a public chat
    account gets ``file`` and not ``terminal``, and that distinction has to be
    expressible before it can be enforced.
    """
    reg = target or registry
    reg.define_toolset(
        Toolset(
            name="file",
            description="Read, search, and edit files in the workspace.",
            tools=("read_file", "write_file", "list_dir", "glob_files", "grep_files"),
        )
    )
    reg.define_toolset(
        Toolset(
            name="core",
            description="Always-on tools: searching the agent's own history.",
            tools=("session_search", "delegate"),
        )
    )
    reg.define_toolset(
        Toolset(
            name="web",
            description="Read pages and search the web. Results are untrusted data.",
            tools=("web_fetch", "web_search"),
        )
    )
    reg.define_toolset(
        Toolset(
            name="data",
            description="Read a database, and call allowlisted HTTP APIs.",
            tools=("sql_schema", "sql_query", "http_request"),
        )
    )
    reg.define_toolset(
        Toolset(
            name="retrieval",
            description="Search the knowledge base for passages to answer from.",
            tools=("kb_search",),
        )
    )
    reg.define_toolset(
        Toolset(
            name="browser",
            description="Read and act on web pages. Page content is untrusted data.",
            tools=(
                "browser_navigate",
                "browser_snapshot",
                "browser_find",
                "browser_click",
                "browser_type",
                "browser_screenshot",
            ),
        )
    )
    reg.define_toolset(
        Toolset(
            name="memory",
            description="Remember facts across sessions. Bounded, and always in context.",
            tools=("memory_add", "memory_replace", "memory_remove", "memory_list"),
        )
    )
    reg.define_toolset(
        Toolset(
            name="skill",
            description="Look up and follow written procedures, and propose new ones.",
            tools=("skill_search", "skill_load", "skill_read", "skill_propose"),
        )
    )
    reg.define_toolset(
        Toolset(
            name="terminal",
            description="Run commands. Requires approval on every surface.",
            tools=("terminal",),
        )
    )
    return reg
