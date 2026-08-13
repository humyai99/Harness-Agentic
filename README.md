# Harness-Agentic

A self-improving agent framework in Python: pluggable model providers, a
registry-based tool system, skills the agent writes and revises from its own
experience, and a chat gateway that serves several messaging platforms from one
process.

> Status: in development. The agent loop, tools, sessions, memory, skills, the
> chat gateway, MCP, the browser toolset, the web UI and voice are implemented.
> See `docs/architecture.md` for the design and the milestone plan.

## Quick start

```bash
uv sync --all-extras --dev
uv run harn --version
uv run harn doctor
```

## Try it without an API key

`fake/scripted` is a real provider that replays a scenario file, so the whole
stack runs with no credential, no network and no bill. Everything except the
model is real: real tool dispatch against the real filesystem, real approval
policy, real session store, real prompt assembly with real cache breakpoints.

```bash
export HARNESS_HOME=/tmp/harn-demo            # keep it out of ~/.harness
export HARNESS_FAKE_SCRIPT=examples/tour.yaml

uv run harn run -m fake/scripted --toolsets file,memory --yes \
  "have a look around this project"
```

```
Let me see what this project is.
  > read_file: pyproject.toml     ✓
  > grep_files: src/harness_agentic/memory     ✓
  > memory_add     ✓
This is harness-agentic, a Python agent framework...
4 step(s)  in 23865  out 68
```

`--yes` approves every tool call and belongs only in a sandbox; without it the
`terminal` tool is refused on any surface with nobody to ask. Then look at what
the run left behind:

```bash
uv run harn memory show        # the fact it recorded, and what it costs per turn
uv run harn sessions           # the transcript, searchable with FTS5
uv run harn skills list        # the catalog, and its token cost
uv run harn tools              # every tool an agent would be offered
```

Write your own scenario by copying `examples/tour.yaml`: each entry under
`turns:` is one model response, `tool:` calls a tool and `text:` answers.

With a real key instead, `harn auth set anthropic` reads it from a hidden prompt
and writes it to `.env` at mode 0600 — never pass a credential as an argument,
where it lands in shell history and in `ps`.

## Design in one page

| Layer | What it owns |
| --- | --- |
| `core/` | Canonical `Message` / `ContentBlock` types, streaming events, cancellation, the single sync↔async bridge |
| `providers/` | `ProviderTransport` per API shape — Anthropic messages, OpenAI chat-completions, OpenAI responses, Gemini — plus the model catalog |
| `tools/` | `ToolRegistry`, JSON-schema dispatch, approval policy, danger classification |
| `envs/` | `ExecEnvironment`: the only path to the filesystem and the shell, so `local` / `docker` / `ssh` swap freely |
| `session/` | SQLite store with FTS5 search and reversible compaction |
| `skills/` | Progressive-disclosure skills, the reflection loop, validation and review gates |
| `gateway/` | Platform adapters, session routing, authorization, streaming to length-limited chats |

Three decisions worth knowing before reading the code:

- **The core is synchronous; the gateway is asyncio.** Exactly one module
  bridges them, and a pre-commit hook keeps event loops out of everything else.
  A blocking call inside an async core would stall every chat platform at once,
  and that mistake would be spread across every tool author.
- **The canonical message format is block-based**, not OpenAI-shaped. Converting
  down to chat-completions is mechanical; converting up loses the block ordering
  and thinking signatures that Anthropic requires on replay.
- **Tools never touch the filesystem directly.** They go through
  `ExecEnvironment`, which is what makes the Docker and SSH backends free.

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest
uv run pre-commit install     # once
```

The test suite is hermetic: an autouse fixture blocks outbound HTTP and another
redirects `~/.harness` to a temporary directory. Tests that need a real provider
carry `@pytest.mark.integration` and are deselected by default.

## Acknowledgement

The subsystem decomposition was informed by studying
[Hermes Agent](https://github.com/NousResearch/hermes-agent) by Nous Research
(MIT). No code or prompt text was copied; see `NOTICE`.

## License

MIT — see `LICENSE`.
