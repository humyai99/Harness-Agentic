# Harness-Agentic

A self-improving agent framework in Python: pluggable model providers, a
registry-based tool system, skills the agent writes and revises from its own
experience, and a chat gateway that serves several messaging platforms from one
process.

> Status: early. M0 (scaffold and toolchain) is in place. See
> `docs/architecture.md` for the design and the milestone plan.

## Quick start

```bash
uv sync --all-extras --dev
uv run harn --version
uv run harn doctor
```

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
