# Harness-Agentic

A self-improving agent framework in Python: pluggable model providers, a
registry-based tool system, skills the agent writes and revises from its own
experience, and a chat gateway that serves several messaging platforms from one
process.

> Status: in development. The agent loop, tools, sessions, memory, skills, the
> chat gateway, MCP, the browser toolset, the web UI and voice are implemented.
> See `docs/architecture.md` for the design and the milestone plan.

## Install

One command, and `harn` is on your PATH — no clone, no virtualenv to manage:

```bash
uv tool install "git+https://github.com/humyai99/Harness-Agentic@claude/hermes-agent-framework-9g8x15"
```

`pipx install git+https://...` works the same way if you would rather. For the
optional pieces, name the extras with a PEP 508 direct reference:

```bash
uv tool install "harness-agentic[gateway,docker] @ git+https://github.com/humyai99/Harness-Agentic@claude/hermes-agent-framework-9g8x15"
```

| extra | brings | for |
| --- | --- | --- |
| `gateway` | uvicorn, websockets | Telegram / LINE / Slack / Discord, the web UI |
| `docker` | docker sdk | running tools in a hardened container |
| `browser` | playwright (~400 MB) | the `browser` toolset |
| `repl` | prompt_toolkit | a nicer `harn chat` |
| `tokens` | tiktoken | exact token counts instead of estimates |

To hack on it instead, clone and `uv sync --all-extras --dev`.

```bash
harn --version
harn doctor          # where state lives, which credentials resolved, from where
```

## Try it without an API key

`fake/scripted` is a real provider that replays a scenario file, so the whole
stack runs with no credential, no network and no bill. Everything except the
model is real: real tool dispatch against the real filesystem, real approval
policy, real session store, real prompt assembly with real cache breakpoints.

A `tour` scenario ships with the package, so this works in any empty directory:

```bash
export HARNESS_HOME=/tmp/harn-demo      # keep the demo out of ~/.harness
export HARNESS_FAKE_SCRIPT=tour         # a bundled scenario, by name

harn run -m fake/scripted --toolsets file,memory --yes "have a look around"
```

```
Let me see where I am.
  > list_dir: .                  ✓
  > write_file: harness-tour.md  ✓
  > read_file: harness-tour.md   ✓
  > memory_add                   ✓
I listed the directory, wrote harness-tour.md, read it back to confirm it
landed, and recorded a note that will be in my prompt next session...
5 step(s)  in 11165  out 75
```

It writes one clearly-named file; delete `harness-tour.md` afterwards. `--yes`
approves every tool call and belongs only in a sandbox — without it, `terminal`
is refused on any surface with nobody to ask.

Then look at what the run left behind:

```bash
harn memory show     # the fact it recorded, and what it costs per turn
harn sessions        # the transcript, searchable with FTS5
harn tools           # every tool an agent would be offered, and its danger level
harn skills list     # the catalog, and its token cost
```

Write your own scenario by copying the bundled one — each entry under `turns:`
is one model response, `tool:` calls a tool and `text:` answers:

```bash
python -c "import harness_agentic.testing.scenario as s; print(s.BUNDLED / 'tour.yaml')"
```

## Use a real model

```bash
harn auth set anthropic      # hidden prompt; writes ~/.harness/.env at mode 0600
harn run -m anthropic/claude-sonnet-4-6 "read pyproject.toml and list the dev dependencies"
```

Never pass a credential as an argument: it lands in shell history and is visible
in `ps` while the command runs. `harn models` shows which providers resolved.

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
