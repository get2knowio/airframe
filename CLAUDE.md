# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project shape

Airframe is "JDBC for LLM agent SDKs": a vendor-neutral `AgentRuntime`
protocol (`src/airframe/protocol.py`) plus pluggable adapters under
`src/airframe/adapters/`. Consumer code depends on the protocol;
each adapter wraps a different vendor SDK (Claude Agent SDK, GitHub
Copilot SDK, Moonshot Kimi Agent SDK, AWS Bedrock Converse via
aioboto3, OpenAI-compatible HTTP). The PyPI distribution name is
`airframe-agents`; the import name is `airframe`.

## Common commands

`mise` is the task runner and toolchain pin (`mise.toml`); `uv` does
the actual work. The verbs are the portfolio contract — the same ones
every repo in `get2knowio` answers to, and the same ones CI invokes.
There is no second list of steps anywhere.

```bash
mise run setup      # uv sync --all-extras --group dev
mise run test       # full pytest suite
mise run test-fast  # excludes the `integration` marker
mise run lint       # ruff check + mypy
mise run fmt        # apply ruff formatting and safe fixes
mise run check      # fmt-check + lint + tests w/ coverage floor — the gate
mise run build      # uv build
```

`mise run check` is what CI runs, and it exits non-zero on any failure.
Run it before pushing.

Run one test: `uv run pytest tests/unit/test_claude_code.py::test_name -q`.

Live probes (require real vendor credentials, kept out of the unit
suite because pytest collects only `test_*.py`):

```bash
uv run python examples/probe_claude_code.py
uv run python examples/probe_copilot.py
uv run python examples/probe_kimi.py
uv run python examples/probe_opencode_zen.py
uv run python examples/probe_supports.py
uv run python tests/probe_list_models.py [--provider claude] [--installed-only=false]
```

## Architecture essentials

The protocol is intentionally narrow — five methods: `execute`,
`reset`, `close`, `validate_binding`, `list_models`, plus capability
predicates `supports(Feature)` and the JDBC-style escape hatch
`unwrap(NativeType)`. Everything above the protocol (retry, fallback,
memory, orchestration) is consumer responsibility.

Key invariants when editing:

- **Wrap vendor SDKs; don't rewrite them.** Airframe's job is to
  expose a vendor-neutral protocol over each vendor's official SDK,
  not to reimplement the vendor's wire format. Before hand-rolling
  HTTP, headers, retry policy, error parsing, or auth handling
  against a vendor endpoint, check whether the official SDK already
  exposes that surface — and use it if it does. Concrete examples:
  call `anthropic.AsyncAnthropic.models.list()` rather than
  `httpx.get("https://api.anthropic.com/v1/models", headers={...})`;
  let the SDK pick the right header set (`x-api-key` vs
  `Authorization: Bearer` + the `oauth-2025-04-20` beta header) and
  handle refresh, retries, and rate-limit telemetry. Local on-disk
  credential helpers (reading `~/.claude/.credentials.json` etc.)
  are a legitimate exception because no SDK exposes them; the rest
  of the auth dance still belongs to the SDK.
- **`close()` is idempotent and never raises.** It runs from
  `finally`/`__aexit__`; shadowing the original exception would be
  catastrophic. `reset()` is also idempotent — it drops scope-bound
  state (sessions/threads) but keeps runtime-wide resources
  (subprocess pool, HTTP client, auth tokens).
- **Errors are vendor-agnostic.** Adapters classify vendor failures
  at the boundary into the `airframe.errors.Runtime*Error` hierarchy.
  Don't leak vendor exception types past the adapter.
- **Lazy SDK imports.** `import airframe` must not pull in
  `claude-agent-sdk`, `openai`, etc. Adapter modules import their
  vendor SDK inside the method that needs it, gated on the
  optional-extra being installed.
- **`PROVIDER_ID` / `REQUIRES_PACKAGE` / `EXTRA_NAME`** are ClassVars
  on every adapter — they drive `airframe.discovery.list_providers`
  filtering by installed extras and the `airframe.adapters`
  entry-point group for third-party adapters.
- **Provider IDs are strict; no aliases.** `"anthropic"` and
  `"openai"` are reserved for future direct-API adapters; today's
  subscription / gateway / managed / agent-server adapters use
  `"claude"`, `"github-copilot"`, `"opencode"`, `"opencode-zen"`,
  `"opencode-go"`, `"openrouter"`, `"bedrock"`, and `"kimi"`. The
  three `opencode*` IDs are deliberately distinct: `"opencode"`
  wraps the local HTTP agent server (`opencode serve`),
  `"opencode-zen"` wraps the per-token gateway at
  `https://opencode.ai/zen/v1`, and `"opencode-go"` wraps the
  flat-fee subscription gateway at `https://opencode.ai/zen/go/v1`
  — different wire formats, different auth, different feature
  surfaces. Reserved-but-not-shipped:
  `"bedrock-agents"` (future sibling wrapping `bedrock-agent-runtime`
  — Knowledge Bases, action groups; must not be folded into
  `"bedrock"`); `"moonshot"` (future OpenAI-compat sibling fronting
  `api.moonshot.ai/v1` chat-completions; must not be folded into
  `"kimi"`, which wraps the Kimi Agent SDK subprocess surface);
  `"codex"` (reserved for a possible future adapter wrapping
  OpenAI's official `openai-codex` Python SDK once it leaves alpha
  — `airframe-agents` 0.7.0 removed the earlier `CodexRuntime`
  that wrapped the now-unmaintained `openai-codex-sdk` package).
- **`CopilotRuntime.validate_binding` deliberately rejects
  `claude-*` model IDs** — Claude via Copilot Chat Completions emits
  markdown-fenced JSON instead of honouring tool calls. Route Claude
  through `ClaudeCodeRuntime`.

### OpenAI-compatible base

Vendors that speak OpenAI Chat Completions inherit from
`OpenAICompatibleRuntime` (`src/airframe/adapters/openai_compatible.py`):
the base implements `execute`, `list_models`, `reset`, `close`, error
classification, and single-key envelope unwrap. Subclasses are
~30 lines — `PROVIDER_ID`, `DEFAULT_BASE_URL`, `DEFAULT_MODEL`,
per-model `_METADATA`, and `_resolve_api_key()`. `opencode_zen.py` is
the canonical example. SDK-based vendors (subprocess / native types)
inherit `AgentRuntime` directly; see `claude_code.py`.

### Test layout

`tests/unit/` is the default suite; `tests/integration/` talks to real
vendor endpoints and is marked `integration` automatically by
`tests/conftest.py`, so a new integration module cannot forget the
marker and quietly join the default run.

### Conformance contracts

`src/airframe/testing/contracts.py` holds shared pytest test functions
every adapter must satisfy. The in-tree tests
(`tests/unit/test_*_conformance.py`) import them and provide an
`adapter_runtime` fixture. Third-party adapters do the same via
`pip install airframe-agents[testing]`. When adding adapter behaviour
that's contract-worthy, add the test to `contracts.py` so every
adapter inherits it.

### Phasing context

`dev-docs/implementation-plan.md` and `dev-docs/feature-roadmap.md`
describe the phased rollout. These docs are dev-internal — they
don't ship in the PyPI sdist (see `[tool.hatch.build.targets.sdist]`).
The current release (v0.3.0) is **Phase 0 —
Foundations**: the `Feature` enum, `ProviderOptions` namespaces,
`unwrap`, and entry-point discovery all shipped intentionally
*before* substantive feature work, so later phases can land without
re-shaping public surface. `ProviderOptions` dataclasses
(`src/airframe/options.py`) are deliberately empty scaffolding —
populate them only when the corresponding feature phase lands. Today
only `Feature.STRUCTURED_OUTPUT_JSON_SCHEMA` returns `True`; other
enum members exist to lock the names, and return `False`.

## Governance

`.specify/memory/constitution.md` carries the project's non-negotiable
principles, each with the command that fails when it is violated.
`tests/unit/test_constitution.py` is that enforcement — a principle
that isn't a test is a suggestion. Amending the constitution means
bumping its semver and updating the Sync Impact Report at its top.

## Style

Python 3.12+, type hints on every public function, Google-style
docstrings (Args / Returns / Raises). Ruff config in `pyproject.toml`
sets line length 100 and enables `E W F I N UP B C4 SIM ASYNC`. Mypy runs
non-strict but with `strict_equality`, `warn_redundant_casts`,
`no_implicit_optional`. Most modules under `src/airframe/` are <500
LOC by convention.
