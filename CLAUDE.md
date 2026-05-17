# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project shape

Airframe is "JDBC for LLM agent SDKs": a vendor-neutral `AgentRuntime`
protocol (`src/airframe/protocol.py`) plus four pluggable adapters
under `src/airframe/adapters/`. Consumer code depends on the protocol;
each adapter wraps a different vendor SDK (Claude Agent SDK, GitHub
Copilot SDK, OpenAI Codex SDK, OpenAI-compatible HTTP). The PyPI
distribution name is `airframe-agents`; the import name is `airframe`.

## Common commands

This project uses `uv` and a `Makefile`.

```bash
make install     # uv sync --all-extras --group dev
make test        # full pytest suite (quiet)
make test-fast   # excludes the `integration` pytest marker
make lint        # ruff check
make typecheck   # mypy on src/airframe
make format-fix  # apply ruff formatting
make ci          # lint + format-check + typecheck + test (pre-push gate)
```

Run one test: `uv run pytest tests/test_claude_code.py::test_name -q`.
Set `VERBOSE=1 make test` for non-quiet pytest/ruff output.

Live probes (require real vendor credentials, kept out of the unit
suite because pytest collects only `test_*.py`):

```bash
uv run python examples/probe_claude_code.py
uv run python examples/probe_codex.py
uv run python examples/probe_copilot.py
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
  subscription-path adapters use `"claude"`, `"github-copilot"`,
  `"codex"`, `"opencode-zen"`, `"opencode-go"`, `"openrouter"`.
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

### Conformance contracts

`src/airframe/testing/contracts.py` holds shared pytest test functions
every adapter must satisfy. The in-tree tests
(`tests/test_*_conformance.py`) import them and provide an
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

## Style

Python 3.11+, type hints on every public function, Google-style
docstrings (Args / Returns / Raises). Ruff config in `pyproject.toml`
sets line length 99 and enables `E W F I N UP B C4 SIM`. Mypy runs
non-strict but with `strict_equality`, `warn_redundant_casts`,
`no_implicit_optional`. Most modules under `src/airframe/` are <500
LOC by convention.
