# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project shape

Airframe is "JDBC for LLM agent SDKs": a vendor-neutral `AgentRuntime`
protocol (`src/airframe/protocol.py`) plus pluggable adapters under
`src/airframe/adapters/`. Consumer code depends on the protocol;
each adapter wraps a different vendor SDK (Claude Agent SDK, GitHub
Copilot SDK, AWS Bedrock Converse via aioboto3, the OpenCode agent
server, OpenAI-compatible HTTP). The PyPI distribution name is
`airframe-agents`; the import name is `airframe`.

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
suite because pytest collects only `test_*.py`). `examples/` holds
about twenty; the feature-specific ones (`probe_thinking.py`,
`probe_tools.py`, `probe_hooks.py`, …) follow the same pattern:

```bash
uv run python examples/probe_claude_code.py
uv run python examples/probe_copilot.py
uv run python examples/probe_bedrock.py
uv run python examples/probe_opencode_zen.py
uv run python examples/probe_opencode_server.py
uv run python examples/probe_supports.py
uv run python examples/smoke_providers.py
uv run python tests/probe_list_models.py [--provider claude] [--installed-only=false]
```

## Architecture essentials

The protocol is intentionally narrow: `execute`, `session`, `reset`,
`close`, `validate_binding`, `list_models`, `count_tokens`, plus the
capability predicates `supports(Feature)` /
`supported_native_tools()` and the JDBC-style escape hatch
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
- **`PROVIDER_ID` identifies a *binding*, not a vendor.** This is the
  rule for when to mint a new ID, and it is easy to get wrong because
  the field is named `provider_id`. A binding is one **(wire protocol,
  endpoint, auth mechanism, feature surface)** tuple. One vendor
  routinely spans several:

  | Vendor | IDs | What differs |
  | --- | --- | --- |
  | Anthropic | `anthropic` (reserved) vs `claude` | direct Messages API vs Agent SDK subprocess |
  | Moonshot | `moonshot` (reserved) vs `kimi` (reserved) | OpenAI-compat HTTP vs Agent SDK subprocess |
  | AWS | `bedrock` vs `bedrock-agents` (reserved) | Converse vs `bedrock-agent-runtime` |
  | OpenCode | `opencode` / `opencode-zen` / `opencode-go` | local server / per-token gateway / flat-fee gateway |

  It is not the framework either: `openrouter`, `opencode-zen`, and
  `opencode-go` all inherit `OpenAICompatibleRuntime` and hold three
  distinct IDs. So the ID tracks neither the company nor the client
  library — it tracks the binding. The JDBC analogy is load-bearing:
  `jdbc:postgresql:` names a subprotocol, not a company, and one DBMS
  can have several drivers. `provider_id` is the subprotocol slot.

  Two mechanisms break if an ID straddles two bindings:

  - `REQUIRES_PACKAGE` / `EXTRA_NAME` gate `list_providers` on which
    client library is importable. An ID spanning two harnesses cannot
    answer "is this usable on this machine?" — the question discovery
    exists to answer.
  - `SUPPORTED_FEATURES` is a ClassVar frozenset: exactly one honest
    capability manifest per ID. Two endpoints whose real capabilities
    differ cannot share one without `supports()` lying about at least
    one of them.

  **Test before minting an ID:** if the new thing needs a different
  auth variable, a different base URL, or a trimmed
  `SUPPORTED_FEATURES` versus an existing adapter, it is a separate
  binding — *even when it reuses that adapter's harness wholesale*.
  Pointing an existing adapter at it through env vars stays available
  as an escape hatch, but an escape hatch is not a supported binding.

  **Name the axis that distinguishes it**, and don't claim a bare
  vendor name when the vendor exposes more than one surface — the
  same reason the Kimi Agent SDK adapter was never named `moonshot`.

- **Provider IDs are strict; no aliases.** `"anthropic"` and
  `"openai"` are reserved for future direct-API adapters; today's
  subscription / gateway / managed / agent-server adapters use
  `"claude"`, `"github-copilot"`, `"opencode"`, `"opencode-zen"`,
  `"opencode-go"`, `"openrouter"`, and `"bedrock"`. The three
  `opencode*` IDs are deliberately distinct: `"opencode"` wraps the
  local HTTP agent server (`opencode serve`), `"opencode-zen"` wraps
  the per-token gateway at `https://opencode.ai/zen/v1`, and
  `"opencode-go"` wraps the flat-fee subscription gateway at
  `https://opencode.ai/zen/go/v1` — different wire formats, different
  auth, different feature surfaces. Reserved-but-not-shipped:
  `"bedrock-agents"` (future sibling wrapping `bedrock-agent-runtime`
  — Knowledge Bases, action groups; must not be folded into
  `"bedrock"`); `"kimi"` (the Kimi Agent SDK subprocess surface — an
  adapter shipped and was removed because `kimi-cli` pinned a
  transitive `mcp<1.17` carrying known advisories; the ID stays
  reserved for its return); `"moonshot"` (future OpenAI-compat sibling
  fronting `api.moonshot.ai/v1` chat-completions; must not be folded
  into `"kimi"`); `"codex"` (reserved for a possible future adapter
  wrapping OpenAI's official `openai-codex` Python SDK once it leaves
  alpha — `airframe-agents` 0.7.0 removed the earlier `CodexRuntime`
  that wrapped the now-unmaintained `openai-codex-sdk` package).

- **Subscription credentials are endpoint-scoped.** Anthropic-minted
  OAuth tokens (`CLAUDE_CODE_OAUTH_TOKEN`,
  `~/.claude/.credentials.json`) authenticate the *user's account* and
  are worthless to any other vendor except as a stolen secret. They
  must never be sent to a base URL that isn't the vendor's own — see
  `_is_anthropic_endpoint` / `_resolve_anthropic_auth` in
  `claude_code.py`. API-key-shaped slots (an explicit `api_key=` arg,
  `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`) are deliberately *not*
  scoped: compatible vendors reuse exactly those to carry their own
  credentials, so scoping them would break the legitimate case. Any
  adapter that lets a base URL be overridden inherits this obligation.

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

Phase 0 — Foundations shipped the `Feature` enum, `ProviderOptions`
namespaces, `unwrap`, and entry-point discovery *before* substantive
feature work, so later phases could land without re-shaping public
surface. That worked: the feature phases since have filled the
scaffolding in rather than replacing it. `Feature` now has 27
members, and coverage varies per binding — `claude` implements 22,
the OpenAI-compatible bindings 15, `bedrock` 13, `opencode` 12.
`ProviderOptions` dataclasses (`src/airframe/options.py`) are
populated as their phases land; a namespace that is still empty is
reserved surface, not an oversight.

Check current coverage rather than trusting any list in this file:

```bash
uv run python examples/probe_supports.py
```

## Style

Python 3.12+, type hints on every public function, Google-style
docstrings (Args / Returns / Raises). Ruff config in `pyproject.toml`
sets line length 99 and enables `E W F I N UP B C4 SIM`. Mypy runs
non-strict but with `strict_equality`, `warn_redundant_casts`,
`no_implicit_optional`. Most modules under `src/airframe/` are <500
LOC by convention.
