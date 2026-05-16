# Changelog

All notable changes to airframe are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] — 2026-05-16

Phase 0 of the [implementation plan](docs/implementation-plan.md):
foundations that lock the shape of every extension point Phases 1–6
will use, plus a v0 contract-gap fix that wires plain-text
`execute(schema=None)` across every built-in adapter.

### Fixed

- Plain-text `execute(schema=None)` is honoured on every built-in
  adapter, closing a v0 contract gap. The
  `AgentRuntime.execute()` docstring has always promised
  "`None` means plain text — text answer on `RuntimeResult.text`,
  `structured=None`," but `ClaudeCodeRuntime`, `CopilotRuntime`, and
  `CodexRuntime` refused with `NotImplementedError`.
  (`OpenAICompatibleRuntime` / `OpenCodeZenRuntime` were already
  correct.) Per-adapter wiring:
  - `ClaudeCodeRuntime` — omits `output_format` from
    `ClaudeAgentOptions` when `schema is None`;
    `ResultMessage.result` carries the final assistant text. Cache
    key uses a `"__plain_text__"` sentinel so plain-text and
    structured sessions don't collide on `(model, system)`.
  - `CopilotRuntime` — doesn't register the `submit_result` tool
    when `schema is None`; doesn't prepend the forced-tool prefix
    to the system message; caller-supplied `system=` passes
    through verbatim.
  - `CodexRuntime` — omits `outputSchema` from `TurnOptions` when
    `schema is None`; `turn.final_response` is the free-form text.
    Empty `final_response` is a legitimate outcome rather than a
    structured-output violation.

  Motivation: a downstream consumer codebase (Maverick) just
  migrated five long-running personas onto airframe, each of
  which grew a single-field Pydantic schema purely to satisfy the
  `schema is None` gate. With the gate gone, those wrappers
  vanish and personas call
  `runtime.execute(prompt, system=PERSONA_SYSTEM_PROMPT)` directly.

### Added

- `airframe.Feature` enum — capability-negotiation predicates
  modelled on JDBC `DatabaseMetaData.supportsXxx()` and SQLAlchemy
  `Dialect.supports_*`. Ships the whole forward-looking set (19
  members covering Phases 1–6); later phases flip `True` bits onto
  each adapter without adding new members. Enum string values are
  public surface and snapshotted in tests.
- `AgentRuntime.supports(feature, model=None) -> bool` — pure,
  cheap lookup against an adapter's `SUPPORTED_FEATURES` class set.
  Today every in-tree adapter declares only
  `Feature.STRUCTURED_OUTPUT_JSON_SCHEMA`; Phase 1+ adds more.
- `AgentRuntime.unwrap(cls) -> cls` — JDBC-`Wrapper`-style escape
  hatch to the native SDK object. Each adapter accepts
  `unwrap(type(self))` plus its native types
  (`ClaudeCodeRuntime.unwrap(ClaudeSDKClient)`,
  `CopilotRuntime.unwrap(CopilotClient | CopilotSession)`,
  `CodexRuntime.unwrap(Codex | Thread)`,
  `OpenAICompatibleRuntime.unwrap(AsyncOpenAI)`). Unsupported types
  raise `TypeError`.
- `airframe.options` module with four empty frozen dataclasses
  (`ClaudeOptions`, `CopilotOptions`, `CodexOptions`,
  `OpenAICompatOptions`) and a `ProviderOptions` tagged-union alias.
  Vercel-style namespaced extension points for the
  `ClaudeAgentOptions`-sprawl problem; Phase 2+ populates each as
  features land. Not wired anywhere in v0.3.0 — Phase 1 attaches them
  to `runtime.session(provider_options=...)`.
- `CostRecord.reasoning_tokens: int` (default `0`) — canonicalises
  hidden reasoning / extended-thinking tokens across vendors. Phase 2
  wires real population from each SDK's native counter.
- Third-party adapter discovery via the `airframe.adapters`
  entry-point group. Third-party packages declare their runtime in
  `pyproject.toml` and `airframe.list_providers()` picks it up
  automatically. Modelled on SLF4J's `ServiceLoader` binding
  discovery; same pip-extras filtering applies. Built-ins shadow
  third-party entries with the same `PROVIDER_ID`. Malformed plugins
  (import errors, missing `PROVIDER_ID`, collisions) log a warning
  and are skipped — discovery never crashes.
- `discovery.ENTRY_POINT_GROUP = "airframe.adapters"` constant —
  public surface, snapshotted in tests.
- `airframe.testing` submodule with shared conformance contracts.
  Adapter authors import test functions from
  `airframe.testing.contracts` and provide an `adapter_runtime`
  pytest fixture; pytest collects the imported tests under the local
  fixture. SQLAlchemy `testing.suite` pattern. Ten structural
  contracts covering `close()` idempotency,
  `unwrap(type(self))` returning self, `supports()` purity,
  `validate_binding` behaviour, and the plain-text `execute()`
  path being wired
  (`test_plain_text_execute_path_is_wired`).
- New `[testing]` pyproject extra brings pytest + pytest-asyncio for
  adapter authors running the conformance suite. The main package
  does not depend on pytest.
- Per-adapter conformance test files (`tests/test_*_conformance.py`)
  for all four in-tree adapters — canonical examples for third-party
  authors.
- `examples/probe_supports.py` — capability matrix probe printing
  the live Feature × adapter truth table.
- `docs/feature-roadmap.md` — feature audit of the four native SDKs
  airframe wraps, prioritised list of cross-vendor features, and
  abstraction patterns borrowed from JDBC / SLF4J / OpenTelemetry /
  SQLAlchemy / JAAS.
- `docs/implementation-plan.md` — phased plan (Phase 0–6) ordered
  by dependency, with version targets, irreversible-shape-lock
  gating points, and adapter migration tables.

### Changed

- `AgentRuntime` protocol now requires two additional methods:
  `supports()` and `unwrap()`. Built-in adapters all implement them;
  external implementations of the protocol (none known) would need
  to add them.
- `CostRecord` constructor accepts an additional optional kwarg
  (`reasoning_tokens=0`); the structured-log dict via `to_dict()`
  always includes the key.

### Deferred

- Behavioural conformance contracts (401 → `RuntimeAuthError`,
  schema= round-trip, `input_tokens > 0` on a successful call,
  plain-text round-trip returns non-empty `text`) require live
  vendor credentials and naturally co-locate with Phase 1
  streaming/multi-turn integration test infrastructure. The
  existing `examples/probe_*.py` scripts already exercise these
  end-to-end against real vendors. Will land as
  `airframe.testing.integration` in v0.4.0.

## [0.2.0] — 2026-05-16

Model discovery API + an OpenAI-compatible base class. The headline
addition is `list_models()`: every adapter can now enumerate the
models a UI consumer can pick from, against the user's resolved
credentials. The headline change is install-state gating: which
providers `list_providers()` returns now reflects which pip extras
the consumer installed, so menus stay honest.

### Added

- `AgentRuntime.list_models() -> list[ModelInfo]` — live model
  discovery against the vendor's models endpoint. Driving UI menus
  is the expected use case; the call requires auth + network, so a
  failure is the consumer's signal to surface the issue before
  letting the user pick a model that would later fail to execute.
- `ModelInfo` — `id`, `display_name`, `provider_id`, optional
  `context_window`, `pricing_input_per_1k_usd`,
  `pricing_output_per_1k_usd`, capability flags, vendor-raw payload.
- Capability constants: `CAPABILITY_VISION`, `CAPABILITY_TOOLS`,
  `CAPABILITY_STRUCTURED_OUTPUT`, `CAPABILITY_STREAMING`,
  `CAPABILITY_REASONING_EFFORT`.
- `airframe.list_providers(installed_only=True)` — sorted list of
  provider IDs. Default filters to providers whose vendor SDK is
  importable (so `pip install airframe-agents[copilot]` users see
  `["github-copilot"]`, not the full menu). Pass
  `installed_only=False` for the full registry.
- `airframe.runtime_for(provider_id)` — returns the adapter class
  for a canonical provider ID. Raises `ValueError` for unknown IDs;
  raises `ImportError` (with a `pip install airframe-agents[...]`
  hint) when the adapter exists but its SDK isn't installed.
- `OpenAICompatibleRuntime` base — shared HTTP / `response_format` /
  `list_models` / error classification for OpenAI-compatible
  vendors. Subclasses (Zen today; Together / Groq / Fireworks /
  OpenRouter possible) ship a `PROVIDER_ID`, default base URL,
  default model, per-model metadata table, and a vendor-specific
  `_resolve_api_key()` hook — typically ~30 lines.
- `examples/probe_list_models.py` (committed as
  `tests/probe_list_models.py`) — end-to-end menu probe against every
  installed adapter; tabular output with context window / pricing /
  capabilities per model.
- `ClassVar` declarations on every adapter: `PROVIDER_ID`,
  `REQUIRES_PACKAGE`, `EXTRA_NAME` — drive discovery / install
  gating / error messages.

### Changed (breaking)

- **Provider IDs are strict.** Aliases dropped; each adapter has one
  canonical `PROVIDER_ID`. Update consumer code that referenced the
  old strings:
  - `ClaudeCodeRuntime`: was `{"anthropic", "claude", "claude-code"}`,
    now `"claude"`. The string `"anthropic"` is reserved for a
    future direct-API `AnthropicRuntime`.
  - `CopilotRuntime`: was `{"github-copilot", "copilot", "github"}`,
    now `"github-copilot"`.
  - `CodexRuntime`: was `{"openai", "codex"}`, now `"codex"`. The
    string `"openai"` is reserved for a future direct-API
    `OpenAIRuntime`.
  - `OpenCodeZenRuntime`: was `{"opencode-zen", "opencode"}`, now
    `"opencode"`.
- **The `[opencode-zen]` extra was renamed to `[openai-compat]`** —
  one extra for the whole OpenAI-compatible family (Zen today; room
  for siblings without exploding the extras matrix). All siblings
  share the `openai>=1.50,<3` SDK, so per-child extras would be a
  fiction. Update install commands:
  `pip install airframe-agents[opencode-zen]` →
  `pip install airframe-agents[openai-compat]`.
- `ClaudeCodeRuntime` now uses the Claude Agent SDK's native
  `output_format={"type": "json_schema", ...}` instead of the
  forced `submit_result` MCP tool. The SDK enforces the schema
  server-side; the validated payload lands on
  `ResultMessage.structured_output`. (Already shipped in 0.1.x via
  `479b576`; called out here because consumers using the lower-level
  capture surface would notice.)
- `CostRecord.provider_id` now reflects the canonical `PROVIDER_ID`
  (e.g. `"claude"`, `"codex"`, `"github-copilot"`) rather than
  vendor-name strings (`"anthropic"`, `"openai"`).

### Migration

```python
# Before (0.1.x)
ProviderModel("anthropic", "claude-haiku-4-5")
ProviderModel("openai", "gpt-5-codex")
ProviderModel("copilot", "gpt-5-mini")
ProviderModel("opencode-zen", "gpt-5-nano")

# After (0.2.0)
ProviderModel("claude", "claude-haiku-4-5")
ProviderModel("codex", "gpt-5-codex")
ProviderModel("github-copilot", "gpt-5-mini")
ProviderModel("opencode", "gpt-5-nano")
```

## [0.1.0] — 2026-05-15

Initial release. Extracted from the Maverick project after stabilising
through five migration phases.

### Added

- `AgentRuntime` protocol with `execute / reset / close /
  validate_binding`.
- `RuntimeResult`, `CostRecord`, `ProviderModel` data types.
- Vendor-agnostic error hierarchy: `AgentRuntimeError` base plus
  `RuntimeAuthError`, `RuntimeModelNotFoundError`,
  `RuntimeTransientError`, `RuntimeStructuredOutputError`,
  `RuntimeContextOverflowError`, `RuntimeProtocolError`,
  `RuntimeServerStartError`, `RuntimeCancelledError`.
- Four adapters:
  - `ClaudeCodeRuntime` via `claude-agent-sdk`. Claude Max
    subscription / OAuth / API-key auth. Forced `submit_result`
    MCP tool for structured output.
  - `CopilotRuntime` via `github-copilot-sdk`. GitHub Copilot
    subscription. Rejects `claude-*` model IDs (Phase 0 finding:
    Claude on Copilot can't honour tool calls).
  - `CodexRuntime` via `openai-codex-sdk`. ChatGPT Plus
    subscription / `OPENAI_API_KEY`. Native JSON-schema mode (no
    tool-forcing required).
  - `OpenCodeZenRuntime` via the `openai` SDK pointed at
    `https://opencode.ai/zen/v1`. Direct HTTP; no subprocess.
- Optional dependency extras: `[claude]`, `[copilot]`, `[codex]`,
  `[opencode-zen]`, `[all]`.
- 115 unit tests; 4 end-to-end probe scripts under `examples/`.
