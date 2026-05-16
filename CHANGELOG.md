# Changelog

All notable changes to airframe are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
