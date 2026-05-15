# Changelog

All notable changes to airframe are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-05-15

Initial release. Extracted from the Maverick project after stabilising
through five migration phases.

### Added

- `AgentRuntime` protocol with `execute / reset / aclose /
  validate_binding`.
- `RuntimeResult`, `CostRecord`, `ProviderModel`, `Tier` data types.
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
