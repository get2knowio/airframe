# `KimiRuntime` adapter plan

Companion to [implementation-plan.md](./implementation-plan.md). Specs
a new built-in adapter wrapping **Moonshot AI's Kimi Agent SDK**
(`kimi-agent-sdk` on PyPI; first-party from the `MoonshotAI` org).
The SDK is a Python wrapper around the `kimi-cli` subprocess — same
architectural class as the Claude Agent SDK / Codex SDK / Copilot
SDK adapters already in the lineup.

The plan mirrors the ABCDEF iteration cadence used by
[`bedrock-adapter-plan.md`](./bedrock-adapter-plan.md) and
[`google-genai-adapter-plan.md`](./google-genai-adapter-plan.md):
scaffold → execute/stream/cancel → polymorphic prompt + reasoning →
tools + permission → hooks + budget → wrap-up. Targeting
~600–800 LOC over 6 iterations, each independently mergeable.

## Motivation

Airframe ships four subprocess-class adapters today
(`ClaudeCodeRuntime`, `CopilotRuntime`, `CodexRuntime`) and one
server-class adapter (`BedrockRuntime`). Every one of them is bound
to a **closed-weight model house** (Anthropic, GitHub/Microsoft,
OpenAI, AWS-curated). The only open-weight access today is via the
OpenAI-compat gateway adapters (`OpenCodeZenRuntime`,
`OpenCodeGoRuntime`, `OpenRouterRuntime`) which deliver tokens but
no agentic surface — chat-completions with no sessions, no
permission gating, no MCP, no lifecycle hooks.

Moonshot's Kimi Agent SDK is the first vendor-shipped agent SDK for
an open-weight model line that meets the agent-SDK bar:

- **Subprocess-class agent.** Thin Python wrapper around the
  `kimi-cli` binary; session lifecycle, approvals, MCP, streaming
  all surface through the same architectural shape as Claude Agent
  SDK. Airframe gets to reuse the patterns already proven in
  `ClaudeCodeRuntime`.
- **Open-weight target.** Default model is `kimi-k2-thinking-turbo`
  (Kimi K2 / K2.6 line — open-weight model with native reasoning
  output, served via Moonshot's hosted endpoint). Closes the lineup
  gap "agent SDK over an open-weight model house."
- **First-party.** `kimi-agent-sdk` is published by the `MoonshotAI`
  GitHub org; not a community wrapper. Apache-2.0 licensed.

Distinct from the planned `OpenCodeServerRuntime` (model-agnostic
agent server fronting *any* backend) — `KimiRuntime` is bound to
the Kimi model line. Both are reasonable answers to "I want
agentics on open-weight models" but for different consumer shapes:
Kimi is the right pick when you want Moonshot-tuned agentic
behaviour; OpenCode is the right pick when you want backend
flexibility.

## Non-goals

- **No bundled `kimi-cli` install.** Same posture as
  `ClaudeCodeRuntime` and `CopilotRuntime`: the user installs the
  underlying CLI themselves (via `kimi-agent-sdk`'s own bootstrap
  or upstream's documented install path). The adapter validates
  the CLI is present at first call and classifies its absence as
  `RuntimeServerStartError` with a clear hint.
- **No `kimi-coder` / `kimi-cli` direct wrapping.** Wrap the
  Python SDK, not the CLI. The SDK is what Moonshot maintains as
  the stable Python surface; the CLI is the implementation detail
  underneath.
- **No `moonshot-python` / Chat Completions integration.** Moonshot
  also exposes Kimi behind an OpenAI-compat endpoint at
  `https://api.moonshot.ai/v1`; that surface is already reachable
  through `OpenAICompatibleRuntime` and a future thin subclass
  (`MoonshotRuntime` — separate provider ID, OpenAI-compat family)
  is the right home for "I want a Kimi chat completion" without
  agentics. `KimiRuntime` is specifically for the agent SDK
  surface.
- **No custom model-routing logic.** If the user wants
  multi-backend, that's the OpenRouter / OpenCode story. Kimi
  Agent SDK targets a single Moonshot endpoint by design.

## Adapter shape

`KimiRuntime(AgentRuntime)` — a direct subclass of `AgentRuntime`,
not `OpenAICompatibleRuntime`. Reasons mirror the rationale for
`ClaudeCodeRuntime` / `CopilotRuntime` / `CodexRuntime`:

1. Wire shape isn't HTTP-chat — it's subprocess + IPC via the
   Python SDK's session abstraction.
2. Session state lives in the SDK's `Session` object (which itself
   shells the CLI). Multi-turn happens inside one `Session`; the
   adapter owns the session lifecycle.
3. Approval requests arrive *during* a turn, not before — that's
   the permission-callback surface and it lives on `Session`.

`KimiSession(AgentSession)` is the primary surface. It owns a
single `Session` instance from `kimi_agent_sdk` opened lazily on
first `execute()` / `stream()` and torn down at `close()`.
`AgentSession.id` populates from whatever stable identifier the
SDK exposes (Iteration A surfaces this — depends on whether
`Session.create` returns an ID or whether one is assigned by the
CLI at first prompt).

```
src/airframe/adapters/kimi.py                  ~600 LOC (target)
tests/test_kimi.py                             ~400 LOC mirroring test_claude_code.py
tests/test_kimi_session.py                     ~200 LOC session-class behaviour
tests/test_kimi_conformance.py                 airframe.testing.contracts driver
tests/test_kimi_integration.py                 pytest-marker-gated live tests
docs/adapters/kimi.md                          ~180 lines
examples/probe_kimi.py                         single-call execute(schema=) probe
```

| Class attribute | Value |
|---|---|
| `PROVIDER_ID` | `"kimi"` |
| `REQUIRES_PACKAGE` | `"kimi-agent-sdk"` |
| `EXTRA_NAME` | `"kimi"` |
| `label` | `"kimi"` |

**Provider ID reservation note.** `"kimi"` is not yet in the
reserved-IDs list in `CLAUDE.md`. Iteration A adds it. `"moonshot"`
is **reserved** for a future OpenAI-compat sibling (`MoonshotRuntime`
fronting the `api.moonshot.ai/v1` chat-completions endpoint) — do
**not** fold the two together. Same vendor with distinct surfaces =
distinct provider IDs, per the lesson documented in
`bedrock-adapter-plan.md` (`"bedrock"` vs `"bedrock-agents"`) and
`opencode-adapter-plan.md` (`"opencode"` vs `"opencode-zen"` vs
`"opencode-go"`).

## SDK surface this adapter wraps

`kimi-agent-sdk` ≥ 0.0.5 (Feb 2026); Python ≥ 3.12. Apache-2.0.
Relevant surface from the quickstart guide:

- **Imports.**
  ```python
  from kimi_agent_sdk import (
      Config, Session, prompt,
      ApprovalRequest, TextPart,
  )
  from kaos.path import KaosPath
  ```
- **High-level `prompt()`** — one-shot helper that auto-creates a
  temp session, streams messages, supports `yolo=True` for
  auto-approval. Equivalent to airframe's `runtime.execute()`
  sugar. Airframe drives the low-level `Session` API instead, for
  uniform behaviour with the per-session features.
- **Low-level `Session`** — async-context-manager lifecycle:
  ```python
  async with await Session.create(work_dir=KaosPath.cwd()) as session:
      ...
  ```
  Persistent across multiple prompts; state resumable; manual
  approval handling; per-session `work_dir`.
- **Approval callbacks.** `ApprovalRequest` objects surface during
  the streamed response; resolved via `req.resolve("approve")` /
  `req.resolve("deny")`. `yolo=True` on the high-level helper
  auto-approves everything. Low-level callers pattern-match on
  request type and decide.
- **Streaming.** Async generators yield typed message parts
  (`TextPart` and friends — Iteration B enumerates the full set).
- **Config.** `Config` object accepts overrides for the env-derived
  defaults. Layered config: env vars → `Config` instance → TOML
  file with explicit precedence.
- **MCP.** Inherits from `kimi-cli`'s MCP support — registered
  servers configured in the SDK's config object are made available
  to the model.

The SDK is a **thin wrapper** around `kimi-cli`. Tool execution,
schema enforcement, and all the heavyweight behaviour live in the
CLI binary; the SDK's role is to spawn it, drive its stdin/stdout,
and surface typed events.

## Feature support matrix (target)

| Feature | Status | Notes |
|---|---|---|
| `STRUCTURED_OUTPUT_JSON_SCHEMA` | ✓ | Verify the SDK exposes a JSON-Schema constraint kwarg on `Session`/`prompt`; if not, implement via forced-tool pattern (mirror `CopilotRuntime`). Decided at Iteration B. |
| `STRUCTURED_OUTPUT_STRICT` | ✗ | No documented strict-mode toggle. |
| `STREAMING` | ✓ | Native via the async-generator surface. Translate `TextPart` → `TextDelta`; reasoning parts → `ReasoningDelta` (Kimi K2 line emits explicit reasoning blocks). |
| `CANCEL` | ✓ | Verify the SDK exposes an `abort()` / `cancel()` on `Session`; if not, fall back to `asyncio.Task.cancel()` + close the session context. |
| `SESSION_RESUME` | ✓ | SDK quickstart claims "state resumption." Verify the exact resume-by-ID mechanism in Iteration B. If only file-based / filesystem resume, surface as documented but flip the feature flag carefully. |
| `REASONING_EFFORT` | ◐ pass-through | Kimi K2-thinking-turbo emits reasoning natively. Whether the SDK accepts a `reasoning_effort` knob or whether the model decides is verified in Iteration C. Conservative default: report True with a debug-log warning if no SDK kwarg exists. |
| `REASONING_BUDGET_TOKENS` | ✗ | No documented budget-tokens knob. |
| `VISION_INPUT` | ◐ model-gated | Kimi K2 supports vision per model card; verify the SDK accepts image content parts in Iteration C. If yes, flip True. |
| `FILE_INPUT` | ◐ model-gated | Same — Kimi CLI reads files via tools; whether the SDK accepts direct file content parts is the question. |
| `TOOLS_FUNCTION` | ✓ | Subprocess-class SDK with a tool registration model. Iteration D wires caller-defined `FunctionTool` registration through the SDK's tool API. |
| `TOOLS_MCP_STDIO` / `_HTTP` / `_SSE` | ✓ | The SDK inherits Kimi CLI's MCP support; `Config` accepts MCP server entries. Iteration D wires the `McpServerRef` translation. |
| `TOOLS_MCP_IN_PROCESS` | ✗ | No in-process MCP slot documented (matches the Bedrock / OpenCode posture). |
| `PERMISSION_CALLBACK` | ✓ | **Natively first-class via `ApprovalRequest`.** Iteration D wires the dispatch. |
| `LIFECYCLE_HOOKS` | ✓ (6 kinds target) | Synthesise from the SDK's streamed message events: `session_start`, `session_end`, `user_prompt_submit`, `pre_tool_use`, `post_tool_use`, `tool_failure`. `pre_compact` / `rate_limit` likely False (Kimi CLI doesn't surface those upstream). |
| `BUDGET_USD_CAP` | ✓ | Client-side accumulation against an in-tree `_KIMI_PRICING` table (Kimi K2 pricing is published on Moonshot's site). |
| `BUDGET_TURN_CAP` | ✓ | Client-side per-session counter, same as siblings. |
| `SANDBOX` | ✗ | Server-side execution is the CLI's job; no airframe-surfaced sandbox kwarg. |
| `SUBAGENTS` | ✗ | No documented subagent API; defer. |

Iteration B's verification step pins each `◐` row to a definite
✓ or ✗ — the matrix above is the *target*; reality is what the
SDK accepts.

## Auth chain

Mirrors the existing four adapters' "checked in order" pattern:

1. **Explicit `api_key=` / `base_url=` / `model=` constructor args.**
   Highest precedence. Forwarded into the SDK's `Config`.
2. **`KIMI_API_KEY` env var.** The SDK's native env var; honoured
   automatically by `Config` when constructor kwargs aren't set.
3. **`KIMI_BASE_URL` env var.** Defaults to
   `https://api.moonshot.ai/v1` when unset.
4. **`KIMI_MODEL_NAME` env var.** Defaults to
   `kimi-k2-thinking-turbo` when unset.

If no API key resolves through any of the three, the first network
call raises `RuntimeAuthError` with a hint pointing at
`https://platform.moonshot.ai/console/api-keys`.

`list_models()` requires a credential (Moonshot's models endpoint
doesn't honour anonymous calls). Same `pytest.skip`-itself pattern
as the other adapters when credentials aren't set.

Auth chain documented in `docs/auth.md#kimiruntime` (added in
Iteration F).

## `KimiOptions` (provider-options namespace)

```python
@dataclass(frozen=True, slots=True)
class KimiOptions:
    # Per-session working directory. The CLI's filesystem-affecting
    # tools (read/write/edit/bash) operate relative to this dir.
    # Maps to Session.create(work_dir=...).
    working_directory: str | None = None

    # YOLO mode — auto-approve all tool/shell invocations. Maps to
    # the high-level `prompt(yolo=True)` shortcut but exposed here
    # for callers using the session API who want the same
    # convenience without registering a PermissionCallback that
    # always returns "allow". Mutually exclusive with on_permission=;
    # adapter raises UnsupportedFeatureError if both are set.
    yolo: bool = False

    # MCP server config entries applied at session start.
    # Translated to the SDK's Config.mcp_servers slot. Use airframe's
    # McpServerRef types in Phase-4-aware code; here as a provider-
    # specific additional channel for vendor-specific knobs.
    additional_mcp_servers: tuple[Any, ...] = ()

    # Skills directory (the SDK inherits Kimi CLI's skills surface).
    skill_directories: tuple[str, ...] = ()

    # Pass-through to the SDK's Config object for vendor-specific
    # knobs we don't surface portably.
    additional_config_fields: dict[str, Any] | None = None
```

Same tagged-union discipline as the existing namespaces — mismatched
type raises `UnsupportedFeatureError` at `session(provider_options=)`.

## Iteration breakdown

### Iteration A — Protocol scaffolding (no behaviour)

~150 LOC. Lands the adapter's shape without wiring substantive
features.

- `src/airframe/adapters/kimi.py` — `KimiRuntime(AgentRuntime)`
  with ClassVars, lazy `kimi_agent_sdk` import (deferred to first
  method call), `validate_binding` (accepts any non-empty
  `model_id` when `provider_id == "kimi"` and the string starts
  with `kimi-` — analogous to how `CopilotRuntime` gates `claude-*`
  bindings).
- `_resolve_auth()` chain implementing the four-step auth above.
- Empty `SUPPORTED_FEATURES = frozenset()` initially; flip flags
  on as each iteration wires them.
- `unwrap(Session)` for runtime/session-level escape hatches.
  Possibly also `unwrap(Config)` if config-time inspection is
  useful.
- `close()` idempotent + never raises (releases the SDK's resource;
  no subprocess to kill at runtime level because subprocess
  lifecycle is per-session).
- `reset()` no-op at the runtime level (sessions own state).
- `list_models()` — first decision point. Two options:
  (a) hit Moonshot's `/v1/models` directly via the OpenAI-compat
  endpoint (Kimi shares the auth scheme); (b) keep a hard-coded
  fallback catalogue. Recommend (a) with (b) as offline fallback,
  matching `ClaudeCodeRuntime`'s pattern.
- Add `"kimi"` to the reserved-IDs paragraph in `CLAUDE.md` and
  reserve `"moonshot"` alongside.
- Discovery registration in `discovery.py` + top-level export in
  `airframe/__init__.py`.
- Conformance contracts pass against a mocked `Session`.

**Stopping point.** `import airframe; airframe.list_providers()`
includes `"kimi"` when the extra is installed. Discovery + capability
predicates work; no live behaviour wired.

### Iteration B — Execute + streaming + cancellation + session-resume

~200 LOC. The Phase-1-equivalent slice.

- `KimiSession(AgentSession)` — owns a lazy `kimi_agent_sdk.Session`
  opened on first `execute()` / `stream()`. The session is built
  inside an `async with` block managed by airframe (so teardown
  goes through the SDK's documented context-manager exit).
- `runtime.session(resume=...)` — verify the SDK's resume model:
  if `Session.create` accepts a `session_id=` or `resume=` kwarg,
  flip `SESSION_RESUME=True`; if resume is only filesystem-based
  (point `work_dir` at a previous run), document that constraint
  and decline `SESSION_RESUME` with a clear message until upstream
  exposes a runtime-resume API.
- `execute(prompt, schema=)` — drive a single turn through
  `Session`; collect the assistant response; close the session.
  Structured-output strategy decided here:
  - **Preferred:** if the SDK accepts a JSON-schema constraint
    kwarg, use it natively.
  - **Fallback:** forced-tool pattern with an in-SDK `submit_result`
    tool (mirror `CopilotRuntime`).
  Pick the path Iteration B's first PR and document.
- `stream(prompt, schema=)` — translate the SDK's async-generator
  output into airframe's `RuntimeEvent` union:
  - `TextPart` → `TextDelta`.
  - Reasoning parts (verify exact type name in Iteration B) →
    `ReasoningDelta`.
  - Tool-call parts → `ToolCallStart` / `ToolCallResult` (minimal
    in B; Iteration D wires the full surface).
  - Terminal event → `TurnComplete` with populated `RuntimeResult`.
- `cancel()` — if `Session.abort()` exists, call it; otherwise
  cancel the wrapping `asyncio.Task` and close the session
  context (the SDK's `__aexit__` terminates the subprocess).
- `close()` — idempotent; awaits the session's `__aexit__` if
  still open; clears state.
- Exception classification: enumerate `kimi-agent-sdk`'s exception
  types (Iteration B's first task) and map to the
  `Runtime*Error` hierarchy. `RuntimeServerStartError` when the
  CLI binary isn't on PATH.
- Flip `Feature.STREAMING`, `Feature.CANCEL`,
  `Feature.STRUCTURED_OUTPUT_JSON_SCHEMA` True. `SESSION_RESUME`
  flip is contingent on the verification.
- `examples/probe_kimi.py` — minimal `execute(schema=)` probe.

**Stopping point.** Single-turn structured output works against a
live Moonshot endpoint with `KIMI_API_KEY` set. Streaming yields
deltas. Cancellation works.

### Iteration C — Polymorphic prompt + reasoning

~100 LOC. The Phase-2-equivalent slice.

- `_split_prompt_parts` integration: `ImageInput` / `FileInput` →
  the SDK's content-part shape (verified in this iteration; if
  the SDK doesn't accept either, decline both features clearly
  rather than emulating).
- `thinking=` kwarg → `reasoning_effort` SDK kwarg if exposed.
  Map `"low"`/`"medium"`/`"high"` to the SDK's accepted strings.
  `"minimal"` coerces to `"low"` with debug log (matches Claude /
  Copilot / Bedrock policy). `{"budget_tokens": N}` declined —
  Kimi has no documented budget-tokens knob.
- Flip `Feature.REASONING_EFFORT` True (if SDK kwarg exists);
  `VISION_INPUT` / `FILE_INPUT` per verified support.
- Probe extensions: `probe_thinking.py`, `probe_vision.py` get
  `kimi` branches.

### Iteration D — Function tools + permission callback + MCP

~200 LOC. Phase-3 + Phase-5 (permission slice) + Phase-4 (MCP)
equivalent.

**Function tools.** Translate `FunctionTool` → the SDK's tool
registration API (verified at iteration start). Airframe owns the
tool-loop surface from the consumer's perspective; the SDK
internally drives the model-side tool calls.

**Permission callback (the natural fit).** The SDK's
`ApprovalRequest` objects surface during the streamed message
flow. Dispatch each request to the registered `PermissionCallback`,
call `req.resolve("approve")` or `req.resolve("deny")` based on
the decision. This is the cleanest `PERMISSION_CALLBACK`
implementation in the lineup alongside `OpenCodeServerRuntime` —
no synthesis from a client-side tool loop, no special handling.

**MCP refs.** Translate airframe's `McpServerRef` shapes to the
SDK's `Config.mcp_servers` slot at session start. `STDIO` →
local-process descriptor; `HTTP`/`SSE` → remote URL descriptor.

- `FunctionTool` ↔ SDK tool registration.
- `PermissionCallback` ↔ `ApprovalRequest` dispatch.
- `McpServerRef` ↔ `Config.mcp_servers`.
- Flip `Feature.TOOLS_FUNCTION`, `Feature.TOOLS_MCP_STDIO`,
  `Feature.TOOLS_MCP_HTTP`, `Feature.TOOLS_MCP_SSE`,
  `Feature.PERMISSION_CALLBACK` True.

`TOOLS_MCP_IN_PROCESS` stays permanently False — no in-process
MCP slot in the SDK.

### Iteration E — Hooks + budget

~100 LOC. Phase-5 equivalent.

- `EMITTABLE_HOOK_KINDS` ClassVar — six kinds initially:
  `session_start`, `session_end`, `user_prompt_submit`,
  `pre_tool_use`, `post_tool_use`, `tool_failure`. Add
  `pre_compact` if the SDK surfaces compaction events; add
  `rate_limit` if the SDK surfaces 429s as typed events (verify
  upstream).
- `_on_event` plumbing on `KimiSession` — synthesise hook events
  from the SDK's streamed message types where airframe's hook
  taxonomy maps cleanly.
- Budget-cap enforcement via the shared
  `_enforce_budget_pre_turn()` helper.
- In-tree `_KIMI_PRICING` table for the published Kimi models.
  Source: Moonshot's pricing page (verify rates at PR time; point-in-
  time-document as the Codex/Bedrock plans do).
- Flip `Feature.LIFECYCLE_HOOKS`, `Feature.BUDGET_USD_CAP`,
  `Feature.BUDGET_TURN_CAP` True.

### Iteration F — Wrap-up

~80 LOC + docs.

- `KimiOptions` dataclass (per the surface above) wired through
  with `_check_provider_options`.
- Conformance contract suite green; integration test wrapper at
  `tests/test_kimi_integration.py` with the standard
  `pytest-marker` gating.
- Per-adapter docs page: `docs/adapters/kimi.md` covering install
  extra (`pip install airframe-agents[kimi]`), CLI prerequisites
  (the user installs `kimi-cli` themselves — same posture as
  `claude.md` documents for the Claude CLI), auth chain
  (cross-link to `docs/auth.md`), supported features, options
  reference, model IDs, structured-output mechanism, vendor
  quirks, native escape hatches.
- `docs/auth.md` — new `## KimiRuntime` section.
- `docs/capabilities.md` matrix gains a Kimi column.
- README "Supported providers" table row (alphabetised between
  `CopilotRuntime` and `OpenCodeGoRuntime`).
- CHANGELOG entry.

## Risks and decisions to flag during execution

1. **Pre-1.0 SDK churn.** `kimi-agent-sdk` is at 0.0.5 (Feb 2026).
   The package is young and the upstream surface may shift faster
   than `claude-agent-sdk`. Pin `kimi-agent-sdk>=0.0.5,<0.1` and
   plan to bump deliberately. Watch the
   `MoonshotAI/kimi-agent-sdk` changelog separately from the
   underlying `kimi-cli` changelog.
2. **Python 3.12 minimum is stricter than airframe's 3.11.**
   The `kimi-agent-sdk` package requires Python ≥3.12; airframe
   itself supports ≥3.11. Implication: users on Python 3.11 can
   `pip install airframe-agents` but the `[kimi]` extra will fail
   to resolve. Document this explicitly in the extra's note and
   in `docs/adapters/kimi.md`; the failure mode is loud (pip
   reports it) so this is annoying-but-survivable, not silent.
3. **`Session.create` semantics not fully documented.** The
   quickstart shows
   `async with await Session.create(work_dir=KaosPath.cwd())`,
   but the resume API, the cancellation API, and the exception
   taxonomy aren't well-documented yet. Iteration B's first task
   is verifying these by reading the SDK source — budget half a
   day for that exploration.
4. **`KaosPath` is a leaky abstraction.** The SDK imports
   `KaosPath` from `kaos.path` — a Moonshot-internal path type.
   Airframe shouldn't expose this; the adapter accepts `str | Path`
   on `KimiOptions.working_directory` and converts to `KaosPath`
   at the boundary.
5. **Subprocess discovery.** Like `ClaudeCodeRuntime`'s posture
   on `claude` CLI, the adapter does not bundle `kimi-cli` —
   the user installs it themselves. First-call reachability
   classification should be high-quality; if the CLI is missing
   from PATH, surface a clear `RuntimeServerStartError` ("install
   kimi-cli — see https://github.com/MoonshotAI/kimi-cli").
6. **Approval-callback ordering.** Verify whether
   `ApprovalRequest` events arrive on the same stream as
   `TextPart` / reasoning parts or on a separate channel. If
   same stream, the adapter must dispatch *during* the stream
   rather than after; if separate channel, a parallel listener
   pattern.
7. **YOLO mode interaction with `PermissionCallback`.**
   `KimiOptions.yolo=True` and `on_permission=callback` are
   mutually exclusive (one means "auto-approve everything," the
   other means "ask the callback"). The adapter raises
   `UnsupportedFeatureError` at `session(...)` if both are set
   non-falsy.
8. **Vendor-specific reasoning behaviour.** Kimi K2-thinking-turbo
   emits reasoning natively but doesn't accept an effort knob in
   the same shape as Claude / Codex. Whether
   `Feature.REASONING_EFFORT` returns True is contingent on the
   SDK accepting *some* knob; if it doesn't, decline cleanly.
9. **Pricing-table maintenance.** Moonshot publishes pricing in
   USD per million tokens. The adapter keeps the table close to
   `_BEDROCK_PRICING` shape so updates are one-PR churn. Consider
   moving the shared shape into `src/airframe/_pricing.py` if a
   fourth pricing-table adapter appears (Codex, Bedrock, Kimi
   would be three).
10. **Subprocess crash on bad model ID.** Same risk as
    `ClaudeCodeRuntime` — the spawned CLI may exit early on
    invalid input. Treat as `RuntimeTransientError` initially;
    refine the classification after Iteration B reveals the
    actual error surface.

## Definition of done

- All six iterations merged.
- `runtime_for("kimi")` returns `KimiRuntime` when the extra is
  installed; clean `ImportError` with `airframe-agents[kimi]` hint
  when not.
- `examples/probe_kimi.py` round-trips a structured-output prompt
  against live Moonshot with `KIMI_API_KEY` set.
- `examples/probe_parity.py` includes `kimi` with no per-vendor
  conditionals; passes on a machine with credentials + Python
  3.12+.
- `examples/probe_supports.py --provider kimi` shows the expected
  feature matrix.
- Conformance contract suite green against a mocked
  `kimi_agent_sdk.Session`; integration suite green against a live
  endpoint.
- `docs/adapters/kimi.md` published; README provider table +
  capability matrix updated.
- CHANGELOG entry with iteration summary.
- `CLAUDE.md` reserved-IDs list includes `"kimi"` and reserves
  `"moonshot"`.

## When to start

**Phase 1 candidate** alongside `OpenCodeServerRuntime`,
`BedrockRuntime` (shipped), `GeminiRuntime`, and `MistralRuntime`.
Kimi is the lowest-friction of the open-weight-agent options
because the SDK shape is so close to `claude-agent-sdk` that most
of `ClaudeCodeRuntime`'s patterns transfer 1:1.

Reasonable cadence: one iteration per week, ~6 weeks end-to-end.
Iteration A is the trivial scaffold; B–E carry the substantive
work; F is wrap-up. Mergeable in parallel with Mistral / OpenCode /
Gemini since they touch disjoint files.

Two triggers would make it Phase-1-priority work:

1. **A consumer asks for open-weight agentics with model-house-
   tuned behaviour.** Kimi K2 is competitive with Claude / GPT-4
   on coding benchmarks and has a distinct agentic style;
   consumers wanting that specifically would pick this over the
   model-agnostic `OpenCodeServerRuntime`.
2. **Cost-sensitive workflows.** Moonshot's pricing is
   significantly below Anthropic / OpenAI for comparable model
   sizes. Consumers running high-volume agentic flows might
   prioritise this over the closed-weight alternatives.

Until either fires, the workaround is `OpenRouterRuntime` against
`moonshotai/kimi-k2-...` model IDs — chat-only, no agentics, but
unblocks "I want to call Kimi" today.

## Open questions for the implementer

1. **`Session.create()` exact signature.** Resume semantics, the
   full kwargs set, and the streaming-iterator protocol need
   verification from the SDK source. Iteration B kickoff.
2. **Tool registration API.** Whether the SDK exposes Python
   callables as tools (like `claude-agent-sdk`'s `@tool` decorator)
   or only configuration-driven tools (TOML / MCP). Iteration D
   kickoff.
3. **Reasoning-effort kwarg.** Whether `Session.run()` /
   `prompt()` accept a `reasoning_effort` kwarg, and what values
   it accepts. Iteration C kickoff.
4. **Exception taxonomy.** What does the SDK raise on auth
   failure, model-not-found, transient errors, subprocess crash?
   Map cleanly to `Runtime*Error`.
5. **`MoonshotRuntime` sibling timing.** A thin
   `OpenAICompatibleRuntime` subclass for the chat-completions
   endpoint is a separate small adapter — ~30 LOC. Worth
   shipping alongside `KimiRuntime` (different provider ID,
   `"moonshot"`) so "I want chat completions on Kimi" has a path
   that doesn't require the heavyweight subprocess SDK. Track
   separately.

## Implementation wiring checklist

Beyond `src/airframe/adapters/kimi.py` itself, every new adapter
needs to touch these files. Easy to forget; easy to verify by
grepping for the closest sibling (`claude_code.py`).

### Source wiring

- [ ] `src/airframe/discovery.py` — add `KimiRuntime` to
      `_builtin_runtime_classes()`.
- [ ] `src/airframe/__init__.py` — `from airframe.adapters.kimi
      import KimiRuntime` at module level + entry in `__all__`
      (alphabetical: between `CopilotRuntime` and
      `OpenAICompatibleRuntime`).
- [ ] `src/airframe/testing/contracts.py` — add `"kimi":
      KimiOptions` to the `matching` dict inside
      `_check_provider_options`.
- [ ] `src/airframe/testing/integration.py` — add `"kimi":
      ["KIMI_API_KEY"]` to `_PROVIDER_AUTH`.

### Probe + examples wiring

- [ ] `examples/probe_budget.py` — add `"kimi"` to the for-loop
      provider tuple.
- [ ] `examples/probe_parity.py` — picks up the new adapter
      automatically. Consider adding an
      `AIRFRAME_PROBE_MODEL_KIMI` env-var hook if the default
      model needs runtime override.

### Packaging

- [ ] `pyproject.toml` — new `kimi = ["kimi-agent-sdk>=0.0.5,<0.1"]`
      extra under `[project.optional-dependencies]`, AND add to
      the `all = [...]` list, AND add to the
      `[dependency-groups].test` list. Document the
      Python ≥3.12 constraint in the extra's comment.

### Documentation

- [ ] `README.md` — provider table row, install one-liner,
      tagline (the seven-adapter list gains an eighth).
- [ ] `docs/auth.md` — quick-reference table row + full
      `## KimiRuntime` section.
- [ ] `docs/reference.md` — adapter table row + `KimiRuntime`
      in the `__all__` snippet.
- [ ] `docs/adapters/kimi.md` — new page; mirror `claude.md`
      structure.
- [ ] `docs/capabilities.md` — add `kimi` column to the matrix.
- [ ] `docs/architecture.md` — new `### Kimi Agent SDK`
      subsection under "Operational landmines" once Iterations
      A–D land (defer the subsection until landmines are
      empirically known).
- [ ] `CLAUDE.md` — add `"kimi"` to canonical IDs list; reserve
      `"moonshot"`.

### Test wiring

- [ ] `tests/test_kimi.py` — unit tests. Mirror
      `tests/test_claude_code.py` for structure (full-bespoke
      subprocess template).
- [ ] `tests/test_kimi_session.py` — session-class behaviour,
      streaming-event translation, approval-callback dispatch.
- [ ] `tests/test_kimi_conformance.py` — drives the shared
      contracts.
- [ ] `tests/test_kimi_integration.py` — pytest-marker-gated
      against live Moonshot endpoint.
- [ ] `tests/test_discovery.py` — update expected sets +
      filtered tests + third-party-discovery test.

### Issue + project housekeeping

- [ ] Open a tracking issue mirroring the Bedrock pattern.
- [ ] CHANGELOG entry at each iteration merge.

## Closest in-tree templates to read first

| File | What to learn from it |
|---|---|
| `src/airframe/adapters/claude_code.py` | **The primary template.** Subprocess + JSON-RPC SDK, session lifecycle, approval callback, MCP. `KimiRuntime` should mirror this structurally. ~800 LOC; Kimi target ~600 LOC because the Kimi SDK does less heavyweight schema enforcement at the Python boundary. |
| `src/airframe/adapters/copilot.py` | Permission-callback pattern + forced-tool-for-structured-output (in case the SDK doesn't ship native JSON-schema enforcement). |
| `src/airframe/adapters/codex.py` | Subprocess-per-turn pattern (if `kimi-agent-sdk` turns out to spawn a fresh subprocess per call rather than holding a long-lived session). The `_strictify_schema` helper if Kimi requires `additionalProperties: false` on tool schemas (likely — most subprocess CLIs do). |
| `src/airframe/adapters/bedrock.py` | Pricing-table pattern (`_BEDROCK_PRICING`) — Kimi adopts the same shape for `_KIMI_PRICING`. |
| `src/airframe/sessions.py` | Shared helpers: `_enforce_budget_pre_turn`, `_check_provider_options`. |

## Naming reservations

Established with this plan:

- `"kimi"` — this adapter (agent SDK; subprocess-class).
- `"moonshot"` — **reserved** for a future `MoonshotRuntime`
  (OpenAI-compat thin subclass fronting `api.moonshot.ai/v1`).
  Different surface, different shape, different feature matrix
  — distinct provider ID per the established pattern.
- `"kimi-cli"`, `"moonshotai"`, `"kimi-code"` — **not reserved**.
  The `"kimi"` namespace covers the agent SDK; the `"moonshot"`
  namespace covers the chat-completions API. No other variants
  needed.

## First commit in a fresh session

A reasonable Iteration A first commit:

```
src/airframe/adapters/kimi.py            # new — Iteration A surface
src/airframe/discovery.py                # +KimiRuntime in builtins
src/airframe/__init__.py                 # +export +__all__
pyproject.toml                           # +[kimi] extra
tests/test_kimi.py                       # new — identity, validate_binding, auth-chain unit tests
tests/test_discovery.py                  # +kimi in expected sets
CLAUDE.md                                # +kimi in reserved-IDs; +moonshot reservation
```

That should pass `make ci` cleanly with `Feature` flags all False
(no behaviour wired yet). After review, Iteration B adds
`execute()` + `stream()` + `cancel()` + `session(resume=...)` and
flips the first three or four `Feature` flags True.
