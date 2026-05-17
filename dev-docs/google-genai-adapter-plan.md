# `GoogleGenaiRuntime` adapter plan

Companion to [implementation-plan.md](./implementation-plan.md).
The phased plan there covers Phase 0–5 across the four built-in
adapters; this doc specs the fifth adapter — a direct wrapper
around the official `google-genai` Python SDK — as signal-gated
post-1.0 work.

The plan deliberately mirrors the ABCD iteration cadence that
shipped Phases 3 / 4 / 5 cleanly: scaffold → hard-feature wiring →
medium-feature wiring → declines + probe + wrap-up. Aiming for
~400–600 LOC and 6 iterations, each independently mergeable.

## Motivation

Airframe has no Gemini path today. The closest workaround
(`OpenCodeZenRuntime` against OpenRouter) loses Gemini-specific
capability — thinking budget, native multimodal Parts, citation
grounding, Vertex AI mode. A direct `google-genai` wrapper:

- Fills the obvious gap in the four-adapter matrix.
- Matches airframe's existing "narrow protocol over a vendor SDK"
  shape (compare `OpenCodeZenRuntime` at ~30 LOC over the
  `OpenAICompatibleRuntime` base).
- Is *not* an ADK wrapper. See [`google-adk-evaluation.md`](./google-adk-evaluation.md)
  (TODO if we decide to formalise that research) for why ADK is
  the wrong first Google adapter.

## Non-goals

- **No subscription-OAuth path.** A Gemini Pro / Ultra consumer
  subscription does **not** unlock `google-genai`; the SDK accepts
  only `GEMINI_API_KEY` / `GOOGLE_API_KEY` env vars or GCP
  Application Default Credentials (Vertex mode). Subscription-only
  users need to mint an API key from `aistudio.google.com` (free
  tier covers most dev work). A separate `GeminiCliRuntime` that
  subprocess-bridges to `gemini-cli` is the right shape for
  subscription auth, but defer until Google ships an official
  agent SDK (`claude-agent-sdk` analogue) to wrap.
- **No ADK wrapping.** ADK is an orchestration framework; airframe
  is a narrow protocol. If a consumer wants ADK they should use
  ADK directly.
- **No Vertex AI-specific features.** Vertex mode is supported via
  the SDK's own env-var switch but the adapter doesn't surface
  Vertex-only knobs (model garden, provisioned throughput,
  managed datasets). Those belong in a future `GeminiVertexOptions`
  dataclass if demand materialises.

## Adapter shape

**Bespoke `AgentRuntime` subclass.** `google-genai` is not OpenAI
Chat-Completions wire-compatible — it has its own
`types.Content` / `types.Part` schema and its own response shape.
`OpenAICompatibleRuntime` is wrong. It's also not subprocess-based
like Claude/Copilot/Codex — `google-genai` runs in-process and
talks direct HTTP. So it inherits `AgentRuntime` and lives at
~400–600 LOC.

| Class attribute | Value |
|---|---|
| `PROVIDER_ID` | `"gemini"` |
| `REQUIRES_PACKAGE` | `"google-genai"` |
| `EXTRA_NAME` | `"gemini"` |
| `label` | `"google_genai"` |

**Provider ID reservation note.** `"google"` is *reserved* for a
future direct Workspace / consumer-Gemini adapter (analogous to how
`"anthropic"` and `"openai"` are reserved). Using `"gemini"` keeps
the door open for `"vertex"`, `"google-adk"`, and
`"google"` later without conflicts.

## SDK surface this adapter wraps

From `google-genai` (current as of late 2025/early 2026):

- **Client.** `genai.Client(api_key=...)` for the Developer API,
  `genai.Client(vertexai=True, project=..., location=...)` for
  Vertex. Async equivalent via `client.aio.*`.
- **Generation.** `client.aio.models.generate_content(model, contents, config)`
  for one-shot; `generate_content_stream` for streaming.
- **Chat.** `client.aio.chats.create(model)` then
  `chat.send_message()` / `chat.send_message_stream()` — manages
  conversation history client-side.
- **Structured output.** `config=types.GenerateContentConfig(response_mime_type='application/json', response_json_schema=MyModel.model_json_schema())`.
- **Tools.** Three patterns. We use the manual `types.FunctionDeclaration`
  → `types.Tool` path; airframe drives the round-trip rather than
  letting the SDK auto-invoke (so the `handler` semantics stay
  consistent with the other adapters and `PermissionCallback`
  fires in a recognisable place).
- **Multimodal.** `types.Part.from_uri(...)`, `types.Part.from_bytes(...)`,
  plus the `client.files.upload()` upload path for large files.
- **Thinking.** `config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=N, include_thoughts=True))`.
  Newer Gemini 3+ uses `thinking_level` (LOW/MEDIUM/HIGH).
- **Usage.** Response carries `usage_metadata` with
  `prompt_token_count` / `candidates_token_count` / `total_token_count`.
  **No per-call dollar cost** — adapter computes from a Gemini
  pricing table (same shape as `_PRICING` on `CodexRuntime`).

## Feature support matrix (target)

| Feature | Support | Mechanism |
|---|---|---|
| `STRUCTURED_OUTPUT_JSON_SCHEMA` | ✓ | Native `response_json_schema` |
| `STRUCTURED_OUTPUT_STRICT` | ✗ | Gemini doesn't expose a strict-mode toggle |
| `STREAMING` | ✓ | `generate_content_stream` / `send_message_stream` |
| `CANCEL` | ✓ | `asyncio.Task.cancel()` propagates to httpx |
| `SESSION_RESUME` | ✗ | Chat history is client-side; no server-side session id. (Newer Interactions API is preview-only — revisit when GA) |
| `REASONING_EFFORT` | ✓ | `ThinkingConfig.thinking_level` |
| `REASONING_BUDGET_TOKENS` | ✓ | `ThinkingConfig.thinking_budget` (Claude-parallel) |
| `VISION_INPUT` | ✓ | Native `types.Part.from_bytes` / `from_uri` |
| `FILE_INPUT` | ✓ | Same Parts API plus `client.files.upload()` for large files |
| `TOOLS_FUNCTION` | ✓ | Manual `FunctionDeclaration` + airframe-driven round-trip |
| `TOOLS_MCP_STDIO` / `_HTTP` / `_SSE` | ✗ | `google-genai` has no MCP-server slot today. ADK does; if airframe ever ships `GoogleAdkRuntime` that covers MCP |
| `PERMISSION_CALLBACK` | ✓ | Airframe-driven (we hold the round-trip), fires per tool-call attempt |
| `LIFECYCLE_HOOKS` | ✓ (6 kinds) | Same shape as `OpenAICompatibleSession` — synthesise from the client-side tool loop. No `pre_compact` (no compaction concept on Gemini), no `rate_limit` (httpx surfaces 429 as a generic transient error) |
| `BUDGET_USD_CAP` | ✓ | Client-side accumulation against an in-tree pricing table |
| `BUDGET_TURN_CAP` | ✓ | Client-side per-session counter (Gemini chat has no native turn cap) |

`SESSION_RESUME=False` is the one notable miss — chat history
lives client-side in the SDK's `Chat` object, and the public API
has no `resume_chat(<id>)` equivalent. If the preview Interactions
API graduates to GA, revisit and flip the flag.

## Auth chain

Mirrors the existing four adapters' "checked in order" pattern:

1. `api_key=` constructor arg.
2. `GEMINI_API_KEY` env var.
3. `GOOGLE_API_KEY` env var (the SDK's preferred name; takes
   precedence in the SDK itself when both are set, but airframe
   checks `GEMINI_API_KEY` first to match the vendor docs that
   prefer the more specific name).
4. **Vertex AI mode** — when `GOOGLE_GENAI_USE_VERTEXAI=true` is
   set, defer to `google-genai`'s ADC chain (gcloud login,
   service-account JSON, workload identity). Requires
   `GOOGLE_CLOUD_PROJECT` + `GOOGLE_CLOUD_LOCATION` env vars or
   their constructor-arg equivalents.

`list_models()` requires a credential — Gemini's models endpoint
won't honour anonymous calls. Same auth-failure-via-skip pattern
as the other adapters: tests `pytest.skip` themselves when no
credential resolves.

Auth chain documented in `docs/auth.md#geminiruntime` (TODO when
this adapter lands; mirror the existing four entries).

## GeminiOptions (provider-options namespace)

Initially three fields — the obvious Gemini-only knobs that don't
fit the portable surface:

```python
@dataclass(frozen=True, slots=True)
class GeminiOptions:
    use_vertexai: bool = False
    project: str | None = None       # Vertex mode
    location: str | None = None      # Vertex mode (e.g. "us-central1")
    safety_settings: tuple[Any, ...] = ()   # HarmCategory / HarmBlockThreshold tuples
    candidate_count: int = 1         # n>1 multi-candidate responses
```

Same tagged-union discipline as the existing four namespaces —
mismatched type raises `UnsupportedFeatureError` at
`session(provider_options=)`.

## Iteration breakdown

### Iteration A — Protocol scaffolding (no behaviour)

~150 LOC. Lands the adapter's shape without wiring substantive
features.

- `src/airframe/adapters/gemini.py` — `GeminiRuntime(AgentRuntime)`
  with ClassVars, lazy SDK import (deferred to first method call),
  `validate_binding` (accepts `model_id` starting with `gemini-`;
  rejects `claude-*` / `gpt-*` / `o5-*` paths to match the
  vendor-routing discipline `CopilotRuntime` follows).
- `_resolve_api_key()` chain implementing the four-step auth above.
- Empty `SUPPORTED_FEATURES = frozenset()` initially; flip flags
  on as each iteration wires them.
- `unwrap(genai.Client)` runtime escape hatch.
- `close()` idempotent + never raises (the SDK has nothing to
  tear down; close is essentially a no-op but kept for protocol
  parity).
- `reset()` no-op (sessionless runtime).
- `list_models()` against the Developer API endpoint with a
  fallback hard-coded catalogue for offline tests (mirroring the
  Codex / OpenCode Zen `_METADATA` pattern).
- Entry-point registration in `pyproject.toml` (`gemini = "airframe.adapters.gemini:GeminiRuntime"`).
- Conformance contracts pass against a mocked client.

**Stopping point.** `import airframe; airframe.list_providers()` includes
`"gemini"` when the extra is installed. Discovery + capability
predicates work; no live behaviour wired.

### Iteration B — Execute + streaming + cancellation

~150 LOC. The Phase-1-equivalent slice — `session` factory,
`AgentSession` subclass, `execute()`, `stream()`, `cancel()`.

- `GeminiSession(AgentSession)` — client-side `Chat` object owned
  per-session (one `client.aio.chats.create()` per `session()`
  call). `id` stays `None` (no server-side session).
- `execute(prompt, schema=)` — single turn via
  `chat.send_message`; structured output via `response_json_schema`.
- `stream(prompt, schema=)` — translate `send_message_stream`
  chunks into airframe's `TextDelta` / `ReasoningDelta` /
  `TurnComplete` events.
- `cancel()` — `asyncio.Task.cancel()` propagates to httpx; the
  awaiting call raises `RuntimeCancelledError`.
- `close()` — idempotent; releases the `Chat` reference.
- Exception classification: `google.api_core.exceptions` and
  `google.genai.errors` → `RuntimeAuthError` /
  `RuntimeModelNotFoundError` / `RuntimeTransientError` /
  `RuntimeProtocolError` per the existing taxonomy.
- Flip `Feature.STREAMING`, `Feature.CANCEL`,
  `Feature.STRUCTURED_OUTPUT_JSON_SCHEMA` True.
- `examples/probe_gemini.py` — minimal `execute(schema=)` probe.

**Stopping point.** Single-turn structured output works against
live Gemini with `GEMINI_API_KEY` set. Streaming yields deltas.
Cancellation works.

### Iteration C — Polymorphic prompt + reasoning

~100 LOC. The Phase-2-equivalent slice.

- `_split_prompt_parts` integration: `ImageInput` / `FileInput` →
  `types.Part.from_bytes` (small) or `client.files.upload()` →
  `types.Part.from_uri` (large; threshold matches OpenAI-compat
  base ~5MB).
- `thinking=` kwarg → `ThinkingConfig`:
  - `"low"` / `"medium"` / `"high"` → `thinking_level` on
    Gemini 3+; falls back to `thinking_budget` mapping on
    Gemini 2.x.
  - `"minimal"` coerces to `"low"` with debug log (matches Claude
    / Copilot policy).
  - `{"budget_tokens": N}` → `thinking_budget=N` (Claude-parallel
    that works natively on Gemini — the only adapter outside
    Claude that honours the dict shape).
- Flip `Feature.REASONING_EFFORT`, `Feature.REASONING_BUDGET_TOKENS`,
  `Feature.VISION_INPUT`, `Feature.FILE_INPUT` True.
- Probe extensions: `probe_thinking.py`, `probe_vision.py` get
  `gemini` branches.

### Iteration D — Function tools

~100 LOC. Phase-3 equivalent.

- Translate `FunctionTool` → `types.FunctionDeclaration` →
  `types.Tool`. Disable auto-calling
  (`AutomaticFunctionCallingConfig(disable=True)`) so airframe
  owns the round-trip — matches `OpenAICompatibleSession`'s shape.
- Tool loop in `_do_execute` / `stream()`: parse
  `response.function_calls`, invoke each handler, append the
  result as a `types.Part.from_function_response`, re-call. Capped
  at `MAX_TOOL_ITERATIONS` (reuse the OpenAI-compat constant).
- `ToolCallStart` / `ToolCallResult` events on `stream()`.
- Flip `Feature.TOOLS_FUNCTION` True.
- Permission callback wiring (Iteration B+? — see below): the
  airframe-owned round-trip means `PermissionCallback` fires
  naturally around each `_invoke_tool` call, same shape as
  OpenAI-compat. **Flip `PERMISSION_CALLBACK=True` here, not in a
  separate iteration** — the round-trip already happens.

**MCP non-goal.** `google-genai` has no MCP-server kwarg today.
Flag `TOOLS_MCP_STDIO/HTTP/SSE` all False permanently for this
adapter; document the decline message pointing at the future
`GoogleAdkRuntime` (which would carry MCP via `McpToolset`).

### Iteration E — Hooks + budget

~100 LOC. Phase-5 equivalent.

- `EMITTABLE_HOOK_KINDS` ClassVar — 6 kinds (matches OpenAI-compat
  / Codex): `session_start`, `session_end`, `user_prompt_submit`,
  `pre_tool_use`, `post_tool_use`, `tool_failure`.
- `_on_event` plumbing on `GeminiSession` synthesised from the
  client-side tool loop, identical pattern to
  `OpenAICompatibleSession`.
- Budget-cap enforcement via the shared
  `_enforce_budget_pre_turn()` helper.
- In-tree `_GEMINI_PRICING` table for common models
  (`gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-1.5-pro`,
  `gemini-1.5-flash`, etc.) — `cost_usd` computed at turn end and
  fed into `CostRecord`.
- Flip `LIFECYCLE_HOOKS`, `BUDGET_USD_CAP`, `BUDGET_TURN_CAP` True.

### Iteration F — Wrap-up

~50 LOC + docs.

- `GeminiOptions` dataclass (per the surface above) wired through
  with `_check_provider_options`.
- Conformance contract suite green; integration test wrapper at
  `tests/test_gemini_integration.py`.
- Per-adapter docs page: `docs/adapters/gemini.md` covering install
  extra, auth chain (cross-link to `docs/auth.md`), supported
  features, `GeminiOptions` reference, model IDs, structured
  output mechanism, vendor quirks, native escape hatches.
- `docs/auth.md` gets a new section.
- `docs/capabilities.md` matrix gains a column.
- README "Supported providers" table + capability matrix updated.
- CHANGELOG entry.

## Risks and decisions to flag during execution

1. **`SESSION_RESUME=False` is the awkward miss.** Consumers
   expecting to resume a Gemini chat across processes need to
   reconstruct history client-side. Document loudly. If the
   preview Interactions API graduates, revisit.
2. **No MCP.** The decline message must point at the future
   `GoogleAdkRuntime` or at running MCP servers as airframe
   `FunctionTool` wrappers (since `google-genai` does expose
   function-calling natively). Don't promise; document the
   workaround.
3. **Pricing-table drift.** Gemini's pricing has changed multiple
   times in 2025. Keep `_GEMINI_PRICING` close to a single
   `_METADATA` dict (Codex pattern) so updates are one-PR
   churn. Consider a `pricing` module split if a fourth adapter
   needs a pricing table.
4. **Thinking-budget API churn.** Gemini 2.x uses
   `thinking_budget` (token count); Gemini 3+ uses
   `thinking_level` (enum). Adapter needs to dispatch on model
   version. Test both branches with mocked SDKs.
5. **Safety-settings shape.** `safety_settings` in `GeminiOptions`
   is `tuple[Any, ...]` initially — Gemini's `HarmCategory` /
   `HarmBlockThreshold` enums are vendor types we don't want to
   re-export. Document with a code example in
   `docs/adapters/gemini.md`.
6. **Vertex AI mode** — accepted via env-var switch but largely
   untested in v1 of this adapter. Integration tests skip Vertex
   mode unless `GOOGLE_GENAI_USE_VERTEXAI=true` plus
   `GOOGLE_CLOUD_PROJECT` are both set.

## Definition of done

- All six iterations merged.
- Conformance contract suite green against a mocked client.
- `examples/probe_gemini.py` runs end-to-end against live
  `GEMINI_API_KEY`.
- `examples/probe_supports.py --provider gemini` shows the
  expected feature matrix.
- `docs/adapters/gemini.md` published.
- `docs/auth.md` and `docs/capabilities.md` updated.
- README capability matrix updated.
- `airframe-agents[gemini]` installs cleanly and `list_providers()`
  surfaces `"gemini"`.

## When to start

Signal-gated. Two triggers would make this Phase-6-priority work:

1. **A Maverick-side consumer asks for Gemini access.** Concrete
   user is the right gate — the same discipline that's kept the
   ProviderOptions namespaces honest.
2. **A Gemini-only capability becomes load-bearing.** Multimodal
   grounding, citation search, or the Interactions API graduating
   to GA would each be reasons to prioritise.

Until then, the OpenRouter workaround via `OpenCodeZenRuntime`
covers the "I just want to call Gemini" case at the cost of
Gemini-specific features.
