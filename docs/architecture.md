# Airframe architecture

## The problem

Modern Python agent code wants to:

1. Send a prompt and a Pydantic schema; get back a validated object.
2. Stream the response as the model produces it.
3. Continue a conversation across multiple turns.
4. Track cost telemetry (tokens, USD) per call.
5. Get useful information back when something fails (was it auth?
   rate limit? capability gap? schema mismatch?).
6. Switch vendors without rewriting agent logic.

Each vendor ships an SDK that solves a *subset* of this and uses
different terminology. Auth chains differ. Error taxonomies differ.
Structured-output forcing differs. Subprocess lifecycle management
differs. Cost reporting differs. Even the obvious things (what to
call "the model identifier") disagree.

Hand-rolling the bridge once per project is tractable. Doing it
across multiple projects, and keeping them in sync as each vendor
SDK evolves, is not.

The mental model is JDBC: one driver-manager interface, many vendor
drivers behind it. Airframe is the driver layer. What an application
*does* with that layer — retry policy, vendor fallback, conversation
memory, multi-agent orchestration — is outside the protocol.

## The shape

Airframe is two collaborating protocols. **`AgentRuntime`** is the
runtime-wide entry point — auth, model discovery, capability flags,
one-shot execution, session factory. **`AgentSession`** is the
multi-turn conversation handle that owns vendor session state for
its lifetime.

```
┌────────────────────────────────────────────────────────────────────┐
│                         your agent code                             │
│  (knows only AgentRuntime, AgentSession, RuntimeEvent, errors)      │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │        AgentRuntime           │
                │  • execute()  sugar over      │
                │      session().execute()      │
                │  • session()  factory         │
                │  • list_models()              │
                │  • validate_binding()         │
                │  • supports(Feature)          │
                │  • unwrap(NativeType)         │
                │  • reset() / close()          │
                └───────────────────────────────┘
                                │
                                ▼  session() returns
                ┌───────────────────────────────┐
                │        AgentSession           │
                │  • execute() — one turn       │
                │  • stream() → RuntimeEvent    │
                │  • cancel()                   │
                │  • close()                    │
                │  • unwrap(NativeType)         │
                │  • id: str | None  (resume)   │
                └───────────────────────────────┘
                                │
        ┌─────────────┬─────────┴───┬────────────┬──────────────┐
        ▼             ▼             ▼            ▼              ▼
   ┌──────────┐ ┌────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
   │ Bedrock  │ │ ClaudeCode │ │ Copilot  │ │  Kimi    │ │ OpenAICompat │
   │ +Session │ │ +Session   │ │ +Session │ │ +Session │ │ +Session     │
   └──────────┘ └────────────┘ └──────────┘ └──────────┘ └──────────────┘
        │             │             │            │              │
        ▼             ▼             ▼            ▼              ▼
   aioboto3      claude-agent-  github-      kimi-agent-   openai
   bedrock-      sdk            copilot-sdk  sdk           chat.completions
   runtime       (subprocess    (subprocess  (subprocess   over https
   (Converse      + JSON-RPC)    + tool reg)  + JSONL)      ↳ subclasses:
    API)                                                    OpenCodeGo,
                                                            OpenCodeZen,
                                                            OpenRouter
```

`runtime.execute(prompt, ...)` is now documented sugar for
`runtime.session(system=..., model=...).execute(prompt, ...) + close()` —
single-turn, ephemeral. The session API is the canonical
multi-turn path; `execute()` is the convenience shorthand.

Anything *above* the protocol — retry, fallback across vendors,
conversation memory above the session, multi-agent orchestration —
is the consumer's responsibility; airframe ships the primitives
(typed results, classified errors, streaming events, capability
flags, binding-validity predicates) and stays out of the way.

## Runtime vs session: who owns what

The runtime/session split draws a clean line:

| Lives on the runtime | Lives on the session |
|---|---|
| Long-lived vendor handles (`CopilotClient`, `AsyncOpenAI`, `aioboto3` client) | Per-conversation vendor handles (`ClaudeSDKClient`, `CopilotSession`, `kimi_agent_sdk.Session`, `messages=[]` buffer) |
| Auth resolution (`api_key`, `github_token`) | The system prompt baked into the vendor session |
| Default model | Per-turn schema (when the vendor bakes it at session-creation, the session may reconnect on schema change) |
| Capability declarations (`supports(Feature.X)`) | `id: str | None` — the live vendor session_id for resume |
| Pricing tables, model metadata | `_in_flight` tracking for `cancel()` |
| The `session()` factory | `_abort_controller` / `_active_stream` (the cancellation primitive) |

The split exists because vendor SDKs split this way too:
`CopilotClient.create_session()` returns a `CopilotSession`; the
client lives across sessions but the session is conversation-scoped.
Mapping airframe's runtime/session split onto the vendor's
client/session split keeps the per-adapter glue thin.

`runtime.unwrap(NativeType)` only handles runtime-level types. For
session-level vendor objects, use `session.unwrap(NativeType)` —
the runtime's `unwrap()` raises with a clear redirect message.

## Streaming: the RuntimeEvent taxonomy

`session.stream(prompt)` yields a discriminated union of five event
types from `airframe.events`:

| Event | When it fires |
|---|---|
| `TextDelta(text)` | A chunk of assistant-visible text. Concatenating all `TextDelta.text` for one turn equals the final result's `text`. |
| `ReasoningDelta(text)` | A chunk of hidden reasoning / extended-thinking text. Distinct from `TextDelta` — the model's private chain-of-thought. |
| `ToolCallStart(tool_name, tool_call_id, arguments_preview)` | The model asked to invoke a tool. User-supplied `FunctionTool` round-trips wire on Claude / Copilot / OpenAI-compat; external `McpServerRef` routes wire on Claude / Copilot. |
| `ToolCallResult(tool_call_id, output, is_error)` | A tool invocation completed. Pairs with the matching `ToolCallStart`. |
| `TurnComplete(result)` | Final event in every successful stream. Carries the same `RuntimeResult` `execute()` would have returned. |

**Shape lock.** The variant set and field-by-field shapes
are public surface — once consumer code does `match event:`,
renaming or removing is a major-version break. Adding a new variant
later is safe (consumers branch with a wildcard / `isinstance`).

Adapters declaring `Feature.STREAMING` emit fine-grained deltas as
the vendor produces them. Adapters that don't (none today) may emit
a single `TextDelta` carrying the full response immediately before
`TurnComplete`.

## Capability negotiation

Each adapter declares which protocol features it implements via
`runtime.supports(Feature.X)`. Current capability matrix:

| Feature | Claude Code | Copilot | Kimi | OpenAI-compat |
|---|---|---|---|---|
| `STRUCTURED_OUTPUT_JSON_SCHEMA` | ✓ | ✓ | ◐ scaffolded | ✓ |
| `STRUCTURED_OUTPUT_STRICT` | ✗ | ✗ | ✗ | ✗ |
| `STREAMING` | ✓ | ✓ | ✓ | ✓ |
| `CANCEL` | ✓ | ✓ | ✓ | ✓ |
| `SESSION_RESUME` | ✓ | ✓ | ✓ | ✗ (no server-side session) |
| `REASONING_EFFORT` | ✓ | ✓ | ✓ (boolean) | ✓ |
| `REASONING_BUDGET_TOKENS` | ✓ | ✗ | ✗ | ✗ |
| `VISION_INPUT` | ✓ | ✓ | ✓ | ✓ |
| `FILE_INPUT` | ✓ | ✓ | ✗ (no SDK channel) | ✗ (chat-completions has no file slot) |
| `TOOLS_FUNCTION` | ✓ | ✓ | ✗ (permanent — use MCP) | ✓ |
| `TOOLS_MCP_STDIO` | ✓ | ✓ | ✓ | ✗ (Responses-only) |
| `TOOLS_MCP_HTTP` | ✓ | ✓ | ✓ | ✗ |
| `TOOLS_MCP_SSE` | ✓ | ✗ (declined, switch to http) | ✓ | ✗ |
| `TOOLS_MCP_IN_PROCESS` | ✗ (internal `tools=` plumbing) | ✗ | ✗ (permanent) | ✗ |
| `PERMISSION_CALLBACK` | ✓ | ✓ | ✓ | ✗ (permanent) |
| `LIFECYCLE_HOOKS` | ✓ (8 kinds) | ✓ (7 kinds, no `rate_limit`) | ✓ (7 kinds, no `rate_limit`) | ✓ (6 kinds, synthesised) |
| `BUDGET_USD_CAP` | ✓ | ✓ | ✓ | ✓ |
| `BUDGET_TURN_CAP` | ✓ | ✗ (vendor caps internally) | ✓ | ✓ |
| `SANDBOX` / `SUBAGENTS` | ✗ (planned, signal-gated) | ✗ (planned) | ✗ (planned) | ✗ (planned) |

`supports()` is a cheap static lookup — no network, no SDK version
sniffing. The conformance suite (`airframe.testing.contracts`)
asserts every adapter's `supports(STRUCTURED_OUTPUT_JSON_SCHEMA)` is
True plus the structural contracts for every feature surface (each
declared-True flag must accept the corresponding kwarg without
raising `UnsupportedFeatureError`; each declared-False flag must
raise with the correct `feature=` attribute). Behavioural contracts
that require live credentials live in
`airframe.testing.integration` (pytest-marker-gated; the live
probes under `examples/probe_*.py` are the manual counterpart).

**Per-adapter hook subsets.** Every adapter that declares
`LIFECYCLE_HOOKS=True` also exposes an `EMITTABLE_HOOK_KINDS:
ClassVar[frozenset[str]]` containing the subset of the eight
[`HookEventKind`](../src/airframe/hooks.py) literals it can
honestly emit. Claude has all 8 (native `PreCompact` + `RateLimit`
events); Copilot drops `rate_limit`; Kimi drops `rate_limit`
(Moonshot raises 429s as exceptions rather than wire events);
OpenAI-compat drops both `pre_compact` and `rate_limit` since it
synthesises events from the tool loop. Consumers
writing portable observers can branch defensively on the per-runtime
set.

## Why the protocol looks like this

* **`execute()` takes a `schema` keyword.** Structured output is
  first-class because that's how typed agent payloads work. Plain
  text is the fallback when `schema=None`.
* **`session()` is a factory; `AgentSession` is the canonical
  multi-turn handle.** Every vendor's SDK has a per-conversation
  abstraction (ClaudeSDKClient, CopilotSession,
  `kimi_agent_sdk.Session`, the client-side message buffer); the
  protocol mirrors that shape so the per-adapter glue is thin.
* **`AgentSession.id` is `str | None`.** Adapters with server-side
  sessions populate it (Claude Code, Copilot, Kimi); chat-completions
  vendors leave it `None` because there's no vendor-side session to
  refer to. Consumer code branching on `session.id is None` can
  treat that as the "stateless adapter" signal.
* **`reset()` is a no-op on every adapter today.** Per-call sessions
  own their own state; the runtime has nothing scope-bound to drop.
* **`validate_binding()` is non-async and cheap.** "Would you serve
  this?" predicate the caller can evaluate before attempting the
  call. Adapters check `provider_id` + maybe a pattern on `model_id`;
  they don't dial home.
* **`unwrap()` lives on both runtime and session.** Runtime-level
  vendor types (`CopilotClient`, `AsyncOpenAI`, `aioboto3` client)
  reach through `runtime.unwrap()`; session-level vendor types
  (`ClaudeSDKClient`, `CopilotSession`, `kimi_agent_sdk.Session`)
  reach through `session.unwrap()`.

## Why errors are vendor-agnostic

Failure modes that *look* the same across vendors should *raise* the
same exception type. That lets consumer code `except` on a neutral
type without needing per-adapter knowledge:

| Error | What it means |
| --- | --- |
| `RuntimeAuthError` | Credential is bad / expired / missing. |
| `RuntimeModelNotFoundError` | Server doesn't serve that model on this binding. |
| `RuntimeTransientError` | Call was attempted; server (or network) returned a recoverable failure (5xx, rate limit). |
| `RuntimeStructuredOutputError` | Transport succeeded but the model didn't produce a payload matching the schema. |
| `RuntimeContextOverflowError` | Prompt exceeded the model's context window. |
| `RuntimeProtocolError` | Adapter saw something it can't interpret (adapter / SDK bug). |
| `RuntimeServerStartError` | Adapter couldn't bring its backend up at all. |
| `RuntimeCancelledError` | Caller-initiated abort (`session.cancel()`). |
| `UnsupportedBindingError` | Programming error — adapter doesn't serve this `(provider, model)` pair. |
| `UnsupportedFeatureError` | Capability decline — adapter doesn't wire this feature. Raised at the boundary so the call fails fast rather than silently downgrading. |

Adapters classify their vendor's failures into these buckets at the
adapter boundary. What to *do* with each — retry, fall back to a
different binding, surface to the user, escalate to a larger model
— is consumer policy. Airframe doesn't prescribe it.

## Operational landmines (and what each adapter does about them)

These are the sharp edges the adapters absorb so you don't have to.

### Claude Agent SDK

* Subprocess crash on bad model ID: `claude-agent-sdk` raises
  `CLIConnectionError` if the spawned subprocess exits early. Treated
  as `RuntimeTransientError`.
* Structured output: native via `ClaudeAgentOptions.output_format`
  (`{"type": "json_schema", "schema": ...}`). The CLI enforces the
  schema server-side and the validated payload lands on
  `ResultMessage.structured_output`. No tool-forcing, no MCP shim.
* Streaming: `include_partial_messages=True` on options surfaces raw
  Anthropic stream events (`StreamEvent`); the adapter translates
  `content_block_delta` with `text_delta` → `TextDelta`,
  `thinking_delta` → `ReasoningDelta`. Fallback: `TextBlock` content
  on `AssistantMessage` emits a `TextDelta` when StreamEvents didn't
  deliver text.
* Resume: `ClaudeAgentOptions.resume` accepts a session_id; the SDK
  materialises the prior conversation from its local-disk session
  store on connect. `AgentSession.id` is seeded with the resume id
  and updated from each `ResultMessage.session_id`.
* Cancellation: `ClaudeSDKClient.interrupt()` aborts the in-flight
  CLI turn; the adapter also cancels the wrapping `asyncio.Task`.
* OAuth refresh: the Claude SDK refreshes on its own when the token
  is expired. Adapter sets `ANTHROPIC_API_KEY` per-spawn via
  `ClaudeAgentOptions.env` when an explicit key is supplied.

### Copilot SDK

* **Claude served via Copilot Chat Completions is broken** for
  structured output. The model emits markdown-fenced JSON instead
  of calling the tool. `CopilotRuntime.validate_binding` rejects
  every `claude-*` model ID so callers filtering bindings by
  `validate_binding` skip this combination before attempting it.
  Route Claude through `ClaudeCodeRuntime` instead.
* Structured output: forced `submit_result` tool registered with
  `copilot.define_tool`. The session's system message gets a
  prefix telling the model to call it; the tool handler captures
  the validated Pydantic payload. Schema is bake-time on
  `create_session`, so the session reconnects when the schema
  fingerprint changes.
* Streaming: per-session `session.on(handler)` subscription pushes
  `AssistantMessageDeltaData` → `TextDelta` and
  `AssistantReasoningDeltaData` → `ReasoningDelta` through an
  `asyncio.Queue` the generator drains. Uses
  `loop.call_soon_threadsafe` since the SDK dispatches off its own
  thread.
* Resume: `CopilotClient.resume_session(session_id, ...)` returns a
  fresh `CopilotSession` with prior history loaded.
* Cancellation: `CopilotSession.abort()` aborts the in-flight CLI
  turn.
* Cost telemetry: `AssistantUsageData` events on the session stream.

### Kimi Agent SDK

* Streaming via `Session.prompt()` — an async generator yielding
  `WireMessage` instances (`TextPart`, `ThinkPart`, `ToolCall`,
  `ToolResult`, `ApprovalRequest`, `TokenUsage`, etc.). The adapter
  classifies on `type(wire).__name__` rather than `isinstance` so
  the unit tests can use `sys.modules`-injected fakes (the SDK can't
  be co-installed with `claude-agent-sdk`).
* Resume: `Session.resume(work_dir, session_id)`; multi-turn state
  persists in the SDK's session store.
* Cancellation: `Session.cancel()` sets the SDK's async cancel
  event; the in-flight `prompt()` raises `RunCancelled` → surfaced
  as `RuntimeCancelledError`.
* `thinking` is session-scoped (`Session.create(thinking: bool)`).
  Toggling between turns rebuilds the SDK session and re-resumes by
  id to preserve conversation state.
* Permission via `ApprovalRequest.resolve()` (per-call); `defer`
  collapses to `reject` with feedback because the SDK's approval
  channel is synchronous.
* Auth: SDK reads `KIMI_API_KEY` from env; the adapter mutates
  env for the call duration when an explicit `api_key=` is passed
  and restores it on `close()`.

### AWS Bedrock (Converse API)

* Stateless from the client's perspective; `BedrockSession` keeps a
  client-side `messages=[]` buffer like the OpenAI-compat family
  does, even though the wire format is Converse rather than Chat
  Completions.
* **Region is mandatory.** Bedrock is region-pinned and per-region
  model catalogs diverge. If `AWS_REGION` doesn't resolve and boto3
  falls through to `us-east-1`, a model that works fine in
  `us-west-2` returns `ValidationException` and looks like
  model-not-found. The adapter surfaces an unresolved region as a
  first-class `RuntimeAuthError` at the first network call rather
  than letting boto3's default cascade into a confusing downstream
  error.
* **Inference-profile and PT ARN model IDs.** Some models *require*
  the prefixed form (`us.anthropic.claude-3-5-sonnet-20241022-v2:0`)
  or a full provisioned-throughput ARN. `validate_binding` accepts
  both shapes; `_BEDROCK_METADATA` lookups fall through to
  `cost_usd=None` on prefixed variants rather than guessing pricing.
  `BedrockOptions.inference_profile_arn` replaces the call-time
  `modelId` when set, so consumers route through a PT or cross-region
  inference profile without rebuilding the session.
* **Guardrail interventions are silent without help.** When a
  Bedrock Guardrails policy fires, Converse returns
  `stopReason == "guardrail_intervened"` with an empty or truncated
  `output`. Pydantic validation would then fail with a confusing
  message. The adapter's `_check_guardrail_intervention()` raises
  `RuntimeProtocolError` with a clear message instead, on both
  `execute()` and `stream()` paths.
* **Redacted reasoning blocks.** Recent Claude versions on Bedrock
  emit `reasoningContent` chunks carrying only a `redactedContent`
  field (safety-redacted thinking — no `text` available). The
  streaming parser skips those without crashing rather than emitting
  empty `ReasoningDelta` events.
* Structured output: forced `submit_result` tool in Converse's
  `toolConfig` slot — same pattern Copilot uses. User-defined
  `FunctionTool`s coexist with the forced tool. Schema is
  fingerprinted so successive turns with the same schema reuse the
  built `toolConfig`.
* Thinking is Anthropic-on-Bedrock only. `thinking=` translates to
  `additionalModelRequestFields={"thinking": {"type": "enabled",
  "budget_tokens": N}}` for Anthropic model IDs and is silently
  omitted (with a debug log) for other vendors. `REASONING_EFFORT` /
  `REASONING_BUDGET_TOKENS` report True via `supports()` but are
  model-gated in practice.
* `additionalModelRequestFields` is a vendor-specific footgun.
  `BedrockOptions.additional_model_fields` passes a dict through
  unchecked. Wrong field for the wrong vendor = silent ignore.
  Documented as an escape hatch where field validity is the caller's
  problem.
* Auth: standard boto3 chain (explicit kwargs → env →
  `AWS_PROFILE` → IAM instance / ECS / Lambda / IRSA roles); region
  resolves on a separate chain (`region_name=` → `AWS_REGION` /
  `AWS_DEFAULT_REGION` → resolved profile's config).

### OpenAI-compatible HTTP (OpenCode Go, OpenCode Zen, OpenRouter)

* Stateless HTTP; the simplest transport. The `OpenAICompatibleSession`
  maintains a client-side `messages=[]` buffer because chat-completions
  has no server-side session — each request resends the full history.
* Structured output: native via
  `response_format={"type": "json_schema", "json_schema": {...}}`.
* Streaming: `stream=True` on `chat.completions.create()` plus
  `stream_options={"include_usage": True}` so the trailing
  `TurnComplete` carries a populated `CostRecord`.
* Cancellation: `execute()` via `asyncio.Task.cancel()` →
  `RuntimeCancelledError`; `stream()` via flag + `AsyncStream.close()`
  → generator raises on next yield boundary.
* No resume: chat-completions has no server-side session.
  `session(resume=...)` raises `UnsupportedFeatureError`. A future
  `OpenAIResponsesRuntime` (separate from this family) could wire it.
* Some Zen-routed models emit a single-key envelope around the
  structured payload (`{"input": {...}}`, `{"content": "<json>"}`).
  The adapter's `_unwrap_envelope` strips one level of wrapper before
  Pydantic validates.

### OpenRouter (within the OpenAI-compatible family)

Same base as the other `OpenAICompatibleRuntime` subclasses; what
follows is the router-specific surface on top.

* **Per-model feature heterogeneity is the headline landmine.**
  OpenRouter is a router, not a vendor. What works in any given call
  depends on which underlying model the request gets routed to —
  function tools, strict JSON schema, vision are all wire-compatible
  but unevenly supported across the catalog. `runtime.supports()`
  declares the *adapter*'s surface; silent degradation per model is
  the failure mode rather than a hard refusal. If your application
  hard-depends on a feature for a specific model, verify against that
  model directly.
* **Model IDs carry vendor prefixes.** Strings are `<vendor>/<model>`
  (`anthropic/claude-3.5-sonnet`, `meta-llama/llama-3.1-70b-instruct`,
  `google/gemini-pro-1.5`, etc.). The adapter passes the string
  through unchanged; copying a bare model name from elsewhere yields
  a 404 from OpenRouter.
* **Curated pricing only.** Around ten model IDs have static
  `ModelMeta` entries; everything else returns `cost_usd=None` from
  `list_models()`. OpenRouter's per-model pricing shifts as upstream
  vendors adjust, so the adapter doesn't guess. Compare to OpenCode
  Go (flat-fee → always `0.0`) and OpenCode Zen / direct OpenAI-compat
  (per-token rates known at the gateway).
* **No on-disk auth fallback.** Unlike the OpenCode adapters which
  fall through to `~/.local/share/opencode/auth.json`, OpenRouter
  reads `OPENROUTER_API_KEY` env or an explicit `api_key=` only.
  Missing either raises `RuntimeAuthError` pointing at
  `https://openrouter.ai/keys`.

## Lifecycle contract

`close()` is idempotent and never raises. Teardown errors get
logged at debug level and swallowed. This matters because
`close()` is called from `finally` blocks and async-context-manager
`__aexit__` paths — the last thing those should do is shadow the
real exception. Both `AgentRuntime.close()` and `AgentSession.close()`
honour this.

`reset()` is also idempotent — and a no-op on every adapter today.
Per-call sessions own their state; the runtime has nothing
scope-bound to drop.

`cancel()` on `AgentSession` is cheap and idempotent: no-op when no
turn is in flight; adapters not declaring `Feature.CANCEL` raise
`UnsupportedFeatureError` when a turn IS in flight. Callers checking
`runtime.supports(Feature.CANCEL)` first never see that error.

## Pricing

Adapters compute `cost_usd` two ways:

1. **Vendor-reported.** Claude Agent SDK exposes `total_cost_usd`
   directly. `ClaudeCodeRuntime` propagates it unchanged.
2. **Computed from token counts.** OpenAI-compat / Copilot / Kimi
   return tokens but not cost. Each adapter ships a stub pricing
   map (USD per 1K tokens) and computes `cost_usd = (in_tokens /
   1000) * in_rate + (out_tokens / 1000) * out_rate`. Models not
   in the map report `cost_usd=None` (tokens are always populated).

The stub maps will migrate to a dedicated `airframe.pricing` module
in a later release. Until then, override per-model rates by editing
each adapter's `_PRICING` / `_METADATA` dict.

## Where to look next

* `dev-docs/implementation-plan.md` — iteration plan, version
  targets, gating decisions, and the criteria for cutting v1.0.
  *(Dev-internal; not published to PyPI.)*
* `dev-docs/feature-roadmap.md` — per-SDK feature audit and
  prioritised cross-vendor work. *(Dev-internal.)*
* `examples/probe_*.py` — live-vendor probes that exercise each
  adapter end-to-end. Including:
  - `probe_streaming.py` — `session.stream()` against any installed
    adapter.
  - `probe_session_resume.py` — two-turn resume via `session(resume=)`
    on the three SDK adapters.
  - `probe_supports.py` — the `Feature × adapter` capability matrix.
  - `probe_thinking.py` / `probe_vision.py` — reasoning controls and
    polymorphic image/file prompts.
  - `probe_tools.py` — `FunctionTool` round-trip.
  - `probe_mcp.py` — `McpServerRef` registration across stdio / http
    / sse transports.
  - `probe_permission.py` — `PermissionCallback` per-call
    interception.
  - `probe_hooks.py` — `HookEvent` observation; prints the
    declared `EMITTABLE_HOOK_KINDS` and the per-kind histogram.
  - `probe_budget.py` — `max_turns=` / `max_budget_usd=`;
    deliberately tiny cap demonstrates `RuntimeBudgetExceededError`.
* `airframe.testing.contracts` — shared structural conformance
  contracts every adapter satisfies (schema round-trip plus the
  full capability-vs-API agreement).
* `airframe.testing.integration` — pytest-marker-gated live-vendor
  probes mirroring the `examples/probe_*.py` set. Run with
  `pytest -m integration` once credentials are configured.
