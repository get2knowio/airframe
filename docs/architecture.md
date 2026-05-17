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
its lifetime. Phase 1 introduced the split (shipped in v0.5.0).

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
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼         ▼
    ┌──────────────┐  ┌──────────────────┐  ┌──────────┐  ┌──────────────┐
    │ ClaudeCode   │  │ Copilot          │  │ Codex    │  │ OpenAICompat │
    │ +Session     │  │ +AgentSession    │  │ +Session │  │ +Session     │
    └──────────────┘  └──────────────────┘  └──────────┘  └──────────────┘
            │                   │                   │            │
            ▼                   ▼                   ▼            ▼
    claude-agent-sdk    github-copilot-sdk   openai-codex   openai (HTTP)
    (subprocess +       (subprocess +        -sdk           https://*/v1
     JSON-RPC)           tool reg)            (subprocess   chat.completions
                                              per turn)
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

Phase 1 Iteration G drew a clean line:

| Lives on the runtime | Lives on the session |
|---|---|
| Long-lived vendor handles (`CopilotClient`, `Codex`, `AsyncOpenAI`) | Per-conversation vendor handles (`ClaudeSDKClient`, `CopilotSession`, `Thread`, `messages=[]` buffer) |
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
| `ToolCallStart(tool_name, tool_call_id, arguments_preview)` | The model asked to invoke a tool. Phase 3 wired user-supplied `FunctionTool` round-trips on Claude / Copilot / OpenAI-compat; Phase 4 added external `McpServerRef` routes on Claude / Copilot. |
| `ToolCallResult(tool_call_id, output, is_error)` | A tool invocation completed. Pairs with the matching `ToolCallStart`. |
| `TurnComplete(result)` | Final event in every successful stream. Carries the same `RuntimeResult` `execute()` would have returned. |

**Shape lock (ADR-003).** The variant set and field-by-field shapes
are public surface — once consumer code does `match event:`,
renaming or removing is a major-version break. Adding a new variant
later is safe (consumers branch with a wildcard / `isinstance`).

Adapters declaring `Feature.STREAMING` emit fine-grained deltas as
the vendor produces them. Adapters that don't (none today after
Phase 1) may emit a single `TextDelta` carrying the full response
immediately before `TurnComplete`.

## Capability negotiation

Each adapter declares which protocol features it implements via
`runtime.supports(Feature.X)`. The capability matrix at the end of
Phase 5 (all four phase rollouts complete):

| Feature | Claude Code | Copilot | Codex | OpenAI-compat |
|---|---|---|---|---|
| `STRUCTURED_OUTPUT_JSON_SCHEMA` | ✓ | ✓ | ✓ | ✓ |
| `STRUCTURED_OUTPUT_STRICT` | ✗ | ✗ | ✗ | ✗ |
| `STREAMING` | ✓ | ✓ | ✓ | ✓ |
| `CANCEL` | ✓ | ✓ | ✓ | ✓ |
| `SESSION_RESUME` | ✓ | ✓ | ✓ | ✗ (no server-side session) |
| `REASONING_EFFORT` | ✓ | ✓ | ✓ | ✓ |
| `REASONING_BUDGET_TOKENS` | ✓ | ✗ | ✗ | ✗ |
| `VISION_INPUT` | ✓ | ✓ | ✓ | ✓ |
| `FILE_INPUT` | ✓ | ✓ | ✓ | ✗ (chat-completions has no file slot) |
| `TOOLS_FUNCTION` | ✓ | ✓ | ✗ (no SDK tool API) | ✓ |
| `TOOLS_MCP_STDIO` | ✓ | ✓ | ✗ (CLI-config only) | ✗ (Responses-only) |
| `TOOLS_MCP_HTTP` | ✓ | ✓ | ✗ | ✗ |
| `TOOLS_MCP_SSE` | ✓ | ✗ (declined, switch to http) | ✗ | ✗ |
| `TOOLS_MCP_IN_PROCESS` | ✗ (internal `tools=` plumbing) | ✗ | ✗ | ✗ |
| `PERMISSION_CALLBACK` | ✓ | ✓ | ✓ (session-wide) | ✗ (permanent) |
| `LIFECYCLE_HOOKS` | ✓ (8 kinds) | ✓ (7 kinds, no `rate_limit`) | ✓ (6 kinds, synthesised) | ✓ (6 kinds, synthesised) |
| `BUDGET_USD_CAP` | ✓ | ✓ | ✓ | ✓ |
| `BUDGET_TURN_CAP` | ✓ | ✗ (vendor caps internally) | ✓ | ✓ |
| `SANDBOX` / `SUBAGENTS` | (Phase 6 / signal-gated) | (Phase 6) | (Phase 6) | (Phase 6) |

`supports()` is a cheap static lookup — no network, no SDK version
sniffing. The conformance suite (`airframe.testing.contracts`)
asserts every adapter's `supports(STRUCTURED_OUTPUT_JSON_SCHEMA)` is
True plus the structural contracts for Phase 1–5 features (each
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
events); Copilot drops `rate_limit`; Codex and OpenAI-compat drop
both `pre_compact` and `rate_limit` since they have to synthesise
events from their respective event streams / tool loops. Consumers
writing portable observers can branch defensively on the per-runtime
set.

## Why the protocol looks like this

* **`execute()` takes a `schema` keyword.** Structured output is
  first-class because that's how typed agent payloads work. Plain
  text is the fallback when `schema=None`.
* **`session()` is a factory; `AgentSession` is the canonical
  multi-turn handle.** Every vendor's SDK has a per-conversation
  abstraction (ClaudeSDKClient, CopilotSession, Thread, the
  client-side message buffer); the protocol mirrors that shape so
  the per-adapter glue is thin.
* **`AgentSession.id` is `str | None`.** Adapters with server-side
  sessions populate it (Claude Code, Copilot, Codex); chat-completions
  vendors leave it `None` because there's no vendor-side session to
  refer to. Consumer code branching on `session.id is None` can
  treat that as the "stateless adapter" signal.
* **`reset()` is a no-op after Iteration G.** Per-call sessions own
  their own state; the runtime has nothing scope-bound to drop.
  Kept on the protocol for completeness and back-compat.
* **`validate_binding()` is non-async and cheap.** "Would you serve
  this?" predicate the caller can evaluate before attempting the
  call. Adapters check `provider_id` + maybe a pattern on `model_id`;
  they don't dial home.
* **`unwrap()` lives on both runtime and session.** Runtime-level
  vendor types (`CopilotClient`, `Codex`, `AsyncOpenAI`) reach
  through `runtime.unwrap()`; session-level vendor types
  (`ClaudeSDKClient`, `CopilotSession`, `Thread`) reach through
  `session.unwrap()`.

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
| `UnsupportedFeatureError` | Capability decline — adapter doesn't wire this feature. Phase 1+ companion to `UnsupportedBindingError` that honours the "no silent fallbacks" principle. |

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

### Codex SDK

* Native structured-output mode: the Codex CLI accepts an
  `--output-schema` flag that constrains the final response to a
  JSON Schema. The adapter passes `schema.model_json_schema()` via
  `TurnOptions.outputSchema` and parses `Turn.final_response` as
  JSON. Simplest adapter as a result. Crucially `outputSchema` is
  per-`TurnOptions`, not per-`ThreadOptions`, so schema can vary
  per turn without rebuilding the Thread.
* Streaming: `Thread.run_streamed()` yields typed thread events.
  Per-item tail tracking on `ItemUpdatedEvent` / `ItemCompletedEvent`
  keeps `TextDelta` instances appendable (concatenated deltas
  reconstruct the message text). `AgentMessageItem` → `TextDelta`;
  `ReasoningItem` → `ReasoningDelta`.
* Resume: `Codex.resume_thread(thread_id, ...)` returns a Thread
  with its id pre-populated; subsequent runs continue the prior
  conversation.
* Cancellation: per-turn `AbortController` plumbed into
  `TurnOptions.signal`; `cancel()` calls `controller.abort()`. The
  awaiting turn raises `AbortError` → surfaced as
  `RuntimeCancelledError`.
* Auth: codex CLI reads `~/.codex/auth.json` directly when present.
  Adapter falls through to that when no env var is set.

### OpenAI-compatible HTTP (OpenCode Zen today)

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

## Lifecycle contract

`close()` is idempotent and never raises. Teardown errors get
logged at debug level and swallowed. This matters because
`close()` is called from `finally` blocks and async-context-manager
`__aexit__` paths — the last thing those should do is shadow the
real exception. Both `AgentRuntime.close()` and `AgentSession.close()`
honour this.

`reset()` is also idempotent — and, after Phase 1 Iteration G, a
no-op on every adapter. Per-call sessions own their state; the
runtime has nothing scope-bound to drop. `reset()` is kept on the
protocol for completeness and back-compat with consumers that
called it pre-Iteration-G.

`cancel()` on `AgentSession` is cheap and idempotent: no-op when no
turn is in flight; adapters not declaring `Feature.CANCEL` raise
`UnsupportedFeatureError` when a turn IS in flight. Callers checking
`runtime.supports(Feature.CANCEL)` first never see that error.

## Pricing

Adapters compute `cost_usd` two ways:

1. **Vendor-reported.** Claude Agent SDK exposes `total_cost_usd`
   directly. `ClaudeCodeRuntime` propagates it unchanged.
2. **Computed from token counts.** OpenAI / Codex / Copilot return
   tokens but not cost. Each adapter ships a stub pricing map
   (USD per 1K tokens) and computes `cost_usd = (in_tokens / 1000)
   * in_rate + (out_tokens / 1000) * out_rate`. Models not in the
   map report `cost_usd=None` (tokens are always populated).

The stub maps will migrate to a dedicated `airframe.pricing` module
in a later release. Until then, override per-model rates by editing
each adapter's `_PRICING` / `_METADATA` dict.

## Where to look next

* `dev-docs/implementation-plan.md` — phased rollout (Phase 0
  through 6 / 7+), version targets, gating decisions, and the
  criteria for cutting v1.0. *(Dev-internal; not published to
  PyPI.)*
* `dev-docs/feature-roadmap.md` — per-SDK feature audit and
  prioritised cross-vendor work. *(Dev-internal.)*
* `examples/probe_*.py` — live-vendor probes that exercise each
  adapter end-to-end. Including:
  - `probe_streaming.py` — `session.stream()` against any installed
    adapter.
  - `probe_session_resume.py` — two-turn resume via `session(resume=)`
    on the three SDK adapters.
  - `probe_supports.py` — the `Feature × adapter` capability matrix.
  - `probe_thinking.py` / `probe_vision.py` — Phase 2 inputs &
    reasoning.
  - `probe_tools.py` — Phase 3 `FunctionTool` round-trip.
  - `probe_mcp.py` — Phase 4 `McpServerRef` registration across
    stdio / http / sse transports.
  - `probe_permission.py` — Phase 5 `PermissionCallback` per-call
    interception.
  - `probe_hooks.py` — Phase 5 `HookEvent` observation; prints the
    declared `EMITTABLE_HOOK_KINDS` and the per-kind histogram.
  - `probe_budget.py` — Phase 5 `max_turns=` / `max_budget_usd=`;
    deliberately tiny cap demonstrates `RuntimeBudgetExceededError`.
* `airframe.testing.contracts` — shared structural conformance
  contracts every adapter satisfies (Phase 0 schema round-trip plus
  Phase 1–5 capability-vs-API agreement).
* `airframe.testing.integration` — pytest-marker-gated live-vendor
  probes mirroring the `examples/probe_*.py` set. Run with
  `pytest -m integration` once credentials are configured.
