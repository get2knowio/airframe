# Airframe architecture

## The problem

Modern Python agent code wants to:

1. Send a prompt and a Pydantic schema; get back a validated object.
2. Track cost telemetry (tokens, USD) per call.
3. Get useful information back when something fails (was it auth?
   rate limit? capability gap? schema mismatch?).
4. Switch vendors without rewriting agent logic.

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

Airframe declares one protocol — `AgentRuntime` — with four methods.
Every vendor lives behind it.

```
┌─────────────────────────────────────────────────────────────────┐
│                       your agent code                            │
│        (knows only AgentRuntime, RuntimeResult, errors)          │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │     AgentRuntime         │
                │  • execute()             │
                │  • reset()               │
                │  • close()              │
                │  • validate_binding()    │
                └──────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼          ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────┐  ┌─────────────┐
    │ ClaudeCode   │  │ Copilot      │  │ Codex    │  │ OpenCodeZen │
    │ Runtime      │  │ Runtime      │  │ Runtime  │  │ Runtime     │
    └──────────────┘  └──────────────┘  └──────────┘  └─────────────┘
            │                 │                 │            │
            ▼                 ▼                 ▼            ▼
    claude-agent-sdk  github-copilot-sdk  openai-codex   openai (HTTP)
    (subprocess +     (subprocess +       -sdk           https://opencode
     JSON-RPC)         tool reg)          (subprocess     .ai/zen/v1
                                          per turn)
```

Agent code never sees a vendor type. Anything *above* the protocol —
retry, fallback across vendors, conversation memory, multi-agent
orchestration — is the consumer's responsibility; airframe ships the
primitives (typed results, classified errors, binding-validity
predicates) and stays out of the way.

## Why the protocol looks like this

* **`execute` takes a `schema` keyword.** Structured output is
  first-class because that's how typed agent payloads work. Plain
  text is the fallback when `schema=None`.
* **`reset` exists separately from `close`.** Reset is "drop the
  conversation, keep the connection"; close is "drop everything."
  In practice consumers call `reset()` between task boundaries to
  keep the vendor's prompt-cache fresh within a scope while dropping
  it between scopes.
* **No `session_id` in the consumer interface.** Sessions exist
  inside the adapter; the consumer just sends prompts and resets.
  Opaque handles in the consumer interface leak across abstractions.
* **`validate_binding` is non-async and cheap.** It's a "would you
  serve this?" predicate the caller can evaluate before attempting
  the call. Adapters check `provider_id` + maybe a pattern on
  `model_id`; they don't dial home.

## Why errors are vendor-agnostic

Failure modes that *look* the same across vendors should *raise* the
same exception type. That lets consumer code `except` on a neutral
type without needing per-adapter knowledge. The hierarchy carves
the failure modes that have meaningfully different shapes:

| Error | What it means |
| --- | --- |
| `RuntimeAuthError` | Credential is bad / expired / missing. |
| `RuntimeModelNotFoundError` | Server doesn't serve that model on this binding. |
| `RuntimeTransientError` | Call was attempted; server (or network) returned a recoverable failure (5xx, rate limit). |
| `RuntimeStructuredOutputError` | Transport succeeded but the model didn't produce a payload matching the schema. |
| `RuntimeContextOverflowError` | Prompt exceeded the model's context window. |
| `RuntimeProtocolError` | Adapter saw something it can't interpret (adapter / SDK bug). |
| `RuntimeServerStartError` | Adapter couldn't bring its backend up at all. |
| `RuntimeCancelledError` | Caller-initiated abort. |

Adapters classify their vendor's failures into these buckets at the
adapter boundary. What to *do* with each — retry, fall back to a
different binding, surface to the user, escalate to a larger model
— is consumer policy. Airframe doesn't prescribe it.

A consumer that wants a cascade can implement one on top of these
primitives (`validate_binding` to filter, `except` clauses to react
to each error class). Maverick — airframe's first consumer — does
exactly that. But it's a layer above the protocol, not part of it.

## Operational landmines (and what each adapter does about them)

These are the sharp edges the adapters absorb so you don't have to.

### Claude Agent SDK

* Subprocess crash on bad model ID: `claude-agent-sdk` raises
  `CLIConnectionError` if the spawned subprocess exits early. Treated
  as `RuntimeTransientError`.
* Structured output forcing: there's no native JSON-schema mode.
  The adapter registers a `submit_result` MCP tool with the schema
  baked in, then prepends a system-prompt prefix telling the model
  to call it. Works reliably with Claude.
* OAuth refresh: the Claude SDK refreshes on its own when token is
  expired. Adapter sets `CLAUDE_CODE_OAUTH_TOKEN` from env when
  available; otherwise relies on `~/.claude/credentials.json`.

### Copilot SDK

* **Claude served via Copilot Chat Completions is broken** for
  structured output. The model emits markdown-fenced JSON instead
  of calling the tool. `CopilotRuntime.validate_binding` rejects
  every `claude-*` model ID so callers filtering bindings by
  `validate_binding` skip this combination before attempting it.
  Route Claude through `ClaudeCodeRuntime` instead.
* Session lifecycle: `create_session` bakes the model + tools +
  system message in. The adapter caches sessions by
  `(model, system, schema)` and recreates when any of those change.
* Cost telemetry comes via `AssistantUsageData` events on the
  session stream — the adapter subscribes via `session.on()` and
  captures the last usage event for the turn.

### Codex SDK

* Native structured-output mode: the Codex CLI accepts an
  `--output-schema` flag that constrains the final response to a
  JSON Schema. The adapter passes `schema.model_json_schema()` via
  `TurnOptions.outputSchema` and parses `Turn.final_response` as
  JSON. Simplest adapter as a result.
* Auth: codex CLI reads `~/.codex/auth.json` directly when present.
  Adapter falls through to that when no env var is set.
* Subprocess is short-lived (one per turn); no session caching
  needed.

### OpenCode Zen

* Stateless HTTP; the simplest case.
* Some Zen-routed models emit a single-key envelope around the
  structured payload (`{"input": {...}}`, `{"content": "<json>"}`).
  The adapter's `_unwrap_envelope` strips one level of wrapper before
  Pydantic validates. Same quirk Maverick's OpenCode HTTP runtime
  saw before extraction.

## Lifecycle contract

`close()` is idempotent and never raises. Teardown errors get
logged at debug level and swallowed. This matters because
`close()` is called from `finally` blocks and async-context-manager
`__aexit__` — the last thing those should do is shadow the real
exception.

`reset()` is also idempotent. It releases scope-bound state
(sessions, threads) but keeps runtime-wide resources (subprocess
pool, HTTP client, auth tokens). Calling `reset()` twice in a row is
fine and cheap.

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
each adapter's `_PRICING` dict.
