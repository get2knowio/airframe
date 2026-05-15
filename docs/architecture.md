# Airframe architecture

## The problem

Modern Python agent code wants to:

1. Send a prompt and a Pydantic schema; get back a validated object.
2. Track cost telemetry (tokens, USD) per call.
3. Recover gracefully when a provider fails (auth, rate limit,
   transient 5xx, structured-output capability gap).
4. Switch vendors without rewriting agent logic.

Each vendor ships an SDK that solves a *subset* of this and uses
different terminology. Auth chains differ. Error taxonomies differ.
Structured-output forcing differs. Subprocess lifecycle management
differs. Cost reporting differs. Even the obvious things (what to
call "the model identifier") disagree.

Hand-rolling the bridge once per project is tractable. Doing it
across multiple projects, and keeping them in sync as each vendor
SDK evolves, is not.

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
                │  • aclose()              │
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

Agent code never sees a vendor type. The cascade machinery (when one
binding fails, fall over to the next) lives in the *consumer* —
airframe ships the data types and the per-failure-mode error classes,
and the consumer decides retry / cascade policy.

## Why the protocol looks like this

* **`execute` takes a `schema` keyword.** Structured output is
  first-class because that's how typed agent payloads work. Plain
  text is the fallback when `schema=None`.
* **`reset` exists separately from `aclose`.** Reset is "drop the
  conversation, keep the connection"; aclose is "drop everything."
  In practice consumers call `reset()` between bead/task boundaries
  to keep the vendor's prompt-cache fresh within a scope while
  dropping it between scopes.
* **No `session_id` in the consumer interface.** Sessions exist
  inside the adapter; the consumer just sends prompts and resets.
  This was the lesson from the OpenCode HTTP runtime — opaque
  handles in the consumer interface leak across abstractions.
* **`validate_binding` is non-async and cheap.** It's a "would you
  serve this?" predicate, evaluated by cascade machinery before
  attempting the call. Adapters check `provider_id` + maybe a
  pattern on `model_id`; they don't dial home.

## Why errors are vendor-agnostic

The cascade decision is the same regardless of vendor: "is this
recoverable on the same binding? on the next binding? not at all?"
That maps cleanly to seven classes:

| Error | Cascade decision |
| --- | --- |
| `RuntimeAuthError` | next binding (re-auth here won't help) |
| `RuntimeModelNotFoundError` | next binding |
| `RuntimeStructuredOutputError` | next binding (capability gap) |
| `RuntimeTransientError` | same binding, backoff retry |
| `RuntimeContextOverflowError` | **escalate to larger context model** (cascading down is wrong) |
| `RuntimeProtocolError` | surface as bug; don't retry |
| `RuntimeServerStartError` | fatal; surface to caller |

Adapters classify their vendor's failures into these buckets at the
adapter boundary. The consumer's cascade logic doesn't need
adapter-specific knowledge to react correctly.

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
  every `claude-*` model ID so the cascade never tries this path.
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

`aclose()` is idempotent and never raises. Teardown errors get
logged at debug level and swallowed. This matters because
`aclose()` is called from `finally` blocks and async-context-manager
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
