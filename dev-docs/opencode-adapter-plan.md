# `OpenCodeServerRuntime` adapter plan

Companion to [implementation-plan.md](./implementation-plan.md). Specs
a new built-in adapter wrapping the **OpenCode agent HTTP server**
(`sst/opencode`) — the bespoke Effect-based REST + SSE server that
`opencode serve` (and the OpenCode TUI) runs locally. This is *not*
the OpenCode Zen / Go OpenAI-compatible gateway; the two existing
`OpenCodeZenRuntime` / `OpenCodeGoRuntime` adapters share a brand but
front a completely different API.

The plan mirrors the ABCDEF iteration cadence used by
[`bedrock-adapter-plan.md`](./bedrock-adapter-plan.md) and
[`google-genai-adapter-plan.md`](./google-genai-adapter-plan.md):
scaffold → execute/stream/cancel → polymorphic prompt + reasoning →
tools + permission → hooks + budget → wrap-up. Targeting ~700–900
LOC over 6 iterations, each independently mergeable.

## Motivation

Airframe ships four agent-server-class adapters today:
`ClaudeCodeRuntime`, `CopilotRuntime`, `CodexRuntime`, and the planned
`BedrockRuntime`. Every one of them is **bound to a single model
house** — Anthropic, OpenAI, GitHub, AWS-curated catalog respectively.
Consumers who want an agent experience that runs **open-weight models**
(Llama, Qwen, DeepSeek, GLM, MiMo, Kimi, Mistral) have no path through
airframe today that gets them more than chat-completions-shaped calls.
The OpenAI-compat adapters (`OpenCodeZen`, `OpenCodeGo`, `OpenRouter`)
deliver tokens but no agent — no server-managed sessions, no built-in
tools, no MCP, no permission gating, no lifecycle observability.

OpenCode is the missing fifth bucket: **a model-agnostic agent server
that fronts any backend** (Anthropic, OpenAI, OpenRouter, Ollama,
vLLM, llama.cpp, Together, Groq, custom OpenAI-compat). Adopting it:

- **Unlocks open-weights agentics.** First adapter in the lineup whose
  value prop is "agent capabilities decoupled from a specific model
  house." That gap is otherwise unreachable through airframe.
- **Mirrors the JDBC-driver pattern.** Same way `BedrockRuntime`
  unlocks a multi-vendor catalog behind one auth scheme, OpenCode
  unlocks a multi-vendor catalog behind one agent loop. The
  feature surface (sessions, MCP, permission, hooks, SSE streaming)
  is genuinely peer-class with Claude / Copilot / Codex / Bedrock —
  this is not a gateway adapter wearing agent costume.
- **Matches airframe's narrow-protocol-over-vendor-SDK shape.** Wraps
  the official `opencode-ai` Stainless-generated Python SDK (HTTP
  client; no subprocess management). Lands closer to `BedrockRuntime`
  in size and complexity than to `ClaudeCodeRuntime`.

## Non-goals

- **No bundled `opencode` install / spawn.** Unlike
  `ClaudeCodeRuntime` (which subprocess-spawns the Claude Agent SDK
  per session), this adapter assumes the user has already started
  `opencode serve` themselves — either headlessly or implicitly via
  the TUI. Reasons: server state lives on disk in a SQLite store
  shared with the user's TUI session; a library-spawned competing
  server would race on it. An opt-in `autostart=True` constructor
  flag is plausible future work but ships False initially.
- **No conflation with `OpenCodeZenRuntime` / `OpenCodeGoRuntime`.**
  Those two adapters target `https://opencode.ai/zen/v1` and
  `https://opencode.ai/zen/go/v1` — the OpenCode Zen / Go billed
  gateways that speak OpenAI Chat Completions. They share a brand
  with this adapter and nothing else; provider ID, base class,
  wire format, and feature surface all differ. The bespoke server
  even exposes a `/zen/v1/chat/completions` pass-through endpoint;
  that pass-through is *out of scope* for this adapter (callers can
  point `OpenCodeZenRuntime` at the local server if they want it).
- **No Bedrock-Agents-style sibling reservation.** OpenCode does not
  ship a managed cloud variant. The `"opencode"` provider ID covers
  the entire surface; no `"opencode-cloud"` / `"opencode-managed"`
  reservation needed.
- **No bring-your-own model provisioning.** Configuring which model
  backends OpenCode itself talks to (Anthropic API key, OpenRouter
  key, Ollama URL, etc.) is a server-side concern handled via
  `opencode auth login` and `opencode.json`. The adapter consumes
  whatever the server has configured; it doesn't reach into that.
- **No filesystem proxying.** OpenCode runs `bash`/`read`/`write`/
  `edit` server-side; those tools see the **server's** filesystem,
  not the adapter's. Documenting this is the adapter's job; mapping
  it isn't.

## Adapter shape

`OpenCodeServerRuntime(AgentRuntime)` — a direct subclass of
`AgentRuntime`, not `OpenAICompatibleRuntime`. Reasons:

1. Wire format isn't OpenAI Chat Completions — it's the OpenCode
   REST API: `POST /session`, `POST /session/:id/message`,
   `GET /event` (SSE), `DELETE /session/:id`, etc.
2. **Sessions live server-side.** OpenCode owns conversation state in
   SQLite; every turn references a session_id. The client-side
   `messages=[]` buffer pattern that every OpenAI-compat adapter
   uses is exactly the wrong abstraction here.
3. Auth chain is HTTP Basic over loopback (or a configured remote),
   not API-key in an `Authorization: Bearer` header.

`OpenCodeServerSession(AgentSession)` is the primary surface. It owns
a server-issued `session_id` (`AgentSession.id` is populated, unlike
every OpenAI-compat adapter and unlike `BedrockSession`) plus a
long-lived SSE subscription on `GET /event` used for streaming,
permission requests, and lifecycle observability. `reset()` is the
first adapter where the call does real work (`DELETE /session/:id`)
rather than being a buffer-clear no-op.

```
src/airframe/adapters/opencode_server.py  ~700 LOC (target)
tests/test_opencode_server.py             ~400 LOC mirroring test_claude_code.py
tests/test_opencode_server_session.py     ~200 LOC session-class behaviour
tests/test_opencode_server_conformance.py airframe.testing.contracts driver
tests/test_opencode_server_integration.py behavioural, pytest-marker gated
docs/adapters/opencode-server.md          ~200 lines
examples/probe_opencode_server.py         single-call execute(schema=) probe
```

| Class attribute | Value |
|---|---|
| `PROVIDER_ID` | `"opencode"` |
| `REQUIRES_PACKAGE` | `"opencode-ai"` |
| `EXTRA_NAME` | `"opencode"` |
| `label` | `"opencode_server"` |

**Provider ID reservation note.** `"opencode"` is **not yet** in the
reserved-IDs list in `CLAUDE.md`. The existing reservations
(`"opencode-zen"`, `"opencode-go"`) are the two billed-gateway
adapters; this lands as their bespoke-agent sibling. The Iteration A
PR adds `"opencode"` to the reserved list.

## SDK surface this adapter wraps

`opencode-ai` (currently `0.1.0a23` — Stainless-generated from the
server's live OpenAPI 3.1 spec at `http://localhost:4096/doc`).
Pre-1.0; breaking changes expected. Pin tightly. Relevant subset:

- **Client.** `opencode_ai.AsyncOpencode(base_url=..., username=...,
  password=...)`. Sync `Opencode` exists; airframe wraps async.
- **Sessions.** `client.session.create(...)`, `client.session.list()`,
  `client.session.get(session_id)`, `client.session.delete(session_id)`,
  `client.session.abort(session_id)`, `client.session.fork(...)`.
- **Messages.** `client.session.message(session_id, ...)` — sync
  request/response (full assistant turn).
  `client.session.prompt_async(session_id, ...)` — fire-and-forget
  (204); reply arrives on the SSE bus.
  `client.session.messages.list(session_id)` — full history.
- **Events.** `client.event.stream()` and `client.event.global_stream()`
  return async iterators of typed events: `server.connected`,
  `message.part.delta`, `message.part.updated`, `message.part.removed`,
  `session.idle`, `session.error`, `permission.asked`,
  `permission.replied`, `question.asked`, lifecycle events.
- **Providers / models.** `client.provider.list()` enumerates the
  server-configured upstream providers and their model catalogs.
- **Permission.** `client.permission.reply(request_id, decision=...)`
  responds to a `permission.asked` event.
- **MCP.** `client.mcp.list()`, `client.mcp.add(...)` — server-side
  MCP server registration.
- **Tools (experimental).** `client.experimental.tool.ids()`,
  `client.experimental.tool.get(...)`. Namespace is explicitly
  unstable; airframe does **not** depend on it for v1.

OpenCode's tool execution is **server-side**. The model emits a tool
call; the server resolves it (bash, file edit, MCP call, etc.); the
client only observes the result on SSE. There is **no client-callback
transport** for caller-defined Python functions, which is the
defining friction this adapter has to confront — see Iteration D.

## Feature support matrix (target)

| Feature | Status | Notes |
|---|---|---|
| `STRUCTURED_OUTPUT_JSON_SCHEMA` | ◐ best-effort | OpenCode has no server-level `response_format=json_schema` knob. Implementation: airframe drives forced-tool-call (`submit_result` tool) by registering a transient MCP server with that single tool — same pattern as `CopilotRuntime`'s structured output, but realised over `POST /mcp` rather than as an inline `tools=` slot. Phase 0 flips this True with a documented caveat: not every backend model OpenCode fronts honours forced tool-use cleanly. |
| `STRUCTURED_OUTPUT_STRICT` | ✗ | No strict-mode equivalent on OpenCode itself; whether the underlying model is strict-capable is invisible from the agent server. |
| `STREAMING` | ✓ | `GET /event` SSE → translate `message.part.delta` (text) and `message.part.delta` (reasoning) → `TextDelta` / `ReasoningDelta`. Terminal `session.idle` → `TurnComplete`. |
| `CANCEL` | ✓ | `POST /session/:id/abort`. The adapter also drops the SSE iterator on cancel to surface `RuntimeCancelledError` synchronously. |
| `SESSION_RESUME` | ✓ | **Natively first-class.** Pass an existing `session_id` to `runtime.session(resume=...)` and skip the `POST /session` step. Server retains full history. Among the cleanest `SESSION_RESUME` stories in the lineup. |
| `REASONING_EFFORT` | ◐ pass-through | The agent server doesn't normalise reasoning across backends. Adapter accepts `"low"`/`"medium"`/`"high"` and forwards via the request's `provider_options` dict to the model layer; non-reasoning models silently ignore. Capability declared True with a per-model warning surfaced via debug logs. |
| `REASONING_BUDGET_TOKENS` | ◐ pass-through | Same channel — forwarded to the underlying model. Honoured on Anthropic / Gemini backends; ignored elsewhere. |
| `VISION_INPUT` | ✓ | `message.part` accepts image parts. Capability declared per-model; for fully accurate gating defer to `client.provider.list()` model metadata when available. |
| `FILE_INPUT` | ✓ | Same parts API supports file attachments. |
| `TOOLS_FUNCTION` | ✗ initially → ✓ Iteration D-extension | OpenCode has **no client-side function-call transport**. To honour `tools=[FunctionTool(...)]`, airframe spins up an in-process MCP server hosting the caller-defined Python handlers, registers it with the OpenCode server via `POST /mcp`, and tears it down at session close. Lands in Iteration D; if scope creep threatens, the iteration ships with `TOOLS_FUNCTION=False` and the MCP-wrapping work moves to Iteration G. |
| `TOOLS_MCP_STDIO` | ✓ | `POST /mcp` accepts `{"type": "local", "command": [...], "env": {...}}`. |
| `TOOLS_MCP_HTTP` | ✓ | `POST /mcp` accepts `{"type": "remote", "url": "https://...", "headers": {...}}`. |
| `TOOLS_MCP_SSE` | ✓ | Same `"remote"` shape with the SSE URL. |
| `TOOLS_MCP_IN_PROCESS` | ✗ | OpenCode's MCP slots are stdio / HTTP / SSE only — there's no in-process Python registration path (unlike Claude's `create_sdk_mcp_server`). Permanent decline. |
| `PERMISSION_CALLBACK` | ✓ | **Natively first-class via the SSE bus.** `permission.asked` events dispatch to the registered `PermissionCallback`; reply via `POST /permission/:id/reply`. Among the cleanest implementations in the lineup. |
| `LIFECYCLE_HOOKS` | ✓ (7 kinds) | The SSE bus emits all the hook-relevant events natively: `session_start`, `session_end`, `user_prompt_submit` (translated from `message.part` user content), `pre_tool_use` / `post_tool_use` / `tool_failure` (from `message.part` tool blocks), and `pre_compact` (from `session.compact` events). No `rate_limit` (OpenCode propagates underlying provider 429s as `session.error`). |
| `BUDGET_USD_CAP` | ✓ | Server reports cost on each `message.part.updated` event (when the underlying provider provides it). Adapter accumulates client-side against `BudgetCap.usd`. |
| `BUDGET_TURN_CAP` | ✓ | Client-side per-session counter, same as siblings. |
| `SANDBOX` | ✗ | Deferred to OpenCode's own server-side sandboxing. Airframe doesn't surface a kwarg. |
| `SUBAGENTS` | ✗ | OpenCode supports parent-session/child-session linking server-side but the wire shape is unstable; defer to Phase 6+ if demand materialises. |

`PERMISSION_CALLBACK=True` and `LIFECYCLE_HOOKS=True` are the
adapter's standout features in the lineup — both fall out
near-for-free from the existing SSE bus, where every other adapter
has to synthesise them from a client-side tool loop.

## Auth chain

OpenCode uses HTTP Basic auth. Order resolved at `__init__()` time
with one server-reachability probe deferred to first call:

1. **Explicit `username=` / `password=` constructor args.** Highest
   precedence. Forwarded to `AsyncOpencode`.
2. **`OPENCODE_SERVER_USERNAME` + `OPENCODE_SERVER_PASSWORD` env vars.**
   Standard. `OPENCODE_SERVER_USERNAME` defaults to `"opencode"` if
   only the password is set — matches the server's documented default
   username.
3. **No auth** — only honoured when the resolved `base_url` is a
   loopback address (`127.0.0.1`, `localhost`, `::1`). Non-loopback
   URLs without basic-auth credentials raise `RuntimeAuthError` at
   `__init__()` with a clear message: "OpenCode server at <host> is
   not loopback; set OPENCODE_SERVER_PASSWORD (and OPENCODE_SERVER_USERNAME
   if non-default)." This guardrail prevents accidentally publishing
   an unauthenticated remote-bash endpoint.

**Base URL resolution.**

1. Explicit `base_url=` constructor arg.
2. `OPENCODE_SERVER_URL` env var.
3. Default: `http://127.0.0.1:4096`.

`validate_binding()` stays cheap (static — accepts any non-empty
`model_id` when `provider_id == "opencode"`, since the available
models depend entirely on what the server has been configured to
expose upstream).

`list_models()` is the first method that probes reachability. It
hits `GET /provider` (returns the server's configured provider list
with model metadata) and classifies:

- `httpx.ConnectError` → `RuntimeServerStartError` with hint to run
  `opencode serve`.
- `401` → `RuntimeAuthError` referencing `OPENCODE_SERVER_PASSWORD`.
- Otherwise → propagate per the standard exception taxonomy.

## `OpenCodeServerOptions` (provider-options namespace)

```python
@dataclass(frozen=True, slots=True)
class OpenCodeServerOptions:
    # Per-call provider routing — picks which upstream model
    # backend OpenCode uses for this turn.
    provider_id: str | None = None
        # e.g. "anthropic", "openai", "openrouter", "ollama". When
        # None, OpenCode's own routing config decides.

    # Allow-/deny-listing built-in tools at session creation time.
    # Maps to POST /session's tool-filter knobs.
    available_tools: tuple[str, ...] | None = None
    excluded_tools: tuple[str, ...] | None = None

    # Working directory the server-side tools operate in. Server
    # interprets relative to its own filesystem, NOT the adapter's.
    working_directory: str | None = None

    # Permission posture forwarded to POST /session — coarse-grained
    # default that the PermissionCallback overrides per-call.
    permission_mode: str | None = None
        # "default" | "accept_edits" | "bypass" — see opencode docs.

    # MCP server registrations applied at session start.
    # Translated to POST /mcp calls. Use airframe's McpServerRef
    # types in Phase 4-aware code; here as a provider-specific
    # additional channel for non-portable knobs.
    additional_mcp_servers: tuple[Any, ...] = ()

    # Forwarded into the request as a typed pass-through for
    # vendor-specific knobs airframe doesn't have first-class
    # support for (provider-specific reasoning depth, sampling
    # overrides, etc.).
    additional_request_fields: dict[str, Any] | None = None
```

Same tagged-union discipline as the existing namespaces — mismatched
type raises `UnsupportedFeatureError` at `session(provider_options=)`.

## Iteration breakdown

### Iteration A — Protocol scaffolding (no behaviour)

~200 LOC. Lands the adapter's shape without wiring substantive
features.

- `src/airframe/adapters/opencode_server.py` —
  `OpenCodeServerRuntime(AgentRuntime)` with ClassVars, lazy
  `opencode_ai` import (deferred to first method call),
  `validate_binding` (accepts any non-empty `model_id` when
  `provider_id == "opencode"`).
- `_resolve_auth()` chain implementing the three-step auth above,
  including the loopback-only guardrail.
- `_resolve_base_url()` for the three-step base URL chain.
- Empty `SUPPORTED_FEATURES = frozenset()` initially; flip flags on
  as each iteration wires them.
- `unwrap(AsyncOpencode)` runtime escape hatch (also `Opencode` for
  callers using the sync client elsewhere).
- `close()` idempotent + never raises (closes the `AsyncOpencode`
  HTTP client; server keeps running independently).
- `reset()` no-op at the runtime level (sessions own the per-session
  state and have their own `DELETE`).
- `list_models()` hits `client.provider.list()`, flattens into
  `ModelInfo`s. Reachability errors classified per the chain above.
- Add `"opencode"` to the reserved-IDs paragraph in `CLAUDE.md`.
- Discovery registration in `discovery.py` + top-level export in
  `airframe/__init__.py`.
- Conformance contracts pass against a mocked `AsyncOpencode`.

**Stopping point.** `import airframe; airframe.list_providers()`
includes `"opencode"` when the extra is installed. Discovery and
capability predicates work; no live behaviour wired.

### Iteration B — Execute + streaming + cancellation + session-resume

~200 LOC. The Phase-1-equivalent slice — `session` factory,
`AgentSession` subclass, `execute()`, `stream()`, `cancel()`, plus
the natively-supported `resume=`.

- `OpenCodeServerSession(AgentSession)` — owns a server-issued
  `session_id`; the `id` attribute is populated (not `None`).
- `runtime.session(resume=...)` — when `resume` is provided, skip
  `POST /session` and adopt the existing session_id. Flip
  `Feature.SESSION_RESUME` True.
- `execute(prompt, schema=)` — single turn via
  `client.session.message(session_id, ...)`. Structured output
  initially via a single in-process MCP server hosting a forced
  `submit_result` tool (mirrors `CopilotRuntime`). The MCP server
  is created lazily on first `execute(schema=)` and torn down at
  session `close()`.
- `stream(prompt, schema=)` — drive `prompt_async` and pull from
  `client.event.stream()`. Translate:
  - `message.part.delta` with `type=="text"` → `TextDelta`.
  - `message.part.delta` with `type=="reasoning"` → `ReasoningDelta`.
  - `message.part.updated` for tool blocks → `ToolCallStart` /
    `ToolCallResult` (Iteration D's surface; emit minimal events
    here that don't claim tool-loop behaviour we haven't wired).
  - `session.idle` → drain to `TurnComplete`.
  - `session.error` → raise the appropriate `Runtime*Error`.
- `cancel()` — `client.session.abort(session_id)` + close the SSE
  iterator. Surfaces `RuntimeCancelledError` to the awaiting call.
- `close()` — idempotent. Calls `client.session.delete(session_id)`
  unless the session was opened via `resume=` (in which case the
  session existed before us — leave it alone). Tears down the SSE
  subscription and any in-process MCP server spun up for structured
  output. The HTTP client itself outlives the session (owned by the
  runtime).
- Exception classification: `opencode_ai.APIError` subclasses →
  `RuntimeAuthError` / `RuntimeModelNotFoundError` /
  `RuntimeTransientError` / `RuntimeProtocolError`.
  `opencode_ai.APIConnectionError` → `RuntimeServerStartError`.
  `httpx.RemoteProtocolError` on SSE → `RuntimeProtocolError` (the
  known [#26697](https://github.com/sst/opencode/issues/26697)
  early-close bug — surface clearly so users know to restart the
  server, don't silently retry).
- Flip `Feature.STREAMING`, `Feature.CANCEL`,
  `Feature.SESSION_RESUME`, `Feature.STRUCTURED_OUTPUT_JSON_SCHEMA`
  True.
- `examples/probe_opencode_server.py` — minimal `execute(schema=)`
  probe + session-resume round-trip.

**Stopping point.** Single-turn structured output works against a
locally-running `opencode serve`. Streaming yields deltas.
Cancellation works. `session(resume=<prior_id>)` resumes a real
conversation.

### Iteration C — Polymorphic prompt + reasoning pass-through

~100 LOC. Phase-2-equivalent slice.

- `_split_prompt_parts` integration: `ImageInput` → OpenCode
  `message.part` shape `{"type": "image", "url": "data:...;base64,..."
  | "https://..."}`. `FileInput` → `{"type": "file", "url": "...",
  "name": "..."}`.
- `thinking=` kwarg → `additional_request_fields` pass-through. The
  agent server doesn't normalise reasoning, so the adapter forwards
  `{"reasoning": {"effort": "low"|"medium"|"high"}}` or
  `{"reasoning": {"budget_tokens": N}}` per the active backend's
  expected shape, gated on `provider_options.provider_id` if set.
  When unset, defaults to the OpenAI Chat Completions shape
  (`reasoning_effort: ...`) since that's the most widely-honoured
  envelope across open-weight backends.
- Document the "your mileage will vary" caveat loudly in
  `docs/adapters/opencode-server.md`: the adapter can't know whether
  the underlying model honours reasoning until the model emits a
  reasoning part. No silent failure — debug log when the response
  has no `reasoning` parts despite a `thinking=` request.
- Flip `Feature.REASONING_EFFORT`, `Feature.REASONING_BUDGET_TOKENS`,
  `Feature.VISION_INPUT`, `Feature.FILE_INPUT` True.
- Probe extensions: `probe_thinking.py`, `probe_vision.py` get
  `opencode` branches.

### Iteration D — Function tools + permission callback + MCP

~250 LOC. Phase-3 + Phase-5 (permission slice) + Phase-4 (MCP)
equivalent. Largest iteration; consider splitting in review if it
exceeds ~300 LOC.

**Function tools (the friction point).** OpenCode has no client-side
function-call transport. Two implementation paths:

- **Path 1 (recommended): in-process MCP wrapping.** When the caller
  passes `tools=[FunctionTool(...)]`, spin up an in-process MCP
  server (`mcp` Python package — already a transitive dep through
  some adapters), expose each `FunctionTool` as an MCP tool, register
  the server with OpenCode via `POST /mcp` at session start, and
  tear it down at `close()`. This is the only path that delivers
  proper tool-loop semantics with caller-defined Python handlers.
- **Path 2 (fallback): decline TOOLS_FUNCTION.** If MCP wrapping
  proves too complex for one iteration, ship the iteration with
  `Feature.TOOLS_FUNCTION=False` and point users at `unwrap()` +
  manual `client.mcp.add()`. Track MCP wrapping in Iteration G.

Path 1 is the target; Path 2 is the safety valve.

**Permission callback (the natural fit).** The SSE bus emits
`permission.asked` events with the request_id, tool name, and
arguments. Dispatch to the registered `PermissionCallback`, reply
via `client.permission.reply(request_id, decision=...)`. This is the
cleanest `PERMISSION_CALLBACK` implementation in the lineup — no
synthesis from a client-side tool loop, no special handling needed.

**MCP refs.** Translate airframe's `McpServerRef` shapes to
`client.mcp.add()` calls at session start. `STDIO` →
`{"type": "local", "command": [...]}`. `HTTP` / `SSE` →
`{"type": "remote", "url": "..."}`. Tear down at session close
(remove just the airframe-added registrations; leave anything the
user registered via `opencode.json` alone).

- Translate `FunctionTool` schemas + handlers → in-process MCP
  server (Path 1).
- Tool loop: airframe doesn't drive one (server does). Instead,
  observe `message.part` tool blocks on SSE, surface as
  `ToolCallStart` / `ToolCallResult` events.
- `PermissionCallback` wiring around `permission.asked` /
  `permission.replied` SSE events.
- `McpServerRef` → `POST /mcp` translation.
- Flip `Feature.TOOLS_FUNCTION` (if Path 1 lands),
  `Feature.TOOLS_MCP_STDIO`, `Feature.TOOLS_MCP_HTTP`,
  `Feature.TOOLS_MCP_SSE`, `Feature.PERMISSION_CALLBACK` True.

**`TOOLS_MCP_IN_PROCESS` non-goal.** OpenCode's MCP slots are
stdio/HTTP/SSE only. The Path 1 implementation above uses an
*airframe-internal* in-process MCP that exposes itself over stdio to
the OpenCode server — that's not the same as registering an
in-process MCP from the consumer's perspective. Flag permanently False.

### Iteration E — Hooks + budget

~100 LOC. Phase-5 equivalent.

- `EMITTABLE_HOOK_KINDS` ClassVar — 7 kinds (one more than other
  adapters because of `pre_compact`): `session_start`, `session_end`,
  `user_prompt_submit`, `pre_tool_use`, `post_tool_use`,
  `tool_failure`, `pre_compact`.
- `_on_event` plumbing on `OpenCodeServerSession` — synthesise hook
  events from the SSE bus rather than from a client-side tool loop.
  This is *easier* than the OpenAI-compat pattern because the events
  already exist on the wire; the adapter just maps them.
- Budget-cap enforcement via the shared `_enforce_budget_pre_turn()`
  helper. Cost-record accumulation reads `cost` fields from
  `message.part.updated` events when the underlying provider reports
  them; when it doesn't (Ollama, vLLM), `cost_usd` stays `None` and
  `BUDGET_USD_CAP` becomes effectively unenforced for that backend
  (document loudly).
- No in-tree `_OPENCODE_PRICING` table. The server reports cost when
  available; when it doesn't, the adapter doesn't synthesise. This
  is different from `BedrockRuntime` / `CodexRuntime` which keep a
  pricing dict — those wrap fixed catalogs; OpenCode fronts an
  open-ended set of backends and we'd be guessing.
- Flip `Feature.LIFECYCLE_HOOKS`, `Feature.BUDGET_USD_CAP`,
  `Feature.BUDGET_TURN_CAP` True.

### Iteration F — Wrap-up

~100 LOC + docs.

- `OpenCodeServerOptions` dataclass (per the surface above) wired
  through with `_check_provider_options`.
- Conformance contract suite green; integration test wrapper at
  `tests/test_opencode_server_integration.py` with `pytest-marker`
  gating (requires a locally-running `opencode serve`).
- Per-adapter docs page: `docs/adapters/opencode-server.md` covering
  install extra, server prerequisites (`opencode serve`), auth chain
  (cross-link to `docs/auth.md`), supported features, options,
  cost-reporting caveat (it depends on which backend the server
  routes to), the filesystem-boundary gotcha (tools run on the
  server's filesystem), vendor quirks (pre-1.0 SDK; SSE bug
  #26697; experimental tool endpoints we don't depend on), escape
  hatches.
- `docs/auth.md` — new `## OpenCodeServerRuntime` section covering
  the HTTP Basic chain + loopback guardrail.
- `docs/capabilities.md` matrix gains an `opencode` column.
- README "Supported providers" table row.
- CHANGELOG entry.

## Risks and decisions to flag during execution

1. **Pre-1.0 SDK churn.** `opencode-ai` is `0.1.0a23`. Server-side
   migration from Hono to Effect HttpApi recently. Expect breaking
   changes. **Pin `opencode-ai>=0.1.0a23,<0.2`** initially and bump
   the upper bound deliberately, not via `pip install -U`. Watch
   Stainless-regenerated client breaks separately from server-side
   API breaks.
2. **SSE stream early-close** ([#26697](https://github.com/sst/opencode/issues/26697)).
   Some setups close `/event` immediately after `server.connected`.
   The adapter must surface this as `RuntimeProtocolError` with a
   clear message ("OpenCode SSE closed prematurely after handshake
   — restart `opencode serve` and retry") rather than silently
   reconnecting (which masks a config bug). When the upstream fix
   lands, revisit; in the meantime the adapter does *not* implement
   automatic reconnection.
3. **Filesystem boundary.** Tools (`bash`, `write`, `edit`) run on
   the **server's** filesystem. For users running `opencode serve`
   on the same host as their adapter calls, this is invisible. For
   users running it in a container or remote host, edits don't land
   where they expect. The adapter cannot fix this; the docs page
   must call it out in a prominent warning block.
4. **Loopback-only auth posture.** Default OpenCode serves
   unauthenticated on 127.0.0.1. The adapter's
   non-loopback-without-password guardrail is essential — without
   it, a misconfigured `opencode serve --hostname 0.0.0.0` becomes
   a remote-bash endpoint. Test the guardrail explicitly.
5. **`Feature.TOOLS_FUNCTION` is contingent.** The in-process-MCP
   wrapping required to honour caller-defined Python tools is real
   engineering. If Iteration D's scope balloons, ship the iteration
   with TOOLS_FUNCTION=False and add Iteration G for MCP wrapping
   alone. Don't half-implement.
6. **Cost reporting is backend-dependent.** When the underlying
   model is hit through a provider that doesn't report cost
   (self-hosted Ollama, llama.cpp), `cost_usd` is `None`. Budget
   caps with `usd` set become unenforceable for that backend.
   Document; don't synthesise prices for backends we can't actually
   price.
7. **Reasoning pass-through is fragile.** Different backends use
   different reasoning envelopes (`reasoning_effort` for OpenAI-shape,
   `thinking` for Anthropic-shape, model-specific dicts for Gemini /
   DeepSeek). The adapter's per-`provider_id` dispatch in Iteration C
   is best-effort; the only fully reliable path is the consumer
   setting `provider_options=OpenCodeServerOptions(provider_id=...)`
   explicitly. Document that constraint.
8. **Server-lifecycle responsibility.** The adapter does not spawn
   `opencode serve`. First-call reachability classification is the
   only signal; users see "OpenCode server not reachable, run
   `opencode serve`" — make sure that error message is high-quality.
   Consider a `runtime.health_check()` helper sugar if first-call
   surface friction is reported.
9. **Experimental tool endpoints.** `/experimental/tool/*` is
   unstable upstream. The adapter does **not** consume those
   endpoints for v1. If a future Iteration G needs them, isolate
   behind a private helper and put it on a separate version-pinned
   path.
10. **Forced-tool structured output dependency on MCP wrapping.**
    Iteration B's `STRUCTURED_OUTPUT_JSON_SCHEMA=True` depends on
    the in-process MCP infrastructure that Iteration D will
    productionise. In B, we ship a minimal version; D refactors it
    onto the shared MCP-wrapping helper. Plan the refactor so B
    doesn't grow MCP code that has to be thrown away.

## Definition of done

- All six iterations merged.
- `runtime_for("opencode")` returns `OpenCodeServerRuntime` when
  the extra is installed; clean `ImportError` with
  `airframe-agents[opencode]` hint when not.
- `examples/probe_opencode_server.py` round-trips a structured-output
  prompt against a locally-running `opencode serve` with a
  configured upstream provider.
- `examples/probe_parity.py` includes `opencode` with no per-vendor
  conditionals; passes on a machine with `opencode serve` running.
- `examples/probe_supports.py --provider opencode` shows the
  expected feature matrix.
- Conformance contract suite green against a mocked
  `AsyncOpencode`; integration suite green against a live server.
- `docs/adapters/opencode-server.md` published; README provider
  table + capability matrix updated.
- CHANGELOG entry with iteration summary.
- `CLAUDE.md` reserved-IDs list includes `"opencode"`.

## When to start

**Phase 1 candidate.** OpenCode is the only adapter in the planned
lineup whose primary value prop is *open-weight agentics*; that
makes it Phase-1-relevant even though it doesn't depend on Phase 1's
`AgentSession` retrofit (the adapter is greenfield, so it lands
directly on the Phase-1-shaped protocol). Reasonable cadence: one
iteration per week, ~6 weeks end-to-end, mergeable in parallel with
Bedrock / Gemini work since they touch disjoint files.

Two triggers would make it Phase-1-priority work:

1. **A consumer asks for open-weights agentics.** Concrete user is
   the right gate — same discipline that's kept the `ProviderOptions`
   namespaces honest.
2. **A capability that only OpenCode delivers becomes load-bearing.**
   Self-hosted model gating, regulatory-isolated agentic workflows,
   or open-weight-only fleets would each be reasons.

Until either fires, the OpenAI-compat workaround via
`OpenCodeZenRuntime` / `OpenRouterRuntime` covers the "I just want
to call an open-weight model" case at the cost of no agentic
features.

## Open questions for the implementer

1. **`autostart=True` constructor mode.** Should the adapter
   optionally subprocess-spawn `opencode serve` when reachability
   fails? Pros: ergonomic first-use experience. Cons: race with the
   user's TUI session on SQLite state; harder to teardown cleanly.
   **Recommendation: defer.** Ship the adapter assuming the user
   manages the server lifecycle. Revisit if first-call friction is
   reported.
2. **Multi-session-per-runtime concurrency.** OpenCode supports any
   number of concurrent server-side sessions; the SSE bus is global
   per the `event.global_stream()` endpoint. Airframe's current
   single-session-per-runtime model can serve them sequentially, but
   nothing about the wire format requires that. Defer to the
   protocol-level decision on concurrent sessions (ADR-004 from
   `implementation-plan.md`).
3. **In-process MCP wrapping helper.** If Iteration D Path 1 lands,
   the wrapping infrastructure (caller `FunctionTool` → in-process
   MCP server) is reusable for any other adapter that needs to bridge
   a Python-handler-shaped tool surface to an MCP-server-shaped
   sink. Worth extracting into `src/airframe/_mcp_bridge.py` once a
   second consumer appears.
4. **Pricing-table opt-in.** Per Risk #6, the adapter doesn't ship a
   pricing table. If a consumer wants `BUDGET_USD_CAP` enforcement
   on self-hosted backends, they could pass a `cost_per_model:
   dict[str, ModelPricing]` to `OpenCodeServerOptions` and have the
   adapter compute cost client-side. Defer until requested.
5. **Per-session vs per-runtime HTTP client.** Currently planned:
   one `AsyncOpencode` per runtime, shared across sessions. If SSE
   connection limits become a bottleneck (the server's HTTP/2 limit
   is configurable; httpx defaults are conservative), revisit.

## Implementation wiring checklist

Beyond `src/airframe/adapters/opencode_server.py` itself, every new
adapter needs to touch these files. Easy to forget; easy to verify
by grepping for the closest sibling (`bedrock.py` for full-bespoke
shape, `opencode_go.py` for OpenAI-compat shape — note that the
*bespoke* shape is the right template here, not `opencode_go.py`).

### Source wiring

- [ ] `src/airframe/discovery.py` — add `OpenCodeServerRuntime` to
      `_builtin_runtime_classes()` (currently 7 entries after Bedrock).
- [ ] `src/airframe/__init__.py` — `from airframe.adapters.opencode_server
      import OpenCodeServerRuntime` at module level + entry in
      `__all__` (alphabetical: between `OpenCodeGoRuntime` and
      `OpenCodeZenRuntime`).
- [ ] `src/airframe/testing/contracts.py` — add `"opencode":
      OpenCodeServerOptions` to the `matching` dict inside
      `_check_provider_options`.
- [ ] `src/airframe/testing/integration.py` — add `"opencode":
      ["OPENCODE_SERVER_PASSWORD"]` to `_PROVIDER_AUTH`, plus a
      "needs-server-running" classification distinct from the
      "needs-credentials" classification (first time we've had this
      split; a `_PROVIDER_RUNTIME_PREREQ` parallel dict is the right
      shape, populated only for `"opencode"` initially).

### Probe + examples wiring

- [ ] `examples/probe_budget.py` — add `"opencode"` to the provider
      tuple in the capability matrix print.
- [ ] `examples/probe_parity.py` — picks up the new adapter
      automatically via `list_providers()`. Add an
      `AIRFRAME_PROBE_MODEL_OPENCODE` env hook for default-model
      override (the server's available models depend on its
      configured upstreams, so no sensible compile-time default).

### Packaging

- [ ] `pyproject.toml` — new `opencode = ["opencode-ai>=0.1.0a23,<0.2"]`
      extra under `[project.optional-dependencies]`, AND add
      `"opencode-ai>=0.1.0a23,<0.2"` to the `all = [...]` list, AND
      add it to the `[dependency-groups].test` list (unit suite
      imports `opencode_ai` for mocking). Note the deliberately
      narrow version range — Risk #1.

### Documentation

- [ ] `README.md` — provider table row (between OpenCode Go and
      OpenRouter alphabetically); update the install-extras example
      to mention `[opencode]`; add `opencode` to the comma-separated
      provider-ID example in the quickstart.
- [ ] `docs/auth.md` — quick-reference table row + a full
      `## OpenCodeServerRuntime` section covering the HTTP Basic
      chain + loopback guardrail.
- [ ] `docs/reference.md` — adapter table row + add
      `OpenCodeServerRuntime` to the `__all__` snippet.
- [ ] `docs/adapters/opencode-server.md` — new page covering identity,
      server prerequisites, model catalog (driven by the configured
      upstream providers — show `client.provider.list()` output as
      example), supported features, options, the cost-reporting
      caveat, the filesystem-boundary warning, escape hatches,
      see-also.
- [ ] `docs/capabilities.md` — add `opencode` column to the
      capability matrix. Distinct from `opencode-zen` / `opencode-go`
      columns.
- [ ] `CLAUDE.md` — add `"opencode"` to the canonical provider IDs
      list in the "Provider IDs are strict" paragraph; clarify it's
      the agent-server adapter distinct from the two existing
      gateway adapters.

### Test wiring

- [ ] `tests/test_opencode_server.py` — unit tests. Mirror
      `tests/test_claude_code.py` for structure (full-bespoke
      template, server-owned sessions, SSE event translation).
- [ ] `tests/test_opencode_server_session.py` — session-class
      behaviour. SSE event → `RuntimeEvent` translation, permission
      callback dispatch, abort propagation.
- [ ] `tests/test_opencode_server_conformance.py` — drives the
      shared `airframe.testing.contracts` suite.
- [ ] `tests/test_opencode_server_integration.py` — pytest-marker-gated
      behavioural tests against a live `opencode serve`. Mirror
      `tests/test_claude_code_integration.py` for shape; skip cleanly
      with a clear message when the server isn't reachable rather
      than failing the suite.
- [ ] `tests/test_discovery.py` — update the
      `test_list_providers_returns_all_when_installed_only_false`
      expected set + filtered tests + third-party-discovery test.

### Issue + project housekeeping

- [ ] Open / update a tracking issue mirroring the Bedrock /
      Gemini pattern. Move from candidate to "shipping" once
      Iteration A merges.
- [ ] CHANGELOG entry with iteration summary at each iteration's
      merge.

## Closest in-tree templates to read first

Open these side-by-side with the plan before writing any code.

| File | What to learn from it |
|---|---|
| `src/airframe/adapters/claude_code.py` | The full-bespoke shape — `OpenCodeServerRuntime` mirrors this structurally (subclasses `AgentRuntime` directly, owns its own `AgentSession` subclass, owns the session lifecycle). Roughly 800 LOC; OpenCode target ~700 is slightly lower because the SDK is HTTP rather than subprocess JSON-RPC. |
| `src/airframe/adapters/bedrock.py` | The recently-merged sibling for translating typed events on a streaming protocol into airframe's `RuntimeEvent` union. The Converse SSE chunk → `TextDelta` / `ReasoningDelta` mapping is the closest pattern to OpenCode's `message.part.delta` translation. |
| `src/airframe/adapters/copilot.py` | The forced-tool-for-structured-output pattern. The Iteration B `submit_result`-via-MCP shim mirrors this; pay attention to the schema-fingerprint caching pattern so we don't rebuild the MCP server on every `execute()`. |
| `src/airframe/adapters/opencode_go.py` | The discovery / `__init__.py` / docs cross-reference pattern (smaller surface; quick to scan). **Note carefully**: this is a different adapter despite the name. The agent-server adapter shares almost no code with it. |
| `src/airframe/sessions.py` | The shared helpers: `_enforce_budget_pre_turn`, `_check_provider_options`. `OpenCodeServerSession` calls into both. |
| `src/airframe/permission.py` | `PermissionCallback` shape — the SSE-driven dispatch in Iteration D is the cleanest implementation in the lineup; make sure the contract suite exercises it against the live server. |

## Naming reservations

Established with this plan:

- `"opencode"` — this adapter (agent HTTP server; `opencode serve`).
- `"opencode-zen"` — existing `OpenCodeZenRuntime` (OpenAI-compat
  gateway at `https://opencode.ai/zen/v1`, per-token billing).
- `"opencode-go"` — existing `OpenCodeGoRuntime` (OpenAI-compat
  gateway at `https://opencode.ai/zen/go/v1`, flat-fee subscription).
- `"opencode-cloud"`, `"opencode-managed"` — **not reserved**.
  OpenCode does not ship a managed cloud variant; if one appears,
  reserve at that point. The bespoke server runs locally or
  user-hosted; the two Zen gateways already cover the hosted-billing
  surface.

The three `opencode*` provider IDs are deliberately distinct. They
share a brand and nothing else — different wire formats, different
auth, different feature surface, different billing. Per the same
lesson `bedrock-adapter-plan.md` documents around
`"bedrock"` vs `"bedrock-agents"`: same-vendor distinct-product =
distinct provider ID.

## First commit in a fresh session

A reasonable Iteration A first commit (the scaffolding-only slice):

```
src/airframe/adapters/opencode_server.py    # new — Iteration A surface
src/airframe/discovery.py                   # +OpenCodeServerRuntime in builtins
src/airframe/__init__.py                    # +export +__all__
pyproject.toml                              # +[opencode] extra
tests/test_opencode_server.py               # new — identity, validate_binding, auth-chain unit tests
tests/test_discovery.py                     # +opencode in expected sets
CLAUDE.md                                   # +opencode in reserved-IDs paragraph
```

That should pass `make ci` cleanly with `Feature` flags all False
(no behaviour wired yet). After review, Iteration B adds `execute()`
+ `stream()` + `cancel()` + `session(resume=...)` and flips the
first four `Feature` flags True.
