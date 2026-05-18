# `MistralRuntime` adapter plan

Companion to [implementation-plan.md](./implementation-plan.md). Specs
a new built-in adapter wrapping **Mistral AI's Agents API** via the
official `mistralai` Python SDK with the `[agents]` extra. Distinct
from any future thin OpenAI-compat adapter against Mistral's
chat-completions endpoint — the Agents API is a proprietary
server-side surface that delivers managed conversation state,
connectors (MCP), agent versioning, and multi-agent handoffs.

The plan mirrors the ABCDEF iteration cadence used by
[`bedrock-adapter-plan.md`](./bedrock-adapter-plan.md) and
[`google-genai-adapter-plan.md`](./google-genai-adapter-plan.md):
scaffold → execute/stream/cancel → polymorphic prompt + reasoning →
tools + permission → hooks + budget → wrap-up. Targeting
~700–900 LOC over 6 iterations, each independently mergeable.

## Motivation

Airframe's adapter lineup covers four architectural shapes today:

- **Subprocess + JSON-RPC** — `ClaudeCodeRuntime`, `CopilotRuntime`,
  `CodexRuntime`. The vendor SDK shells a local CLI; airframe drives
  it across IPC.
- **OpenAI-compat HTTP** — `OpenCodeZenRuntime`, `OpenCodeGoRuntime`,
  `OpenRouterRuntime`. Stateless chat-completions request/response;
  airframe maintains the message buffer client-side.
- **AWS-billed Converse HTTP** — `BedrockRuntime`. Stateless from
  the client but with vendor-normalised envelopes.
- **Planned: local-server agent** — `OpenCodeServerRuntime`. The
  user runs an HTTP server locally; the adapter is a typed client.

Mistral's Agents API introduces a **fifth shape**: a hosted,
multi-tenant agent service where the vendor owns the conversation
state, the tool-execution loop, and the connector lifecycle on
their servers. The Python SDK is a typed REST client; the
*agentic intelligence* lives behind the API.

Why this matters for airframe:

- **Capability surface only Mistral covers.** Multi-agent handoffs
  (with explicit `server`/`client` execution semantics), agent
  versioning + aliases (a production rollout primitive),
  vendor-managed MCP connectors with auto-discovery, persistent
  conversations referenced by ID across processes. None of the
  existing adapters surface this combination.
- **Architectural prep for `BedrockAgentsRuntime`.** AWS's
  `bedrock-agent-runtime` service (Knowledge Bases, action groups,
  server-side orchestration — reserved as `"bedrock-agents"` in
  `CLAUDE.md`) has the same shape: hosted agent, REST client,
  server-managed conversation. Building Mistral first teaches us
  the right base class for both.
- **First-party European model house.** Mistral is the only major
  European foundation-model vendor with a published agent SDK.
  Useful for consumers with data-residency or sovereignty
  constraints that rule out AWS Bedrock and OpenAI direct.

Wire format is **not** OpenAI Chat Completions. The Agents API has
its own envelope: `/v1/agents`, `/v1/conversations`,
`/v1/connectors`. `OpenAICompatibleRuntime` cannot reach this
surface; it can reach Mistral's plain `/v1/chat/completions` endpoint
which is a *different* (lower-feature) thing.

## Non-goals

- **No chat-completions integration.** Mistral exposes its models
  via `/v1/chat/completions` as well as via Agents. The
  chat-completions surface is reachable through
  `OpenAICompatibleRuntime` and a thin future
  `MistralCompletionsRuntime` is the right home if demand
  materialises (separate provider ID: `"mistral-completions"`).
  `MistralRuntime` is specifically for the Agents API and the
  capabilities it unlocks (handoffs, connectors, versioning,
  conversations).
- **No agent-definition CRUD as airframe protocol surface.**
  Creating, updating, versioning, and deleting agents is an
  out-of-band operation handled via the Mistral console, IaC, or
  direct `mistral.beta.agents.create()` calls (reachable via
  `unwrap()`). Airframe consumes existing agents by ID and runs
  conversations against them; the protocol stays narrow.
- **No `mistralai[gcp]` integration.** Mistral on Vertex AI is a
  separate auth scheme + endpoint and a different adapter
  surface. If demand materialises, ship as `MistralVertexRuntime`
  with provider ID `"mistral-vertex"`.
- **No connector OAuth flow management.** Some Mistral connectors
  (Gmail, Drive) require OAuth from the calling principal. That
  flow lives entirely in the Mistral console / Agents UI;
  airframe consumes pre-authorised connectors and doesn't broker
  the OAuth dance.
- **No workflow / payload-encryption / payload-offloading extras.**
  Those `mistralai[workflow-*]` extras serve a different
  feature surface (long-running workflows; out-of-band payload
  storage). Out of scope for the agent adapter.

## Adapter shape

`MistralRuntime(AgentRuntime)` — a direct subclass of `AgentRuntime`,
not `OpenAICompatibleRuntime`. Reasons:

1. Wire format isn't OpenAI Chat Completions — it's Mistral's
   Agents/Conversations envelope, served by
   `mistralai.Mistral(api_key=...)`.
2. **Conversations live server-side.** Mistral owns conversation
   state, accessible by `conversation_id`; the client-side
   `messages=[]` buffer pattern that every OpenAI-compat adapter
   uses is the wrong abstraction.
3. Auth is bearer API key only (`MISTRAL_API_KEY`), but the SDK's
   error taxonomy (`MistralError`, `HTTPValidationError`,
   `ObservabilityError`) needs explicit translation to airframe's
   hierarchy.

`MistralSession(AgentSession)` is the primary surface. It owns:

- An `agent_id` — either provided by the caller (referencing a
  pre-defined agent created out-of-band) or synthesised at session
  start for ad-hoc conversational use.
- A `conversation_id` — server-issued at first `execute()` /
  `stream()`; `AgentSession.id` is populated from this.
- An SSE iterator for streaming responses.

`reset()` drops the conversation reference (subsequent calls start
a new conversation). `close()` is idempotent; tears down the
client. Server-side conversations linger until garbage-collected
by Mistral's retention policy — airframe doesn't proactively
delete them unless an explicit `delete_on_close=True` knob is
added later.

```
src/airframe/adapters/mistral.py             ~700 LOC (target)
tests/test_mistral.py                        ~400 LOC mirroring test_bedrock.py
tests/test_mistral_session.py                ~200 LOC session-class behaviour
tests/test_mistral_conformance.py            airframe.testing.contracts driver
tests/test_mistral_integration.py            pytest-marker-gated live tests
docs/adapters/mistral.md                     ~200 lines
examples/probe_mistral.py                    single-call execute(schema=) probe
```

| Class attribute | Value |
|---|---|
| `PROVIDER_ID` | `"mistral"` |
| `REQUIRES_PACKAGE` | `"mistralai"` |
| `EXTRA_NAME` | `"mistral"` |
| `label` | `"mistral"` |

The `[agents]` sub-extra on the upstream package is what enables
agent completion + streaming; airframe's `[mistral]` extra pulls
`mistralai[agents]` rather than the bare `mistralai`.

**Provider ID reservation note.** `"mistral"` is not yet in the
reserved-IDs list in `CLAUDE.md`. Iteration A adds it.
`"mistral-completions"` and `"mistral-vertex"` are **reserved**
alongside for the future chat-completions and Vertex AI siblings.
Same lesson as `bedrock-adapter-plan.md` (`"bedrock"` vs
`"bedrock-agents"`): same vendor with distinct surfaces = distinct
provider IDs.

## SDK surface this adapter wraps

`mistralai` 2.4.5 (May 2026), Python ≥3.10, with the `[agents]`
extra. Relevant surface:

- **Client.** `mistralai.Mistral(api_key=...)`. Sync and async
  use the same class; async methods are suffixed `_async`
  (`complete_async`, `upload_async`, etc.). Airframe uses async
  throughout.
- **Agent management** (not airframe's primary surface; reachable
  via `unwrap()`). `mistral.beta.agents.create(...)`,
  `mistral.beta.agents.list(...)`, `mistral.beta.agents.update(...)`,
  `mistral.beta.agents.version(...)`, `mistral.beta.agents.alias(...)`.
- **Agent execution (airframe's primary surface).**
  `mistral.agents.complete(messages=[...], agent_id="<id>", stream=False,
   response_format={"type": "json_schema", "json_schema": {...}})`
  for one-shot; `complete_async` for async; an `agents.stream(...)`
  surface for streaming (verify exact name in Iteration B).
- **Conversations.** `mistral.beta.conversations.start(...)` /
  `append(...)` / `restart(...)` for managed multi-turn flows
  (alternative to passing the full `messages=[...]` history on
  every call). Verify exact API in Iteration B.
- **Connectors (Mistral's MCP equivalent).**
  `mistral.beta.connectors.*` — server-managed MCP registrations.
  Connectors are configured per-agent; the agent definition
  references them.
- **Built-in tools.** Web search, code interpreter, image
  generation, document library. Toggled via the agent definition
  rather than per-call (so airframe doesn't gate these — the
  pre-defined agent decides).
- **Streaming.** SSE iterator exposed as a Python generator;
  context-manager-able (`with response as event_stream:`).
- **Models.** `mistral.models.list()` returns available models.
- **Errors.** `MistralError` base; subclasses include
  `HTTPValidationError` (422) and `ObservabilityError` (invalid
  parameters). The error object carries `message`, `status_code`,
  `headers`, `body`, `raw_response`.

The key shape mismatch with airframe's existing adapters: Mistral's
`agents.complete()` is a *stateless* helper that takes the full
`messages=[...]` history each call AND an `agent_id`. To support
multi-turn properly with server-side state, airframe uses the
`beta.conversations.*` surface, which gives a `conversation_id`
that survives across calls. Iteration B picks between these two
paths and probably uses both: `execute()` against `agents.complete()`
for one-shot (faster startup, no server-side conversation
creation), `session()` against `beta.conversations.*` for
multi-turn (state lives server-side).

## Feature support matrix (target)

| Feature | Status | Notes |
|---|---|---|
| `STRUCTURED_OUTPUT_JSON_SCHEMA` | ✓ | Native via `response_format={"type": "json_schema", "json_schema": {...}}` on `agents.complete()`. Mistral's documented schema-mode is closer to OpenAI's than to Anthropic's. |
| `STRUCTURED_OUTPUT_STRICT` | ◐ verify | Mistral docs reference a `strict` flag on `response_format`; verify behaviour in Iteration B (does it reject non-strict-compatible schemas, or just enforce harder?). Flip True if it matches OpenAI's strict-mode contract. |
| `STREAMING` | ✓ | `agents.stream(...)` (or `complete(stream=True)`) returns an SSE iterator. Translate to `TextDelta` / `ReasoningDelta` / `TurnComplete`. |
| `CANCEL` | ✓ | `asyncio.Task.cancel()` propagates to the underlying httpx; SSE stream gets closed via its context-manager exit. |
| `SESSION_RESUME` | ✓ | **Natively first-class via `conversation_id`.** Pass an existing conversation_id to `runtime.session(resume=...)` and the adapter skips creating a new conversation; server has full history. Among the cleanest `SESSION_RESUME` stories in the lineup. |
| `REASONING_EFFORT` | ✓ | Mistral's reasoning models (Magistral / future) accept a `reasoning_effort` knob; older models silently ignore. Adapter declares True with model-gating. |
| `REASONING_BUDGET_TOKENS` | ✗ | No documented budget-tokens knob (Anthropic-only convention). |
| `VISION_INPUT` | ✓ | Pixtral models accept image content parts. Wire-format identical to OpenAI's `[{"type": "image_url", ...}]`. |
| `FILE_INPUT` | ✓ | Documents supported via the `agents` surface; uses Mistral's document library when configured on the agent. |
| `TOOLS_FUNCTION` | ✓ | Mistral's `tools=[...]` on `agents.complete()` accepts function-call definitions wire-compatible with OpenAI's. **Subtlety:** function tools are typically baked into the agent definition rather than passed per-call. Iteration D verifies whether per-call `tools=` is honoured by the Agents API; if only agent-definition-level, document the constraint and flip True with caveat. |
| `TOOLS_MCP_STDIO` | ✗ | Mistral's connectors run server-side; stdio (local-process) MCP doesn't translate. |
| `TOOLS_MCP_HTTP` | ✓ | Connectors are remote-URL HTTP; airframe's `McpServerRef(transport="http", ...)` translates to a connector registration. |
| `TOOLS_MCP_SSE` | ✓ | Connectors support the SSE variant of HTTP MCP. |
| `TOOLS_MCP_IN_PROCESS` | ✗ | No in-process slot. |
| `PERMISSION_CALLBACK` | ◐ defer | Mistral runs tool execution server-side without per-call permission gates (unlike Claude / Copilot / OpenCode). For client-side handoffs (`handoff_execution=client`), the adapter receives the handoff request and the consumer decides whether to invoke; that's adjacent to but not the same as `PermissionCallback`. Mark False initially; revisit if Mistral exposes a tool-approval webhook. |
| `LIFECYCLE_HOOKS` | ✓ (6 kinds) | Synthesise from SSE event types: `session_start`, `session_end`, `user_prompt_submit`, `pre_tool_use` / `post_tool_use` / `tool_failure` (from agent-level tool execution events). `pre_compact` False (no compaction concept); `rate_limit` False (httpx surfaces 429 as transient). |
| `BUDGET_USD_CAP` | ✓ | Server reports usage on each response; adapter accumulates client-side against an in-tree `_MISTRAL_PRICING` table for cost computation. |
| `BUDGET_TURN_CAP` | ✓ | Client-side counter, same as siblings. |
| `SANDBOX` | ✗ | Server-side; nothing for airframe to surface. |
| `SUBAGENTS` | ◐ defer | Mistral's multi-agent handoffs are the closest analogue. Defer to Phase 6+ when airframe's subagent kwarg lands; the wire shape is ready when airframe is. |

## Auth chain

Mirrors the existing adapters' "checked in order" pattern:

1. **Explicit `api_key=` constructor argument.** Highest precedence.
2. **`MISTRAL_API_KEY` env var.** Mistral's documented env var name.

If neither resolves, the first call raises `RuntimeAuthError`
pointing at `https://console.mistral.ai/api-keys`.

**No on-disk auth-file convention.** Mistral has no equivalent of
opencode's `~/.local/share/opencode/auth.json` — env var or
explicit kwarg only. Documented in `docs/auth.md#mistralruntime`.

`list_models()` requires a credential. Standard `pytest.skip`
when credentials aren't set.

## `MistralOptions` (provider-options namespace)

```python
@dataclass(frozen=True, slots=True)
class MistralOptions:
    # Reference an existing pre-defined agent by ID (created via
    # the Mistral console or `mistral.beta.agents.create(...)`).
    # When set, `session()` uses this agent_id; when None, the
    # adapter uses a default agent or synthesises one for ad-hoc
    # use (Iteration B decides which).
    agent_id: str | None = None

    # Pin to a specific agent version. Mistral supports version
    # aliases (`production`, `staging`, ...) which are themselves
    # mutable references; passing `agent_version_id=` pins to an
    # immutable version for reproducibility.
    agent_version_id: str | None = None

    # Handoff execution policy when this agent invokes another.
    # "server" — Mistral routes the handoff transparently.
    # "client" — the adapter receives the handoff request and the
    # consumer decides what to do with it (typically: run a
    # different agent or surface to the user).
    # None — use the agent's default.
    handoff_execution: Literal["server", "client", None] = None

    # Per-request tool overrides for ad-hoc tool registration when
    # the pre-defined agent doesn't already include them. Note:
    # not all built-in tools are togglable per-call; only function
    # tools and registered connectors. The model decides which
    # tools to use given availability.
    additional_tools: tuple[Any, ...] = ()

    # Connector overrides — additional MCP-style remote tools
    # registered for this session beyond the agent's defaults.
    additional_connectors: tuple[Any, ...] = ()

    # Forwarded into the SDK request for vendor-specific knobs
    # airframe doesn't have first-class support for.
    additional_request_fields: dict[str, Any] | None = None

    # When True, the adapter deletes the server-side conversation
    # at session close (otherwise it lingers per Mistral's retention
    # policy). Useful for ephemeral / privacy-sensitive flows.
    delete_on_close: bool = False
```

Same tagged-union discipline as the existing namespaces — mismatched
type raises `UnsupportedFeatureError` at `session(provider_options=)`.

## Iteration breakdown

### Iteration A — Protocol scaffolding (no behaviour)

~150 LOC. Lands the adapter's shape without wiring substantive
features.

- `src/airframe/adapters/mistral.py` —
  `MistralRuntime(AgentRuntime)` with ClassVars, lazy `mistralai`
  import (deferred to first method call), `validate_binding`
  (accepts any non-empty `model_id` when `provider_id == "mistral"`;
  Mistral's model catalogue is too dynamic to gate by prefix).
- `_resolve_api_key()` chain implementing the two-step auth above.
- Empty `SUPPORTED_FEATURES = frozenset()` initially; flip flags
  on as each iteration wires them.
- `unwrap(Mistral)` runtime escape hatch.
- `close()` idempotent + never raises (closes the `Mistral` client
  context).
- `reset()` no-op at the runtime level (sessions own state).
- `list_models()` hits `client.models.list()`. Mistral's models
  endpoint authenticates per the bearer token; classify auth
  failures cleanly.
- Add `"mistral"` to the reserved-IDs paragraph in `CLAUDE.md`
  and reserve `"mistral-completions"` and `"mistral-vertex"`
  alongside.
- Discovery registration in `discovery.py` + top-level export.
- Conformance contracts pass against a mocked `Mistral` client.

**Stopping point.** `import airframe; airframe.list_providers()`
includes `"mistral"` when the extra is installed. Discovery +
capability predicates work; no live behaviour wired.

### Iteration B — Execute + streaming + cancellation + session-resume

~250 LOC. The Phase-1-equivalent slice. Largest iteration because
of the agents-vs-conversations decision; budget extra review time.

**Two-path decision.** Mistral has two flows that both feel like
"run an agent turn":

- `mistral.agents.complete(messages=[...], agent_id=..., stream=...)`
  — stateless; caller passes full history each call.
- `mistral.beta.conversations.{start, append, restart}(...)` —
  stateful; server owns the conversation by ID.

Airframe uses both. Mapping:
- `runtime.execute(...)` (one-shot sugar) → `agents.complete()`.
  Faster, no server-side conversation creation. Each call is
  independent.
- `runtime.session().execute(...)` and `.stream(...)` (multi-turn)
  → `beta.conversations.start()` + `append()`. State on the
  server.
- `runtime.session(resume=conversation_id)` → adopt the
  conversation_id and `append()` to it directly.

Implementation:

- `MistralSession(AgentSession)` — owns an `agent_id`,
  `conversation_id`, and an SSE iterator slot.
- `runtime.session(resume=...)` — populates `conversation_id` from
  the resume value; first `execute()` calls `append()` rather than
  `start()`.
- `execute(prompt, schema=)` — single turn. Either:
  - First call on this session: `beta.conversations.start(
      agent_id=..., inputs=[{"role": "user", "content": prompt}])`
  - Subsequent: `beta.conversations.append(
      conversation_id=..., inputs=[...])`
  Structured output via `response_format={"type": "json_schema",
  "json_schema": {...}}` (verified at iteration kickoff).
- `stream(prompt, schema=)` — same two-path logic, with the
  streaming variant of the same endpoints. Translate SSE events:
  - Content-delta events → `TextDelta`.
  - Reasoning-delta events (if Mistral surfaces these) →
    `ReasoningDelta`.
  - Tool-use events → `ToolCallStart` / `ToolCallResult`
    (Iteration D wires the full surface; emit minimally here).
  - Terminal event → `TurnComplete` with populated `RuntimeResult`
    including `CostRecord`.
- `cancel()` — `asyncio.Task.cancel()` + close the SSE iterator
  via its context-manager exit.
- `close()` — idempotent. If `MistralOptions.delete_on_close=True`
  *and* the conversation was created by us (not resumed), call
  `beta.conversations.delete(conversation_id)`; otherwise leave
  the server-side conversation in place for Mistral's retention
  policy.
- Exception classification: `MistralError` →
  - `HTTPValidationError` (422) → `RuntimeProtocolError`
  - status 401/403 → `RuntimeAuthError`
  - status 404 → `RuntimeModelNotFoundError` if message references
    model/agent; else `RuntimeProtocolError`
  - status 5xx / 429 → `RuntimeTransientError`
  - `ObservabilityError` → `RuntimeProtocolError`
- Flip `Feature.STREAMING`, `Feature.CANCEL`,
  `Feature.SESSION_RESUME`, `Feature.STRUCTURED_OUTPUT_JSON_SCHEMA`
  True. `STRUCTURED_OUTPUT_STRICT` flip contingent on the strict-
  mode verification.
- `examples/probe_mistral.py` — minimal `execute(schema=)` probe.

**Stopping point.** Single-turn structured output works against
live Mistral with `MISTRAL_API_KEY` set. Streaming yields deltas.
Cancellation works. `session(resume=<prior_id>)` resumes a real
conversation.

### Iteration C — Polymorphic prompt + reasoning

~100 LOC. The Phase-2-equivalent slice.

- `_split_prompt_parts` integration: `ImageInput` → Mistral's
  `{"type": "image_url", "image_url": {"url": "..."}}` content
  part (wire-compatible with OpenAI). `FileInput` → Mistral's
  document-content shape (verify exact field name in iteration
  kickoff).
- `thinking=` kwarg → `reasoning_effort` on the request when set,
  honoured on reasoning models (Magistral and successors) and
  silently ignored on others. Map `"low"`/`"medium"`/`"high"`
  directly; `"minimal"` coerces to `"low"`. `{"budget_tokens": N}`
  declined (no Mistral knob).
- Flip `Feature.REASONING_EFFORT`, `Feature.VISION_INPUT`,
  `Feature.FILE_INPUT` True.
- Probe extensions: `probe_thinking.py`, `probe_vision.py` get
  `mistral` branches.

### Iteration D — Function tools + connectors (MCP)

~150 LOC. Phase-3 + Phase-4 equivalent. **Permission callback
is NOT in this iteration** — see Risk #5.

**Function tools.** Translate `FunctionTool` → Mistral's tool
schema shape on the request. Verify whether per-call `tools=`
is honoured or only agent-definition-baked tools fire (Iteration
D's first task). If per-call works, ship full
`Feature.TOOLS_FUNCTION=True`; if only agent-baked, document the
constraint and flip True with caveat ("function tools must be
declared on the agent; per-call additions ignored").

**Connectors (MCP refs).** Translate airframe's `McpServerRef`
HTTP/SSE shapes to Mistral's connector-registration API.
`additional_connectors` on `MistralOptions` accepts pre-built
Mistral connector objects via `unwrap()`-style pass-through for
callers needing connector features airframe doesn't surface.

- `FunctionTool` ↔ Mistral tool-schema shape.
- `McpServerRef(transport="http"|"sse")` ↔ connector registration.
- Tool-use events on SSE → `ToolCallStart` / `ToolCallResult`
  surfaced to consumer.
- Flip `Feature.TOOLS_FUNCTION` (gated as above),
  `Feature.TOOLS_MCP_HTTP`, `Feature.TOOLS_MCP_SSE` True.

`TOOLS_MCP_STDIO` and `TOOLS_MCP_IN_PROCESS` stay permanently
False — Mistral connectors are remote URLs only.

### Iteration E — Hooks + budget

~100 LOC. Phase-5 equivalent.

- `EMITTABLE_HOOK_KINDS` ClassVar — six kinds: `session_start`,
  `session_end`, `user_prompt_submit`, `pre_tool_use`,
  `post_tool_use`, `tool_failure`. No `pre_compact` (Mistral has
  no compaction concept); no `rate_limit` (httpx 429s are generic
  transients, not typed events).
- `_on_event` plumbing on `MistralSession` — synthesise hook
  events from the SSE event types.
- Budget-cap enforcement via the shared
  `_enforce_budget_pre_turn()` helper.
- In-tree `_MISTRAL_PRICING` table for the published Mistral
  models (Mistral Large, Small, Medium, Pixtral, Codestral,
  Magistral, etc.). Rates from
  [Mistral's pricing page](https://mistral.ai/pricing) at PR
  time, documented as point-in-time.
- Flip `Feature.LIFECYCLE_HOOKS`, `Feature.BUDGET_USD_CAP`,
  `Feature.BUDGET_TURN_CAP` True.

### Iteration F — Wrap-up

~80 LOC + docs.

- `MistralOptions` dataclass (per the surface above) wired
  through with `_check_provider_options`.
- Conformance contract suite green; integration test wrapper at
  `tests/test_mistral_integration.py`.
- Per-adapter docs page: `docs/adapters/mistral.md` covering
  install extra, auth chain (cross-link to `docs/auth.md`),
  supported features, options reference, model IDs, agent-vs-
  conversation distinction (the architectural quirk that
  distinguishes this adapter), connectors / handoffs / agent
  versioning, structured output mechanism, vendor quirks,
  native escape hatches.
- `docs/auth.md` — new `## MistralRuntime` section.
- `docs/capabilities.md` matrix gains a Mistral column.
- README "Supported providers" table row (alphabetised between
  `CopilotRuntime` and `OpenCodeGoRuntime`).
- `docs/architecture.md` — new `### Mistral Agents API`
  subsection under "Operational landmines" with the verified-by-
  this-point landmines from Risks below.
- CHANGELOG entry.

## Risks and decisions to flag during execution

1. **API stability — `beta.*` namespace.** Mistral's conversations
   and connectors live under `mistral.beta.*` at the SDK level
   today. Beta status implies breaking-change risk. Pin
   `mistralai>=2.4.5,<3` initially and watch the upstream
   changelog separately for `beta` → `v1` graduations.
2. **Two-path complexity — `agents.complete` vs
   `beta.conversations.*`.** Iteration B is larger than usual
   because of this; the adapter has to honestly model two paths
   without doubling its surface. Decision: `execute()` (one-shot
   sugar) uses `agents.complete()`; `session()` uses
   `beta.conversations.*`. Document the implication: cost
   characteristics differ (one-shot resends full history each
   call; conversation uses server state once initialised).
3. **Pre-defined-agent dependency.** Mistral agents are typically
   created out-of-band (via console or
   `mistral.beta.agents.create`). The adapter assumes
   `MistralOptions.agent_id` is provided OR a default-agent
   resolution happens. Iteration B picks: either ship with a
   *required* `agent_id` (caller must create an agent first) or
   ship with a synthesised default that calls
   `agents.create()` lazily on first use. Synthesised default is
   ergonomic; required agent_id is honest. Recommend synthesised
   default for `execute()` (one-shot, ad-hoc) and required for
   multi-turn `session()` (consumers should be deliberate when
   creating server-side state).
4. **Connector availability differs per Mistral plan.** Some
   connectors (Gmail, Drive) require workspace-level
   authorisation. If the caller registers a connector their plan
   doesn't authorise, Mistral returns a permission error.
   Surface as `RuntimeAuthError` with a clear message rather
   than letting the SDK's `MistralError` cascade.
5. **`PERMISSION_CALLBACK=False` is the surprising decline.**
   Three of four current adapters honour permission callbacks;
   Mistral's server-side execution model doesn't expose
   per-tool-call gates. Document this clearly as a tradeoff: the
   capability gap is the cost of server-managed agents. If
   Mistral adds approval webhooks later, revisit.
6. **Handoff execution semantics.** `handoff_execution=client`
   is the path where airframe matters most — the consumer
   receives the handoff request and decides what to do. Iteration
   D establishes the wire shape and decides whether to surface
   handoffs as `RuntimeEvent`s, `HookEvent`s, or a new
   discriminated-union variant. **This is a shape-lock decision**
   worth an ADR if it ships; document the choice carefully.
7. **Pricing-table drift.** Mistral's prices have shifted multiple
   times in 2025–2026. `_MISTRAL_PRICING` is point-in-time;
   document loudly. Shared shape with `_BEDROCK_PRICING` /
   `_CODEX_PRICING` so updates are one-PR churn. By the third
   in-tree pricing table, consider extracting to
   `src/airframe/_pricing.py`.
8. **Vertex AI mode out of scope.** Mistral on Vertex
   (`mistralai[gcp]`) is a separate auth scheme + endpoint. Don't
   half-wire it under the `[mistral]` extra; ship as a separate
   `MistralVertexRuntime` if/when demand fires.
9. **`MistralError` carries `raw_response`.** Useful for debug
   logging but mustn't leak `raw_response.body` (could contain
   the request payload including user content) into airframe's
   public error messages. Sanitise at the boundary; debug-log the
   full thing.
10. **Conversation retention.** Server-side conversations linger
    per Mistral's retention policy (verify default; document).
    For privacy-sensitive flows, `MistralOptions.delete_on_close=
    True` lets the consumer opt in to explicit teardown. Default
    False because deletion is irreversible and most flows want
    history retention.
11. **Built-in tools opacity.** Web search / code interpreter /
    image generation fire entirely server-side; airframe sees
    only the synthesised tool-call events. Document that
    `runtime.session(tools=...)` doesn't enable these — they're
    configured on the agent definition.

## Definition of done

- All six iterations merged.
- `runtime_for("mistral")` returns `MistralRuntime` when the
  extra is installed; clean `ImportError` with
  `airframe-agents[mistral]` hint when not.
- `examples/probe_mistral.py` round-trips a structured-output
  prompt against live Mistral with `MISTRAL_API_KEY` set.
- `examples/probe_parity.py` includes `mistral` with no per-vendor
  conditionals; passes on a machine with credentials.
- `examples/probe_supports.py --provider mistral` shows the
  expected feature matrix.
- Conformance contract suite green against a mocked `Mistral`
  client; integration suite green against the live Agents API.
- `docs/adapters/mistral.md` published; README provider table +
  capability matrix updated.
- `docs/architecture.md` "Operational landmines" gains a Mistral
  subsection.
- CHANGELOG entry with iteration summary.
- `CLAUDE.md` reserved-IDs list includes `"mistral"`; reserves
  `"mistral-completions"` and `"mistral-vertex"`.

## When to start

**Phase 1 candidate** alongside `OpenCodeServerRuntime`,
`BedrockRuntime` (shipped), `GeminiRuntime`, and `KimiRuntime`.
Mistral is the highest-architectural-novelty of the planned
adapters — its server-managed-agent shape is new in the lineup
and the patterns it establishes will inform `BedrockAgentsRuntime`
later.

Reasonable cadence: one iteration per week, ~6 weeks end-to-end.
Iteration B is the heavy one (two-path complexity); plan extra
review time there. Mergeable in parallel with Kimi / OpenCode /
Gemini since they touch disjoint files.

Two triggers would make it Phase-1-priority work:

1. **A consumer asks for managed-agent capabilities** — handoffs,
   versioning, server-side connectors. Mistral is the only
   adapter in the planned lineup that offers these.
2. **An EU-jurisdictional consumer asks for European model-house
   coverage.** Data-residency or sovereignty constraints often
   rule out US-only adapters; Mistral is the obvious answer.

Until either fires, the workaround is `OpenRouterRuntime` against
`mistralai/mistral-large-...` model IDs — chat-only, no agentics,
but unblocks "I want to call a Mistral model" today.

## Open questions for the implementer

1. **`agents.complete()` vs `beta.conversations.*` — exact
   precedence and feature parity.** Verify which features work on
   which surface at Iteration B kickoff: structured output,
   streaming, tools, vision, file input. The adapter may have to
   forbid certain combinations (e.g. structured output may only
   work on `agents.complete()`).
2. **Default-agent resolution.** Should the adapter create a
   throwaway agent on first `execute()` when no `agent_id` is
   provided, or require explicit `MistralOptions(agent_id=...)`?
   Per Risk #3.
3. **Handoff event shape.** Surface handoffs as `RuntimeEvent`
   variants, `HookEvent`s, or both? ADR-worthy.
4. **Connector ↔ `McpServerRef` impedance.** Mistral connectors
   carry auth state (OAuth tokens, API keys) managed in the
   Mistral console. Airframe's `McpServerRef` carries auth too,
   but the assumed-good-token vs assumed-needs-OAuth posture
   differs. Iteration D decides whether to require pre-authorised
   connectors only.
5. **`MistralCompletionsRuntime` sibling timing.** A thin
   `OpenAICompatibleRuntime` subclass for the chat-completions
   endpoint is ~30 LOC. Worth shipping alongside `MistralRuntime`
   so "I want chat completions on Mistral" has a path that
   doesn't require the heavyweight Agents API. Track separately;
   not a blocker for `MistralRuntime`.

## Implementation wiring checklist

Beyond `src/airframe/adapters/mistral.py` itself, every new
adapter needs to touch these files. Closest sibling for the
full-bespoke shape is `bedrock.py` (HTTP client, typed envelope,
client-side or server-side state depending on path).

### Source wiring

- [ ] `src/airframe/discovery.py` — add `MistralRuntime` to
      `_builtin_runtime_classes()`.
- [ ] `src/airframe/__init__.py` — `from airframe.adapters.mistral
      import MistralRuntime` at module level + entry in `__all__`
      (alphabetical: between `KimiRuntime` (planned) /
      `CopilotRuntime` and `OpenAICompatibleRuntime`).
- [ ] `src/airframe/testing/contracts.py` — add `"mistral":
      MistralOptions` to the `matching` dict inside
      `_check_provider_options`.
- [ ] `src/airframe/testing/integration.py` — add `"mistral":
      ["MISTRAL_API_KEY"]` to `_PROVIDER_AUTH`.

### Probe + examples wiring

- [ ] `examples/probe_budget.py` — add `"mistral"` to the
      provider tuple.
- [ ] `examples/probe_parity.py` — picks up automatically. Add
      `AIRFRAME_PROBE_MODEL_MISTRAL` env hook for default-model
      override.

### Packaging

- [ ] `pyproject.toml` — new
      `mistral = ["mistralai[agents]>=2.4.5,<3"]` extra under
      `[project.optional-dependencies]`, AND add to the
      `all = [...]` list, AND add to the
      `[dependency-groups].test` list.

### Documentation

- [ ] `README.md` — provider table row, install one-liner.
- [ ] `docs/auth.md` — quick-reference table row + full
      `## MistralRuntime` section.
- [ ] `docs/reference.md` — adapter table row +
      `MistralRuntime` in the `__all__` snippet.
- [ ] `docs/adapters/mistral.md` — new page; mirror `bedrock.md`
      structure plus a dedicated section on the agent-vs-
      conversation distinction.
- [ ] `docs/capabilities.md` — add `mistral` column.
- [ ] `docs/architecture.md` — new `### Mistral Agents API`
      subsection under "Operational landmines" (Iteration F).
- [ ] `CLAUDE.md` — add `"mistral"` to canonical IDs; reserve
      `"mistral-completions"` and `"mistral-vertex"`.

### Test wiring

- [ ] `tests/test_mistral.py` — unit tests. Mirror
      `tests/test_bedrock.py` for structure (HTTP-client template).
- [ ] `tests/test_mistral_session.py` — session-class behaviour,
      SSE event translation, conversation vs agent path selection.
- [ ] `tests/test_mistral_conformance.py` — drives the shared
      contracts.
- [ ] `tests/test_mistral_integration.py` — pytest-marker-gated
      against live Mistral API.
- [ ] `tests/test_discovery.py` — update expected sets +
      filtered tests + third-party-discovery test.

### Issue + project housekeeping

- [ ] Open a tracking issue mirroring the Bedrock pattern.
- [ ] CHANGELOG entry at each iteration merge.

## Closest in-tree templates to read first

| File | What to learn from it |
|---|---|
| `src/airframe/adapters/bedrock.py` | **The primary template.** HTTP client over a typed envelope, async-context-manager session, error classification, in-tree pricing table. `MistralRuntime` mirrors this shape. ~700 LOC; Mistral target ~700 LOC because the agent-vs-conversation two-path adds complexity that Bedrock avoids. |
| `src/airframe/adapters/copilot.py` | Schema-fingerprint caching pattern (in case Mistral's strict-mode JSON schema rebuilds need similar caching). |
| `src/airframe/adapters/openai_compatible.py` | `_apply_provider_options()` helper pattern for merging vendor-specific knobs into the request. Mistral's `additional_request_fields` uses the same idea. |
| `src/airframe/sessions.py` | Shared helpers: `_enforce_budget_pre_turn`, `_check_provider_options`. |
| `dev-docs/opencode-adapter-plan.md` | The server-managed-conversation pattern. Mistral and OpenCode share the "session lives server-side, client tracks the ID" shape; OpenCode's plan documents the friction points (reset() doing real work, close()'s deletion semantics, SESSION_RESUME being natively first-class). |

## Naming reservations

Established with this plan:

- `"mistral"` — this adapter (Agents API; server-managed
  conversations).
- `"mistral-completions"` — **reserved** for a future
  `MistralCompletionsRuntime` (OpenAI-compat thin subclass
  fronting Mistral's chat-completions endpoint). Different
  surface, different shape, lower feature matrix — distinct
  provider ID per the established pattern.
- `"mistral-vertex"` — **reserved** for a future
  `MistralVertexRuntime` (Vertex AI deployment via
  `mistralai[gcp]`). Different auth (GCP ADC), different endpoint,
  different feature gating — distinct provider ID.
- `"mistralai"` — **not reserved**. Treated as a typo of
  `"mistral"`; `runtime_for("mistralai")` raises with a clear
  hint.

## First commit in a fresh session

A reasonable Iteration A first commit:

```
src/airframe/adapters/mistral.py            # new — Iteration A surface
src/airframe/discovery.py                   # +MistralRuntime in builtins
src/airframe/__init__.py                    # +export +__all__
pyproject.toml                              # +[mistral] extra
tests/test_mistral.py                       # new — identity, validate_binding, auth-chain unit tests
tests/test_discovery.py                     # +mistral in expected sets
CLAUDE.md                                   # +mistral; reserve mistral-completions + mistral-vertex
```

That should pass `make ci` cleanly with `Feature` flags all False
(no behaviour wired yet). After review, Iteration B adds
`execute()` (against `agents.complete()`) + `session()` (against
`beta.conversations.*`) + `stream()` + `cancel()` and flips the
first four `Feature` flags True.
