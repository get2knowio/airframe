# Phased implementation plan

Companion to [feature-roadmap.md](./feature-roadmap.md). The roadmap
identifies *what* should change; this doc orders the work so each
phase unblocks the next without locking the wrong shape in.

The dominant constraint is shape lock-in. Once a `Feature` enum
value, a `RuntimeEvent` variant, or a `ProviderOptions` field name
ships and third-party code branches on it, changing it is painful.
The plan therefore lands the *shape-defining* primitives in
**Phase 0** before any of the substantive feature work in
roadmap §3.

The current public surface is v0.2.0 (see
[CHANGELOG.md](../CHANGELOG.md)). Pre-1.0 is breaking-change tolerant
by semver convention, and airframe has one production consumer
(Maverick); this plan exploits both — most phases are minor bumps,
with one major bump scheduled when the protocol stabilises.

---

## Dependency graph

```
Phase 0 — Foundations (shape locks)
   │  Feature enum, ProviderOptions namespaces, unwrap,
   │  reasoning_tokens on CostRecord, entry-points, airframe-tck
   ▼
Phase 1 — AgentSession + streaming (the hinge)
   │  AgentSession protocol; stream(); cancellation
   ▼
Phase 2 — Inputs & reasoning
   │  thinking= kwarg; Prompt polymorphism (vision/file)
   ▼
Phase 3 — Function tools (Tier 1)
   │  tools= on AgentSession; tool-result round-trip
   ▼
Phase 4 — MCP server refs (Tier 2)            [gated on divergence signal]
   │
   ▼
Phase 5 — Permission, hooks, budget
   │  PermissionCallback; HookEvent observation; budget caps
   ▼
Phase 6 — Middleware + sandbox + subagents    [optional / signal-gated]
```

Why this ordering:

* **Phase 0 must precede Phase 1** because `AgentSession`'s
  constructor will accept `provider_options=` and its method
  signatures will be capability-gated. Defining `ProviderOptions`
  and `Feature` after the fact means rewriting the session API.
* **Phase 1 must precede Phase 2–6** because every later kwarg
  (`thinking=`, `tools=`, `mcp_servers=`, `on_permission=`) attaches
  to `AgentSession`, not `AgentRuntime`. Bolting them onto
  `execute()` directly inflates the runtime surface and forces a
  second migration later.
* **Phase 3 must precede Phase 4** because MCP server registration
  reuses the function-tool result-routing plumbing. Doing MCP first
  forces inventing a more abstract tool-call pipeline up front
  without a concrete user.
* **Phase 5 attaches to AgentSession** and is independent of 3/4 —
  could land in parallel.

---

## Phase 0 — Foundations

**Goal.** Lock the shape of the extension points before any
substantive feature work uses them. Every later phase relies on
these primitives existing.

**Scope.**

1. `Feature` enum + `runtime.supports(feature, model=None) -> bool`.
2. `ProviderOptions` tagged union + per-vendor dataclasses
   (`ClaudeOptions`, `CopilotOptions`, `CodexOptions`,
   `OpenAICompatOptions`) — initially nearly empty; populated as
   each adapter's vendor-specific knobs land in later phases.
3. `AgentRuntime.unwrap(cls: type[T]) -> T` — formalised escape
   hatch around what `RuntimeResult.raw` already provides.
4. `CostRecord.reasoning_tokens: int` — additive field; every
   adapter populates 0 until Phase 2 wires it.
5. Entry-point registration: third-party adapters discoverable via
   the `airframe.adapters` entry-point group, in addition to the
   built-in `airframe.adapters.*` module scan.
6. `airframe.testing` submodule — shared conformance contracts
   (idempotency, `supports()` ↔ `execute()` agreement, error
   classification, structured-output round-trip) that adapter
   authors run against their adapter via a `adapter_runtime`
   pytest fixture.

   Modelled on [SQLAlchemy's `testing.suite`](https://docs.sqlalchemy.org/en/20/dialects/) —
   a submodule of the main package, not a separate distribution.
   The "TCK / JCK" framing in §6.8 of the roadmap describes the
   *pattern*; the *implementation* uses Python's idiom for that
   pattern (shared test functions + a fixture name as the
   contract), not a Java-style separate-package conformance
   battery. Pytest is gated behind a new optional extra
   (`airframe-agents[testing]`); the main package doesn't depend
   on it. A separate `airframe-testing` PyPI distribution is
   deferred per §6.15 #6 until a third-party adapter author
   actually asks.

**Non-goals (deferred).**

* No new feature kwargs on `execute()`. The `Feature` enum's initial
  values describe what airframe *will* expose in Phases 1–5, but
  Phase 0 doesn't surface the corresponding APIs yet — only the
  capability flags so adapters and consumers can declare truth
  ahead of the API landing.
* No `AgentSession` yet. `execute()` keeps its current single-turn
  shape.
* No `ProviderOptions` *fields* beyond the empty dataclass scaffold
  — each subsequent phase fills its dataclass as the relevant
  feature lands.

**Public-API surface changes.**

```python
# airframe/features.py  (new)
from enum import Enum

class Feature(str, Enum):
    # Phase 1 will surface these:
    STREAMING = "streaming"
    SESSION_RESUME = "session_resume"
    CANCEL = "cancel"
    # Phase 2:
    REASONING_EFFORT = "reasoning_effort"
    REASONING_BUDGET_TOKENS = "reasoning_budget_tokens"
    VISION_INPUT = "vision_input"
    FILE_INPUT = "file_input"
    # Phase 3:
    TOOLS_FUNCTION = "tools_function"
    # Phase 4:
    TOOLS_MCP_STDIO = "tools_mcp_stdio"
    TOOLS_MCP_HTTP = "tools_mcp_http"
    TOOLS_MCP_IN_PROCESS = "tools_mcp_in_process"
    # Phase 5:
    PERMISSION_CALLBACK = "permission_callback"
    LIFECYCLE_HOOKS = "lifecycle_hooks"
    BUDGET_USD_CAP = "budget_usd_cap"
    BUDGET_TURN_CAP = "budget_turn_cap"
    # Phase 6:
    SANDBOX = "sandbox"
    SUBAGENTS = "subagents"
    # Structured output is already universal:
    STRUCTURED_OUTPUT_JSON_SCHEMA = "structured_output_json_schema"
    STRUCTURED_OUTPUT_STRICT = "structured_output_strict"

# airframe/protocol.py  (extended)
class AgentRuntime(Protocol):
    ...  # existing methods unchanged
    def supports(self, feature: Feature, model: ProviderModel | None = None) -> bool: ...
    def unwrap(self, cls: type[T]) -> T: ...

# airframe/options.py  (new)
@dataclass(frozen=True, slots=True)
class ClaudeOptions: ...      # empty for now
@dataclass(frozen=True, slots=True)
class CopilotOptions: ...
@dataclass(frozen=True, slots=True)
class CodexOptions: ...
@dataclass(frozen=True, slots=True)
class OpenAICompatOptions: ...

ProviderOptions = ClaudeOptions | CopilotOptions | CodexOptions | OpenAICompatOptions

# airframe/cost.py  (extended)
@dataclass(frozen=True, slots=True)
class CostRecord:
    ...  # existing fields unchanged
    reasoning_tokens: int = 0   # NEW, default 0 for back-compat
```

**Backward compatibility.**

* Adding methods to a `Protocol` is technically breaking for
  third-party implementations of `AgentRuntime` — but airframe has
  none yet outside the four built-ins. Acceptable.
* `CostRecord.reasoning_tokens: int = 0` is fully additive for
  consumers reading the field; the dataclass constructor gains an
  optional kwarg.
* `unwrap()` default implementation in a base mixin can return
  `self` when `isinstance(self, cls)` and raise `TypeError`
  otherwise; adapters override to expose their native client.

**Adapter migration.**

| Adapter | Work |
| --- | --- |
| `ClaudeCodeRuntime` | Implement `supports()` (returns True for most Phase 1–5 features per the matrix in §1 of the roadmap); implement `unwrap(ClaudeSDKClient)`. |
| `CopilotRuntime` | Same — `unwrap(CopilotClient)` / `unwrap(CopilotSession)`. |
| `CodexRuntime` | `unwrap(Codex)` / `unwrap(Thread)`. `supports(TOOLS_FUNCTION)` returns False. |
| `OpenAICompatibleRuntime` | `unwrap(AsyncOpenAI)`. `supports(TOOLS_MCP_*)` returns False unconditionally (Responses-only). |

Each adapter's `supports()` implementation is a static lookup table
keyed off `Feature` — no network calls, no SDK-version sniffing.

**Definition of done.**

* All four adapters report capability truth via `supports()`.
* `airframe.testing.contracts` ships the following test functions
  and every in-tree adapter has a conformance test file that
  imports them, provides an `adapter_runtime` fixture, and passes:
  - `close()` is idempotent.
  - `validate_binding(b) == True` ⇒ `execute(model=b, ...)` does not
    raise `UnsupportedBindingError`.
  - `supports(STRUCTURED_OUTPUT_JSON_SCHEMA)` ⇒ schema round-trip
    succeeds against a canonical Pydantic model.
  - 401 from the vendor ⇒ `RuntimeAuthError`.
  - Successful call ⇒ `CostRecord.input_tokens > 0`.
  - `unwrap(self.__class__)` returns `self`.
* README documents how to register a third-party adapter via the
  `airframe.adapters` entry-point group.
* New `examples/probe_supports.py` lists every adapter × feature
  combination.

**Version target.** `v0.3.0` (minor bump — pre-1.0, but the new
protocol methods are large enough to warrant signalling).

**External-contributor unblocks.** This is the key phase. Once
entry-point discovery + TCK ship, third-party adapter packages
(`airframe-adapters-together`, `-groq`, `-fireworks`, ...) can land
without core changes. **Phase 0 is the unblocker; nothing later is.**

**Gating / irreversible decisions.** ⚠️ Two shapes lock here:

1. **The `Feature` enum string values.** Once consumer code branches
   on `Feature.REASONING_EFFORT`, renaming it requires a major
   bump. Review carefully; prefer descriptive over short.
2. **The `ProviderOptions` tagged-union shape.** Choosing one
   dataclass per provider (vs. a single dict, vs. mixed
   inheritance) is the biggest API decision in the whole plan.
   Worth a written ADR.

The `airframe.testing` contracts are also sticky — once contributors
target them, removing a contract is fine but tightening one isn't.

---

## Phase 1 — `AgentSession` + streaming

**Goal.** The hinge phase. Establish the
factory → session → operation hierarchy (roadmap §6.6) and the
streaming event taxonomy (roadmap §3 P0). Every later phase attaches
kwargs to `AgentSession`, not `AgentRuntime`.

**Scope.**

1. `AgentSession` protocol.
2. `AgentRuntime.session(*, resume=None, system=None, model=None,
   provider_options=None) -> AgentSession` factory.
3. `AgentSession.execute()` / `AgentSession.stream()` /
   `AgentSession.cancel()` / `AgentSession.close()`.
4. `RuntimeEvent` discriminated union for streaming (`TextDelta`,
   `ReasoningDelta`, `ToolCallStart`, `ToolCallResult`,
   `TurnComplete`).
5. `AgentRuntime.execute(...)` becomes documented sugar for
   `runtime.session().execute(...).` + `close()`. Behaviour
   identical to today — single-turn, ephemeral.

**Non-goals (deferred).**

* No tools, no vision, no thinking kwargs yet. Those land in
  Phases 2–3 as additive `session.execute()` / `runtime.session()`
  kwargs.
* No multi-session orchestration. One session at a time per runtime.

**Public-API surface changes.**

```python
# airframe/events.py  (new)
@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str

@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    text: str

@dataclass(frozen=True, slots=True)
class ToolCallStart:
    tool_name: str
    tool_call_id: str
    arguments_preview: str

@dataclass(frozen=True, slots=True)
class ToolCallResult:
    tool_call_id: str
    output: Any
    is_error: bool

@dataclass(frozen=True, slots=True)
class TurnComplete:
    result: RuntimeResult

RuntimeEvent = TextDelta | ReasoningDelta | ToolCallStart | ToolCallResult | TurnComplete

# airframe/protocol.py  (extended)
class AgentSession(Protocol):
    id: str | None

    async def execute(
        self,
        prompt: str,
        *,
        schema: type[BaseModel] | None = None,
        timeout: float = 600.0,
    ) -> RuntimeResult: ...

    async def stream(
        self,
        prompt: str,
        *,
        schema: type[BaseModel] | None = None,
        timeout: float = 600.0,
    ) -> AsyncIterator[RuntimeEvent]: ...

    async def cancel(self) -> None: ...
    async def close(self) -> None: ...

class AgentRuntime(Protocol):
    ...
    def session(
        self,
        *,
        resume: str | None = None,
        system: str | None = None,
        model: ProviderModel | None = None,
        provider_options: ProviderOptions | None = None,
    ) -> AgentSession: ...
```

**Backward compatibility.**

* `AgentRuntime.execute()` keeps its signature; internally it
  becomes `await self.session(system=..., model=...).execute(...)`
  followed by `close()`. The existing `reset()` semantics map to
  "discard the current ad-hoc session"; behaviour is preserved.
* `RuntimeResult` is unchanged.
* `RuntimeEvent`'s union is *additive over time*. Adding a new
  variant is safe (consumers do `match event:` with a wildcard or
  `isinstance`); removing one is breaking.

**Adapter migration.**

| Adapter | Session implementation |
| --- | --- |
| `ClaudeCodeRuntime` | `AgentSession` wraps the `ClaudeSDKClient` lifecycle that currently lives inside `_ensure_client`. `resume=` maps to `ClaudeAgentOptions.resume`. Streaming via `include_partial_messages=True` + `receive_messages()` filtering. `cancel()` calls `client.interrupt()`. |
| `CopilotRuntime` | Session wraps `CopilotSession`. `resume=` maps to `client.resume_session()`. Streaming via `session.on(handler)` filtering for `ASSISTANT_MESSAGE_DELTA` / `ASSISTANT_REASONING_DELTA`. `cancel()` calls `session.abort()`. |
| `CodexRuntime` | Session wraps `Thread`; lifecycle is one subprocess per `execute()` regardless. `resume=` maps to `client.resume_thread(thread_id)`. Streaming via `thread.run_streamed()`. `cancel()` triggers `AbortController.abort()`. |
| `OpenAICompatibleRuntime` | Session keeps a client-side `messages=[]` buffer. `resume=` raises `UnsupportedFeatureError` for any non-falsy value (chat-completions vendors); subclasses backed by Responses can override. Streaming via `stream=True`. `cancel()` cancels the underlying `asyncio.Task`. |

**Definition of done.**

* All four adapters implement `AgentSession`.
* `AgentRuntime.execute()` keeps existing semantics — verified by
  re-running the existing `probe_*.py` scripts unchanged.
* TCK adds: streaming contract (yields ≥1 event ending in
  `TurnComplete`); session-resume round-trip for adapters that
  declare `Feature.SESSION_RESUME`; cancellation propagates within
  100 ms of `cancel()`.
* New examples: `examples/probe_streaming.py`,
  `examples/probe_session_resume.py`.
* Architecture doc updated with the new staircase diagram.

**Version target.** `v0.4.0` (minor; substantial new surface but
pre-1.0).

**External-contributor impact.** Third-party adapters now have a
larger surface to implement. The TCK additions are the contract.

**Gating / irreversible decisions.** ⚠️ Three:

1. **`RuntimeEvent` discriminated union.** Adding variants later is
   safe; getting the *current* variants wrong (e.g. omitting
   `ToolCallStart.arguments_preview`, or merging
   reasoning/text deltas) forces consumer rewrites. Worth an ADR.
2. **`AgentSession.id` is `Optional[str]`.** That's the right call
   (OAI-compat has no server-side id), but consumers must learn
   to treat it as a hint, not a key.
3. **Single-session-per-runtime vs. concurrent.** This plan
   recommends single-session-per-runtime in v0.4. Going
   concurrent later is additive (new `session()` returning a new
   handle); going from concurrent to single is breaking.

---

## Phase 2 — Inputs and reasoning

**Goal.** Land the two roadmap §3 P2 features that are
straightforward kwarg additions once `AgentSession` exists.

**Scope.**

1. `thinking: ThinkingMode | None` kwarg on `session.execute()` /
   `session.stream()`.
2. Polymorphic `prompt: str | list[PromptPart]` where `PromptPart`
   is `str | ImageInput | FileInput`.
3. Adapter mapping per the matrix in §1 of the roadmap.
4. Populate `CostRecord.reasoning_tokens` (groundwork laid in
   Phase 0).

**Non-goals (deferred).**

* No audio inputs. OpenAI-only; not justified yet.
* No batch image upload (Claude file upload API not yet in the
  agent SDK). Per-call only.

**Public-API surface changes.**

```python
# airframe/inputs.py  (new)
@dataclass(frozen=True, slots=True)
class ImageInput:
    path: str | None = None
    bytes_: bytes | None = None
    url: str | None = None
    media_type: str | None = None

@dataclass(frozen=True, slots=True)
class FileInput:
    path: str
    media_type: str | None = None

PromptPart = str | ImageInput | FileInput
Prompt = str | list[PromptPart]

# airframe/thinking.py  (new)
ReasoningEffort = Literal["minimal", "low", "medium", "high"]
ThinkingMode = (
    None
    | ReasoningEffort
    | dict          # {"budget_tokens": int}
    | Literal["disabled"]
)

# AgentSession (extended)
async def execute(
    self,
    prompt: Prompt,
    *,
    schema: type[BaseModel] | None = None,
    thinking: ThinkingMode = None,
    timeout: float = 600.0,
) -> RuntimeResult: ...
```

**Backward compatibility.** Both kwargs are additive with safe
defaults. `prompt: str` continues to work everywhere.

**Adapter migration.**

* Claude / Copilot / Codex / OpenAI-compat each map
  `thinking=` to their native field per roadmap §3 P2.
* Each declares the intersection (`low/medium/high`) via
  `supports(REASONING_EFFORT)`; Claude additionally supports
  `REASONING_BUDGET_TOKENS`.
* For `minimal` on adapters that don't have it (Claude, Copilot),
  the adapter coerces to `low` with a debug-level log — NOT
  silently. Consumers using `runtime.supports(REASONING_EFFORT,
  model)` aren't surprised.
* Image input: Claude routes through Read tool (auto-allowed for
  prompt-attached paths); Copilot → `FileAttachment`; Codex →
  `LocalImageInput`; OpenAI-compat → content parts.

**Definition of done.**

* TCK adds: `thinking=` round-trip on every adapter declaring
  `REASONING_EFFORT`; image-input round-trip on every adapter
  declaring `VISION_INPUT`.
* `CostRecord.reasoning_tokens > 0` after a thinking-enabled call on
  Claude / Codex / OpenAI-compat with reasoning model.
* Examples: `examples/probe_thinking.py`,
  `examples/probe_vision.py`.

**Version target.** `v0.5.0` (minor).

**External-contributor impact.** New adapters now have a defined
shape for the most-common per-vendor knobs. ProviderOptions tables
gain their first real fields (e.g.
`ClaudeOptions.thinking_budget_tokens` for the explicit-budget
variant; the literal-effort `minimal/low/medium/high` stays on the
portable kwarg).

**Gating / irreversible decisions.** ⚠️ One:

1. **`ThinkingMode` union shape.** Specifically the choice to put
   `budget_tokens` inside an inline dict rather than a dataclass.
   Inline dict matches Anthropic's wire format and is what consumers
   already write; a dataclass would be slightly more type-safe but
   more verbose. Recommend: dict literal for portability, document
   the keys, accept the trade-off.

---

## Phase 3 — Function tools (Tier 1)

**Goal.** Roadmap §3 P2 "Tool / MCP server registration" — the
Tier-1 (function-tools) half. Land this before MCP because the
tool-result round-trip plumbing this builds is what MCP reuses.

**Scope.**

1. `FunctionTool` dataclass.
2. `tools: list[FunctionTool] | None` kwarg on
   `runtime.session(...)`.
3. Tool-result round-trip across all four adapters (with
   `CodexRuntime` raising `UnsupportedFeatureError` on tool
   registration — its Python SDK has no tool API; tools must be
   wired into the Codex CLI config externally).
4. `ToolCallStart` / `ToolCallResult` events on `stream()` reflect
   tool calls in flight.

**Non-goals (deferred).**

* No MCP server refs (Phase 4).
* No remote tool execution. The tool's `handler` runs in the
  caller's process.
* No tool-choice / parallel-tool-calls knobs yet. Wait for a
  concrete user before exposing.

**Public-API surface changes.**

```python
# airframe/tools.py  (new)
@dataclass(frozen=True, slots=True)
class FunctionTool:
    name: str
    description: str
    params: type[BaseModel]
    handler: Callable[[BaseModel], Awaitable[Any]]

# AgentRuntime.session (extended)
def session(
    self,
    *,
    resume: str | None = None,
    system: str | None = None,
    model: ProviderModel | None = None,
    tools: list[FunctionTool] | None = None,
    provider_options: ProviderOptions | None = None,
) -> AgentSession: ...
```

**Backward compatibility.** Additive kwarg. Sessions without
`tools=` behave identically to today.

**Adapter migration.**

| Adapter | Tool implementation |
| --- | --- |
| `ClaudeCodeRuntime` | Build an in-process MCP server via `create_sdk_mcp_server(...)` + `@tool` from each `FunctionTool`. |
| `CopilotRuntime` | Translate each `FunctionTool` to `define_tool(name, description, handler, params_type)`. |
| `CodexRuntime` | `supports(TOOLS_FUNCTION)` returns False; passing `tools=` raises `UnsupportedFeatureError`. |
| `OpenAICompatibleRuntime` | Pass `tools=[{"type":"function","function":{...}}]` to chat completions; implement the round-trip (capture `tool_calls` in response, invoke handler, append `role="tool"` message, re-call). |

The OpenAI-compat case is the most complex — multi-turn within a
single `execute()` call. Done well, it transparently supports
parallel tool calls (the chat completions API returns them all in
one assistant message).

**Coexistence with forced-`submit_result` structured output on
Copilot.** When both `schema=` and `tools=` are passed to
`CopilotRuntime`, the adapter prepends `submit_result` to the tool
list — same forcing pattern as today, just sharing the slot.

**Definition of done.**

* TCK adds: tool round-trip on every adapter declaring
  `TOOLS_FUNCTION`; parallel-tool-call handling on adapters where it
  works.
* `RuntimeEvent.ToolCallStart` / `ToolCallResult` fire correctly on
  the streaming path.
* Examples: `examples/probe_tools.py` with a simple calculator
  tool.

**Version target.** `v0.6.0` (minor).

**Gating / irreversible decisions.** ⚠️ One:

1. **`FunctionTool.handler` shape.** Specifically: receives the
   parsed `BaseModel` instance vs. raw dict, returns `Any` vs. a
   typed result wrapper. Recommend: typed-in (the model the schema
   produced), `Any`-out with the convention that the return value
   is JSON-serialisable. Matches Copilot's `define_tool` shape and
   LangChain's `tool` decorator.

---

## Phase 4 — MCP server refs (Tier 2)

**Goal.** Roadmap §3 P2 "Tool / MCP server registration" — the
Tier-2 half.

**Status: signal-gated.** This phase lands only when the divergence
signal in the next paragraph fires.

**Divergence signal.** Today: Claude (in-process + stdio + sse +
http), Copilot (stdio + http). Codex has MCP plumbing only via CLI
config — not Python SDK. OpenAI-compat is Responses-only and the
`OpenAICompatibleRuntime` family is Chat Completions. So airframe
can either:

(a) Ship MCP refs with two adapters fully supporting it and the
other two declining via `supports()`, OR

(b) Wait until at least one of Codex / OpenAI-compat gains
first-class MCP, on the theory that two-of-four is too low a
coverage ratio.

**Recommendation: ship (a) once a consumer asks.** Two-of-four is
fine when those two are the most-MCP-adopted of the four. The
capability-gate pattern makes the asymmetry honest.

**Scope (when triggered).**

```python
# airframe/tools.py  (extended)
@dataclass(frozen=True, slots=True)
class McpServerRef:
    name: str
    transport: Literal["stdio", "http", "sse"]
    command: list[str] | None = None
    url: str | None = None
    headers: dict[str, str] | None = None
    auth_token: str | None = None

# AgentRuntime.session (extended)
def session(
    self,
    *,
    ...,
    mcp_servers: list[McpServerRef] | None = None,
) -> AgentSession: ...
```

**Adapter migration.**

| Adapter | MCP implementation |
| --- | --- |
| `ClaudeCodeRuntime` | Translate each `McpServerRef` to the appropriate `McpStdioServerConfig` / `McpHttpServerConfig` / `McpSSEServerConfig` and pass via `ClaudeAgentOptions.mcp_servers`. |
| `CopilotRuntime` | Translate to `MCPStdioServerConfig` / `MCPHTTPServerConfig` and pass via `create_session(mcp_servers=...)`. |
| `CodexRuntime` | `supports(TOOLS_MCP_*)` returns False; `mcp_servers=` raises `UnsupportedFeatureError`. |
| `OpenAICompatibleRuntime` | Same — raises. A future `OpenAIResponsesRuntime` (separate from the compat family) could translate to the Responses-API `{"type":"mcp",...}` tool shape. |

**Definition of done.**

* TCK adds: a stdio MCP server round-trip (e.g. with a fixture
  echo-server) on every adapter declaring `TOOLS_MCP_STDIO`.
* Example: `examples/probe_mcp.py` using a public MCP server.

**Version target.** `v0.7.0` (minor) if (a); deferred if (b).

**Gating / irreversible decisions.** ⚠️ One:

1. **`McpServerRef` field set.** Hard to extend later if a vendor
   adds a transport we didn't model (e.g. websocket). Recommend:
   ship the three common transports + a `provider_options=` dict
   for transport-specific knobs (`stdio.env`, `http.timeout`, etc.)
   to keep the dataclass small.

---

## Phase 5 — Permission, hooks, budget

**Goal.** Roadmap §3 P3 items. Independent of Phase 3/4 — could land
in parallel.

**Scope.**

1. `PermissionCallback` protocol attached to `runtime.session(...)`.
2. `HookEvent` typed union + `on_event` callback on
   `runtime.session(...)`.
3. `max_turns: int | None` and `max_budget_usd: float | None`
   kwargs on `session.execute()` / `session.stream()`.

**Non-goals (deferred).**

* No transcript export / import. Use `unwrap(NativeClient)` to reach
  Claude's `session_store` for now.
* No subagent definitions (Phase 6).

**Public-API surface changes.**

```python
# airframe/permission.py  (new)
@dataclass(frozen=True, slots=True)
class PermissionRequest:
    tool_name: str
    tool_args: dict[str, Any]
    reason: str | None

PermissionDecision = Literal["allow", "deny", "defer"]

class PermissionCallback(Protocol):
    async def handle(self, request: PermissionRequest) -> PermissionDecision: ...

# airframe/hooks.py  (new)
@dataclass(frozen=True, slots=True)
class HookEvent:
    kind: Literal[
        "session_start", "session_end",
        "user_prompt_submit",
        "pre_tool_use", "post_tool_use", "tool_failure",
        "pre_compact", "rate_limit",
    ]
    session_id: str | None
    payload: dict[str, Any]

# AgentRuntime.session (extended)
def session(
    self,
    *,
    ...,
    on_permission: PermissionCallback | None = None,
    on_event: Callable[[HookEvent], None] | None = None,
) -> AgentSession: ...

# AgentSession.execute / stream (extended)
async def execute(
    self,
    prompt: Prompt,
    *,
    ...,
    max_turns: int | None = None,
    max_budget_usd: float | None = None,
) -> RuntimeResult: ...
```

**Backward compatibility.** Additive. Sessions without these
callbacks behave identically.

**Adapter migration.**

| Adapter | Permission wiring | Hook wiring | Budget wiring |
| --- | --- | --- | --- |
| `ClaudeCodeRuntime` | `can_use_tool=` on `ClaudeAgentOptions`. | `hooks=` on options + `include_hook_events=True`. | `max_turns` + `max_budget_usd` native. |
| `CopilotRuntime` | `on_permission_request=` (mandatory). | `SessionHooks` typed dict + `session.on(...)` filtering. | No native — adapter tracks cumulative cost and raises if exceeded; `max_turns` no-op. |
| `CodexRuntime` | `approval_policy=` (coarser; map allow/deny → `never`/`untrusted`). | Synthesize from `ItemStartedEvent` / `ItemCompletedEvent`. | Same client-side enforcement. |
| `OpenAICompatibleRuntime` | `supports(PERMISSION_CALLBACK)` returns False; `on_permission=` is ignored with a debug log. | Synthesize from chat-completions tool-call rounds. | Client-side `max_budget_usd`; native `max_tokens` for `max_turns` (loose mapping). |

**Definition of done.**

* TCK adds: permission callback fires on every adapter declaring
  `PERMISSION_CALLBACK`; hook events fire in expected order on every
  adapter declaring `LIFECYCLE_HOOKS`; budget cap aborts a
  long-running session at the threshold.
* Examples: `examples/probe_permission.py`,
  `examples/probe_budget.py`.

**Version target.** `v0.8.0` (minor).

**Gating / irreversible decisions.** ⚠️ Two:

1. **`HookEvent.kind` enum string values.** Same shape lock as
   `Feature` — see Phase 0.
2. **`PermissionDecision = Literal["allow", "deny", "defer"]`.**
   Adding new decisions later is fine; renaming existing ones is
   breaking.

---

## Phase 6 — Middleware, sandbox, subagents

**Goal.** Roadmap §3 P4 + §6.9 (middleware). The "everything else"
phase. Each item is independent and signal-gated.

**Scope (each item ships when its signal fires).**

1. **`RuntimeMiddleware` protocol** + `with_middleware(runtime,
   [...])`. Signal: a second consumer asks for tracing or retry
   policy that's not Maverick-specific.
2. **`cwd` / `sandbox` / `network_access` kwargs** on
   `runtime.session(...)`. Signal: a consumer with a real sandbox
   need (e.g., untrusted prompt processing). Three of four adapters
   support this natively; capability gate handles OpenAI-compat
   refusal.
3. **Subagent definitions** via `agents: dict[str, AgentDefinition]`
   on `runtime.session(...)`. Signal: a consumer wants programmatic
   subagents portably (only Claude + Copilot support today;
   two-of-four; defer until a third joins).

**Non-goals (likely permanent).**

* Batch APIs, fine-tuning, audio/TTS/STT, file checkpointing,
  external session storage — see roadmap §3 "Out of scope."

**Version target.** Each lands as a minor bump (`v0.9.0`,
`v0.10.0`, ...) when its signal fires.

---

## Phase 7+ — Long-tail

Items that *might* land later but don't have a clear shape today:

* **Direct API adapters.** `AnthropicRuntime` for the Messages API;
  `OpenAIRuntime` for the Responses API directly. Provider IDs
  `"anthropic"` and `"openai"` are already reserved in the README
  for these. Both unblock MCP for the OpenAI side and avoid the
  CLI-subscription auth chain for Claude.
* **`airframe-spec` distribution split.** Promote the protocol +
  types to a zero-dependency package once a library consumer is
  bottlenecked. Until then, the pip-extra model works.
* **Concurrent sessions per runtime.** Today `runtime.session()`
  returns one active at a time; concurrency is a future
  refactoring once a real consumer needs it.

---

## When to cut v1.0

Recommend cutting `v1.0.0` after **Phase 5** lands, with a stability
freeze on the surface shipped through it. Rationale:

* `Feature` enum and `ProviderOptions` namespaces have been
  exercised by five phases of feature work — their shape is proven.
* `AgentSession` has been the hinge for tools, vision, thinking,
  permission, and hooks — it has earned a 1.0 contract.
* TCK has accumulated meaningful contracts across permission,
  streaming, tools, structured output, error classification — third
  parties have a real spec to target.

Subsequent phases (6, 7+) land as `v1.x` minors per their additive
nature.

---

## Summary table — phases at a glance

| Phase | Theme | Version | External unblocks | Major shape-lock decisions |
| --- | --- | --- | --- | --- |
| 0 | Foundations | v0.3.0 | ⚡ **Third-party adapters** (entry-points + TCK) | `Feature` enum values; `ProviderOptions` shape |
| 1 | `AgentSession` + streaming | v0.4.0 | Streaming consumers | `RuntimeEvent` union; session-id optional |
| 2 | Inputs + reasoning | v0.5.0 | Vision/reasoning consumers | `ThinkingMode` shape |
| 3 | Function tools | v0.6.0 | Tool-using consumers | `FunctionTool.handler` shape |
| 4 | MCP refs (signal-gated) | v0.7.0 | MCP consumers | `McpServerRef` field set |
| 5 | Permission, hooks, budget | v0.8.0 | Agentic / audit consumers | `HookEvent.kind` values; `PermissionDecision` literals |
| — | **v1.0 cut** | **v1.0.0** | Spec-stable contract | Surface freeze |
| 6+ | Middleware, sandbox, subagents | v1.x | Signal-gated | n/a |

---

## Cross-cutting principles

These apply to every phase:

1. **Additive over breaking.** Every new kwarg is optional with a
   safe default. Every new method has a default
   implementation in a shared base class where possible.
2. **Capability-gate everything.** A new feature lands with its
   `Feature.X` enum value, every adapter declares its support
   truthfully, the TCK contract verifies the declaration matches
   behaviour.
3. **No silent fallbacks.** A capability declined ⇒ a clear
   `UnsupportedFeatureError`, never "best effort succeed."
4. **No vendor-specific field on canonical types.** `CostRecord`,
   `RuntimeResult`, `RuntimeEvent` stay canonical; vendor specifics
   live in `ProviderOptions` or behind `unwrap()`.
5. **The TCK grows with every phase.** A feature without a TCK
   contract is shipped half-done — third parties can't verify
   parity.
6. **Match the JDBC discipline on escape hatches.** `unwrap()` is
   the documented way to reach vendor-specific behaviour; it's
   neither hidden nor encouraged. Consumers paying the abstraction
   tax get portability; consumers wanting vendor power get
   one-call access without `# type: ignore`.

---

## Open questions / ADRs to write

These deserve written decisions before the corresponding phase
lands, not in-PR debate:

1. **ADR-001 (Phase 0):** `ProviderOptions` tagged-union shape —
   one dataclass per provider vs. inheritance vs. dict-based.
2. **ADR-002 (Phase 0):** Whether `Feature` enum members map 1:1
   to method/kwarg names, or are semantically independent. (Recommend
   1:1 — easier consumer mental model.)
3. **ADR-003 (Phase 1):** `RuntimeEvent` variant set — final
   field-by-field shapes before any consumer ships.
4. **ADR-004 (Phase 1):** `AgentSession` concurrency model —
   single-active vs. multi-active per runtime.
5. **ADR-005 (Phase 3):** `FunctionTool` parameter/return shape —
   Pydantic in / JSON-serialisable out vs. raw dict both ways.
6. **ADR-006 (Phase 4 trigger):** MCP refs ship gate — write the
   "two of four is enough" rationale or the "wait for three"
   rationale explicitly so the bar isn't relitigated in every PR.
