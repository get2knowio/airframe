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
Phase 4 — MCP server refs (Tier 2)            [✅ shipped]
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

7. **Plain-text `execute(schema=None)` across all built-in adapters**
   — closes a v0 contract gap where the
   `AgentRuntime.execute()` docstring promised
   "`None` means plain text — text answer on `RuntimeResult.text`,
   `structured=None`" but `ClaudeCodeRuntime`,
   `CopilotRuntime`, and `CodexRuntime` refused with
   `NotImplementedError`. `OpenAICompatibleRuntime` (and therefore
   `OpenCodeZenRuntime`) already honoured the contract — only the
   conformance test ratifying it is new for that adapter.

   **Motivation.** A downstream consumer codebase (Maverick) just
   migrated five long-running personas onto airframe. Each grew a
   single-field Pydantic schema (`Payload(text: str)`) purely to
   satisfy the `schema is None` gate — markdown summaries, free-form
   analyses, agents that write files via tools and only need a
   "done" signal all paid the schema-wrapper tax. Once the gate is
   gone, those wrappers vanish and personas call
   `runtime.execute(prompt, system=PERSONA_SYSTEM_PROMPT)` directly.

   **Per-adapter implementation notes:**

   * `ClaudeCodeRuntime` — don't pass `output_format` on
     `ClaudeAgentOptions` when `schema is None`.
     `ResultMessage.result` already carries the concatenated final
     assistant text. The `_ensure_client` cache key uses the literal
     sentinel `"__plain_text__"` so plain-text and structured
     sessions don't collide on `(model, system)`.
   * `CopilotRuntime` — don't register the `submit_result` tool
     when `schema is None`; don't prepend the forced-tool prefix to
     the system message. The final `AssistantMessageData.content`
     event lands on `RuntimeResult.text`. Caller-supplied
     `system=` passes through verbatim — load-bearing for
     downstream personas.
   * `CodexRuntime` — omit `outputSchema` from `TurnOptions` when
     `schema is None`. `turn.final_response` is the free-form text.
     Empty `final_response` is a legitimate outcome (tool-only
     turn that wrote files and stopped) rather than a
     structured-output violation.
   * `OpenAICompatibleRuntime` — already correct (the body short-
     circuits on `schema is not None` before building
     `response_format`); only the conformance test is added.

   **Conformance.** `airframe.testing.contracts` gains
   `test_plain_text_execute_path_is_wired` — a structural check that
   the `execute()` signature accepts `schema=None` (default) and the
   implementation doesn't carry the historical
   `"plain-text execute() is not wired in v0"` sentinel. Every
   in-tree adapter's conformance file imports it and passes. A
   behavioural variant validating the actual returned text shape
   needs live vendor credentials and lives in
   `airframe.testing.integration` (deferred to v0.4.0 alongside the
   other network-required contracts).

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
* All four adapters honour the protocol's plain-text contract:
  `execute(prompt, schema=None)` returns a `RuntimeResult` with
  `text` populated and `structured=None`; `system=` forwarded to
  the vendor verbatim; `persona=` accepted (no crash) regardless
  of whether the adapter consumes it.
* `airframe.testing.contracts` ships the following test functions
  and every in-tree adapter has a conformance test file that
  imports them, provides an `adapter_runtime` fixture, and passes:
  - `close()` is idempotent.
  - `unwrap(type(self))` returns `self`.
  - `unwrap(<unrelated>)` raises `TypeError`.
  - `supports()` returns `bool` for every `Feature` member.
  - `supports()` is idempotent.
  - `supports(STRUCTURED_OUTPUT_JSON_SCHEMA)` is `True`.
  - `supports(feature, model=binding)` accepts the model kwarg.
  - `validate_binding(binding)` returns `bool`; rejects foreign
    provider IDs.
  - `execute()` signature accepts `schema=None` (default) and the
    plain-text path is wired (no legacy
    `"plain-text execute() is not wired in v0"` gate).
* README documents how to register a third-party adapter via the
  `airframe.adapters` entry-point group.
* New `examples/probe_supports.py` lists every adapter × feature
  combination.

**Deferred to `airframe.testing.integration` (v0.4.0).** The
behavioural contracts that require live vendor credentials —
schema round-trip succeeds, 401 ⇒ `RuntimeAuthError`, successful
call ⇒ `CostRecord.input_tokens > 0`, plain-text round-trip
returns non-empty `text` — naturally co-locate with Phase 1's
streaming/multi-turn integration test infrastructure. The
existing `examples/probe_*.py` scripts already exercise these
end-to-end against real vendors.

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

### Iteration breakdown

Phase 3 lands in four iterations, each ending with a `mise run check`-green
stopping point. Same shape as Phase 2 Iterations A–D. Status: **Phase
3 complete** on the `phase-3-function-tools` branch — Iterations A
through D all green.

#### Iteration A — Protocol scaffolding (no behaviour) ✅

Lock the public surface; defer the wiring.

- New module `airframe/tools.py` with `FunctionTool` dataclass
  (frozen+slots: `name`, `description`, `params: type[BaseModel]`,
  `handler: Callable[[BaseModel], Awaitable[Any]]`).
- Extend `AgentRuntime.session(...)` Protocol and every adapter's
  `session()` method with `tools: list[FunctionTool] | None = None`.
- Shared helper `airframe.sessions._check_tools_supported(tools, *,
  adapter_label, feature_supported)` raises
  `UnsupportedFeatureError(feature=Feature.TOOLS_FUNCTION)` on a
  non-None list while the adapter's capability flag is False.
- Every adapter's `session()` calls the helper at the top.
- Top-level export: `FunctionTool`.
- `TurnComplete` docstring clarified: one per *user turn*, not per
  *model turn* (tool round-trips produce intermediate
  `ToolCallStart`/`ToolCallResult` pairs but only one trailing
  `TurnComplete`).
- Tests: `tests/test_tools.py` pinning the dataclass shape;
  cross-adapter feature-matrix tests confirming `TOOLS_FUNCTION` is
  False everywhere and `session(tools=...)` raises everywhere.

**Stopping point.** `tools=` accepted by every adapter's `session()`
signature; non-None raises immediately. No per-adapter behaviour yet.

#### Iteration B — Wire OpenAI-compat (the hard one first) ✅

Sets the canonical event-emission pattern; client-side tool-loop is
the most complex piece in Phase 3.

- `_translate_tools_for_openai(tools) → list[dict]` helper —
  `FunctionTool` → `{"type":"function","function":{"name":..,
  "description":..,"parameters":<json_schema>}}`.
- `OpenAICompatibleSession`: cache tools at construction; pass
  `tools=` to every `chat.completions.create()`.
- Tool-loop in `_do_execute`: when response has `tool_calls`,
  invoke each handler with parsed `params`, append `role="tool"`
  messages with JSON-serialised returns, re-call. Cap at
  `MAX_TOOL_ITERATIONS=20` (raise `RuntimeProtocolError` on cap —
  runaway agents are a real failure mode worth surfacing).
- Tool-loop in `stream()`: detect `delta.tool_calls`, accumulate
  the `arguments` JSON across chunks, emit `ToolCallStart` with
  `arguments_preview`, run handler, emit `ToolCallResult`, append
  the message, continue. One `TurnComplete` at the very end.
- Handler errors → `ToolCallResult(is_error=True, output=<repr>)`
  and a `role="tool"` message so the model can recover.
- Flip `Feature.TOOLS_FUNCTION` True on `OpenAICompatibleRuntime`.
- Tests: round-trip with one tool, parallel tool calls (multiple
  in one assistant message), handler raises → propagates, iteration
  cap.

**Stopping point.** OpenAI-compat fully wired. Claude / Copilot
still raise on `tools=`.

#### Iteration C — Wire Claude + Copilot (SDK does the dispatch) ✅

Both vendors handle tool dispatch *inside the SDK*. No client-side
loop — register the tools at session-creation, let the SDK invoke
handlers, translate the SDK's tool events into airframe events.

- **Claude:** `_translate_tools_for_claude(tools) → mcp_server_config`.
  Build an in-process MCP server via
  `claude_agent_sdk.create_sdk_mcp_server(...)`; wrap each
  `FunctionTool` with the SDK's `@tool` decorator. Pass via
  `ClaudeAgentOptions.mcp_servers={...}`. Detect `ToolUseBlock` in
  `receive_response()` → emit `ToolCallStart`; matching
  `ToolResultBlock` → emit `ToolCallResult`. Tools enter the
  existing `_ensure_client` cache key (a tools-fingerprint
  fragment) so a tools-change forces reconnect — symmetric with how
  `thinking` and `has_attachments` work.
- **Copilot:** `_translate_tools_for_copilot(tools) → list[copilot.Tool]`.
  Each `FunctionTool` becomes a `define_tool(name, description,
  handler, params_type)` registration. Pass via
  `create_session(tools=...)`. Coexistence rule: when both `tools=`
  and `schema=` are present, the adapter prepends the existing
  `submit_result` tool to the user's list. Translate `TOOL_USE_*`
  session events into `ToolCallStart`/`ToolCallResult`. Tools join
  the existing cache key.
- Flip `Feature.TOOLS_FUNCTION` True on both adapters.
- Tests: round-trip with one tool on each; Copilot `submit_result`
  + `tools=` coexistence (forced structured output still works);
  Claude tools event-emission shape.

**Stopping point.** Three of four adapters wired. Codex still
accepts `tools=` and ignores them — fixed in D.

#### Iteration D — Codex rejection, probe, wrap-up ✅

- **Codex:** `CodexRuntime.session(tools=<non-None>)` raises
  `UnsupportedFeatureError(feature=Feature.TOOLS_FUNCTION)` with a
  "wire tools through the Codex CLI config file; airframe can't
  expose them programmatically" message. `Feature.TOOLS_FUNCTION`
  stays False on Codex.
- `examples/probe_tools.py` — multi-provider probe registering a
  tiny `calculator` tool (`add(a, b: float) -> float`), prompts
  "what is 17 × 23?", reports the `ToolCallStart`/`ToolCallResult`
  sequence from `stream()`.
- Tests: Codex tools= rejection; `test_features.py` asserts
  `TOOLS_FUNCTION` is universal-except-Codex; update
  `test_unwired_features_stay_false` (TOOLS_FUNCTION joins
  `any_adapter_may_support`).
- CHANGELOG Iteration D entry; trim the "Deferred (Phase 3)" block.

**Stopping point.** Phase 3 complete. Ready for release (`v0.6.0`)
or to roll straight into Phase 4 (MCP server refs).

### Risks and decisions to flag during execution

1. **`FunctionTool.handler` signature.** Plan-recommended: parsed
   `BaseModel` in, `Any` out (JSON-serialisable by convention).
   Locked in Iteration A. Matches Copilot's `define_tool` and
   LangChain's `@tool`. Alternative (pass an invocation-context
   object too) rejected — v0 keeps the shape minimal; can be added
   later via a kwarg-only optional parameter.
2. **OpenAI-compat tool-loop iteration cap.** A model that keeps
   requesting tool calls indefinitely is a real failure mode.
   Hard-fail at 20 iterations with `RuntimeProtocolError`;
   consumers wanting more can override via a future
   `provider_options` field. Not exposing as a kwarg yet — wait
   for a concrete user.
3. **Tools change between turns.** On Claude / Copilot the tools
   list is baked at session-creation; changing it mid-session means
   a new session. The plan attaches `tools=` to `session(...)`
   rather than `execute()` precisely for this reason. Worth
   re-confirming when wiring B and C.
4. **`stream()` contract clarification.** One `TurnComplete` per
   *user turn*, not per *model turn*. Tool round-trips produce
   intermediate `ToolCallStart`/`ToolCallResult` pairs but only one
   trailing `TurnComplete`. `TurnComplete` docstring updated in
   Iteration A.

---

## Phase 4 — MCP server refs (Tier 2)

**Goal.** Roadmap §3 P2 "Tool / MCP server registration" — the
Tier-2 half.

**Status: ✅ complete.** Shipped per (a) below on the
`phase-4-mcp-server-refs` branch. Iterations A–D landed sequentially;
all four adapters report their final truth (Claude all three
transports, Copilot stdio + http, Codex + OpenAI-compat permanent
declines with vendor-specific actionable messages).

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

### Iteration breakdown

Phase 4 lands in four iterations, mirroring Phase 3's A–D shape.
Status: **Phase 4 complete** on the `phase-4-mcp-server-refs`
branch — Iterations A through D all green.

Coverage matrix (target after Iteration D):

| Adapter | stdio | http | sse | Notes |
| --- | --- | --- | --- | --- |
| `ClaudeCodeRuntime` | ✓ | ✓ | ✓ | All three transports native to the SDK. |
| `CopilotRuntime` | ✓ | ✓ | ✗ | Copilot SDK has no SSE channel; decline at translation time. |
| `CodexRuntime` | ✗ | ✗ | ✗ | Permanent decline — MCP wired through `~/.codex/config.toml` only. |
| `OpenAICompatibleRuntime` | ✗ | ✗ | ✗ | Permanent decline on this family — MCP-as-tool is Responses-only; a future `OpenAIResponsesRuntime` could wire it. |

#### Iteration A — Protocol scaffolding (no behaviour) ✅

Lock the public surface; defer the wiring.

- New `airframe.tools.McpServerRef` (frozen+slots) — `name`,
  `transport: Literal["stdio", "http", "sse"]`, `command: list[str]
  | None`, `url: str | None`, `headers: dict[str, str] | None`,
  `auth_token: str | None`. `__post_init__` validates the
  transport/field combo (stdio needs `command`; http/sse need `url`;
  raises `ValueError` otherwise).
- New `Feature.TOOLS_MCP_SSE = "tools_mcp_sse"` enum member. The
  existing `TOOLS_MCP_STDIO` / `TOOLS_MCP_HTTP` /
  `TOOLS_MCP_IN_PROCESS` stay. (Adding an enum member is additive
  per ADR-002 — but once consumer code branches on
  `Feature.TOOLS_MCP_SSE`, renaming is breaking. Lock the name
  now.)
- Extend `AgentRuntime.session(...)` Protocol and every adapter's
  `session()` method with `mcp_servers: list[McpServerRef] | None
  = None`.
- Shared helper `airframe.sessions._check_mcp_servers_supported(
  refs, *, adapter_label, supports)` — iterates the list,
  looks up the matching `Feature.TOOLS_MCP_{STDIO,HTTP,SSE}` per
  ref, and raises `UnsupportedFeatureError(feature=<the missing
  one>)` on the first decline. `supports` is a callable so the
  helper doesn't need a runtime reference (same shape as
  `_check_tools_supported`).
- Every adapter's `session()` calls the helper at the top.
- Top-level export: `McpServerRef`.
- Tests: `tests/test_tools.py` extended with `McpServerRef` shape
  lock (frozen+slots, field order, post-init validation);
  `tests/test_features.py` snapshots `Feature.TOOLS_MCP_SSE`
  string value; cross-adapter feature-matrix tests confirm
  `TOOLS_MCP_{STDIO,HTTP,SSE}` is False everywhere and
  `session(mcp_servers=[...])` raises everywhere with the right
  `feature=` attribute.

**Stopping point.** `mcp_servers=` accepted by every adapter's
`session()` signature; non-None list raises immediately. No
per-adapter behaviour yet.

#### Iteration B — Wire Claude (broadest transport coverage) ✅

Claude's SDK has typed configs for all three transports
(`McpStdioServerConfig` / `McpHttpServerConfig` /
`McpSSEServerConfig`); wiring it first establishes the per-transport
pattern for Copilot to follow.

- `_translate_mcp_servers_for_claude(refs) → dict[str,
  McpStdioServerConfig | McpHttpServerConfig | McpSSEServerConfig]`
  — keyed by `ref.name`. Each ref's `auth_token` becomes the
  `Authorization: Bearer …` header (merged with caller-supplied
  `headers=`); `headers=` passes through verbatim.
- `ClaudeCodeSession` accepts `mcp_servers=` at construction;
  passes the translated dict via `ClaudeAgentOptions.mcp_servers`,
  **merged with** the in-process server Phase 3 builds for
  `tools=` (so `tools=` + `mcp_servers=` coexist cleanly).
- `_ensure_client` cache key gains an `mcp_servers=<fingerprint>`
  fragment so a refs-change forces reconnect. Fingerprint excludes
  `auth_token` and `headers` values to avoid caching sensitive
  material — only `name`, `transport`, `command`, `url`, and
  `sorted(headers.keys())` participate.
- Stream-event translation (Phase 3's `ToolUseBlock` →
  `ToolCallStart`, `ToolResultBlock` → `ToolCallResult`)
  automatically covers MCP-routed tool calls. Prefix-stripping is
  generalised: `mcp__<known_server_name>__` for any server in the
  registered set (in-process `airframe_tools` plus every
  `McpServerRef.name`) gets trimmed so consumers see the bare tool
  name.
- Flip `Feature.TOOLS_MCP_STDIO`, `_HTTP`, `_SSE` True on
  `ClaudeCodeRuntime.SUPPORTED_FEATURES`.
- Tests: stdio + http + sse ref translation (each wire-shape
  asserted); auth_token → Authorization-header injection; mixed
  list (stdio + http + sse) in one session; `tools=` +
  `mcp_servers=` coexistence; cache invalidation on refs change.

**Stopping point.** Claude fully wired. Copilot still raises on
`mcp_servers=`.

#### Iteration C — Wire Copilot (stdio + http; decline SSE) ✅

Copilot's SDK has stdio + http but no SSE. The decline message for
SSE refs needs to be specific.

- `_translate_mcp_servers_for_copilot(refs) → list[MCPStdioServerConfig
  | MCPHTTPServerConfig]`. SSE refs raise
  `UnsupportedFeatureError(feature=Feature.TOOLS_MCP_SSE)` with a
  "Copilot SDK has no SSE transport channel; use http transport
  instead" message.
- `CopilotAgentSession` accepts `mcp_servers=`; passes via
  `CopilotClient.create_session(mcp_servers=[...])`. Tools join the
  existing session cache key (already keyed on schema + effort +
  tools fingerprint).
- Stream-event translation (Phase 3's `ToolExecutionStartData` →
  `ToolCallStart`, `ToolExecutionCompleteData` → `ToolCallResult`)
  automatically covers MCP-routed tool calls. The
  `submit_result`-suppression set already drops one specific tool
  name; no new filtering needed.
- Flip `Feature.TOOLS_MCP_STDIO`, `_HTTP` True on
  `CopilotRuntime.SUPPORTED_FEATURES`. `TOOLS_MCP_SSE` stays
  False; refs of that transport surface the explicit decline.
- Tests: stdio + http ref translation; SSE decline carries the
  http-transport hint; `tools=` + `mcp_servers=` coexistence
  (including the `submit_result` + custom tools + MCP three-way
  combination); cache invalidation.

**Stopping point.** Three of four adapters report their final
truth. Codex + OpenAI-compat still accept `mcp_servers=` and
ignore it — fixed in D.

#### Iteration D — Codex + OpenAI-compat declination, probe, wrap-up ✅

- **Codex.** `CodexRuntime.session(mcp_servers=<non-empty>)` raises
  `UnsupportedFeatureError(feature=Feature.TOOLS_MCP_STDIO)` (the
  first ref's transport, since the Codex Python SDK declines all
  transports equally) with a "wire MCP servers through the
  ``codex`` CLI's config file (`~/.codex/config.toml`
  `[[mcp_servers]]` block) instead — the Codex Python SDK has no
  programmatic MCP-registration channel" message. Symmetric with
  Phase 3 Iteration D's tools= decline.
- **OpenAI-compat.** `OpenAICompatibleRuntime.session(mcp_servers=<non-empty>)`
  raises with an OpenAI-compat-specific message: "Chat Completions
  doesn't support MCP-as-tool; that wire shape is Responses-API
  only. A future `OpenAIResponsesRuntime` (separate from this
  compat family) could translate to the Responses-API
  `{"type":"mcp",...}` tool shape." Pointer to the future direct-
  API option.
- `examples/probe_mcp.py` — multi-provider probe registering a
  small public stdio MCP server (target candidate: the
  `@modelcontextprotocol/server-everything` reference server, or a
  bundled fixture echo-server). Stream events; report tool
  invocations and the trailing `TurnComplete`. Default to
  `claude` (broadest transport support); accept
  `--provider claude|github-copilot|codex|opencode` (the latter
  two surface the decline messages verbatim, same probe-as-docs
  pattern Phase 3 used).
- Tests: Codex + OpenAI-compat rejection messages (both content
  *and* `feature=` attribute);
  `tests/test_features.py::test_unwired_features_stay_false`
  permits `TOOLS_MCP_{STDIO,HTTP,SSE}` to vary per adapter;
  `test_features.py::test_mcp_transports_final_matrix` pins the
  table from the iteration-breakdown header (Claude all three;
  Copilot stdio + http; Codex + OpenAI-compat none). The
  `TOOLS_MCP_IN_PROCESS` flag stays False on every adapter —
  Phase 4 doesn't expose an in-process transport on
  `McpServerRef`; the Phase 3 in-process MCP path is internal
  plumbing for `tools=` rather than a user-facing capability.
- CHANGELOG Iteration D entry; trim the "Deferred (Phase 4)"
  block; mark Phase 4 ✅ in this doc.

**Stopping point.** Phase 4 complete. Ready for release (`v0.7.0`)
or to roll straight into Phase 5 (permission, hooks, budget).

### Risks and decisions to flag during execution

1. **`McpServerRef` field set.** Shape-locked in Iteration A. The
   plan-doc paragraph above recommends a `provider_options=` dict
   slot for transport-specific knobs (`stdio.env`,
   `http.timeout`); pragmatic decision in Iteration A: defer that
   slot until a consumer asks. Adding a kwarg later is additive;
   removing one isn't. Today's three fields cover the common case.
2. **`Feature.TOOLS_MCP_SSE` name lock.** Adding the enum member
   is additive but once consumer code branches on it the string
   value is sticky. Snapshot in `test_feature_string_values_are_stable`
   from Iteration A onwards.
3. **Auth-token caching.** `McpServerRef.auth_token` and any
   `headers=` value may contain sensitive material. Strategy in
   Iteration B: fingerprint participates from `name`, `transport`,
   `command`, `url`, and the *sorted keys* of `headers` only;
   never the values, never `auth_token`. Re-confirm when wiring B.
4. **`mcp_servers=` + `tools=` coexistence.** Both end up routed
   through the same vendor MCP plumbing (Claude's `mcp_servers`
   slot; Copilot's `tools=` slot for FunctionTool + dedicated
   `mcp_servers=` slot for external refs). Verify in Iteration B
   that the merged dict on Claude has both
   `airframe_tools` (in-process for FunctionTool) and each external
   server's `name`-keyed entry; verify in Iteration C that
   Copilot's `tools=` list and `mcp_servers=` list don't accidentally
   shadow each other on tool-name collisions. **Decision: tool-name
   collisions raise at session-construction time** rather than
   silently shadowing — same "no silent fallbacks" principle the
   plan calls out elsewhere.
5. **In-process MCP capability flag.** `TOOLS_MCP_IN_PROCESS`
   stays False on every adapter in Phase 4. Phase 3 uses Claude's
   in-process MCP server as an internal implementation detail for
   `tools=`, but that's not a *user-facing* MCP-registration API.
   If a later phase exposes
   `McpServerRef(transport="in_process", server_factory=...)`
   that flag flips True; until then leave it false to keep
   `supports()` honest.
6. **MCP tool prefix-stripping in stream events.** Phase 3's
   `_strip_mcp_prefix` only knows about `mcp__airframe_tools__`.
   Iteration B generalises it: the session tracks the set of
   registered server names (Phase 3 in-process + Phase 4 external)
   and strips any `mcp__<known>__` prefix. Unrecognised prefixes
   pass through verbatim so consumers can still inspect raw vendor
   tool names if needed.

---

## Phase 5 — Permission, hooks, budget ✅

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

### Iteration breakdown

Phase 5 lands in four iterations, mirroring Phase 3/4's A–D shape.
The plan's three feature buckets (permission, hooks, budget) are
independent; each gets its own wiring iteration. Status: pending;
work begins on a fresh `phase-5-permission-hooks-budget` branch.

Coverage matrix (target after Iteration D):

| Adapter | PERMISSION | HOOKS | BUDGET_USD | BUDGET_TURN | Notes |
| --- | --- | --- | --- | --- | --- |
| `ClaudeCodeRuntime` | ✓ | ✓ | ✓ | ✓ | All native via SDK channels. |
| `CopilotRuntime` | ✓ | ✓ | ✓ | ✗ | Vendor enforces turn caps internally; `max_turns` no-op. |
| `CodexRuntime` | ✓ | ✓ | ✓ | ✓ | Permission mapping is coarser (session-wide `approval_policy`). |
| `OpenAICompatibleRuntime` | ✗ | ✓ | ✓ | ✓ | Chat Completions has no permission concept; declines permanently. |

#### Iteration A — Protocol scaffolding (no behaviour) ✅

Lock the public surface; defer the wiring.

- New `airframe/permission.py` — `PermissionRequest` (frozen+slots:
  `tool_name`, `tool_args: dict[str, Any]`, `reason: str | None`),
  `PermissionDecision = Literal["allow", "deny", "defer"]`,
  `PermissionCallback` Protocol with
  `async def handle(request) -> PermissionDecision`.
- New `airframe/hooks.py` — `HookEvent` (frozen+slots: `kind`
  literal of the 8 strings, `session_id: str | None`,
  `payload: dict[str, Any]`).
- Extend `AgentRuntime.session(...)` Protocol and every adapter's
  `session()` with:
  - `on_permission: PermissionCallback | None = None`
  - `on_event: Callable[[HookEvent], None] | None = None`
- Extend `AgentSession.execute()` / `.stream()` (Protocol and every
  bespoke session) with:
  - `max_turns: int | None = None`
  - `max_budget_usd: float | None = None`
- Shared helpers in `airframe.sessions`:
  - `_check_permission_supported(cb, *, adapter_label, supports)` —
    raises `UnsupportedFeatureError(feature=Feature.PERMISSION_CALLBACK)`
    when `cb is not None` and the flag is False.
  - `_check_hooks_supported(cb, *, adapter_label, supports)` — same
    shape for `Feature.LIFECYCLE_HOOKS`.
  - `_check_budget_supported(*, max_turns, max_budget_usd,
    adapter_label, supports)` — raises with the matching feature
    (`BUDGET_TURN_CAP` / `BUDGET_USD_CAP`) on the first non-None
    kwarg whose flag is False.
- Every adapter's `session()` / `execute()` / `stream()` calls the
  helpers at the top.
- Top-level exports: `PermissionRequest`, `PermissionDecision`,
  `PermissionCallback`, `HookEvent`.
- Tests:
  - `tests/test_permission.py` — shape lock on the new dataclasses
    + protocol; `PermissionDecision` literal snapshot.
  - `tests/test_hooks.py` — shape lock on `HookEvent`; the 8
    `kind` strings snapshotted in `test_hook_event_kind_strings_are_stable`
    so a rename is caught at PR time (same discipline as
    `test_feature_string_values_are_stable` for `Feature`).
  - `tests/test_features.py` extended: cross-adapter matrix
    confirming `PERMISSION_CALLBACK` / `LIFECYCLE_HOOKS` /
    `BUDGET_USD_CAP` / `BUDGET_TURN_CAP` False everywhere; the
    four new kwargs raise on every adapter with the right
    `feature=` attribute.

**Stopping point.** New kwargs accepted by every adapter's
`session()` / `execute()` / `stream()` signature; non-None values
raise immediately. No per-adapter behaviour yet.

#### Iteration B — Permission callback (3 wire, 1 declines) ✅

Sets the per-vendor permission-channel pattern. Same "wire the
hardest one first" cadence as Phase 4 Iteration B (Claude).

- **Claude:** `_translate_permission_for_claude(cb)` →
  `ClaudeAgentOptions.can_use_tool=<adapter>`. The adapter builds
  a :class:`PermissionRequest` from the SDK's tool-call dict, awaits
  the user's callback, maps `"allow"`/`"deny"` to the SDK's boolean
  return; `"defer"` falls through to the SDK's default policy with
  a debug log. Callback identity joins the existing
  `_ensure_client` cache key — changing the callback forces
  reconnect (same pattern as `tools=`).
- **Copilot:** translate to `on_permission_request=` (mandatory at
  `create_session`). Map airframe's `PermissionDecision` literal to
  Copilot's `PermissionDecision` enum
  (`approve_once`/`approve_for_session`/`reject`). Callback joins
  the existing session cache key.
- **Codex:** translate to `approval_policy=` (session-wide enum:
  `never`/`untrusted`/`on-request`/`on-failure`). The user's
  callback runs **once** at session creation to derive the policy
  enum — per-call interception isn't possible through the SDK.
  Document the limitation loudly so consumers aren't surprised.
- **OpenAI-compat:** inline decline with a permanent message —
  chat-completions has no tool-permission wire shape. Same shape
  as Phase 4 Iteration D's `mcp_servers=` decline.
- Flip `Feature.PERMISSION_CALLBACK` True on Claude / Copilot /
  Codex.
- `examples/probe_permission.py` — multi-provider live probe with
  a callback that logs every request and approves. OpenAI-compat
  branch surfaces the decline verbatim (probe-as-docs).
- Tests: round-trip on each accepting adapter (mock callback fired
  with the expected `PermissionRequest`, decision honoured);
  OpenAI-compat decline (`.feature == PERMISSION_CALLBACK` +
  actionable message); cache invalidation when callback identity
  changes between turns on Claude + Copilot; Codex per-session-only
  semantics documented in test docstring.

**Stopping point.** Three of four adapters wire permission;
OpenAI-compat declines. Hooks and budget still raise.

#### Iteration C — Lifecycle hooks (4 wire, mixed mechanisms) ✅

The most surface-area iteration; each adapter takes a different
route to emit the 8 `HookEvent.kind` literals.

- **Claude:** pass `hooks=<adapter>` to `ClaudeAgentOptions` plus
  `include_hook_events=True`. Translate the SDK's native hook
  stream into `HookEvent` and fan out to `on_event`.
- **Copilot:** install a `session.on(...)` subscriber at session
  creation that translates the SDK's typed `*Data` events into the
  8 `kind` literals.
- **Codex:** synthesize from `ItemStartedEvent` /
  `ItemCompletedEvent` on the thread event stream. Emits
  `session_start`/`session_end`/`user_prompt_submit`/
  `pre_tool_use`/`post_tool_use`.
- **OpenAI-compat:** synthesize from the existing client-side
  tool-loop in `OpenAICompatibleSession._do_execute` / `stream`.
  Emits the same five kinds Codex does, plus `tool_failure`.
  Cannot emit `pre_compact` or `rate_limit` honestly — document.
- Per-adapter docstring: the *emittable kinds set* for that
  adapter so consumers know what to expect.
- Flip `Feature.LIFECYCLE_HOOKS` True on all four.
- `examples/probe_hooks.py` — multi-provider live probe printing
  every observed `HookEvent` in order.
- Tests: ordered-event matcher on each adapter for a deterministic
  prompt (one that triggers `user_prompt_submit` → `pre_tool_use`
  → `post_tool_use` → `session_end`); per-adapter
  emittable-kinds-set test pinned (so a regression where Codex
  stops emitting `post_tool_use` is caught at PR time).

**Stopping point.** All four adapters fire hooks; emittable-kinds
sets are documented and tested.

#### Iteration D — Budget caps + probe + wrap-up ✅

- **Claude:** `max_turns` → `ClaudeAgentOptions.max_turns`
  (overrides the hard-coded `DEFAULT_MAX_TURNS=60` when set);
  `max_budget_usd` via vendor field if available, else client-side
  accumulation against `RuntimeResult.cost.cost_usd` (same pattern
  the other three adapters use).
- **Copilot / Codex / OpenAI-compat:** client-side accumulation in
  the session. `max_turns` is a per-`execute()` turn counter
  (Copilot: no-op since vendor caps internally; Codex /
  OpenAI-compat: real counter). `max_budget_usd` aborts the next
  turn when the running total exceeds the cap.
- New error: `RuntimeBudgetExceededError(AgentRuntimeError)` with
  `cap: float`, `current: float`,
  `kind: Literal["usd","turns"]` attributes. Raised at turn
  boundary in v0 — no mid-turn interrupts (additive later via the
  existing `cancel()` plumbing).
- Flip `Feature.BUDGET_USD_CAP` True on all four (client-side
  enforcement counts as honest support).
  `Feature.BUDGET_TURN_CAP` True on Claude (native) + Codex +
  OpenAI-compat (client-side); False on Copilot.
- `examples/probe_budget.py` — multi-provider live probe with a
  deliberately tiny cap; verifies the error fires and prints the
  matrix table at the end.
- `tests/test_features.py::test_phase_5_final_matrix` pins the
  endgame (Permission / Hooks / Budget_USD / Budget_Turn columns
  per the coverage table above).
- CHANGELOG Iteration D entry; trim the "Deferred (Phase 5)"
  block; mark Phase 5 ✅ in this doc.

**Stopping point.** Phase 5 complete. Ready for release as
`v0.8.0` or to roll into Phase 6 (middleware, sandbox, subagents)
— or the `v1.0` cut per the "When to cut v1.0" section above.

### Risks and decisions to flag during execution

1. **`HookEvent.kind` literal lock (8 strings).** Phase-0-style
   shape lock; snapshot the wire values in `tests/test_hooks.py`
   from Iteration A onwards. Once consumer code branches on
   `event.kind == "pre_tool_use"`, renaming is a major bump.
2. **`PermissionDecision = "defer"` semantics.** Plan doesn't
   define cross-vendor. Decision in Iteration B: fall through to
   the vendor's default policy with a debug log. Lock the
   semantics in the per-adapter docstrings.
3. **Codex permission fidelity.** Codex's `approval_policy` is
   *session-wide*, not per-call. The user's `PermissionCallback`
   fires once (to derive the policy enum); per-call interception
   isn't possible through the SDK. Document loudly so consumers
   aren't surprised.
4. **`max_budget_usd` granularity.** Turn-boundary abort vs
   mid-turn interrupt. Recommend turn-boundary in v0 (simpler,
   predictable cost shape); mid-turn cancellation is additive
   later via the existing `cancel()` plumbing.
5. **`max_turns` vs `MAX_TOOL_ITERATIONS` on OpenAI-compat.** Keep
   distinct: `max_turns` is a user-facing budget; `MAX_TOOL_ITERATIONS=20`
   stays as the runaway guard. Document the relationship in the
   adapter docstring.
6. **`RuntimeBudgetExceededError` as new error class** vs reusing
   `RuntimeProtocolError`. Recommend new class with
   `cap`/`current`/`kind` attributes so consumers can
   retry-with-larger-cap without parsing strings.
7. **`HookEvent.payload: dict[str, Any]`** — lowest-common-
   denominator shape. Per-`kind` typed payloads would be more
   rigorous but additive later. Recommend dict for v0.
8. **Cache-key churn from `on_permission` / `on_event`.** Both
   bake at session-creation on Claude / Copilot, so
   callable-identity changes force a session rebuild. Document so
   consumers don't pass freshly-constructed lambdas per call by
   accident.

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
| 0 | Foundations + plain-text `execute()` fix | v0.3.0 | ⚡ **Third-party adapters** (entry-points + TCK) | `Feature` enum values; `ProviderOptions` shape |
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
