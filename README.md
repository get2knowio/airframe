# airframe

**JDBC for LLM agent SDKs.** Vendor-neutral agent runtime for
Python — write your agent against a small `AgentRuntime` protocol
and switch between Claude Code, GitHub Copilot, OpenAI Codex, or
OpenCode Zen by swapping a single object.

Drives both *execution* (one prompt → typed payload + cost) and
*discovery* (which providers does this install support? which models
can the user pick from?). The same five-method protocol covers
agent-CLI subprocesses, OpenAI-compatible HTTP, and everything in
between.

```python
from airframe import ClaudeCodeRuntime, ProviderModel
from pydantic import BaseModel

class Brief(BaseModel):
    summary: str
    risks: list[str]

runtime = ClaudeCodeRuntime()
result = await runtime.execute(
    "Brief me on the project structure.",
    schema=Brief,
    model=ProviderModel("claude", "claude-haiku-4-5"),
)
print(result.structured)     # {"summary": "...", "risks": [...]}
print(result.cost.cost_usd)  # 0.0042
await runtime.close()
```

Swap the adapter, keep the schema:

```python
# Same Brief schema; different vendor, different auth path.
from airframe import CopilotRuntime, ProviderModel
runtime = CopilotRuntime()
result = await runtime.execute(
    "Brief me on the project structure.",
    schema=Brief,
    model=ProviderModel("github-copilot", "gpt-5-mini"),
)
```

Plain text works too — omit `schema=` and read `result.text`:

```python
runtime = ClaudeCodeRuntime()
result = await runtime.execute(
    "Summarise this in two paragraphs of free-form markdown.",
    system="You are a thoughtful technical writer.",
)
print(result.text)            # free-form markdown
assert result.structured is None
```

Discover providers and models at runtime (for menus, config UIs,
"what can I install / which models can I pick?"):

```python
from airframe import list_providers, runtime_for

# Filtered by which pip extras the consumer has installed.
for provider_id in list_providers():
    runtime_cls = runtime_for(provider_id)
    runtime = runtime_cls()
    try:
        models = await runtime.list_models()
        for m in models:
            print(provider_id, m.id, m.display_name, m.context_window)
    finally:
        await runtime.close()
```

## Why?

Each vendor ships a Python SDK that does something subtly different:
the Claude Agent SDK exposes a subprocess + JSON-RPC interface;
GitHub's Copilot SDK exposes a session + tool registration model;
OpenAI's Codex SDK passes a JSON Schema flag to a CLI subprocess;
the opencode-go Zen gateway speaks OpenAI-compatible HTTP. Each has
its own auth chain, error taxonomy, cost-reporting shape, structured-
output mechanism, and models-endpoint shape. Airframe collapses those
differences behind one
`execute / reset / close / validate_binding / list_models` interface,
classifies every vendor's failures into a single hierarchy, and
produces a single `CostRecord` / `ModelInfo` shape regardless of the
vendor.

The protocol is intentionally narrow. The five methods are the
contract; everything else (auth chains, session caching, tool-call
forcing, JSON-schema mode, envelope unwrapping, per-model metadata
joining) lives inside each adapter, where vendor-specific behaviour
belongs.

Anything *above* the protocol — retry policy, fallback across
vendors, conversation memory, multi-agent orchestration — is left to
the consumer. Airframe is the driver layer; the application
composes its own behaviour on top.

## Install

The base package ships only the protocol + error types. Each
adapter's SDK is an optional extra so you only pull what you need:

```bash
pip install airframe-agents[claude]         # ClaudeCodeRuntime
pip install airframe-agents[copilot]        # CopilotRuntime
pip install airframe-agents[codex]          # CodexRuntime
pip install airframe-agents[openai-compat]  # OpenCodeZenRuntime (+ future OAI-compat)
pip install airframe-agents[all]            # Everything
```

`list_providers()` filters by which extras you installed:
``pip install airframe-agents[copilot]`` makes ``list_providers()``
return ``["github-copilot"]``; the other adapters are silently filtered
so menus stay honest about what the local install can serve. Pass
``installed_only=False`` to see every built-in provider regardless of
SDK presence (useful for documentation UIs).

The import name is `airframe`. The PyPI dist name is
`airframe-agents` (the unqualified `airframe` slot on PyPI was taken
by an unrelated abandoned project).

## The four adapters

| Adapter | `PROVIDER_ID` | Vendor SDK | Auth | Structured output | Subprocess? |
| --- | --- | --- | --- | --- | --- |
| `ClaudeCodeRuntime` | `claude` | `claude-agent-sdk` | Claude Max OAuth → `~/.claude/credentials.json` → `ANTHROPIC_API_KEY` | Native `output_format={"type":"json_schema",...}` | yes, per-runtime |
| `CopilotRuntime` | `github-copilot` | `github-copilot-sdk` | `GITHUB_TOKEN` → `gh auth` | Forced `submit_result` tool (Copilot's native `define_tool`) | yes, per-runtime |
| `CodexRuntime` | `codex` | `openai-codex-sdk` | `OPENAI_API_KEY` → opencode `auth.json` → `~/.codex/auth.json` | Native JSON-schema flag (`--output-schema`) | yes, per-turn |
| `OpenCodeZenRuntime` | `opencode` | `openai` (HTTP) | `OPENCODE_API_KEY` → opencode `auth.json` | `response_format={"type":"json_schema",...}` | no (direct HTTP) |

The OpenAI-compatible family (`OpenCodeZenRuntime` today; Together /
Groq / Fireworks / OpenRouter as future siblings) shares the
`OpenAICompatibleRuntime` base class and the single `[openai-compat]`
pip extra. Subclasses are ~30 lines: declare `PROVIDER_ID`, a default
base URL + model, a per-model metadata table, and a vendor-specific
`_resolve_api_key()` hook — everything else (HTTP execute, structured
output, `list_models`, error classification) is inherited.

The provider IDs are deliberately strict: one canonical ID per
adapter, no aliases. ``"anthropic"`` is reserved for a future direct-
API `AnthropicRuntime`; ``"openai"`` is reserved for a future direct-
API `OpenAIRuntime`. The current adapters cover the *subscription*
paths (Claude Max, Copilot, ChatGPT Plus, opencode-go) — API-key-only
direct-vendor adapters can land later without breaking these IDs.

`ClaudeCodeRuntime` is the only adapter that accepts Claude bindings.
`CopilotRuntime.validate_binding` *rejects* Claude bindings on
purpose — Claude served via Copilot Chat Completions emits markdown-
fenced JSON instead of honouring tool calls, so it can't satisfy the
structured-output contract. Route Claude through `ClaudeCodeRuntime`.

## The protocol

```python
class AgentRuntime(Protocol):
    label: str

    async def execute(
        self,
        prompt: Prompt,
        *,
        schema: type[BaseModel] | None = None,
        system: str | None = None,
        persona: str | None = None,
        model: ProviderModel | None = None,
        thinking: ThinkingMode = None,
        timeout: float = 600.0,
    ) -> RuntimeResult: ...

    def session(
        self,
        *,
        resume: str | None = None,
        system: str | None = None,
        model: ProviderModel | None = None,
        tools: list[FunctionTool] | None = None,
        mcp_servers: list[McpServerRef] | None = None,
        on_permission: PermissionCallback | None = None,
        on_event: Callable[[HookEvent], None] | None = None,
        provider_options: ProviderOptions | None = None,
    ) -> AgentSession: ...

    async def reset(self) -> None: ...
    async def close(self) -> None: ...
    def validate_binding(self, binding: ProviderModel) -> bool: ...
    async def list_models(self) -> list[ModelInfo]: ...
    def supports(self, feature: Feature, model: ProviderModel | None = None) -> bool: ...
    def unwrap(self, cls: type[T]) -> T: ...
```

* **`execute`** — single-turn convenience over `session().execute() + close()`.
  Returns a `RuntimeResult` with text + (optional) structured
  payload + cost + finish reason.
* **`session`** — open a multi-turn `AgentSession` (see below).
  Every later feature kwarg (`tools=`, `mcp_servers=`,
  `on_permission=`, `on_event=`, `provider_options=`) attaches here,
  not to `execute()` — this keeps the per-turn surface narrow.
* **`reset`** — no-op on every built-in adapter as of v0.5.0; kept
  on the protocol for completeness. Sessions own per-conversation
  state; the runtime owns nothing scope-bound.
* **`close`** — full teardown. Idempotent; never raises.
* **`validate_binding`** — predicate: does this runtime serve a
  given `(provider_id, model_id)`? Cheap and non-async; suitable for
  filtering bindings before attempting them.
* **`list_models`** — hit the vendor's models endpoint with the
  user's resolved credentials, return a list of `ModelInfo` (id,
  display name, context window, pricing, capability flags). Drives
  UI menus.
* **`supports`** — capability predicate. `runtime.supports(Feature.TOOLS_MCP_HTTP)`
  before passing `mcp_servers=[McpServerRef(transport="http", ...)]`.
* **`unwrap`** — JDBC-`Wrapper`-style escape hatch to the native
  vendor object. `runtime.unwrap(ClaudeSDKClient)` /
  `session.unwrap(CopilotSession)`, etc.

### `AgentSession`

```python
class AgentSession(Protocol):
    id: str | None  # vendor session id, or None for stateless adapters

    async def execute(
        self,
        prompt: Prompt,
        *,
        schema: type[BaseModel] | None = None,
        thinking: ThinkingMode = None,
        max_turns: int | None = None,
        max_budget_usd: float | None = None,
        timeout: float = 600.0,
    ) -> RuntimeResult: ...

    def stream(self, prompt: Prompt, *, ...) -> AsyncIterator[RuntimeEvent]: ...
    async def cancel(self) -> None: ...
    async def close(self) -> None: ...
    def unwrap(self, cls: type[T]) -> T: ...
```

The session owns conversation state (`messages=[]` buffer for OAI-compat,
`ClaudeSDKClient` for Claude Code, `CopilotSession` for Copilot,
`Thread` for Codex). Same `close()` discipline — idempotent, never
raises. `cancel()` is cheap and idempotent: no-op when no turn is
in flight; abort+raise `RuntimeCancelledError` mid-turn.

```python
runtime = ClaudeCodeRuntime()
sess = runtime.session(system="You are concise.")
try:
    async for event in sess.stream("Tell me about Python."):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, TurnComplete):
            print(f"\nfinal: {event.result.cost.cost_usd}")
finally:
    await sess.close()
```

`stream()` yields a discriminated union of five
[`RuntimeEvent`](src/airframe/events.py) variants:
`TextDelta`, `ReasoningDelta`, `ToolCallStart`, `ToolCallResult`,
`TurnComplete`. The variant set is shape-locked — consumer
`match event:` is safe across releases.

### Errors

Adapters classify vendor failures into a small hierarchy so consumer
`except` clauses don't need vendor-specific knowledge:

| Error | What it means |
| --- | --- |
| `RuntimeAuthError` | Credentials bad / expired / missing. |
| `RuntimeModelNotFoundError` | Server doesn't serve that model on this binding. |
| `RuntimeTransientError` | 5xx, rate limit, brief outage. Call was attempted; failure was recoverable. |
| `RuntimeStructuredOutputError` | Transport succeeded but model didn't produce a payload matching the schema. |
| `RuntimeContextOverflowError` | Prompt exceeded the model's context window. |
| `RuntimeProtocolError` | Adapter saw something it can't interpret (adapter / SDK bug). |
| `RuntimeServerStartError` | Adapter couldn't bring its backend up at all. |
| `RuntimeCancelledError` | Caller-initiated abort. |

What to *do* with each — retry, fall back to another binding,
surface to the user, escalate to a larger model — is consumer
policy. Airframe doesn't ship a retry / fallback engine; consumers
compose their own from these primitives.

See [docs/architecture.md](docs/architecture.md) for the design
rationale and the operational landmines each adapter mitigates.

## Examples

Probe scripts under `examples/` exercise each adapter end-to-end
against a real CLI / HTTP endpoint. They're runnable demos, not part
of `make test`. Auth issues surface as classified `Runtime*Error`
so you see exactly what would happen in production if credentials
were misconfigured.

```bash
# Per-adapter execute() probes (require auth for that vendor).
uv run python examples/probe_claude_code.py
uv run python examples/probe_copilot.py
uv run python examples/probe_codex.py
uv run python examples/probe_opencode_zen.py

# Live capability matrix across every installed adapter.
uv run python examples/probe_supports.py

# Phase 1+: streaming, session resume, cancellation.
uv run python examples/probe_streaming.py
uv run python examples/probe_session_resume.py

# Phase 2: thinking effort, vision/file inputs.
uv run python examples/probe_thinking.py
uv run python examples/probe_vision.py

# Phase 3: function-tool round-trip.
uv run python examples/probe_tools.py

# Phase 4: external MCP server registration.
uv run python examples/probe_mcp.py

# Phase 5: permission callback, lifecycle hooks, budget caps.
uv run python examples/probe_permission.py
uv run python examples/probe_hooks.py
uv run python examples/probe_budget.py
```

## Capability negotiation

Each adapter declares which protocol features it implements via
`runtime.supports(Feature.X)`. The `Feature` enum's string values
were fixed at v0.3.0; consumer code branches on them safely:

```python
from airframe import ClaudeCodeRuntime, Feature, McpServerRef

runtime = ClaudeCodeRuntime()
if runtime.supports(Feature.TOOLS_MCP_HTTP):
    sess = runtime.session(
        mcp_servers=[McpServerRef(name="docs", transport="http", url="...")]
    )
```

End-of-Phase-5 capability matrix (run
`uv run python examples/probe_supports.py` for the live version):

| Feature | Claude | Copilot | Codex | OpenAI-compat |
|---|---|---|---|---|
| `STRUCTURED_OUTPUT_JSON_SCHEMA` | ✓ | ✓ | ✓ | ✓ |
| `STREAMING` | ✓ | ✓ | ✓ | ✓ |
| `CANCEL` | ✓ | ✓ | ✓ | ✓ |
| `SESSION_RESUME` | ✓ | ✓ | ✓ | ✗ |
| `REASONING_EFFORT` | ✓ | ✓ | ✓ | ✓ |
| `REASONING_BUDGET_TOKENS` | ✓ | ✗ | ✗ | ✗ |
| `VISION_INPUT` | ✓ | ✓ | ✓ | ✓ |
| `FILE_INPUT` | ✓ | ✓ | ✓ | ✗ |
| `TOOLS_FUNCTION` | ✓ | ✓ | ✗ | ✓ |
| `TOOLS_MCP_STDIO` | ✓ | ✓ | ✗ | ✗ |
| `TOOLS_MCP_HTTP` | ✓ | ✓ | ✗ | ✗ |
| `TOOLS_MCP_SSE` | ✓ | ✗ | ✗ | ✗ |
| `PERMISSION_CALLBACK` | ✓ | ✓ | ✓ (session-wide) | ✗ |
| `LIFECYCLE_HOOKS` | ✓ (8 kinds) | ✓ (7 kinds) | ✓ (6 kinds) | ✓ (6 kinds) |
| `BUDGET_USD_CAP` | ✓ | ✓ | ✓ | ✓ |
| `BUDGET_TURN_CAP` | ✓ | ✗ | ✓ | ✓ |

A declined capability raises `UnsupportedFeatureError` with a
`feature=` attribute, never a silent fallback.

## Sessions, streaming, and the new kwargs

Beyond `execute(schema=)`, every Phase 1–5 capability attaches to
`runtime.session(...)` / `session.execute(...)` / `session.stream(...)`:

```python
from airframe import (
    ClaudeCodeRuntime, FunctionTool, McpServerRef,
    PermissionCallback, PermissionDecision, PermissionRequest,
    HookEvent, ClaudeOptions,
)

class _AddArgs(BaseModel):
    a: float
    b: float

async def add(args: _AddArgs) -> float:
    return args.a + args.b

class ApproveAll(PermissionCallback):
    async def handle(self, req: PermissionRequest) -> PermissionDecision:
        return "allow"

def log_event(e: HookEvent) -> None:
    print(f"[{e.kind}] {e.payload}")

runtime = ClaudeCodeRuntime()
sess = runtime.session(
    system="You are a careful math assistant.",
    tools=[FunctionTool(name="add", description="Add two numbers.",
                         params=_AddArgs, handler=add)],
    mcp_servers=[McpServerRef(name="docs", transport="http",
                              url="https://mcp.example.com",
                              auth_token="...")],
    on_permission=ApproveAll(),
    on_event=log_event,
    provider_options=ClaudeOptions(strict_mcp_config=True),
)
try:
    result = await sess.execute(
        "What is 17 + 25?",
        thinking="medium",
        max_turns=10,
        max_budget_usd=0.05,
    )
    print(result.text)
finally:
    await sess.close()
```

* **`tools=`** — `FunctionTool` instances the model may invoke. The
  session drives the round-trip (Claude/Copilot/OpenAI-compat) or
  declines permanently (Codex — no SDK tool-registration channel).
* **`mcp_servers=`** — `McpServerRef` entries pointing at external
  Model Context Protocol servers. Stdio + http on Claude/Copilot;
  SSE on Claude only.
* **`on_permission=`** — callback receives `PermissionRequest` and
  returns `"allow"` / `"deny"` / `"defer"`. Per-call on Claude /
  Copilot; session-wide on Codex (fires once to derive the policy
  enum); declined on OpenAI-compat.
* **`on_event=`** — synchronous observer receives `HookEvent`
  instances (`session_start`, `pre_tool_use`, `post_tool_use`, etc.).
  Per-adapter `EMITTABLE_HOOK_KINDS` ClassVar pins which subset of
  the eight canonical kinds each adapter fires.
* **`max_turns=`** / **`max_budget_usd=`** — cumulative caps
  enforced at the turn boundary; trip `RuntimeBudgetExceededError`
  with `cap` / `current` / `kind` attributes.
* **`provider_options=`** — vendor-specific extension namespace
  (`ClaudeOptions`, `CopilotOptions`, `CodexOptions`,
  `OpenAICompatOptions`). Tagged union — passing the wrong namespace
  raises at the adapter boundary.

## Escape hatch: `runtime.unwrap()`

When the portable surface doesn't expose a vendor-specific knob,
reach the native SDK object directly via JDBC-`Wrapper`-style
`unwrap()`:

```python
from claude_agent_sdk import ClaudeSDKClient

runtime = ClaudeCodeRuntime()
await runtime.execute("hello", schema=Brief)

# Now reach the underlying SDK for vendor-specific behaviour:
client: ClaudeSDKClient = runtime.unwrap(ClaudeSDKClient)
await client.interrupt()
```

Each adapter accepts `unwrap(type(self))` (returning `self`) plus
its native types: `ClaudeCodeRuntime.unwrap(ClaudeSDKClient)`,
`CopilotRuntime.unwrap(CopilotClient | CopilotSession)`,
`CodexRuntime.unwrap(Codex | Thread)`,
`OpenAICompatibleRuntime.unwrap(AsyncOpenAI)`. Unsupported types
raise `TypeError`.

## Adding an adapter

A new adapter is one class implementing `AgentRuntime`:

1. Map your vendor's auth chain into a `_resolve_api_key()` / token
   helper.
2. Choose a structured-output mechanism — native JSON-schema if the
   vendor supports it, otherwise force a `submit_result` tool call.
3. Classify the vendor's exceptions into the
   `airframe.errors.Runtime*Error` hierarchy.
4. Populate a `CostRecord` from the vendor's usage report. Use the
   per-model pricing table when the vendor doesn't return cost
   directly.
5. Declare `PROVIDER_ID`, `REQUIRES_PACKAGE`, and `EXTRA_NAME` as
   ClassVars; implement `validate_binding`.
6. Implement `list_models()` against the vendor's models endpoint;
   enrich each entry with whatever metadata the adapter knows
   (context window, pricing, capability flags).

**OpenAI-compatible HTTP vendor?** Subclass `OpenAICompatibleRuntime`
instead — the base implements `execute`, `list_models`, `reset`,
`close`, error classification, and envelope-unwrap. The subclass is
~30 lines: ClassVars + `_resolve_api_key()`. See
`src/airframe/adapters/opencode_zen.py`.

**SDK-based vendor (subprocess / native types)?** Inherit
`AgentRuntime` directly and implement all five methods. See
`src/airframe/adapters/claude_code.py` for the canonical example.

## Third-party adapters

Adapters can live in their own package and be discovered via the
`airframe.adapters` entry-point group:

```toml
# pyproject.toml of, say, airframe-adapters-together
[project.entry-points."airframe.adapters"]
together = "airframe_adapters_together:TogetherRuntime"
```

Once installed, `airframe.list_providers()` picks the runtime up
automatically — same pip-extras filtering applies as for built-ins.

To run the shared conformance contracts against a third-party
adapter:

```bash
pip install airframe-agents[testing]
```

`airframe.testing.contracts` exposes ~25 structural tests covering
Phase 0–5: lifecycle (`close()` idempotency), `unwrap()` escape
hatch, `supports()` purity, session factory shape, and per-phase
capability-vs-API agreement (`tools=` raises iff
`TOOLS_FUNCTION=False`, `on_permission=` raises iff
`PERMISSION_CALLBACK=False`, etc.).

```python
# tests/test_my_adapter_conformance.py
import pytest
from airframe.testing.contracts import (
    test_close_is_idempotent,
    test_session_factory_returns_agent_session,
    test_session_tools_kwarg_agrees_with_tools_function_capability,
    test_session_mcp_servers_kwarg_agrees_with_transport_capabilities,
    test_session_on_permission_agrees_with_permission_callback_capability,
    test_session_on_event_agrees_with_lifecycle_hooks_capability,
    test_session_rejects_wrong_provider_options_namespace,
    # ...full list in airframe.testing.contracts.__all__
)
from airframe_adapters_together import TogetherRuntime

@pytest.fixture
def adapter_runtime():
    return TogetherRuntime(api_key="test-key")
```

For behavioural coverage against live vendor endpoints,
`airframe.testing.integration` provides the same import-into-suite
pattern with `pytest.mark.integration` gating. Run with
`pytest -m integration` after configuring the relevant adapter's
credentials. Tests `pytest.skip` themselves when credentials are
absent — the suite stays usable on partially-configured machines.

Modelled on SQLAlchemy's `testing.suite` pattern. See
`tests/test_claude_code_conformance.py` and
`tests/test_claude_code_integration.py` for the canonical in-tree
examples.

## Development

```bash
uv sync --all-extras --group dev
make test       # All tests
make lint       # Ruff
make typecheck  # mypy
make ci         # Lint + format + typecheck + test
```

## License

MIT — see [LICENSE](LICENSE).
