# API reference

Hand-curated reference for everything exported from `airframe`.
Cross-links into the source files for full docstrings.

```python
import airframe
# All public names listed under "Top-level exports" below are
# importable directly from `airframe`.
```

## Runtime classes

| Class | Provider ID | See |
|---|---|---|
| `ClaudeCodeRuntime` | `claude` | [adapters/claude.md](./adapters/claude.md) |
| `CopilotRuntime` | `github-copilot` | [adapters/copilot.md](./adapters/copilot.md) |
| `CodexRuntime` | `codex` | [adapters/codex.md](./adapters/codex.md) |
| `OpenCodeZenRuntime` | `opencode` | [adapters/opencode-zen.md](./adapters/opencode-zen.md) |
| `OpenCodeGoRuntime` | `opencode-go` | [adapters/opencode-go.md](./adapters/opencode-go.md) |
| `OpenRouterRuntime` | `openrouter` | [adapters/openrouter.md](./adapters/openrouter.md) |
| `BedrockRuntime` | `bedrock` | [adapters/bedrock.md](./adapters/bedrock.md) |

Every runtime implements the `AgentRuntime` protocol.

## `AgentRuntime` protocol

`src/airframe/protocol.py`. Eight methods + one attribute:

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

- **`execute`** — single-turn convenience over `session().execute() + close()`.
- **`session`** — multi-turn `AgentSession` factory (see below).
- **`reset`** — no-op on every built-in adapter today (the runtime
  is sessionless).
- **`close`** — idempotent, never raises. Safe in `finally` blocks.
- **`validate_binding`** — cheap predicate; doesn't make network calls.
- **`list_models`** — hits the vendor's models endpoint; requires
  credentials. Raises `RuntimeAuthError` / `RuntimeTransientError`.
- **`supports`** — capability predicate. See [capabilities.md](./capabilities.md).
- **`unwrap`** — vendor-native escape hatch. Returns the
  underlying SDK object; raises `TypeError` for unsupported casts.

## `AgentSession` protocol

`src/airframe/protocol.py`. Owns per-conversation state.

```python
class AgentSession(Protocol):
    id: str | None  # vendor session id; None for stateless adapters

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

    def stream(
        self,
        prompt: Prompt,
        *,
        schema: type[BaseModel] | None = None,
        thinking: ThinkingMode = None,
        max_turns: int | None = None,
        max_budget_usd: float | None = None,
        timeout: float = 600.0,
    ) -> AsyncIterator[RuntimeEvent]: ...

    async def cancel(self) -> None: ...
    async def close(self) -> None: ...
    def unwrap(self, cls: type[T]) -> T: ...
```

- **`id`** — surfaces the vendor session/thread id when one exists;
  `None` on OpenAI-compat (no server-side session). Seeded from
  `resume=` before the first turn; populated from vendor events
  after.
- **`execute` / `stream`** — multi-turn paths. Both accept the same
  per-turn kwargs (`max_turns=`, `max_budget_usd=` are the budget
  caps).
- **`cancel`** — no-op when no turn is in flight; mid-turn raises
  `RuntimeCancelledError` on the awaiting call.
- **`close`** — idempotent, never raises. Safe in `finally`.
- **`unwrap`** — session-level escape hatch. Use
  `session.unwrap(ClaudeSDKClient)` for the live SDK client (after
  the first turn).

## Discovery

`src/airframe/discovery.py`.

```python
from airframe import list_providers, runtime_for

# Default: filtered by which extras are installed.
installed = list_providers()  # → list[str] of provider IDs

# Include declined-but-known providers (for UI menus).
all_known = list_providers(installed_only=False)

cls = runtime_for("claude")  # → ClaudeCodeRuntime class
rt = cls()
```

`list_providers()` walks the `airframe.adapters` entry-point group
plus the built-in `airframe.adapters.*` module scan. Third-party
adapters discoverable via setuptools entry points.

## Bindings

```python
from airframe import ProviderModel

binding = ProviderModel("claude", "claude-haiku-4-5")
binding.label   # "claude/claude-haiku-4-5"
binding.to_dict()  # {"providerID": "...", "modelID": "..."}
```

Frozen dataclass. The two-string pair is airframe's primitive for
"which vendor + model".

## Results & cost

```python
from airframe import RuntimeResult, CostRecord

# RuntimeResult fields:
result.text         # str
result.structured   # dict | None
result.cost         # CostRecord
result.finish       # "stop" | "length" | "tool_calls" | "end_turn" | None
result.raw          # vendor-specific result object (opaque to consumers)

# CostRecord fields:
result.cost.provider_id      # str
result.cost.model_id         # str
result.cost.cost_usd         # float | None (None when vendor doesn't compute / unknown model)
result.cost.input_tokens     # int
result.cost.output_tokens    # int
result.cost.cache_read_tokens
result.cost.cache_write_tokens
result.cost.reasoning_tokens
result.cost.finish           # mirror of result.finish
result.cost.to_dict()        # log-friendly serialisation
```

Both `RuntimeResult` and `CostRecord` are frozen dataclasses.

## Streaming events

`src/airframe/events.py`.

```python
from airframe import (
    RuntimeEvent,  # the union
    TextDelta, ReasoningDelta, ToolCallStart, ToolCallResult, TurnComplete,
)

async for event in session.stream(prompt):
    match event:
        case TextDelta(text):
            ...
        case ReasoningDelta(text):
            ...  # hidden chain-of-thought, distinct from TextDelta
        case ToolCallStart(tool_name, tool_call_id, arguments_preview):
            ...
        case ToolCallResult(tool_call_id, output, is_error):
            ...
        case TurnComplete(result):
            ...  # always the last event; carries the same RuntimeResult execute() would return
```

The variant set is shape-locked. Adding new variants later is safe
(consumers branch with a wildcard); renaming or removing is a
major-version break.

Concatenating all `TextDelta.text` for one turn equals
`turn_complete.result.text`. Exactly one `TurnComplete` per
successful stream.

## Inputs (polymorphic prompts)

`src/airframe/inputs.py`.

```python
from airframe import Prompt, PromptPart, ImageInput, FileInput

# Prompt = str | list[PromptPart]
# PromptPart = str | ImageInput | FileInput

await session.execute("plain text prompt")
await session.execute(["caption this:", ImageInput(path="/tmp/x.png")])
await session.execute(["summarise:", FileInput(path="/tmp/spec.pdf")])

# ImageInput accepts:
ImageInput(path="/tmp/x.png")
ImageInput(bytes_=b"\x89PNG...", media_type="image/png")
ImageInput(url="https://example.com/x.png")

# FileInput is path-based today; bytes_ deferred.
FileInput(path="/tmp/spec.pdf")
```

Per-adapter input support: see
[capabilities.md](./capabilities.md#vision_input) and
[capabilities.md](./capabilities.md#file_input).

## Thinking (reasoning effort)

`src/airframe/thinking.py`.

```python
from airframe import ThinkingMode, ReasoningEffort

# ThinkingMode = None | ReasoningEffort | dict
# ReasoningEffort = Literal["minimal", "low", "medium", "high", "disabled"]

await session.execute(prompt, thinking="medium")
await session.execute(prompt, thinking="disabled")  # explicitly off
await session.execute(prompt, thinking=None)         # default for the model

# Claude-only — explicit token budget:
await session.execute(prompt, thinking={"budget_tokens": 5000})
```

`"minimal"` is coerced to `"low"` on adapters that don't support
it (Claude, Copilot) with a debug log. The dict form raises
`UnsupportedFeatureError` on non-Claude adapters.

## Function tools

`src/airframe/tools.py`.

```python
from airframe import FunctionTool
from pydantic import BaseModel

class AddParams(BaseModel):
    a: float
    b: float

async def add(p: AddParams) -> float:
    return p.a + p.b

tool = FunctionTool(
    name="add",
    description="Add two numbers and return the sum.",
    params=AddParams,
    handler=add,
)

sess = runtime.session(tools=[tool])
```

`handler` is awaited with the validated Pydantic instance.
Returns are JSON-serialised back to the model. Exceptions become
`is_error=True` tool results so the model can recover.

## MCP server refs

```python
from airframe import McpServerRef

McpServerRef(name="docs", transport="stdio", command=["uvx", "mcp-server-docs"])
McpServerRef(name="docs", transport="http", url="https://mcp.example.com",
             auth_token="bearer-tok", headers={"X-Trace": "abc"})
McpServerRef(name="docs", transport="sse", url="https://sse.example.com")

sess = runtime.session(mcp_servers=[...])
```

Per-adapter transport support: see
[capabilities.md](./capabilities.md#tools_mcp_stdio-_http-_sse).
Adapters decline unsupported transports with
`UnsupportedFeatureError(feature=Feature.TOOLS_MCP_<TRANSPORT>)`.

## Permission callbacks

`src/airframe/permission.py`.

```python
from airframe import PermissionCallback, PermissionDecision, PermissionRequest

class ApproveAll(PermissionCallback):
    async def handle(self, req: PermissionRequest) -> PermissionDecision:
        print(f"tool={req.tool_name!r} args={req.tool_args!r} reason={req.reason!r}")
        return "allow"

sess = runtime.session(on_permission=ApproveAll())

# PermissionDecision = Literal["allow", "deny", "defer"]
# - "defer" falls through to the vendor's default policy
```

Per-adapter shape: per-call on Claude / Copilot; session-wide on
Codex (callback fires once to derive policy enum); declined on
OpenAI-compat.

## Lifecycle hooks

`src/airframe/hooks.py`.

```python
from airframe import HookEvent, HookEventKind

def observer(event: HookEvent) -> None:
    print(f"[{event.kind}] session={event.session_id} {event.payload}")

sess = runtime.session(on_event=observer)

# HookEventKind = Literal[
#     "session_start", "session_end", "user_prompt_submit",
#     "pre_tool_use", "post_tool_use", "tool_failure",
#     "pre_compact", "rate_limit",
# ]
```

Synchronous — don't block. Raising observers are caught and
debug-logged; the session continues.

Per-adapter emittable subset on `<RuntimeClass>.EMITTABLE_HOOK_KINDS`.
See [capabilities.md](./capabilities.md#lifecycle_hooks).

## Budget caps

```python
await session.execute(prompt, max_turns=10, max_budget_usd=0.05)
```

Both caps are cumulative across all turns of the session, checked
at the start of every turn. Trip `RuntimeBudgetExceededError`:

```python
try:
    await session.execute(prompt, max_budget_usd=0.05)
except RuntimeBudgetExceededError as exc:
    print(f"kind={exc.kind} cap={exc.cap} current={exc.current}")
    # kind: "usd" | "turns"
```

## Provider options (vendor namespaces)

`src/airframe/options.py`.

```python
from airframe import ClaudeOptions, CopilotOptions, CodexOptions, OpenAICompatOptions

sess = runtime.session(provider_options=ClaudeOptions(strict_mcp_config=True))
```

Tagged union — passing the wrong namespace raises
`UnsupportedFeatureError` at the adapter boundary.

Per-namespace fields: see each adapter page
([Claude](./adapters/claude.md#claudeoptions-provider-options-namespace),
[Copilot](./adapters/copilot.md#copilotoptions-provider-options-namespace),
[Codex](./adapters/codex.md#codexoptions-provider-options-namespace),
[OpenAI-compat](./adapters/opencode-zen.md#openaicompatoptions-provider-options-namespace)).

## Errors

`src/airframe/errors.py`. All inherit from `AgentRuntimeError`.

| Error | Meaning |
|---|---|
| `RuntimeAuthError` | Credentials bad / expired / missing |
| `RuntimeModelNotFoundError` | Server doesn't serve that model on this binding |
| `RuntimeTransientError` | 5xx, rate limit, brief outage — recoverable |
| `RuntimeStructuredOutputError` | Transport OK but model didn't produce schema-matching payload |
| `RuntimeContextOverflowError` | Prompt exceeded the model's context window |
| `RuntimeProtocolError` | Adapter saw something it can't interpret (adapter/SDK bug) |
| `RuntimeServerStartError` | Adapter couldn't bring its backend up |
| `RuntimeCancelledError` | Caller-initiated abort |
| `RuntimeBudgetExceededError` | `max_turns=` / `max_budget_usd=` cap tripped. Attributes: `cap`, `current`, `kind` |
| `UnsupportedFeatureError` | Capability declined. Attribute: `feature` (the `Feature` enum value, or its string) |
| `UnsupportedBindingError` | Adapter can't serve a `(provider_id, model_id)` binding |

What to *do* with each (retry / fallback / surface / escalate) is
consumer policy. Airframe doesn't ship a retry / fallback engine.

## Model info

```python
from airframe import ModelInfo

# Returned by runtime.list_models() — list[ModelInfo].
m.id              # str
m.display_name    # str
m.context_window  # int | None
m.input_per_1k    # float | None
m.output_per_1k   # float | None
m.capabilities    # frozenset[str] — see CAPABILITY_* constants
```

Capability constants (re-exported at top level for branching on
`ModelInfo.capabilities`):

- `CAPABILITY_STRUCTURED_OUTPUT`
- `CAPABILITY_STREAMING`
- `CAPABILITY_TOOLS`
- `CAPABILITY_VISION`
- `CAPABILITY_REASONING_EFFORT`

These are static strings that mirror per-model marketing capability
flags vendors expose on their models endpoint. They're *not* the
same as `Feature` (which is the runtime-level protocol-API
capability flag).

## Testing scaffolding

```python
from airframe.testing.contracts import (...)        # structural conformance
from airframe.testing.integration import (...)      # behavioural conformance
```

See [adapters/third-party.md](./adapters/third-party.md) for the
import-into-suite pattern used by both the in-tree and
third-party adapters.

## Top-level exports

Everything below is importable from `airframe`:

```
AgentRuntime, AgentSession, AgentRuntimeError
ClaudeCodeRuntime, CopilotRuntime, CodexRuntime,
OpenCodeZenRuntime, OpenCodeGoRuntime, OpenRouterRuntime, BedrockRuntime
ClaudeOptions, CopilotOptions, CodexOptions, OpenAICompatOptions, BedrockOptions, ProviderOptions
ProviderModel, RuntimeResult, ModelInfo, CostRecord
Feature
Prompt, PromptPart, ImageInput, FileInput
ThinkingMode, ReasoningEffort
FunctionTool, McpServerRef
PermissionCallback, PermissionDecision, PermissionRequest
HookEvent, HookEventKind
RuntimeEvent, TextDelta, ReasoningDelta, ToolCallStart, ToolCallResult, TurnComplete
RuntimeAuthError, RuntimeBudgetExceededError, RuntimeCancelledError,
RuntimeContextOverflowError, RuntimeModelNotFoundError, RuntimeProtocolError,
RuntimeServerStartError, RuntimeStructuredOutputError, RuntimeTransientError,
UnsupportedFeatureError, UnsupportedBindingError
CAPABILITY_STRUCTURED_OUTPUT, CAPABILITY_STREAMING, CAPABILITY_TOOLS,
CAPABILITY_VISION, CAPABILITY_REASONING_EFFORT
list_providers, runtime_for
__version__
```
