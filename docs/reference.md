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
| `BedrockRuntime` | `bedrock` | [adapters/bedrock.md](./adapters/bedrock.md) |
| `ClaudeCodeRuntime` | `claude` | [adapters/claude.md](./adapters/claude.md) |
| `CopilotRuntime` | `github-copilot` | [adapters/copilot.md](./adapters/copilot.md) |
| `OpenCodeGoRuntime` | `opencode-go` | [adapters/opencode-go.md](./adapters/opencode-go.md) |
| `OpenCodeServerRuntime` | `opencode` | [adapters/opencode-server.md](./adapters/opencode-server.md) |
| `OpenCodeZenRuntime` | `opencode-zen` | [adapters/opencode-zen.md](./adapters/opencode-zen.md) |
| `OpenRouterRuntime` | `openrouter` | [adapters/openrouter.md](./adapters/openrouter.md) |
| `ZaiAnthropicRuntime` | `zai-anthropic` | [adapters/zai-anthropic.md](./adapters/zai-anthropic.md) |

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
        metadata: RequestMetadata | None = None,
        cache: CacheConfig | None = None,
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
        metadata: RequestMetadata | None = None,
        cache: CacheConfig | None = None,
        slash_commands: SlashCommandsConfig | None = None,
    ) -> AgentSession: ...

    async def reset(self) -> None: ...
    async def close(self) -> None: ...
    def validate_binding(self, binding: ProviderModel) -> bool: ...
    async def list_models(self) -> list[ModelInfo]: ...
    async def count_tokens(
        self,
        prompt: Prompt,
        *,
        system: str | None = None,
        model: ProviderModel | None = None,
    ) -> int: ...
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
- **`count_tokens`** — pre-flight tokeniser-accurate count for a
  prompt against the runtime's model. Claude hits
  `messages.count_tokens`; OpenAI-compat uses `tiktoken`. Adapters
  not declaring `Feature.COUNT_TOKENS` raise
  `UnsupportedFeatureError`. See [capabilities.md#count_tokens](./capabilities.md#count_tokens).
- **`supports`** — capability predicate. See [capabilities.md](./capabilities.md).
- **`unwrap`** — vendor-native escape hatch. Returns the
  underlying SDK object; raises `TypeError` for unsupported casts.

**New session kwargs** (Phase 6): `metadata=`, `cache=`,
`slash_commands=` — see [`RequestMetadata`](#request-metadata),
[`CacheConfig`](#cache-config), and
[`SlashCommandsConfig`](#slash-commands) below.

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

    async def list_slash_commands(self) -> list[SlashCommand]: ...
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
- **`list_slash_commands`** — discover user-authored slash commands
  from the filesystem. Returns `list[SlashCommand]` (name +
  description + body template + source_path + frontmatter). See
  [`SlashCommandsConfig`](#slash-commands) below for the search
  paths + filtering.
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
result.reasoning    # str | None — model's finalised reasoning trace
                    #   (Feature.REASONING_OUTPUT); see capabilities.md
result.rate_limit   # RateLimitInfo | None — typed quota snapshot
                    #   (Feature.RATE_LIMIT_TELEMETRY); see below
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

Per-adapter shape: per-call on Claude / Copilot; declined on
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

## Rate-limit telemetry

`src/airframe/rate_limit.py`. Typed quota snapshot surfaced on
successful calls and on throttle errors.

```python
from airframe import RateLimitInfo, RateLimitWindow

# RuntimeResult.rate_limit — populated when the vendor sent quota
# data on a successful call (None otherwise).
info: RateLimitInfo | None = result.rate_limit

# RuntimeTransientError.rate_limit — populated when a 429 carried
# quota data (None for 5xx / network blips).
try:
    await session.execute(prompt)
except RuntimeTransientError as exc:
    if exc.rate_limit is not None:
        for w in exc.rate_limit.windows:
            print(f"{w.name}: {w.remaining}/{w.limit}, reset {w.reset_at}")

# RateLimitWindow fields:
window.name                    # vendor's window id ("requests", "tokens",
                               # "five_hour", "seven_day", ...)
window.remaining               # int | None
window.limit                   # int | None
window.utilization             # float | None (0.0-1.0; Claude-style)
window.reset_at                # datetime | None
window.retry_after_seconds     # float | None
window.status                  # "allowed" | "allowed_warning" | "rejected" | None
```

Adapters declaring `Feature.RATE_LIMIT_TELEMETRY`: Claude, OpenAI-compat.
See [capabilities.md](./capabilities.md#rate_limit_telemetry).

## Request metadata

`src/airframe/metadata.py`. Per-request observation tag forwarded
to the vendor for abuse detection / per-tenant attribution / audit.

```python
from airframe import RequestMetadata

md = RequestMetadata(
    user_id="acct_123",                  # → OpenAI user=, Claude metadata.user_id
    request_id="req-abc",                # → X-Request-ID header on OAI-compat
    tags={"tenant": "acme", "env": "prod"},  # → OpenAI metadata= dict
)
sess = runtime.session(metadata=md)
# or per-call:
result = await runtime.execute(prompt, metadata=md)
```

**Soft contract** — adapters returning False on
`Feature.REQUEST_METADATA` silently drop the kwarg rather than
raising. The call still succeeds; only the attribution tag is
lost. See [capabilities.md](./capabilities.md#request_metadata).

## Cache config

`src/airframe/cache.py`. Portable prompt-cache key for vendors that
expose explicit cache control.

```python
from airframe import CacheConfig

cfg = CacheConfig(
    key="agent-foo:session-42",          # stable identifier
    retention="long",                     # "short" (5min) | "long" (24h+)
)
sess = runtime.session(cache=cfg)
```

Per-adapter mapping (OpenAI-compat): `key` → `prompt_cache_key=`;
`retention="short"` → `prompt_cache_retention="in_memory"`;
`retention="long"` → `prompt_cache_retention="24h"`. The portable
`cache=` takes precedence over the OpenAI-specific
`OpenAICompatOptions.prompt_cache_key` when both are set.

**Soft contract** — non-supporting adapters silently drop the
kwarg (the call succeeds without the cache speed-up). See
[capabilities.md](./capabilities.md#prompt_cache_control).

## Slash commands

`src/airframe/slash_commands.py`. Filesystem discovery of
user-authored slash commands — for rendering a palette UI.

```python
from airframe import SlashCommand, SlashCommandsConfig
from airframe.slash_commands import discover

# Session-level config controls which directories to search.
config = SlashCommandsConfig(
    enabled="all",                       # "all" | list[str] | None
    search_paths=[Path("./my-commands")],
    include_user_global=True,            # ~/.claude/commands/, etc.
)
sess = runtime.session(slash_commands=config)
cmds: list[SlashCommand] = await sess.list_slash_commands()
for cmd in cmds:
    print(f"/{cmd.name} — {cmd.description}")
    print(cmd.body)            # template; substitute args yourself

# Module-level discover() — for use outside a session.
cmds = discover(config, cwd=Path("/path/to/project"))

# SlashCommand fields:
cmd.name              # str — file stem or frontmatter "name:" override
cmd.description       # str | None — frontmatter "description:"
cmd.body              # str — Markdown body (template; substitute placeholders)
cmd.source_path       # Path — for "edit this command" UX
cmd.frontmatter       # dict[str, str] — raw frontmatter values
```

Discovery walks `.claude/commands/`, `.opencode/command/`,
`.agents/commands/` upward from `cwd` to the git worktree root,
plus the matching user-global paths. Later-found (more specific)
paths win on name collision.

**Invocation differs per adapter.** Claude's SDK auto-expands
`/commandname args` natively when passed through `execute()` — the
SDK reads the same files airframe discovers. Other adapters: the
consumer substitutes placeholders (`$ARGUMENTS`, `$1`, `{file}`,
etc.) into `cmd.body` and calls `execute(expanded_text)`. See
[capabilities.md](./capabilities.md#slash_commands).

## Provider options (vendor namespaces)

`src/airframe/options.py`.

```python
from airframe import (
    BedrockOptions,
    ClaudeOptions,
    CopilotOptions,
    OpenAICompatOptions,
    OpenCodeServerOptions,
)

sess = runtime.session(provider_options=ClaudeOptions(strict_mcp_config=True))
```

Tagged union — passing the wrong namespace raises
`UnsupportedFeatureError` at the adapter boundary.

Per-namespace fields: see each adapter page
([Bedrock](./adapters/bedrock.md#bedrockoptions),
[Claude](./adapters/claude.md#claudeoptions-provider-options-namespace),
[Copilot](./adapters/copilot.md#copilotoptions-provider-options-namespace),
[OpenAI-compat](./adapters/opencode-zen.md#openaicompatoptions-provider-options-namespace),
[OpenCode Server](./adapters/opencode-server.md#opencodeserveroptions)).

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
BedrockRuntime, ClaudeCodeRuntime, CopilotRuntime,
OpenCodeGoRuntime, OpenCodeServerRuntime, OpenCodeZenRuntime, OpenRouterRuntime
BedrockOptions, ClaudeOptions, CopilotOptions,
OpenAICompatOptions, OpenCodeServerOptions, ProviderOptions
ProviderModel, RuntimeResult, ModelInfo, CostRecord
Feature
Prompt, PromptPart, ImageInput, FileInput
ThinkingMode, ReasoningEffort
FunctionTool, McpServerRef
PermissionCallback, PermissionDecision, PermissionRequest
HookEvent, HookEventKind
RuntimeEvent, TextDelta, ReasoningDelta, ToolCallStart, ToolCallResult, TurnComplete
RateLimitInfo, RateLimitWindow                          # Phase 6
RequestMetadata                                          # Phase 6
CacheConfig                                              # Phase 6
SlashCommand, SlashCommandsConfig                        # Phase 6
RuntimeAuthError, RuntimeBudgetExceededError, RuntimeCancelledError,
RuntimeContextOverflowError, RuntimeModelNotFoundError, RuntimeProtocolError,
RuntimeServerStartError, RuntimeStructuredOutputError, RuntimeTransientError,
UnsupportedFeatureError, UnsupportedBindingError
CAPABILITY_STRUCTURED_OUTPUT, CAPABILITY_STREAMING, CAPABILITY_TOOLS,
CAPABILITY_VISION, CAPABILITY_REASONING_EFFORT
list_providers, runtime_for
__version__
```
