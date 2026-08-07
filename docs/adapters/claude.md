# ClaudeCodeRuntime

Routes work through Anthropic's Claude family via the official
[`claude-agent-sdk`](https://pypi.org/project/claude-agent-sdk/)
package. The SDK spawns and manages the `claude` CLI subprocess;
airframe doesn't allocate ports, juggle credentials, validate model
IDs at startup, or maintain any client code.

| | |
|---|---|
| **PROVIDER_ID** | `claude` |
| **Pip extra** | `airframe-agents[claude]` |
| **Vendor SDK** | `claude-agent-sdk` |
| **Transport** | Subprocess + JSON-RPC (one `ClaudeSDKClient` per session) |
| **Authentication** | See [auth.md](../auth.md#claudecoderuntime) |

## Install

```bash
pip install airframe-agents[claude]
```

## Quickstart

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
print(result.structured)
await runtime.close()
```

## Supported features

| Feature | Status | Notes |
|---|---|---|
| `STRUCTURED_OUTPUT_JSON_SCHEMA` | ✓ | Native — `output_format={"type":"json_schema",...}` baked at connect time |
| `STREAMING` | ✓ | `include_partial_messages=True` → `StreamEvent` |
| `SESSION_RESUME` | ✓ | `ClaudeAgentOptions.resume=<session_id>` |
| `CANCEL` | ✓ | `ClaudeSDKClient.interrupt()` |
| `REASONING_EFFORT` | ✓ | `effort=` on `ClaudeAgentOptions` |
| `REASONING_BUDGET_TOKENS` | ✓ | Only adapter that supports the `{"budget_tokens": N}` shape |
| `VISION_INPUT` | ✓ | Via the Read tool (filesystem paths) |
| `FILE_INPUT` | ✓ | Same Read-tool path |
| `TOOLS_FUNCTION` | ✓ | In-process MCP server via `create_sdk_mcp_server()` |
| `TOOLS_MCP_STDIO` | ✓ | All three transports natively supported |
| `TOOLS_MCP_HTTP` | ✓ | |
| `TOOLS_MCP_SSE` | ✓ | Only adapter that supports SSE |
| `TOOLS_NATIVE` | ✓ | Hosted `WebSearch` / `WebFetch` via `native_tools=` → `allowed_tools` (`supported_native_tools()` → `{WEB_SEARCH, WEB_FETCH}`) |
| `PERMISSION_CALLBACK` | ✓ | Per-call via a native `PreToolUse` hook (`can_use_tool` is inert under `permission_mode="bypassPermissions"`) |
| `LIFECYCLE_HOOKS` | ✓ | All 8 kinds emittable (native `PreCompact` + `RateLimit` events) |
| `BUDGET_USD_CAP` | ✓ | Client-side accumulation against `total_cost_usd` |
| `BUDGET_TURN_CAP` | ✓ | `ClaudeAgentOptions.max_turns` (overrides `DEFAULT_MAX_TURNS=60`) |

`STRUCTURED_OUTPUT_STRICT` stays False (Anthropic's API doesn't
expose a strict-mode toggle).

## ClaudeOptions (provider-options namespace)

```python
from airframe import ClaudeCodeRuntime, ClaudeOptions

runtime = ClaudeCodeRuntime()
sess = runtime.session(
    provider_options=ClaudeOptions(
        append_system_prompt="Always cite line numbers.",
        fork_session=True,        # combined with resume=, forks a copy
        strict_mcp_config=True,   # fail closed on unknown MCP tools
    )
)
```

| Field | Type | Maps to |
|---|---|---|
| `append_system_prompt` | `str \| None` | `ClaudeAgentOptions.append_system_prompt` — appended to (not replacing) the resolved system prompt |
| `fork_session` | `bool` | `ClaudeAgentOptions.fork_session` — when combined with `resume=<id>`, forks instead of resuming. Ignored when `resume=` is None |
| `strict_mcp_config` | `bool` | `ClaudeAgentOptions.strict_mcp_config` — reject MCP refs whose advertised tools don't match the compiled config (fail closed) |

All three fields bake at connect time, so the namespace fingerprint
joins the session cache key — a change between turns forces a
reconnect.

## Model IDs

`ClaudeCodeRuntime.list_models()` hits Anthropic's `/v1/models`
endpoint and returns the live catalogue. Common picks:

- `claude-haiku-4-5` — fastest, cheapest. Good default for
  high-volume work.
- `claude-sonnet-4-5` — middle tier; the workhorse.
- `claude-opus-4-7` — strongest, most expensive.

The vendor adds new model IDs frequently; treat `list_models()` as
the source of truth.

## Structured output

Native — passes `schema.model_json_schema()` straight to
`ClaudeAgentOptions.output_format={"type":"json_schema","schema":...}`.
The CLI enforces the schema server-side and the validated payload
lands on `ResultMessage.structured_output`. No tool-forcing, no MCP
shim, no system-prompt prefix.

## Cost reporting

The SDK exposes `total_cost_usd` on the `ResultMessage` — populated
directly into `CostRecord.cost_usd`. Token counts come from
`ResultMessage.usage`. Reasoning-token counts surface under
`usage.thinking_tokens` when extended thinking is enabled and feed
`CostRecord.reasoning_tokens`.

## Vendor quirks & landmines

- **OAuth tokens don't work for `/v1/models`.** Anthropic's models
  endpoint requires an API key. If your auth chain resolves to an
  OAuth path, `list_models()` raises `RuntimeAuthError` with an
  actionable message. Subscription users can still drive `execute()`
  / `session()` fine — the limitation is models-listing only.
- **The CLI auto-allows the Read tool** when an `ImageInput` /
  `FileInput` is in the prompt. We add it to `allowed_tools=` at
  connect time so attachments work without explicit allowlisting.
- **External MCP server names must not collide** with airframe's
  internal `_airframe_tools` MCP server name (the in-process
  function-tool dispatcher). Adapter raises `ValueError` on
  collision.
- **`max_turns` defaults to 60** when no `max_turns=` is supplied
  to `execute()`/`stream()`. Override per-call via the kwarg; cap
  the runtime-wide default via `ClaudeCodeRuntime(max_turns=N)`.

## Native escape hatches

```python
from claude_agent_sdk import ClaudeSDKClient
sess = runtime.session()
await sess.execute("hi")
client: ClaudeSDKClient = sess.unwrap(ClaudeSDKClient)
await client.interrupt()
```

Runtime-level: `runtime.unwrap(ClaudeCodeRuntime)` returns self.
Session-level: `session.unwrap(ClaudeSDKClient)` returns the live
SDK client (after the first turn has run).

## See also

- [auth.md](../auth.md#claudecoderuntime) — credential resolution
  details and CI patterns.
- [capabilities.md](../capabilities.md) — per-`Feature` semantics
  across adapters.
- [Claude Agent SDK docs](https://github.com/anthropics/claude-agent-sdk-python)
  — vendor SDK reference.
