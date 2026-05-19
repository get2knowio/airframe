# CopilotRuntime

Routes work through GitHub Copilot's models (OpenAI / GPT family
plus xAI) via the official
[`github-copilot-sdk`](https://pypi.org/project/github-copilot-sdk/)
package (imports as `copilot`). The SDK spawns and manages the
`copilot` CLI subprocess.

| | |
|---|---|
| **PROVIDER_ID** | `github-copilot` |
| **Pip extra** | `airframe-agents[copilot]` |
| **Vendor SDK** | `github-copilot-sdk` |
| **Transport** | Subprocess + JSON-RPC (one `CopilotSession` per session) |
| **Authentication** | See [auth.md](../auth.md#copilotruntime) |

## Install

```bash
pip install airframe-agents[copilot]
```

## Quickstart

```python
from airframe import CopilotRuntime, ProviderModel
from pydantic import BaseModel

class Brief(BaseModel):
    summary: str

runtime = CopilotRuntime()
result = await runtime.execute(
    "Brief me on the project structure.",
    schema=Brief,
    model=ProviderModel("github-copilot", "gpt-5-mini"),
)
print(result.structured)
await runtime.close()
```

## Supported features

| Feature | Status | Notes |
|---|---|---|
| `STRUCTURED_OUTPUT_JSON_SCHEMA` | ✓ | Forced via a hidden `submit_result` tool (Copilot has no native JSON-schema mode) |
| `STREAMING` | ✓ | `session.on(...)` deltas |
| `SESSION_RESUME` | ✓ | `CopilotClient.resume_session(<id>)` |
| `CANCEL` | ✓ | `CopilotSession.abort()` |
| `REASONING_EFFORT` | ✓ | `reasoning_effort=` on `create_session` |
| `REASONING_BUDGET_TOKENS` | ✗ | No token-budget channel on Copilot |
| `VISION_INPUT` | ✓ | `attachments=[FileAttachment{path}]` |
| `FILE_INPUT` | ✓ | Same `FileAttachment` channel |
| `TOOLS_FUNCTION` | ✓ | `define_tool()` registrations |
| `TOOLS_MCP_STDIO` | ✓ | `mcp_servers={"name": {"type":"local","command":...}}` |
| `TOOLS_MCP_HTTP` | ✓ | `mcp_servers={"name": {"type":"http","url":...}}` |
| `TOOLS_MCP_SSE` | ✗ | Declined per the plan; use `transport="http"` instead |
| `PERMISSION_CALLBACK` | ✓ | `on_permission_request=` per-call |
| `LIFECYCLE_HOOKS` | ✓ | 7 of 8 kinds emittable (no `rate_limit` — Copilot surfaces rate-limit failures as `SessionErrorData` instead) |
| `BUDGET_USD_CAP` | ✓ | Client-side accumulation against `AssistantUsageData.cost` |
| `BUDGET_TURN_CAP` | ✗ | Declined permanently — Copilot CLI caps internal turns at the vendor level (the `--max-turns` config); exposing a user-facing `max_turns=` would be misleading |

`STRUCTURED_OUTPUT_STRICT` stays False.

## CopilotOptions (provider-options namespace)

```python
from airframe import CopilotRuntime, CopilotOptions

runtime = CopilotRuntime()
sess = runtime.session(
    provider_options=CopilotOptions(
        available_tools=("read", "shell"),       # built-in allowlist
        excluded_tools=("write",),               # built-in denylist
        skill_directories=("/opt/skills",),
        working_directory="/workspaces/my-repo",
    )
)
```

| Field | Type | Maps to |
|---|---|---|
| `available_tools` | `tuple[str, ...] \| None` | `CopilotClient.create_session.available_tools` — allowlist Copilot built-in tools the model may invoke |
| `excluded_tools` | `tuple[str, ...]` | `excluded_tools=` — denylist applied after allowlist |
| `skill_directories` | `tuple[str, ...]` | `skill_directories=` — extra dirs scanned for skill packs |
| `working_directory` | `str \| None` | `working_directory=` — override the Copilot CLI's cwd |

All four fields bake at `create_session()` time; namespace
fingerprint joins the session cache key.

## Model IDs

`CopilotRuntime.list_models()` returns the live menu from
`CopilotClient.list_models()` — rich vendor metadata (display
name, context window, capability flags, billing multiplier). Common
picks:

- `gpt-5-mini` — default; balanced cost / capability.
- `gpt-5` — full-strength frontier.
- `o5-preview` — reasoning-optimised.
- `gpt-5-codex` — coding tasks.

**Claude models are rejected.** `validate_binding()` returns False
for any `model_id` starting with `claude-` — Claude via Copilot
Chat Completions emits markdown-fenced JSON instead of honouring
tool calls, so it can't satisfy airframe's structured-output
contract. Route Claude work through `ClaudeCodeRuntime`.

## Structured output

Implemented via a hidden `submit_result` tool registered with the
agent's schema via `copilot.define_tool`. The model is forced (via
a system-message append) to call `submit_result` exactly once with
the typed payload; the runtime captures the validated Pydantic
instance and returns its dict form as `RuntimeResult.structured`.

This is the same pattern many OpenAI-compat tool-using setups use
to enforce structured output without a native JSON-schema flag.
The `submit_result` tool is filtered out of `ToolCallStart` /
`ToolCallResult` streaming events and `HookEvent`s — consumers
never see the internal plumbing.

## Cost reporting

`AssistantUsageData` carries `cost` (vendor-computed USD),
`input_tokens`, `output_tokens`, `cache_read_tokens`,
`cache_write_tokens`. All five populate `CostRecord` directly.

## Vendor quirks & landmines

- **SSE MCP transport declined** with an actionable pointer at
  `transport="http"`. Copilot's wire shape supports stdio + http
  natively but not SSE; the http transport handles the same remote
  servers in most cases.
- **`submit_result` is internal plumbing** — never surfaces in
  streaming events or hooks, even though it's a real tool call
  under the hood. Consumers don't need to special-case it.
- **`BUDGET_TURN_CAP` is permanently False** — see the table above.
- **Auth chain is unique** — `use_logged_in_user=True` (default)
  uses the `gh auth login` storage, which is the right behaviour
  for local dev but surprises CI users who expect env-var-only.
  Set `GITHUB_TOKEN` explicitly in CI to avoid the fallback.

## Native escape hatches

```python
from copilot.session import CopilotSession
sess = runtime.session()
await sess.execute("hi")
vendor_sess: CopilotSession = sess.unwrap(CopilotSession)
await vendor_sess.snapshot()
```

Runtime-level: `runtime.unwrap(CopilotClient)`. Session-level:
`session.unwrap(CopilotSession)`.

## See also

- [auth.md](../auth.md#copilotruntime)
- [capabilities.md](../capabilities.md)
- [GitHub Copilot SDK docs](https://github.com/github/copilot-sdk)
