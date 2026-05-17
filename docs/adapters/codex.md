# CodexRuntime

Routes work through OpenAI's Codex models via the official
[`openai-codex-sdk`](https://pypi.org/project/openai-codex-sdk/)
package. The SDK spawns a `codex` CLI subprocess per turn.

| | |
|---|---|
| **PROVIDER_ID** | `codex` |
| **Pip extra** | `airframe-agents[codex]` |
| **Vendor SDK** | `openai-codex-sdk` |
| **Transport** | Subprocess per turn (JSONL events) |
| **Authentication** | See [auth.md](../auth.md#codexruntime) |

## Install

```bash
pip install airframe-agents[codex]
```

## Quickstart

```python
from airframe import CodexRuntime, ProviderModel
from pydantic import BaseModel

class Brief(BaseModel):
    summary: str

runtime = CodexRuntime()
result = await runtime.execute(
    "Brief me on the project structure.",
    schema=Brief,
    model=ProviderModel("codex", "gpt-5-codex"),
)
print(result.structured)
await runtime.close()
```

## Supported features

| Feature | Status | Notes |
|---|---|---|
| `STRUCTURED_OUTPUT_JSON_SCHEMA` | ✓ | Native via `TurnOptions.outputSchema` → CLI `--output-schema` |
| `STREAMING` | ✓ | `Thread.run_streamed()` |
| `SESSION_RESUME` | ✓ | `Codex.resume_thread(<thread_id>)` |
| `CANCEL` | ✓ | `AbortController` on `TurnOptions.signal` |
| `REASONING_EFFORT` | ✓ | `ThreadOptions.model_reasoning_effort` |
| `REASONING_BUDGET_TOKENS` | ✗ | No token-budget channel on Codex |
| `VISION_INPUT` | ✓ | `LocalImageInput(path)` |
| `FILE_INPUT` | ✓ | Via working-directory + Read tool |
| `TOOLS_FUNCTION` | ✗ | **Declined permanently** — the Codex Python SDK has no tool-registration API. Configure tools through `~/.codex/config.toml` instead |
| `TOOLS_MCP_STDIO` | ✗ | **Declined permanently** — same reason: no programmatic MCP-registration channel. Use `~/.codex/config.toml`'s `[[mcp_servers]]` block |
| `TOOLS_MCP_HTTP` | ✗ | Same as STDIO |
| `TOOLS_MCP_SSE` | ✗ | Same |
| `PERMISSION_CALLBACK` | ✓ (session-wide) | Codex's `approval_policy` is per-session, not per-call. The user's callback fires **once** at first `execute()` with a sentinel `PermissionRequest` to derive the policy enum (`"never"` / `"untrusted"` / `"on-request"` / `"on-failure"`); the enum applies for the session's lifetime |
| `LIFECYCLE_HOOKS` | ✓ | 6 of 8 kinds emittable (no `pre_compact`, no `rate_limit` — Codex SDK has neither event) |
| `BUDGET_USD_CAP` | ✓ | Client-side accumulation against the pricing table |
| `BUDGET_TURN_CAP` | ✓ | Client-side per-session counter |

`STRUCTURED_OUTPUT_STRICT` stays False.

## CodexOptions (provider-options namespace)

```python
from airframe import CodexRuntime, CodexOptions

runtime = CodexRuntime()
sess = runtime.session(
    provider_options=CodexOptions(
        working_directory="/workspaces/my-repo",
        additional_directories=("/etc/configs",),
        network_access_enabled=True,
        web_search_enabled=True,
    )
)
```

| Field | Type | Maps to |
|---|---|---|
| `working_directory` | `str \| None` | `ThreadOptions.workingDirectory` — override the Codex CLI cwd |
| `additional_directories` | `tuple[str, ...]` | `ThreadOptions.additionalDirectories` — extra dirs the sandboxed shell may read/write |
| `network_access_enabled` | `bool` | `ThreadOptions.networkAccessEnabled` — let the sandboxed shell make network calls |
| `web_search_enabled` | `bool` | `ThreadOptions.webSearchEnabled` — allow the built-in web-search tool |

All four bake at `start_thread()` / `resume_thread()` time;
namespace fingerprint joins the thread cache key.

## Model IDs

`CodexRuntime.list_models()` filters OpenAI's `/v1/models` endpoint
down to codex-shaped IDs (the raw endpoint returns every model the
account has access to, including embeddings). Common picks:

- `gpt-5-codex` — default; the standard codex model.
- `gpt-5-codex-mini` — smaller / cheaper.
- `o5-codex` — reasoning-optimised variant.

**Claude models are rejected.** `validate_binding()` returns False
for any `model_id` starting with `claude-`. Codex is OpenAI-only
by design.

## Structured output

Native — passes `schema.model_json_schema()` via
`TurnOptions.outputSchema` (translates to the CLI's
`--output-schema` flag). The codex CLI enforces the schema on the
final response; the adapter parses `Turn.final_response` as JSON
into `RuntimeResult.structured`.

## Cost reporting

Codex returns token counts but no per-call USD cost. The adapter
ships a stub pricing map (`_PRICING` in `codex.py`) and computes
`cost_usd = (in_tokens / 1000) * in_rate + (out_tokens / 1000) * out_rate`.
Unknown model IDs return `cost_usd=None` with tokens populated.

## Vendor quirks & landmines

- **No tool registration API.** Both `tools=` and `mcp_servers=`
  raise `UnsupportedFeatureError` with an actionable pointer at
  `~/.codex/config.toml`. The decline is *permanent* —
  airframe-side wiring would require Codex SDK changes.
- **Per-turn subprocess.** Each `thread.run()` spawns a fresh
  `codex exec` subprocess and drains its JSONL event stream.
  Multi-turn within one `session` reuses the `Thread` object but
  every turn pays the subprocess startup cost.
- **`approval_policy` is session-wide.** Codex's permission model
  is coarser than Claude / Copilot. The user's `PermissionCallback`
  fires *once* per session at first `execute()`, with a sentinel
  `PermissionRequest(tool_name="*", ...)` — the returned decision
  maps to `approval_policy: "never" | "untrusted" | "on-request" | "on-failure"`.
  Per-call interception isn't possible through the SDK; document
  this clearly to consumers expecting Claude-style fidelity.
- **Sandbox mode lives on the runtime, not in `CodexOptions`.**
  `CodexRuntime(sandbox_mode="workspace-write")` — security-relevant
  default that shouldn't vary per session.
- **The codex CLI reads `~/.codex/auth.json` directly.** Auth chain
  step 4 isn't airframe code — the subprocess just inherits the
  user's CLI login. Means `CodexRuntime()` with no env vars may
  still work if the user has run `codex login` previously.

## Native escape hatches

```python
from openai_codex_sdk import Codex, Thread
sess = runtime.session()
await sess.execute("hi")
thread: Thread = sess.unwrap(Thread)
print(thread.id)
client: Codex = runtime.unwrap(Codex)
```

## See also

- [auth.md](../auth.md#codexruntime)
- [capabilities.md](../capabilities.md)
- [OpenAI Codex SDK docs](https://github.com/openai/codex-cli)
