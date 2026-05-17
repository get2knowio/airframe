# OpenCodeZenRuntime

Routes work through the [opencode-go](https://github.com/opencode-ai/opencode)
Zen gateway via the standard `openai` Python SDK over HTTP. The
canonical example of the `OpenAICompatibleRuntime` base class —
any vendor speaking OpenAI's Chat Completions wire format
(Together, Groq, Fireworks, OpenRouter, vLLM, LM Studio,
Anthropic's `/v1/messages/openai` proxy) gets the same surface in
~30 lines of subclass code.

| | |
|---|---|
| **PROVIDER_ID** | `opencode` |
| **Pip extra** | `airframe-agents[openai-compat]` |
| **Vendor SDK** | `openai` (HTTP only — no subprocess) |
| **Transport** | Direct HTTP via `AsyncOpenAI` |
| **Authentication** | See [auth.md](../auth.md#opencodezenruntime) |

## Install

```bash
pip install airframe-agents[openai-compat]
```

## Quickstart

```python
from airframe import OpenCodeZenRuntime, ProviderModel
from pydantic import BaseModel

class Brief(BaseModel):
    summary: str

runtime = OpenCodeZenRuntime(api_key="opc_...")
result = await runtime.execute(
    "Brief me on the project structure.",
    schema=Brief,
    model=ProviderModel("opencode", "gpt-5-nano"),
)
print(result.structured)
await runtime.close()
```

## Supported features

| Feature | Status | Notes |
|---|---|---|
| `STRUCTURED_OUTPUT_JSON_SCHEMA` | ✓ | Native via `response_format={"type":"json_schema",...}` |
| `STREAMING` | ✓ | `stream=True` on `chat.completions.create()` |
| `SESSION_RESUME` | ✗ | Chat Completions has no server-side session — `session(resume=...)` raises `UnsupportedFeatureError` |
| `CANCEL` | ✓ | `asyncio.Task.cancel()` propagates to httpx; `stream()` cancellation via flag + `AsyncStream.close()` |
| `REASONING_EFFORT` | ✓ | `reasoning_effort=` on `chat.completions.create()`; vendor / model rejects unsupported levels |
| `REASONING_BUDGET_TOKENS` | ✗ | Claude-specific |
| `VISION_INPUT` | ✓ | OpenAI content-parts shape (`{"type":"image_url","image_url":{"url":"data:image/..;base64,.."}}`) |
| `FILE_INPUT` | ✗ | File-routing varies wildly across compat vendors; explicit per-subclass opt-in only |
| `TOOLS_FUNCTION` | ✓ | `tools=[{"type":"function",...}]` + client-side tool loop |
| `TOOLS_MCP_STDIO` | ✗ | **Declined permanently** — Chat Completions has no MCP-as-tool slot; that lives on the Responses API |
| `TOOLS_MCP_HTTP` | ✗ | Same |
| `TOOLS_MCP_SSE` | ✗ | Same |
| `PERMISSION_CALLBACK` | ✗ | **Declined permanently** — Chat Completions has no tool-permission wire shape. A future `OpenAIResponsesRuntime` (separate adapter family) could wire it |
| `LIFECYCLE_HOOKS` | ✓ | 6 of 8 kinds emittable (no `pre_compact` — chat-completions has no compaction concept; no `rate_limit` — SDK doesn't surface discrete throttle events) |
| `BUDGET_USD_CAP` | ✓ | Client-side accumulation against the pricing table |
| `BUDGET_TURN_CAP` | ✓ | Client-side per-session counter (distinct from the internal `MAX_TOOL_ITERATIONS=20` runaway guard) |

`STRUCTURED_OUTPUT_STRICT` stays False — the OpenAI SDK accepts
`strict: True` natively, but compat-vendor coverage is uneven
(Together / Groq / Fireworks handle it differently). The base
passes `strict: False` for portability; a future per-subclass
opt-in could flip it.

## OpenAICompatOptions (provider-options namespace)

Most fields are OpenAI-only — compat vendors silently ignore
unrecognised kwargs in their server-side validation, so passing
these to non-OpenAI compat endpoints is a no-op rather than an
error.

```python
from airframe import OpenCodeZenRuntime, OpenAICompatOptions

runtime = OpenCodeZenRuntime(api_key="opc_...")
sess = runtime.session(
    provider_options=OpenAICompatOptions(
        prompt_cache_key="user-42",
        prompt_cache_retention="24h",
        service_tier="priority",
        safety_identifier="user-99",
        verbosity="low",
        store=True,
    )
)
```

| Field | Type | Maps to |
|---|---|---|
| `prompt_cache_key` | `str \| None` | `prompt_cache_key=` — explicit OpenAI prompt-cache key |
| `prompt_cache_retention` | `str \| None` | `prompt_cache_retention=` — `"in_memory"` or `"24h"` |
| `service_tier` | `str \| None` | `service_tier=` — `"auto"` / `"default"` / `"flex"` / `"priority"` |
| `safety_identifier` | `str \| None` | `safety_identifier=` — opaque per-user identifier for abuse-detection routing |
| `verbosity` | `str \| None` | `verbosity=` — `"low"` / `"medium"` / `"high"` response-length hint |
| `store` | `bool \| None` | `store=` — persist request/response for Responses-API retrieval |

These are merged into **every** `chat.completions.create()` call
on both the execute and stream paths via the `_apply_provider_options()`
helper.

## Model IDs (Zen-routed)

`list_models()` hits `GET <base_url>/models` via the standard
OpenAI client. Common picks on the Zen gateway:

- `gpt-5-nano` — default; cheap.
- `gpt-5-mini` — middle tier.
- `glm-5` / `big-pickle` / `qwen3.6-plus` — non-OpenAI options
  routed through Zen.
- Free-tier models: `minimax-m2.5-free`, `deepseek-v4-flash-free`,
  `qwen3.6-plus-free`, `nemotron-3-super-free` — `cost_usd=0`.

The `METADATA` table in `opencode_zen.py` enriches known IDs with
display name, context window, and pricing. Unknown IDs come back
with sensible defaults.

## Structured output

Native via `response_format={"type":"json_schema","json_schema":{"name":"...","schema":...,"strict":false}}`.
Strict mode is off for compat-vendor portability (see the table
above).

Some Zen-routed models emit a single-key envelope around the
structured payload (`{"input": {...}}`, `{"content": "<json>"}`).
The adapter's `_unwrap_envelope` strips one level of wrapper before
Pydantic validates.

## Cost reporting

The vendor returns tokens but no per-call USD cost. The adapter
computes `cost_usd` from the `METADATA` pricing table per model
ID. Models marked free-tier return `cost_usd=0`; unknown IDs
return `cost_usd=None`.

## Vendor quirks & landmines

- **No server-side session** — `messages=[]` lives entirely on the
  client. Every `session.execute()` resends the full history; the
  adapter manages the buffer.
- **Permission callback permanently declined.** Chat Completions
  has no tool-permission wire shape — the *caller* decides whether
  to execute a returned `tool_call`. That decision sits above the
  adapter boundary, not inside it.
- **MCP servers permanently declined.** Same boundary issue — MCP
  as a tool slot lives on the Responses API only.
- **`MAX_TOOL_ITERATIONS=20`** caps the client-side tool loop to
  prevent runaway invocations. Distinct from
  `max_turns=` on `execute()` — the latter is a caller-facing
  budget; the former is a fail-safe for misbehaving tool loops.
- **Envelope unwrap is single-key only.** Two-level wrappers
  (`{"data": {"input": {...}}}`) require the consumer to unwrap
  manually or use `unwrap(AsyncOpenAI)` and call the API directly.

## Native escape hatches

```python
from openai import AsyncOpenAI
client: AsyncOpenAI = runtime.unwrap(AsyncOpenAI)
resp = await client.chat.completions.create(...)
```

`OpenAICompatibleSession` has no vendor session object — it just
holds the `messages=[]` buffer. `session.unwrap()` works only for
the session class itself.

## Building your own compat-vendor adapter

Subclass `OpenAICompatibleRuntime` — see
[`src/airframe/adapters/opencode_zen.py`](../../src/airframe/adapters/opencode_zen.py)
for the canonical 30-line example. Override:

- `PROVIDER_ID`, `EXTRA_NAME`, `DEFAULT_BASE_URL`, `DEFAULT_MODEL`
- `METADATA: dict[str, ModelMeta]`
- `_resolve_api_key(api_key)` — vendor-specific auth chain

Everything else (execute, list_models, error classification,
envelope unwrap) is inherited.

## See also

- [auth.md](../auth.md#opencodezenruntime)
- [capabilities.md](../capabilities.md)
- [third-party.md](./third-party.md) — writing your own adapter
  (compat-vendor subclass or full bespoke).
