# OpenCodeGoRuntime

Routes work through the [opencode.ai
subscription gateway](https://opencode.ai/docs/zen) at
`https://opencode.ai/zen/go/v1`. The 14 bundled models come included
with the opencode-go monthly subscription — every token is $0 at the
caller's margin. For per-token access to a much wider model catalog
(GPT-5, Claude, Gemini, etc.), use
[`OpenCodeZenRuntime`](./opencode-zen.md) instead.

| | |
|---|---|
| **PROVIDER_ID** | `opencode-go` |
| **Pip extra** | `airframe-agents[openai-compat]` (shared with Zen) |
| **Vendor SDK** | `openai` (HTTP only — no subprocess) |
| **Transport** | Direct HTTP via `AsyncOpenAI` |
| **Authentication** | See [auth.md](../auth.md#opencodegoruntime) |
| **Billing** | Flat-fee monthly subscription — `cost_usd` reports `0.0` per turn |

## Install

```bash
pip install airframe-agents[openai-compat]
```

## Quickstart

```python
from airframe import OpenCodeGoRuntime, ProviderModel
from pydantic import BaseModel

class Brief(BaseModel):
    summary: str

runtime = OpenCodeGoRuntime()  # picks up OPENCODE_API_KEY from env
result = await runtime.execute(
    "Brief me on the project structure.",
    schema=Brief,
    model=ProviderModel("opencode-go", "glm-5.1"),
)
print(result.structured)
print(result.cost.input_tokens, result.cost.output_tokens)  # tokens still tracked
print(result.cost.cost_usd)  # 0.0 — flat-fee at the margin
await runtime.close()
```

## Subscription model catalog

The 14 models the gateway bundles with the subscription:

| Model | Context window | Family |
|---|---|---|
| `deepseek-v4-flash` | 1,000,000 | DeepSeek |
| `deepseek-v4-pro` | 1,000,000 | DeepSeek |
| `glm-5` | 202,752 | GLM |
| `glm-5.1` | 202,752 | GLM (default) |
| `kimi-k2.5` | 262,144 | Kimi |
| `kimi-k2.6` | 262,144 | Kimi (default — cleanest JSON-schema support) |
| `mimo-v2-omni` | 262,144 | MiMo |
| `mimo-v2-pro` | 1,048,576 | MiMo |
| `mimo-v2.5` | 1,000,000 | MiMo |
| `mimo-v2.5-pro` | 1,048,576 | MiMo |
| `minimax-m2.5` | 204,800 | MiniMax |
| `minimax-m2.7` | 204,800 | MiniMax |
| `qwen3.5-plus` | 262,144 | Qwen |
| `qwen3.6-plus` | 262,144 | Qwen |

No Claude / GPT / Gemini in the subscription catalog — those live on
the per-token Zen catalog.

## Supported features

Identical to [`OpenCodeZenRuntime`](./opencode-zen.md#supported-features)
— both adapters inherit the same `OpenAICompatibleRuntime` base.
Highlights:

| Feature | Status | Notes |
|---|---|---|
| `STRUCTURED_OUTPUT_JSON_SCHEMA` | ✓ | Native via `response_format={"type":"json_schema",...}` |
| `STREAMING` / `CANCEL` | ✓ | |
| `TOOLS_FUNCTION` | ✓ | Client-side tool loop |
| `SESSION_RESUME` | ✗ | Chat Completions has no server-side session |
| `TOOLS_MCP_*` / `PERMISSION_CALLBACK` | ✗ | Declined permanently on Chat Completions transport |
| `LIFECYCLE_HOOKS` | ✓ | 6 of 8 kinds (no `pre_compact`, no `rate_limit`) |
| `BUDGET_USD_CAP` | ✓ | Always `0.0` at the margin — caps still apply to multi-call flows that mix providers |
| `BUDGET_TURN_CAP` | ✓ | Client-side per-session counter |

## OpenAICompatOptions

Same provider-options namespace as `OpenCodeZenRuntime` — see
[opencode-zen.md](./opencode-zen.md#openaicompatoptions-provider-options-namespace).
The Go gateway silently ignores OpenAI-specific knobs that the
underlying non-OpenAI models don't recognise.

## Cost reporting

Token counts come through normally (`input_tokens`, `output_tokens`).
`cost_usd` is always `0.0` because the subscription is billed
monthly, not per call. The pricing entries in `METADATA` set
`(0.0, 0.0)` per-1k rates to reflect that — distinct from the Zen
per-token entries which use real rates.

## Vendor quirks & landmines

- **Same auth scheme as Zen, different URL.** Both adapters honour
  `OPENCODE_API_KEY` and `~/.local/share/opencode/auth.json`, but
  Go reads the `opencode-go` slot and posts to `/zen/go/v1`; Zen
  reads the `opencode` slot and posts to `/zen/v1`. Pick the
  runtime that matches your billing.
- **No subscription model overlap with Zen.** If you need GPT-5,
  Claude, or Gemini you must go through the per-token Zen catalog
  (or a vendor-direct adapter) — they're not part of the
  opencode-go bundle.
- **`cost_usd=0.0` is a real value, not "unknown".** Tools that
  use `BUDGET_USD_CAP` to gate spending will never trigger on Go
  traffic alone; cap-driven failover policies should branch on
  `provider_id == "opencode-go"` when zero-cost is meaningful.
- **Structured-output support varies wildly across the subscription
  catalog.** Empirically (smoke probe with a Pydantic schema):
    - `kimi-k2.6` — clean round-trip (the default for this reason).
    - `glm-5.1`, `minimax-m2.7` — accept `response_format` but
      silently return non-JSON.
    - `deepseek-v4-pro` — rejects with `"response_format type is
      unavailable now"`.
    - `qwen3.6-plus` — requires the word "json" in the prompt.
    - `mimo-v2.5-pro` — emits JSON but ignores the schema shape.

  If you need a non-Kimi model, run a probe against your schema
  first and treat compliance as model-specific. The adapter passes
  `strict: false` for portability, so silent non-compliance is the
  failure mode rather than a hard refusal.

## Native escape hatches

```python
from openai import AsyncOpenAI
client: AsyncOpenAI = runtime.unwrap(AsyncOpenAI)
```

## See also

- [opencode-zen.md](./opencode-zen.md) — sibling adapter for the
  per-token catalog.
- [auth.md](../auth.md#opencodegoruntime) — full auth chain.
- [capabilities.md](../capabilities.md) — per-feature semantics.
