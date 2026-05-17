# OpenRouterRuntime

Routes work through the [OpenRouter](https://openrouter.ai/) gateway
at `https://openrouter.ai/api/v1` — OpenRouter's OpenAI Chat
Completions-compatible router fronts 200+ models from Anthropic,
OpenAI, Google, Meta, Mistral, DeepSeek, and others behind one HTTP
endpoint and one billing relationship.

| | |
|---|---|
| **PROVIDER_ID** | `openrouter` |
| **Pip extra** | `airframe-agents[openai-compat]` |
| **Vendor SDK** | `openai` (HTTP only — no subprocess) |
| **Transport** | Direct HTTP via `AsyncOpenAI` |
| **Authentication** | See [auth.md](../auth.md#openrouterruntime) |
| **Billing** | Per-token, against OpenRouter credit balance |

## Install

```bash
pip install airframe-agents[openai-compat]
```

## Quickstart

```python
from airframe import OpenRouterRuntime, ProviderModel
from pydantic import BaseModel

class Brief(BaseModel):
    summary: str

runtime = OpenRouterRuntime()  # picks up OPENROUTER_API_KEY from env
result = await runtime.execute(
    "Brief me on the project structure.",
    schema=Brief,
    model=ProviderModel("openrouter", "anthropic/claude-3.5-haiku"),
)
print(result.structured)
print(result.cost.cost_usd)  # computed when the model is in METADATA
await runtime.close()
```

## Model identifiers carry vendor prefixes

OpenRouter routes by `<vendor>/<model>` strings. Examples:

```
openai/gpt-4o-mini           openai/gpt-4o            openai/o1
anthropic/claude-3.5-haiku   anthropic/claude-3.5-sonnet
google/gemini-pro-1.5        google/gemini-flash-1.5
meta-llama/llama-3.1-70b-instruct
deepseek/deepseek-chat
mistralai/mistral-large
```

The adapter passes the string through unchanged — pass whatever
identifier OpenRouter publishes at
[https://openrouter.ai/models](https://openrouter.ai/models). The
`METADATA` table in `openrouter.py` enriches a curated subset with
pricing and context windows; unknown IDs return `cost_usd=None`
from `list_models()` (and computed cost) rather than a guess.

## Supported features

Identical to the sibling OpenAI-compatible adapters — all inherit
from `OpenAICompatibleRuntime`:

| Feature | Status | Notes |
|---|---|---|
| `STRUCTURED_OUTPUT_JSON_SCHEMA` | ✓ | Native via `response_format={"type":"json_schema",...}` |
| `STREAMING` / `CANCEL` | ✓ | |
| `TOOLS_FUNCTION` | ✓ | Client-side tool loop |
| `SESSION_RESUME` | ✗ | Chat Completions has no server-side session |
| `TOOLS_MCP_*` / `PERMISSION_CALLBACK` | ✗ | Declined on Chat Completions transport |
| `LIFECYCLE_HOOKS` | ✓ | 6 of 8 kinds (no `pre_compact`, no `rate_limit`) |
| `BUDGET_USD_CAP` / `BUDGET_TURN_CAP` | ✓ | Caps apply normally; cost computed per-model |

## Per-model feature heterogeneity

OpenRouter is a **router**, not a single vendor. A given model's
support for `tools=`, strict-mode JSON schema, vision inputs, and
reasoning controls depends on the underlying vendor it routes to.

- ✓ The adapter exposes the OpenAI Chat Completions wire format
  uniformly — every feature flag in the table above is "supported"
  in the sense that the call won't be rejected at airframe's
  boundary.
- ⚠️ Whether the underlying model actually honours the directive
  is per-model. Anthropic/OpenAI/Google flagship models handle
  structured output cleanly; open-weights and budget routes vary.
- ❌ Some routes silently degrade to plain text when asked for JSON
  schema — the failure mode is `RuntimeStructuredOutputError` on
  Pydantic validation, not a hard refusal from OpenRouter.

For any model your application hard-depends on, run a probe against
your schema before promoting it past development.

## OpenAICompatOptions

Same provider-options namespace as the OpenCode adapters — see
[opencode-zen.md](./opencode-zen.md#openaicompatoptions-provider-options-namespace).
OpenRouter forwards most fields to the upstream vendor unchanged;
fields the upstream doesn't recognise are silently ignored at the
gateway.

## Cost reporting

OpenRouter publishes per-token pricing at
[https://openrouter.ai/models](https://openrouter.ai/models). The
adapter ships a curated `METADATA` map with the most-used routes;
each `ModelMeta` carries input/output per-1k rates. `cost_usd` on
the `CostRecord` is computed from these rates when the model ID is
known; unknown IDs report `cost_usd=None` (tokens still populated).

To enrich a new model, add a `ModelMeta` entry in
`src/airframe/adapters/openrouter.py`. Rates on OpenRouter's site
are listed per million tokens — divide by 1000 to get the per-1k
values airframe stores.

## Vendor quirks & landmines

- **No on-disk auth-file convention.** OpenRouter doesn't have an
  equivalent of `~/.local/share/opencode/auth.json`. Auth resolves
  from explicit `api_key=` or `OPENROUTER_API_KEY` env only.
- **Pricing drifts.** Upstream vendors adjust rates and OpenRouter
  reflects them. The METADATA map is a point-in-time snapshot;
  treat as a hint, not authoritative. For production budget gating,
  pull live rates from OpenRouter's `/models` endpoint or override
  `METADATA` at construction time.
- **Vendor prefixes are part of the model ID.** Don't strip them
  before passing to `ProviderModel(...)`. `"claude-3.5-sonnet"`
  alone won't route; you need `"anthropic/claude-3.5-sonnet"`.
- **Route availability is dynamic.** Models can be added,
  deprecated, or rate-limited by their upstream vendors without
  OpenRouter changing. A 404 / 502 from a previously-working route
  is expected behaviour, not an airframe regression.

## Native escape hatches

```python
from openai import AsyncOpenAI
client: AsyncOpenAI = runtime.unwrap(AsyncOpenAI)
```

## See also

- [opencode-zen.md](./opencode-zen.md) / [opencode-go.md](./opencode-go.md)
  — sibling adapters built on the same base.
- [auth.md](../auth.md#openrouterruntime) — full auth chain.
- [capabilities.md](../capabilities.md) — per-feature semantics
  inherited from `OpenAICompatibleRuntime`.
