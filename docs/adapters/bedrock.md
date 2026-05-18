# BedrockRuntime

Wraps AWS Bedrock's [Converse
API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html)
— the vendor-normalised model-invocation endpoint that fronts
Anthropic Claude, Meta Llama, Mistral, Cohere, Amazon Nova, and
AI21 Jamba behind one AWS-billed envelope with IAM-rooted auth and
region pinning.

| | |
|---|---|
| **PROVIDER_ID** | `bedrock` |
| **Pip extra** | `airframe-agents[bedrock]` |
| **Vendor SDK** | `aioboto3` (≥13) — async wrapper around `botocore` |
| **Transport** | Async HTTPS via `aioboto3.Session().client("bedrock-runtime")` |
| **Authentication** | See [auth.md](../auth.md#bedrockruntime) — boto3's four-step chain |
| **Billing** | AWS-billed per token; in-tree pricing table covers the curated catalog |

## Install

```bash
pip install airframe-agents[bedrock]
```

Brings `aioboto3>=13` (which brings `aiobotocore` + `boto3` + `botocore`
as transitive deps).

## Quickstart

```python
from airframe import BedrockRuntime, ProviderModel
from pydantic import BaseModel

class Brief(BaseModel):
    summary: str

runtime = BedrockRuntime(region_name="us-east-1")  # or via AWS_REGION env
result = await runtime.execute(
    "Brief me on the project structure.",
    schema=Brief,
    model=ProviderModel("bedrock", "anthropic.claude-3-5-haiku-20241022-v1:0"),
)
print(result.structured)         # {"summary": "..."}
print(result.cost.cost_usd)      # ~0.001 — computed from in-tree pricing
print(result.cost.input_tokens)  # populated from Converse's usage
await runtime.close()
```

`region_name=` is **required** (constructor arg or `AWS_REGION` env).
Bedrock catalogs differ per region; silent fallback to a default region
would route traffic to a different model surface than the caller
expects, so the adapter raises `RuntimeAuthError` rather than
fall through.

## Model catalog

`list_models()` hits
`bedrock.list_foundation_models(byOutputModality="TEXT")` and returns
the catalog the resolved AWS identity can see in the resolved region.
The adapter enriches a curated subset with display names + context
windows + per-1k-token pricing (`docs/adapters/bedrock.md` lives
alongside the source for easy review):

| Model ID | Family | Context |
|---|---|---|
| `anthropic.claude-3-5-haiku-20241022-v1:0` | Anthropic (default) | 200K |
| `anthropic.claude-3-5-sonnet-20241022-v2:0` | Anthropic | 200K |
| `anthropic.claude-3-opus-20240229-v1:0` | Anthropic | 200K |
| `amazon.nova-micro-v1:0` | Amazon Nova | 128K |
| `amazon.nova-lite-v1:0` | Amazon Nova | 300K |
| `amazon.nova-pro-v1:0` | Amazon Nova | 300K |
| `meta.llama3-1-8b-instruct-v1:0` | Meta Llama | 128K |
| `meta.llama3-1-70b-instruct-v1:0` | Meta Llama | 128K |
| `meta.llama3-1-405b-instruct-v1:0` | Meta Llama | 128K |
| `mistral.mistral-large-2407-v1:0` | Mistral | 128K |
| `cohere.command-r-plus-v1:0` | Cohere | 128K |

Models the catalog returns but the table doesn't recognise still
surface — they just come back without context-window / pricing
enrichment.

### Inference-profile and PT-ARN gotcha

Bedrock supports cross-region [inference
profiles](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html)
(IDs like `us.anthropic.claude-3-5-sonnet-20241022-v2:0` — the leading
`us.` / `eu.` is the profile prefix) and provisioned-throughput ARNs
(`arn:aws:bedrock:us-east-1:…:provisioned-model/…`). Some models
*require* the inference-profile form rather than the bare ID.

* `validate_binding` accepts both shapes — anything non-empty under
  `provider_id="bedrock"` is valid.
* `_BEDROCK_PRICING` keys on the un-prefixed form, so inference-profile
  IDs report `cost_usd=None` — by design rather than as a bug. If you
  rely on `BUDGET_USD_CAP` for an inference-profile-routed session,
  populate `additional_model_fields` is not the right knob — fork the
  pricing table or set the cap on a wrapping session that owns its
  own cost telemetry.

## Supported features

| Feature | Status | Notes |
|---|---|---|
| `STRUCTURED_OUTPUT_JSON_SCHEMA` | ✓ | Forced `submit_result` tool in Converse `toolConfig` |
| `STRUCTURED_OUTPUT_STRICT` | ✗ | Converse has no "strict" tool-schema mode |
| `STREAMING` | ✓ | `converse_stream` typed events → `TextDelta` / `ReasoningDelta` |
| `CANCEL` | ✓ | `asyncio.Task.cancel()` for execute; stream-iterator close for stream |
| `SESSION_RESUME` | ✗ | Converse is stateless from the client; messages buffer doesn't survive restart |
| `REASONING_EFFORT` / `REASONING_BUDGET_TOKENS` | ✓ | Anthropic-on-Bedrock via `additionalModelRequestFields={"thinking":...}`; silently dropped on non-Anthropic models per Bedrock's per-vendor field handling |
| `VISION_INPUT` | ✓ | Converse `{"image": {"format": ..., "source": {"bytes": ...}}}` content blocks; `url=` raises (needs bytes locally) |
| `FILE_INPUT` | ✓ | Converse `{"document": ...}` content blocks; Anthropic-only today on Bedrock |
| `TOOLS_FUNCTION` | ✓ | Native Converse `toolConfig` + client-side tool loop (cap: 20 iterations) |
| `TOOLS_MCP_*` | ✗ | Bedrock Converse has no MCP slot — **permanent decline** |
| `PERMISSION_CALLBACK` | ✓ | Fires around each tool-handler invocation in the client-side loop |
| `LIFECYCLE_HOOKS` | ✓ | 6 of 8 kinds emittable (no `pre_compact`, no `rate_limit`) |
| `BUDGET_USD_CAP` | ✓ | Computed from the in-tree pricing table; turns whose model is in the table count toward the cap |
| `BUDGET_TURN_CAP` | ✓ | Per-session counter, enforced at turn boundary |

## BedrockOptions

```python
from airframe import BedrockOptions

opts = BedrockOptions(
    region_name=None,                  # per-session region override
    inference_profile_arn=None,        # use this ARN as modelId for every call
    guardrail_id=None,                 # Bedrock Guardrails policy id
    guardrail_version=None,            # optional companion version pin
    performance_latency=None,          # "standard" | "optimized"
    additional_model_fields=None,      # pass-through to additionalModelRequestFields
)
runtime.session(provider_options=opts)
```

* **`region_name`** — opens a session-private `aioboto3` client pinned
  to a different region than the runtime's default. Useful for routing
  one session through `us-west-2` while the runtime defaults to
  `us-east-1`. Per-session clients are torn down on `session.close()`.
* **`inference_profile_arn`** — replaces the session's `modelId` with
  the full ARN on every Converse call. The `model=` you'd otherwise
  pass on `session()` is overridden.
* **`guardrail_id`** (+ `guardrail_version`) — runs the session under
  the named [Bedrock
  Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html)
  policy. When a guardrail blocks the response,
  `stopReason="guardrail_intervened"` surfaces as
  `RuntimeProtocolError` rather than failing structured-output
  validation on a truncated payload — see *Vendor quirks* below.
* **`performance_latency`** — `"optimized"` opts into Bedrock's
  latency-optimised inference path. Available on a subset of models;
  unsupported models silently ignore the field per Bedrock's
  per-vendor `performanceConfig` handling.
* **`additional_model_fields`** — merged into
  `additionalModelRequestFields` alongside airframe's `thinking`
  field. **User keys win on collision** — the pass-through is the
  honest escape hatch; field-validity is the caller's problem.

## Cost reporting

`cost.input_tokens` / `cost.output_tokens` / `cost.cache_read_tokens` /
`cost.cache_write_tokens` come through directly from Converse's
`usage`. `cost.cost_usd` is computed from `_BEDROCK_PRICING` (an
in-tree, point-in-time `us-east-1` rate table) for known models;
`None` for unknown ids. Region-specific pricing and PT rates are out
of scope for the table.

## Vendor quirks & landmines

- **`AWS_REGION` is required.** No silent fallback to `us-east-1`.
  Pass `region_name=` to the constructor or set the env var; missing
  region surfaces as `RuntimeAuthError` at first call.
- **Bedrock catalogs differ per region.** A model that works in
  `us-east-1` may be unavailable in `us-west-2` and vice-versa. The
  adapter surfaces `ValidationException` with "model" in the message
  as `RuntimeModelNotFoundError` so the failure mode is clear.
- **Bedrock Guardrails interact opaquely with structured output.** A
  blocked response truncates the payload and sets
  `stopReason="guardrail_intervened"`. The adapter detects this and
  raises `RuntimeProtocolError` with a clear message rather than
  letting Pydantic validation fail on an empty `submit_result` payload.
- **Encrypted reasoning blocks on Anthropic-on-Bedrock.** Recent
  Claude versions emit `reasoningContent` chunks with a
  `redactedContent` field for safety-redacted thinking. The adapter
  skips these without crashing and does not surface them as
  `ReasoningDelta` events (no usable text).
- **`additionalModelRequestFields` is a footgun.** Vendor-specific
  knobs (Anthropic `top_k`, Meta `top_p`, Cohere
  `search_result_format`) pass through unchecked. Wrong field for
  the wrong vendor = silent ignore (Bedrock per-vendor handling
  drops unknown keys).
- **MCP is a permanent decline.** Bedrock Converse has no MCP slot.
  `session(mcp_servers=...)` raises `UnsupportedFeatureError`
  pointing at `runtime.unwrap(BedrockRuntimeClient)` for callers who
  want to hand-craft an MCP shim.
- **Throttling is silent.** `botocore`'s transient-retry chain
  handles throttle events before the adapter sees them, which is why
  `rate_limit` is not in the emittable hook-kinds set.

## Native escape hatches

The aioboto3 bedrock-runtime client (the live, async-context-managed
object inside the runtime) is reachable via `unwrap` once
`execute()` / `stream()` has built it:

```python
client = runtime.unwrap(type(runtime._runtime_client))  # opaque botocore-generated class
# Or, by importing the typed shim if installed:
from types_aiobotocore_bedrock_runtime import BedrockRuntimeClient
client = runtime.unwrap(BedrockRuntimeClient)
```

The class is dynamically generated by botocore; `isinstance` is the
honest check.

## See also

- [auth.md](../auth.md#bedrockruntime) — full boto3 four-step auth
  chain and region resolution.
- [capabilities.md](../capabilities.md) — per-feature semantics
  across every adapter.
- [`examples/probe_bedrock.py`](../../examples/probe_bedrock.py) —
  end-to-end probe (structured output + plain text) against live
  AWS.
- [`dev-docs/bedrock-adapter-plan.md`](../../dev-docs/bedrock-adapter-plan.md)
  — design decisions and the per-iteration scope sheet.
