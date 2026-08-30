# `BedrockRuntime` adapter plan

Companion to [implementation-plan.md](./implementation-plan.md).
Specs a new built-in adapter wrapping AWS Bedrock's [Converse
API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html)
— the vendor-normalised model-invocation endpoint that fronts
Anthropic / Meta / Mistral / Cohere / Amazon Nova models behind one
AWS-flavoured wire format.

The plan deliberately mirrors the ABCDEF iteration cadence used by
[`google-genai-adapter-plan.md`](./google-genai-adapter-plan.md):
scaffold → execute/stream/cancel → polymorphic prompt + reasoning
→ tools → hooks + budget → wrap-up. Targeting ~600–800 LOC and 6
iterations, each independently mergeable.

## Motivation

Airframe has no enterprise / managed-cloud path today. Every shipped
adapter targets either a personal-subscription product (Claude Max,
Copilot, ChatGPT Plus, opencode-go) or a developer-API gateway
(OpenRouter, OpenCode Zen). Bedrock is the missing fourth bucket —
**AWS-billed access to a multi-vendor model catalog** with IAM-rooted
auth, region pinning, and provisioned throughput options. That
audience (enterprise, government, regulated industries) can't use
any of the existing adapters because the others depend on creds /
endpoints that aren't reachable from their VPCs.

A `BedrockRuntime` over the Converse API specifically:

- **Vendor-normalised wire format.** Converse abstracts the
  per-vendor request/response shapes (Anthropic's
  `messages`+`anthropic-version`, Meta's prompt-templated form,
  Mistral's instruction format, Cohere's `chat_history`) into one
  `messages` / `system` / `toolConfig` envelope. Airframe's
  vendor-neutral surface maps cleanly onto that — same way
  `OpenAICompatibleRuntime` maps onto Chat Completions.
- **One adapter unlocks many models.** Anthropic Claude, Meta
  Llama, Mistral Large, Cohere Command, Amazon Nova, AI21 Jamba.
  Roughly the same multi-vendor leverage OpenRouter gives
  consumer-credit users, but with AWS auth and SLAs.
- **Matches airframe's existing "narrow protocol over a vendor
  SDK" shape.** Compare `OpenAICompatibleRuntime` at ~30 LOC over
  HTTP, `ClaudeCodeRuntime` at ~800 LOC over a subprocess SDK.
  Bedrock lands between — closer to 600 LOC because the Converse
  envelope is richer than chat-completions but the SDK is just
  `aioboto3`.

## Non-goals

- **No Bedrock Agents wrapping.** `bedrock-agent-runtime` (the
  service that owns Knowledge Bases, action groups, agent
  orchestration) is a different endpoint with a different request
  shape. It deserves a sibling adapter, not conflation. Track
  separately if/when there's demand.
- **No legacy `InvokeModel` path.** The pre-Converse API requires
  per-vendor JSON shapes — equivalent to what `BedrockRuntime` is
  meant to abstract away. Consumers stuck on a model that doesn't
  support Converse should use `boto3` directly and unwrap.
- **No provisioned throughput management.** PT setup is an
  AWS-console / IaC concern. The adapter accepts a PT ARN as a
  model identifier (Bedrock honours those in Converse) but doesn't
  provision or report on it.
- **No cross-account assumed-role bootstrap.** Use the standard
  AWS practice — `aws sts assume-role` ahead of time, export the
  resulting session creds, then construct `BedrockRuntime()`. The
  adapter doesn't reinvent role chaining.
- **No KMS-encrypted-prompt routing.** Bedrock supports CMK
  encryption on the wire; the adapter passes whatever boto3
  resolves through. Per-call encryption-key selection (rare) is
  reachable via `unwrap(BedrockRuntimeClient)`.

## Adapter shape

`BedrockRuntime(AgentRuntime)` — a direct subclass of `AgentRuntime`,
not `OpenAICompatibleRuntime`. Reasons:

1. Wire format isn't OpenAI Chat Completions — it's Converse's
   `messages` + `system` + `toolConfig` envelope, served by
   `aioboto3.Session().client("bedrock-runtime")`.
2. Auth chain is boto3-native (env → ~/.aws/credentials → IAM
   instance profile → role assumption), not API-key style.
3. Region is a first-class constructor arg, not a base-URL override.

`BedrockSession(AgentSession)` owns a per-conversation `messages=[]`
buffer (Converse is stateless from the client's perspective, same
as Chat Completions — each call resends the full history).
`id` stays `None`.

```
src/airframe/adapters/bedrock.py        ~600 LOC (target)
tests/test_bedrock.py                   ~400 LOC mirroring test_opencode_go.py
tests/test_bedrock_integration.py       behavioural, pytest-marker gated
docs/adapters/bedrock.md                ~150 lines, sibling to opencode-go.md
examples/probe_bedrock.py               single-call execute(schema=) probe
```

## SDK surface this adapter wraps

`aioboto3` ≥ 13.x is the async wrapper around `botocore`; we use it
for non-blocking calls inside airframe's async surface. The Bedrock
Runtime service exposes (relevant subset):

- `client.converse(modelId, messages, system, inferenceConfig,
  toolConfig, additionalModelRequestFields)` — single-turn
  request/response. Returns `{output, usage, stopReason,
  performanceConfig}`.
- `client.converse_stream(...)` — same shape, returns an async
  iterator of typed event chunks (`messageStart`, `contentBlockDelta`,
  `messageStop`, `metadata`).
- `client.list_foundation_models(byProvider=, byOutputModality=)` —
  vendor-curated catalog (we filter to text-output, exclude
  embeddings).

Per-vendor knobs live in `additionalModelRequestFields` — e.g.,
Anthropic extended thinking is
`{"thinking": {"type": "enabled", "budget_tokens": N}}`. The
adapter exposes this through `thinking=` and the `BedrockOptions`
namespace (see below).

## Feature support matrix (target)

| Feature | Status | Notes |
|---|---|---|
| `STRUCTURED_OUTPUT_JSON_SCHEMA` | ✓ | Converse `toolConfig` with a forced `submit_result` tool — same pattern as `CopilotRuntime`. The Converse Tool API takes a JSON Schema as the tool's `inputSchema`. |
| `STRUCTURED_OUTPUT_STRICT` | ✗ | Bedrock has no "strict" tool-schema mode equivalent to OpenAI's. |
| `STREAMING` | ✓ | `converse_stream` typed event chunks → `TextDelta` / `ReasoningDelta`. |
| `CANCEL` | ✓ | `asyncio.Task.cancel()` propagates to `aiohttp` underneath; `stream()` cancels by closing the stream iterator. |
| `SESSION_RESUME` | ✗ | Converse is stateless from the client side. The session buffer doesn't survive process restart. |
| `REASONING_EFFORT` | ✓ | Maps to Anthropic-on-Bedrock's `thinking_budget` via `additionalModelRequestFields`. **Model-gated** — non-Anthropic models on Bedrock decline silently per vendor. |
| `REASONING_BUDGET_TOKENS` | ✓ | Same channel — `{"budget_tokens": N}` honoured on Anthropic variants. |
| `VISION_INPUT` | ✓ | Converse content blocks: `{"image": {"format": "png\|jpeg\|gif\|webp", "source": {"bytes": ...}}}`. Anthropic / Nova / Llama 3.2 vision all supported. |
| `FILE_INPUT` | ✓ | Converse supports `{"document": {...}}` content blocks (Anthropic only today; others ignore). |
| `TOOLS_FUNCTION` | ✓ | Native Converse `toolConfig` — `tools: [{"toolSpec": {name, description, inputSchema}}]`. Client-side tool loop owned by airframe (matches OpenAI-compat). |
| `TOOLS_MCP_STDIO` / `_HTTP` / `_SSE` | ✗ | Bedrock has no MCP-as-tool slot. Permanent decline. |
| `PERMISSION_CALLBACK` | ✓ | Client-side tool loop means `PermissionCallback` fires naturally around each `_invoke_tool` (same pattern as OpenAI-compat + Gemini). |
| `LIFECYCLE_HOOKS` | ✓ | 6 of 8 kinds — `session_start`, `session_end`, `user_prompt_submit`, `pre_tool_use`, `post_tool_use`, `tool_failure`. No `pre_compact` (Converse has no compaction concept); no `rate_limit` (boto3 retries are silent). |
| `BUDGET_USD_CAP` | ✓ | Computed from Converse's `usage.inputTokens` / `outputTokens` against an in-tree `_BEDROCK_PRICING` table keyed by model ID. |
| `BUDGET_TURN_CAP` | ✓ | Client-side counter, same as siblings. |

## Auth chain

Order resolved at `execute()` time:

1. **Explicit `aws_access_key_id` / `aws_secret_access_key` /
   `aws_session_token` constructor args.** Highest precedence.
   Forwards to `aioboto3.Session(...)`.
2. **`AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` (+ optional
   `AWS_SESSION_TOKEN`) env vars.** The standard AWS vars.
3. **`AWS_PROFILE` env var** → `~/.aws/credentials` / `~/.aws/config`
   profile resolution. Honoured by boto3 natively.
4. **Default credential chain** — IAM instance profile (EC2),
   ECS task role, Lambda execution role, IRSA (EKS). boto3
   handles all of these; we just don't override.

**Region resolution** (separate from credentials):

1. Explicit `region_name=` constructor arg.
2. `AWS_REGION` / `AWS_DEFAULT_REGION` env var.
3. `~/.aws/config` `region` for the resolved profile.

If no region resolves, the first call raises a
`RuntimeAuthError`-style classification with a clear "set
AWS_REGION" message — Bedrock is region-pinned and silent fallback
to a default region would route traffic to a different model
catalog than the user expects.

## BedrockOptions (provider-options namespace)

```python
@dataclass(frozen=True)
class BedrockOptions:
    # Per-call AWS knobs.
    region_name: str | None = None              # override constructor region per session
    inference_profile_arn: str | None = None    # for provisioned throughput / cross-region
    guardrail_id: str | None = None             # Bedrock Guardrails policy id
    guardrail_version: str | None = None
    # Performance / latency hints (Converse-native).
    performance_latency: str | None = None      # "standard" | "optimized"
    # Vendor-specific pass-through.
    additional_model_fields: dict[str, Any] | None = None
        # forwarded into Converse's additionalModelRequestFields
        # — for things airframe doesn't have first-class support for
        # (Anthropic top_k, Meta top_p, Cohere search-result formatting).
```

Same tagged-union discipline as the existing five namespaces —
mismatched type raises `UnsupportedFeatureError` at
`session(provider_options=)`.

## Iteration breakdown

### Iteration A — Protocol scaffolding (no behaviour)

~150 LOC. Lands the adapter's shape without wiring substantive
features.

- `src/airframe/adapters/bedrock.py` — `BedrockRuntime(AgentRuntime)`
  with ClassVars, lazy `aioboto3` import (deferred to first method
  call), `validate_binding` (accepts any string — Bedrock's catalog
  is too dynamic to gate by prefix; `validate_binding` returns True
  for any non-empty model_id when `provider_id == "bedrock"`).
- `_resolve_aws_credentials()` chain implementing the four-step
  auth above. Region resolution is independent.
- Empty `SUPPORTED_FEATURES = frozenset()` initially; flip flags
  on as each iteration wires them.
- `unwrap(BedrockRuntimeClient)` runtime escape hatch (the aioboto3
  client type).
- `close()` idempotent + never raises (closes the aioboto3 session
  context).
- `reset()` no-op (sessionless runtime — the session buffer lives
  on `BedrockSession`).
- `list_models()` hits `list_foundation_models(byOutputModality="TEXT")`,
  enriches against `_BEDROCK_METADATA` for known IDs.
- Discovery registration in `discovery.py` + top-level export.
- Conformance contracts pass against a mocked aioboto3 client.

**Stopping point.** `import airframe; airframe.list_providers()`
includes `"bedrock"` when the extra is installed. Discovery +
capability predicates work; no live behaviour wired.

### Iteration B — Execute + streaming + cancellation

~150 LOC. The Phase-1-equivalent slice — `session` factory,
`AgentSession` subclass, `execute()`, `stream()`, `cancel()`.

- `BedrockSession(AgentSession)` — owns the `messages=[]` buffer.
  No per-session aioboto3 client (the runtime-level client serves
  every session; cheaper than spinning one up per turn).
- `execute(prompt, schema=)` — single turn via `client.converse(...)`.
  Structured output: bake a forced `submit_result` tool into
  `toolConfig` with the schema as its `inputSchema`; parse the
  tool-use response block. Mirrors `CopilotRuntime` exactly.
- `stream(prompt, schema=)` — translate `converse_stream` chunks:
  `contentBlockDelta.delta.text` → `TextDelta`,
  `contentBlockDelta.delta.reasoningContent.text` → `ReasoningDelta`,
  `messageStop` → captures `stopReason` for `TurnComplete`,
  `metadata.usage` → cost record.
- `cancel()` — `asyncio.Task.cancel()` propagates; for `stream()`,
  closes the async iterator (boto3 drops the underlying connection).
- `close()` — idempotent; clears the `messages` buffer; runtime
  client stays alive for sibling sessions.
- Exception classification: `botocore.exceptions.ClientError` →
  `RuntimeAuthError` (4xx auth codes), `RuntimeModelNotFoundError`
  (`ValidationException` with "model" in message),
  `RuntimeTransientError` (5xx, throttling), `RuntimeProtocolError`
  (otherwise unclassifiable). `aiohttp.ClientError` →
  `RuntimeTransientError`.
- Flip `Feature.STREAMING`, `Feature.CANCEL`,
  `Feature.STRUCTURED_OUTPUT_JSON_SCHEMA` True.
- `examples/probe_bedrock.py` — minimal `execute(schema=)` probe.

**Stopping point.** Single-turn structured output works against
live Bedrock with valid AWS creds + region. Streaming yields
deltas. Cancellation works.

### Iteration C — Polymorphic prompt + reasoning

~100 LOC. The Phase-2-equivalent slice.

- `_split_prompt_parts` integration: `ImageInput` → Converse
  `{"image": {"format": "...", "source": {"bytes": <data>}}}`
  content block. Auto-detect format from bytes header / file
  extension. `FileInput` → `{"document": {"format": "pdf|md|...",
  "name": "...", "source": {"bytes": ...}}}`.
- `thinking=` kwarg →
  `additionalModelRequestFields={"thinking": {"type": "enabled",
   "budget_tokens": N}}` for Anthropic models. Non-Anthropic models
  log at debug and pass through (Bedrock ignores unknown fields per
  vendor).
- Mapping: `"low"` → 1024, `"medium"` → 8192, `"high"` → 32768,
  `{"budget_tokens": N}` → N exact, `"disabled"` → omits the
  field. Matches `ClaudeCodeRuntime`.
- Flip `Feature.REASONING_EFFORT`, `Feature.REASONING_BUDGET_TOKENS`,
  `Feature.VISION_INPUT`, `Feature.FILE_INPUT` True.
- Probe extensions: `probe_thinking.py`, `probe_vision.py` get
  `bedrock` branches.

### Iteration D — Function tools + permission

~150 LOC. Phase-3 + Phase-5 (permission slice) equivalent.

- Translate `FunctionTool` → Converse `{"toolSpec": {"name",
  "description", "inputSchema": {"json": <schema dict>}}}`.
- Tool loop in `_do_execute` / `stream()`: parse `output.message`
  for `toolUse` content blocks, invoke each handler, append the
  result as a `{"toolResult": {"toolUseId", "content": [{"json"|"text":
  ...}], "status": "success|error"}}` content block, re-call
  `converse(...)`. Capped at `MAX_TOOL_ITERATIONS` (reuse the
  OpenAI-compat constant).
- `ToolCallStart` / `ToolCallResult` events on `stream()`.
- `PermissionCallback` fires around each `_invoke_tool` — same
  pattern as OpenAI-compat and the planned Gemini adapter.
- Flip `Feature.TOOLS_FUNCTION`, `Feature.PERMISSION_CALLBACK` True.

**MCP non-goal.** Bedrock Converse has no MCP slot. Flag
`TOOLS_MCP_STDIO/HTTP/SSE` all False permanently; decline message
points users at `unwrap(BedrockRuntimeClient)` if they want to
hand-craft an MCP-shim themselves.

### Iteration E — Hooks + budget

~100 LOC. Phase-5 equivalent.

- `EMITTABLE_HOOK_KINDS` ClassVar — 6 kinds (matches OpenAI-compat
  / Codex / Gemini): `session_start`, `session_end`,
  `user_prompt_submit`, `pre_tool_use`, `post_tool_use`,
  `tool_failure`.
- `_on_event` plumbing on `BedrockSession` synthesised from the
  client-side tool loop, identical pattern to
  `OpenAICompatibleSession`.
- Budget-cap enforcement via the shared
  `_enforce_budget_pre_turn()` helper.
- In-tree `_BEDROCK_PRICING` table for the curated catalog:
  - Anthropic Claude variants (haiku/sonnet/opus per generation)
  - Amazon Nova (Pro/Lite/Micro)
  - Meta Llama 3.x (8B/70B/405B Instruct)
  - Mistral Large
  - Cohere Command R+
  - Rates pulled from
    [AWS Bedrock pricing](https://aws.amazon.com/bedrock/pricing/)
    at the time of writing; documented as point-in-time.
- Flip `LIFECYCLE_HOOKS`, `BUDGET_USD_CAP`, `BUDGET_TURN_CAP` True.

### Iteration F — Wrap-up

~50 LOC + docs.

- `BedrockOptions` dataclass (per the surface above) wired through
  with `_check_provider_options`.
- Conformance contract suite green; integration test wrapper at
  `tests/test_bedrock_integration.py`.
- Per-adapter docs page: `docs/adapters/bedrock.md` covering install
  extra, auth chain (cross-link to `docs/auth.md`), supported
  features, `BedrockOptions` reference, model IDs (with the
  inference-profile-prefix gotcha called out), structured output
  mechanism, vendor quirks, native escape hatches.
- `docs/auth.md` gets a new section covering the boto3 chain +
  region resolution.
- `docs/capabilities.md` matrix gains a column.
- README "Supported providers" table updated.
- CHANGELOG entry.
- Update Issue #8 to mark Bedrock as shipped; move from Tier 1.

## Risks and decisions to flag during execution

1. **Inference-profile vs base-model IDs.** Bedrock recently
   introduced inference profiles for cross-region routing — model
   IDs like `us.anthropic.claude-3-5-sonnet-20241022-v2:0` (with a
   region prefix) or pure ARNs. Some models *require* the
   inference-profile form. `validate_binding` should accept both
   shapes; `_BEDROCK_METADATA` should key on the un-prefixed form
   and fall through to "unknown" for the prefixed variants
   (cost-reporting becomes None, model still works).
2. **Region defaults are dangerous.** Bedrock catalogs differ
   per region. If `AWS_REGION` isn't set and boto3 falls through to
   `us-east-1`, the user may hit `ValidationException` ("model not
   available in this region") on a model that works fine in
   `us-west-2`. Resolution: surface "region not set" as a
   first-class auth-class error rather than letting it fall through
   to a confusing model-not-found.
3. **`aioboto3` vs `botocore.session`.** `aioboto3` adds a
   dependency layer. Alternative: use `botocore` directly with
   `httpx` + manual sigv4 signing (no extra dep). Decision: stick
   with `aioboto3` — its API matches boto3 1:1, sigv4 is correct
   by construction, and the dep is well-maintained.
4. **`additionalModelRequestFields` is a footgun.** Vendor-specific
   knobs (Anthropic top_k, Meta top_p, Cohere search-result format)
   pass through unchecked. Wrong field for the wrong vendor =
   silent ignore. `BedrockOptions.additional_model_fields` is the
   honest escape hatch; document that field-validity is the
   caller's problem.
5. **Bedrock Guardrails interact opaquely with structured output.**
   Guardrails policies can block tool-use responses if they trigger
   content filters, surfacing as truncated or empty `output`. When
   `guardrail_id` is set, the adapter should detect
   `stopReason == "guardrail_intervened"` and raise
   `RuntimeProtocolError` with a clear message rather than failing
   Pydantic validation on an empty payload.
6. **`PERMISSION_CALLBACK=True` is a real claim.** Unlike Codex
   (session-wide) or OpenAI Chat Completions (declined), Bedrock's
   client-side tool loop genuinely supports per-call permission
   gating. Make sure the contract suite exercises this against the
   live API.
7. **Streaming on Anthropic-via-Bedrock includes encrypted
   reasoning blocks.** Recent Claude versions on Bedrock emit
   `reasoningContent` blocks with a `redactedContent` field for
   safety-redacted thinking. The adapter must skip those without
   crashing (they have no `text` field) and not surface them as
   `ReasoningDelta` events.

## Definition of done

- `runtime_for("bedrock")` returns `BedrockRuntime` when the
  extra is installed; clean `ImportError` with `airframe-agents[bedrock]`
  hint when not.
- `examples/probe_bedrock.py` round-trips a structured-output prompt
  against live Bedrock in `us-east-1` and `us-west-2` (one Anthropic
  + one Amazon Nova model).
- `examples/probe_parity.py` includes `bedrock` with no per-vendor
  conditionals; passes on a machine with AWS creds + region.
- Conformance contract suite green; integration suite green
  against live AWS.
- `docs/adapters/bedrock.md` complete; README provider table +
  capability matrix updated.
- CHANGELOG entry; Issue #8 updated.

## When to start

After OpenRouter ships (done in PR following this plan's commit).
Bedrock is the largest single-adapter scope airframe has yet
attempted (~600 LOC + boto dep + AWS test-account requirement).
Reasonable cadence: one iteration per week, ~6 weeks end-to-end.
Iteration A can land alongside other work since it's behaviour-free;
B onward should batch together for review since they touch the same
files.

## Open questions for the implementer

1. **`BedrockAgentRuntime` as a sibling?** The Bedrock Agents
   service (`bedrock-agent-runtime` — Knowledge Bases, action
   groups, server-side orchestration) is a separate API. Worth
   shipping as a sibling adapter once this lands, particularly for
   consumers using Bedrock for retrieval-grounded workflows. Track
   separately.
2. **Pricing table maintenance.** AWS publishes pricing per region
   per model. `_BEDROCK_PRICING` is point-in-time; should the
   adapter fetch live rates from a pricing API on first call? Cost:
   one extra HTTP per process start. Benefit: no stale pricing.
   Defer until staleness becomes a reported issue.
3. **Vertical integration with airframe's planned `AnthropicRuntime`.**
   The direct Anthropic API and Bedrock-hosted Anthropic are
   functionally close but auth-distinct. When `AnthropicRuntime`
   lands, document the choice matrix clearly (IAM vs API key,
   regional availability, billing path).

## Implementation wiring checklist

Beyond `src/airframe/adapters/bedrock.py` itself, every new
adapter needs to touch these files to be fully wired. Easy to
forget; easy to verify by grepping for the closest sibling
(`opencode_go.py` for OpenAI-compat shape, `claude_code.py` for
full-bespoke shape).

### Source wiring

- [ ] `src/airframe/discovery.py` — add `BedrockRuntime` to
      `_builtin_runtime_classes()` (currently 6 entries).
- [ ] `src/airframe/__init__.py` — `from airframe.adapters.bedrock
      import BedrockRuntime` at module level + `"BedrockRuntime"`
      entry in `__all__` (alphabetical).
- [ ] `src/airframe/testing/contracts.py` — add `"bedrock":
      BedrockOptions` to the `matching` dict inside
      `_check_provider_options` (the test that asserts every
      adapter rejects foreign provider-options namespaces).
- [ ] `src/airframe/testing/integration.py` — add `"bedrock":
      ["AWS_ACCESS_KEY_ID", "AWS_PROFILE"]` to `_PROVIDER_AUTH`.
      The integration suite uses this map to self-skip when no
      credentials are present.

### Probe + examples wiring

- [ ] `examples/probe_budget.py` — add `"bedrock"` to the for-loop
      provider tuple inside the capability matrix print (the line
      that iterates over `("claude", "github-copilot", "codex",
      "opencode-zen", "opencode-go", "openrouter")`).
- [ ] `examples/probe_parity.py` — picks up the new adapter
      automatically via `list_providers()`. No source change needed.
      Consider adding an `AIRFRAME_PROBE_MODEL_BEDROCK` env-var
      hook in `_model_override` if the default model is
      region-sensitive (look at the codex `_codex_subscription_model`
      auto-detection pattern for precedent).

### Packaging

- [ ] `pyproject.toml` — new `bedrock = ["aioboto3>=13"]` extra
      under `[project.optional-dependencies]`, AND add
      `"aioboto3>=13"` to the `all = [...]` list, AND add
      `"aioboto3>=13"` to the `[dependency-groups].test` list (the
      unit suite imports `aioboto3` for mocking even though calls
      are stubbed).

### Documentation

- [ ] `README.md` — provider table row (between the OpenRouter
      and any future row); update the `pip install
      airframe-agents[openai-compat]` adjacent example to mention
      `[bedrock]`; add `bedrock` to the comma-separated provider
      ID example in the quickstart.
- [ ] `docs/auth.md` — quick-reference table row + a full
      `## BedrockRuntime` section covering the boto3 chain and
      region resolution. Cross-link from the adapter page.
- [ ] `docs/reference.md` — adapter table row + add
      `BedrockRuntime` to the `__all__` snippet near the end.
- [ ] `docs/adapters/bedrock.md` — new page (mirror
      `docs/adapters/opencode-go.md` structure: identity table,
      quickstart, model catalog, supported features, options,
      cost reporting, vendor quirks, escape hatches, see-also).
- [ ] `docs/capabilities.md` — the existing matrix uses
      `OpenAI-compat` as a single column header for the OpenAI-
      compatible family. Bedrock needs its own column.
- [ ] `CLAUDE.md` — add `"bedrock"` to the canonical provider IDs
      list in the "Provider IDs are strict" paragraph.

### Test wiring

- [ ] `tests/test_bedrock.py` — unit tests. Mirror
      `tests/test_claude_code.py` for structure (it's the closest
      full-bespoke template, ~600 LOC) rather than
      `tests/test_opencode_go.py` (which is a ~150-LOC test for a
      ~30-LOC subclass of the OpenAI-compat base). Bedrock owns
      its own session class so the test surface is larger.
- [ ] `tests/test_bedrock_integration.py` — pytest-marker-gated
      behavioural tests against live AWS. Mirror
      `tests/test_opencode_zen_integration.py` for shape.
- [ ] `tests/test_discovery.py` — update the
      `test_list_providers_returns_all_when_installed_only_false`
      expected set + the filtered `test_list_providers_filters_...`
      tests + the third-party-discovery test that lists builtins.

### Issue + project housekeeping

- [ ] Update [Issue #8](https://github.com/get2knowio/airframe/issues/8)
      — move Bedrock from Tier 1 to the "Recently shipped" section
      with commit/PR references.
- [ ] CHANGELOG entry with the iteration summary.

## Closest in-tree templates to read first

Open these side-by-side with the plan before writing any code.

| File | What to learn from it |
|---|---|
| `src/airframe/adapters/claude_code.py` | The full-bespoke shape — `BedrockRuntime` mirrors this structurally (subclasses `AgentRuntime` directly, owns its own `AgentSession` subclass). Roughly 800 LOC; the Bedrock target ~600 is lower because Converse abstracts what Claude Code does at the SDK level. |
| `src/airframe/adapters/copilot.py` | The forced-tool-for-structured-output pattern. Bedrock's `toolConfig` + `submit_result` tool works identically. Pay attention to the schema-fingerprint caching pattern — Bedrock may want similar to avoid rebuilding `toolConfig` when the schema doesn't change. |
| `src/airframe/adapters/codex.py` (lines ~140-170) | The `_strictify_schema` helper. Anthropic-on-Bedrock will almost certainly require `additionalProperties: false` on every object node in the tool `inputSchema` — same as Codex's Responses-backed endpoints. Either move the helper to a shared module (`src/airframe/_schema.py`?) or copy into `bedrock.py`. Recommend the move during Iteration B. |
| `src/airframe/adapters/opencode_go.py` | The discovery / `__init__.py` / docs cross-reference pattern (smaller surface, easier to scan). |
| `src/airframe/sessions.py` | The shared helpers: `_enforce_budget_pre_turn`, `_check_provider_options`. Bedrock's `BedrockSession` calls into both. |

## Naming reservations

Established during the OpenCode rename:

- `"bedrock"` — this adapter (Converse API, model invocation).
- `"bedrock-agents"` — **reserved** for the future
  `BedrockAgentsRuntime` wrapping `bedrock-agent-runtime`
  (Knowledge Bases, action groups, server-side orchestration).
  Do **not** fold Agents into `"bedrock"`. Same vendor with
  distinct billing / capability surface = distinct provider ID,
  per the lesson from
  `OpenCodeZenRuntime` (`"opencode-zen"`) vs
  `OpenCodeGoRuntime` (`"opencode-go"`).
- `"aws-bedrock"`, `"amazon-bedrock"` — **not** used. The
  `"bedrock"` namespace is unambiguous and matches AWS's own
  service-name shorthand.

## First commit in a fresh session

A reasonable Iteration A first commit (the scaffolding-only slice):

```
src/airframe/adapters/bedrock.py        # new — Iteration A surface
src/airframe/discovery.py               # +BedrockRuntime in builtins
src/airframe/__init__.py                # +export +__all__
pyproject.toml                          # +[bedrock] extra
tests/test_bedrock.py                   # new — identity, validate_binding, auth-chain unit tests
tests/test_discovery.py                 # +bedrock in expected sets
```

That should pass `mise run check` cleanly with `Feature` flags all False
(no behaviour wired yet). After review, Iteration B adds
`execute()` + `stream()` + `cancel()` and flips the first three
`Feature` flags True.
