# Capabilities

`Feature` is airframe's typed capability enum. Every
:class:`AgentRuntime.supports(feature, model=None)` call is a cheap
static lookup — no network, no SDK version probe. Consumer code
branches on `supports()` before invoking the matching API, so
declined capabilities never reach a `RuntimeAuthError` at the call
site; they raise `UnsupportedFeatureError` with a `feature=`
attribute at the gate.

The string values of every `Feature` member are public surface.
Renaming would be a major-version break.

## Capability matrix

| Feature | Bedrock | Claude | Copilot | Kimi | OpenAI-compat | OpenCode |
|---|---|---|---|---|---|---|
| `STRUCTURED_OUTPUT_JSON_SCHEMA` | ✓ | ✓ | ✓ | ◐ scaffolded | ✓ | ✗ (SDK gap) |
| `STRUCTURED_OUTPUT_STRICT` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `STREAMING` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `CANCEL` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SESSION_RESUME` | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ |
| `REASONING_EFFORT` | ✓ (Anthropic-on-Bedrock) | ✓ | ✓ | ✓ (boolean) | ✓ | ✓ (per-upstream) |
| `REASONING_BUDGET_TOKENS` | ✓ (Anthropic-on-Bedrock) | ✓ | ✗ | ✗ | ✗ | ✓ (Anthropic upstream) |
| `VISION_INPUT` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `FILE_INPUT` | ✓ (Anthropic-on-Bedrock) | ✓ | ✓ | ✗ | ✗ | ✓ |
| `TOOLS_FUNCTION` | ✓ | ✓ | ✓ | ✗ (permanent) | ✓ | ✗ (SDK gap) |
| `TOOLS_MCP_STDIO` | ✗ (permanent) | ✓ | ✓ | ✓ | ✗ | ✗ (SDK gap) |
| `TOOLS_MCP_HTTP` | ✗ (permanent) | ✓ | ✓ | ✓ | ✗ | ✗ (SDK gap) |
| `TOOLS_MCP_SSE` | ✗ (permanent) | ✓ | ✗ | ✓ | ✗ | ✗ (SDK gap) |
| `TOOLS_MCP_IN_PROCESS` | ✗ | (internal) | (internal) | ✗ (permanent) | ✗ | ✗ (permanent) |
| `PERMISSION_CALLBACK` | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ (SDK gap) |
| `LIFECYCLE_HOOKS` | ✓ (6 kinds) | ✓ (8 kinds) | ✓ (7 kinds) | ✓ (7 kinds) | ✓ (6 kinds) | ✓ (6 kinds) |
| `BUDGET_USD_CAP` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (best-effort) |
| `BUDGET_TURN_CAP` | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| `RATE_LIMIT_TELEMETRY` | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ |
| `REASONING_OUTPUT` | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ |
| `REQUEST_METADATA` | ✗ (soft drop) | ✓ | ✗ (soft drop) | ✗ (soft drop) | ✓ | ✗ (soft drop) |
| `COUNT_TOKENS` | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ |
| `PROMPT_CACHE_CONTROL` | ✗ (soft drop) | ✗ (soft drop) | ✗ (soft drop) | ✗ (soft drop) | ✓ | ✗ (soft drop) |
| `SLASH_COMMANDS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SANDBOX` / `SUBAGENTS` | ✗ (planned) | ✗ (planned) | ✗ (planned) | ✗ (planned) | ✗ (planned) | ✗ (planned) |

The OpenCode "SDK gap" entries are not adapter declines — OpenCode the *server* supports those features. The opencode-ai 0.1.0a36 Python SDK simply hasn't surfaced the matching endpoints yet (no `client.mcp` / `client.permission` resources). The flags will flip True once the SDK catches up. See `docs/adapters/opencode-server.md` for the full story.

**Soft-drop entries** (`REQUEST_METADATA`, `PROMPT_CACHE_CONTROL`) follow a deliberately different contract from the rest: an adapter declaring `False` *accepts* the kwarg structurally and silently drops it rather than raising `UnsupportedFeatureError`. The call's correctness doesn't depend on the tag / cache key reaching the vendor — at worst the consumer loses abuse-detection attribution or cache speed-up. Consumers who care branch on `supports()` first; consumers who just want best-effort pass the kwarg and forget. See each feature's section below.

Run `uv run python examples/probe_supports.py` for the live matrix
against your installed adapters.

---

## Per-feature semantics

### `STRUCTURED_OUTPUT_JSON_SCHEMA`

Wire shape: `runtime.execute(prompt, schema=MyPydanticModel)` /
`session.execute(prompt, schema=...)`. Always returns
`RuntimeResult.structured` as a dict matching the schema.

Per-adapter mechanism:
- **Claude:** native `output_format={"type":"json_schema","schema":...}`
- **Copilot:** forced `submit_result` tool call (hidden from
  streaming events)
- **Kimi:** scaffolded (declared True) but `execute(schema=…)` raises
  `NotImplementedError` until the MCP forced-tool path lands
- **OpenAI-compat:** native `response_format={"type":"json_schema",...}`
  (strict=False for compat-vendor portability)

### `STRUCTURED_OUTPUT_STRICT`

Reserved for "JSON schema enforced at the wire level with
no field omission tolerance" — OpenAI's `strict: True` mode is the
canonical example. False everywhere today; compat-vendor coverage
is uneven. Likely flips on a future per-vendor opt-in.

### `STREAMING`

Wire shape: `session.stream(prompt) → AsyncIterator[RuntimeEvent]`.
Five event variants — `TextDelta`, `ReasoningDelta`,
`ToolCallStart`, `ToolCallResult`, `TurnComplete`. Exactly one
`TurnComplete` per successful stream; it carries the same
`RuntimeResult` `execute()` would have returned.

Concatenating all `TextDelta.text` for one turn equals the final
`result.text`. The variant set is shape-locked.

### `CANCEL`

Wire shape: `await session.cancel()`. No-op when no turn is in
flight; mid-turn it aborts and the awaiting call raises
`RuntimeCancelledError`.

Per-adapter mechanism:
- **Claude:** `ClaudeSDKClient.interrupt()` + task cancellation
- **Copilot:** `CopilotSession.abort()`
- **Kimi:** `Session.cancel()` sets the SDK's async cancel event
- **OpenAI-compat:** `asyncio.Task.cancel()` → httpx; for `stream()`,
  a flag + `AsyncStream.close()`

### `SESSION_RESUME`

Wire shape: `runtime.session(resume=<id>)`. `<id>` is the value
surfaced on a prior session's `AgentSession.id` field.

OpenAI-compat declines permanently — Chat Completions has no
server-side session concept. The other three honour vendor-native
resume APIs.

### `REASONING_EFFORT`

Wire shape: `session.execute(prompt, thinking="low" | "medium" | "high")`.
Maps to each vendor's reasoning-effort knob. `"minimal"` is also
accepted; vendors that don't support it (Claude, Copilot) coerce
to `"low"` with a debug log. Pass `thinking="disabled"` or
`thinking=None` to skip.

### `REASONING_BUDGET_TOKENS`

Wire shape: `session.execute(prompt, thinking={"budget_tokens": N})`.
Claude-only — explicit reasoning-token budget rather than an
effort enum. Other adapters raise `UnsupportedFeatureError` on the
dict shape.

### `VISION_INPUT`

Wire shape: `session.execute([text, ImageInput(path="..."), ...])`.
The prompt accepts a list of `PromptPart` (str / `ImageInput` /
`FileInput`); the adapter routes each part to the vendor's matching
input channel.

Per-adapter mechanism:
- **Claude:** Read tool (auto-allowed for attachments)
- **Copilot:** `attachments=[FileAttachment{path}]`
- **Kimi:** kosong `ImageURLPart` (URL pass-through; bytes / path
  → `data:` URI)
- **OpenAI-compat:** content-parts (`{"type":"image_url","image_url":{"url":"data:..base64..."}}`)

`ImageInput` accepts `path=`, `bytes_= + media_type=`, or `url=`.
Each adapter declines paths it can't honour (e.g. Copilot has no
URL channel — `url=` raises).

### `FILE_INPUT`

Same `PromptPart` channel as VISION_INPUT but for non-image files
(`FileInput(path=...)`). OpenAI-compat declines — Chat Completions
has no file slot; vendors using OpenAI's `client.files.create`
API would need their own subclass that overrides this.

### `TOOLS_FUNCTION`

Wire shape: `runtime.session(tools=[FunctionTool(name, description, params=PydanticModel, handler=async_fn)])`.
The model invokes registered tools; airframe drives the round-trip
and surfaces results.

Kimi declines permanently — its Python SDK has no
Python-callable tool channel. Wrap the function as an MCP server
and pass via `mcp_servers=` instead.

### `TOOLS_MCP_STDIO` / `_HTTP` / `_SSE`

Wire shape: `runtime.session(mcp_servers=[McpServerRef(name, transport, command=..., url=..., auth_token=..., headers=...)])`.

Per-transport coverage:
- **Claude:** all three transports natively
- **Copilot:** stdio + http; SSE declined (use http instead)
- **Kimi:** all three transports via fastmcp's `MCPConfig` dict shape
- **OpenAI-compat:** all three declined (Chat Completions has no
  MCP-as-tool slot)

The decline raises `UnsupportedFeatureError` with `feature=`
carrying the *specific* transport's flag, so consumer code
branching on `supports(Feature.TOOLS_MCP_STDIO)` etc. works
correctly.

### `TOOLS_MCP_IN_PROCESS`

Internal-only — describes the in-process MCP server that
`tools=[FunctionTool]` compiles to on Claude / Copilot. Never
exposed as a user-facing transport on `McpServerRef`. Always
False at the runtime level.

### `PERMISSION_CALLBACK`

Wire shape: `runtime.session(on_permission=PermissionCallback)`.
The callback receives a `PermissionRequest(tool_name, tool_args, reason)`
and returns `"allow"` / `"deny"` / `"defer"`.

Per-adapter shape:
- **Claude:** per-call via `can_use_tool`
- **Copilot:** per-call via `on_permission_request`
- **Kimi:** per-call dispatch from the SDK's `ApprovalRequest` wire
  messages (`yolo=False` mode). `allow → approve`, `deny → reject`,
  `defer → reject` with feedback (the SDK's approval channel is
  synchronous so defer collapses)
- **OpenAI-compat:** **permanently declined** — Chat Completions
  has no tool-permission wire shape; the *caller* decides whether
  to execute a returned `tool_call`

`"defer"` semantics: falls through to the vendor's default policy
with a debug log. Claude's binary `PermissionResultAllow`/`Deny`
maps `"defer"` to `PermissionResultAllow` because the default
`permission_mode="bypassPermissions"` already matches the "defer"
intent.

### `LIFECYCLE_HOOKS`

Wire shape: `runtime.session(on_event=Callable[[HookEvent], None])`.
Synchronous observer — don't block; await work belongs in
`on_permission`.

The `HookEvent.kind` enum has eight literals, shape-locked:
`"session_start"`, `"session_end"`, `"user_prompt_submit"`,
`"pre_tool_use"`, `"post_tool_use"`, `"tool_failure"`,
`"pre_compact"`, `"rate_limit"`.

Per-adapter emittable subset (`EMITTABLE_HOOK_KINDS` ClassVar):
- **Claude:** all 8 (native `PreCompact` and `RateLimit` events)
- **Copilot:** 7 (no `rate_limit` — surfaces as `SessionErrorData`)
- **Kimi:** 7 (no `rate_limit` — Moonshot raises 429s as
  `APIStatusError` exceptions, not wire events)
- **OpenAI-compat:** 6 (no `pre_compact` — no compaction concept;
  no `rate_limit` — no discrete throttle event)

`session_start` and `session_end` are universal. `session_end` is
synthesised at `close()` if the vendor never fired it, gated on
idempotency.

### `BUDGET_USD_CAP`

Wire shape: `session.execute(prompt, max_budget_usd=0.05)`.
Cumulative across all turns of the session; checked at the start
of every turn. Trips `RuntimeBudgetExceededError(kind="usd",
cap=0.05, current=0.07)` *before* the about-to-fire turn — so
the cap acts as a fail-closed gate, not a refund mechanism.

Every adapter accumulates client-side against
`RuntimeResult.cost.cost_usd`. Claude's vendor-computed
`total_cost_usd` is authoritative; the others compute from the
pricing tables (see each adapter page for the rate source).

Vendors that report `cost_usd=None` (free tiers, unknown model
IDs) can't trip the cap — the running total stays at 0. The
integration suite skips `test_integration_budget_usd_cap_trips`
in that case.

### `BUDGET_TURN_CAP`

Wire shape: `session.execute(prompt, max_turns=10)`. Counts
user-visible turns (one per `session.execute()` /
`session.stream()`); checked at the start of every turn.

Copilot declines permanently — its vendor SDK caps internal turns
at the CLI level via the runtime's `--max-turns` config. Exposing
a user-facing `max_turns=` on per-execute would be misleading.

On Claude, the kwarg additionally rides into
`ClaudeAgentOptions.max_turns` to override the runtime-default
`DEFAULT_MAX_TURNS=60` for SDK-internal turn limiting.

### `RATE_LIMIT_TELEMETRY`

Wire shape: `result.rate_limit: RateLimitInfo | None` on
`RuntimeResult` (when the vendor surfaces quota data on a
successful call) and `RuntimeTransientError.rate_limit` (on a
throttle response).

`RateLimitInfo` wraps a tuple of `RateLimitWindow` snapshots — each
carries `name` (vendor's window identifier — `"requests"` /
`"tokens"` / `"five_hour"` / etc.), `remaining`, `limit`,
`utilization`, `reset_at`, `retry_after_seconds`, `status`. Fields
vary per vendor: OpenAI populates `remaining` / `limit`; Claude
populates `utilization` / `status`. Consumers should treat any
single field as a hint and check `is not None` before using it.

Per-adapter mechanism:
- **Claude:** consumes `RateLimitEvent` instances from the SDK
  message stream, accumulating per-window state across emitted
  events (`five_hour`, `seven_day`, `seven_day_opus`,
  `seven_day_sonnet`, `overage`).
- **OpenAI-compat:** parses the six `x-ratelimit-*` headers
  (limit/remaining/reset for requests + tokens) + `retry-after` via
  `chat.completions.with_raw_response.create()`. Includes OpenAI's
  `"6m0s"` / `"42ms"` duration-string format.

Adapters returning `False` leave `result.rate_limit=None`
unconditionally. Adapters returning `True` may still leave it
`None` when the vendor didn't send quota data on that turn —
consumers should branch on `rate_limit is not None` regardless of
`supports()`.

### `REASONING_OUTPUT`

Wire shape: `result.reasoning: str | None` on `RuntimeResult` —
the model's finalised reasoning / extended-thinking trace as plain
text. Streaming pairs naturally: concatenating every
`ReasoningDelta.text` for one turn equals `turn_complete.result.reasoning`.

Distinct from `REASONING_EFFORT` (the *input* side: "I can ask the
model to think harder"); this is the *output* side ("I can show you
what it thought").

Per-adapter mechanism:
- **Claude:** consumes `ThinkingBlock` content from
  `AssistantMessage` (non-streaming) + accumulates streamed
  `ReasoningDelta` from `thinking_delta` wire events.
- **OpenAI-compat:** defensively reads `message.reasoning_content`
  (DeepSeek-R1 derivatives) or `message.reasoning` on the response;
  per-chunk `delta.reasoning_content` / `delta.reasoning` on the
  stream. Vendors that don't surface reasoning text (OpenAI Chat
  Completions on the `o1`/`gpt-5` family — which exposes only the
  token count) leave the field `None`.

### `REQUEST_METADATA`

Wire shape: `runtime.session(metadata=RequestMetadata(user_id=...,
request_id=..., tags={...}))` — forwards a per-request observation
tag to the vendor for abuse detection / per-tenant usage
attribution / audit trails.

**Soft contract.** Adapters returning `False` *accept* the kwarg
and silently drop it. The call's correctness doesn't depend on the
tag reaching the vendor; consumers who care branch on `supports()`
first.

Per-adapter mapping:
- **Claude:** `user_id` → `ClaudeAgentOptions.user`. `tags` /
  `request_id` silently dropped (no agent-SDK channel).
- **OpenAI-compat:** `user_id` → `user=` kwarg; `tags` →
  `metadata=` kwarg (typed `Dict[str, str]` on the OpenAI SDK);
  `request_id` → `extra_headers={"X-Request-ID": ...}`. Pre-existing
  values on the create kwargs are preserved + extended rather than
  overwritten.
- **Copilot / Kimi / Bedrock / OpenCode-server:** silently dropped
  (no native metadata channel today).

### `COUNT_TOKENS`

Wire shape: `await runtime.count_tokens(prompt, *, system=None, model=None) -> int`.
Pre-flight token count — answer "is this prompt going to blow the
context window / break my budget" *before* paying for a turn.

Adapters returning `False` raise `UnsupportedFeatureError` when
called. v1 supports plain-text and string-only multi-part prompts;
list-shaped prompts with image / file attachments raise
`UnsupportedFeatureError` (base64 expansion + per-vendor counting
heuristics deferred).

Per-adapter mechanism:
- **Claude:** delegates to
  `anthropic.AsyncAnthropic.messages.count_tokens(...)` — same
  auth-resolution chain as `list_models()`. Requires `[claude]`
  extra. Network call.
- **OpenAI-compat:** `tiktoken.encoding_for_model(model_id)` with
  a fall-back to `o200k_base` (GPT-4o tokeniser) when the model
  isn't in tiktoken's registry. Best-effort approximation for
  compat-vendor models using non-OpenAI tokenisers (DeepSeek,
  Llama, etc.) — typically within 5–10% but not exact. Requires
  `[openai-compat]` extra (pulls `tiktoken`).
- **Copilot / Kimi / Bedrock / OpenCode-server:** declined —
  neither the vendor SDK exposes a counter endpoint nor is the
  per-family tokeniser bundled. Wrap with the consumer's own
  estimator if needed.

### `PROMPT_CACHE_CONTROL`

Wire shape: `runtime.session(cache=CacheConfig(key=..., retention="short"|"long"))` —
the consumer hands the vendor a stable cache key so repeated
prompts hit the cached prefix rather than recompute.

**Soft contract.** Adapters returning `False` accept the kwarg and
silently drop it. The call still succeeds — just without the
speed-up / cost-reduction explicit caching would have provided.

Per-adapter mapping:
- **OpenAI-compat:** `key` → `prompt_cache_key=`;
  `retention="short"` → `prompt_cache_retention="in_memory"`
  (5-minute window); `retention="long"` →
  `prompt_cache_retention="24h"`. The portable `cache=` value
  takes precedence over the OpenAI-specific
  `OpenAICompatOptions.prompt_cache_key` (consumers who set both
  get the cross-vendor surface through).
- **Claude / Copilot / Kimi / Bedrock / OpenCode-server:**
  silently dropped — these adapters either manage caching via
  session warmth (Claude) or expose no explicit cache-key channel.

The `retention` literal is deliberately coarse (`"short"` /
`"long"`) because vendor windows differ. Consumers wanting precise
control should reach the vendor's native field via
`provider_options=`.

### `SLASH_COMMANDS`

Wire shape: `runtime.session(slash_commands=SlashCommandsConfig(...))`
+ `await session.list_slash_commands() -> list[SlashCommand]`.
Discovery of user-authored slash commands from the filesystem —
the consumer's UI uses the returned list to render a palette.

Discovery walks `.claude/commands/*.md`, `.opencode/command/*.md`,
`.agents/commands/*.md` upward from `cwd` to the git worktree root,
plus the matching user-global directories
(`~/.claude/commands/`, etc.) when
`SlashCommandsConfig.include_user_global` is `True`. Plus any extra
`SlashCommandsConfig.search_paths`. The YAML frontmatter at the
top of each file is parsed via a minimal hand-rolled parser (no
`pyyaml` dep); the rest is the body template.

Each `SlashCommand` exposes `name`, `description`, `body`,
`source_path`, and the raw `frontmatter` dict. Consumers expand
the body template themselves (substituting `$ARGUMENTS` / `$1` /
`{file}` per the vendor convention) before passing the expanded
text to `session.execute()`.

**Every adapter declares `Feature.SLASH_COMMANDS = True`** —
discovery is filesystem-only and adapter-agnostic. **Invocation
semantics differ:**
- **Claude Agent SDK:** auto-expands `/commandname args` when
  passed verbatim through `execute()` — the SDK reads the same
  `.claude/commands/` directory airframe discovers. Consumers can
  use either path (call `execute("/refactor foo.py")` directly, or
  enumerate via `list_slash_commands()` for a palette and pass the
  expanded body).
- **OpenAI-compat / Bedrock / Copilot / Kimi / OpenCode-server:**
  no native slash-command channel; the consumer expands
  `SlashCommand.body` and calls `execute(expanded_text)`. The
  model receives the substituted body as a normal user prompt.

The model itself does not do template-syntax expansion — it sees
whatever text reaches it after the consumer (or the Claude SDK)
substitutes placeholders.

### `SANDBOX` / `SUBAGENTS`

Planned, signal-gated features. Will flip True when the
corresponding API ships. `runtime.supports(...)` returns False
today on every adapter; the kwargs aren't exposed yet, so there's
nothing to gate.

---

## Capability + ProviderOptions interaction

`provider_options=` (the per-vendor namespace, see each adapter
page) is independent of `Feature`. A field can be set even when no
corresponding `Feature` exists — it's a vendor-specific knob
without a portable equivalent.

Cross-namespace mistakes raise `UnsupportedFeatureError` at the
adapter boundary: passing `CopilotOptions` to `ClaudeCodeRuntime`
fails fast with a clear message rather than silently ignoring.

## See also

- [reference.md](./reference.md) — API reference for `Feature`
  members and the kwargs that gate on them.
- [adapters/*.md](./adapters/) — per-adapter feature subset and
  vendor quirks.
- `airframe.testing.contracts` — structural conformance tests that
  pin capability-vs-API agreement for every adapter.
