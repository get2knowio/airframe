# Capabilities

`Feature` is airframe's typed capability enum. Every
:class:`AgentRuntime.supports(feature, model=None)` call is a cheap
static lookup — no network, no SDK version probe. Consumer code
branches on `supports()` before invoking the matching API, so
declined capabilities never reach a `RuntimeAuthError` at the call
site; they raise `UnsupportedFeatureError` with a `feature=`
attribute at the gate.

The string values of every `Feature` member are public surface —
locked at v0.3.0. Renaming would be a major-version break.

## Capability matrix

| Feature | Claude | Copilot | Codex | OpenAI-compat |
|---|---|---|---|---|
| `STRUCTURED_OUTPUT_JSON_SCHEMA` | ✓ | ✓ | ✓ | ✓ |
| `STRUCTURED_OUTPUT_STRICT` | ✗ | ✗ | ✗ | ✗ |
| `STREAMING` | ✓ | ✓ | ✓ | ✓ |
| `CANCEL` | ✓ | ✓ | ✓ | ✓ |
| `SESSION_RESUME` | ✓ | ✓ | ✓ | ✗ |
| `REASONING_EFFORT` | ✓ | ✓ | ✓ | ✓ |
| `REASONING_BUDGET_TOKENS` | ✓ | ✗ | ✗ | ✗ |
| `VISION_INPUT` | ✓ | ✓ | ✓ | ✓ |
| `FILE_INPUT` | ✓ | ✓ | ✓ | ✗ |
| `TOOLS_FUNCTION` | ✓ | ✓ | ✗ | ✓ |
| `TOOLS_MCP_STDIO` | ✓ | ✓ | ✗ | ✗ |
| `TOOLS_MCP_HTTP` | ✓ | ✓ | ✗ | ✗ |
| `TOOLS_MCP_SSE` | ✓ | ✗ | ✗ | ✗ |
| `TOOLS_MCP_IN_PROCESS` | (internal) | (internal) | ✗ | ✗ |
| `PERMISSION_CALLBACK` | ✓ | ✓ | ✓ (session-wide) | ✗ |
| `LIFECYCLE_HOOKS` | ✓ (8 kinds) | ✓ (7 kinds) | ✓ (6 kinds) | ✓ (6 kinds) |
| `BUDGET_USD_CAP` | ✓ | ✓ | ✓ | ✓ |
| `BUDGET_TURN_CAP` | ✓ | ✗ | ✓ | ✓ |
| `SANDBOX` / `SUBAGENTS` | ✗ (planned) | ✗ (planned) | ✗ (planned) | ✗ (planned) |

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
- **Codex:** native `TurnOptions.outputSchema`
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
- **Codex:** `AbortController` plumbed into `TurnOptions.signal`
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
- **Codex:** `LocalImageInput(path)`
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

Codex declines permanently — its Python SDK has no
tool-registration channel. Configure tools through
`~/.codex/config.toml`'s `[[mcp_servers]]` block instead and
register them as MCP servers on the CLI side.

### `TOOLS_MCP_STDIO` / `_HTTP` / `_SSE`

Wire shape: `runtime.session(mcp_servers=[McpServerRef(name, transport, command=..., url=..., auth_token=..., headers=...)])`.

Per-transport coverage:
- **Claude:** all three transports natively
- **Copilot:** stdio + http; SSE declined (use http instead)
- **Codex:** all three declined (no Python SDK channel)
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
- **Codex:** **session-wide** — the callback fires *once* at first
  `execute()` with a sentinel request to derive the
  `approval_policy` enum (`"never"` / `"untrusted"` /
  `"on-request"` / `"on-failure"`); per-call interception isn't
  possible through the SDK
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
- **Codex:** 6 (no `pre_compact`, no `rate_limit` — SDK has neither)
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
