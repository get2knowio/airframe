# OpenCodeServerRuntime

`OpenCodeServerRuntime` wraps the **OpenCode HTTP agent server** —
the bespoke REST + SSE server `opencode serve` runs locally (or
that you've stood up on a remote host). Provider ID: `"opencode"`.

This is distinct from the two existing `opencode*` adapters that
share the brand:

| Provider ID | Adapter | Wraps |
|---|---|---|
| `"opencode"` | `OpenCodeServerRuntime` | The local **agent server** (`opencode serve`) |
| `"opencode-zen"` | `OpenCodeZenRuntime` | The OpenCode Zen per-token gateway at `https://opencode.ai/zen/v1` |
| `"opencode-go"` | `OpenCodeGoRuntime` | The OpenCode Go subscription gateway at `https://opencode.ai/zen/go/v1` |

Different wire formats, different auth, different feature surfaces.
Don't mix them up.

## Why use it

The agent server is **model-agnostic**: it fronts whatever upstream
providers `opencode auth login` has configured — Anthropic, OpenAI
(including ChatGPT-OAuth subscriptions), OpenRouter, Ollama, vLLM,
llama.cpp, Together, Groq, MoonshotAI. Wrapping it gives airframe
one agent loop that delivers streaming, session resume, permission
gating (when the SDK exposes it), lifecycle hooks, and budget caps
across both open-weight and subscription model houses.

A concrete use case: **route Codex via your ChatGPT subscription**.
Run `opencode auth login openai`, complete the OAuth flow against
your ChatGPT account, then use `OpenCodeServerRuntime` to drive
chat turns. OpenCode handles the subscription auth; airframe gets a
clean `RuntimeResult`. No API-key billing.

## Install

```bash
pip install airframe-agents[opencode]
```

Brings in `opencode-ai>=0.1.0a36,<0.2` (the official Stainless-
generated Python SDK). Pre-1.0; pinned tightly to a single alpha
series — bump the upper bound deliberately, not via `pip install -U`.

## Server prerequisites

The adapter does **not** spawn `opencode serve` for you. Start it
yourself before calling:

```bash
opencode serve                           # default: 127.0.0.1:4096
opencode serve --port 9000               # custom port
opencode serve --hostname 0.0.0.0 ...    # remote (requires auth)
```

You also need at least one upstream configured for routing:

```bash
opencode auth login openai               # OpenAI / Codex subscription
opencode auth login anthropic            # Anthropic API key or Pro auth
opencode auth login openrouter
# ...
```

Run `opencode providers` to see what's wired.

## Quickstart

```python
from airframe import OpenCodeServerRuntime, ProviderModel

# Loopback default — no auth needed.
runtime = OpenCodeServerRuntime()

result = await runtime.execute(
    "Brief me on what Python's GIL is, in one paragraph.",
    model=ProviderModel("opencode", "gpt-5-codex"),
)
print(result.text)
print(result.cost.cost_usd)         # if the upstream reports cost
await runtime.close()
```

Streaming + session resume:

```python
sess = runtime.session(model=ProviderModel("opencode", "claude-haiku-4-5"))

async for event in sess.stream("Explain the GIL."):
    if hasattr(event, "text"):
        print(event.text, end="", flush=True)
print(f"\nsession id: {sess.id}")

# Resume later in a different process:
sess2 = runtime.session(
    resume=sess.id,
    model=ProviderModel("opencode", "claude-haiku-4-5"),
)
await sess2.execute("Now do the same for memory management.")
```

## Auth

See [`docs/auth.md` § OpenCodeServerRuntime](../auth.md#opencodeserverruntime)
for the full chain. TL;DR:

- Loopback (`127.0.0.1` / `localhost` / `::1`) — unauthenticated by
  default; matches `opencode serve`'s default posture.
- Remote URLs — `username=` / `password=` constructor args or
  `OPENCODE_SERVER_USERNAME` + `OPENCODE_SERVER_PASSWORD` env vars.
  Non-loopback URLs without credentials raise `RuntimeAuthError` to
  prevent an accidental remote-bash endpoint.

## Routing — picking which upstream to call

OpenCode's chat API requires both an **upstream provider id** and a
model id. Airframe resolves them in this order:

1. **Explicit routing**:
   ```python
   from airframe import OpenCodeServerOptions
   sess = runtime.session(
       model=ProviderModel("opencode", "gpt-5-codex"),
       provider_options=OpenCodeServerOptions(provider_id="openai"),
   )
   ```
2. **Auto-discovery** via `client.app.providers()` when the model id
   uniquely identifies one upstream. Cached per session.
3. **Ambiguous match** (same model on multiple upstreams) raises
   `UnsupportedFeatureError` asking for explicit routing.

## Supported features

Iteration B–E land most of the protocol; the remaining declines are
all gated on `opencode-ai` SDK 0.1.0a36 not surfacing the matching
endpoints yet. They'll flip True when the SDK catches up.

| Feature | Status | Notes |
|---|---|---|
| `STRUCTURED_OUTPUT_JSON_SCHEMA` | ✗ | **SDK gap** — no `client.mcp` resource for the forced-tool shim. |
| `STREAMING` | ✓ | `client.event.list()` SSE bus; deltas computed client-side from `message.part.updated` snapshots. |
| `CANCEL` | ✓ | `client.session.abort()` + chat-task cancel. |
| `SESSION_RESUME` | ✓ | Pass the prior session id via `runtime.session(resume=...)`. |
| `REASONING_EFFORT` | ✓ | Per-upstream envelope: OpenAI-shape (`reasoning_effort`) by default, Anthropic-shape (`thinking={"type": "enabled", "budget_tokens": ...}`) when upstream is Anthropic. |
| `REASONING_BUDGET_TOKENS` | ✓ | `thinking={"budget_tokens": N}` — Anthropic upstreams only. |
| `VISION_INPUT` | ✓ | `ImageInput(path/bytes_/url)` translated to OpenCode's `FilePartInputParam`. |
| `FILE_INPUT` | ✓ | `FileInput(path)` translated to `FilePartInputParam` with inferred MIME. |
| `TOOLS_FUNCTION` | ✗ | **SDK gap** — no MCP runtime registration. Pre-register MCP servers via `opencode.json` instead. |
| `TOOLS_MCP_*` | ✗ | **SDK gap** — same. |
| `PERMISSION_CALLBACK` | ✗ | **SDK gap** — no permission-reply endpoint. `permission.updated` events fire on the bus but can't be replied to. |
| `LIFECYCLE_HOOKS` | ✓ (6 of 8 kinds) | `session_start`, `session_end`, `user_prompt_submit`, `pre_tool_use`, `post_tool_use`, `tool_failure`. Not emitted: `pre_compact`, `rate_limit`. |
| `BUDGET_USD_CAP` | ✓ (best-effort) | Depends on upstream reporting cost. Ollama / llama.cpp don't; the cap is effectively unenforced for those backends. |
| `BUDGET_TURN_CAP` | ✓ | Client-side per-session turn counter. |

## OpenCodeServerOptions

```python
OpenCodeServerOptions(
    provider_id=None,                # explicit upstream routing
    available_tools=None,            # tuple[str, ...] — built-in tool allowlist
    excluded_tools=(),               # tuple[str, ...] — denylist
    working_directory=None,          # server-side cwd (interpreted by the server)
    permission_mode=None,            # reserved for future SDK support
    additional_mcp_servers=(),       # reserved
    additional_request_fields=None,  # reserved
)
```

`available_tools` / `excluded_tools` wire through to
`session.chat(tools={"bash": True, "write": False, ...})` —
OpenCode's per-session enable/disable map for its built-in tools.
Denylist entries override allowlist on overlap.

Other fields are scaffolded for forward compatibility — when the
SDK exposes the matching endpoints, the wiring lands without
breaking the namespace.

## Vendor quirks & landmines

1. **Pre-1.0 SDK churn.** `opencode-ai` is `0.1.0a*`. Stainless
   regenerates the client off the server's OpenAPI spec; breaking
   field changes ship between alphas. We pin tightly.
2. **Server-side filesystem.** Tools (`bash`, `write`, `edit`) run on
   the **OpenCode server's** filesystem — not the adapter's. For
   loopback `opencode serve` they're the same; for remote / container
   deployments, edits land where the server sees them. The adapter
   surfaces this in the `working_directory` option but cannot fix it.
3. **Permission events are observe-only.** The SSE bus emits
   `permission.updated` events you'd normally gate with a
   `PermissionCallback`, but the 0.1.0a36 SDK has no reply endpoint.
   `on_permission=` raises `UnsupportedFeatureError` until the SDK
   catches up. Observers can watch `permission.updated` flow through
   `on_event=` once airframe surfaces them (currently observation is
   on the airframe side via the streaming bus only).
4. **Streaming deltas are synthetic.** OpenCode emits "part updated"
   snapshots, not native delta events. Airframe computes the diff
   against the prior snapshot per part id. Mostly transparent, but a
   server that retransmits the entire part (rather than appending)
   would surface as a single large delta rather than a continuous
   stream.
5. **Reasoning pass-through is per-upstream.** The envelope we send
   depends on the upstream: OpenAI-shape for OpenAI / OpenRouter /
   Ollama, Anthropic-shape for Anthropic. If you switch upstreams
   mid-session via `OpenCodeServerOptions(provider_id=)`, the
   envelope re-resolves per turn.
6. **Cost reporting is backend-dependent.** When the upstream
   reports cost, `RuntimeResult.cost.cost_usd` is populated; when it
   doesn't (self-hosted Ollama, llama.cpp), it stays `None` and
   `BUDGET_USD_CAP` becomes effectively unenforced for that turn —
   we debug-log it so it's visible.

## Native escape hatches

```python
from opencode_ai import AsyncOpencode

client = runtime.unwrap(AsyncOpencode)  # the live SDK client
sessions = await client.session.list()  # SDK-typed call
```

Reach the raw HTTP client when you need an endpoint the airframe
adapter doesn't expose. Sessions have no native unwrap target —
the session id is the handle.

## See also

- [`docs/auth.md` § OpenCodeServerRuntime](../auth.md#opencodeserverruntime)
  — full auth chain.
- [`docs/capabilities.md`](../capabilities.md) — feature matrix
  across adapters.
- [`OpenCode docs`](https://opencode.ai/docs/) — server config, tool
  list, mode definitions.
- [`opencode-ai` PyPI](https://pypi.org/project/opencode-ai/) —
  upstream SDK release notes.
