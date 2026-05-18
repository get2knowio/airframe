# KimiRuntime

Routes work through Moonshot AI's Kimi K2 line via the official
[`kimi-agent-sdk`](https://pypi.org/project/kimi-agent-sdk/) package
— a thin Python surface over the `kimi-cli` subprocess. The
architectural sibling to `ClaudeCodeRuntime`: a subprocess-class
agent SDK with sessions, streaming, approvals, and MCP.

| | |
|---|---|
| **PROVIDER_ID** | `kimi` |
| **Pip extra** | `airframe-agents[kimi]` |
| **Vendor SDK** | `kimi-agent-sdk` |
| **Transport** | Subprocess + WireMessage stream |
| **Python floor** | 3.12 (stricter than airframe's 3.11) |
| **Authentication** | See [auth.md](../auth.md#kimiruntime) |

## Install

```bash
pip install airframe-agents[kimi]
```

**Co-installation conflict with Claude.** `kimi-agent-sdk` 0.0.5 →
`kimi-cli` 1.12 → `fastmcp` 2.12.5 → `mcp<1.17`; `claude-agent-sdk`
0.2 requires `mcp>=1.23`. The two SDKs cannot co-install in one
environment until upstream resolves. airframe declares this in
`[tool.uv.conflicts]` so `uv sync` picks one side at lock time;
users wanting both must split into separate venvs:

```bash
python3.12 -m venv .venv-kimi
.venv-kimi/bin/pip install -U pip
.venv-kimi/bin/pip install 'airframe-agents[kimi]'
```

You also need the `kimi-cli` subprocess on your PATH —
[install instructions](https://github.com/MoonshotAI/kimi-cli).

## Quickstart

```python
from airframe import KimiRuntime, ProviderModel

runtime = KimiRuntime()
result = await runtime.execute(
    "In one sentence: what is airframe-agents?",
    model=ProviderModel("kimi", "kimi-k2-thinking-turbo"),
)
print(result.text)
print(f"{result.cost.input_tokens=} {result.cost.output_tokens=}")
print(f"{result.cost.cost_usd=}")
await runtime.close()
```

## Supported features

| Feature | Status | Notes |
|---|---|---|
| `STRUCTURED_OUTPUT_JSON_SCHEMA` | ◐ scaffolded | Declared True so the conformance floor is honoured, but `execute(schema=…)` raises `NotImplementedError`. The Kimi SDK has no JSON-schema constraint knob; a future iteration will wire structured output via an MCP-based forced-tool, mirroring CopilotRuntime's pattern |
| `STREAMING` | ✓ | `Session.prompt()` is a native async generator over `WireMessage` |
| `SESSION_RESUME` | ✓ | `Session.resume(work_dir, session_id)` |
| `CANCEL` | ✓ | `Session.cancel()` sets the SDK's cancel event |
| `REASONING_EFFORT` | ✓ (boolean) | The SDK accepts a boolean `thinking` knob — every airframe effort literal collapses to `True`; the model itself decides depth |
| `REASONING_BUDGET_TOKENS` | ✗ | No token-budget channel on Kimi |
| `VISION_INPUT` | ✓ | `ImageInput(url=…)` passes through; `bytes_=` / `path=` become `data:` URIs |
| `FILE_INPUT` | ✗ | No prompt-side file slot in the SDK — surface files through `KimiOptions.working_directory` + tool reads instead |
| `TOOLS_FUNCTION` | ✗ | **Declined permanently** — kimi-agent-sdk's Python surface has no Python-callable tool channel. Wrap your function as an MCP server and pass it via `mcp_servers=` instead |
| `TOOLS_MCP_STDIO` | ✓ | Translated to fastmcp's `MCPConfig` dict shape |
| `TOOLS_MCP_HTTP` | ✓ | Same; `auth_token=` becomes `Authorization: Bearer …` |
| `TOOLS_MCP_SSE` | ✓ | Same; `transport="sse"` round-trips |
| `TOOLS_MCP_IN_PROCESS` | ✗ (permanent) | No in-process MCP slot in the SDK |
| `PERMISSION_CALLBACK` | ✓ (per-call) | Each `ApprovalRequest` dispatches to your callback. `allow→approve`, `deny→reject`, `defer→reject+feedback` (Kimi's approval channel is synchronous so defer collapses to a rejection — the feedback string explains why to the model) |
| `LIFECYCLE_HOOKS` | ✓ (7 kinds) | All but `rate_limit` — Moonshot raises 429s as `APIStatusError` exceptions rather than wire events |
| `BUDGET_USD_CAP` | ✓ | Pre-turn enforcement against the cumulative cost tracker |
| `BUDGET_TURN_CAP` | ✓ | Pre-turn enforcement against the session turn counter |

`STRUCTURED_OUTPUT_STRICT` stays False.

## KimiOptions (provider-options namespace)

```python
from airframe import KimiRuntime, KimiOptions

runtime = KimiRuntime()
sess = runtime.session(
    provider_options=KimiOptions(
        working_directory="/workspaces/my-repo",
        yolo=False,                         # don't auto-approve
        additional_mcp_servers=(
            {"mcpServers": {"telemetry": {"command": "uvx", "args": ["mcp-telemetry"]}}},
        ),
        skill_directories=("/opt/team-skills",),
        additional_config_fields={"telemetry": False},
    )
)
```

| Field | Type | Maps to |
|---|---|---|
| `working_directory` | `str \| None` | `Session.create(work_dir=KaosPath(...))` — the cwd Kimi's filesystem tools operate against |
| `yolo` | `bool` | `Session.create(yolo=...)` — auto-approve every tool call. **Mutually exclusive with `on_permission=`** — passing both raises `UnsupportedFeatureError` at `runtime.session()` |
| `additional_mcp_servers` | `tuple[Any, ...]` | Extra entries appended to `Session.create(mcp_configs=...)` after the airframe-synthesised entries from `mcp_servers=` |
| `skill_directories` | `tuple[str, ...]` | `Session.create(skills_dir=KaosPath(first))` — Kimi's skill discovery dir. The SDK accepts a single dir today; airframe surfaces a tuple so a future SDK widening requires no caller-side change |
| `additional_config_fields` | `dict[str, Any] \| None` | Documented escape hatch for vendor-specific `Config` slots airframe doesn't surface portably |

All fields default to "no-op" — `KimiOptions()` is the empty default.

## Model IDs

`KimiRuntime.list_models()` returns a curated fallback catalogue today
(live `/v1/models` via Moonshot's OpenAI-compatible endpoint lands
in a later iteration once we have a non-conflicting venv to verify
against). Pricing comes from the in-tree `_KIMI_PRICING` table
(point-in-time numbers — see Cost reporting below).

- `kimi-k2-thinking-turbo` — default; the most capable reasoning
  variant.
- `kimi-k2-thinking` — slower / cheaper.

**Only `kimi-*` model IDs accepted.** `validate_binding()` returns
False for anything else (analogous to how Copilot rejects `claude-*`
and Codex rejects `claude-*`). Foreign provider IDs return False
rather than raise.

## Structured output

**Not yet wired.** `execute(schema=…)` raises `NotImplementedError`.
The Kimi SDK has no JSON-schema constraint knob; the wrap-don't-rewrite
principle (see `CLAUDE.md`) rules out prompt-engineering it inside the
adapter. A future iteration will wire it via an MCP-based forced-tool,
the same pattern `CopilotRuntime` uses.

Workarounds today: request JSON via prompt-engineering in your
application and parse the response yourself, OR check
`runtime.supports(Feature.STRUCTURED_OUTPUT_JSON_SCHEMA)` before
relying on the kwarg.

## Cost reporting

Token counts come from the SDK's `TokenUsage` wire messages.
`CostRecord.cost_usd` populates from `_KIMI_PRICING` for models in
the table; unknown model IDs return `cost_usd=None` with tokens
still populated.

Pricing table (point-in-time as of 2026-05-18; re-verify against
[Moonshot's pricing page](https://platform.moonshot.ai/docs/pricing)
before relying on it):

| Model | Input (per 1M) | Output (per 1M) | Cache read (per 1M) |
|---|---|---|---|
| `kimi-k2-thinking` | $0.60 | $2.50 | $0.15 |
| `kimi-k2-thinking-turbo` | $1.50 | $5.00 | $0.15 |

Cached input tokens bill at the cache rate (the cheaper rate);
fresh-input tokens at the input rate. Cache-write isn't billed
separately on Moonshot today — those tokens already count as
fresh input on the turn that wrote them.

## Vendor quirks & landmines

- **mcp version conflict with Claude.** Documented under Install
  above — the canonical airframe-wide co-installation hazard. Use
  separate venvs until upstream resolves.
- **Python 3.12 floor.** `kimi-agent-sdk` requires Python ≥ 3.12.
  airframe's overall floor is 3.11; on 3.11 the `[kimi]` extra
  installs as a no-op (the `python_version` marker on the dep
  skips it) and the adapter's lazy import surfaces a clear error
  at first use.
- **`thinking` is boolean.** Every effort literal (`"minimal" |
  "low" | "medium" | "high"`) collapses to `thinking=True` —
  granularity is lost on the SDK boundary; the model decides
  depth.
- **`thinking` toggle between turns rebuilds the SDK session.**
  The SDK bakes `thinking` at `Session.create` time and never
  re-evaluates. Toggling rebuilds while preserving the session ID
  (re-resumes by ID), so multi-turn state survives.
- **`tools=` is a permanent decline.** No Python-callable
  tool-registration channel in the SDK. Wrap as MCP via
  `mcp_servers=` instead.
- **`defer` permission decisions collapse to reject + feedback.**
  Kimi's `ApprovalRequest` channel is synchronous — there's no
  "ask the human later" path. The feedback string explains the
  situation to the model.
- **No `pre_compact` / `rate_limit` events.** Wait — `pre_compact`
  *is* emitted (from `CompactionBegin`); `rate_limit` is not
  (Moonshot raises 429s as exceptions rather than wire events).
- **`KimiOptions(yolo=True)` and `on_permission=callback` are
  mutually exclusive.** Pick one. Passing both raises
  `UnsupportedFeatureError` at `runtime.session()`.
- **subprocess discovery.** Like `ClaudeCodeRuntime`'s posture, the
  adapter does *not* bundle `kimi-cli` — the user installs it
  themselves.

## Native escape hatches

```python
import kimi_agent_sdk
sess = runtime.session()
await sess.execute("hi")
sdk_session: kimi_agent_sdk.Session = sess.unwrap(kimi_agent_sdk.Session)
# Use the underlying SDK session for status snapshots, etc.
print(sdk_session.status)
```

## See also

- [auth.md](../auth.md#kimiruntime)
- [capabilities.md](../capabilities.md)
- [Kimi Agent SDK on GitHub](https://github.com/MoonshotAI/kimi-agent-sdk)
- [Kimi CLI on GitHub](https://github.com/MoonshotAI/kimi-cli)
- [Moonshot platform docs](https://platform.moonshot.ai/docs)
