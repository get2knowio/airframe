# airframe

[![PyPI version](https://img.shields.io/pypi/v/airframe-agents.svg)](https://pypi.org/project/airframe-agents/)
[![Python versions](https://img.shields.io/pypi/pyversions/airframe-agents.svg)](https://pypi.org/project/airframe-agents/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/get2knowio/airframe/blob/main/LICENSE)
[![CI](https://github.com/get2knowio/airframe/actions/workflows/ci.yml/badge.svg)](https://github.com/get2knowio/airframe/actions/workflows/ci.yml)

**One protocol, every agent SDK.** Vendor-neutral runtime for
Python AI agents — write once against a small `AgentRuntime`
protocol and run on AWS Bedrock, Claude Code, GitHub Copilot,
Moonshot Kimi, OpenCode Go, OpenCode Zen, or OpenRouter by
changing a single config value.

## Quickstart

```bash
pip install airframe-agents[claude]   # or [bedrock] / [copilot] / [kimi] / [openai-compat] / [all]
```

```python
from airframe import runtime_for, ProviderModel
from pydantic import BaseModel

class Brief(BaseModel):
    summary: str
    risks: list[str]

# Provider ID comes from config — YAML, env, CLI flag, whatever.
provider_id = "claude"  # or bedrock / github-copilot / kimi / opencode-go / opencode-zen / openrouter

cls = runtime_for(provider_id)       # discovery lookup by ID
runtime = cls()                      # auth resolves from env / credential files
result = await runtime.execute(
    "Brief me on the project structure.",
    schema=Brief,
    model=ProviderModel(provider_id, "claude-haiku-4-5"),
)
print(result.structured)     # {"summary": "...", "risks": [...]}
print(result.cost.cost_usd)  # 0.0042
await runtime.close()
```

The same agent code now serves any installed adapter — swap
`provider_id` (and `model`) in config, no import or instantiation
changes. Add a new vendor to your project's YAML and ship.

Direct imports still work when you only ever need one adapter:

```python
from airframe import ClaudeCodeRuntime
runtime = ClaudeCodeRuntime()
```

Use `list_providers()` to enumerate installed adapters at startup
(handy for validating YAML config):

```python
from airframe import list_providers
list_providers()  # ["claude", "github-copilot"]  — whichever extras are installed
```

The PyPI distribution name is `airframe-agents`. The import name
is `airframe`.

## Supported providers

| Adapter | `PROVIDER_ID` | Vendor SDK | Auth |
|---|---|---|---|
| [`BedrockRuntime`](https://github.com/get2knowio/airframe/blob/main/docs/adapters/bedrock.md) | `bedrock` | `aioboto3` | boto3 chain (env / `AWS_PROFILE` / IAM role) + `AWS_REGION` |
| [`ClaudeCodeRuntime`](https://github.com/get2knowio/airframe/blob/main/docs/adapters/claude.md) | `claude` | `claude-agent-sdk` | Claude Max OAuth → `~/.claude/credentials.json` → `ANTHROPIC_API_KEY` |
| [`CopilotRuntime`](https://github.com/get2knowio/airframe/blob/main/docs/adapters/copilot.md) | `github-copilot` | `github-copilot-sdk` | `GITHUB_TOKEN` → `gh auth` |
| [`KimiRuntime`](https://github.com/get2knowio/airframe/blob/main/docs/adapters/kimi.md) | `kimi` | `kimi-agent-sdk` | `KIMI_API_KEY` (Python 3.12+ only; mcp-version conflict with `[claude]`) |
| [`OpenCodeGoRuntime`](https://github.com/get2knowio/airframe/blob/main/docs/adapters/opencode-go.md) | `opencode-go` | OpenAI compatible | `OPENCODE_API_KEY` → opencode `auth.json::opencode-go.key` |
| [`OpenCodeZenRuntime`](https://github.com/get2knowio/airframe/blob/main/docs/adapters/opencode-zen.md) | `opencode-zen` | OpenAI compatible | `OPENCODE_API_KEY` → opencode `auth.json::opencode.key` |
| [`OpenRouterRuntime`](https://github.com/get2knowio/airframe/blob/main/docs/adapters/openrouter.md) | `openrouter` | OpenAI compatible | `OPENROUTER_API_KEY` |

The OpenAI-compatible family (`OpenCodeZenRuntime` per-token,
`OpenCodeGoRuntime` subscription, `OpenRouterRuntime` multi-vendor
gateway today; Together / Groq / Fireworks as future siblings) shares
the `OpenAICompatibleRuntime` base — subclasses are ~30 lines. See
[`docs/adapters/third-party.md`](https://github.com/get2knowio/airframe/blob/main/docs/adapters/third-party.md).

`BedrockRuntime` is the enterprise / managed-cloud path —
AWS-billed access to a multi-vendor catalog (Anthropic, Meta,
Mistral, Cohere, Amazon Nova) behind IAM-rooted auth and region
pinning. Distinct from the OpenAI-compatible family because Bedrock
speaks Converse, not Chat Completions.

Each adapter has one canonical provider ID. `"anthropic"` is
reserved for a future direct-API `AnthropicRuntime`; `"openai"`
for a future `OpenAIRuntime`. Current adapters cover the
*subscription* paths (Claude Max, Copilot, ChatGPT Plus,
opencode-go), the *per-token gateways* (OpenCode Zen, OpenRouter),
and the *AWS-billed managed* path (Bedrock).

`ClaudeCodeRuntime` is the only adapter that accepts Claude
bindings. `CopilotRuntime` declines them — Claude served via
Copilot Chat Completions emits markdown-fenced JSON instead of
honouring tool calls, so it can't satisfy the structured-output
contract.

## Capability matrix

Current snapshot (run
`uv run python examples/probe_supports.py` for the live version):

| Feature | Bedrock | Claude | Copilot | Kimi | OpenAI-compat |
|---|---|---|---|---|---|
| `STRUCTURED_OUTPUT_JSON_SCHEMA` | ✓ | ✓ | ✓ | ◐ | ✓ |
| `STREAMING` / `CANCEL` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SESSION_RESUME` | ✗ | ✓ | ✓ | ✓ | ✗ |
| `REASONING_EFFORT` | ✓ (Anthropic) | ✓ | ✓ | ✓ (bool) | ✓ |
| `REASONING_BUDGET_TOKENS` | ✓ (Anthropic) | ✓ | ✗ | ✗ | ✗ |
| `VISION_INPUT` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `FILE_INPUT` | ✓ (Anthropic) | ✓ | ✓ | ✗ | ✗ |
| `TOOLS_FUNCTION` | ✓ | ✓ | ✓ | ✗ | ✓ |
| `TOOLS_MCP_STDIO` / `_HTTP` | ✗ | ✓ | ✓ | ✓ | ✗ |
| `TOOLS_MCP_SSE` | ✗ | ✓ | ✗ | ✓ | ✗ |
| `PERMISSION_CALLBACK` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `LIFECYCLE_HOOKS` | ✓ (6 kinds) | ✓ (8 kinds) | ✓ (7 kinds) | ✓ (7 kinds) | ✓ (6 kinds) |
| `BUDGET_USD_CAP` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `BUDGET_TURN_CAP` | ✓ | ✓ | ✗ | ✓ | ✓ |

Capability flags are statically declared per adapter. Check
`runtime.supports(Feature.X)` before invoking a feature; declined
capabilities raise `UnsupportedFeatureError` with a `feature=`
attribute so the call fails fast.

Full per-feature semantics in [`docs/capabilities.md`](https://github.com/get2knowio/airframe/blob/main/docs/capabilities.md);
per-adapter quirks under [`docs/adapters/`](https://github.com/get2knowio/airframe/tree/main/docs/adapters).

## Why?

Each vendor ships a Python SDK that does something subtly different:
the Claude Agent SDK exposes a subprocess + JSON-RPC interface;
GitHub's Copilot SDK exposes a session + tool registration model;
Moonshot's Kimi Agent SDK spawns the `kimi-cli` subprocess and
streams typed `WireMessage` events; AWS Bedrock's `aioboto3` client
fronts a multi-vendor model catalog behind the Converse envelope
and IAM auth; the OpenAI-compatible gateways (OpenCode Zen,
OpenCode Go, OpenRouter) speak Chat Completions HTTP. Each has its
own auth chain, error taxonomy, cost-reporting shape,
structured-output mechanism, and
models-endpoint shape. Airframe
collapses those differences behind one
`execute / session / reset / close / validate_binding / list_models /
supports / unwrap` interface, classifies every vendor's failures
into a single hierarchy, and produces a single `CostRecord` /
`ModelInfo` shape regardless of the vendor.

The protocol is intentionally narrow. The eight methods are the
contract; everything else (auth chains, session caching, tool-call
forcing, JSON-schema mode, envelope unwrapping, per-model metadata
joining) lives inside each adapter, where vendor-specific
behaviour belongs.

Anything *above* the protocol — retry policy, fallback across
vendors, conversation memory, multi-agent orchestration — is left
to the consumer. Airframe is the adapter layer; the application
composes its own behaviour on top.

The shape — one narrow protocol plus pluggable vendor adapters,
discovered by ID — is borrowed from JDBC, with the same goal:
let the application code stay vendor-agnostic while each adapter
absorbs its vendor's quirks.

## Install

```bash
pip install airframe-agents[bedrock]        # BedrockRuntime
pip install airframe-agents[claude]         # ClaudeCodeRuntime
pip install airframe-agents[copilot]        # CopilotRuntime
pip install airframe-agents[kimi]           # KimiRuntime (Python 3.12+, separate venv — see note below)
pip install airframe-agents[openai-compat]  # OpenCodeGoRuntime + OpenCodeZenRuntime + OpenRouterRuntime (+ future siblings)
pip install airframe-agents[all]            # Everything except [kimi] (mcp-version conflict)
pip install airframe-agents[testing]        # Conformance contract suite (pytest)
```

**`[kimi]` co-installation note.** `kimi-agent-sdk` 0.0.5 →
`kimi-cli` 1.12 → `fastmcp` 2.12.5 → `mcp<1.17`, but
`claude-agent-sdk` 0.2 requires `mcp>=1.23`. The two SDKs cannot
co-install in one environment until upstream resolves; airframe
declares the conflict in `[tool.uv.conflicts]` and excludes
`[kimi]` from `[all]`. Users wanting both extras must split into
separate venvs.

`list_providers()` filters by which extras you installed:
`airframe-agents[copilot]` makes `list_providers()` return
`["github-copilot"]`. Pass `installed_only=False` to see every
built-in provider for documentation UIs.

## Sessions, streaming, and the new kwargs

`runtime.execute(...)` is convenient single-turn sugar. The full
surface lives on `runtime.session(...)`:

```python
from airframe import (
    ClaudeCodeRuntime, FunctionTool, McpServerRef,
    PermissionCallback, PermissionDecision, PermissionRequest,
    HookEvent, ClaudeOptions, TextDelta, TurnComplete,
)
from pydantic import BaseModel

class AddArgs(BaseModel):
    a: float
    b: float

async def add(args: AddArgs) -> float:
    return args.a + args.b

class ApproveAll(PermissionCallback):
    async def handle(self, req: PermissionRequest) -> PermissionDecision:
        return "allow"

def log_event(e: HookEvent) -> None:
    print(f"[{e.kind}] {e.payload}")

runtime = ClaudeCodeRuntime()
sess = runtime.session(
    system="You are a careful math assistant.",
    tools=[FunctionTool(name="add", description="Add two numbers.",
                        params=AddArgs, handler=add)],
    mcp_servers=[McpServerRef(name="docs", transport="http",
                              url="https://mcp.example.com",
                              auth_token="...")],
    on_permission=ApproveAll(),
    on_event=log_event,
    provider_options=ClaudeOptions(strict_mcp_config=True),
)
try:
    async for event in sess.stream("What is 17 + 25?",
                                    max_turns=10, max_budget_usd=0.05):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, TurnComplete):
            print(f"\nfinal cost: ${event.result.cost.cost_usd}")
finally:
    await sess.close()
```

`session.stream()` yields a discriminated union of five event
variants: `TextDelta`, `ReasoningDelta`, `ToolCallStart`,
`ToolCallResult`, `TurnComplete`. The variant set is shape-locked.

Per-kwarg semantics live in
[`docs/capabilities.md`](https://github.com/get2knowio/airframe/blob/main/docs/capabilities.md); per-adapter quirks
in each [`docs/adapters/`](https://github.com/get2knowio/airframe/tree/main/docs/adapters) page.

## Errors

Adapters classify vendor failures into a small hierarchy:

| Error | Meaning |
|---|---|
| `RuntimeAuthError` | Credentials bad / expired / missing |
| `RuntimeModelNotFoundError` | Server doesn't serve that model on this binding |
| `RuntimeTransientError` | 5xx, rate limit, brief outage — recoverable |
| `RuntimeStructuredOutputError` | Transport OK; payload didn't match schema |
| `RuntimeBudgetExceededError` | `max_turns=` / `max_budget_usd=` cap tripped |
| `UnsupportedFeatureError` | Capability declined (carries `feature=` attr) |

Full list and the rest of the hierarchy in
[`docs/reference.md#errors`](https://github.com/get2knowio/airframe/blob/main/docs/reference.md#errors).

## Escape hatch: `unwrap()`

When the portable surface doesn't expose a vendor-specific knob,
reach the native SDK object via `unwrap()`:

```python
from claude_agent_sdk import ClaudeSDKClient
sess = runtime.session()
await sess.execute("hi")
client: ClaudeSDKClient = sess.unwrap(ClaudeSDKClient)
await client.interrupt()
```

Each adapter declares the native types it accepts; unsupported
types raise `TypeError`. Runtime-level types via
`runtime.unwrap(...)`; session-level vendor objects via
`session.unwrap(...)`.

## Live probes

`examples/probe_*.py` exercise each adapter end-to-end against a
real CLI / HTTP endpoint. They're runnable demos, not part of
`make test`. Auth issues surface as classified `Runtime*Error`.

```bash
uv run python examples/probe_supports.py        # capability matrix
uv run python examples/probe_streaming.py       # stream() against any installed adapter
uv run python examples/probe_tools.py           # FunctionTool round-trip
uv run python examples/probe_mcp.py             # external MCP server
uv run python examples/probe_permission.py      # PermissionCallback
uv run python examples/probe_hooks.py           # HookEvent observation
uv run python examples/probe_budget.py          # max_turns / max_budget_usd
```

Full list with one-line descriptions in
[`docs/cookbook.md`](https://github.com/get2knowio/airframe/blob/main/docs/cookbook.md).

## Documentation

- **[Architecture & design](https://github.com/get2knowio/airframe/blob/main/docs/architecture.md)** — protocol
  shape, runtime-vs-session split, streaming event taxonomy.
- **[Capabilities](https://github.com/get2knowio/airframe/blob/main/docs/capabilities.md)** — per-`Feature`
  semantics across adapters.
- **[Authentication](https://github.com/get2knowio/airframe/blob/main/docs/auth.md)** — per-adapter credential
  resolution chains and CI patterns.
- **[API reference](https://github.com/get2knowio/airframe/blob/main/docs/reference.md)** — every public name with
  cross-links into the source.
- **[Cookbook](https://github.com/get2knowio/airframe/blob/main/docs/cookbook.md)** — runnable recipes via the
  probe scripts.
- **[Per-adapter notes](https://github.com/get2knowio/airframe/tree/main/docs/adapters)** —
  [Bedrock](https://github.com/get2knowio/airframe/blob/main/docs/adapters/bedrock.md) ·
  [Claude](https://github.com/get2knowio/airframe/blob/main/docs/adapters/claude.md) ·
  [Copilot](https://github.com/get2knowio/airframe/blob/main/docs/adapters/copilot.md) ·
  [Kimi](https://github.com/get2knowio/airframe/blob/main/docs/adapters/kimi.md) ·
  [OpenCode Go](https://github.com/get2knowio/airframe/blob/main/docs/adapters/opencode-go.md) ·
  [OpenCode Zen](https://github.com/get2knowio/airframe/blob/main/docs/adapters/opencode-zen.md) ·
  [OpenRouter](https://github.com/get2knowio/airframe/blob/main/docs/adapters/openrouter.md).
- **[Writing your own adapter](https://github.com/get2knowio/airframe/blob/main/docs/adapters/third-party.md)** —
  the `airframe.adapters` entry-point group + conformance
  contracts.
- **[Changelog](https://github.com/get2knowio/airframe/blob/main/CHANGELOG.md)** · **[Contributing](https://github.com/get2knowio/airframe/blob/main/CONTRIBUTING.md)** · **[Security](https://github.com/get2knowio/airframe/blob/main/SECURITY.md)**.

## Development

```bash
uv sync --all-extras --group dev
make test          # full suite (incl. integration tests, which self-skip without creds)
make test-fast     # exclude `integration` marker
make lint          # ruff
make typecheck     # mypy
make ci            # lint + format + typecheck + test
```

Integration tests run automatically when credentials for an
adapter are configured (see [auth.md](https://github.com/get2knowio/airframe/blob/main/docs/auth.md)).

## License

MIT — see [LICENSE](https://github.com/get2knowio/airframe/blob/main/LICENSE).
