# airframe

[![PyPI version](https://img.shields.io/pypi/v/airframe-agents.svg)](https://pypi.org/project/airframe-agents/)
[![Python versions](https://img.shields.io/pypi/pyversions/airframe-agents.svg)](https://pypi.org/project/airframe-agents/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/get2knowio/airframe/actions/workflows/ci.yml/badge.svg)](https://github.com/get2knowio/airframe/actions/workflows/ci.yml)

**One protocol, every agent SDK.** Vendor-neutral runtime for
Python AI agents — write once against a small `AgentRuntime`
protocol and run on Claude Code, GitHub Copilot, OpenAI Codex, or
OpenCode Zen by changing a single config value.

## Quickstart

```bash
pip install airframe-agents[claude]   # or [copilot] / [codex] / [openai-compat] / [all]
```

```python
from airframe import runtime_for, ProviderModel
from pydantic import BaseModel

class Brief(BaseModel):
    summary: str
    risks: list[str]

# Provider ID comes from config — YAML, env, CLI flag, whatever.
provider_id = "claude"  # or "github-copilot", "codex", "opencode-zen", "opencode-go"

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
list_providers()  # ["claude", "codex"]  — whichever extras are installed
```

The PyPI distribution name is `airframe-agents`. The import name
is `airframe`.

## Supported providers

| Adapter | `PROVIDER_ID` | Vendor SDK | Auth | Subprocess? |
|---|---|---|---|---|
| [`ClaudeCodeRuntime`](docs/adapters/claude.md) | `claude` | `claude-agent-sdk` | Claude Max OAuth → `~/.claude/credentials.json` → `ANTHROPIC_API_KEY` | yes, per-runtime |
| [`CopilotRuntime`](docs/adapters/copilot.md) | `github-copilot` | `github-copilot-sdk` | `GITHUB_TOKEN` → `gh auth` | yes, per-runtime |
| [`CodexRuntime`](docs/adapters/codex.md) | `codex` | `openai-codex-sdk` | `OPENAI_API_KEY` → opencode `auth.json` → `~/.codex/auth.json` | yes, per-turn |
| [`OpenCodeZenRuntime`](docs/adapters/opencode-zen.md) | `opencode-zen` | `openai` (HTTP) | `OPENCODE_API_KEY` → opencode `auth.json::opencode.key` | no (direct HTTP) |
| [`OpenCodeGoRuntime`](docs/adapters/opencode-go.md) | `opencode-go` | `openai` (HTTP) | `OPENCODE_API_KEY` → opencode `auth.json::opencode-go.key` | no (direct HTTP) |

The OpenAI-compatible family (`OpenCodeZenRuntime` per-token and
`OpenCodeGoRuntime` subscription today; Together / Groq / Fireworks /
OpenRouter as future siblings) shares the `OpenAICompatibleRuntime`
base — subclasses are ~30 lines. See
[`docs/adapters/third-party.md`](docs/adapters/third-party.md).

Each adapter has one canonical provider ID. `"anthropic"` is
reserved for a future direct-API `AnthropicRuntime`; `"openai"`
for a future `OpenAIRuntime`. Current adapters cover the
*subscription* paths (Claude Max, Copilot, ChatGPT Plus,
opencode-go).

`ClaudeCodeRuntime` is the only adapter that accepts Claude
bindings. `CopilotRuntime` declines them — Claude served via
Copilot Chat Completions emits markdown-fenced JSON instead of
honouring tool calls, so it can't satisfy the structured-output
contract.

## Capability matrix

Current snapshot (run
`uv run python examples/probe_supports.py` for the live version):

| Feature | Claude | Copilot | Codex | OpenAI-compat |
|---|---|---|---|---|
| `STRUCTURED_OUTPUT_JSON_SCHEMA` | ✓ | ✓ | ✓ | ✓ |
| `STREAMING` / `CANCEL` | ✓ | ✓ | ✓ | ✓ |
| `SESSION_RESUME` | ✓ | ✓ | ✓ | ✗ |
| `REASONING_EFFORT` | ✓ | ✓ | ✓ | ✓ |
| `REASONING_BUDGET_TOKENS` | ✓ | ✗ | ✗ | ✗ |
| `VISION_INPUT` | ✓ | ✓ | ✓ | ✓ |
| `FILE_INPUT` | ✓ | ✓ | ✓ | ✗ |
| `TOOLS_FUNCTION` | ✓ | ✓ | ✗ | ✓ |
| `TOOLS_MCP_STDIO` / `_HTTP` | ✓ | ✓ | ✗ | ✗ |
| `TOOLS_MCP_SSE` | ✓ | ✗ | ✗ | ✗ |
| `PERMISSION_CALLBACK` | ✓ | ✓ | ✓ (session-wide) | ✗ |
| `LIFECYCLE_HOOKS` | ✓ (8 kinds) | ✓ (7 kinds) | ✓ (6 kinds) | ✓ (6 kinds) |
| `BUDGET_USD_CAP` | ✓ | ✓ | ✓ | ✓ |
| `BUDGET_TURN_CAP` | ✓ | ✗ | ✓ | ✓ |

Capability flags are statically declared per adapter. Check
`runtime.supports(Feature.X)` before invoking a feature; declined
capabilities raise `UnsupportedFeatureError` with a `feature=`
attribute so the call fails fast.

Full per-feature semantics in [`docs/capabilities.md`](docs/capabilities.md);
per-adapter quirks under [`docs/adapters/`](docs/adapters/).

## Why?

Each vendor ships a Python SDK that does something subtly different:
the Claude Agent SDK exposes a subprocess + JSON-RPC interface;
GitHub's Copilot SDK exposes a session + tool registration model;
OpenAI's Codex SDK passes a JSON Schema flag to a CLI subprocess;
the opencode-go Zen gateway speaks OpenAI-compatible HTTP. Each
has its own auth chain, error taxonomy, cost-reporting shape,
structured-output mechanism, and models-endpoint shape. Airframe
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
pip install airframe-agents[claude]         # ClaudeCodeRuntime
pip install airframe-agents[copilot]        # CopilotRuntime
pip install airframe-agents[codex]          # CodexRuntime
pip install airframe-agents[openai-compat]  # OpenCodeZenRuntime + OpenCodeGoRuntime (+ future siblings)
pip install airframe-agents[all]            # Everything
pip install airframe-agents[testing]        # Conformance contract suite (pytest)
```

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
[`docs/capabilities.md`](docs/capabilities.md); per-adapter quirks
in each [`docs/adapters/`](docs/adapters/) page.

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
[`docs/reference.md#errors`](docs/reference.md#errors).

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
[`docs/cookbook.md`](docs/cookbook.md).

## Documentation

- **[Architecture & design](docs/architecture.md)** — protocol
  shape, runtime-vs-session split, streaming event taxonomy.
- **[Capabilities](docs/capabilities.md)** — per-`Feature`
  semantics across adapters.
- **[Authentication](docs/auth.md)** — per-adapter credential
  resolution chains and CI patterns.
- **[API reference](docs/reference.md)** — every public name with
  cross-links into the source.
- **[Cookbook](docs/cookbook.md)** — runnable recipes via the
  probe scripts.
- **[Per-adapter notes](docs/adapters/)** —
  [Claude](docs/adapters/claude.md) ·
  [Copilot](docs/adapters/copilot.md) ·
  [Codex](docs/adapters/codex.md) ·
  [OpenCode Zen](docs/adapters/opencode-zen.md) ·
  [OpenCode Go](docs/adapters/opencode-go.md).
- **[Writing your own adapter](docs/adapters/third-party.md)** —
  the `airframe.adapters` entry-point group + conformance
  contracts.
- **[Changelog](CHANGELOG.md)** · **[Contributing](CONTRIBUTING.md)** · **[Security](SECURITY.md)**.

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
adapter are configured (see [auth.md](docs/auth.md)).

## License

MIT — see [LICENSE](LICENSE).
