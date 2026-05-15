# airframe

**One protocol, pluggable adapters.** Vendor-neutral agent runtime
for Python — write your agent against a small `AgentRuntime`
protocol and switch between Claude Code, GitHub Copilot, OpenAI
Codex, or OpenCode Zen by swapping a single object.

```python
from airframe import ClaudeCodeRuntime, ProviderModel
from pydantic import BaseModel

class Brief(BaseModel):
    summary: str
    risks: list[str]

runtime = ClaudeCodeRuntime()
result = await runtime.execute(
    "Brief me on the project structure.",
    schema=Brief,
    model=ProviderModel("anthropic", "claude-haiku-4-5"),
)
print(result.structured)     # {"summary": "...", "risks": [...]}
print(result.cost.cost_usd)  # 0.0042
await runtime.aclose()
```

Swap the adapter, keep the schema:

```python
# Same Brief schema; different vendor, different auth path.
from airframe import CopilotRuntime, ProviderModel
runtime = CopilotRuntime()
result = await runtime.execute(
    "Brief me on the project structure.",
    schema=Brief,
    model=ProviderModel("copilot", "gpt-5-mini"),
)
```

## Why?

Each vendor ships a Python SDK that does something subtly different:
the Claude Agent SDK exposes a subprocess + JSON-RPC interface;
GitHub's Copilot SDK exposes a session + tool registration model;
OpenAI's Codex SDK passes a JSON Schema flag to a CLI subprocess;
the opencode-go Zen gateway speaks OpenAI-compatible HTTP. Each has
its own auth chain, error taxonomy, cost-reporting shape, and
structured-output mechanism. Airframe collapses those differences
behind one `execute / reset / aclose / validate_binding` interface,
classifies every vendor's failures into a single hierarchy
(`RuntimeAuthError`, `RuntimeTransientError`,
`RuntimeStructuredOutputError`, etc.), and produces a single
`CostRecord` shape regardless of the vendor.

The protocol is intentionally narrow. The four methods are the
contract; everything else (auth chains, session caching, tool-call
forcing, JSON-schema mode, envelope unwrapping) lives inside each
adapter, where vendor-specific behaviour belongs.

## Install

The base package ships only the protocol + error types. Each
adapter's SDK is an optional extra so you only pull what you need:

```bash
pip install airframe-agents[claude]        # ClaudeCodeRuntime
pip install airframe-agents[copilot]       # CopilotRuntime
pip install airframe-agents[codex]         # CodexRuntime
pip install airframe-agents[opencode-zen]  # OpenCodeZenRuntime
pip install airframe-agents[all]           # Everything
```

The import name is `airframe`. The PyPI dist name is
`airframe-agents` (the unqualified `airframe` slot on PyPI was taken
by an unrelated abandoned project).

## The four adapters

| Adapter | Vendor SDK | Auth | Structured output | Subprocess? |
| --- | --- | --- | --- | --- |
| `ClaudeCodeRuntime` | `claude-agent-sdk` | Claude Max OAuth → `~/.claude/credentials.json` → `ANTHROPIC_API_KEY` | Forced `submit_result` MCP tool | yes, per-runtime |
| `CopilotRuntime` | `github-copilot-sdk` | `GITHUB_TOKEN` → `gh auth` | Forced `submit_result` tool | yes, per-runtime |
| `CodexRuntime` | `openai-codex-sdk` | `OPENAI_API_KEY` → opencode `auth.json` → `~/.codex/auth.json` | Native JSON-schema flag (`--output-schema`) | yes, per-turn |
| `OpenCodeZenRuntime` | `openai` (HTTP) | `OPENCODE_API_KEY` → opencode `auth.json` | `response_format={type: "json_schema"}` | no (direct HTTP) |

`ClaudeCodeRuntime` is the only adapter that accepts Claude bindings.
`CopilotRuntime.validate_binding` *rejects* Claude bindings on
purpose — Claude served via Copilot Chat Completions emits markdown-
fenced JSON instead of honouring tool calls, so it can't satisfy the
structured-output contract. Route Claude through `ClaudeCodeRuntime`.

## The protocol

```python
class AgentRuntime(Protocol):
    label: str

    async def execute(
        self,
        prompt: str,
        *,
        schema: type[BaseModel] | None = None,
        system: str | None = None,
        persona: str | None = None,
        model: ProviderModel | None = None,
        timeout: float = 600.0,
    ) -> RuntimeResult: ...

    async def reset(self) -> None: ...
    async def aclose(self) -> None: ...
    def validate_binding(self, binding: ProviderModel) -> bool: ...
```

* **`execute`** — send one prompt, get back a `RuntimeResult` with
  text + (optional) structured payload + cost + finish reason.
* **`reset`** — drop accumulated context for a fresh scope
  (typically between tasks). Runtime-wide resources (subprocess
  pool, HTTP client) survive.
* **`aclose`** — full teardown. Idempotent; never raises.
* **`validate_binding`** — quick check the runtime can serve a given
  `(provider_id, model_id)`. Cascade machinery uses this to skip
  bindings before attempting them.

See [docs/architecture.md](docs/architecture.md) for the design
rationale and the operational landmines each adapter mitigates.

## Examples

Probe scripts under `examples/` exercise each adapter end-to-end
against a real CLI / HTTP endpoint:

```bash
uv run python examples/probe_claude_code.py
uv run python examples/probe_copilot.py
uv run python examples/probe_codex.py
uv run python examples/probe_opencode_zen.py
```

## Adding an adapter

A new adapter is one class implementing `AgentRuntime`:

1. Map your vendor's auth chain into a `_resolve_api_key()` / token
   helper.
2. Choose a structured-output mechanism — native JSON-schema if the
   vendor supports it, otherwise force a `submit_result` tool call.
3. Classify the vendor's exceptions into the
   `airframe.errors.Runtime*Error` hierarchy.
4. Populate a `CostRecord` from the vendor's usage report. Use the
   per-model pricing table when the vendor doesn't return cost
   directly.
5. Add `SUPPORTED_PROVIDER_IDS` and implement `validate_binding`.

See `src/airframe/adapters/opencode_zen.py` for the simplest example
(stateless HTTP, no subprocess), and `src/airframe/adapters/claude_code.py`
for the most complex (subprocess + MCP tool registration + session
caching).

## Development

```bash
uv sync --all-extras --group dev
make test       # All tests
make lint       # Ruff
make typecheck  # mypy
make ci         # Lint + format + typecheck + test
```

## License

MIT — see [LICENSE](LICENSE).
