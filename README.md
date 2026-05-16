# airframe

**JDBC for LLM agent SDKs.** Vendor-neutral agent runtime for
Python — write your agent against a small `AgentRuntime` protocol
and switch between Claude Code, GitHub Copilot, OpenAI Codex, or
OpenCode Zen by swapping a single object.

Drives both *execution* (one prompt → typed payload + cost) and
*discovery* (which providers does this install support? which models
can the user pick from?). The same five-method protocol covers
agent-CLI subprocesses, OpenAI-compatible HTTP, and everything in
between.

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
    model=ProviderModel("claude", "claude-haiku-4-5"),
)
print(result.structured)     # {"summary": "...", "risks": [...]}
print(result.cost.cost_usd)  # 0.0042
await runtime.close()
```

Swap the adapter, keep the schema:

```python
# Same Brief schema; different vendor, different auth path.
from airframe import CopilotRuntime, ProviderModel
runtime = CopilotRuntime()
result = await runtime.execute(
    "Brief me on the project structure.",
    schema=Brief,
    model=ProviderModel("github-copilot", "gpt-5-mini"),
)
```

Discover providers and models at runtime (for menus, config UIs,
"what can I install / which models can I pick?"):

```python
from airframe import list_providers, runtime_for

# Filtered by which pip extras the consumer has installed.
for provider_id in list_providers():
    runtime_cls = runtime_for(provider_id)
    runtime = runtime_cls()
    try:
        models = await runtime.list_models()
        for m in models:
            print(provider_id, m.id, m.display_name, m.context_window)
    finally:
        await runtime.close()
```

## Why?

Each vendor ships a Python SDK that does something subtly different:
the Claude Agent SDK exposes a subprocess + JSON-RPC interface;
GitHub's Copilot SDK exposes a session + tool registration model;
OpenAI's Codex SDK passes a JSON Schema flag to a CLI subprocess;
the opencode-go Zen gateway speaks OpenAI-compatible HTTP. Each has
its own auth chain, error taxonomy, cost-reporting shape, structured-
output mechanism, and models-endpoint shape. Airframe collapses those
differences behind one
`execute / reset / close / validate_binding / list_models` interface,
classifies every vendor's failures into a single hierarchy, and
produces a single `CostRecord` / `ModelInfo` shape regardless of the
vendor.

The protocol is intentionally narrow. The five methods are the
contract; everything else (auth chains, session caching, tool-call
forcing, JSON-schema mode, envelope unwrapping, per-model metadata
joining) lives inside each adapter, where vendor-specific behaviour
belongs.

Anything *above* the protocol — retry policy, fallback across
vendors, conversation memory, multi-agent orchestration — is left to
the consumer. Airframe is the driver layer; the application
composes its own behaviour on top.

## Install

The base package ships only the protocol + error types. Each
adapter's SDK is an optional extra so you only pull what you need:

```bash
pip install airframe-agents[claude]         # ClaudeCodeRuntime
pip install airframe-agents[copilot]        # CopilotRuntime
pip install airframe-agents[codex]          # CodexRuntime
pip install airframe-agents[openai-compat]  # OpenCodeZenRuntime (+ future OAI-compat)
pip install airframe-agents[all]            # Everything
```

`list_providers()` filters by which extras you installed:
``pip install airframe-agents[copilot]`` makes ``list_providers()``
return ``["github-copilot"]``; the other adapters are silently filtered
so menus stay honest about what the local install can serve. Pass
``installed_only=False`` to see every built-in provider regardless of
SDK presence (useful for documentation UIs).

The import name is `airframe`. The PyPI dist name is
`airframe-agents` (the unqualified `airframe` slot on PyPI was taken
by an unrelated abandoned project).

## The four adapters

| Adapter | `PROVIDER_ID` | Vendor SDK | Auth | Structured output | Subprocess? |
| --- | --- | --- | --- | --- | --- |
| `ClaudeCodeRuntime` | `claude` | `claude-agent-sdk` | Claude Max OAuth → `~/.claude/credentials.json` → `ANTHROPIC_API_KEY` | Native `output_format={"type":"json_schema",...}` | yes, per-runtime |
| `CopilotRuntime` | `github-copilot` | `github-copilot-sdk` | `GITHUB_TOKEN` → `gh auth` | Forced `submit_result` tool (Copilot's native `define_tool`) | yes, per-runtime |
| `CodexRuntime` | `codex` | `openai-codex-sdk` | `OPENAI_API_KEY` → opencode `auth.json` → `~/.codex/auth.json` | Native JSON-schema flag (`--output-schema`) | yes, per-turn |
| `OpenCodeZenRuntime` | `opencode` | `openai` (HTTP) | `OPENCODE_API_KEY` → opencode `auth.json` | `response_format={"type":"json_schema",...}` | no (direct HTTP) |

The OpenAI-compatible family (`OpenCodeZenRuntime` today; Together /
Groq / Fireworks / OpenRouter as future siblings) shares the
`OpenAICompatibleRuntime` base class and the single `[openai-compat]`
pip extra. Subclasses are ~30 lines: declare `PROVIDER_ID`, a default
base URL + model, a per-model metadata table, and a vendor-specific
`_resolve_api_key()` hook — everything else (HTTP execute, structured
output, `list_models`, error classification) is inherited.

The provider IDs are deliberately strict: one canonical ID per
adapter, no aliases. ``"anthropic"`` is reserved for a future direct-
API `AnthropicRuntime`; ``"openai"`` is reserved for a future direct-
API `OpenAIRuntime`. The current adapters cover the *subscription*
paths (Claude Max, Copilot, ChatGPT Plus, opencode-go) — API-key-only
direct-vendor adapters can land later without breaking these IDs.

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
    async def close(self) -> None: ...
    def validate_binding(self, binding: ProviderModel) -> bool: ...
    async def list_models(self) -> list[ModelInfo]: ...
```

* **`execute`** — send one prompt, get back a `RuntimeResult` with
  text + (optional) structured payload + cost + finish reason.
* **`reset`** — drop accumulated context for a fresh scope
  (typically between tasks). Runtime-wide resources (subprocess
  pool, HTTP client) survive.
* **`close`** — full teardown. Idempotent; never raises.
* **`validate_binding`** — predicate: does this runtime serve a
  given `(provider_id, model_id)`? Cheap and non-async; suitable for
  filtering bindings before attempting them.
* **`list_models`** — hit the vendor's models endpoint with the
  user's resolved credentials, return a list of `ModelInfo` (id,
  display name, context window, pricing, capability flags). Drives
  UI menus. Async + auth-aware; raises `RuntimeAuthError` /
  `RuntimeTransientError` so the consumer can surface the failure
  before letting the user pick a model that would later fail to
  execute.

### Errors

Adapters classify vendor failures into a small hierarchy so consumer
`except` clauses don't need vendor-specific knowledge:

| Error | What it means |
| --- | --- |
| `RuntimeAuthError` | Credentials bad / expired / missing. |
| `RuntimeModelNotFoundError` | Server doesn't serve that model on this binding. |
| `RuntimeTransientError` | 5xx, rate limit, brief outage. Call was attempted; failure was recoverable. |
| `RuntimeStructuredOutputError` | Transport succeeded but model didn't produce a payload matching the schema. |
| `RuntimeContextOverflowError` | Prompt exceeded the model's context window. |
| `RuntimeProtocolError` | Adapter saw something it can't interpret (adapter / SDK bug). |
| `RuntimeServerStartError` | Adapter couldn't bring its backend up at all. |
| `RuntimeCancelledError` | Caller-initiated abort. |

What to *do* with each — retry, fall back to another binding,
surface to the user, escalate to a larger model — is consumer
policy. Airframe doesn't ship a retry / fallback engine; consumers
compose their own from these primitives.

See [docs/architecture.md](docs/architecture.md) for the design
rationale and the operational landmines each adapter mitigates.

## Examples

Probe scripts under `tests/` exercise each adapter end-to-end against
a real CLI / HTTP endpoint. They're not part of the unit suite — pytest
collects `test_*.py` only, so the `probe_*.py` scripts are runnable
demos that won't be picked up by `make test`.

```bash
# Per-adapter execute() probes (require auth for that vendor).
uv run python tests/probe_claude_code.py
uv run python tests/probe_copilot.py
uv run python tests/probe_codex.py
uv run python tests/probe_opencode_zen.py

# Live model-menu probe across every installed adapter.
uv run python tests/probe_list_models.py
uv run python tests/probe_list_models.py --provider claude
uv run python tests/probe_list_models.py --installed-only=false
```

Each probe surfaces auth / network issues as classified
`Runtime*Error` so you can see exactly what would happen in production
if credentials were misconfigured.

## Capability negotiation

Each adapter declares which protocol features it implements via
`runtime.supports(Feature.X)`. The `Feature` enum ships the whole
forward-looking set as of v0.3.0; later releases flip more bits on as
each phase lands its corresponding API:

```python
from airframe import ClaudeCodeRuntime, Feature

runtime = ClaudeCodeRuntime()
if runtime.supports(Feature.STRUCTURED_OUTPUT_JSON_SCHEMA):
    result = await runtime.execute(prompt, schema=MySchema)
```

Today only `Feature.STRUCTURED_OUTPUT_JSON_SCHEMA` returns `True` —
every other capability (`STREAMING`, `SESSION_RESUME`,
`REASONING_EFFORT`, `TOOLS_FUNCTION`, `TOOLS_MCP_*`, …) returns
`False` and will flip on in its respective phase. See
[docs/implementation-plan.md](docs/implementation-plan.md) for the
phasing.

Run `uv run python examples/probe_supports.py` for the live
Feature × adapter matrix.

## Escape hatch: `runtime.unwrap()`

When the portable surface doesn't expose a vendor-specific knob,
reach the native SDK object directly via JDBC-`Wrapper`-style
`unwrap()`:

```python
from claude_agent_sdk import ClaudeSDKClient

runtime = ClaudeCodeRuntime()
await runtime.execute("hello", schema=Brief)

# Now reach the underlying SDK for vendor-specific behaviour:
client: ClaudeSDKClient = runtime.unwrap(ClaudeSDKClient)
await client.interrupt()
```

Each adapter accepts `unwrap(type(self))` (returning `self`) plus
its native types: `ClaudeCodeRuntime.unwrap(ClaudeSDKClient)`,
`CopilotRuntime.unwrap(CopilotClient | CopilotSession)`,
`CodexRuntime.unwrap(Codex | Thread)`,
`OpenAICompatibleRuntime.unwrap(AsyncOpenAI)`. Unsupported types
raise `TypeError`.

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
5. Declare `PROVIDER_ID`, `REQUIRES_PACKAGE`, and `EXTRA_NAME` as
   ClassVars; implement `validate_binding`.
6. Implement `list_models()` against the vendor's models endpoint;
   enrich each entry with whatever metadata the adapter knows
   (context window, pricing, capability flags).

**OpenAI-compatible HTTP vendor?** Subclass `OpenAICompatibleRuntime`
instead — the base implements `execute`, `list_models`, `reset`,
`close`, error classification, and envelope-unwrap. The subclass is
~30 lines: ClassVars + `_resolve_api_key()`. See
`src/airframe/adapters/opencode_zen.py`.

**SDK-based vendor (subprocess / native types)?** Inherit
`AgentRuntime` directly and implement all five methods. See
`src/airframe/adapters/claude_code.py` for the canonical example.

## Third-party adapters

Adapters can live in their own package and be discovered via the
`airframe.adapters` entry-point group:

```toml
# pyproject.toml of, say, airframe-adapters-together
[project.entry-points."airframe.adapters"]
together = "airframe_adapters_together:TogetherRuntime"
```

Once installed, `airframe.list_providers()` picks the runtime up
automatically — same pip-extras filtering applies as for built-ins.

To run the shared conformance contracts against a third-party
adapter:

```bash
pip install airframe-agents[testing]
```

Then in the adapter's test suite:

```python
# tests/test_my_adapter_conformance.py
import pytest
from airframe.testing.contracts import (
    test_close_is_idempotent,
    test_close_on_fresh_runtime,
    test_unwrap_returns_self,
    test_unwrap_unrelated_type_raises_typeerror,
    test_supports_returns_bool_for_every_feature,
    test_supports_is_idempotent,
    test_supports_structured_output_json_schema_is_true,
    test_supports_accepts_model_kwarg,
    test_validate_binding_returns_bool,
)
from airframe_adapters_together import TogetherRuntime

@pytest.fixture
def adapter_runtime():
    return TogetherRuntime(api_key="test-key")
```

Pytest collects the imported test functions and runs them against
the local fixture. Modelled on SQLAlchemy's `testing.suite` pattern.
See `tests/test_claude_code_conformance.py` for the canonical
in-tree example.

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
