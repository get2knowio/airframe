# Writing your own adapter

Airframe's adapter surface is intentionally pluggable — third-party
adapters live in their own pip package, register via the
`airframe.adapters` entry-point group, and inherit the shared
conformance contract suite. Same pattern as SQLAlchemy dialects.

Two shapes of adapter:

1. **OpenAI-compatible HTTP** — subclass `OpenAICompatibleRuntime`.
   ~30 lines of code; ships the base's HTTP execute, list_models,
   error classification, envelope unwrap, and the full
   session class. Example: `OpenCodeZenRuntime` in
   `src/airframe/adapters/opencode_zen.py`.
2. **SDK-based (subprocess / native types)** — inherit
   `AgentRuntime` directly, implement the full protocol. Same
   shape as `ClaudeCodeRuntime`, `CopilotRuntime`, `CodexRuntime`.

This guide walks through shape (1) first because it's the most
common case for new vendors.

## Shape 1: OpenAI-compatible vendor

Many vendors speak OpenAI's Chat Completions wire format. Together,
Groq, Fireworks, Anyscale, OpenRouter, vLLM, LM Studio, and
Anthropic's `/v1/messages/openai` proxy all do — adding any of
them as an airframe adapter is a sub-50-line subclass.

```python
# airframe_adapters_together/runtime.py
import os
from typing import ClassVar
from airframe.adapters.openai_compatible import (
    OpenAICompatibleRuntime,
    ModelMeta,
)
from airframe.errors import RuntimeAuthError


_METADATA: dict[str, ModelMeta] = {
    "Qwen/Qwen2.5-72B-Instruct-Turbo": ModelMeta(
        "Qwen 2.5 72B Instruct Turbo", 32_768, 0.0012, 0.0012,
    ),
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": ModelMeta(
        "Llama 3.3 70B Turbo", 131_072, 0.00088, 0.00088,
    ),
    # ...
}


class TogetherRuntime(OpenAICompatibleRuntime):
    """OpenAI-compatible adapter for together.ai."""

    label = "together"

    PROVIDER_ID: ClassVar[str] = "together"
    EXTRA_NAME: ClassVar[str] = "together"
    REQUIRES_PACKAGE: ClassVar[str] = "openai"

    DEFAULT_BASE_URL: ClassVar[str] = "https://api.together.xyz/v1"
    DEFAULT_MODEL: ClassVar[str] = "Qwen/Qwen2.5-72B-Instruct-Turbo"
    METADATA: ClassVar[dict[str, ModelMeta]] = _METADATA

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None) -> None:
        super().__init__(
            api_key=api_key or os.environ.get("TOGETHER_API_KEY"),
            base_url=base_url,
            model=model,
        )

    def _resolve_api_key(self, api_key: str | None) -> str:
        if api_key:
            return api_key
        env = os.environ.get("TOGETHER_API_KEY")
        if env:
            return env
        raise RuntimeAuthError(
            "TogetherRuntime: no API key found. Set TOGETHER_API_KEY "
            "or pass api_key= explicitly."
        )
```

That's it. The base class handles:

- Async HTTP via `openai.AsyncOpenAI`.
- Structured output via `response_format={"type":"json_schema",...}`.
- `list_models()` enriched from `METADATA`.
- The full session class (`OpenAICompatibleSession`) with
  streaming, cancellation, tool loops, lifecycle hooks, budget
  caps, polymorphic prompts.
- Error classification onto `Runtime*Error`.
- Single-key envelope unwrap.

The capability flags inherited from `OpenAICompatibleRuntime` apply
verbatim — see the
[`docs/adapters/opencode-zen.md`](./opencode-zen.md) table for the
defaults. Override `SUPPORTED_FEATURES` on your subclass to flip
extras on/off (e.g. if your vendor supports `STRUCTURED_OUTPUT_STRICT`,
add it).

## Shape 2: SDK-based (full bespoke)

When your vendor ships its own Python SDK (subprocess-based or
otherwise) that doesn't fit OpenAI's wire format, inherit
`AgentRuntime` directly. Implement all the methods on the
protocol. This is more work — typically 500–1500 lines depending
on which features you wire — but you get full control over
session lifecycle, native streaming, vendor-specific tool channels,
etc.

Use the four in-tree adapters as templates:

- **`src/airframe/adapters/claude_code.py`** — full feature wiring;
  the broadest reference.
- **`src/airframe/adapters/copilot.py`** — subprocess + JSON-RPC,
  forced `submit_result` tool for structured output.
- **`src/airframe/adapters/codex.py`** — per-turn subprocess, no
  tool registration channel, session-wide permission policy.

Each adapter file is fully self-contained except for the shared
helpers in `airframe.sessions`.

## Entry-point registration

Once your runtime class exists, expose it via the
`airframe.adapters` entry-point group in your `pyproject.toml`:

```toml
# pyproject.toml of airframe-adapters-together

[project]
name = "airframe-adapters-together"
version = "0.1.0"
dependencies = [
    "airframe-agents>=0.6.0",
    "openai>=1.0",
]

[project.entry-points."airframe.adapters"]
together = "airframe_adapters_together.runtime:TogetherRuntime"
```

The key (`together`) **must match `PROVIDER_ID`**. Once installed,
`airframe.list_providers()` picks the runtime up automatically and
the same `[testing]` pip extras pattern applies.

```python
from airframe import list_providers, runtime_for
print(list_providers())            # ['claude', 'github-copilot', 'together']
cls = runtime_for("together")
rt = cls()
```

## Conformance contracts

`airframe.testing.contracts` ships ~27 structural tests every
adapter must satisfy. They run without credentials — mock the
vendor SDK at the boundary if needed.

```python
# tests/test_my_adapter_conformance.py
import pytest
from airframe.testing.contracts import (
    test_close_is_idempotent,
    test_close_on_fresh_runtime,
    test_emittable_hook_kinds_subset_of_eight_literals,
    test_plain_text_execute_path_is_wired,
    test_session_cancel_when_idle_is_noop,
    test_session_close_is_idempotent,
    test_session_close_on_fresh_session_is_safe,
    test_session_execute_signature_accepts_budget_kwargs,
    test_session_execute_signature_accepts_thinking_kwarg,
    test_session_factory_returns_agent_session,
    test_session_max_budget_usd_declined_when_budget_usd_cap_false,
    test_session_max_turns_declined_when_budget_turn_cap_false,
    test_session_mcp_servers_kwarg_agrees_with_transport_capabilities,
    test_session_on_event_agrees_with_lifecycle_hooks_capability,
    test_session_on_permission_agrees_with_permission_callback_capability,
    test_session_polymorphic_prompt_declined_when_vision_false,
    test_session_rejects_wrong_provider_options_namespace,
    test_session_resume_not_implemented_until_feature_flips,
    test_session_stream_is_async_generator,
    test_session_thinking_kwarg_declined_when_capability_false,
    test_session_tools_kwarg_agrees_with_tools_function_capability,
    test_supports_accepts_model_kwarg,
    test_supports_is_idempotent,
    test_supports_returns_bool_for_every_feature,
    test_supports_structured_output_json_schema_is_true,
    test_unwrap_returns_self,
    test_unwrap_unrelated_type_raises_typeerror,
    test_validate_binding_returns_bool,
)
from airframe_adapters_together import TogetherRuntime


@pytest.fixture
def adapter_runtime() -> TogetherRuntime:
    return TogetherRuntime(api_key="dummy-for-conformance")
```

Pytest collects the imported tests against your fixture. Same
pattern SQLAlchemy uses in `sqlalchemy.testing.suite`.

The contracts check structural properties only — never make live
calls. Each declared-True `Feature` must accept the corresponding
kwarg; each declared-False `Feature` must raise
`UnsupportedFeatureError` with the right `feature=` attribute.

## Integration tests (optional)

For behavioural coverage against your vendor's actual endpoint,
`airframe.testing.integration` provides the same import-into-suite
pattern with `pytest.mark.integration` gating:

```python
# tests/test_my_adapter_integration.py
import pytest
pytest.importorskip("openai")

from airframe.testing.integration import (
    test_integration_budget_usd_cap_trips,
    test_integration_function_tool_round_trip,
    test_integration_hook_observer_receives_events,
    test_integration_list_models,
    test_integration_plain_text_execute,
    test_integration_schema_round_trip,
    test_integration_stream_yields_text_then_turn_complete,
    test_integration_thinking_round_trip,
)
from airframe_adapters_together import TogetherRuntime


@pytest.fixture
async def adapter_runtime() -> TogetherRuntime:
    import os
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        pytest.skip("TOGETHER_API_KEY not set")
    rt = TogetherRuntime(api_key=api_key)
    try:
        yield rt
    finally:
        await rt.close()
```

Run with `pytest -m integration`. Tests self-skip when credentials
are absent so the suite stays usable on partially-configured
machines.

## Installing the testing extras

To run the contract suite, your adapter's dev dependencies need
the `[testing]` extra:

```bash
pip install airframe-agents[testing]
```

This pulls in pytest + pytest-asyncio.

## Provider IDs and reservations

Pick a provider ID that's both clear and unclaimed. The strings
`"anthropic"` and `"openai"` are reserved for future direct-API
adapters (when airframe ships an `AnthropicRuntime` against the
Messages API or `OpenAIRuntime` against the Responses API).

A few existing IDs are taken by the built-in adapters:
`"claude"`, `"github-copilot"`, `"codex"`, `"opencode-zen"`,
`"opencode-go"`. If your
adapter targets one of these vendors with a different transport
or auth chain, choose a discriminating suffix:
`"github-copilot-pro"`, `"opencode-zen-fork"`, etc.

## A `ProviderOptions` namespace?

If your adapter has vendor-specific knobs that don't fit the
portable surface, ship a `<Vendor>Options` frozen+slots dataclass
in your package and accept it via
`session(provider_options=<Vendor>Options(...))`. Call
`airframe.sessions._check_provider_options(provider_options,
expected_type=<Vendor>Options, adapter_label=self.label)` at the
top of `session()` to enforce the tagged-union contract.

The four built-in namespaces (`ClaudeOptions`, `CopilotOptions`,
`CodexOptions`, `OpenAICompatOptions`) are independent — your
namespace doesn't need to inherit from anything. The
`ProviderOptions` type alias is just a union of the in-tree
namespaces; third-party namespaces are accepted by structural
typing.

## Checklist

Before publishing your adapter to PyPI:

- [ ] `PROVIDER_ID`, `REQUIRES_PACKAGE`, `EXTRA_NAME` ClassVars
      declared.
- [ ] `SUPPORTED_FEATURES` ClassVar honestly declares which
      features you wire.
- [ ] `EMITTABLE_HOOK_KINDS` ClassVar (if you declare
      `LIFECYCLE_HOOKS=True`) names the subset of the 8 canonical
      kinds you emit.
- [ ] `validate_binding(binding)` returns False for foreign
      provider IDs (never raises).
- [ ] `close()` is idempotent and never raises (wrap teardown in
      `try/except` + debug log).
- [ ] `reset()` is idempotent.
- [ ] Vendor exceptions are classified into `Runtime*Error` at the
      adapter boundary — never leak past `_classify_exception`.
- [ ] No top-level vendor SDK imports — defer to method bodies or
      `TYPE_CHECKING` blocks so `import airframe` stays light.
- [ ] Conformance suite passes against your adapter with at most
      a `mock_sdk` fixture wherever vendor SDK objects need
      stand-ins.
- [ ] Entry point registered in `pyproject.toml`
      (`[project.entry-points."airframe.adapters"]`).
- [ ] If you ship a `<Vendor>Options` namespace, document its
      fields and add a `_check_provider_options` call to your
      `session()` factory.
- [ ] Per-adapter documentation page in your package's README or
      docs/ — see [adapters/claude.md](./claude.md) for the canonical
      template (auth chain, supported features, options, model IDs,
      structured output mechanism, vendor quirks, native escape
      hatches).

## See also

- [`docs/architecture.md`](../architecture.md) — runtime-vs-session
  split, error hierarchy, conformance philosophy.
- [`docs/capabilities.md`](../capabilities.md) — per-`Feature`
  semantics; informs which flags your adapter declares.
- [`docs/reference.md`](../reference.md) — every public type and
  function with cross-links.
- The four in-tree adapters under
  [`src/airframe/adapters/`](../../src/airframe/adapters/).
