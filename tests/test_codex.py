"""Unit tests for :class:`CodexRuntime`.

Mocks :mod:`openai_codex_sdk` at the boundary — no real ``codex`` CLI
subprocess, no real OpenAI calls. Validates:

* Binding validation (Codex bindings pass; non-Codex providers
  rejected; ``claude-*`` model IDs rejected even with an Openai/Codex
  provider).
* Structured-output happy path: JSON Schema passed via
  ``outputSchema``, ``Turn.final_response`` parsed as JSON.
* Empty / non-JSON final response → :class:`RuntimeStructuredOutputError`.
* SDK error classification: auth / install / exec / thread-run.
* Cost record populated from ``Turn.usage``.
* Lifecycle: ``reset()`` drops the thread; ``close()`` drops the client.
* Thread caching by model.
* Auth resolution chain: explicit key → env → opencode auth.json →
  fall-through.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from airframe.adapters.codex import (
    DEFAULT_CODEX_MODEL,
    CodexRuntime,
    _resolve_api_key,
    _strictify_schema,
)
from airframe.errors import (
    AgentRuntimeError,
    RuntimeAuthError,
    RuntimeServerStartError,
    RuntimeStructuredOutputError,
    RuntimeTransientError,
)
from airframe.protocol import ProviderModel, UnsupportedBindingError


class _Schema(BaseModel):
    summary: str
    count: int


class _FakeUsage:
    """Stand-in for ``openai_codex_sdk.Usage``."""

    def __init__(
        self,
        *,
        input_tokens: int = 120,
        output_tokens: int = 240,
        cached_input_tokens: int = 30,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cached_input_tokens = cached_input_tokens


_UNSET = object()


class _FakeTurn:
    """Stand-in for ``openai_codex_sdk.Turn``."""

    def __init__(
        self,
        *,
        final_response: str = '{"summary": "ok", "count": 42}',
        items: list[Any] | None = None,
        usage: Any = _UNSET,
    ) -> None:
        self.final_response = final_response
        self.items = items or []
        # Default to a populated _FakeUsage; pass usage=None to explicitly
        # exercise the cost-without-usage path.
        self.usage = _FakeUsage() if usage is _UNSET else usage


@pytest.fixture
def mock_sdk(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Mock the ``openai_codex_sdk`` symbols ``CodexRuntime`` imports lazily."""
    import openai_codex_sdk as sdk

    mock_thread = MagicMock()
    mock_thread.run = AsyncMock(return_value=_FakeTurn())
    mock_thread.id = None

    mock_client = MagicMock()
    mock_client.start_thread = MagicMock(return_value=mock_thread)

    captured_codex_options: dict[str, Any] = {}

    def fake_codex_factory(options: dict[str, Any] | None = None) -> Any:
        captured_codex_options.update(options or {})
        return mock_client

    monkeypatch.setattr(sdk, "Codex", fake_codex_factory)

    return {
        "client": mock_client,
        "thread": mock_thread,
        "codex_options": captured_codex_options,
    }


# ---------------------------------------------------------------------------
# Binding validation
# ---------------------------------------------------------------------------


def test_validate_binding_accepts_canonical_provider() -> None:
    """v0.2.0 dropped the `openai` alias — just `codex`."""
    rt = CodexRuntime()
    assert rt.validate_binding(ProviderModel("codex", "gpt-5-codex"))
    assert rt.validate_binding(ProviderModel("codex", "o5-codex"))


def test_validate_binding_rejects_aliases_and_others() -> None:
    rt = CodexRuntime()
    # `openai` reserved for a future direct-API OpenAIRuntime.
    assert not rt.validate_binding(ProviderModel("openai", "gpt-5-codex"))
    assert not rt.validate_binding(ProviderModel("claude", "claude-haiku-4-5"))
    assert not rt.validate_binding(ProviderModel("github-copilot", "gpt-5-mini"))
    assert not rt.validate_binding(ProviderModel("opencode-zen", "gpt-5-nano"))


def test_validate_binding_rejects_claude_models_on_codex() -> None:
    rt = CodexRuntime()
    assert not rt.validate_binding(ProviderModel("codex", "claude-sonnet-4.6"))
    assert not rt.validate_binding(ProviderModel("codex", "claude-opus-4.7"))


# ---------------------------------------------------------------------------
# execute() — shape checks before we hit the SDK
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_plain_text_returns_text(mock_sdk: dict[str, Any]) -> None:
    """``schema=None`` honours the protocol's plain-text contract.

    The Codex CLI's ``Turn.final_response`` is the free-form text
    answer when no ``outputSchema`` constraint is applied. The
    adapter must return it on ``RuntimeResult.text`` with
    ``structured=None``.
    """
    rt = CodexRuntime()
    mock_sdk["thread"].run = AsyncMock(return_value=_FakeTurn(final_response="The answer is 42."))

    result = await rt.execute(
        "ask something",
        model=ProviderModel("codex", "gpt-5-codex"),
    )

    assert result.text == "The answer is 42."
    assert result.structured is None
    assert result.cost.input_tokens == 120
    assert result.cost.output_tokens == 240
    assert result.cost.provider_id == "codex"


@pytest.mark.asyncio
async def test_execute_plain_text_omits_output_schema(mock_sdk: dict[str, Any]) -> None:
    """No ``outputSchema`` on TurnOptions when schema is None.

    The Codex CLI sees no JSON-schema constraint and produces
    free-form output. Confirms the plain-text path doesn't leak the
    structured-output constraint.
    """
    rt = CodexRuntime()
    mock_sdk["thread"].run = AsyncMock(return_value=_FakeTurn(final_response="hi"))

    await rt.execute("hello")

    call_args = mock_sdk["thread"].run.call_args
    _prompt, turn_options = call_args.args
    assert "outputSchema" not in turn_options


@pytest.mark.asyncio
async def test_execute_plain_text_forwards_system_via_prompt(
    mock_sdk: dict[str, Any],
) -> None:
    """``system=`` lands prepended to the prompt (Codex SDK shape).

    Codex has no SDK-level system-message slot; the adapter
    concatenates ``system`` and the user prompt with a blank line.
    Same shape as the structured path — text mode must not drop the
    persona.
    """
    rt = CodexRuntime()
    mock_sdk["thread"].run = AsyncMock(return_value=_FakeTurn(final_response="ok"))

    await rt.execute("user prompt", system="You are the navigator.")

    call_args = mock_sdk["thread"].run.call_args
    prompt_arg, _opts = call_args.args
    assert prompt_arg == "You are the navigator.\n\nuser prompt"


@pytest.mark.asyncio
async def test_execute_plain_text_ignores_persona(mock_sdk: dict[str, Any]) -> None:
    """``persona=`` accepted but unused by CodexRuntime."""
    rt = CodexRuntime()
    mock_sdk["thread"].run = AsyncMock(return_value=_FakeTurn(final_response="ok"))

    result = await rt.execute("hi", persona="navigator")
    assert result.text == "ok"


@pytest.mark.asyncio
async def test_execute_plain_text_empty_response_is_not_an_error(
    mock_sdk: dict[str, Any],
) -> None:
    """Empty ``final_response`` is acceptable in plain-text mode.

    Differs from the structured-output path: a tool-only turn that
    writes files and stops without an agent message is a legitimate
    outcome in plain-text mode, not a contract violation.
    """
    rt = CodexRuntime()
    mock_sdk["thread"].run = AsyncMock(return_value=_FakeTurn(final_response=""))

    result = await rt.execute("do the work")
    assert result.text == ""
    assert result.structured is None


@pytest.mark.asyncio
async def test_execute_plain_text_classifies_auth_error(
    mock_sdk: dict[str, Any],
) -> None:
    """Auth-classified Codex errors propagate the same in text mode."""
    from openai_codex_sdk.errors import CodexAuthError

    rt = CodexRuntime()
    mock_sdk["thread"].run = AsyncMock(side_effect=CodexAuthError("no auth.json found"))

    with pytest.raises(RuntimeAuthError):
        await rt.execute("hi")


@pytest.mark.asyncio
async def test_execute_rejects_unsupported_binding() -> None:
    rt = CodexRuntime()
    with pytest.raises(UnsupportedBindingError):
        await rt.execute(
            "hi",
            schema=_Schema,
            model=ProviderModel("anthropic", "claude-haiku-4-5"),
        )


@pytest.mark.asyncio
async def test_execute_rejects_claude_on_codex() -> None:
    rt = CodexRuntime()
    with pytest.raises(UnsupportedBindingError):
        await rt.execute(
            "hi",
            schema=_Schema,
            model=ProviderModel("codex", "claude-sonnet-4.6"),
        )


# ---------------------------------------------------------------------------
# execute() — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_captures_structured_output(mock_sdk: dict[str, Any]) -> None:
    rt = CodexRuntime()

    result = await rt.execute(
        "What is 17 + 25?",
        schema=_Schema,
        model=ProviderModel("codex", "gpt-5-codex"),
    )

    assert result.structured == {"summary": "ok", "count": 42}
    assert result.finish == "stop"
    assert result.text == '{"summary": "ok", "count": 42}'
    assert result.cost.cost_usd is not None  # gpt-5-codex is in the pricing map
    assert result.cost.input_tokens == 120
    assert result.cost.output_tokens == 240
    assert result.cost.cache_read_tokens == 30
    assert result.cost.cache_write_tokens == 0
    assert result.cost.provider_id == "codex"
    assert result.cost.model_id == "gpt-5-codex"


@pytest.mark.asyncio
async def test_execute_passes_output_schema(mock_sdk: dict[str, Any]) -> None:
    rt = CodexRuntime()
    await rt.execute("hi", schema=_Schema)

    call_args = mock_sdk["thread"].run.call_args
    prompt_arg, turn_options = call_args.args
    assert prompt_arg == "hi"
    # outputSchema is the Pydantic schema with additionalProperties:false
    # injected — required by OpenAI Responses-backed Codex endpoints.
    schema_sent = turn_options["outputSchema"]
    assert schema_sent["type"] == "object"
    assert schema_sent["additionalProperties"] is False
    assert schema_sent["properties"] == _Schema.model_json_schema()["properties"]
    assert schema_sent["required"] == _Schema.model_json_schema()["required"]


@pytest.mark.asyncio
async def test_execute_prepends_system_to_prompt(mock_sdk: dict[str, Any]) -> None:
    rt = CodexRuntime()
    await rt.execute("user prompt", schema=_Schema, system="You are the implementer.")

    call_args = mock_sdk["thread"].run.call_args
    prompt_arg, _opts = call_args.args
    assert prompt_arg == "You are the implementer.\n\nuser prompt"


@pytest.mark.asyncio
async def test_execute_uses_default_model_when_unspecified(mock_sdk: dict[str, Any]) -> None:
    rt = CodexRuntime(model="o5-codex")
    result = await rt.execute("hi", schema=_Schema)

    assert result.cost.model_id == "o5-codex"
    thread_opts = mock_sdk["client"].start_thread.call_args.args[0]
    assert thread_opts["model"] == "o5-codex"


@pytest.mark.asyncio
async def test_execute_falls_back_to_default_when_no_model_in_runtime(
    mock_sdk: dict[str, Any],
) -> None:
    rt = CodexRuntime()
    result = await rt.execute("hi", schema=_Schema)

    assert result.cost.model_id == DEFAULT_CODEX_MODEL


@pytest.mark.asyncio
async def test_execute_threads_sandbox_mode_and_skip_git(mock_sdk: dict[str, Any]) -> None:
    rt = CodexRuntime(sandbox_mode="workspace-write", skip_git_repo_check=False)
    await rt.execute("hi", schema=_Schema)

    thread_opts = mock_sdk["client"].start_thread.call_args.args[0]
    assert thread_opts["sandboxMode"] == "workspace-write"
    assert thread_opts["skipGitRepoCheck"] is False


# ---------------------------------------------------------------------------
# execute() — error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_empty_final_response_raises_structured_output_error(
    mock_sdk: dict[str, Any],
) -> None:
    mock_sdk["thread"].run = AsyncMock(return_value=_FakeTurn(final_response=""))
    rt = CodexRuntime()

    with pytest.raises(RuntimeStructuredOutputError):
        await rt.execute("hi", schema=_Schema)


@pytest.mark.asyncio
async def test_execute_non_json_final_response_raises_structured_output_error(
    mock_sdk: dict[str, Any],
) -> None:
    mock_sdk["thread"].run = AsyncMock(
        return_value=_FakeTurn(final_response="here's your answer: maybe 42")
    )
    rt = CodexRuntime()

    with pytest.raises(RuntimeStructuredOutputError):
        await rt.execute("hi", schema=_Schema)


@pytest.mark.asyncio
async def test_execute_thread_run_auth_failure(mock_sdk: dict[str, Any]) -> None:
    from openai_codex_sdk.errors import ThreadRunError

    mock_sdk["thread"].run = AsyncMock(
        side_effect=ThreadRunError("401 unauthorized: missing credentials")
    )
    rt = CodexRuntime()

    with pytest.raises(RuntimeAuthError):
        await rt.execute("hi", schema=_Schema)


@pytest.mark.asyncio
async def test_execute_thread_run_rate_limit(mock_sdk: dict[str, Any]) -> None:
    from openai_codex_sdk.errors import ThreadRunError

    mock_sdk["thread"].run = AsyncMock(side_effect=ThreadRunError("rate limit: 429"))
    rt = CodexRuntime()

    with pytest.raises(RuntimeTransientError):
        await rt.execute("hi", schema=_Schema)


@pytest.mark.asyncio
async def test_execute_thread_run_schema_failure(mock_sdk: dict[str, Any]) -> None:
    from openai_codex_sdk.errors import ThreadRunError

    mock_sdk["thread"].run = AsyncMock(
        side_effect=ThreadRunError("output schema violation: missing required field")
    )
    rt = CodexRuntime()

    with pytest.raises(RuntimeStructuredOutputError):
        await rt.execute("hi", schema=_Schema)


@pytest.mark.asyncio
async def test_execute_thread_run_generic_failure(mock_sdk: dict[str, Any]) -> None:
    from openai_codex_sdk.errors import ThreadRunError

    mock_sdk["thread"].run = AsyncMock(side_effect=ThreadRunError("upstream meltdown"))
    rt = CodexRuntime()

    with pytest.raises(AgentRuntimeError):
        await rt.execute("hi", schema=_Schema)


@pytest.mark.asyncio
async def test_execute_codex_auth_error(mock_sdk: dict[str, Any]) -> None:
    from openai_codex_sdk.errors import CodexAuthError

    mock_sdk["thread"].run = AsyncMock(side_effect=CodexAuthError("no auth.json found"))
    rt = CodexRuntime()

    with pytest.raises(RuntimeAuthError):
        await rt.execute("hi", schema=_Schema)


@pytest.mark.asyncio
async def test_execute_codex_install_error(mock_sdk: dict[str, Any]) -> None:
    from openai_codex_sdk.errors import CodexInstallError

    mock_sdk["thread"].run = AsyncMock(side_effect=CodexInstallError("install missing"))
    rt = CodexRuntime()

    with pytest.raises(RuntimeServerStartError):
        await rt.execute("hi", schema=_Schema)


@pytest.mark.asyncio
async def test_execute_codex_exec_error(mock_sdk: dict[str, Any]) -> None:
    from openai_codex_sdk.errors import CodexExecError

    mock_sdk["thread"].run = AsyncMock(side_effect=CodexExecError("subprocess died"))
    rt = CodexRuntime()

    with pytest.raises(RuntimeTransientError):
        await rt.execute("hi", schema=_Schema)


@pytest.mark.asyncio
async def test_execute_cli_not_found_raises_server_start_error(
    mock_sdk: dict[str, Any],
) -> None:
    mock_sdk["thread"].run = AsyncMock(side_effect=FileNotFoundError("no codex CLI"))
    rt = CodexRuntime()

    with pytest.raises(RuntimeServerStartError):
        await rt.execute("hi", schema=_Schema)


@pytest.mark.asyncio
async def test_execute_timeout_raises_transient_error(mock_sdk: dict[str, Any]) -> None:
    rt = CodexRuntime()

    async def slow(*args: Any, **kwargs: Any) -> Any:
        import asyncio

        await asyncio.sleep(10)

    mock_sdk["thread"].run = slow

    with pytest.raises(RuntimeTransientError):
        await rt.execute("hi", schema=_Schema, timeout=0.05)


@pytest.mark.asyncio
async def test_execute_event_parse_error(mock_sdk: dict[str, Any]) -> None:
    from openai_codex_sdk.errors import EventParseError

    mock_sdk["thread"].run = AsyncMock(side_effect=EventParseError("bad event line"))
    rt = CodexRuntime()

    with pytest.raises(RuntimeStructuredOutputError):
        await rt.execute("hi", schema=_Schema)


@pytest.mark.asyncio
async def test_execute_unknown_exception_classifies_as_agent_error(
    mock_sdk: dict[str, Any],
) -> None:
    mock_sdk["thread"].run = AsyncMock(side_effect=ValueError("totally unexpected"))
    rt = CodexRuntime()

    with pytest.raises(AgentRuntimeError):
        await rt.execute("hi", schema=_Schema)


# ---------------------------------------------------------------------------
# Cost path when usage is missing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cost_record_returns_zero_when_no_usage(mock_sdk: dict[str, Any]) -> None:
    mock_sdk["thread"].run = AsyncMock(return_value=_FakeTurn(usage=None))
    rt = CodexRuntime()

    result = await rt.execute("hi", schema=_Schema)

    assert result.cost.cost_usd is None
    assert result.cost.input_tokens == 0
    assert result.cost.output_tokens == 0
    assert result.cost.provider_id == "codex"


@pytest.mark.asyncio
async def test_cost_usd_none_for_unknown_model(mock_sdk: dict[str, Any]) -> None:
    rt = CodexRuntime(model="some-future-codex-variant")
    result = await rt.execute("hi", schema=_Schema)
    assert result.cost.cost_usd is None
    assert result.cost.input_tokens == 120


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reset_is_noop_after_iteration_g() -> None:
    """Phase 1 Iteration G: the runtime no longer caches a Thread.

    Per-call sessions own that. reset() / close() on a never-used
    runtime are no-ops; close() additionally drops the cached Codex
    client (cheap, no subprocess until thread.run()).
    """
    rt = CodexRuntime()
    await rt.reset()
    await rt.close()


@pytest.mark.asyncio
async def test_close_drops_codex_client(mock_sdk: dict[str, Any]) -> None:
    """close() drops the runtime's cached Codex reference."""
    rt = CodexRuntime()
    rt._ensure_client()  # noqa: SLF001 — populate the runtime-level client
    assert rt._client is not None  # noqa: SLF001
    await rt.close()
    assert rt._client is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_execute_creates_thread_per_call(mock_sdk: dict[str, Any]) -> None:
    """Iteration G: each execute() opens a session → spawns one Thread, closes it.

    Two execute() calls → two start_thread invocations. Consumers
    wanting Thread reuse open a session explicitly and call
    ``session.execute()`` repeatedly.
    """
    rt = CodexRuntime(model="gpt-5-codex")
    await rt.execute("hi", schema=_Schema)
    await rt.execute("hi again", schema=_Schema)
    assert mock_sdk["client"].start_thread.call_count == 2


@pytest.mark.asyncio
async def test_session_reuses_thread_across_turns(mock_sdk: dict[str, Any]) -> None:
    """Session-level thread reuse — covered in detail by test_codex_session.py.

    Smoke check here that the runtime exposes a session that holds
    onto its Thread across multiple turns, distinct from the per-call
    sugar above.
    """
    rt = CodexRuntime(model="gpt-5-codex")
    sess = rt.session()
    try:
        await sess.execute("hi", schema=_Schema)
        await sess.execute("hi again", schema=_Schema)
    finally:
        await sess.close()
    assert mock_sdk["client"].start_thread.call_count == 1


# ---------------------------------------------------------------------------
# Auth resolution
# ---------------------------------------------------------------------------


def test_resolve_api_key_uses_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    assert _resolve_api_key("explicit-key") == "explicit-key"


def test_resolve_api_key_uses_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai-key")
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    assert _resolve_api_key(None) == "env-openai-key"


def test_resolve_api_key_uses_codex_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CODEX_API_KEY", "env-codex-key")
    assert _resolve_api_key(None) == "env-codex-key"


def test_resolve_api_key_uses_opencode_auth_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"openai": {"key": "opencode-stored-key"}}))
    monkeypatch.setenv("OPENCODE_AUTH_PATH", str(auth_file))

    assert _resolve_api_key(None) == "opencode-stored-key"


def test_resolve_api_key_returns_none_when_nothing_resolves(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    monkeypatch.setenv("OPENCODE_AUTH_PATH", str(tmp_path / "nonexistent.json"))

    assert _resolve_api_key(None) is None


def test_strictify_schema_adds_additional_properties_false_to_objects() -> None:
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "integer"},
            "nested": {
                "type": "object",
                "properties": {"deep": {"type": "string"}},
                "required": ["deep"],
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"x": {"type": "number"}},
                    "required": ["x"],
                },
            },
        },
        "required": ["answer", "nested", "items"],
    }
    out = _strictify_schema(schema)
    assert out["additionalProperties"] is False
    assert out["properties"]["nested"]["additionalProperties"] is False
    assert out["properties"]["items"]["items"]["additionalProperties"] is False


def test_strictify_schema_walks_defs_and_anyof() -> None:
    schema = {
        "$defs": {
            "Inner": {"type": "object", "properties": {"y": {"type": "string"}}},
        },
        "type": "object",
        "properties": {
            "field": {
                "anyOf": [
                    {"$ref": "#/$defs/Inner"},
                    {"type": "null"},
                ],
            },
        },
    }
    out = _strictify_schema(schema)
    assert out["additionalProperties"] is False
    assert out["$defs"]["Inner"]["additionalProperties"] is False


def test_strictify_schema_respects_explicit_additional_properties() -> None:
    schema = {
        "type": "object",
        "properties": {"loose": {"type": "object", "additionalProperties": True}},
        "additionalProperties": True,
    }
    out = _strictify_schema(schema)
    assert out["additionalProperties"] is True
    assert out["properties"]["loose"]["additionalProperties"] is True


def test_resolve_api_key_ignores_malformed_auth_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    bad = tmp_path / "auth.json"
    bad.write_text("not valid json {")
    monkeypatch.setenv("OPENCODE_AUTH_PATH", str(bad))

    assert _resolve_api_key(None) is None


@pytest.mark.asyncio
async def test_explicit_api_key_threaded_to_codex_options(
    mock_sdk: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    monkeypatch.setenv("OPENCODE_AUTH_PATH", "/nonexistent/path")
    rt = CodexRuntime(api_key="sk-test-key")

    await rt.execute("hi", schema=_Schema)

    assert mock_sdk["codex_options"].get("apiKey") == "sk-test-key"


@pytest.mark.asyncio
async def test_codex_path_override_threaded_to_codex_options(
    mock_sdk: dict[str, Any],
) -> None:
    rt = CodexRuntime(codex_path="/opt/custom/codex")
    await rt.execute("hi", schema=_Schema)

    assert mock_sdk["codex_options"].get("codexPathOverride") == "/opt/custom/codex"
