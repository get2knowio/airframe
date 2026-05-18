"""Unit tests for :class:`OpenRouterRuntime`.

The behavioural surface comes from :class:`OpenAICompatibleRuntime`
(already covered by ``test_opencode_zen.py``); this file pins what
the OpenRouter subclass adds:

* Distinct ``PROVIDER_ID`` + base URL.
* Auth chain: explicit → ``OPENROUTER_API_KEY`` env. No on-disk
  auth file (unlike the OpenCode adapters).
* validate_binding only accepts ``openrouter``.
* Vendor-prefixed model IDs round-trip through ``ProviderModel``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from airframe.adapters.openrouter import (
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_OPENROUTER_MODEL,
    OpenRouterRuntime,
)
from airframe.errors import RuntimeAuthError
from airframe.protocol import ProviderModel, UnsupportedBindingError


def _resolve_api_key(api_key: str | None) -> str:
    return OpenRouterRuntime()._resolve_api_key(api_key)


def _compute_cost_usd(model_id: str, *, input_tokens: int, output_tokens: int) -> float | None:
    return OpenRouterRuntime()._compute_cost_usd(
        model_id, input_tokens=input_tokens, output_tokens=output_tokens
    )


class _Schema(BaseModel):
    summary: str
    count: int


def _make_response() -> Any:
    msg = MagicMock()
    msg.content = '{"summary": "ok", "count": 42}'
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 50
    usage.prompt_tokens_details = None
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model_dump = MagicMock(return_value={"id": "resp-or-123"})
    return response


@pytest.fixture
def mock_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    import openai

    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_make_response())
    client.close = AsyncMock()

    factory = MagicMock(return_value=client)
    monkeypatch.setattr(openai, "AsyncOpenAI", factory)
    return client


# --- identity + defaults ------------------------------------------------------


def test_provider_identity() -> None:
    assert OpenRouterRuntime.PROVIDER_ID == "openrouter"
    assert OpenRouterRuntime.DEFAULT_BASE_URL == DEFAULT_OPENROUTER_BASE_URL
    assert DEFAULT_OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"
    assert DEFAULT_OPENROUTER_MODEL in OpenRouterRuntime.METADATA


def test_metadata_uses_vendor_prefixed_ids() -> None:
    # OpenRouter routes by `<vendor>/<model>`; the METADATA keys should
    # reflect that so list_models() enrichment matches what callers
    # pass back through ProviderModel.
    for model_id in OpenRouterRuntime.METADATA:
        assert "/" in model_id, f"{model_id!r} missing vendor prefix"


# --- _resolve_api_key ---------------------------------------------------------


def test_resolve_api_key_explicit() -> None:
    assert _resolve_api_key("sk-or-explicit") == "sk-or-explicit"


def test_resolve_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
    assert _resolve_api_key(None) == "sk-or-env"


def test_resolve_api_key_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeAuthError) as exc_info:
        _resolve_api_key(None)
    # Error message should point users at where to mint a key.
    assert "openrouter.ai/keys" in str(exc_info.value)


def test_resolve_api_key_does_not_read_opencode_auth_file(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OpenRouter has no on-disk auth-file convention — even with
    OPENCODE_AUTH_PATH set, OpenRouter must not pick up that key."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # Even with opencode's auth file present, OpenRouter ignores it.
    auth = tmp_path / "auth.json"
    auth.write_text('{"openrouter": {"key": "should-not-be-read"}}')
    monkeypatch.setenv("OPENCODE_AUTH_PATH", str(auth))
    with pytest.raises(RuntimeAuthError):
        _resolve_api_key(None)


# --- validate_binding ---------------------------------------------------------


def test_validate_binding_accepts_canonical_provider() -> None:
    rt = OpenRouterRuntime()
    assert rt.validate_binding(ProviderModel("openrouter", "openai/gpt-4o-mini"))


def test_validate_binding_accepts_vendor_prefixed_models() -> None:
    """OpenRouter model IDs are `<vendor>/<model>` — the slash isn't
    a special character to the adapter, just part of the string."""
    rt = OpenRouterRuntime()
    assert rt.validate_binding(ProviderModel("openrouter", "anthropic/claude-3.5-sonnet"))
    assert rt.validate_binding(ProviderModel("openrouter", "google/gemini-pro-1.5"))
    assert rt.validate_binding(ProviderModel("openrouter", "meta-llama/llama-3.1-70b-instruct"))


def test_validate_binding_rejects_other_providers() -> None:
    rt = OpenRouterRuntime()
    assert not rt.validate_binding(ProviderModel("opencode-zen", "gpt-5-nano"))
    assert not rt.validate_binding(ProviderModel("opencode-go", "glm-5.1"))
    # Naked vendor IDs without the openrouter wrapper get rejected.
    assert not rt.validate_binding(ProviderModel("openai", "gpt-4o"))
    assert not rt.validate_binding(ProviderModel("anthropic", "claude-3.5-sonnet"))


@pytest.mark.asyncio
async def test_execute_rejects_unsupported_binding() -> None:
    rt = OpenRouterRuntime(api_key="sk-or-test")
    with pytest.raises(UnsupportedBindingError):
        await rt.execute(
            "hi",
            schema=_Schema,
            model=ProviderModel("openai", "gpt-4o-mini"),
        )


# --- _compute_cost_usd --------------------------------------------------------


def test_compute_cost_usd_known_model() -> None:
    # openai/gpt-4o-mini: 0.00015 in, 0.0006 out per 1k
    # 1.0 * 0.00015 + 0.5 * 0.0006 = 0.00045
    cost = _compute_cost_usd("openai/gpt-4o-mini", input_tokens=1000, output_tokens=500)
    assert cost is not None
    assert cost == pytest.approx(0.00045, rel=1e-3)


def test_compute_cost_usd_unknown_model_is_none() -> None:
    """Anything not in METADATA — including obscure or new OpenRouter routes —
    returns None rather than a guessed value."""
    assert _compute_cost_usd("nobody/never-shipped", input_tokens=1, output_tokens=1) is None


# --- execute() ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_returns_typed_payload(mock_client: MagicMock) -> None:
    rt = OpenRouterRuntime(api_key="sk-or-test")
    result = await rt.execute("hi", schema=_Schema)
    assert result.structured == {"summary": "ok", "count": 42}
    assert result.finish == "stop"
    assert result.cost.input_tokens == 100
    assert result.cost.output_tokens == 50
    assert result.cost.provider_id == "openrouter"
    assert result.cost.model_id == DEFAULT_OPENROUTER_MODEL


@pytest.mark.asyncio
async def test_execute_threads_base_url_to_client(mock_client: MagicMock) -> None:
    import openai

    rt = OpenRouterRuntime(api_key="sk-or-test")
    await rt.execute("hi", schema=_Schema)
    factory: MagicMock = openai.AsyncOpenAI  # type: ignore[assignment]
    call_kwargs = factory.call_args.kwargs
    assert call_kwargs["base_url"] == DEFAULT_OPENROUTER_BASE_URL


# --- discovery integration ----------------------------------------------------


def test_runtime_for_resolves_openrouter() -> None:
    from airframe import list_providers, runtime_for

    assert "openrouter" in list_providers(installed_only=False)
    cls = runtime_for("openrouter")
    assert cls is OpenRouterRuntime
