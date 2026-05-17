"""Unit tests for :class:`OpenCodeGoRuntime`.

The behavioural surface comes from :class:`OpenAICompatibleRuntime`
(already covered exhaustively by ``test_opencode_zen.py``); this file
focuses on what the Go subclass adds:

* Distinct ``PROVIDER_ID`` and base URL.
* Subscription-tier metadata (cost rates of 0.0 since flat-fee billing).
* Auth chain reads the ``opencode-go`` slot from ``auth.json`` (the
  sibling key from the Zen slot the per-token adapter reads).
* validate_binding only accepts ``opencode-go``.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from airframe.adapters.opencode_go import (
    DEFAULT_GO_BASE_URL,
    DEFAULT_GO_MODEL,
    OpenCodeGoRuntime,
)
from airframe.errors import RuntimeAuthError
from airframe.protocol import ProviderModel, UnsupportedBindingError


def _resolve_api_key(api_key: str | None) -> str:
    return OpenCodeGoRuntime()._resolve_api_key(api_key)


def _compute_cost_usd(model_id: str, *, input_tokens: int, output_tokens: int) -> float | None:
    return OpenCodeGoRuntime()._compute_cost_usd(
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
    response.model_dump = MagicMock(return_value={"id": "resp-123"})
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
    assert OpenCodeGoRuntime.PROVIDER_ID == "opencode-go"
    assert OpenCodeGoRuntime.DEFAULT_BASE_URL == DEFAULT_GO_BASE_URL
    assert DEFAULT_GO_BASE_URL == "https://opencode.ai/zen/go/v1"
    assert DEFAULT_GO_MODEL in OpenCodeGoRuntime.METADATA


def test_metadata_covers_subscription_catalog() -> None:
    # The Go subscription bundles 14 models — keeping the catalog pinned
    # so accidental drift surfaces as a test failure.
    assert len(OpenCodeGoRuntime.METADATA) == 14
    # Every entry is flat-fee billed → zero marginal cost.
    for meta in OpenCodeGoRuntime.METADATA.values():
        assert meta.input_per_1k == 0.0
        assert meta.output_per_1k == 0.0


# --- _resolve_api_key ---------------------------------------------------------


def test_resolve_api_key_explicit() -> None:
    assert _resolve_api_key("sk-explicit") == "sk-explicit"


def test_resolve_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCODE_API_KEY", "sk-env-key")
    monkeypatch.setenv("OPENCODE_AUTH_PATH", "/nonexistent")
    assert _resolve_api_key(None) == "sk-env-key"


def test_resolve_api_key_from_auth_file(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"opencode-go": {"key": "sk-go-key"}}))
    monkeypatch.setenv("OPENCODE_AUTH_PATH", str(auth))
    assert _resolve_api_key(None) == "sk-go-key"


def test_resolve_api_key_does_not_steal_zen_per_token_key(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Go and Zen share an auth file but live under distinct slots."""
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"opencode": {"key": "sk-zen-only"}}))
    monkeypatch.setenv("OPENCODE_AUTH_PATH", str(auth))
    with pytest.raises(RuntimeAuthError):
        _resolve_api_key(None)


def test_resolve_api_key_missing_raises(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    monkeypatch.setenv("OPENCODE_AUTH_PATH", str(tmp_path / "absent.json"))
    with pytest.raises(RuntimeAuthError) as exc_info:
        _resolve_api_key(None)
    # Error message should point users at the right login command.
    assert "opencode auth login opencode-go" in str(exc_info.value)


# --- validate_binding ---------------------------------------------------------


def test_validate_binding_accepts_canonical_provider() -> None:
    rt = OpenCodeGoRuntime()
    assert rt.validate_binding(ProviderModel("opencode-go", "glm-5.1"))


def test_validate_binding_rejects_other_providers() -> None:
    rt = OpenCodeGoRuntime()
    assert not rt.validate_binding(ProviderModel("opencode", "gpt-5-nano"))
    assert not rt.validate_binding(ProviderModel("anthropic", "claude-haiku-4-5"))
    assert not rt.validate_binding(ProviderModel("github-copilot", "gpt-5-mini"))


@pytest.mark.asyncio
async def test_execute_rejects_unsupported_binding() -> None:
    rt = OpenCodeGoRuntime(api_key="sk-test")
    with pytest.raises(UnsupportedBindingError):
        await rt.execute(
            "hi",
            schema=_Schema,
            model=ProviderModel("opencode", "gpt-5-nano"),
        )


# --- _compute_cost_usd --------------------------------------------------------


def test_compute_cost_usd_subscription_model_is_zero() -> None:
    """Subscription models bill flat-fee — per-call cost is $0."""
    cost = _compute_cost_usd("glm-5.1", input_tokens=10_000, output_tokens=5_000)
    assert cost == 0.0


def test_compute_cost_usd_unknown_model_is_none() -> None:
    assert _compute_cost_usd("not-in-table", input_tokens=1, output_tokens=1) is None


# --- execute() ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_returns_typed_payload(mock_client: MagicMock) -> None:
    rt = OpenCodeGoRuntime(api_key="sk-test")
    result = await rt.execute("hi", schema=_Schema)
    assert result.structured == {"summary": "ok", "count": 42}
    assert result.finish == "stop"
    assert result.cost.input_tokens == 100
    assert result.cost.output_tokens == 50
    assert result.cost.provider_id == "opencode-go"
    assert result.cost.model_id == DEFAULT_GO_MODEL
    # Subscription model: $0 per call at the margin.
    assert result.cost.cost_usd == 0.0


@pytest.mark.asyncio
async def test_execute_threads_base_url_to_client(mock_client: MagicMock) -> None:
    import openai

    rt = OpenCodeGoRuntime(api_key="sk-test")
    await rt.execute("hi", schema=_Schema)
    factory: MagicMock = openai.AsyncOpenAI  # type: ignore[assignment]
    call_kwargs = factory.call_args.kwargs
    assert call_kwargs["base_url"] == DEFAULT_GO_BASE_URL


# --- discovery integration ----------------------------------------------------


def test_runtime_for_resolves_opencode_go() -> None:
    from airframe import list_providers, runtime_for

    assert "opencode-go" in list_providers(installed_only=False)
    cls = runtime_for("opencode-go")
    assert cls is OpenCodeGoRuntime
