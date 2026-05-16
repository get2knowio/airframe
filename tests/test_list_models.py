"""Unit tests for :meth:`AgentRuntime.list_models` across all adapters.

Each adapter has its own model-discovery path:

* :class:`ClaudeCodeRuntime` — hits ``GET https://api.anthropic.com/v1/models``
  directly via :mod:`httpx`, joins against ``_METADATA``.
* :class:`CopilotRuntime` — native ``CopilotClient.list_models()``
  (Copilot ships rich metadata: vision, reasoning_effort, context window).
* :class:`CodexRuntime` — ``AsyncOpenAI.models.list()`` filtered to
  codex-shaped IDs.
* :class:`OpenCodeZenRuntime` (via :class:`OpenAICompatibleRuntime`
  base) — ``AsyncOpenAI.models.list()`` joined with ``_METADATA``.

These tests mock at the vendor-SDK boundary and assert that:

1. Successful calls return :class:`ModelInfo` entries with correct
   ``provider_id`` and enriched metadata (where the adapter knows it).
2. Auth failures raise :class:`RuntimeAuthError`.
3. Transient HTTP / SDK errors raise :class:`RuntimeTransientError`.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from airframe.adapters.claude_code import ClaudeCodeRuntime
from airframe.adapters.codex import CodexRuntime
from airframe.adapters.copilot import CopilotRuntime
from airframe.adapters.opencode_zen import OpenCodeZenRuntime
from airframe.errors import (
    RuntimeAuthError,
    RuntimeProtocolError,
    RuntimeTransientError,
)
from airframe.models import (
    CAPABILITY_REASONING_EFFORT,
    CAPABILITY_STREAMING,
    CAPABILITY_STRUCTURED_OUTPUT,
    CAPABILITY_TOOLS,
    CAPABILITY_VISION,
    ModelInfo,
)

# ---------------------------------------------------------------------------
# ClaudeCodeRuntime.list_models — httpx against /v1/models
# ---------------------------------------------------------------------------


class _FakeClaudeResponse:
    def __init__(self, *, status_code: int, payload: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = ""

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            req = httpx.Request("GET", "https://api.anthropic.com/v1/models")
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=req,
                response=httpx.Response(self.status_code, request=req),
            )

    def json(self) -> dict[str, Any]:
        return self._payload


def _patch_claude_http(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response: _FakeClaudeResponse | None = None,
    raise_exc: Exception | None = None,
) -> dict[str, Any]:
    """Patch ``httpx.AsyncClient`` to return one canned response."""

    captured: dict[str, Any] = {}

    class _FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured["client_kwargs"] = kwargs

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

        async def get(self, url: str, *, headers: dict[str, str]) -> Any:
            captured["url"] = url
            captured["headers"] = headers
            if raise_exc is not None:
                raise raise_exc
            return response

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    return captured


@pytest.mark.asyncio
async def test_claude_list_models_returns_enriched_model_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Known IDs get full metadata from ``_METADATA``."""
    payload = {
        "data": [
            {"id": "claude-haiku-4-5", "display_name": "Claude Haiku 4.5", "type": "model"},
            {"id": "claude-sonnet-4-6", "display_name": "Claude Sonnet 4.6", "type": "model"},
            {"id": "claude-opus-4-7", "display_name": "Claude Opus 4.7", "type": "model"},
        ]
    }
    captured = _patch_claude_http(
        monkeypatch, response=_FakeClaudeResponse(status_code=200, payload=payload)
    )

    rt = ClaudeCodeRuntime(api_key="sk-ant-test")
    models = await rt.list_models()

    assert len(models) == 3
    assert all(isinstance(m, ModelInfo) for m in models)
    assert all(m.provider_id == "claude" for m in models)

    haiku = next(m for m in models if m.id == "claude-haiku-4-5")
    assert haiku.display_name == "Claude Haiku 4.5"
    assert haiku.context_window == 200_000
    assert haiku.pricing_input_per_1k_usd == 0.0010
    assert haiku.pricing_output_per_1k_usd == 0.0050
    # All Claude models declare these capabilities.
    assert CAPABILITY_TOOLS in haiku.capabilities
    assert CAPABILITY_STRUCTURED_OUTPUT in haiku.capabilities
    assert CAPABILITY_STREAMING in haiku.capabilities
    assert CAPABILITY_VISION in haiku.capabilities

    # We sent the x-api-key header (not Authorization Bearer).
    assert captured["headers"]["x-api-key"] == "sk-ant-test"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["url"] == "https://api.anthropic.com/v1/models"


@pytest.mark.asyncio
async def test_claude_list_models_handles_unknown_model_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown IDs come back without enrichment — display_name from the API."""
    payload = {
        "data": [
            {"id": "claude-mystery-5-0", "display_name": "Claude Mystery 5.0", "type": "model"},
        ]
    }
    _patch_claude_http(monkeypatch, response=_FakeClaudeResponse(status_code=200, payload=payload))

    rt = ClaudeCodeRuntime(api_key="sk-ant-test")
    models = await rt.list_models()
    assert len(models) == 1
    mystery = models[0]
    assert mystery.id == "claude-mystery-5-0"
    assert mystery.display_name == "Claude Mystery 5.0"
    assert mystery.context_window is None
    assert mystery.pricing_input_per_1k_usd is None


@pytest.mark.asyncio
async def test_claude_list_models_without_api_key_raises_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OAuth tokens don't work for /v1/models — we surface that cleanly."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    rt = ClaudeCodeRuntime()  # no api_key override
    with pytest.raises(RuntimeAuthError) as excinfo:
        await rt.list_models()
    assert "ANTHROPIC_API_KEY" in str(excinfo.value)


@pytest.mark.asyncio
async def test_claude_list_models_401_raises_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_claude_http(monkeypatch, response=_FakeClaudeResponse(status_code=401))
    rt = ClaudeCodeRuntime(api_key="sk-ant-bad")
    with pytest.raises(RuntimeAuthError):
        await rt.list_models()


@pytest.mark.asyncio
async def test_claude_list_models_503_raises_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_claude_http(monkeypatch, response=_FakeClaudeResponse(status_code=503))
    rt = ClaudeCodeRuntime(api_key="sk-ant-test")
    with pytest.raises(RuntimeTransientError):
        await rt.list_models()


@pytest.mark.asyncio
async def test_claude_list_models_network_error_raises_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_claude_http(monkeypatch, raise_exc=httpx.ConnectError("DNS failure"))
    rt = ClaudeCodeRuntime(api_key="sk-ant-test")
    with pytest.raises(RuntimeTransientError):
        await rt.list_models()


@pytest.mark.asyncio
async def test_claude_list_models_unexpected_status_raises_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_claude_http(monkeypatch, response=_FakeClaudeResponse(status_code=418))
    rt = ClaudeCodeRuntime(api_key="sk-ant-test")
    with pytest.raises(RuntimeProtocolError):
        await rt.list_models()


# ---------------------------------------------------------------------------
# CopilotRuntime.list_models — native SDK
# ---------------------------------------------------------------------------


class _FakeCopilotCaps:
    def __init__(
        self,
        *,
        vision: bool = False,
        reasoning_effort: bool = False,
        max_context_window_tokens: int = 128_000,
    ) -> None:
        self.supports = type(
            "_FakeSupports",
            (),
            {"vision": vision, "reasoning_effort": reasoning_effort},
        )()
        self.limits = type(
            "_FakeLimits",
            (),
            {"max_context_window_tokens": max_context_window_tokens},
        )()


class _FakeCopilotModel:
    def __init__(
        self,
        *,
        id: str,
        name: str,
        capabilities: _FakeCopilotCaps,
    ) -> None:
        self.id = id
        self.name = name
        self.capabilities = capabilities


def _patch_copilot_client(
    monkeypatch: pytest.MonkeyPatch, *, models: list[_FakeCopilotModel]
) -> MagicMock:
    """Patch ``copilot.CopilotClient`` so ``list_models()`` returns fakes."""
    import copilot

    mock_client = MagicMock()
    mock_client.list_models = AsyncMock(return_value=models)
    mock_client.stop = AsyncMock()
    monkeypatch.setattr(copilot, "CopilotClient", MagicMock(return_value=mock_client))
    monkeypatch.setattr(
        copilot, "SubprocessConfig", MagicMock(side_effect=lambda **kwargs: kwargs)
    )
    return mock_client


@pytest.mark.asyncio
async def test_copilot_list_models_returns_enriched_model_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_models = [
        _FakeCopilotModel(
            id="gpt-5-mini",
            name="GPT-5 Mini",
            capabilities=_FakeCopilotCaps(
                vision=False, reasoning_effort=False, max_context_window_tokens=128_000
            ),
        ),
        _FakeCopilotModel(
            id="gpt-5",
            name="GPT-5",
            capabilities=_FakeCopilotCaps(
                vision=True, reasoning_effort=True, max_context_window_tokens=256_000
            ),
        ),
    ]
    _patch_copilot_client(monkeypatch, models=fake_models)

    rt = CopilotRuntime(github_token="ghs_test")
    models = await rt.list_models()

    assert len(models) == 2
    assert all(m.provider_id == "github-copilot" for m in models)

    gpt5 = next(m for m in models if m.id == "gpt-5")
    assert gpt5.display_name == "GPT-5"
    assert gpt5.context_window == 256_000
    # Copilot is subscription-priced; we don't surface per-1K pricing.
    assert gpt5.pricing_input_per_1k_usd is None
    assert gpt5.pricing_output_per_1k_usd is None
    # Capability translation from Copilot's typed flags.
    assert CAPABILITY_VISION in gpt5.capabilities
    assert CAPABILITY_REASONING_EFFORT in gpt5.capabilities
    assert CAPABILITY_TOOLS in gpt5.capabilities
    assert CAPABILITY_STRUCTURED_OUTPUT in gpt5.capabilities
    assert CAPABILITY_STREAMING in gpt5.capabilities

    mini = next(m for m in models if m.id == "gpt-5-mini")
    assert CAPABILITY_VISION not in mini.capabilities
    assert CAPABILITY_REASONING_EFFORT not in mini.capabilities


@pytest.mark.asyncio
async def test_copilot_list_models_empty_when_sdk_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_copilot_client(monkeypatch, models=[])
    rt = CopilotRuntime(github_token="ghs_test")
    assert await rt.list_models() == []


@pytest.mark.asyncio
async def test_copilot_list_models_auth_error_classifies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import copilot

    mock_client = MagicMock()
    mock_client.list_models = AsyncMock(side_effect=Exception("401 unauthorized"))
    mock_client.stop = AsyncMock()
    monkeypatch.setattr(copilot, "CopilotClient", MagicMock(return_value=mock_client))
    monkeypatch.setattr(
        copilot, "SubprocessConfig", MagicMock(side_effect=lambda **kwargs: kwargs)
    )

    rt = CopilotRuntime(github_token="ghs_test")
    with pytest.raises(RuntimeAuthError):
        await rt.list_models()


# ---------------------------------------------------------------------------
# CodexRuntime.list_models — AsyncOpenAI filtered to codex-shaped IDs
# ---------------------------------------------------------------------------


class _FakeOpenAIModel:
    def __init__(self, id: str) -> None:
        self.id = id


class _FakeOpenAIPage:
    def __init__(self, data: list[_FakeOpenAIModel]) -> None:
        self.data = data


def _patch_async_openai(
    monkeypatch: pytest.MonkeyPatch,
    *,
    page: _FakeOpenAIPage | Exception,
    capture: dict[str, Any] | None = None,
) -> None:
    """Patch ``openai.AsyncOpenAI`` so ``.models.list()`` returns a fake page."""
    import openai

    mock_models = MagicMock()
    if isinstance(page, Exception):
        mock_models.list = AsyncMock(side_effect=page)
    else:
        mock_models.list = AsyncMock(return_value=page)

    mock_client = MagicMock()
    mock_client.models = mock_models
    mock_client.close = AsyncMock()

    def factory(**kwargs: Any) -> Any:
        if capture is not None:
            capture.update(kwargs)
        return mock_client

    monkeypatch.setattr(openai, "AsyncOpenAI", factory)


@pytest.mark.asyncio
async def test_codex_list_models_filters_to_codex_shaped_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI /v1/models returns *everything*; codex adapter filters to its tier."""
    fake_page = _FakeOpenAIPage(
        data=[
            _FakeOpenAIModel("gpt-5-codex"),
            _FakeOpenAIModel("gpt-5-codex-mini"),
            _FakeOpenAIModel("o5-codex"),
            # Models we don't surface from the codex adapter:
            _FakeOpenAIModel("gpt-4o"),
            _FakeOpenAIModel("text-embedding-3-small"),
            _FakeOpenAIModel("whisper-1"),
        ]
    )
    _patch_async_openai(monkeypatch, page=fake_page)

    rt = CodexRuntime(api_key="sk-test")
    models = await rt.list_models()

    ids = {m.id for m in models}
    assert ids == {"gpt-5-codex", "gpt-5-codex-mini", "o5-codex"}
    assert all(m.provider_id == "codex" for m in models)

    flagship = next(m for m in models if m.id == "gpt-5-codex")
    assert flagship.display_name == "GPT-5 Codex"
    assert flagship.context_window == 256_000
    assert flagship.pricing_input_per_1k_usd == 0.0015
    assert flagship.pricing_output_per_1k_usd == 0.0060


@pytest.mark.asyncio
async def test_codex_list_models_without_api_key_raises_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    # Steer the opencode-auth.json fallback at a path that doesn't exist.
    monkeypatch.setenv("OPENCODE_AUTH_PATH", "/tmp/airframe-nonexistent-auth.json")
    rt = CodexRuntime()
    with pytest.raises(RuntimeAuthError) as excinfo:
        await rt.list_models()
    assert "OPENAI_API_KEY" in str(excinfo.value)


@pytest.mark.asyncio
async def test_codex_list_models_auth_error_from_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openai import AuthenticationError

    err = AuthenticationError(message="bad key", response=MagicMock(status_code=401), body=None)
    _patch_async_openai(monkeypatch, page=err)

    rt = CodexRuntime(api_key="sk-bad")
    with pytest.raises(RuntimeAuthError):
        await rt.list_models()


# ---------------------------------------------------------------------------
# OpenCodeZenRuntime.list_models — via OpenAICompatibleRuntime base
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_opencode_zen_list_models_returns_enriched_model_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Known Zen model IDs get display name + context window + pricing."""
    fake_page = _FakeOpenAIPage(
        data=[
            _FakeOpenAIModel("gpt-5-nano"),
            _FakeOpenAIModel("gpt-5-mini"),
            _FakeOpenAIModel("minimax-m2.5-free"),
        ]
    )
    capture: dict[str, Any] = {}
    _patch_async_openai(monkeypatch, page=fake_page, capture=capture)

    rt = OpenCodeZenRuntime(api_key="ok-test")
    models = await rt.list_models()

    assert len(models) == 3
    assert all(m.provider_id == "opencode" for m in models)

    nano = next(m for m in models if m.id == "gpt-5-nano")
    assert nano.display_name == "GPT-5 Nano"
    assert nano.context_window == 128_000
    assert nano.pricing_input_per_1k_usd == 0.0001
    assert nano.pricing_output_per_1k_usd == 0.0002

    free = next(m for m in models if m.id == "minimax-m2.5-free")
    assert free.pricing_input_per_1k_usd == 0.0

    # The client was constructed with the right base URL.
    assert capture["base_url"] == "https://opencode.ai/zen/v1"
    assert capture["api_key"] == "ok-test"


@pytest.mark.asyncio
async def test_opencode_zen_list_models_handles_unknown_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new Zen model we haven't catalogued yet comes back unenriched."""
    fake_page = _FakeOpenAIPage(data=[_FakeOpenAIModel("zen-mystery-9000")])
    _patch_async_openai(monkeypatch, page=fake_page)

    rt = OpenCodeZenRuntime(api_key="ok-test")
    models = await rt.list_models()

    assert len(models) == 1
    mystery = models[0]
    assert mystery.id == "zen-mystery-9000"
    assert mystery.display_name == "zen-mystery-9000"  # falls back to id
    assert mystery.context_window is None
    assert mystery.pricing_input_per_1k_usd is None


@pytest.mark.asyncio
async def test_opencode_zen_list_models_without_api_key_raises_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    monkeypatch.setenv("OPENCODE_AUTH_PATH", "/tmp/airframe-nonexistent-auth.json")
    rt = OpenCodeZenRuntime()
    with pytest.raises(RuntimeAuthError):
        await rt.list_models()


@pytest.mark.asyncio
async def test_opencode_zen_list_models_auth_error_from_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openai import AuthenticationError

    err = AuthenticationError(message="bad key", response=MagicMock(status_code=401), body=None)
    _patch_async_openai(monkeypatch, page=err)

    rt = OpenCodeZenRuntime(api_key="ok-bad")
    with pytest.raises(RuntimeAuthError):
        await rt.list_models()
