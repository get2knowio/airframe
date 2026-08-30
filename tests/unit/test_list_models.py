"""Unit tests for :meth:`AgentRuntime.list_models` across all adapters.

Each adapter has its own model-discovery path:

* :class:`ClaudeCodeRuntime` — wraps the official ``anthropic`` SDK's
  :meth:`AsyncAnthropic.models.list`. The SDK handles both API-key
  (``x-api-key``) and OAuth-Bearer (``Authorization: Bearer …`` +
  ``anthropic-beta: oauth-2025-04-20``) auth; airframe picks the
  slot based on which credential resolved through the four-step
  chain and lets the SDK send the right headers.
* :class:`CopilotRuntime` — native ``CopilotClient.list_models()``
  (Copilot ships rich metadata: vision, reasoning_effort, context window).
* :class:`OpenCodeZenRuntime` (via :class:`OpenAICompatibleRuntime`
  base) — ``AsyncOpenAI.models.list()`` joined with ``_METADATA``.

These tests mock at the vendor-SDK boundary and assert that:

1. Successful calls return :class:`ModelInfo` entries with correct
   ``provider_id`` and enriched metadata (where the adapter knows it).
2. Auth failures raise :class:`RuntimeAuthError`.
3. Transient HTTP / SDK errors raise :class:`RuntimeTransientError`.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from airframe.adapters.claude_code import ClaudeCodeRuntime
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
# ClaudeCodeRuntime.list_models — wraps anthropic.AsyncAnthropic.models.list
# ---------------------------------------------------------------------------


class _FakeAnthropicModel:
    def __init__(self, *, id: str, display_name: str | None = None) -> None:
        self.id = id
        if display_name is not None:
            self.display_name = display_name


def _patch_anthropic_sdk(
    monkeypatch: pytest.MonkeyPatch,
    *,
    models: list[_FakeAnthropicModel] | None = None,
    raise_exc: Exception | None = None,
) -> dict[str, Any]:
    """Patch ``anthropic.AsyncAnthropic`` to return a fake client.

    Returns the captured-kwargs dict the test can introspect to verify
    which auth slot airframe picked (api_key= vs auth_token=).
    """
    import anthropic

    captured: dict[str, Any] = {}

    mock_models = MagicMock()
    if raise_exc is not None:
        mock_models.list = AsyncMock(side_effect=raise_exc)
    else:
        page = MagicMock()
        page.data = models or []
        mock_models.list = AsyncMock(return_value=page)

    mock_client = MagicMock()
    mock_client.models = mock_models
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    def factory(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return mock_client

    monkeypatch.setattr(anthropic, "AsyncAnthropic", factory)
    return captured


def _anthropic_api_error(status_code: int) -> Exception:
    """Build an ``APIStatusError`` instance airframe should classify."""
    import anthropic
    import httpx

    response = httpx.Response(status_code, request=httpx.Request("GET", "https://example.com"))
    return anthropic.APIStatusError(
        message=f"HTTP {status_code}",
        response=response,
        body=None,
    )


def _anthropic_connection_error() -> Exception:
    import anthropic
    import httpx

    return anthropic.APIConnectionError(request=httpx.Request("GET", "https://example.com"))


@pytest.mark.asyncio
async def test_claude_list_models_uses_api_key_kwarg_when_constructor_arg_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit ``api_key=`` constructor arg → ``api_key=`` on AsyncAnthropic."""
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_CREDENTIALS_PATH", "/nonexistent/path")

    captured = _patch_anthropic_sdk(
        monkeypatch,
        models=[
            _FakeAnthropicModel(id="claude-haiku-4-5", display_name="Claude Haiku 4.5"),
            _FakeAnthropicModel(id="claude-sonnet-4-6", display_name="Claude Sonnet 4.6"),
            _FakeAnthropicModel(id="claude-opus-4-7", display_name="Claude Opus 4.7"),
        ],
    )

    rt = ClaudeCodeRuntime(api_key="sk-ant-test")
    models = await rt.list_models()

    # Auth slot selection: explicit constructor api_key wins.
    assert captured == {"api_key": "sk-ant-test"}

    assert len(models) == 3
    assert all(isinstance(m, ModelInfo) for m in models)
    assert all(m.provider_id == "claude" for m in models)

    haiku = next(m for m in models if m.id == "claude-haiku-4-5")
    assert haiku.display_name == "Claude Haiku 4.5"
    assert haiku.context_window == 200_000
    assert haiku.pricing_input_per_1k_usd == 0.0010
    assert haiku.pricing_output_per_1k_usd == 0.0050
    assert CAPABILITY_TOOLS in haiku.capabilities
    assert CAPABILITY_STRUCTURED_OUTPUT in haiku.capabilities
    assert CAPABILITY_STREAMING in haiku.capabilities
    assert CAPABILITY_VISION in haiku.capabilities


@pytest.mark.asyncio
async def test_claude_list_models_uses_auth_token_kwarg_for_oauth_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``CLAUDE_CODE_OAUTH_TOKEN`` env → ``auth_token=`` on AsyncAnthropic.

    The SDK then sends ``Authorization: Bearer …`` + ``anthropic-beta:
    oauth-2025-04-20`` automatically — the combination that
    ``/v1/models`` actually accepts for subscription tokens.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat-from-env")
    monkeypatch.setenv("CLAUDE_CREDENTIALS_PATH", "/nonexistent/path")

    captured = _patch_anthropic_sdk(monkeypatch, models=[])

    rt = ClaudeCodeRuntime()  # no explicit api_key
    await rt.list_models()

    assert captured == {"auth_token": "sk-ant-oat-from-env"}


@pytest.mark.asyncio
async def test_claude_list_models_defers_to_sdk_when_only_anthropic_api_key_env_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ANTHROPIC_API_KEY`` env (no explicit arg) → SDK self-resolves.

    Airframe passes no auth kwargs in this case; the SDK reads the
    env var itself via its native auth chain.
    """
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
    monkeypatch.setenv("CLAUDE_CREDENTIALS_PATH", "/nonexistent/path")

    captured = _patch_anthropic_sdk(monkeypatch, models=[])

    rt = ClaudeCodeRuntime()
    await rt.list_models()

    # No kwargs to AsyncAnthropic — the SDK reads ANTHROPIC_API_KEY itself.
    assert captured == {}


@pytest.mark.asyncio
async def test_claude_list_models_falls_back_to_credentials_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """``~/.claude/.credentials.json`` is the last-resort OAuth fallback."""
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    cred_path = tmp_path / "credentials.json"
    cred_path.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "sk-ant-oat-from-file",
                    "refreshToken": "refresh-...",
                    "expiresAt": 9_999_999_999,
                    "subscriptionType": "max",
                }
            }
        )
    )
    monkeypatch.setenv("CLAUDE_CREDENTIALS_PATH", str(cred_path))

    captured = _patch_anthropic_sdk(monkeypatch, models=[])

    rt = ClaudeCodeRuntime()
    await rt.list_models()

    assert captured == {"auth_token": "sk-ant-oat-from-file"}


@pytest.mark.asyncio
async def test_claude_list_models_handles_unknown_model_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown IDs come back without enrichment — display_name from SDK."""
    _patch_anthropic_sdk(
        monkeypatch,
        models=[_FakeAnthropicModel(id="claude-mystery-5-0", display_name="Claude Mystery 5.0")],
    )

    rt = ClaudeCodeRuntime(api_key="sk-ant-test")
    models = await rt.list_models()
    assert len(models) == 1
    mystery = models[0]
    assert mystery.id == "claude-mystery-5-0"
    assert mystery.display_name == "Claude Mystery 5.0"
    assert mystery.context_window is None
    assert mystery.pricing_input_per_1k_usd is None


@pytest.mark.asyncio
async def test_claude_list_models_no_credentials_anywhere_raises_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every credential source exhausted → RuntimeAuthError with clear hint."""
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_CREDENTIALS_PATH", "/nonexistent/path")

    rt = ClaudeCodeRuntime()
    with pytest.raises(RuntimeAuthError) as excinfo:
        await rt.list_models()
    msg = str(excinfo.value)
    # The error message should name all the recoverable paths.
    assert "CLAUDE_CODE_OAUTH_TOKEN" in msg
    assert "ANTHROPIC_API_KEY" in msg
    assert "claude setup-token" in msg


@pytest.mark.asyncio
async def test_claude_list_models_401_raises_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_anthropic_sdk(monkeypatch, raise_exc=_anthropic_api_error(401))
    rt = ClaudeCodeRuntime(api_key="sk-ant-bad")
    with pytest.raises(RuntimeAuthError):
        await rt.list_models()


@pytest.mark.asyncio
async def test_claude_list_models_503_raises_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_anthropic_sdk(monkeypatch, raise_exc=_anthropic_api_error(503))
    rt = ClaudeCodeRuntime(api_key="sk-ant-test")
    with pytest.raises(RuntimeTransientError):
        await rt.list_models()


@pytest.mark.asyncio
async def test_claude_list_models_network_error_raises_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_anthropic_sdk(monkeypatch, raise_exc=_anthropic_connection_error())
    rt = ClaudeCodeRuntime(api_key="sk-ant-test")
    with pytest.raises(RuntimeTransientError):
        await rt.list_models()


@pytest.mark.asyncio
async def test_claude_list_models_unexpected_status_raises_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_anthropic_sdk(monkeypatch, raise_exc=_anthropic_api_error(418))
    rt = ClaudeCodeRuntime(api_key="sk-ant-test")
    with pytest.raises(RuntimeProtocolError):
        await rt.list_models()


# ---------------------------------------------------------------------------
# _read_claude_credentials_oauth_token helper — defensive parsing
# ---------------------------------------------------------------------------


def test_credentials_helper_returns_none_for_missing_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from airframe.adapters.claude_code import _read_claude_credentials_oauth_token

    monkeypatch.setenv("CLAUDE_CREDENTIALS_PATH", "/definitely/not/a/path")
    assert _read_claude_credentials_oauth_token() is None


def test_credentials_helper_returns_none_for_malformed_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from airframe.adapters.claude_code import _read_claude_credentials_oauth_token

    bad = tmp_path / "credentials.json"
    bad.write_text("not valid json{{{")
    monkeypatch.setenv("CLAUDE_CREDENTIALS_PATH", str(bad))
    assert _read_claude_credentials_oauth_token() is None


def test_credentials_helper_returns_none_for_missing_oauth_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from airframe.adapters.claude_code import _read_claude_credentials_oauth_token

    p = tmp_path / "credentials.json"
    p.write_text(json.dumps({"otherKey": {"accessToken": "x"}}))
    monkeypatch.setenv("CLAUDE_CREDENTIALS_PATH", str(p))
    assert _read_claude_credentials_oauth_token() is None


def test_credentials_helper_returns_none_for_empty_access_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from airframe.adapters.claude_code import _read_claude_credentials_oauth_token

    p = tmp_path / "credentials.json"
    p.write_text(json.dumps({"claudeAiOauth": {"accessToken": ""}}))
    monkeypatch.setenv("CLAUDE_CREDENTIALS_PATH", str(p))
    assert _read_claude_credentials_oauth_token() is None


def test_credentials_helper_returns_token_when_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from airframe.adapters.claude_code import _read_claude_credentials_oauth_token

    p = tmp_path / "credentials.json"
    p.write_text(json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat-VALID"}}))
    monkeypatch.setenv("CLAUDE_CREDENTIALS_PATH", str(p))
    assert _read_claude_credentials_oauth_token() == "sk-ant-oat-VALID"


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
    mock_client.start = AsyncMock()
    mock_client.stop = AsyncMock()
    monkeypatch.setattr(copilot, "CopilotClient", MagicMock(return_value=mock_client))
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
    mock_client.start = AsyncMock()
    mock_client.stop = AsyncMock()
    monkeypatch.setattr(copilot, "CopilotClient", MagicMock(return_value=mock_client))

    rt = CopilotRuntime(github_token="ghs_test")
    with pytest.raises(RuntimeAuthError):
        await rt.list_models()


# ---------------------------------------------------------------------------
# OpenCodeZenRuntime.list_models — via OpenAICompatibleRuntime base
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
    assert all(m.provider_id == "opencode-zen" for m in models)

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
