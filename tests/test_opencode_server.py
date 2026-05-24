"""Unit tests for :class:`OpenCodeServerRuntime` — Iteration A scaffolding.

This file pins the scaffolding-only surface: discovery wiring,
identity, ``validate_binding``, auth chain (loopback no-auth /
explicit / env / non-loopback guardrail), capability predicates (all
flags False), ``unwrap`` shape, and the ``list_models`` happy path +
error classification against a mocked ``AsyncOpencode``.

Behavioural tests (``execute`` / ``stream`` / ``cancel`` /
``session(resume=)``) land in Iteration B's
``tests/test_opencode_server_session.py``.
"""

from __future__ import annotations

import base64
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from airframe.adapters.opencode_server import (
    DEFAULT_BASE_URL,
    DEFAULT_USERNAME,
    OpenCodeServerRuntime,
)
from airframe.errors import (
    RuntimeAuthError,
    RuntimeProtocolError,
    RuntimeServerStartError,
    RuntimeTransientError,
    UnsupportedFeatureError,
)
from airframe.features import Feature
from airframe.options import (
    BedrockOptions,
    ClaudeOptions,
    OpenCodeServerOptions,
)
from airframe.protocol import ProviderModel

# --- identity + defaults ------------------------------------------------------


def test_provider_identity() -> None:
    assert OpenCodeServerRuntime.PROVIDER_ID == "opencode"
    assert OpenCodeServerRuntime.REQUIRES_PACKAGE == "opencode_ai"
    assert OpenCodeServerRuntime.EXTRA_NAME == "opencode"
    assert OpenCodeServerRuntime.label == "opencode_server"


def test_default_base_url_matches_opencode_serve_default() -> None:
    """``opencode serve`` defaults to 127.0.0.1:4096."""
    assert DEFAULT_BASE_URL == "http://127.0.0.1:4096"
    assert DEFAULT_USERNAME == "opencode"


def test_iteration_e_supported_features() -> None:
    """Iteration E adds LIFECYCLE_HOOKS / BUDGET_USD_CAP / BUDGET_TURN_CAP.

    STRUCTURED_OUTPUT_JSON_SCHEMA / TOOLS_FUNCTION / TOOLS_MCP_* /
    PERMISSION_CALLBACK stay False — the 0.1.0a36 SDK doesn't surface
    the matching endpoints yet (see Iteration D module docstring).
    """
    assert set(OpenCodeServerRuntime.SUPPORTED_FEATURES) == {
        Feature.STREAMING,
        Feature.CANCEL,
        Feature.SESSION_RESUME,
        Feature.VISION_INPUT,
        Feature.FILE_INPUT,
        Feature.REASONING_EFFORT,
        Feature.REASONING_BUDGET_TOKENS,
        Feature.LIFECYCLE_HOOKS,
        Feature.BUDGET_USD_CAP,
        Feature.BUDGET_TURN_CAP,
        # Phase 6: filesystem-only discovery is adapter-agnostic.
        Feature.SLASH_COMMANDS,
    }
    rt = OpenCodeServerRuntime()
    assert rt.supports(Feature.LIFECYCLE_HOOKS)
    assert rt.supports(Feature.BUDGET_USD_CAP)
    assert rt.supports(Feature.BUDGET_TURN_CAP)
    # Features still False after E (await SDK):
    assert not rt.supports(Feature.STRUCTURED_OUTPUT_JSON_SCHEMA)
    assert not rt.supports(Feature.TOOLS_FUNCTION)
    assert not rt.supports(Feature.PERMISSION_CALLBACK)


def test_emittable_hook_kinds_after_iteration_e() -> None:
    """Six of airframe's eight kinds are emitted; ``pre_compact`` /
    ``rate_limit`` stay unemittable.
    """
    assert set(OpenCodeServerRuntime.EMITTABLE_HOOK_KINDS) == {
        "session_start",
        "session_end",
        "user_prompt_submit",
        "pre_tool_use",
        "post_tool_use",
        "tool_failure",
    }


# --- validate_binding ---------------------------------------------------------


def test_validate_binding_accepts_canonical_provider_with_any_model_id() -> None:
    """The server fronts whatever upstream is configured — accept any non-empty model_id."""
    rt = OpenCodeServerRuntime()
    assert rt.validate_binding(ProviderModel("opencode", "claude-haiku-4-5"))
    assert rt.validate_binding(ProviderModel("opencode", "gpt-5-codex"))
    assert rt.validate_binding(ProviderModel("opencode", "llama-3.3-70b"))
    assert rt.validate_binding(ProviderModel("opencode", "anything"))


def test_validate_binding_rejects_other_providers() -> None:
    rt = OpenCodeServerRuntime()
    assert not rt.validate_binding(ProviderModel("opencode-zen", "kimi-k2.6"))
    assert not rt.validate_binding(ProviderModel("opencode-go", "glm-5.1"))
    assert not rt.validate_binding(ProviderModel("claude", "claude-haiku-4-5"))
    assert not rt.validate_binding(ProviderModel("openai", "gpt-5"))


def test_validate_binding_rejects_empty_model_id() -> None:
    rt = OpenCodeServerRuntime()
    assert not rt.validate_binding(ProviderModel("opencode", ""))


# --- auth chain ---------------------------------------------------------------


def test_loopback_no_auth_constructs_cleanly() -> None:
    """Default base URL is loopback — no creds needed."""
    rt = OpenCodeServerRuntime()
    # Both halves None when loopback + no creds set anywhere.
    assert rt._username is None
    assert rt._password is None


def test_explicit_credentials_override_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCODE_SERVER_USERNAME", "env-user")
    monkeypatch.setenv("OPENCODE_SERVER_PASSWORD", "env-pass")
    rt = OpenCodeServerRuntime(username="explicit-user", password="explicit-pass")
    assert rt._username == "explicit-user"
    assert rt._password == "explicit-pass"


def test_password_only_env_uses_default_username(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCODE_SERVER_PASSWORD", "env-pass")
    monkeypatch.delenv("OPENCODE_SERVER_USERNAME", raising=False)
    rt = OpenCodeServerRuntime()
    assert rt._username == DEFAULT_USERNAME
    assert rt._password == "env-pass"


def test_non_loopback_without_credentials_raises_runtime_auth_error() -> None:
    """The guardrail prevents an accidental remote-bash endpoint."""
    with pytest.raises(RuntimeAuthError) as exc_info:
        OpenCodeServerRuntime(base_url="http://example.com:4096")
    msg = str(exc_info.value)
    assert "loopback" in msg.lower()
    assert "OPENCODE_SERVER_PASSWORD" in msg


def test_non_loopback_with_credentials_constructs() -> None:
    rt = OpenCodeServerRuntime(
        base_url="https://remote.example.com",
        username="u",
        password="p",
    )
    assert rt._base_url == "https://remote.example.com"
    assert rt._username == "u"
    assert rt._password == "p"


def test_ipv6_loopback_is_treated_as_loopback() -> None:
    rt = OpenCodeServerRuntime(base_url="http://[::1]:4096")
    assert rt._username is None
    assert rt._password is None


def test_localhost_alias_is_treated_as_loopback() -> None:
    rt = OpenCodeServerRuntime(base_url="http://localhost:4096")
    assert rt._username is None
    assert rt._password is None


def test_env_url_resolved_when_constructor_arg_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCODE_SERVER_URL", "http://127.0.0.1:9999")
    rt = OpenCodeServerRuntime()
    assert rt._base_url == "http://127.0.0.1:9999"


def test_explicit_base_url_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCODE_SERVER_URL", "http://127.0.0.1:9999")
    rt = OpenCodeServerRuntime(base_url="http://127.0.0.1:1234")
    assert rt._base_url == "http://127.0.0.1:1234"


# --- _ensure_client (Basic-auth header construction) -------------------------


def test_ensure_client_threads_basic_auth_header_to_async_opencode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When credentials resolve, the SDK gets an Authorization: Basic header."""
    import opencode_ai

    client = MagicMock()
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(opencode_ai, "AsyncOpencode", factory)

    rt = OpenCodeServerRuntime(
        base_url="https://remote.example.com",
        username="alice",
        password="s3cret",
    )
    rt._ensure_client()
    factory.assert_called_once()
    kwargs = factory.call_args.kwargs
    assert kwargs["base_url"] == "https://remote.example.com"
    assert "Authorization" in kwargs["default_headers"]
    expected = base64.b64encode(b"alice:s3cret").decode("ascii")
    assert kwargs["default_headers"]["Authorization"] == f"Basic {expected}"


def test_ensure_client_omits_authorization_when_no_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loopback-no-auth path: no Authorization header at all."""
    import opencode_ai

    client = MagicMock()
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(opencode_ai, "AsyncOpencode", factory)

    rt = OpenCodeServerRuntime()  # loopback default
    rt._ensure_client()
    kwargs = factory.call_args.kwargs
    assert "default_headers" not in kwargs


def test_ensure_client_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeat calls return the same client instance."""
    import opencode_ai

    client = MagicMock()
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(opencode_ai, "AsyncOpencode", factory)

    rt = OpenCodeServerRuntime()
    first = rt._ensure_client()
    second = rt._ensure_client()
    assert first is second
    factory.assert_called_once()


# --- unwrap -------------------------------------------------------------------


def test_unwrap_self_returns_self() -> None:
    rt = OpenCodeServerRuntime()
    assert rt.unwrap(OpenCodeServerRuntime) is rt


def test_unwrap_async_opencode_before_call_raises() -> None:
    """Pre-construction: AsyncOpencode unwrap raises with a clear pointer."""
    from opencode_ai import AsyncOpencode

    rt = OpenCodeServerRuntime()
    with pytest.raises(TypeError, match="no AsyncOpencode client exists yet"):
        rt.unwrap(AsyncOpencode)


def test_unwrap_async_opencode_after_call_returns_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import opencode_ai
    from opencode_ai import AsyncOpencode

    client = MagicMock(spec=AsyncOpencode)
    monkeypatch.setattr(opencode_ai, "AsyncOpencode", MagicMock(return_value=client))

    rt = OpenCodeServerRuntime()
    rt._ensure_client()
    assert rt.unwrap(AsyncOpencode) is client


def test_unwrap_unrelated_type_raises_typeerror() -> None:
    class _NotMine:
        pass

    rt = OpenCodeServerRuntime()
    with pytest.raises(TypeError, match="OpenCodeServerRuntime"):
        rt.unwrap(_NotMine)


# --- session (Iteration A — _ThinAgentSession placeholder) -------------------


def test_session_returns_a_session_for_iteration_a() -> None:
    """Iteration A returns a placeholder; later iterations replace it."""
    rt = OpenCodeServerRuntime()
    sess = rt.session()
    assert sess is not None
    assert sess.id is None  # B fills this in with the server-issued session_id


def test_session_rejects_tools_until_iteration_d() -> None:
    from airframe.tools import FunctionTool

    def _handler(args: Any) -> str:
        return "ok"

    from pydantic import BaseModel

    class _Params(BaseModel):
        pass

    rt = OpenCodeServerRuntime()
    tool = FunctionTool(name="probe", description="d", params=_Params, handler=_handler)
    with pytest.raises(UnsupportedFeatureError) as exc_info:
        rt.session(tools=[tool])
    assert exc_info.value.feature == Feature.TOOLS_FUNCTION


def test_session_rejects_mcp_servers_until_iteration_d() -> None:
    from airframe.tools import McpServerRef

    rt = OpenCodeServerRuntime()
    refs = [McpServerRef(name="probe", transport="stdio", command=["x"])]
    with pytest.raises(UnsupportedFeatureError) as exc_info:
        rt.session(mcp_servers=refs)
    assert exc_info.value.feature == Feature.TOOLS_MCP_STDIO


def test_session_rejects_on_permission_until_iteration_d() -> None:
    async def _cb(_: Any) -> str:
        return "allow"

    rt = OpenCodeServerRuntime()
    with pytest.raises(UnsupportedFeatureError) as exc_info:
        rt.session(on_permission=_cb)
    assert exc_info.value.feature == Feature.PERMISSION_CALLBACK


def test_session_accepts_on_event_after_iteration_e() -> None:
    """``on_event=`` is honoured now that LIFECYCLE_HOOKS is True."""
    rt = OpenCodeServerRuntime()
    events: list[Any] = []
    sess = rt.session(on_event=events.append)
    assert sess is not None
    # The callback isn't fired at session-open time — only on the
    # first turn / close. Sanity: no error.
    assert events == []


def test_session_rejects_foreign_provider_options() -> None:
    rt = OpenCodeServerRuntime()
    with pytest.raises(UnsupportedFeatureError):
        rt.session(provider_options=ClaudeOptions())
    with pytest.raises(UnsupportedFeatureError):
        rt.session(provider_options=BedrockOptions())


def test_session_accepts_own_options_namespace() -> None:
    rt = OpenCodeServerRuntime()
    sess = rt.session(provider_options=OpenCodeServerOptions())
    assert sess is not None


# --- list_models (mocked) -----------------------------------------------------


def _make_providers_payload() -> Any:
    """Build a fake ``AppProvidersResponse``-shaped payload."""
    # Use plain dicts; the adapter's payload-translation helpers
    # handle both Pydantic models and dicts.
    return {
        "default": {"openai": "gpt-5"},
        "providers": [
            {
                "id": "openai",
                "name": "OpenAI",
                "env": ["OPENAI_API_KEY"],
                "models": {
                    "gpt-5": {
                        "id": "gpt-5",
                        "name": "GPT-5",
                        "limit": {"context": 400000, "output": 16000},
                        "cost": {"input": 1.25, "output": 10.0},
                        "reasoning": True,
                        "tool_call": True,
                        "attachment": True,
                    },
                    "gpt-5-nano": {
                        "id": "gpt-5-nano",
                        "name": "GPT-5 Nano",
                        "limit": {"context": 200000, "output": 8000},
                        "cost": {"input": 0.10, "output": 0.40},
                        "reasoning": False,
                        "tool_call": True,
                        "attachment": False,
                    },
                },
            },
            {
                "id": "anthropic",
                "name": "Anthropic",
                "env": ["ANTHROPIC_API_KEY"],
                "models": {
                    "claude-haiku-4-5": {
                        "id": "claude-haiku-4-5",
                        "name": "Claude Haiku 4.5",
                        "limit": {"context": 200000, "output": 8192},
                        "cost": {"input": 1.0, "output": 5.0},
                        "reasoning": False,
                        "tool_call": True,
                        "attachment": True,
                    },
                },
            },
        ],
    }


@pytest.fixture
def mock_async_opencode(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch ``opencode_ai.AsyncOpencode`` with a mock client."""
    import opencode_ai

    client = MagicMock()
    client.app = MagicMock()
    client.app.providers = AsyncMock(return_value=_make_providers_payload())
    client.close = AsyncMock()
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(opencode_ai, "AsyncOpencode", factory)
    return client


@pytest.mark.asyncio
async def test_list_models_flattens_provider_tree(mock_async_opencode: MagicMock) -> None:
    rt = OpenCodeServerRuntime()
    models = await rt.list_models()
    ids = {m.id for m in models}
    assert ids == {"gpt-5", "gpt-5-nano", "claude-haiku-4-5"}
    # Every entry is tagged with this adapter's PROVIDER_ID — not the
    # upstream provider's ID.
    assert all(m.provider_id == "opencode" for m in models)
    # The upstream provider lives in raw["provider"] for callers that need it.
    upstreams = {m.raw["provider"] for m in models}  # type: ignore[index]
    assert upstreams == {"openai", "anthropic"}


@pytest.mark.asyncio
async def test_list_models_normalises_cost_to_per_1k(mock_async_opencode: MagicMock) -> None:
    """OpenCode reports per-million-tokens; airframe expects per-1k."""
    rt = OpenCodeServerRuntime()
    models = await rt.list_models()
    gpt5 = next(m for m in models if m.id == "gpt-5")
    # OpenCode says 1.25 / 10.0 per million → 0.00125 / 0.01 per 1k.
    assert gpt5.pricing_input_per_1k_usd == pytest.approx(0.00125)
    assert gpt5.pricing_output_per_1k_usd == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_list_models_pulls_context_window_from_limit(
    mock_async_opencode: MagicMock,
) -> None:
    rt = OpenCodeServerRuntime()
    models = await rt.list_models()
    gpt5 = next(m for m in models if m.id == "gpt-5")
    assert gpt5.context_window == 400000


@pytest.mark.asyncio
async def test_list_models_capabilities_from_typed_flags(
    mock_async_opencode: MagicMock,
) -> None:
    rt = OpenCodeServerRuntime()
    models = await rt.list_models()
    gpt5 = next(m for m in models if m.id == "gpt-5")
    assert "reasoning" in gpt5.capabilities
    assert "tools" in gpt5.capabilities
    assert "vision" in gpt5.capabilities
    nano = next(m for m in models if m.id == "gpt-5-nano")
    assert "reasoning" not in nano.capabilities
    assert "tools" in nano.capabilities
    assert "vision" not in nano.capabilities


@pytest.mark.asyncio
async def test_list_models_classifies_unreachable_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`opencode serve` not running → RuntimeServerStartError with hint."""
    import opencode_ai
    from opencode_ai import APIConnectionError

    client = MagicMock()
    client.app = MagicMock()
    client.app.providers = AsyncMock(side_effect=APIConnectionError(request=MagicMock()))
    client.close = AsyncMock()
    monkeypatch.setattr(opencode_ai, "AsyncOpencode", MagicMock(return_value=client))

    rt = OpenCodeServerRuntime()
    with pytest.raises(RuntimeServerStartError) as exc_info:
        await rt.list_models()
    assert "opencode serve" in str(exc_info.value)


@pytest.mark.asyncio
async def test_list_models_classifies_401_as_auth_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import opencode_ai
    from opencode_ai import APIStatusError

    err = APIStatusError(
        message="unauthorized",
        response=MagicMock(status_code=401),
        body=None,
    )
    client = MagicMock()
    client.app = MagicMock()
    client.app.providers = AsyncMock(side_effect=err)
    client.close = AsyncMock()
    monkeypatch.setattr(opencode_ai, "AsyncOpencode", MagicMock(return_value=client))

    rt = OpenCodeServerRuntime()
    with pytest.raises(RuntimeAuthError) as exc_info:
        await rt.list_models()
    assert "OPENCODE_SERVER_PASSWORD" in str(exc_info.value)


@pytest.mark.asyncio
async def test_list_models_classifies_429_as_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import opencode_ai
    from opencode_ai import APIStatusError

    err = APIStatusError(
        message="rate-limited",
        response=MagicMock(status_code=429),
        body=None,
    )
    client = MagicMock()
    client.app = MagicMock()
    client.app.providers = AsyncMock(side_effect=err)
    client.close = AsyncMock()
    monkeypatch.setattr(opencode_ai, "AsyncOpencode", MagicMock(return_value=client))

    rt = OpenCodeServerRuntime()
    with pytest.raises(RuntimeTransientError):
        await rt.list_models()


@pytest.mark.asyncio
async def test_list_models_classifies_5xx_as_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import opencode_ai
    from opencode_ai import APIStatusError

    err = APIStatusError(
        message="boom",
        response=MagicMock(status_code=502),
        body=None,
    )
    client = MagicMock()
    client.app = MagicMock()
    client.app.providers = AsyncMock(side_effect=err)
    client.close = AsyncMock()
    monkeypatch.setattr(opencode_ai, "AsyncOpencode", MagicMock(return_value=client))

    rt = OpenCodeServerRuntime()
    with pytest.raises(RuntimeTransientError):
        await rt.list_models()


@pytest.mark.asyncio
async def test_list_models_classifies_4xx_as_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import opencode_ai
    from opencode_ai import APIStatusError

    err = APIStatusError(
        message="bad request",
        response=MagicMock(status_code=400),
        body=None,
    )
    client = MagicMock()
    client.app = MagicMock()
    client.app.providers = AsyncMock(side_effect=err)
    client.close = AsyncMock()
    monkeypatch.setattr(opencode_ai, "AsyncOpencode", MagicMock(return_value=client))

    rt = OpenCodeServerRuntime()
    with pytest.raises(RuntimeProtocolError):
        await rt.list_models()


# --- close idempotency --------------------------------------------------------


@pytest.mark.asyncio
async def test_close_is_idempotent() -> None:
    rt = OpenCodeServerRuntime()
    await rt.close()
    await rt.close()
    await rt.close()


@pytest.mark.asyncio
async def test_close_tears_down_built_client(monkeypatch: pytest.MonkeyPatch) -> None:
    import opencode_ai

    client = MagicMock()
    client.app = MagicMock()
    client.close = AsyncMock()
    monkeypatch.setattr(opencode_ai, "AsyncOpencode", MagicMock(return_value=client))

    rt = OpenCodeServerRuntime()
    rt._ensure_client()
    await rt.close()
    client.close.assert_awaited_once()
    # Second close is a no-op (client already torn down).
    await rt.close()
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_reset_is_a_noop() -> None:
    rt = OpenCodeServerRuntime()
    await rt.reset()
    await rt.reset()  # idempotent


# --- discovery integration ----------------------------------------------------


def test_runtime_for_resolves_opencode() -> None:
    from airframe import list_providers, runtime_for

    assert "opencode" in list_providers(installed_only=False)
    cls = runtime_for("opencode")
    assert cls is OpenCodeServerRuntime


def test_provider_id_namespace_separation() -> None:
    """The bare ``"opencode"`` ID is distinct from the gateway IDs.

    ``OpenCodeServerRuntime.PROVIDER_ID`` deliberately overlaps with
    nothing on ``OpenCodeZenRuntime.PROVIDER_ID`` /
    ``OpenCodeGoRuntime.PROVIDER_ID``. If a future refactor collapses
    or aliases them, this test trips.
    """
    from airframe.adapters.opencode_go import OpenCodeGoRuntime
    from airframe.adapters.opencode_zen import OpenCodeZenRuntime

    ids = {
        OpenCodeServerRuntime.PROVIDER_ID,
        OpenCodeZenRuntime.PROVIDER_ID,
        OpenCodeGoRuntime.PROVIDER_ID,
    }
    assert ids == {"opencode", "opencode-zen", "opencode-go"}
