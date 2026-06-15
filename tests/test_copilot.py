"""Unit tests for :class:`CopilotRuntime` — Phase 1 Iteration G surface.

After Iteration G, ``CopilotRuntime.execute()`` is pure delegation to
``runtime.session().execute() + close()``. The conversational
behaviour (plain-text vs structured, session error classification,
cost computation, capture slots) is exercised by
``tests/test_copilot_session.py`` against the bespoke
:class:`CopilotAgentSession`.

What's left runtime-level — and tested here:

* Binding validation (Copilot bindings pass; Claude bindings rejected
  even with a Copilot provider, per Phase 0 spike finding).
* ``CopilotClient`` construction + auth chain (kwargs wiring:
  explicit token → env → ``use_logged_in_user``).
* ``runtime.execute()`` smoke — delegates correctly to a session and
  returns the result it produced.
* Lifecycle: ``reset()`` is a no-op; ``close()`` releases the
  long-lived ``CopilotClient``.
* Canonical name of the forced ``submit_result`` tool.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from airframe.adapters.copilot import (
    SUBMIT_RESULT_TOOL,
    CopilotRuntime,
)
from airframe.protocol import ProviderModel, UnsupportedBindingError


class _Schema(BaseModel):
    summary: str
    count: int


# ---------------------------------------------------------------------------
# Fake event-data classes for the smoke test
# ---------------------------------------------------------------------------


class _FakeUsage:
    def __init__(
        self,
        *,
        cost: float | None = 0.001,
        input_tokens: float | None = 5,
        output_tokens: float | None = 5,
        cache_read_tokens: float | None = 0,
        cache_write_tokens: float | None = 0,
    ) -> None:
        self.cost = cost
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_tokens = cache_read_tokens
        self.cache_write_tokens = cache_write_tokens


class _FakeAssistantMessage:
    def __init__(self, content: str = "") -> None:
        self.content = content


class _FakeSessionError:
    def __init__(
        self,
        *,
        message: str = "boom",
        error_type: str = "x",
        status_code: int | None = None,
    ) -> None:
        self.message = message
        self.error_type = error_type
        self.status_code = status_code


class _FakeEvent:
    def __init__(self, data: Any) -> None:
        self.data = data


# ---------------------------------------------------------------------------
# Mock fixture: captures subscribed handlers + subprocess kwargs
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sdk(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Mock the ``copilot`` SDK at the boundary.

    The fixture tracks every handler subscribed via ``session.on(handler)``
    so tests can fire events through ``_fire(handlers, event)`` exactly
    as the SDK would after Iteration G's session-based architecture.
    """
    import copilot
    from copilot import session as session_mod
    from copilot.generated import session_events as se_mod

    handlers: list[Any] = []

    def fake_on(handler: Any) -> Any:
        handlers.append(handler)

        def _unsub() -> None:
            if handler in handlers:
                handlers.remove(handler)

        return _unsub

    mock_session = MagicMock()
    mock_session.session_id = "live-sess-id"
    mock_session.send_and_wait = AsyncMock()
    mock_session.destroy = AsyncMock()
    mock_session.abort = AsyncMock()
    mock_session.on = fake_on

    mock_client = MagicMock()
    mock_client.create_session = AsyncMock(return_value=mock_session)
    mock_client.resume_session = AsyncMock(return_value=mock_session)
    mock_client.start = AsyncMock()
    mock_client.stop = AsyncMock()

    # github-copilot-sdk 1.x dropped SubprocessConfig — construction kwargs
    # (github_token / use_logged_in_user / connection) go straight onto
    # CopilotClient(...). Capture them off the factory call.
    captured_client_kwargs: dict[str, Any] = {}

    def fake_client_factory(**kwargs: Any) -> Any:
        captured_client_kwargs.update(kwargs)
        return mock_client

    mock_client_factory = MagicMock(side_effect=fake_client_factory)

    captured_tools: list[dict[str, Any]] = []

    def fake_define_tool(
        name: str,
        *,
        description: str,
        handler: Any,
        params_type: type[BaseModel],
        skip_permission: bool = False,
    ) -> Any:
        tool_obj = MagicMock()
        tool_obj.name = name
        captured_tools.append(
            {
                "name": name,
                "description": description,
                "handler": handler,
                "params_type": params_type,
                "skip_permission": skip_permission,
            }
        )
        return tool_obj

    mock_perm = MagicMock()
    mock_perm.approve_all = lambda req, inv: MagicMock(kind="approve-once")

    monkeypatch.setattr(copilot, "CopilotClient", mock_client_factory)
    monkeypatch.setattr(copilot, "define_tool", fake_define_tool)
    monkeypatch.setattr(session_mod, "PermissionHandler", mock_perm)
    monkeypatch.setattr(se_mod, "AssistantUsageData", _FakeUsage)
    monkeypatch.setattr(se_mod, "AssistantMessageData", _FakeAssistantMessage)
    monkeypatch.setattr(se_mod, "SessionErrorData", _FakeSessionError)

    return {
        "client_factory": mock_client_factory,
        "client": mock_client,
        "session": mock_session,
        "handlers": handlers,
        "client_kwargs": captured_client_kwargs,
        "captured_tools": captured_tools,
    }


def _fire(handlers: list[Any], event: Any) -> None:
    """Invoke every subscribed handler synchronously with one event."""
    for h in handlers:
        h(event)


# ---------------------------------------------------------------------------
# Binding validation
# ---------------------------------------------------------------------------


def test_validate_binding_accepts_canonical_provider() -> None:
    """v0.2.0 dropped the `copilot` / `github` aliases — just `github-copilot`."""
    rt = CopilotRuntime()
    assert rt.validate_binding(ProviderModel("github-copilot", "gpt-4o"))


def test_validate_binding_rejects_aliases_and_others() -> None:
    rt = CopilotRuntime()
    assert not rt.validate_binding(ProviderModel("copilot", "gpt-4o"))
    assert not rt.validate_binding(ProviderModel("github", "o5"))
    assert not rt.validate_binding(ProviderModel("claude", "claude-haiku-4-5"))
    assert not rt.validate_binding(ProviderModel("openai", "gpt-5-mini"))
    assert not rt.validate_binding(ProviderModel("opencode-zen", "gpt-5-nano"))


def test_validate_binding_rejects_claude_models_on_copilot() -> None:
    """Phase 0 finding: Claude served via Copilot can't honour tool calls."""
    rt = CopilotRuntime()
    assert not rt.validate_binding(ProviderModel("github-copilot", "claude-opus-4.7"))


@pytest.mark.asyncio
async def test_execute_rejects_unsupported_binding() -> None:
    rt = CopilotRuntime()
    with pytest.raises(UnsupportedBindingError):
        await rt.execute(
            "hi",
            schema=_Schema,
            model=ProviderModel("anthropic", "claude-haiku-4-5"),
        )


@pytest.mark.asyncio
async def test_execute_rejects_claude_on_copilot() -> None:
    rt = CopilotRuntime()
    with pytest.raises(UnsupportedBindingError):
        await rt.execute(
            "hi",
            schema=_Schema,
            model=ProviderModel("github-copilot", "claude-opus-4.7"),
        )


# ---------------------------------------------------------------------------
# execute() smoke — verify the sugar delegates correctly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_plain_text_smoke(mock_sdk: dict[str, Any]) -> None:
    """``rt.execute(prompt)`` opens a session, runs one turn, closes.

    Iteration G refactored execute() into pure delegation. Behaviour
    is exercised by ``test_copilot_session.py``; this is the smoke test
    that the sugar wire-up returns the session's result through to the
    runtime caller.
    """

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage(content="Done.")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    result = await CopilotRuntime().execute(
        "ask something",
        model=ProviderModel("github-copilot", "gpt-5-mini"),
    )

    assert result.text == "Done."
    assert result.structured is None
    assert result.cost.provider_id == "github-copilot"
    # The session was destroyed after the call (sugar tears it down).
    mock_sdk["session"].destroy.assert_awaited()


@pytest.mark.asyncio
async def test_execute_persona_kwarg_does_not_crash(mock_sdk: dict[str, Any]) -> None:
    """``persona=`` accepted but unused by CopilotRuntime."""

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage(content="ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)
    result = await CopilotRuntime().execute("hi", persona="navigator")
    assert result.text == "ok"


# ---------------------------------------------------------------------------
# Lifecycle — Iteration G: reset() is a no-op; close() releases the client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reset_is_noop_after_iteration_g() -> None:
    """Per-call sessions own their own state; runtime has nothing scope-bound."""
    rt = CopilotRuntime()
    await rt.reset()
    await rt.reset()


@pytest.mark.asyncio
async def test_close_releases_runtime_client(mock_sdk: dict[str, Any]) -> None:
    """close() stops the long-lived CopilotClient even if no session ran."""
    rt = CopilotRuntime()
    # Eagerly build the client via _ensure_client.
    await rt._ensure_client()  # noqa: SLF001
    assert rt._client is not None  # noqa: SLF001
    await rt.close()
    mock_sdk["client"].stop.assert_awaited()
    assert rt._client is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_close_with_no_client_is_noop() -> None:
    rt = CopilotRuntime()
    await rt.close()
    await rt.close()


# ---------------------------------------------------------------------------
# Auth — CopilotClient construction (runtime-owned, long-lived)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explicit_github_token_threaded_to_client(
    mock_sdk: dict[str, Any],
) -> None:
    rt = CopilotRuntime(github_token="ghu_test_token")
    await rt._ensure_client()  # noqa: SLF001 — runtime-level wiring under test
    kwargs = mock_sdk["client_kwargs"]
    assert kwargs.get("github_token") == "ghu_test_token"
    assert "use_logged_in_user" not in kwargs


@pytest.mark.asyncio
async def test_no_token_falls_back_to_logged_in_user(
    mock_sdk: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    rt = CopilotRuntime()
    await rt._ensure_client()  # noqa: SLF001
    kwargs = mock_sdk["client_kwargs"]
    assert kwargs.get("use_logged_in_user") is True
    assert "github_token" not in kwargs


@pytest.mark.asyncio
async def test_env_token_picked_up(
    mock_sdk: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_from_env")
    rt = CopilotRuntime()
    await rt._ensure_client()  # noqa: SLF001
    kwargs = mock_sdk["client_kwargs"]
    assert kwargs.get("github_token") == "ghp_from_env"


# ---------------------------------------------------------------------------
# Canonical naming
# ---------------------------------------------------------------------------


def test_submit_result_tool_name_canonical() -> None:
    """The tool name is stable — referenced by the forced system prefix."""
    assert SUBMIT_RESULT_TOOL == "submit_result"
