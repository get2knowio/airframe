"""Behavioural tests for :class:`OpenCodeServerSession` (Iteration B).

Covers ``execute`` / ``stream`` / ``cancel`` / ``close`` /
``session(resume=)`` round-trips against a mocked
``AsyncOpencode`` client. The 0.1.0a36 SDK's actual surface — chat
returns ``AssistantMessage`` metadata; the global event bus
(``client.event.list``) emits ``message.part.updated`` snapshots
the adapter has to delta — is mocked here. Live integration lives
in ``tests/test_opencode_server_integration.py`` (Iteration F).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from airframe.adapters.opencode_server import OpenCodeServerRuntime
from airframe.errors import (
    RuntimeAuthError,
    RuntimeProtocolError,
    UnsupportedFeatureError,
)
from airframe.events import (
    ReasoningDelta,
    TextDelta,
    ToolCallResult,
    ToolCallStart,
    TurnComplete,
)
from airframe.features import Feature
from airframe.options import OpenCodeServerOptions
from airframe.protocol import ProviderModel

# --- Helpers to fabricate Pydantic-shaped SDK objects ------------------------


def _msg(**kw: Any) -> MagicMock:
    """Stand-in for an SDK Pydantic model with attribute access."""
    m = MagicMock()
    for k, v in kw.items():
        setattr(m, k, v)
    return m


def _tokens(input_: int, output: int, *, reasoning: int = 0, cache_read: int = 0) -> MagicMock:
    cache = _msg(read=cache_read, write=0)
    return _msg(input=input_, output=output, reasoning=reasoning, cache=cache)


def _assistant_message(
    *,
    cost: float = 0.0042,
    tokens: MagicMock | None = None,
    model_id: str = "gpt-5-codex",
    error: Any = None,
    summary: str = "",
) -> MagicMock:
    return _msg(
        id="msg-1",
        cost=cost,
        tokens=tokens if tokens is not None else _tokens(100, 50),
        api_model_id=model_id,
        provider_id="openai",
        role="assistant",
        session_id="sess-1",
        error=error,
        summary=summary,
        mode="chat",
    )


def _providers_payload() -> dict[str, Any]:
    """Two upstreams, each hosting one model, for routing lookup."""
    return {
        "default": {"openai": "gpt-5-codex"},
        "providers": [
            {
                "id": "openai",
                "name": "OpenAI",
                "env": ["OPENAI_API_KEY"],
                "models": {"gpt-5-codex": {"id": "gpt-5-codex", "name": "GPT-5 Codex"}},
            },
            {
                "id": "anthropic",
                "name": "Anthropic",
                "env": ["ANTHROPIC_API_KEY"],
                "models": {
                    "claude-haiku-4-5": {"id": "claude-haiku-4-5", "name": "Claude Haiku 4.5"}
                },
            },
        ],
    }


@pytest.fixture
def mock_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch ``opencode_ai.AsyncOpencode`` and return the mock client."""
    import opencode_ai

    client = MagicMock()
    client.session = MagicMock()
    client.session.create = AsyncMock(return_value=_msg(id="sess-1"))
    client.session.chat = AsyncMock(return_value=_assistant_message())
    client.session.delete = AsyncMock()
    client.session.abort = AsyncMock()
    client.app = MagicMock()
    client.app.providers = AsyncMock(return_value=_providers_payload())
    client.event = MagicMock()
    client.close = AsyncMock()
    monkeypatch.setattr(opencode_ai, "AsyncOpencode", MagicMock(return_value=client))
    return client


# --- execute() — happy path + routing ----------------------------------------


@pytest.mark.asyncio
async def test_execute_creates_server_session_when_none(mock_client: MagicMock) -> None:
    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    result = await sess.execute("hi")
    mock_client.session.create.assert_awaited_once()
    mock_client.session.chat.assert_awaited_once()
    assert sess.id == "sess-1"
    assert result.cost.input_tokens == 100
    assert result.cost.output_tokens == 50
    assert result.cost.cost_usd == pytest.approx(0.0042)
    assert result.cost.provider_id == "opencode"
    assert result.cost.model_id == "gpt-5-codex"


@pytest.mark.asyncio
async def test_execute_with_resume_skips_session_create(mock_client: MagicMock) -> None:
    rt = OpenCodeServerRuntime()
    sess = rt.session(
        resume="prior-session-id",
        model=ProviderModel("opencode", "gpt-5-codex"),
    )
    await sess.execute("hi")
    mock_client.session.create.assert_not_awaited()
    mock_client.session.chat.assert_awaited_once()
    # Chat hit the resumed session id.
    args, kwargs = mock_client.session.chat.call_args
    assert args[0] == "prior-session-id"


@pytest.mark.asyncio
async def test_execute_routes_via_explicit_provider_options(mock_client: MagicMock) -> None:
    """``OpenCodeServerOptions(provider_id=)`` short-circuits the catalog lookup."""
    rt = OpenCodeServerRuntime()
    sess = rt.session(
        model=ProviderModel("opencode", "some-private-model"),
        provider_options=OpenCodeServerOptions(provider_id="openrouter"),
    )
    await sess.execute("hi")
    mock_client.app.providers.assert_not_awaited()
    _, kwargs = mock_client.session.chat.call_args
    assert kwargs["provider_id"] == "openrouter"
    assert kwargs["model_id"] == "some-private-model"


@pytest.mark.asyncio
async def test_execute_routes_via_providers_catalog_when_unique(
    mock_client: MagicMock,
) -> None:
    """Without explicit routing, ``app.providers()`` is consulted."""
    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "claude-haiku-4-5"))
    await sess.execute("hi")
    mock_client.app.providers.assert_awaited_once()
    _, kwargs = mock_client.session.chat.call_args
    assert kwargs["provider_id"] == "anthropic"
    assert kwargs["model_id"] == "claude-haiku-4-5"


@pytest.mark.asyncio
async def test_execute_caches_routing_within_session(mock_client: MagicMock) -> None:
    """Once a model→provider mapping is resolved, repeats don't re-query."""
    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    await sess.execute("first")
    await sess.execute("second")
    # providers() hit once; chat hit twice.
    mock_client.app.providers.assert_awaited_once()
    assert mock_client.session.chat.await_count == 2


@pytest.mark.asyncio
async def test_execute_raises_when_no_upstream_hosts_model(
    mock_client: MagicMock,
) -> None:
    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "model-no-one-has"))
    with pytest.raises(UnsupportedFeatureError) as exc_info:
        await sess.execute("hi")
    msg = str(exc_info.value)
    assert "cannot route model" in msg
    assert "OpenCodeServerOptions" in msg


@pytest.mark.asyncio
async def test_execute_raises_when_no_model_resolved(mock_client: MagicMock) -> None:
    rt = OpenCodeServerRuntime()  # no default model env var
    sess = rt.session()  # no binding either
    with pytest.raises(UnsupportedFeatureError) as exc_info:
        await sess.execute("hi")
    assert "no model resolved" in str(exc_info.value)


@pytest.mark.asyncio
async def test_execute_threads_system_prompt(mock_client: MagicMock) -> None:
    rt = OpenCodeServerRuntime()
    sess = rt.session(
        system="You are a precise assistant.",
        model=ProviderModel("opencode", "gpt-5-codex"),
    )
    await sess.execute("hi")
    _, kwargs = mock_client.session.chat.call_args
    assert kwargs["system"] == "You are a precise assistant."


@pytest.mark.asyncio
async def test_execute_parts_carry_user_prompt(mock_client: MagicMock) -> None:
    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    await sess.execute("freight train problem")
    _, kwargs = mock_client.session.chat.call_args
    parts = kwargs["parts"]
    assert len(parts) == 1
    assert parts[0] == {"type": "text", "text": "freight train problem"}


# --- execute() — body text assembled from message parts ----------------------


@pytest.mark.asyncio
async def test_execute_assembles_text_from_message_parts(mock_client: MagicMock) -> None:
    """chat() returns metadata-only AssistantMessage (empty summary); the body
    text is fetched from client.session.messages() and lands on result.text."""
    mock_client.session.messages = AsyncMock(
        return_value=[
            {"info": {"role": "user"}, "parts": [{"type": "text", "text": "ignored"}]},
            {
                "info": {"role": "assistant"},
                "parts": [
                    {"type": "text", "text": "The answer is "},
                    {"type": "text", "text": "42."},
                ],
            },
        ]
    )
    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    result = await sess.execute("What is 17 + 25?")
    assert result.text == "The answer is 42."
    # Cost metadata still comes from the chat() AssistantMessage, unchanged.
    assert result.cost.input_tokens == 100
    mock_client.session.messages.assert_awaited_once_with("sess-1")


@pytest.mark.asyncio
async def test_execute_falls_back_to_summary_when_parts_fetch_fails(
    mock_client: MagicMock,
) -> None:
    """A failing messages() fetch degrades to the AssistantMessage summary
    rather than raising."""
    mock_client.session.chat = AsyncMock(return_value=_assistant_message(summary="fallback text"))
    mock_client.session.messages = AsyncMock(side_effect=RuntimeError("AccessDenied"))
    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    result = await sess.execute("hi")
    assert result.text == "fallback text"


def test_assistant_text_from_messages_picks_last_assistant() -> None:
    from airframe.adapters.opencode_server import _assistant_text_from_messages

    resp = [
        {"info": {"role": "assistant"}, "parts": [{"type": "text", "text": "old"}]},
        {"info": {"role": "user"}, "parts": [{"type": "text", "text": "q"}]},
        {
            "info": {"role": "assistant"},
            "parts": [
                {"type": "reasoning", "text": "thinking"},  # non-text part skipped
                {"type": "text", "text": "new answer"},
            ],
        },
    ]
    assert _assistant_text_from_messages(resp) == "new answer"


def test_assistant_text_from_messages_handles_paginated_and_empty() -> None:
    from airframe.adapters.opencode_server import _assistant_text_from_messages

    # .data wrapper (paginated shape).
    wrapped = _msg(
        data=[{"info": {"role": "assistant"}, "parts": [{"type": "text", "text": "hi"}]}]
    )
    assert _assistant_text_from_messages(wrapped) == "hi"
    # No assistant message → "".
    assert _assistant_text_from_messages([{"info": {"role": "user"}, "parts": []}]) == ""
    # Unexpected shape → "".
    assert _assistant_text_from_messages(None) == ""


# --- execute() — unsupported kwargs raise ------------------------------------


@pytest.mark.asyncio
async def test_execute_schema_raises_until_iteration_d(mock_client: MagicMock) -> None:
    from pydantic import BaseModel

    class _Brief(BaseModel):
        summary: str

    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    with pytest.raises(UnsupportedFeatureError) as exc_info:
        await sess.execute("hi", schema=_Brief)
    assert exc_info.value.feature == Feature.STRUCTURED_OUTPUT_JSON_SCHEMA


# max_turns / max_budget_usd are now honoured; see Iteration E tests below.


# --- execute() — error classification ----------------------------------------


@pytest.mark.asyncio
async def test_execute_classifies_sdk_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import opencode_ai
    from opencode_ai import APIStatusError

    err = APIStatusError(message="nope", response=MagicMock(status_code=401), body=None)
    client = MagicMock()
    client.session = MagicMock()
    client.session.create = AsyncMock(return_value=_msg(id="sess-1"))
    client.session.chat = AsyncMock(side_effect=err)
    client.session.delete = AsyncMock()
    client.app = MagicMock()
    client.app.providers = AsyncMock(return_value=_providers_payload())
    client.event = MagicMock()
    client.close = AsyncMock()
    monkeypatch.setattr(opencode_ai, "AsyncOpencode", MagicMock(return_value=client))

    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    with pytest.raises(RuntimeAuthError):
        await sess.execute("hi")


# --- close() behaviour --------------------------------------------------------


@pytest.mark.asyncio
async def test_close_deletes_owned_session(mock_client: MagicMock) -> None:
    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    await sess.execute("hi")  # populates self.id, sets _owned=True
    await sess.close()
    mock_client.session.delete.assert_awaited_once_with("sess-1")


@pytest.mark.asyncio
async def test_close_does_not_delete_resumed_session(mock_client: MagicMock) -> None:
    rt = OpenCodeServerRuntime()
    sess = rt.session(
        resume="prior-id",
        model=ProviderModel("opencode", "gpt-5-codex"),
    )
    await sess.execute("hi")
    await sess.close()
    mock_client.session.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_close_is_idempotent(mock_client: MagicMock) -> None:
    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    await sess.execute("hi")
    await sess.close()
    await sess.close()
    await sess.close()
    mock_client.session.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_after_close_raises(mock_client: MagicMock) -> None:
    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    await sess.close()
    with pytest.raises(RuntimeError, match="session is closed"):
        await sess.execute("hi")


# --- cancel() -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_is_noop_when_no_session(mock_client: MagicMock) -> None:
    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    # No execute() yet → no session.id → cancel is a no-op.
    await sess.cancel()
    mock_client.session.abort.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_calls_session_abort_when_session_exists(
    mock_client: MagicMock,
) -> None:
    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    await sess.execute("hi")  # populates session id
    await sess.cancel()
    mock_client.session.abort.assert_awaited_once_with("sess-1")


# --- unwrap -------------------------------------------------------------------


def test_session_unwrap_self_returns_self(mock_client: MagicMock) -> None:
    from airframe.adapters.opencode_server import OpenCodeServerSession

    rt = OpenCodeServerRuntime()
    sess = rt.session()
    assert sess.unwrap(OpenCodeServerSession) is sess


def test_session_unwrap_unrelated_raises(mock_client: MagicMock) -> None:
    rt = OpenCodeServerRuntime()
    sess = rt.session()

    class _NotMine:
        pass

    with pytest.raises(TypeError, match="OpenCodeServerSession cannot unwrap"):
        sess.unwrap(_NotMine)


# --- stream() — event translation --------------------------------------------


class _FakeAsyncIter:
    """Minimal async iterator over a fixed event list."""

    def __init__(self, events: list[Any]) -> None:
        self._events = list(events)
        self.close = MagicMock()

    def __aiter__(self) -> _FakeAsyncIter:
        return self

    async def __anext__(self) -> Any:
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


def _event(type_: str, **props: Any) -> MagicMock:
    return _msg(type=type_, properties=_msg(**props))


def _text_part(*, id: str, text: str, session_id: str = "sess-1") -> MagicMock:
    return _msg(id=id, type="text", text=text, session_id=session_id, message_id="msg-1")


@pytest.mark.asyncio
async def test_stream_yields_text_deltas_and_turn_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import opencode_ai

    events = [
        _event(
            "message.part.updated",
            session_id="sess-1",
            part=_text_part(id="p1", text="Hello"),
        ),
        _event(
            "message.part.updated",
            session_id="sess-1",
            part=_text_part(id="p1", text="Hello, world"),
        ),
        _event(
            "message.part.updated",
            session_id="sess-1",
            part=_text_part(id="p1", text="Hello, world!"),
        ),
        _event("session.idle", session_id="sess-1"),
    ]
    fake_stream = _FakeAsyncIter(events)

    client = MagicMock()
    client.session = MagicMock()
    client.session.create = AsyncMock(return_value=_msg(id="sess-1"))
    client.session.chat = AsyncMock(return_value=_assistant_message())
    client.session.delete = AsyncMock()
    client.session.abort = AsyncMock()
    client.app = MagicMock()
    client.app.providers = AsyncMock(return_value=_providers_payload())
    client.event = MagicMock()
    client.event.list = AsyncMock(return_value=fake_stream)
    client.close = AsyncMock()
    monkeypatch.setattr(opencode_ai, "AsyncOpencode", MagicMock(return_value=client))

    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    collected = [evt async for evt in sess.stream("hi")]

    text_deltas = [e for e in collected if isinstance(e, TextDelta)]
    assert [d.text for d in text_deltas] == ["Hello", ", world", "!"]
    # Final event is TurnComplete carrying the AssistantMessage-derived result.
    assert isinstance(collected[-1], TurnComplete)
    assert collected[-1].result.cost.input_tokens == 100


@pytest.mark.asyncio
async def test_stream_filters_events_from_other_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import opencode_ai

    events = [
        # Event from a different session — must be ignored.
        _event(
            "message.part.updated",
            session_id="not-ours",
            part=_text_part(id="p1", text="ghost", session_id="not-ours"),
        ),
        _event(
            "message.part.updated",
            session_id="sess-1",
            part=_text_part(id="p1", text="real"),
        ),
        _event("session.idle", session_id="sess-1"),
    ]
    fake_stream = _FakeAsyncIter(events)

    client = MagicMock()
    client.session = MagicMock()
    client.session.create = AsyncMock(return_value=_msg(id="sess-1"))
    client.session.chat = AsyncMock(return_value=_assistant_message())
    client.session.delete = AsyncMock()
    client.app = MagicMock()
    client.app.providers = AsyncMock(return_value=_providers_payload())
    client.event = MagicMock()
    client.event.list = AsyncMock(return_value=fake_stream)
    client.close = AsyncMock()
    monkeypatch.setattr(opencode_ai, "AsyncOpencode", MagicMock(return_value=client))

    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    collected = [evt async for evt in sess.stream("hi")]
    text_deltas = [e for e in collected if isinstance(e, TextDelta)]
    assert [d.text for d in text_deltas] == ["real"]


@pytest.mark.asyncio
async def test_stream_translates_tool_part_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import opencode_ai

    tool_part_running = _msg(
        id="t1",
        type="tool",
        session_id="sess-1",
        message_id="msg-1",
        tool="bash",
        state=_msg(status="running", input={"cmd": "ls"}, output=None, error=None),
    )
    tool_part_done = _msg(
        id="t1",
        type="tool",
        session_id="sess-1",
        message_id="msg-1",
        tool="bash",
        state=_msg(status="completed", input={"cmd": "ls"}, output="file1\nfile2", error=None),
    )
    events = [
        _event("message.part.updated", session_id="sess-1", part=tool_part_running),
        _event("message.part.updated", session_id="sess-1", part=tool_part_done),
        _event("session.idle", session_id="sess-1"),
    ]
    fake_stream = _FakeAsyncIter(events)

    client = MagicMock()
    client.session = MagicMock()
    client.session.create = AsyncMock(return_value=_msg(id="sess-1"))
    client.session.chat = AsyncMock(return_value=_assistant_message())
    client.session.delete = AsyncMock()
    client.app = MagicMock()
    client.app.providers = AsyncMock(return_value=_providers_payload())
    client.event = MagicMock()
    client.event.list = AsyncMock(return_value=fake_stream)
    client.close = AsyncMock()
    monkeypatch.setattr(opencode_ai, "AsyncOpencode", MagicMock(return_value=client))

    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    collected = [evt async for evt in sess.stream("hi")]

    starts = [e for e in collected if isinstance(e, ToolCallStart)]
    results = [e for e in collected if isinstance(e, ToolCallResult)]
    assert len(starts) == 1
    assert starts[0].tool_name == "bash"
    assert starts[0].tool_call_id == "t1"
    assert len(results) == 1
    assert results[0].tool_call_id == "t1"
    assert results[0].is_error is False
    assert results[0].output == "file1\nfile2"


@pytest.mark.asyncio
async def test_stream_translates_reasoning_part(monkeypatch: pytest.MonkeyPatch) -> None:
    import opencode_ai

    reasoning_a = _msg(
        id="r1", type="reasoning", text="Thinking", session_id="sess-1", message_id="msg-1"
    )
    reasoning_b = _msg(
        id="r1",
        type="reasoning",
        text="Thinking about the answer",
        session_id="sess-1",
        message_id="msg-1",
    )
    events = [
        _event("message.part.updated", session_id="sess-1", part=reasoning_a),
        _event("message.part.updated", session_id="sess-1", part=reasoning_b),
        _event("session.idle", session_id="sess-1"),
    ]
    fake_stream = _FakeAsyncIter(events)

    client = MagicMock()
    client.session = MagicMock()
    client.session.create = AsyncMock(return_value=_msg(id="sess-1"))
    client.session.chat = AsyncMock(return_value=_assistant_message())
    client.session.delete = AsyncMock()
    client.app = MagicMock()
    client.app.providers = AsyncMock(return_value=_providers_payload())
    client.event = MagicMock()
    client.event.list = AsyncMock(return_value=fake_stream)
    client.close = AsyncMock()
    monkeypatch.setattr(opencode_ai, "AsyncOpencode", MagicMock(return_value=client))

    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    collected = [evt async for evt in sess.stream("hi")]
    reasonings = [e for e in collected if isinstance(e, ReasoningDelta)]
    assert [r.text for r in reasonings] == ["Thinking", " about the answer"]


@pytest.mark.asyncio
async def test_stream_raises_on_session_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import opencode_ai

    error_evt = _event(
        "session.error",
        session_id="sess-1",
        error=_msg(name="ProviderAuthError", message="bad key"),
    )
    fake_stream = _FakeAsyncIter([error_evt])

    client = MagicMock()
    client.session = MagicMock()
    client.session.create = AsyncMock(return_value=_msg(id="sess-1"))
    client.session.chat = AsyncMock(return_value=_assistant_message())
    client.session.delete = AsyncMock()
    client.app = MagicMock()
    client.app.providers = AsyncMock(return_value=_providers_payload())
    client.event = MagicMock()
    client.event.list = AsyncMock(return_value=fake_stream)
    client.close = AsyncMock()
    monkeypatch.setattr(opencode_ai, "AsyncOpencode", MagicMock(return_value=client))

    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    with pytest.raises(RuntimeAuthError):
        async for _evt in sess.stream("hi"):
            pass


# --- _ensure_session_id error paths -----------------------------------------


@pytest.mark.asyncio
async def test_execute_raises_when_session_create_returns_no_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import opencode_ai

    client = MagicMock()
    client.session = MagicMock()
    # SDK returns a Session-shaped object missing ``id``.
    client.session.create = AsyncMock(return_value=_msg())
    client.session.chat = AsyncMock(return_value=_assistant_message())
    client.session.delete = AsyncMock()
    client.app = MagicMock()
    client.app.providers = AsyncMock(return_value=_providers_payload())
    client.event = MagicMock()
    client.close = AsyncMock()
    monkeypatch.setattr(opencode_ai, "AsyncOpencode", MagicMock(return_value=client))

    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    with pytest.raises(RuntimeProtocolError, match="without an id"):
        await sess.execute("hi")


# --- Iteration C — polymorphic prompts ---------------------------------------


@pytest.mark.asyncio
async def test_execute_image_url_passes_through(mock_client: MagicMock) -> None:
    from airframe.inputs import ImageInput

    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    await sess.execute(
        [
            "Describe this:",
            ImageInput(url="https://example.com/cat.png", media_type="image/png"),
        ]
    )
    _, kwargs = mock_client.session.chat.call_args
    parts = kwargs["parts"]
    assert parts[0] == {"type": "text", "text": "Describe this:"}
    assert parts[1] == {
        "type": "file",
        "mime": "image/png",
        "url": "https://example.com/cat.png",
    }


@pytest.mark.asyncio
async def test_execute_image_bytes_encodes_as_data_url(mock_client: MagicMock) -> None:
    import base64 as _b64

    from airframe.inputs import ImageInput

    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    raw = b"\x89PNG\r\n\x1a\n-pretend-image"
    await sess.execute(["Look:", ImageInput(bytes_=raw, media_type="image/png")])
    _, kwargs = mock_client.session.chat.call_args
    parts = kwargs["parts"]
    expected = "data:image/png;base64," + _b64.b64encode(raw).decode("ascii")
    assert parts[1] == {"type": "file", "mime": "image/png", "url": expected}


@pytest.mark.asyncio
async def test_execute_image_path_reads_and_encodes(mock_client: MagicMock, tmp_path: Any) -> None:
    import base64 as _b64

    from airframe.inputs import ImageInput

    img_path = tmp_path / "photo.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n-bytes-on-disk")

    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    await sess.execute(["Photo:", ImageInput(path=str(img_path))])
    _, kwargs = mock_client.session.chat.call_args
    parts = kwargs["parts"]
    assert parts[1]["type"] == "file"
    assert parts[1]["mime"] == "image/png"
    assert parts[1]["filename"] == "photo.png"
    expected = "data:image/png;base64," + _b64.b64encode(
        b"\x89PNG\r\n\x1a\n-bytes-on-disk"
    ).decode("ascii")
    assert parts[1]["url"] == expected


@pytest.mark.asyncio
async def test_execute_file_path_reads_and_encodes(mock_client: MagicMock, tmp_path: Any) -> None:
    import base64 as _b64

    from airframe.inputs import FileInput

    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n-not-a-real-pdf")

    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    await sess.execute(["Summarise:", FileInput(path=str(pdf_path))])
    _, kwargs = mock_client.session.chat.call_args
    parts = kwargs["parts"]
    assert parts[1]["type"] == "file"
    assert parts[1]["mime"] == "application/pdf"
    assert parts[1]["filename"] == "doc.pdf"
    expected = "data:application/pdf;base64," + _b64.b64encode(
        b"%PDF-1.4\n-not-a-real-pdf"
    ).decode("ascii")
    assert parts[1]["url"] == expected


@pytest.mark.asyncio
async def test_execute_multiple_attachments_preserve_order(
    mock_client: MagicMock,
) -> None:
    from airframe.inputs import FileInput, ImageInput

    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    await sess.execute(
        [
            "First part",
            ImageInput(url="https://example.com/a.jpg", media_type="image/jpeg"),
            "Second part",
            ImageInput(url="https://example.com/b.png", media_type="image/png"),
        ]
    )
    _, kwargs = mock_client.session.chat.call_args
    parts = kwargs["parts"]
    # Text parts collapse into a single ``\n\n``-joined text element;
    # both images follow as separate file parts (in input order).
    assert parts[0] == {"type": "text", "text": "First part\n\nSecond part"}
    assert parts[1]["url"] == "https://example.com/a.jpg"
    assert parts[2]["url"] == "https://example.com/b.png"
    # Sanity: no FileInput was involved.
    assert all(p.get("mime", "").startswith("image/") for p in parts[1:])
    _ = FileInput  # silence unused-import lint in this test


# --- Iteration C — reasoning pass-through ------------------------------------


@pytest.mark.asyncio
async def test_thinking_effort_routes_through_openai_shape_for_non_anthropic(
    mock_client: MagicMock,
) -> None:
    """OpenAI / OpenRouter / Ollama etc. get the ``reasoning_effort`` envelope."""
    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    await sess.execute("solve a hard problem", thinking="high")
    _, kwargs = mock_client.session.chat.call_args
    assert kwargs["extra_body"] == {"reasoning_effort": "high"}


@pytest.mark.asyncio
async def test_thinking_effort_routes_through_anthropic_thinking_envelope(
    mock_client: MagicMock,
) -> None:
    """Anthropic upstream gets the ``thinking={'budget_tokens': ...}`` shape."""
    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "claude-haiku-4-5"))
    await sess.execute("solve", thinking="medium")
    _, kwargs = mock_client.session.chat.call_args
    assert kwargs["extra_body"] == {"thinking": {"type": "enabled", "budget_tokens": 8192}}


@pytest.mark.asyncio
async def test_thinking_budget_tokens_to_anthropic(mock_client: MagicMock) -> None:
    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "claude-haiku-4-5"))
    await sess.execute("solve", thinking={"budget_tokens": 32_000})
    _, kwargs = mock_client.session.chat.call_args
    assert kwargs["extra_body"] == {"thinking": {"type": "enabled", "budget_tokens": 32_000}}


@pytest.mark.asyncio
async def test_thinking_budget_tokens_on_non_anthropic_raises(
    mock_client: MagicMock,
) -> None:
    """Anthropic-shape dict on a non-Anthropic upstream raises rather than silently dropping."""
    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    with pytest.raises(UnsupportedFeatureError) as exc_info:
        await sess.execute("solve", thinking={"budget_tokens": 4096})
    assert exc_info.value.feature == Feature.REASONING_BUDGET_TOKENS


@pytest.mark.asyncio
async def test_thinking_none_sends_no_envelope(mock_client: MagicMock) -> None:
    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    await sess.execute("hi", thinking=None)
    _, kwargs = mock_client.session.chat.call_args
    assert "extra_body" not in kwargs


@pytest.mark.asyncio
async def test_thinking_disabled_sends_no_envelope(mock_client: MagicMock) -> None:
    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    await sess.execute("hi", thinking="disabled")
    _, kwargs = mock_client.session.chat.call_args
    assert "extra_body" not in kwargs


@pytest.mark.asyncio
async def test_thinking_invalid_dict_raises(mock_client: MagicMock) -> None:
    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "claude-haiku-4-5"))
    with pytest.raises(UnsupportedFeatureError):
        await sess.execute("hi", thinking={"budget_tokens": -1})
    with pytest.raises(UnsupportedFeatureError):
        await sess.execute("hi", thinking={"budget_tokens": "lots"})  # type: ignore[arg-type]


# --- Iteration D — built-in tool allow/denylist ------------------------------


@pytest.mark.asyncio
async def test_available_tools_threads_allowlist(mock_client: MagicMock) -> None:
    rt = OpenCodeServerRuntime()
    sess = rt.session(
        model=ProviderModel("opencode", "gpt-5-codex"),
        provider_options=OpenCodeServerOptions(available_tools=("bash", "read")),
    )
    await sess.execute("hi")
    _, kwargs = mock_client.session.chat.call_args
    assert kwargs["tools"] == {"bash": True, "read": True}


@pytest.mark.asyncio
async def test_excluded_tools_threads_denylist(mock_client: MagicMock) -> None:
    rt = OpenCodeServerRuntime()
    sess = rt.session(
        model=ProviderModel("opencode", "gpt-5-codex"),
        provider_options=OpenCodeServerOptions(excluded_tools=("write", "edit")),
    )
    await sess.execute("hi")
    _, kwargs = mock_client.session.chat.call_args
    assert kwargs["tools"] == {"write": False, "edit": False}


@pytest.mark.asyncio
async def test_allowlist_and_denylist_merge_with_deny_winning(
    mock_client: MagicMock,
) -> None:
    """Denying a tool already on the allowlist resolves to False — explicit deny wins."""
    rt = OpenCodeServerRuntime()
    sess = rt.session(
        model=ProviderModel("opencode", "gpt-5-codex"),
        provider_options=OpenCodeServerOptions(
            available_tools=("bash", "read", "write"),
            excluded_tools=("write",),
        ),
    )
    await sess.execute("hi")
    _, kwargs = mock_client.session.chat.call_args
    assert kwargs["tools"] == {"bash": True, "read": True, "write": False}


@pytest.mark.asyncio
async def test_no_tool_filter_sends_no_tools_kwarg(mock_client: MagicMock) -> None:
    """Default options → no ``tools=`` threaded; server defaults apply."""
    rt = OpenCodeServerRuntime()
    sess = rt.session(
        model=ProviderModel("opencode", "gpt-5-codex"),
        provider_options=OpenCodeServerOptions(),
    )
    await sess.execute("hi")
    _, kwargs = mock_client.session.chat.call_args
    assert "tools" not in kwargs


# --- Iteration D — declines cite SDK constraint ------------------------------


def test_session_tools_decline_cites_sdk_gap(mock_client: MagicMock) -> None:
    """Declining FunctionTool wiring points at the SDK constraint, not "later iteration"."""
    from pydantic import BaseModel

    from airframe.tools import FunctionTool

    class _Params(BaseModel):
        pass

    async def _handler(_: BaseModel) -> str:
        return "ok"

    rt = OpenCodeServerRuntime()
    tool = FunctionTool(name="probe", description="d", params=_Params, handler=_handler)
    with pytest.raises(UnsupportedFeatureError) as exc_info:
        rt.session(tools=[tool])
    msg = str(exc_info.value)
    assert "opencode-ai 0.1.0a36" in msg
    assert "MCP-runtime-registration" in msg
    assert "OpenCodeServerOptions" in msg


def test_session_mcp_servers_decline_cites_sdk_gap(mock_client: MagicMock) -> None:
    from airframe.tools import McpServerRef

    rt = OpenCodeServerRuntime()
    ref = McpServerRef(name="probe", transport="stdio", command=["x"])
    with pytest.raises(UnsupportedFeatureError) as exc_info:
        rt.session(mcp_servers=[ref])
    msg = str(exc_info.value)
    assert "opencode-ai 0.1.0a36" in msg
    assert "opencode.json" in msg


def test_session_on_permission_decline_cites_sdk_gap(mock_client: MagicMock) -> None:
    async def _cb(_: Any) -> str:
        return "allow"

    rt = OpenCodeServerRuntime()
    with pytest.raises(UnsupportedFeatureError) as exc_info:
        rt.session(on_permission=_cb)
    msg = str(exc_info.value)
    assert "opencode-ai 0.1.0a36" in msg
    assert "permission-reply endpoint" in msg


# --- Iteration D — feature flags stay False ----------------------------------


def test_iteration_d_does_not_flip_tools_permission_mcp_flags() -> None:
    """D's SDK constraints mean these flags remain False; later iteration flips them."""
    from airframe.adapters.opencode_server import OpenCodeServerRuntime as Rt

    for feature in (
        Feature.TOOLS_FUNCTION,
        Feature.TOOLS_MCP_STDIO,
        Feature.TOOLS_MCP_HTTP,
        Feature.TOOLS_MCP_SSE,
        Feature.PERMISSION_CALLBACK,
    ):
        assert feature not in Rt.SUPPORTED_FEATURES, (
            f"OpenCodeServerRuntime should not yet declare {feature.name} — "
            "the opencode-ai 0.1.0a36 SDK has no matching endpoint."
        )


# --- Iteration E — lifecycle hooks -------------------------------------------


@pytest.mark.asyncio
async def test_session_start_fires_on_first_turn(mock_client: MagicMock) -> None:
    events: list[Any] = []
    rt = OpenCodeServerRuntime()
    sess = rt.session(
        model=ProviderModel("opencode", "gpt-5-codex"),
        on_event=events.append,
    )
    await sess.execute("hi")
    kinds = [e.kind for e in events]
    # session_start fires once at first turn, before user_prompt_submit.
    assert kinds.index("session_start") < kinds.index("user_prompt_submit")
    start_event = next(e for e in events if e.kind == "session_start")
    assert start_event.session_id == "sess-1"
    assert start_event.payload["resumed"] is False
    assert start_event.payload["model"] == "gpt-5-codex"


@pytest.mark.asyncio
async def test_session_start_fires_only_once_across_turns(mock_client: MagicMock) -> None:
    events: list[Any] = []
    rt = OpenCodeServerRuntime()
    sess = rt.session(
        model=ProviderModel("opencode", "gpt-5-codex"),
        on_event=events.append,
    )
    await sess.execute("first")
    await sess.execute("second")
    starts = [e for e in events if e.kind == "session_start"]
    assert len(starts) == 1


@pytest.mark.asyncio
async def test_session_start_marks_resumed_session(mock_client: MagicMock) -> None:
    events: list[Any] = []
    rt = OpenCodeServerRuntime()
    sess = rt.session(
        resume="prior-id",
        model=ProviderModel("opencode", "gpt-5-codex"),
        on_event=events.append,
    )
    await sess.execute("hi")
    start = next(e for e in events if e.kind == "session_start")
    assert start.payload["resumed"] is True


@pytest.mark.asyncio
async def test_user_prompt_submit_carries_preview_and_length(
    mock_client: MagicMock,
) -> None:
    events: list[Any] = []
    rt = OpenCodeServerRuntime()
    sess = rt.session(
        model=ProviderModel("opencode", "gpt-5-codex"),
        on_event=events.append,
    )
    await sess.execute("freight-train problem")
    submit = next(e for e in events if e.kind == "user_prompt_submit")
    assert submit.payload["prompt"] == "freight-train problem"
    assert submit.payload["length"] == 21


@pytest.mark.asyncio
async def test_user_prompt_submit_truncates_long_prompts(mock_client: MagicMock) -> None:
    events: list[Any] = []
    rt = OpenCodeServerRuntime()
    sess = rt.session(
        model=ProviderModel("opencode", "gpt-5-codex"),
        on_event=events.append,
    )
    long_prompt = "x" * 5000
    await sess.execute(long_prompt)
    submit = next(e for e in events if e.kind == "user_prompt_submit")
    # Preview is bounded; length is the true length.
    assert len(submit.payload["prompt"]) <= 201  # 200 + "…"
    assert submit.payload["length"] == 5000


@pytest.mark.asyncio
async def test_session_end_fires_on_close(mock_client: MagicMock) -> None:
    events: list[Any] = []
    rt = OpenCodeServerRuntime()
    sess = rt.session(
        model=ProviderModel("opencode", "gpt-5-codex"),
        on_event=events.append,
    )
    await sess.execute("hi")
    await sess.close()
    end_events = [e for e in events if e.kind == "session_end"]
    assert len(end_events) == 1
    assert end_events[0].payload["turn_count"] == 1
    assert end_events[0].payload["cost_usd"] == pytest.approx(0.0042)


@pytest.mark.asyncio
async def test_session_end_is_idempotent(mock_client: MagicMock) -> None:
    events: list[Any] = []
    rt = OpenCodeServerRuntime()
    sess = rt.session(
        model=ProviderModel("opencode", "gpt-5-codex"),
        on_event=events.append,
    )
    await sess.execute("hi")
    await sess.close()
    await sess.close()
    await sess.close()
    end_events = [e for e in events if e.kind == "session_end"]
    assert len(end_events) == 1


@pytest.mark.asyncio
async def test_observer_raise_does_not_break_session(mock_client: MagicMock) -> None:
    """A buggy observer must not propagate into the session's vendor call."""

    def boom(_event: Any) -> None:
        raise RuntimeError("observer is broken")

    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"), on_event=boom)
    # Despite the observer raising on every hook, execute completes.
    result = await sess.execute("hi")
    assert result.cost.input_tokens == 100


@pytest.mark.asyncio
async def test_stream_fires_tool_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    """``pre_tool_use`` / ``post_tool_use`` synthesised from tool-part lifecycle."""
    import opencode_ai

    tool_running = _msg(
        id="t1",
        type="tool",
        session_id="sess-1",
        message_id="msg-1",
        tool="bash",
        state=_msg(status="running", input={"cmd": "ls"}, output=None, error=None),
    )
    tool_done = _msg(
        id="t1",
        type="tool",
        session_id="sess-1",
        message_id="msg-1",
        tool="bash",
        state=_msg(status="completed", input={"cmd": "ls"}, output="files", error=None),
    )
    sse = _FakeAsyncIter(
        [
            _event("message.part.updated", session_id="sess-1", part=tool_running),
            _event("message.part.updated", session_id="sess-1", part=tool_done),
            _event("session.idle", session_id="sess-1"),
        ]
    )
    client = MagicMock()
    client.session = MagicMock()
    client.session.create = AsyncMock(return_value=_msg(id="sess-1"))
    client.session.chat = AsyncMock(return_value=_assistant_message())
    client.session.delete = AsyncMock()
    client.app = MagicMock()
    client.app.providers = AsyncMock(return_value=_providers_payload())
    client.event = MagicMock()
    client.event.list = AsyncMock(return_value=sse)
    client.close = AsyncMock()
    monkeypatch.setattr(opencode_ai, "AsyncOpencode", MagicMock(return_value=client))

    events: list[Any] = []
    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"), on_event=events.append)
    async for _ in sess.stream("hi"):
        pass
    kinds = [e.kind for e in events]
    assert "pre_tool_use" in kinds
    assert "post_tool_use" in kinds
    pre = next(e for e in events if e.kind == "pre_tool_use")
    assert pre.payload["tool_name"] == "bash"
    assert pre.payload["tool_call_id"] == "t1"
    post = next(e for e in events if e.kind == "post_tool_use")
    assert post.payload["output"] == "files"


@pytest.mark.asyncio
async def test_stream_fires_tool_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import opencode_ai

    tool_err = _msg(
        id="t1",
        type="tool",
        session_id="sess-1",
        message_id="msg-1",
        tool="bash",
        state=_msg(status="error", input={"cmd": "false"}, output=None, error="exit 1"),
    )
    sse = _FakeAsyncIter(
        [
            _event("message.part.updated", session_id="sess-1", part=tool_err),
            _event("session.idle", session_id="sess-1"),
        ]
    )
    client = MagicMock()
    client.session = MagicMock()
    client.session.create = AsyncMock(return_value=_msg(id="sess-1"))
    client.session.chat = AsyncMock(return_value=_assistant_message())
    client.session.delete = AsyncMock()
    client.app = MagicMock()
    client.app.providers = AsyncMock(return_value=_providers_payload())
    client.event = MagicMock()
    client.event.list = AsyncMock(return_value=sse)
    client.close = AsyncMock()
    monkeypatch.setattr(opencode_ai, "AsyncOpencode", MagicMock(return_value=client))

    events: list[Any] = []
    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"), on_event=events.append)
    async for _ in sess.stream("hi"):
        pass
    fail = next(e for e in events if e.kind == "tool_failure")
    assert fail.payload["error"] == "exit 1"
    assert fail.payload["tool_name"] == "bash"


# --- Iteration E — budget caps -----------------------------------------------


@pytest.mark.asyncio
async def test_max_turns_zero_blocks_first_turn(mock_client: MagicMock) -> None:
    from airframe.errors import RuntimeBudgetExceededError

    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    with pytest.raises(RuntimeBudgetExceededError) as exc_info:
        await sess.execute("hi", max_turns=0)
    assert exc_info.value.kind == "turns"


@pytest.mark.asyncio
async def test_max_turns_one_allows_then_blocks(mock_client: MagicMock) -> None:
    from airframe.errors import RuntimeBudgetExceededError

    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    await sess.execute("first", max_turns=1)  # succeeds
    with pytest.raises(RuntimeBudgetExceededError):
        await sess.execute("second", max_turns=1)


@pytest.mark.asyncio
async def test_max_budget_usd_blocks_when_cap_already_spent(
    mock_client: MagicMock,
) -> None:
    """Running cost ≥ cap → next turn aborts before chat fires."""
    from airframe.errors import RuntimeBudgetExceededError

    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    # First turn returns cost=0.0042; cap=0.001 → cumulative trips before turn 2.
    await sess.execute("first", max_budget_usd=0.10)
    with pytest.raises(RuntimeBudgetExceededError) as exc_info:
        await sess.execute("second", max_budget_usd=0.001)
    assert exc_info.value.kind == "usd"


@pytest.mark.asyncio
async def test_cost_accumulates_across_turns(mock_client: MagicMock) -> None:
    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    await sess.execute("first")
    await sess.execute("second")
    await sess.execute("third")
    assert sess._cumulative_cost_usd == pytest.approx(0.0042 * 3)
    assert sess._turn_count == 3


@pytest.mark.asyncio
async def test_unreported_cost_does_not_break_budget_tracking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the upstream doesn't report cost, the cap is best-effort.

    Token counters still increment so ``max_turns`` works; the dollar
    accumulator stays at 0 and ``max_budget_usd`` becomes effectively
    unenforced for that turn (debug-logged in the source).
    """
    import opencode_ai

    msg = _assistant_message(cost=None)  # type: ignore[arg-type]
    client = MagicMock()
    client.session = MagicMock()
    client.session.create = AsyncMock(return_value=_msg(id="sess-1"))
    client.session.chat = AsyncMock(return_value=msg)
    client.session.delete = AsyncMock()
    client.app = MagicMock()
    client.app.providers = AsyncMock(return_value=_providers_payload())
    client.event = MagicMock()
    client.close = AsyncMock()
    monkeypatch.setattr(opencode_ai, "AsyncOpencode", MagicMock(return_value=client))

    rt = OpenCodeServerRuntime()
    sess = rt.session(model=ProviderModel("opencode", "gpt-5-codex"))
    await sess.execute("hi")
    assert sess._turn_count == 1
    assert sess._cumulative_cost_usd == 0.0


# --- Native (vendor-hosted) tools: WEB_SEARCH/WEB_FETCH → tools allow-map ---


@pytest.mark.asyncio
async def test_native_tools_force_enabled_in_chat_tools_map(
    mock_client: MagicMock,
) -> None:
    """A native request sets ``websearch``/``webfetch`` True in the chat
    ``tools=`` allow-map."""
    from airframe.native_tools import NativeTool

    rt = OpenCodeServerRuntime()
    sess = rt.session(
        model=ProviderModel("opencode", "some-private-model"),
        provider_options=OpenCodeServerOptions(provider_id="openrouter"),
        native_tools=[NativeTool.web_search(), NativeTool.web_fetch()],
    )
    await sess.execute("hi")
    _, kwargs = mock_client.session.chat.call_args
    assert kwargs["tools"]["websearch"] is True
    assert kwargs["tools"]["webfetch"] is True


@pytest.mark.asyncio
async def test_native_tool_overrides_denylist(mock_client: MagicMock) -> None:
    """An explicit native request wins over an ``excluded_tools`` denylist."""
    from airframe.native_tools import NativeTool

    rt = OpenCodeServerRuntime()
    sess = rt.session(
        model=ProviderModel("opencode", "some-private-model"),
        provider_options=OpenCodeServerOptions(
            provider_id="openrouter", excluded_tools=("websearch",)
        ),
        native_tools=[NativeTool.web_search()],
    )
    await sess.execute("hi")
    _, kwargs = mock_client.session.chat.call_args
    assert kwargs["tools"]["websearch"] is True
