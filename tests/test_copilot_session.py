"""Unit tests for :class:`CopilotAgentSession`.

Phase 1 Iteration E — third per-vendor session (after OpenAI-compat
and Claude Code). Mocks the ``copilot`` SDK at the boundary so we
exercise the session.on(handler) → asyncio.Queue → AsyncIterator
streaming plumbing, the resume= path through
:meth:`CopilotClient.resume_session`, and cancellation via
:meth:`CopilotSession.abort` without spawning a CLI subprocess.

Live-vendor probes belong in ``airframe.testing.integration``
(Phase 1 work).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from airframe.adapters.copilot import (
    SUBMIT_RESULT_TOOL,
    CopilotAgentSession,
    CopilotRuntime,
)
from airframe.events import ReasoningDelta, TextDelta, TurnComplete
from airframe.features import Feature
from airframe.protocol import AgentSession


class _Schema(BaseModel):
    summary: str
    count: int


# ---------------------------------------------------------------------------
# Fake event-data classes — real classes so isinstance() dispatch fires.
# ---------------------------------------------------------------------------


class _FakeUsage:
    def __init__(
        self,
        *,
        cost: float | None = 0.001,
        input_tokens: float | None = 50,
        output_tokens: float | None = 25,
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


class _FakeMessageDelta:
    def __init__(self, *, delta_content: str, message_id: str = "msg-1") -> None:
        self.delta_content = delta_content
        self.message_id = message_id
        self.parent_tool_call_id = None


class _FakeReasoningDelta:
    def __init__(self, *, delta_content: str, reasoning_id: str = "r-1") -> None:
        self.delta_content = delta_content
        self.reasoning_id = reasoning_id


class _FakeEvent:
    def __init__(self, data: Any) -> None:
        self.data = data


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sdk(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace copilot SDK symbols CopilotAgentSession imports lazily."""
    import copilot
    from copilot import session as session_mod
    from copilot.generated import session_events as se_mod

    # Captured handler bag: on(handler) appends; unsubscribe pops.
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
    mock_client.stop = AsyncMock()

    def fake_subprocess_config(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    mock_client_factory = MagicMock(return_value=mock_client)

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
    monkeypatch.setattr(copilot, "SubprocessConfig", fake_subprocess_config)
    monkeypatch.setattr(copilot, "define_tool", fake_define_tool)
    monkeypatch.setattr(session_mod, "PermissionHandler", mock_perm)

    # Fake event-data classes so isinstance() dispatch in the session
    # picks them up.
    monkeypatch.setattr(se_mod, "AssistantUsageData", _FakeUsage)
    monkeypatch.setattr(se_mod, "AssistantMessageData", _FakeAssistantMessage)
    monkeypatch.setattr(se_mod, "SessionErrorData", _FakeSessionError)
    monkeypatch.setattr(se_mod, "AssistantMessageDeltaData", _FakeMessageDelta)
    monkeypatch.setattr(se_mod, "AssistantReasoningDeltaData", _FakeReasoningDelta)

    return {
        "client_factory": mock_client_factory,
        "client": mock_client,
        "session": mock_session,
        "handlers": handlers,
        "captured_tools": captured_tools,
    }


def _fire(handlers: list[Any], event: Any) -> None:
    """Invoke every subscribed handler synchronously with one event."""
    for h in handlers:
        h(event)


# ---------------------------------------------------------------------------
# Factory + capability surface
# ---------------------------------------------------------------------------


async def test_session_factory_returns_bespoke_session(mock_sdk: dict[str, Any]) -> None:
    rt = CopilotRuntime()
    sess = rt.session()
    try:
        assert isinstance(sess, CopilotAgentSession)
        assert isinstance(sess, AgentSession)
        assert sess.id is None
    finally:
        await sess.close()


def test_streaming_resume_cancel_features_declared() -> None:
    rt = CopilotRuntime()
    assert rt.supports(Feature.STREAMING)
    assert rt.supports(Feature.SESSION_RESUME)
    assert rt.supports(Feature.CANCEL)


async def test_resume_seeds_id_and_calls_resume_session(mock_sdk: dict[str, Any]) -> None:
    """resume=<id> seeds session.id immediately and routes to client.resume_session."""
    rt = CopilotRuntime()
    sess = rt.session(resume="sess-resumed")
    try:
        assert sess.id == "sess-resumed"  # surfaced before any turn

        # Drive a turn — that triggers _ensure_session, which should
        # call client.resume_session (not create_session).
        async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
            _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
            _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

        mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)
        await sess.execute("hi")
    finally:
        await sess.close()

    mock_sdk["client"].resume_session.assert_awaited()
    mock_sdk["client"].create_session.assert_not_called()
    # Live session_id from the underlying CopilotSession surfaces.
    assert sess.id == "live-sess-id"


async def test_fresh_session_uses_create_session(mock_sdk: dict[str, Any]) -> None:
    """resume=None → create_session, not resume_session."""

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session()
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    mock_sdk["client"].create_session.assert_awaited()
    mock_sdk["client"].resume_session.assert_not_called()


# ---------------------------------------------------------------------------
# Execute — plain text + structured paths
# ---------------------------------------------------------------------------


async def test_execute_plain_text_returns_assistant_message_content(
    mock_sdk: dict[str, Any],
) -> None:
    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage(input_tokens=10, output_tokens=5)))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("hello world")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session()
    try:
        result = await sess.execute("hi")
    finally:
        await sess.close()

    assert result.text == "hello world"
    assert result.structured is None
    assert result.cost.input_tokens == 10
    assert result.cost.output_tokens == 5


async def test_execute_structured_captures_submit_result_payload(
    mock_sdk: dict[str, Any],
) -> None:
    """The submit_result tool handler captures the typed payload."""
    captured_handler: dict[str, Any] = {}

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        # Drive the submit_result handler to populate the capture.
        tool = mock_sdk["captured_tools"][0]
        captured_handler["params_type"] = tool["params_type"]
        await tool["handler"](_Schema(summary="ok", count=42), MagicMock())
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session(system="be helpful")
    try:
        result = await sess.execute("brief me", schema=_Schema)
    finally:
        await sess.close()

    assert result.structured == {"summary": "ok", "count": 42}
    assert captured_handler["params_type"] is _Schema
    # The submit_result tool was registered with the schema baked in.
    tool = mock_sdk["captured_tools"][0]
    assert tool["name"] == SUBMIT_RESULT_TOOL
    assert tool["skip_permission"] is True


async def test_client_reused_across_same_schema_turns(mock_sdk: dict[str, Any]) -> None:
    """Two same-schema turns share one create_session call."""

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("r")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session()
    try:
        await sess.execute("a")
        await sess.execute("b")
    finally:
        await sess.close()

    assert mock_sdk["client"].create_session.await_count == 1


async def test_client_rebuilds_when_schema_changes(mock_sdk: dict[str, Any]) -> None:
    """Schema fingerprint change destroys + rebuilds — submit_result is bake-time."""
    call_idx = {"n": 0}

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        idx = call_idx["n"]
        call_idx["n"] += 1
        if idx == 0:
            # Plain-text turn.
            _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
            _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("plain")))
        else:
            # Structured turn — fire the submit_result handler.
            tool = mock_sdk["captured_tools"][-1]
            await tool["handler"](_Schema(summary="ok", count=1), MagicMock())
            _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session()
    try:
        await sess.execute("plain")
        await sess.execute("structured", schema=_Schema)
    finally:
        await sess.close()

    assert mock_sdk["client"].create_session.await_count == 2
    assert mock_sdk["session"].destroy.await_count >= 1


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


async def test_stream_yields_text_deltas_then_turn_complete(
    mock_sdk: dict[str, Any],
) -> None:
    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        # Fire two text deltas, then capture the final assistant message
        # + usage so _build_result has something to populate the
        # TurnComplete with.
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeMessageDelta(delta_content="Hel")))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeMessageDelta(delta_content="lo")))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage(input_tokens=5, output_tokens=2)))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("Hello")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session()
    yielded: list[Any] = []
    try:
        async for ev in sess.stream("hi"):
            yielded.append(ev)
    finally:
        await sess.close()

    text_events = [e for e in yielded if isinstance(e, TextDelta)]
    turn_events = [e for e in yielded if isinstance(e, TurnComplete)]
    assert [e.text for e in text_events] == ["Hel", "lo"]
    assert len(turn_events) == 1
    assert turn_events[0].result.text == "Hello"
    assert turn_events[0].result.cost.input_tokens == 5


async def test_stream_yields_reasoning_deltas(mock_sdk: dict[str, Any]) -> None:
    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeReasoningDelta(delta_content="let me think")))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeMessageDelta(delta_content="done")))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("done")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session()
    yielded: list[Any] = []
    try:
        async for ev in sess.stream("think hard"):
            yielded.append(ev)
    finally:
        await sess.close()

    reasoning = [e for e in yielded if isinstance(e, ReasoningDelta)]
    assert [e.text for e in reasoning] == ["let me think"]


# ---------------------------------------------------------------------------
# thinking= (Phase 2 Iteration B)
# ---------------------------------------------------------------------------


async def test_thinking_effort_forwarded_to_create_session(
    mock_sdk: dict[str, Any],
) -> None:
    """thinking=<effort> becomes the ``reasoning_effort`` kwarg on create_session."""

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session()
    try:
        await sess.execute("hi", thinking="high")
    finally:
        await sess.close()

    call = mock_sdk["client"].create_session.await_args_list[0]
    assert call.kwargs["reasoning_effort"] == "high"


async def test_thinking_minimal_coerces_to_low(mock_sdk: dict[str, Any]) -> None:
    """Copilot SDK lacks 'minimal' — adapter coerces to 'low' at debug log level."""

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session()
    try:
        await sess.execute("hi", thinking="minimal")
    finally:
        await sess.close()

    call = mock_sdk["client"].create_session.await_args_list[0]
    assert call.kwargs["reasoning_effort"] == "low"


async def test_thinking_none_or_disabled_omits_kwarg(mock_sdk: dict[str, Any]) -> None:
    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session()
    try:
        await sess.execute("plain")
        await sess.execute("explicit-off", thinking="disabled")
    finally:
        await sess.close()

    for call in mock_sdk["client"].create_session.await_args_list:
        assert "reasoning_effort" not in call.kwargs


async def test_thinking_change_between_turns_rebuilds_session(
    mock_sdk: dict[str, Any],
) -> None:
    """``reasoning_effort`` is baked at create_session() time → rebuild on change."""

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session()
    try:
        await sess.execute("a", thinking="low")
        await sess.execute("b", thinking="high")
    finally:
        await sess.close()

    assert mock_sdk["client"].create_session.await_count == 2
    efforts = [
        c.kwargs.get("reasoning_effort") for c in mock_sdk["client"].create_session.await_args_list
    ]
    assert efforts == ["low", "high"]


async def test_same_thinking_across_turns_reuses_session(
    mock_sdk: dict[str, Any],
) -> None:
    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session()
    try:
        await sess.execute("a", thinking="medium")
        await sess.execute("b", thinking="medium")
    finally:
        await sess.close()

    assert mock_sdk["client"].create_session.await_count == 1


async def test_thinking_dict_raises_unsupported_feature_error(
    mock_sdk: dict[str, Any],
) -> None:
    from airframe.errors import UnsupportedFeatureError

    rt = CopilotRuntime()
    sess = rt.session()
    try:
        with pytest.raises(UnsupportedFeatureError) as exc_info:
            await sess.execute("hi", thinking={"budget_tokens": 5000})
    finally:
        await sess.close()

    assert exc_info.value.feature == Feature.REASONING_BUDGET_TOKENS


# ---------------------------------------------------------------------------
# Vision / file input (Phase 2 Iteration C)
# ---------------------------------------------------------------------------


async def test_image_input_routes_to_attachments(mock_sdk: dict[str, Any]) -> None:
    """ImageInput(path=) becomes ``{"type":"file","path":...}`` in attachments=."""
    from airframe.inputs import ImageInput

    captured: dict[str, Any] = {}

    async def fake_send(prompt: str, *, timeout: float, **kwargs: Any) -> None:
        captured["prompt"] = prompt
        captured["attachments"] = kwargs.get("attachments")
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session()
    try:
        await sess.execute(["caption:", ImageInput(path="/tmp/x.png")])
    finally:
        await sess.close()

    assert captured["prompt"] == "caption:"
    assert captured["attachments"] == [{"type": "file", "path": "/tmp/x.png"}]


async def test_file_input_routes_to_attachments(mock_sdk: dict[str, Any]) -> None:
    """FileInput(path=) also maps to FileAttachment — Copilot handles them uniformly."""
    from airframe.inputs import FileInput

    captured: dict[str, Any] = {}

    async def fake_send(prompt: str, *, timeout: float, **kwargs: Any) -> None:
        captured["attachments"] = kwargs.get("attachments")
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session()
    try:
        await sess.execute(["summarise:", FileInput(path="/tmp/spec.pdf")])
    finally:
        await sess.close()

    assert captured["attachments"] == [{"type": "file", "path": "/tmp/spec.pdf"}]


async def test_plain_string_uses_no_attachments(mock_sdk: dict[str, Any]) -> None:
    """No parts → attachments kwarg is None (SDK keeps its default behaviour)."""
    captured: dict[str, Any] = {}

    async def fake_send(prompt: str, *, timeout: float, **kwargs: Any) -> None:
        captured["attachments"] = kwargs.get("attachments")
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session()
    try:
        await sess.execute("just text")
    finally:
        await sess.close()

    assert captured["attachments"] is None


async def test_image_input_bytes_routes_to_blob_attachment(
    mock_sdk: dict[str, Any],
) -> None:
    """Iteration D: bytes_= → BlobAttachment ({"type":"blob","data":<b64>,...})."""
    import base64

    from airframe.inputs import ImageInput

    captured: dict[str, Any] = {}

    async def fake_send(prompt: str, *, timeout: float, **kwargs: Any) -> None:
        captured["attachments"] = kwargs.get("attachments")
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session()
    try:
        await sess.execute(["caption:", ImageInput(bytes_=b"raw", media_type="image/jpeg")])
    finally:
        await sess.close()

    assert captured["attachments"] == [
        {
            "type": "blob",
            "data": base64.b64encode(b"raw").decode("ascii"),
            "mimeType": "image/jpeg",
        }
    ]


async def test_image_input_bytes_default_media_type(mock_sdk: dict[str, Any]) -> None:
    """bytes_= without media_type defaults to image/png."""
    from airframe.inputs import ImageInput

    captured: dict[str, Any] = {}

    async def fake_send(prompt: str, *, timeout: float, **kwargs: Any) -> None:
        captured["attachments"] = kwargs.get("attachments")
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session()
    try:
        await sess.execute([ImageInput(bytes_=b"raw")])
    finally:
        await sess.close()

    assert captured["attachments"][0]["mimeType"] == "image/png"


async def test_image_input_url_raises_unsupported_feature(
    mock_sdk: dict[str, Any],
) -> None:
    """Copilot SDK has no URL channel — url= raises with VISION_INPUT."""
    from airframe.errors import UnsupportedFeatureError
    from airframe.inputs import ImageInput

    rt = CopilotRuntime()
    sess = rt.session()
    try:
        with pytest.raises(UnsupportedFeatureError) as exc_info:
            await sess.execute(["x", ImageInput(url="https://example.com/x.png")])
    finally:
        await sess.close()
    assert exc_info.value.feature == Feature.VISION_INPUT


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


async def test_cancel_calls_abort(mock_sdk: dict[str, Any]) -> None:
    """cancel() during execute() calls session.abort()."""
    cancel_event = asyncio.Event()

    async def hang_send(prompt: str, *, timeout: float, **_: Any) -> None:
        await cancel_event.wait()

    def _trigger_unblock() -> None:
        # Sync side_effect — AsyncMock awaits this and returns its
        # result. Setting the event unblocks hang_send so the awaiting
        # execute() task can return.
        cancel_event.set()

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=hang_send)
    mock_sdk["session"].abort = AsyncMock(side_effect=_trigger_unblock)

    rt = CopilotRuntime()
    sess = rt.session()
    try:
        exec_task = asyncio.create_task(sess.execute("hangs"))
        # Wait until the session sets _in_flight.
        for _ in range(50):
            await asyncio.sleep(0)
            if sess._in_flight:  # noqa: SLF001 — invariant probe
                break
        assert sess._in_flight  # noqa: SLF001
        await sess.cancel()
        # send_and_wait returns when abort unblocks; the result is
        # built from whatever was captured (empty here), which surfaces
        # as an empty plain-text result rather than RuntimeCancelledError.
        # The contract verified here is just: abort() was called.
        await exec_task
    finally:
        await sess.close()

    mock_sdk["session"].abort.assert_awaited()


async def test_cancel_when_idle_is_noop(mock_sdk: dict[str, Any]) -> None:
    rt = CopilotRuntime()
    sess = rt.session()
    try:
        await sess.cancel()
        await sess.cancel()
    finally:
        await sess.close()
    mock_sdk["session"].abort.assert_not_called()


# ---------------------------------------------------------------------------
# close() lifecycle
# ---------------------------------------------------------------------------


async def test_close_destroys_session_and_blocks_further_execute(
    mock_sdk: dict[str, Any],
) -> None:
    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("r")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session()
    await sess.execute("warm")
    await sess.close()
    mock_sdk["session"].destroy.assert_awaited()
    with pytest.raises(RuntimeError):
        await sess.execute("nope")


async def test_close_is_idempotent(mock_sdk: dict[str, Any]) -> None:
    rt = CopilotRuntime()
    sess = rt.session()
    await sess.close()
    await sess.close()
    await sess.close()


async def test_close_does_not_stop_runtime_client(mock_sdk: dict[str, Any]) -> None:
    """Session.close() destroys the session but leaves the runtime's client alive."""

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("r")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session()
    await sess.execute("hi")
    await sess.close()
    mock_sdk["client"].stop.assert_not_called()
