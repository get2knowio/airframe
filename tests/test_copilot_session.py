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


class _FakeToolStart:
    def __init__(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments: Any = None,
    ) -> None:
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.arguments = arguments
        self.mcp_server_name = None
        self.mcp_tool_name = None
        self.parent_tool_call_id = None


class _FakeToolResult:
    def __init__(self, *, content: str = "") -> None:
        self.content = content
        self.contents = None
        self.detailed_content = None


class _FakeToolError:
    def __init__(self, *, message: str, code: str | None = None) -> None:
        self.message = message
        self.code = code


class _FakeToolComplete:
    def __init__(
        self,
        *,
        tool_call_id: str,
        success: bool = True,
        result: Any = None,
        error: Any = None,
    ) -> None:
        self.tool_call_id = tool_call_id
        self.success = success
        self.result = result
        self.error = error
        self.interaction_id = None
        self.is_user_requested = None
        self.model = None
        self.parent_tool_call_id = None
        self.tool_telemetry = None


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
    monkeypatch.setattr(se_mod, "ToolExecutionStartData", _FakeToolStart)
    monkeypatch.setattr(se_mod, "ToolExecutionCompleteData", _FakeToolComplete)

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


# ---------------------------------------------------------------------------
# Function tools (Phase 3 Iteration C)
# ---------------------------------------------------------------------------


class _AddParams(BaseModel):
    a: float
    b: float


async def _add(params: _AddParams) -> float:
    return params.a + params.b


def _build_tool() -> Any:
    from airframe import FunctionTool

    return FunctionTool(
        name="add",
        description="Add two numbers.",
        params=_AddParams,
        handler=_add,
    )


async def test_tools_passed_to_create_session(mock_sdk: dict[str, Any]) -> None:
    """tools= becomes the ``tools=`` kwarg on CopilotClient.create_session."""

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("done")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session(tools=[_build_tool()])
    try:
        await sess.execute("call add")
    finally:
        await sess.close()

    create_call = mock_sdk["client"].create_session.await_args_list[0]
    assert "tools" in create_call.kwargs
    # Exactly one tool (no submit_result; schema=None).
    assert len(create_call.kwargs["tools"]) == 1
    # define_tool was called with the FunctionTool's metadata.
    captured = mock_sdk["captured_tools"]
    assert any(
        t["name"] == "add"
        and t["description"] == "Add two numbers."
        and t["params_type"] is _AddParams
        and t["skip_permission"] is True
        for t in captured
    )


async def test_tools_handler_signature_adapted_to_copilot(
    mock_sdk: dict[str, Any],
) -> None:
    """Copilot hands the handler ``(params, invocation_context)`` — the wrapper
    discards the second arg and awaits the airframe handler."""

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("done")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session(tools=[_build_tool()])
    try:
        await sess.execute("anything")
    finally:
        await sess.close()

    # Pull the captured handler and call it with the SDK's (params,
    # invocation) shape; the wrapper should ignore the invocation.
    captured = next(t for t in mock_sdk["captured_tools"] if t["name"] == "add")
    params = _AddParams(a=17, b=23)
    result = await captured["handler"](params, MagicMock())
    assert result == 40.0


async def test_submit_result_coexists_with_user_tools(mock_sdk: dict[str, Any]) -> None:
    """schema= + tools= → submit_result first, then user tools."""

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        # Find the captured submit_result handler in mock_sdk and invoke it
        # with a payload that matches the schema, so _build_result returns
        # the structured payload (not a structured-output failure).
        submit_entry = next(
            t for t in mock_sdk["captured_tools"] if t["name"] == SUBMIT_RESULT_TOOL
        )
        await submit_entry["handler"](_Schema(summary="ok", count=1), MagicMock())
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session(tools=[_build_tool()])
    try:
        result = await sess.execute("brief me", schema=_Schema)
    finally:
        await sess.close()

    create_call = mock_sdk["client"].create_session.await_args_list[0]
    tools = create_call.kwargs["tools"]
    assert len(tools) == 2
    # submit_result is in slot 0 (the model sees the gate first).
    captured_order = [t["name"] for t in mock_sdk["captured_tools"]]
    assert captured_order.index(SUBMIT_RESULT_TOOL) < captured_order.index("add")
    # The structured-output round-trip still produced its dict payload.
    assert result.structured == {"summary": "ok", "count": 1}


async def test_no_tools_omits_tools_kwarg(mock_sdk: dict[str, Any]) -> None:
    """No tools= and no schema= → no ``tools=`` on create_session."""

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

    create_call = mock_sdk["client"].create_session.await_args_list[0]
    assert "tools" not in create_call.kwargs


async def test_stream_emits_tool_call_start_and_result(mock_sdk: dict[str, Any]) -> None:
    """stream() translates ToolExecutionStart/Complete events into airframe events."""
    from airframe.events import ToolCallResult, ToolCallStart

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(
            mock_sdk["handlers"],
            _FakeEvent(
                _FakeToolStart(
                    tool_call_id="tc-1",
                    tool_name="add",
                    arguments={"a": 17, "b": 23},
                )
            ),
        )
        _fire(
            mock_sdk["handlers"],
            _FakeEvent(
                _FakeToolComplete(
                    tool_call_id="tc-1",
                    success=True,
                    result=_FakeToolResult(content="40.0"),
                )
            ),
        )
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeMessageDelta(delta_content="17+23=40")))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("17+23=40")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session(tools=[_build_tool()])
    events: list[Any] = []
    try:
        async for event in sess.stream("what's 17 + 23?"):
            events.append(event)
    finally:
        await sess.close()

    starts = [e for e in events if isinstance(e, ToolCallStart)]
    results = [e for e in events if isinstance(e, ToolCallResult)]
    turns = [e for e in events if isinstance(e, TurnComplete)]
    assert len(starts) == 1
    assert starts[0].tool_name == "add"
    assert starts[0].tool_call_id == "tc-1"
    assert "17" in starts[0].arguments_preview and "23" in starts[0].arguments_preview
    assert len(results) == 1
    assert results[0].tool_call_id == "tc-1"
    assert results[0].output == "40.0"
    assert results[0].is_error is False
    assert len(turns) == 1
    assert events[-1] is turns[0]


async def test_stream_submit_result_tool_call_is_filtered(
    mock_sdk: dict[str, Any],
) -> None:
    """The forced submit_result tool is structured-output plumbing, not a
    user-visible tool call — the streaming events suppress it."""
    from airframe.events import ToolCallResult, ToolCallStart

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        # Drive the submit_result handler so _build_result yields a structured
        # payload (and we can be sure suppression doesn't break that path).
        submit_entry = next(
            t for t in mock_sdk["captured_tools"] if t["name"] == SUBMIT_RESULT_TOOL
        )
        await submit_entry["handler"](_Schema(summary="ok", count=1), MagicMock())
        _fire(
            mock_sdk["handlers"],
            _FakeEvent(
                _FakeToolStart(
                    tool_call_id="submit-1",
                    tool_name=SUBMIT_RESULT_TOOL,
                    arguments={"summary": "ok", "count": 1},
                )
            ),
        )
        _fire(
            mock_sdk["handlers"],
            _FakeEvent(
                _FakeToolComplete(
                    tool_call_id="submit-1",
                    success=True,
                    result=_FakeToolResult(content='{"ok": true}'),
                )
            ),
        )
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session(tools=[_build_tool()])
    events: list[Any] = []
    try:
        async for event in sess.stream("brief me", schema=_Schema):
            events.append(event)
    finally:
        await sess.close()

    # The submit_result tool call did not produce visible events.
    starts = [e for e in events if isinstance(e, ToolCallStart)]
    results = [e for e in events if isinstance(e, ToolCallResult)]
    assert starts == []
    assert results == []
    # But the final TurnComplete still carries the structured payload.
    turn = next(e for e in events if isinstance(e, TurnComplete))
    assert turn.result.structured == {"summary": "ok", "count": 1}


async def test_stream_tool_failure_surfaces_is_error(mock_sdk: dict[str, Any]) -> None:
    """ToolExecutionCompleteData(success=False) maps to is_error=True."""
    from airframe.events import ToolCallResult

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(
            mock_sdk["handlers"],
            _FakeEvent(
                _FakeToolStart(
                    tool_call_id="tc-x",
                    tool_name="add",
                    arguments={"a": 1, "b": 2},
                )
            ),
        )
        _fire(
            mock_sdk["handlers"],
            _FakeEvent(
                _FakeToolComplete(
                    tool_call_id="tc-x",
                    success=False,
                    error=_FakeToolError(message="kaboom", code="E_ARG"),
                )
            ),
        )
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("apologies")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session(tools=[_build_tool()])
    results: list[ToolCallResult] = []
    try:
        async for event in sess.stream("call add"):
            if isinstance(event, ToolCallResult):
                results.append(event)
    finally:
        await sess.close()

    assert len(results) == 1
    assert results[0].is_error is True
    assert "E_ARG" in str(results[0].output)
    assert "kaboom" in str(results[0].output)


async def test_tools_change_between_turns_rebuilds_session(
    mock_sdk: dict[str, Any],
) -> None:
    """A tools-list change invalidates the cached CopilotSession."""

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("r")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    from airframe import FunctionTool

    async def _other(_: _AddParams) -> float:
        return 0.0

    other = FunctionTool(name="other", description="X", params=_AddParams, handler=_other)

    rt = CopilotRuntime()
    sess1 = rt.session(tools=[_build_tool()])
    try:
        await sess1.execute("turn 1")
    finally:
        await sess1.close()
    first_create_count = mock_sdk["client"].create_session.await_count

    sess2 = rt.session(tools=[other])
    try:
        await sess2.execute("turn 2")
    finally:
        await sess2.close()

    # A fresh session always creates a fresh CopilotSession (sessions
    # don't share state across the factory boundary).
    assert mock_sdk["client"].create_session.await_count > first_create_count


# ---------------------------------------------------------------------------
# External MCP server refs (Phase 4 Iteration C)
# ---------------------------------------------------------------------------


def _mcp_send_factory(mock_sdk: dict[str, Any]) -> Any:
    """Standard ``send_and_wait`` side effect: fire one usage + one
    assistant-message event so :meth:`_build_result` succeeds."""

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    return fake_send


async def test_mcp_stdio_translates_to_local_dict_with_args(
    mock_sdk: dict[str, Any],
) -> None:
    """``transport='stdio'`` builds ``{type:"local", command, args}`` —
    Copilot's wire enum is ``local``, not ``stdio``."""
    from airframe import McpServerRef

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=_mcp_send_factory(mock_sdk))

    rt = CopilotRuntime()
    sess = rt.session(
        mcp_servers=[
            McpServerRef(
                name="everything",
                transport="stdio",
                command=["uvx", "mcp-server-everything", "--flag"],
            )
        ]
    )
    try:
        await sess.execute("anything")
    finally:
        await sess.close()

    create_call = mock_sdk["client"].create_session.await_args_list[0]
    assert "mcp_servers" in create_call.kwargs
    servers = create_call.kwargs["mcp_servers"]
    assert "everything" in servers
    cfg = servers["everything"]
    assert cfg["type"] == "local"
    assert cfg["command"] == "uvx"
    assert cfg["args"] == ["mcp-server-everything", "--flag"]
    # No url/headers leakage on a local config.
    assert "url" not in cfg
    assert "headers" not in cfg


async def test_mcp_stdio_single_element_command_emits_empty_args(
    mock_sdk: dict[str, Any],
) -> None:
    """Copilot's wire schema requires ``args`` even when empty — emit ``[]``."""
    from airframe import McpServerRef

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=_mcp_send_factory(mock_sdk))

    rt = CopilotRuntime()
    sess = rt.session(
        mcp_servers=[McpServerRef(name="solo", transport="stdio", command=["mcp-bin"])]
    )
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    cfg = mock_sdk["client"].create_session.await_args_list[0].kwargs["mcp_servers"]["solo"]
    assert cfg == {"type": "local", "command": "mcp-bin", "args": []}


async def test_mcp_http_translates_with_auth_header(mock_sdk: dict[str, Any]) -> None:
    """``auth_token`` becomes ``Authorization: Bearer …`` on http transport."""
    from airframe import McpServerRef

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=_mcp_send_factory(mock_sdk))

    rt = CopilotRuntime()
    sess = rt.session(
        mcp_servers=[
            McpServerRef(
                name="remote",
                transport="http",
                url="https://mcp.example.com",
                auth_token="secret-token",
            )
        ]
    )
    try:
        await sess.execute("anything")
    finally:
        await sess.close()

    cfg = mock_sdk["client"].create_session.await_args_list[0].kwargs["mcp_servers"]["remote"]
    assert cfg["type"] == "http"
    assert cfg["url"] == "https://mcp.example.com"
    assert cfg["headers"]["Authorization"] == "Bearer secret-token"
    assert "command" not in cfg


async def test_mcp_http_caller_headers_override_auth_token(
    mock_sdk: dict[str, Any],
) -> None:
    """Caller-supplied ``Authorization`` in ``headers=`` wins over ``auth_token``.

    Same precedence rule as Claude (Iteration B); shared
    :func:`~airframe.sessions._compose_mcp_headers` governs both.
    """
    from airframe import McpServerRef

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=_mcp_send_factory(mock_sdk))

    rt = CopilotRuntime()
    sess = rt.session(
        mcp_servers=[
            McpServerRef(
                name="remote",
                transport="http",
                url="https://mcp.example.com",
                auth_token="shorthand",
                headers={"Authorization": "Bearer caller-explicit", "X-Trace": "abc"},
            )
        ]
    )
    try:
        await sess.execute("anything")
    finally:
        await sess.close()

    cfg = mock_sdk["client"].create_session.await_args_list[0].kwargs["mcp_servers"]["remote"]
    assert cfg["headers"]["Authorization"] == "Bearer caller-explicit"
    assert cfg["headers"]["X-Trace"] == "abc"


async def test_mcp_sse_ref_raises_before_create_session(mock_sdk: dict[str, Any]) -> None:
    """SSE refs are rejected at :meth:`session` — Copilot never builds a session."""
    from airframe import McpServerRef
    from airframe.errors import UnsupportedFeatureError

    rt = CopilotRuntime()
    with pytest.raises(UnsupportedFeatureError) as exc_info:
        rt.session(
            mcp_servers=[McpServerRef(name="feed", transport="sse", url="https://example.com/sse")]
        )
    assert exc_info.value.feature == Feature.TOOLS_MCP_SSE
    text = str(exc_info.value).lower()
    # The decline points at the working alternative.
    assert "http" in text
    assert "sse" in text
    # And no session was ever built.
    mock_sdk["client"].create_session.assert_not_called()
    mock_sdk["client"].resume_session.assert_not_called()


async def test_mcp_mixed_stdio_and_http_in_one_session(
    mock_sdk: dict[str, Any],
) -> None:
    """A list combining stdio + http lands as one dict keyed by name."""
    from airframe import McpServerRef

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=_mcp_send_factory(mock_sdk))

    rt = CopilotRuntime()
    sess = rt.session(
        mcp_servers=[
            McpServerRef(name="local", transport="stdio", command=["a"]),
            McpServerRef(name="rest", transport="http", url="https://h.example.com"),
        ]
    )
    try:
        await sess.execute("anything")
    finally:
        await sess.close()

    servers = mock_sdk["client"].create_session.await_args_list[0].kwargs["mcp_servers"]
    assert set(servers.keys()) == {"local", "rest"}
    assert servers["local"]["type"] == "local"
    assert servers["rest"]["type"] == "http"


async def test_mcp_servers_coexist_with_tools(mock_sdk: dict[str, Any]) -> None:
    """``tools=`` and ``mcp_servers=`` pass through separate create_session kwargs."""
    from airframe import McpServerRef

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=_mcp_send_factory(mock_sdk))

    rt = CopilotRuntime()
    sess = rt.session(
        tools=[_build_tool()],
        mcp_servers=[McpServerRef(name="external", transport="stdio", command=["x"])],
    )
    try:
        await sess.execute("anything")
    finally:
        await sess.close()

    create_call = mock_sdk["client"].create_session.await_args_list[0]
    # tools= still landed (the FunctionTool registration).
    assert "tools" in create_call.kwargs
    assert len(create_call.kwargs["tools"]) == 1
    # mcp_servers= landed alongside it.
    assert "mcp_servers" in create_call.kwargs
    assert "external" in create_call.kwargs["mcp_servers"]


async def test_mcp_servers_coexist_with_tools_and_submit_result(
    mock_sdk: dict[str, Any],
) -> None:
    """Three-way coexistence: forced ``submit_result`` + user tools + external MCP.

    Plan-required scenario — schema= forces the submit_result tool to
    slot 0, user tools fill the rest of the tools= kwarg, mcp_servers
    rides in its own kwarg. None of the three buckets shadow each
    other.
    """
    from airframe import McpServerRef

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        # Drive submit_result so _build_result returns the structured payload.
        submit_entry = next(
            t for t in mock_sdk["captured_tools"] if t["name"] == SUBMIT_RESULT_TOOL
        )
        await submit_entry["handler"](_Schema(summary="ok", count=1), MagicMock())
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session(
        tools=[_build_tool()],
        mcp_servers=[McpServerRef(name="external", transport="http", url="https://e")],
    )
    try:
        result = await sess.execute("brief me", schema=_Schema)
    finally:
        await sess.close()

    create_call = mock_sdk["client"].create_session.await_args_list[0]
    tools = create_call.kwargs["tools"]
    # submit_result + add (user tool).
    assert len(tools) == 2
    captured_order = [t["name"] for t in mock_sdk["captured_tools"]]
    assert captured_order.index(SUBMIT_RESULT_TOOL) < captured_order.index("add")
    # External MCP server rides separately and doesn't bleed into tools=.
    assert create_call.kwargs["mcp_servers"] == {"external": {"type": "http", "url": "https://e"}}
    # Structured-output round-trip succeeded.
    assert result.structured == {"summary": "ok", "count": 1}


async def test_no_mcp_servers_omits_kwarg(mock_sdk: dict[str, Any]) -> None:
    """No ``mcp_servers=`` → the kwarg is never passed to create_session."""
    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=_mcp_send_factory(mock_sdk))

    rt = CopilotRuntime()
    sess = rt.session()
    try:
        await sess.execute("anything")
    finally:
        await sess.close()

    create_call = mock_sdk["client"].create_session.await_args_list[0]
    assert "mcp_servers" not in create_call.kwargs


async def test_mcp_refs_change_between_sessions_rebuilds_session(
    mock_sdk: dict[str, Any],
) -> None:
    """Switching ``mcp_servers=`` invalidates the cached CopilotSession.

    ``mcp_servers`` is baked at :meth:`create_session` time on Copilot,
    just like ``tools=`` — the fingerprint must join the cache key so
    a refs-change forces a destroy + rebuild.
    """
    from airframe import McpServerRef

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=_mcp_send_factory(mock_sdk))

    rt = CopilotRuntime()
    sess1 = rt.session(mcp_servers=[McpServerRef(name="one", transport="stdio", command=["x"])])
    try:
        await sess1.execute("turn 1")
    finally:
        await sess1.close()
    first_create_count = mock_sdk["client"].create_session.await_count

    sess2 = rt.session(mcp_servers=[McpServerRef(name="two", transport="stdio", command=["x"])])
    try:
        await sess2.execute("turn 2")
    finally:
        await sess2.close()

    assert mock_sdk["client"].create_session.await_count > first_create_count


async def test_mcp_capability_flags_final_truth() -> None:
    """Sanity-check the Iteration C matrix at the adapter level."""
    rt = CopilotRuntime()
    assert rt.supports(Feature.TOOLS_MCP_STDIO)
    assert rt.supports(Feature.TOOLS_MCP_HTTP)
    assert not rt.supports(Feature.TOOLS_MCP_SSE)
    assert not rt.supports(Feature.TOOLS_MCP_IN_PROCESS)


# ---------------------------------------------------------------------------
# Permission callback (Phase 5 Iteration B)
# ---------------------------------------------------------------------------


class _FakeCopilotPermissionRequest:
    """Minimal stand-in for copilot.session.PermissionRequest in unit tests."""

    def __init__(
        self,
        *,
        tool_name: str = "",
        tool_title: str | None = None,
        kind: str = "tool_use",
        tool_args: Any = None,
        args: Any = None,
        reason: str | None = None,
        tool_description: str | None = None,
        intention: str | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.tool_title = tool_title
        self.kind = kind
        self.tool_args = tool_args
        self.args = args
        self.reason = reason
        self.tool_description = tool_description
        self.intention = intention


async def test_permission_callback_replaces_approve_all_handler(
    mock_sdk: dict[str, Any],
) -> None:
    """on_permission= overrides the default PermissionHandler.approve_all."""
    from airframe import PermissionDecision, PermissionRequest

    received: list[PermissionRequest] = []

    class _RecordingCallback:
        async def handle(self, request: PermissionRequest) -> PermissionDecision:
            received.append(request)
            return "allow"

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session(on_permission=_RecordingCallback())
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    create_call = mock_sdk["client"].create_session.await_args_list[0]
    handler = create_call.kwargs["on_permission_request"]
    # The handler isn't the default approve_all anymore — it's
    # airframe's wrapper. Invoke it directly to verify the mapping.
    from copilot.session import PermissionRequestResult

    result = await handler(
        _FakeCopilotPermissionRequest(
            tool_name="read_file",
            tool_args={"path": "/etc/hosts"},
            reason="reads filesystem",
        ),
        {},
    )
    assert isinstance(result, PermissionRequestResult)
    assert result.kind == "approve-once"
    assert len(received) == 1
    req = received[0]
    assert req.tool_name == "read_file"
    assert req.tool_args == {"path": "/etc/hosts"}
    assert req.reason == "reads filesystem"


async def test_permission_callback_deny_maps_to_reject(
    mock_sdk: dict[str, Any],
) -> None:
    """``deny`` → ``"reject"`` PermissionRequestResultKind."""
    from airframe import PermissionDecision, PermissionRequest

    class _DenyAll:
        async def handle(self, request: PermissionRequest) -> PermissionDecision:
            return "deny"

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session(on_permission=_DenyAll())
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    handler = mock_sdk["client"].create_session.await_args_list[0].kwargs["on_permission_request"]
    result = await handler(_FakeCopilotPermissionRequest(tool_name="x"), {})
    assert result.kind == "reject"


async def test_permission_callback_defer_maps_to_user_not_available(
    mock_sdk: dict[str, Any],
) -> None:
    """``defer`` → ``"user-not-available"`` — Copilot's default policy takes over."""
    from airframe import PermissionDecision, PermissionRequest

    class _Defer:
        async def handle(self, request: PermissionRequest) -> PermissionDecision:
            return "defer"

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session(on_permission=_Defer())
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    handler = mock_sdk["client"].create_session.await_args_list[0].kwargs["on_permission_request"]
    result = await handler(_FakeCopilotPermissionRequest(tool_name="x"), {})
    assert result.kind == "user-not-available"


async def test_no_permission_callback_keeps_approve_all_default(
    mock_sdk: dict[str, Any],
) -> None:
    """Without on_permission= the adapter uses PermissionHandler.approve_all."""

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

    handler = mock_sdk["client"].create_session.await_args_list[0].kwargs["on_permission_request"]
    # The mock fixture replaced PermissionHandler.approve_all with a
    # lambda; check identity against what the fixture installed
    # (anything that's not the airframe wrapper would pass — we just
    # care that the user's callback isn't being routed through).
    assert handler is not None
    assert not asyncio.iscoroutinefunction(handler) or handler.__name__ == "<lambda>"


async def test_permission_callback_change_between_sessions_rebuilds(
    mock_sdk: dict[str, Any],
) -> None:
    """Switching ``on_permission=`` invalidates the cached session.

    on_permission_request is baked at create_session time; a callback
    swap must force a session rebuild.
    """
    from airframe import PermissionDecision, PermissionRequest

    class _Cb1:
        async def handle(self, request: PermissionRequest) -> PermissionDecision:
            return "allow"

    class _Cb2:
        async def handle(self, request: PermissionRequest) -> PermissionDecision:
            return "deny"

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("r")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess1 = rt.session(on_permission=_Cb1())
    try:
        await sess1.execute("turn 1")
    finally:
        await sess1.close()
    first_count = mock_sdk["client"].create_session.await_count

    sess2 = rt.session(on_permission=_Cb2())
    try:
        await sess2.execute("turn 2")
    finally:
        await sess2.close()

    assert mock_sdk["client"].create_session.await_count > first_count


def test_copilot_permission_fingerprint_distinguishes_identity() -> None:
    """The Copilot session also uses identity-based fingerprinting."""
    from airframe import PermissionDecision, PermissionRequest
    from airframe.adapters.copilot import _permission_fingerprint

    class _Cb:
        async def handle(self, request: PermissionRequest) -> PermissionDecision:
            return "allow"

    a, b = _Cb(), _Cb()
    assert _permission_fingerprint(None) == "__no_permission__"
    assert _permission_fingerprint(a) != _permission_fingerprint(b)
    assert _permission_fingerprint(a) == _permission_fingerprint(a)


# ---------------------------------------------------------------------------
# Lifecycle hooks (Phase 5 Iteration C)
# ---------------------------------------------------------------------------


class _FakeSessionStartData:
    def __init__(self, *, session_id: str = "live-sess-id") -> None:
        self.session_id = session_id


class _FakeSessionShutdownData:
    def __init__(self) -> None:
        pass


class _FakeUserMessageData:
    def __init__(self, *, content: str = "") -> None:
        self.content = content


class _FakeSessionCompactionStartData:
    def __init__(self) -> None:
        pass


def _patch_hook_event_types(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install fake hook-related event-data classes so isinstance() dispatch
    in ``_on_copilot_hook_event`` recognises them.

    The standard ``mock_sdk`` fixture monkeypatches the
    ``ToolExecution*`` / ``AssistantUsage*`` family; hook tests also
    need ``SessionStartData`` / ``SessionShutdownData`` /
    ``UserMessageData`` / ``SessionCompactionStartData``.
    """
    from copilot.generated import session_events as se_mod

    monkeypatch.setattr(se_mod, "SessionStartData", _FakeSessionStartData)
    monkeypatch.setattr(se_mod, "SessionShutdownData", _FakeSessionShutdownData)
    monkeypatch.setattr(se_mod, "UserMessageData", _FakeUserMessageData)
    monkeypatch.setattr(se_mod, "SessionCompactionStartData", _FakeSessionCompactionStartData)


def test_copilot_runtime_declares_lifecycle_hooks() -> None:
    """LIFECYCLE_HOOKS is True at the runtime level."""
    rt = CopilotRuntime()
    assert rt.supports(Feature.LIFECYCLE_HOOKS)


def test_copilot_emittable_hook_kinds_matches_plan() -> None:
    """The seven kinds Copilot can emit; ``rate_limit`` is excluded
    because Copilot's SDK has no explicit rate-limit event."""
    assert (
        frozenset(
            {
                "session_start",
                "session_end",
                "user_prompt_submit",
                "pre_tool_use",
                "post_tool_use",
                "tool_failure",
                "pre_compact",
            }
        )
        == CopilotRuntime.EMITTABLE_HOOK_KINDS
    )


async def test_on_event_subscribes_hook_handler_alongside_capture(
    mock_sdk: dict[str, Any],
) -> None:
    """on_event= installs a second session.on() handler beside the
    capture handler — both fire for the same event stream."""
    from airframe.hooks import HookEvent

    events: list[HookEvent] = []

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session(on_event=events.append)
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    # The capture handler + the hook-event handler — two subscriptions
    # were installed on session.on() during _ensure_session.
    # After close(), both unsubscribe — handlers list is empty.
    assert mock_sdk["handlers"] == []


async def test_on_event_session_start_translates_from_native_event(
    mock_sdk: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """SessionStartData → session_start HookEvent with model in payload."""
    from airframe.hooks import HookEvent

    _patch_hook_event_types(monkeypatch)
    events: list[HookEvent] = []

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeSessionStartData()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session(on_event=events.append)
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    starts = [e for e in events if e.kind == "session_start"]
    assert len(starts) == 1
    assert "model" in starts[0].payload


async def test_on_event_user_message_translates_to_user_prompt_submit(
    mock_sdk: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """UserMessageData → user_prompt_submit with prompt + length payload."""
    from airframe.hooks import HookEvent

    _patch_hook_event_types(monkeypatch)
    events: list[HookEvent] = []

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUserMessageData(content="hi there")))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session(on_event=events.append)
    try:
        await sess.execute("hi there")
    finally:
        await sess.close()

    prompts = [e for e in events if e.kind == "user_prompt_submit"]
    assert len(prompts) == 1
    assert prompts[0].payload["prompt"] == "hi there"
    assert prompts[0].payload["length"] == len("hi there")


async def test_on_event_tool_execution_translates_to_pre_and_post_tool_use(
    mock_sdk: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """ToolExecutionStart → pre_tool_use; ToolExecutionComplete(success=True)
    → post_tool_use; the tool_name + tool_call_id ride the payload."""
    from airframe.hooks import HookEvent

    _patch_hook_event_types(monkeypatch)
    events: list[HookEvent] = []

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(
            mock_sdk["handlers"],
            _FakeEvent(
                _FakeToolStart(
                    tool_call_id="tc-1",
                    tool_name="add",
                    arguments={"a": 1, "b": 2},
                )
            ),
        )
        _fire(
            mock_sdk["handlers"],
            _FakeEvent(
                _FakeToolComplete(
                    tool_call_id="tc-1",
                    success=True,
                    result=_FakeToolResult(content="3"),
                )
            ),
        )
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("3")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session(on_event=events.append)
    try:
        await sess.execute("call add")
    finally:
        await sess.close()

    pre = [e for e in events if e.kind == "pre_tool_use"]
    post = [e for e in events if e.kind == "post_tool_use"]
    assert len(pre) == 1
    assert pre[0].payload["tool_name"] == "add"
    assert pre[0].payload["tool_call_id"] == "tc-1"
    assert len(post) == 1
    assert post[0].payload["tool_call_id"] == "tc-1"
    assert post[0].payload["success"] is True


async def test_on_event_tool_failure_translates_to_tool_failure_kind(
    mock_sdk: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """ToolExecutionComplete(success=False) → tool_failure HookEvent."""
    from airframe.hooks import HookEvent

    _patch_hook_event_types(monkeypatch)
    events: list[HookEvent] = []

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(
            mock_sdk["handlers"],
            _FakeEvent(_FakeToolStart(tool_call_id="t-1", tool_name="add")),
        )
        _fire(
            mock_sdk["handlers"],
            _FakeEvent(
                _FakeToolComplete(
                    tool_call_id="t-1",
                    success=False,
                    error=_FakeToolError(message="boom", code="E_X"),
                )
            ),
        )
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("apologies")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session(on_event=events.append)
    try:
        await sess.execute("call add")
    finally:
        await sess.close()

    failures = [e for e in events if e.kind == "tool_failure"]
    posts = [e for e in events if e.kind == "post_tool_use"]
    assert len(failures) == 1
    assert failures[0].payload["success"] is False
    assert posts == []


async def test_on_event_pre_compact_translates_from_native(
    mock_sdk: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """SessionCompactionStartData → pre_compact HookEvent."""
    from airframe.hooks import HookEvent

    _patch_hook_event_types(monkeypatch)
    events: list[HookEvent] = []

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeSessionCompactionStartData()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session(on_event=events.append)
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    compacts = [e for e in events if e.kind == "pre_compact"]
    assert len(compacts) == 1


async def test_on_event_submit_result_tool_calls_are_suppressed(
    mock_sdk: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The forced submit_result tool is structured-output plumbing —
    hook emission suppresses it consistently with the streaming events."""
    from airframe.hooks import HookEvent

    _patch_hook_event_types(monkeypatch)
    events: list[HookEvent] = []

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        # Drive the submit_result handler so _build_result yields a
        # structured payload.
        submit_entry = next(
            t for t in mock_sdk["captured_tools"] if t["name"] == SUBMIT_RESULT_TOOL
        )
        await submit_entry["handler"](_Schema(summary="ok", count=1), MagicMock())
        _fire(
            mock_sdk["handlers"],
            _FakeEvent(
                _FakeToolStart(
                    tool_call_id="submit-1",
                    tool_name=SUBMIT_RESULT_TOOL,
                    arguments={"summary": "ok", "count": 1},
                )
            ),
        )
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session(on_event=events.append)
    try:
        await sess.execute("brief me", schema=_Schema)
    finally:
        await sess.close()

    pre = [e for e in events if e.kind == "pre_tool_use"]
    assert pre == []


async def test_close_synthesises_session_end_when_native_event_missing(
    mock_sdk: dict[str, Any],
) -> None:
    """If the SDK never fires SessionShutdownData, close() must still
    emit a synthetic session_end so consumer observers see the
    end-of-life marker."""
    from airframe.hooks import HookEvent

    events: list[HookEvent] = []

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session(on_event=events.append)
    await sess.execute("hi")
    # Before close: no session_end yet.
    assert [e for e in events if e.kind == "session_end"] == []
    await sess.close()
    ends = [e for e in events if e.kind == "session_end"]
    assert len(ends) == 1


async def test_close_session_end_is_idempotent(mock_sdk: dict[str, Any]) -> None:
    """Multiple close() calls fire session_end exactly once."""
    from airframe.hooks import HookEvent

    events: list[HookEvent] = []

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session(on_event=events.append)
    await sess.execute("hi")
    await sess.close()
    await sess.close()
    await sess.close()
    ends = [e for e in events if e.kind == "session_end"]
    assert len(ends) == 1


async def test_close_skips_session_end_if_native_already_fired(
    mock_sdk: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """If SessionShutdownData fired during the turn, close() must NOT
    fire a second synthetic session_end."""
    from airframe.hooks import HookEvent

    _patch_hook_event_types(monkeypatch)
    events: list[HookEvent] = []

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeSessionShutdownData()))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session(on_event=events.append)
    await sess.execute("hi")
    await sess.close()
    ends = [e for e in events if e.kind == "session_end"]
    assert len(ends) == 1


async def test_no_on_event_skips_hook_subscription(mock_sdk: dict[str, Any]) -> None:
    """No on_event= → only the capture handler is installed."""

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session()
    # During the turn there's exactly one handler (the capture slot);
    # the hook subscription is absent because on_event=None.

    handlers_during_turn: list[int] = []

    async def probe_send(prompt: str, *, timeout: float, **_: Any) -> None:
        handlers_during_turn.append(len(mock_sdk["handlers"]))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=probe_send)
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    assert handlers_during_turn == [1]


async def test_on_event_observer_that_raises_does_not_break_session(
    mock_sdk: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the observer raises, the session continues — _fire_hook_event
    catches everything except KeyboardInterrupt/SystemExit."""
    from airframe.hooks import HookEvent

    _patch_hook_event_types(monkeypatch)
    calls = {"n": 0}

    def boom(event: HookEvent) -> None:
        calls["n"] += 1
        raise RuntimeError("observer broke")

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeSessionStartData()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)

    rt = CopilotRuntime()
    sess = rt.session(on_event=boom)
    try:
        # The turn completes despite the raising observer.
        result = await sess.execute("hi")
    finally:
        await sess.close()

    assert result.text == "ok"
    assert calls["n"] >= 1


# ---------------------------------------------------------------------------
# Budget caps (Phase 5 Iteration D)
# ---------------------------------------------------------------------------


def test_copilot_runtime_declares_budget_usd_cap_only() -> None:
    """Copilot flips BUDGET_USD_CAP True; BUDGET_TURN_CAP stays False.

    The Copilot SDK caps internal turns at the CLI level via the
    runtime's ``--max-turns`` config; exposing a user-facing
    ``max_turns=`` on per-execute would mislead consumers, so we
    decline.
    """
    rt = CopilotRuntime()
    assert rt.supports(Feature.BUDGET_USD_CAP)
    assert not rt.supports(Feature.BUDGET_TURN_CAP)


async def test_max_turns_raises_unsupported_feature(mock_sdk: dict[str, Any]) -> None:
    """``max_turns=`` always raises on Copilot (capability declined)."""
    from airframe.errors import UnsupportedFeatureError

    rt = CopilotRuntime()
    sess = rt.session()
    try:
        with pytest.raises(UnsupportedFeatureError) as exc_info:
            await sess.execute("hi", max_turns=5)
        assert exc_info.value.feature == Feature.BUDGET_TURN_CAP
    finally:
        await sess.close()


async def test_max_budget_usd_cap_raises_when_cumulative_cost_reached(
    mock_sdk: dict[str, Any],
) -> None:
    """Once cumulative cost >= max_budget_usd, the next execute()
    raises RuntimeBudgetExceededError(kind='usd')."""
    from airframe.errors import RuntimeBudgetExceededError

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage(cost=0.03)))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)
    rt = CopilotRuntime()
    sess = rt.session()
    try:
        await sess.execute("turn 1", max_budget_usd=0.05)
        await sess.execute("turn 2", max_budget_usd=0.05)
        with pytest.raises(RuntimeBudgetExceededError) as exc_info:
            await sess.execute("turn 3", max_budget_usd=0.05)
    finally:
        await sess.close()
    err = exc_info.value
    assert err.kind == "usd"
    assert err.cap == 0.05


async def test_budget_caps_none_open_cleanly(mock_sdk: dict[str, Any]) -> None:
    """No cap → no enforcement; turns run freely."""

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage(cost=1.0)))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)
    rt = CopilotRuntime()
    sess = rt.session()
    try:
        await sess.execute("a")
        await sess.execute("b")
        await sess.execute("c")
    finally:
        await sess.close()


async def test_stream_honours_max_budget_usd_cap(mock_sdk: dict[str, Any]) -> None:
    """Stream path uses the same enforce — exhausted cap raises."""
    from airframe.errors import RuntimeBudgetExceededError

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage(cost=0.02)))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)
    rt = CopilotRuntime()
    sess = rt.session()
    try:
        async for _ in sess.stream("a", max_budget_usd=0.01):
            pass
        with pytest.raises(RuntimeBudgetExceededError):
            async for _ in sess.stream("b", max_budget_usd=0.01):
                pass
    finally:
        await sess.close()


# ---------------------------------------------------------------------------
# Provider options (v0.5.0-readiness — CopilotOptions wired)
# ---------------------------------------------------------------------------


async def test_copilot_options_available_tools_lands_on_create_session(
    mock_sdk: dict[str, Any],
) -> None:
    from airframe import CopilotOptions

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)
    rt = CopilotRuntime()
    sess = rt.session(provider_options=CopilotOptions(available_tools=("read", "shell")))
    try:
        await sess.execute("hi")
    finally:
        await sess.close()
    call = mock_sdk["client"].create_session.await_args_list[0]
    assert call.kwargs["available_tools"] == ["read", "shell"]


async def test_copilot_options_excluded_tools_lands_on_create_session(
    mock_sdk: dict[str, Any],
) -> None:
    from airframe import CopilotOptions

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)
    rt = CopilotRuntime()
    sess = rt.session(provider_options=CopilotOptions(excluded_tools=("write",)))
    try:
        await sess.execute("hi")
    finally:
        await sess.close()
    call = mock_sdk["client"].create_session.await_args_list[0]
    assert call.kwargs["excluded_tools"] == ["write"]


async def test_copilot_options_skill_directories_and_working_directory(
    mock_sdk: dict[str, Any],
) -> None:
    from airframe import CopilotOptions

    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)
    rt = CopilotRuntime()
    sess = rt.session(
        provider_options=CopilotOptions(
            skill_directories=("/tmp/skills", "/opt/skills"),
            working_directory="/workspaces/foo",
        )
    )
    try:
        await sess.execute("hi")
    finally:
        await sess.close()
    call = mock_sdk["client"].create_session.await_args_list[0]
    assert call.kwargs["skill_directories"] == ["/tmp/skills", "/opt/skills"]
    assert call.kwargs["working_directory"] == "/workspaces/foo"


async def test_copilot_options_default_omits_kwargs(mock_sdk: dict[str, Any]) -> None:
    """No ``provider_options=`` → none of the CopilotOptions kwargs land."""

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
    call = mock_sdk["client"].create_session.await_args_list[0]
    assert "available_tools" not in call.kwargs
    assert "excluded_tools" not in call.kwargs
    assert "skill_directories" not in call.kwargs
    assert "working_directory" not in call.kwargs


async def test_copilot_options_wrong_namespace_raises_unsupported_feature(
    mock_sdk: dict[str, Any],
) -> None:
    from airframe import ClaudeOptions
    from airframe.errors import UnsupportedFeatureError

    rt = CopilotRuntime()
    with pytest.raises(UnsupportedFeatureError) as exc_info:
        rt.session(provider_options=ClaudeOptions())
    assert "ClaudeOptions" in str(exc_info.value)
    assert "CopilotOptions" in str(exc_info.value)


# --- Native (vendor-hosted) tools: WEB_FETCH → fetch_webpage ---------------


def _native_fake_send(mock_sdk: dict[str, Any]) -> None:
    async def fake_send(prompt: str, *, timeout: float, **_: Any) -> None:
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeUsage()))
        _fire(mock_sdk["handlers"], _FakeEvent(_FakeAssistantMessage("ok")))

    mock_sdk["session"].send_and_wait = AsyncMock(side_effect=fake_send)


async def test_native_web_fetch_with_allowlist_unions_fetch_webpage(
    mock_sdk: dict[str, Any],
) -> None:
    """A native WEB_FETCH request unions ``fetch_webpage`` into an existing
    allowlist rather than replacing it."""
    from airframe import NativeTool
    from airframe.options import CopilotOptions

    _native_fake_send(mock_sdk)
    rt = CopilotRuntime()
    sess = rt.session(
        native_tools=[NativeTool.web_fetch()],
        provider_options=CopilotOptions(available_tools=("read",)),
    )
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    call = mock_sdk["client"].create_session.await_args_list[0]
    assert set(call.kwargs["available_tools"]) == {"read", "fetch_webpage"}


async def test_native_web_fetch_removes_tool_from_denylist(
    mock_sdk: dict[str, Any],
) -> None:
    """An explicit native request wins over a provider-options denylist."""
    from airframe import NativeTool
    from airframe.options import CopilotOptions

    _native_fake_send(mock_sdk)
    rt = CopilotRuntime()
    sess = rt.session(
        native_tools=[NativeTool.web_fetch()],
        provider_options=CopilotOptions(excluded_tools=("fetch_webpage",)),
    )
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    call = mock_sdk["client"].create_session.await_args_list[0]
    # fetch_webpage was the only denied tool → the kwarg drops out entirely.
    assert "excluded_tools" not in call.kwargs


async def test_native_web_fetch_no_options_leaves_available_tools_unset(
    mock_sdk: dict[str, Any],
) -> None:
    """With no allowlist, the hosted built-in is on by default — we must not
    introduce a restrictive allowlist that disables everything else."""
    from airframe import NativeTool

    _native_fake_send(mock_sdk)
    rt = CopilotRuntime()
    sess = rt.session(native_tools=[NativeTool.web_fetch()])
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    call = mock_sdk["client"].create_session.await_args_list[0]
    assert "available_tools" not in call.kwargs
