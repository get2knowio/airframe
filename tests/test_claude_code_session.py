"""Unit tests for :class:`ClaudeCodeSession`.

Phase 1 Iteration D — second per-vendor session (after OpenAI-compat).
Mocks ``claude_agent_sdk`` at the boundary so we exercise the
session's connect cache, the streaming translation of Anthropic
``StreamEvent`` deltas, the resume= path through
``ClaudeAgentOptions.resume``, and cancellation via
``ClaudeSDKClient.interrupt()`` without spawning a subprocess.

Live-vendor probes belong in ``airframe.testing.integration``
(Phase 1 work).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from airframe.adapters.claude_code import ClaudeCodeRuntime, ClaudeCodeSession
from airframe.errors import RuntimeCancelledError
from airframe.events import ReasoningDelta, TextDelta, TurnComplete
from airframe.features import Feature
from airframe.protocol import AgentSession, ProviderModel


class _Schema(BaseModel):
    summary: str
    count: int


# ---------------------------------------------------------------------------
# Fake SDK types — real classes so isinstance() checks work
# ---------------------------------------------------------------------------


class _FakeResultMessage:
    def __init__(
        self,
        *,
        is_error: bool = False,
        stop_reason: str | None = "end_turn",
        result: str | None = "",
        structured_output: Any = None,
        total_cost_usd: float | None = 0.001,
        usage: dict[str, Any] | None = None,
        session_id: str = "sess-123",
        subtype: str | None = None,
        errors: list[str] | None = None,
    ) -> None:
        self.is_error = is_error
        self.stop_reason = stop_reason
        self.result = result
        self.structured_output = structured_output
        self.total_cost_usd = total_cost_usd
        self.usage = usage or {
            "input_tokens": 10,
            "output_tokens": 20,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        self.session_id = session_id
        self.subtype = subtype
        self.errors = errors


class _FakeStreamEvent:
    def __init__(self, event: dict[str, Any]) -> None:
        self.uuid = "u"
        self.session_id = "sess-123"
        self.event = event
        self.parent_tool_use_id = None


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeAssistantMessage:
    def __init__(self, content: list[Any], session_id: str = "sess-123") -> None:
        self.content = content
        self.model = "claude-haiku-4-5"
        self.session_id = session_id


class _FakeToolUseBlock:
    def __init__(self, id: str, name: str, input: dict[str, Any]) -> None:
        self.id = id
        self.name = name
        self.input = input


class _FakeToolResultBlock:
    def __init__(
        self,
        tool_use_id: str,
        content: Any = None,
        is_error: bool | None = None,
    ) -> None:
        self.tool_use_id = tool_use_id
        self.content = content
        self.is_error = is_error


class _FakeUserMessage:
    def __init__(self, content: Any, session_id: str = "sess-123") -> None:
        self.content = content
        self.session_id = session_id


# ---------------------------------------------------------------------------
# Fixture: patch the SDK symbols the adapter imports lazily
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sdk(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the claude_agent_sdk symbols ClaudeCodeSession imports."""
    import claude_agent_sdk as sdk

    client = MagicMock()
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.query = AsyncMock()
    client.interrupt = AsyncMock()

    captured_options: list[dict[str, Any]] = []

    def capturing_options(**kwargs: Any) -> MagicMock:
        captured_options.append(kwargs)
        return MagicMock()

    factory = MagicMock(return_value=client)
    monkeypatch.setattr(sdk, "ClaudeSDKClient", factory)
    monkeypatch.setattr(sdk, "ClaudeAgentOptions", capturing_options)
    monkeypatch.setattr(sdk, "ResultMessage", _FakeResultMessage)
    monkeypatch.setattr(sdk, "AssistantMessage", _FakeAssistantMessage)
    monkeypatch.setattr(sdk, "StreamEvent", _FakeStreamEvent)
    monkeypatch.setattr(sdk, "TextBlock", _FakeTextBlock)
    monkeypatch.setattr(sdk, "ToolUseBlock", _FakeToolUseBlock)
    monkeypatch.setattr(sdk, "ToolResultBlock", _FakeToolResultBlock)
    monkeypatch.setattr(sdk, "UserMessage", _FakeUserMessage)

    # create_sdk_mcp_server is invoked at connect time when tools= is
    # passed; return a sentinel the adapter can pass through to
    # ClaudeAgentOptions.mcp_servers. The @tool decorator is called
    # once per FunctionTool — return a stable identifier so assertions
    # can inspect which tools were registered.
    server_calls: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []

    def fake_create_server(*, name: str, tools: list[Any]) -> Any:
        server_calls.append({"name": name, "tools": tools})
        return {"_server": name, "_tools": tools}

    def fake_tool_decorator(name: str, description: str, input_schema: Any) -> Any:
        def _wrap(func: Any) -> Any:
            tool_calls.append({"name": name, "description": description, "schema": input_schema})
            func._airframe_tool_name = name  # type: ignore[attr-defined]
            return func

        return _wrap

    monkeypatch.setattr(sdk, "create_sdk_mcp_server", fake_create_server)
    monkeypatch.setattr(sdk, "tool", fake_tool_decorator)

    return {
        "client": client,
        "factory": factory,
        "options_kwargs": captured_options,
        "server_calls": server_calls,
        "tool_calls": tool_calls,
    }


# ---------------------------------------------------------------------------
# Factory + capability surface
# ---------------------------------------------------------------------------


async def test_session_factory_returns_bespoke_session(mock_sdk: dict[str, Any]) -> None:
    rt = ClaudeCodeRuntime()
    sess = rt.session()
    try:
        assert isinstance(sess, ClaudeCodeSession)
        assert isinstance(sess, AgentSession)
        # No turn run yet — id is None unless the caller seeded resume=.
        assert sess.id is None
    finally:
        await sess.close()


async def test_streaming_resume_cancel_features_declared() -> None:
    rt = ClaudeCodeRuntime()
    assert rt.supports(Feature.STREAMING)
    assert rt.supports(Feature.SESSION_RESUME)
    assert rt.supports(Feature.CANCEL)


async def test_resume_seeds_session_id_and_forwards_to_options(
    mock_sdk: dict[str, Any],
) -> None:
    """resume=<id> appears on session.id immediately and on the SDK options at connect."""
    final = _FakeResultMessage(result="resumed reply", session_id="sess-resumed")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session(resume="sess-resumed")
    try:
        assert sess.id == "sess-resumed"  # surfaced before any turn
        await sess.execute("continue please")
    finally:
        await sess.close()

    opts = mock_sdk["options_kwargs"][0]
    assert opts["resume"] == "sess-resumed"


async def test_options_carry_system_and_model_and_partial_messages(
    mock_sdk: dict[str, Any],
) -> None:
    """system_prompt + model + include_partial_messages all reach ClaudeAgentOptions."""
    final = _FakeResultMessage(result="ok")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session(
        system="be brief",
        model=ProviderModel("claude", "claude-sonnet-4-6"),
    )
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    opts = mock_sdk["options_kwargs"][0]
    assert opts["system_prompt"] == "be brief"
    assert opts["model"] == "claude-sonnet-4-6"
    assert opts["include_partial_messages"] is True
    assert "resume" not in opts  # fresh session


# ---------------------------------------------------------------------------
# Multi-turn lifecycle
# ---------------------------------------------------------------------------


async def test_id_populated_after_first_turn(mock_sdk: dict[str, Any]) -> None:
    """When the SDK reports a session_id on the ResultMessage, surface it on .id."""
    final = _FakeResultMessage(result="reply", session_id="sess-XYZ")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session()
    try:
        assert sess.id is None
        await sess.execute("first")
        assert sess.id == "sess-XYZ"
    finally:
        await sess.close()


async def test_client_reused_across_turns_with_same_schema(
    mock_sdk: dict[str, Any],
) -> None:
    """Two turns with the same (plain-text) schema fingerprint share one connect."""
    final = _FakeResultMessage(result="r")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session()
    try:
        await sess.execute("a")
        await sess.execute("b")
    finally:
        await sess.close()

    assert mock_sdk["client"].connect.await_count == 1


async def test_client_reconnects_when_schema_changes(mock_sdk: dict[str, Any]) -> None:
    """Schema fingerprint change forces a reconnect — output_format is bake-time."""
    final_plain = _FakeResultMessage(result="plain")
    final_structured = _FakeResultMessage(
        result="", structured_output={"summary": "ok", "count": 1}
    )

    async def make_iter(msg: Any) -> Any:
        async def gen() -> Any:
            yield msg

        return gen()

    call_idx = {"n": 0}

    async def receive_response() -> Any:
        idx = call_idx["n"]
        call_idx["n"] += 1
        if idx == 0:
            yield final_plain
        else:
            yield final_structured

    mock_sdk["client"].receive_response = receive_response

    rt = ClaudeCodeRuntime()
    sess = rt.session()
    try:
        await sess.execute("plain text turn")
        await sess.execute("structured turn", schema=_Schema)
    finally:
        await sess.close()

    # Two distinct schema fingerprints → reconnect between them.
    assert mock_sdk["client"].connect.await_count == 2
    assert mock_sdk["client"].disconnect.await_count >= 1


# ---------------------------------------------------------------------------
# thinking= (Phase 2 Iteration B)
# ---------------------------------------------------------------------------


async def test_thinking_effort_forwarded_as_options_effort(
    mock_sdk: dict[str, Any],
) -> None:
    """thinking=<effort> ends up on ClaudeAgentOptions.effort."""
    final = _FakeResultMessage(result="ok")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session()
    try:
        await sess.execute("plan it", thinking="high")
    finally:
        await sess.close()

    opts = mock_sdk["options_kwargs"][0]
    assert opts["effort"] == "high"
    assert "thinking" not in opts


async def test_thinking_minimal_coerces_to_low(mock_sdk: dict[str, Any]) -> None:
    """Anthropic has no 'minimal' tier — coerce to 'low' with a debug log."""
    final = _FakeResultMessage(result="ok")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session()
    try:
        await sess.execute("hi", thinking="minimal")
    finally:
        await sess.close()

    opts = mock_sdk["options_kwargs"][0]
    assert opts["effort"] == "low"


async def test_thinking_disabled_sets_thinking_config_disabled(
    mock_sdk: dict[str, Any],
) -> None:
    """thinking='disabled' sends ``thinking={"type": "disabled"}`` (explicit off)."""
    final = _FakeResultMessage(result="ok")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session()
    try:
        await sess.execute("hi", thinking="disabled")
    finally:
        await sess.close()

    opts = mock_sdk["options_kwargs"][0]
    assert opts["thinking"] == {"type": "disabled"}
    assert "effort" not in opts


async def test_thinking_dict_sets_budget_tokens(mock_sdk: dict[str, Any]) -> None:
    """``{"budget_tokens": N}`` becomes ``thinking={"type":"enabled","budget_tokens":N}``."""
    final = _FakeResultMessage(result="ok")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session()
    try:
        await sess.execute("hi", thinking={"budget_tokens": 8000})
    finally:
        await sess.close()

    opts = mock_sdk["options_kwargs"][0]
    assert opts["thinking"] == {"type": "enabled", "budget_tokens": 8000}
    assert "effort" not in opts


async def test_thinking_dict_without_budget_raises(mock_sdk: dict[str, Any]) -> None:
    """The dict shape requires an integer ``budget_tokens`` key."""
    from airframe.errors import UnsupportedFeatureError

    rt = ClaudeCodeRuntime()
    sess = rt.session()
    try:
        with pytest.raises(UnsupportedFeatureError) as exc_info:
            await sess.execute("hi", thinking={"wrong_key": 5})
    finally:
        await sess.close()

    assert exc_info.value.feature == Feature.REASONING_BUDGET_TOKENS


async def test_thinking_none_omits_both_knobs(mock_sdk: dict[str, Any]) -> None:
    final = _FakeResultMessage(result="ok")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session()
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    opts = mock_sdk["options_kwargs"][0]
    assert "effort" not in opts
    assert "thinking" not in opts


async def test_thinking_change_between_turns_reconnects(
    mock_sdk: dict[str, Any],
) -> None:
    """``effort`` is baked at connect time — change forces a reconnect."""
    final = _FakeResultMessage(result="ok")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session()
    try:
        await sess.execute("a", thinking="low")
        await sess.execute("b", thinking="high")
    finally:
        await sess.close()

    assert mock_sdk["client"].connect.await_count == 2
    efforts = [opts.get("effort") for opts in mock_sdk["options_kwargs"]]
    assert efforts == ["low", "high"]


async def test_same_thinking_across_turns_reuses_client(
    mock_sdk: dict[str, Any],
) -> None:
    final = _FakeResultMessage(result="ok")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session()
    try:
        await sess.execute("a", thinking="medium")
        await sess.execute("b", thinking="medium")
    finally:
        await sess.close()

    assert mock_sdk["client"].connect.await_count == 1


# ---------------------------------------------------------------------------
# Vision / file input (Phase 2 Iteration C)
# ---------------------------------------------------------------------------


async def test_image_input_appends_read_tool_hint_and_allows_read(
    mock_sdk: dict[str, Any],
) -> None:
    """Image attachments add a Read-tool hint to the prompt and Read to allowed_tools."""
    captured: dict[str, Any] = {}

    async def fake_query(prompt: Any) -> None:
        captured["prompt"] = prompt

    mock_sdk["client"].query = AsyncMock(side_effect=fake_query)

    final = _FakeResultMessage(result="ok")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    from airframe.inputs import ImageInput

    rt = ClaudeCodeRuntime()
    sess = rt.session()
    try:
        await sess.execute(["look:", ImageInput(path="/tmp/x.png")])
    finally:
        await sess.close()

    assert "Attached files (use the Read tool to access):" in captured["prompt"]
    assert "- /tmp/x.png" in captured["prompt"]
    opts = mock_sdk["options_kwargs"][0]
    assert opts["allowed_tools"] == ["Read"]


async def test_file_input_appends_hint_with_media_type(
    mock_sdk: dict[str, Any],
) -> None:
    """FileInput hint includes media_type when provided."""
    captured: dict[str, Any] = {}

    async def fake_query(prompt: Any) -> None:
        captured["prompt"] = prompt

    mock_sdk["client"].query = AsyncMock(side_effect=fake_query)

    final = _FakeResultMessage(result="ok")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    from airframe.inputs import FileInput

    rt = ClaudeCodeRuntime()
    sess = rt.session()
    try:
        await sess.execute(
            ["summarise:", FileInput(path="/tmp/spec.pdf", media_type="application/pdf")]
        )
    finally:
        await sess.close()

    assert "- /tmp/spec.pdf (application/pdf)" in captured["prompt"]


async def test_plain_string_keeps_options_clean(mock_sdk: dict[str, Any]) -> None:
    """No attachments → allowed_tools is NOT set (preserves the default behaviour)."""
    final = _FakeResultMessage(result="ok")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session()
    try:
        await sess.execute("just text")
    finally:
        await sess.close()

    opts = mock_sdk["options_kwargs"][0]
    assert "allowed_tools" not in opts


async def test_attachments_change_between_turns_reconnects(
    mock_sdk: dict[str, Any],
) -> None:
    """has_attachments=False → True between turns rebuilds the client (allowed_tools changes)."""
    final = _FakeResultMessage(result="ok")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    from airframe.inputs import ImageInput

    rt = ClaudeCodeRuntime()
    sess = rt.session()
    try:
        await sess.execute("plain")
        await sess.execute(["with image:", ImageInput(path="/tmp/x.png")])
    finally:
        await sess.close()

    assert mock_sdk["client"].connect.await_count == 2
    assert "allowed_tools" not in mock_sdk["options_kwargs"][0]
    assert mock_sdk["options_kwargs"][1]["allowed_tools"] == ["Read"]


async def test_image_bytes_raises_with_helpful_message(
    mock_sdk: dict[str, Any],
) -> None:
    """Claude's Read-tool fallback is path-only; bytes_= raises with a write-to-disk hint."""
    from airframe.errors import UnsupportedFeatureError
    from airframe.inputs import ImageInput

    rt = ClaudeCodeRuntime()
    sess = rt.session()
    try:
        with pytest.raises(UnsupportedFeatureError) as exc_info:
            await sess.execute(["x", ImageInput(bytes_=b"raw")])
    finally:
        await sess.close()
    assert exc_info.value.feature == Feature.VISION_INPUT
    assert "tempfile" in str(exc_info.value)


async def test_image_url_raises_with_helpful_message(
    mock_sdk: dict[str, Any],
) -> None:
    from airframe.errors import UnsupportedFeatureError
    from airframe.inputs import ImageInput

    rt = ClaudeCodeRuntime()
    sess = rt.session()
    try:
        with pytest.raises(UnsupportedFeatureError) as exc_info:
            await sess.execute(["x", ImageInput(url="https://example.com/x.png")])
    finally:
        await sess.close()
    assert exc_info.value.feature == Feature.VISION_INPUT
    assert "path=" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


async def test_stream_yields_text_deltas_from_stream_events(
    mock_sdk: dict[str, Any],
) -> None:
    """content_block_delta events with text_delta become TextDelta."""
    events = [
        _FakeStreamEvent(
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hel"}}
        ),
        _FakeStreamEvent(
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "lo"}}
        ),
        _FakeResultMessage(result="Hello", stop_reason="end_turn"),
    ]

    async def fake_receive() -> Any:
        for e in events:
            yield e

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
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
    assert turn_events[0].result.finish == "end_turn"


async def test_stream_yields_reasoning_deltas(mock_sdk: dict[str, Any]) -> None:
    """thinking_delta events become ReasoningDelta."""
    events = [
        _FakeStreamEvent(
            {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "let me think"},
            }
        ),
        _FakeStreamEvent(
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "done"}}
        ),
        _FakeResultMessage(result="done"),
    ]

    async def fake_receive() -> Any:
        for e in events:
            yield e

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session()
    yielded: list[Any] = []
    try:
        async for ev in sess.stream("think hard"):
            yielded.append(ev)
    finally:
        await sess.close()

    reasoning = [e for e in yielded if isinstance(e, ReasoningDelta)]
    assert [e.text for e in reasoning] == ["let me think"]


async def test_stream_falls_back_to_assistant_message_textblocks(
    mock_sdk: dict[str, Any],
) -> None:
    """If StreamEvents don't deliver text, TextBlocks on AssistantMessage do."""
    msg = _FakeAssistantMessage(content=[_FakeTextBlock("aggregated reply")])
    events = [msg, _FakeResultMessage(result="aggregated reply")]

    async def fake_receive() -> Any:
        for e in events:
            yield e

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session()
    yielded: list[Any] = []
    try:
        async for ev in sess.stream("hi"):
            yielded.append(ev)
    finally:
        await sess.close()

    text_events = [e for e in yielded if isinstance(e, TextDelta)]
    assert [e.text for e in text_events] == ["aggregated reply"]


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


async def test_cancel_calls_interrupt_and_raises_runtime_cancelled(
    mock_sdk: dict[str, Any],
) -> None:
    """cancel() during execute() interrupts the SDK and surfaces RuntimeCancelledError."""

    async def hanging_receive() -> Any:
        await asyncio.sleep(5.0)
        yield _FakeResultMessage(result="never")

    mock_sdk["client"].receive_response = hanging_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session()
    try:
        exec_task = asyncio.create_task(sess.execute("hang"))
        # Wait until the inner _do_execute task has connected the client
        # and entered the receive_response loop — that's when _in_flight
        # flips to True and cancel() has something to interrupt.
        for _ in range(50):
            await asyncio.sleep(0)
            if sess._in_flight:  # noqa: SLF001 — invariant probe
                break
        assert sess._in_flight, "execute didn't enter the in-flight state"  # noqa: SLF001
        await sess.cancel()
        with pytest.raises(RuntimeCancelledError):
            await exec_task
    finally:
        await sess.close()

    mock_sdk["client"].interrupt.assert_awaited()


async def test_cancel_when_idle_is_noop(mock_sdk: dict[str, Any]) -> None:
    rt = ClaudeCodeRuntime()
    sess = rt.session()
    try:
        await sess.cancel()
        await sess.cancel()
    finally:
        await sess.close()
    # interrupt() may or may not be called when there's no live client;
    # the contract is just "doesn't raise".


# ---------------------------------------------------------------------------
# close() lifecycle
# ---------------------------------------------------------------------------


async def test_close_disconnects_and_blocks_further_execute(
    mock_sdk: dict[str, Any],
) -> None:
    final = _FakeResultMessage(result="r")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session()
    await sess.execute("warm up")
    await sess.close()
    mock_sdk["client"].disconnect.assert_awaited()
    with pytest.raises(RuntimeError):
        await sess.execute("nope")


async def test_close_is_idempotent(mock_sdk: dict[str, Any]) -> None:
    rt = ClaudeCodeRuntime()
    sess = rt.session()
    await sess.close()
    await sess.close()


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


async def test_tools_register_mcp_server_and_allowed_tools(
    mock_sdk: dict[str, Any],
) -> None:
    """tools= triggers an in-process MCP server and the matching allowed-tools entry."""
    final = _FakeResultMessage(result="ok")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session(tools=[_build_tool()])
    try:
        await sess.execute("what's 17 + 23?")
    finally:
        await sess.close()

    # The MCP server was built once with our one tool.
    assert len(mock_sdk["server_calls"]) == 1
    server_call = mock_sdk["server_calls"][0]
    assert server_call["name"] == "airframe_tools"
    assert len(server_call["tools"]) == 1

    # The @tool decorator captured the FunctionTool's metadata.
    assert len(mock_sdk["tool_calls"]) == 1
    tool_call = mock_sdk["tool_calls"][0]
    assert tool_call["name"] == "add"
    assert tool_call["description"] == "Add two numbers."
    assert tool_call["schema"] is _AddParams

    # ClaudeAgentOptions carries both mcp_servers and allowed_tools.
    opts = mock_sdk["options_kwargs"][0]
    assert "mcp_servers" in opts
    assert "airframe_tools" in opts["mcp_servers"]
    assert opts["allowed_tools"] == ["mcp__airframe_tools__add"]


async def test_tools_handler_invocation_through_mcp_wrapper(
    mock_sdk: dict[str, Any],
) -> None:
    """The @tool wrapper validates args, awaits the handler, returns the MCP envelope."""
    final = _FakeResultMessage(result="done")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session(tools=[_build_tool()])
    try:
        await sess.execute("anything")
    finally:
        await sess.close()

    # Pull the wrapped coroutine out of the server config and invoke it
    # directly — same path the SDK would take when Claude calls the tool.
    server_call = mock_sdk["server_calls"][0]
    wrapped = server_call["tools"][0]
    result = await wrapped({"a": 17, "b": 23})
    assert result == {"content": [{"type": "text", "text": "40.0"}]}


async def test_tools_handler_exception_returns_is_error_envelope(
    mock_sdk: dict[str, Any],
) -> None:
    """A raising handler comes back as ``isError=True`` so the model can recover."""
    from airframe import FunctionTool

    async def boom(_: _AddParams) -> float:
        raise ValueError("the maths fell apart")

    tool = FunctionTool(name="add", description="Add.", params=_AddParams, handler=boom)

    final = _FakeResultMessage(result="apologies")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session(tools=[tool])
    try:
        await sess.execute("call add(1,2)")
    finally:
        await sess.close()

    wrapped = mock_sdk["server_calls"][0]["tools"][0]
    result = await wrapped({"a": 1, "b": 2})
    assert result["isError"] is True
    assert "ValueError" in result["content"][0]["text"]
    assert "the maths fell apart" in result["content"][0]["text"]


async def test_tools_invalid_arguments_return_validation_error(
    mock_sdk: dict[str, Any],
) -> None:
    """Args that don't validate against the params model surface as a tool error."""
    final = _FakeResultMessage(result="ok")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session(tools=[_build_tool()])
    try:
        await sess.execute("anything")
    finally:
        await sess.close()

    wrapped = mock_sdk["server_calls"][0]["tools"][0]
    result = await wrapped({"a": "not-a-number", "b": 2})
    assert result["isError"] is True
    assert "_AddParams" in result["content"][0]["text"]


async def test_stream_emits_tool_call_start_and_result_events(
    mock_sdk: dict[str, Any],
) -> None:
    """stream() translates ToolUseBlock/ToolResultBlock into airframe events."""
    from airframe.events import ToolCallResult, ToolCallStart

    assistant = _FakeAssistantMessage(
        content=[
            _FakeTextBlock("Let me add those..."),
            _FakeToolUseBlock(
                id="toolu_01", name="mcp__airframe_tools__add", input={"a": 17, "b": 23}
            ),
        ]
    )
    user = _FakeUserMessage(
        content=[
            _FakeToolResultBlock(
                tool_use_id="toolu_01",
                content=[{"type": "text", "text": "40.0"}],
            )
        ]
    )
    final_assistant = _FakeAssistantMessage(content=[_FakeTextBlock("17 + 23 = 40.")])
    final = _FakeResultMessage(result="17 + 23 = 40.")

    async def fake_receive() -> Any:
        yield assistant
        yield user
        yield final_assistant
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
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
    # The MCP prefix is stripped so consumers see the FunctionTool.name.
    assert starts[0].tool_name == "add"
    assert starts[0].tool_call_id == "toolu_01"
    assert "17" in starts[0].arguments_preview and "23" in starts[0].arguments_preview
    assert len(results) == 1
    assert results[0].tool_call_id == "toolu_01"
    assert results[0].output == "40.0"
    assert results[0].is_error is False
    assert len(turns) == 1
    assert events[-1] is turns[0]


async def test_stream_surfaces_is_error_on_tool_result_block(
    mock_sdk: dict[str, Any],
) -> None:
    """ToolResultBlock(is_error=True) propagates to ToolCallResult.is_error."""
    from airframe.events import ToolCallResult

    assistant = _FakeAssistantMessage(
        content=[
            _FakeToolUseBlock(
                id="toolu_42", name="mcp__airframe_tools__add", input={"a": 1, "b": 2}
            )
        ]
    )
    user = _FakeUserMessage(
        content=[
            _FakeToolResultBlock(
                tool_use_id="toolu_42",
                content="kaboom",
                is_error=True,
            )
        ]
    )
    final = _FakeResultMessage(result="apologies")

    async def fake_receive() -> Any:
        yield assistant
        yield user
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
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
    assert results[0].output == "kaboom"


async def test_tools_change_between_turns_reconnects(
    mock_sdk: dict[str, Any],
) -> None:
    """Switching tools= invalidates the cached client (mcp_servers is connect-bound)."""
    from airframe import FunctionTool

    async def _other(_: _AddParams) -> float:
        return 0.0

    other_tool = FunctionTool(
        name="other",
        description="Different tool.",
        params=_AddParams,
        handler=_other,
    )

    final = _FakeResultMessage(result="r")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess1 = rt.session(tools=[_build_tool()])
    try:
        await sess1.execute("turn 1")
    finally:
        await sess1.close()

    factory_after_first = mock_sdk["factory"].call_count

    sess2 = rt.session(tools=[other_tool])
    try:
        await sess2.execute("turn 2")
    finally:
        await sess2.close()

    # A fresh session built a fresh client, regardless of caching.
    assert mock_sdk["factory"].call_count > factory_after_first


async def test_no_tools_omits_mcp_servers_and_allowed_tools(
    mock_sdk: dict[str, Any],
) -> None:
    """When tools= is absent, ClaudeAgentOptions stays clean."""
    final = _FakeResultMessage(result="ok")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session()
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    opts = mock_sdk["options_kwargs"][0]
    assert "mcp_servers" not in opts
    assert "allowed_tools" not in opts
    # And no MCP server was created.
    assert mock_sdk["server_calls"] == []
    await sess.close()
