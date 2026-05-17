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


# ---------------------------------------------------------------------------
# External MCP server refs (Phase 4 Iteration B)
# ---------------------------------------------------------------------------


def _make_final_only_receive() -> Any:
    """Convenience: a ``receive_response`` that yields one ResultMessage."""

    async def fake_receive() -> Any:
        yield _FakeResultMessage(result="ok")

    return fake_receive


async def test_mcp_stdio_translates_to_typed_dict(mock_sdk: dict[str, Any]) -> None:
    """``McpServerRef(transport='stdio', command=[...])`` builds the SDK's
    stdio TypedDict (``type``, ``command``, ``args``)."""
    from airframe import McpServerRef

    mock_sdk["client"].receive_response = _make_final_only_receive()

    rt = ClaudeCodeRuntime()
    ref = McpServerRef(
        name="everything",
        transport="stdio",
        command=["uvx", "mcp-server-everything", "--flag"],
    )
    sess = rt.session(mcp_servers=[ref])
    try:
        await sess.execute("anything")
    finally:
        await sess.close()

    opts = mock_sdk["options_kwargs"][0]
    assert "mcp_servers" in opts
    assert "everything" in opts["mcp_servers"]
    cfg = opts["mcp_servers"]["everything"]
    assert cfg["type"] == "stdio"
    # argv head/tail split — SDK takes command: str, args: list[str].
    assert cfg["command"] == "uvx"
    assert cfg["args"] == ["mcp-server-everything", "--flag"]
    # No url / headers leaks onto a stdio config.
    assert "url" not in cfg
    assert "headers" not in cfg


async def test_mcp_stdio_single_element_command_omits_args(
    mock_sdk: dict[str, Any],
) -> None:
    """A 1-element ``command`` produces no ``args`` key (minimal wire shape)."""
    from airframe import McpServerRef

    mock_sdk["client"].receive_response = _make_final_only_receive()

    rt = ClaudeCodeRuntime()
    sess = rt.session(
        mcp_servers=[McpServerRef(name="solo", transport="stdio", command=["mcp-bin"])]
    )
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    cfg = mock_sdk["options_kwargs"][0]["mcp_servers"]["solo"]
    assert cfg == {"type": "stdio", "command": "mcp-bin"}


async def test_mcp_http_translates_to_typed_dict_with_auth_header(
    mock_sdk: dict[str, Any],
) -> None:
    """``auth_token=`` becomes ``Authorization: Bearer …`` on http transport."""
    from airframe import McpServerRef

    mock_sdk["client"].receive_response = _make_final_only_receive()

    rt = ClaudeCodeRuntime()
    ref = McpServerRef(
        name="remote",
        transport="http",
        url="https://mcp.example.com",
        auth_token="secret-token",
    )
    sess = rt.session(mcp_servers=[ref])
    try:
        await sess.execute("anything")
    finally:
        await sess.close()

    cfg = mock_sdk["options_kwargs"][0]["mcp_servers"]["remote"]
    assert cfg["type"] == "http"
    assert cfg["url"] == "https://mcp.example.com"
    assert cfg["headers"]["Authorization"] == "Bearer secret-token"
    # And no stdio leakage.
    assert "command" not in cfg


async def test_mcp_sse_translates_to_typed_dict(mock_sdk: dict[str, Any]) -> None:
    """``transport='sse'`` produces the SSE TypedDict shape."""
    from airframe import McpServerRef

    mock_sdk["client"].receive_response = _make_final_only_receive()

    rt = ClaudeCodeRuntime()
    sess = rt.session(
        mcp_servers=[
            McpServerRef(
                name="sse-feed",
                transport="sse",
                url="https://mcp.example.com/sse",
                headers={"X-Trace": "trace-123"},
            )
        ]
    )
    try:
        await sess.execute("anything")
    finally:
        await sess.close()

    cfg = mock_sdk["options_kwargs"][0]["mcp_servers"]["sse-feed"]
    assert cfg["type"] == "sse"
    assert cfg["url"] == "https://mcp.example.com/sse"
    assert cfg["headers"] == {"X-Trace": "trace-123"}


async def test_mcp_caller_headers_override_auth_token(
    mock_sdk: dict[str, Any],
) -> None:
    """Caller-supplied ``Authorization`` in ``headers=`` wins over ``auth_token=``."""
    from airframe import McpServerRef

    mock_sdk["client"].receive_response = _make_final_only_receive()

    rt = ClaudeCodeRuntime()
    sess = rt.session(
        mcp_servers=[
            McpServerRef(
                name="remote",
                transport="http",
                url="https://mcp.example.com",
                auth_token="shorthand",
                headers={"Authorization": "Bearer caller-explicit"},
            )
        ]
    )
    try:
        await sess.execute("anything")
    finally:
        await sess.close()

    cfg = mock_sdk["options_kwargs"][0]["mcp_servers"]["remote"]
    assert cfg["headers"]["Authorization"] == "Bearer caller-explicit"


async def test_mcp_mixed_transports_in_one_session(mock_sdk: dict[str, Any]) -> None:
    """A list with all three transports lands as one dict keyed by name."""
    from airframe import McpServerRef

    mock_sdk["client"].receive_response = _make_final_only_receive()

    rt = ClaudeCodeRuntime()
    sess = rt.session(
        mcp_servers=[
            McpServerRef(name="local", transport="stdio", command=["a"]),
            McpServerRef(name="rest", transport="http", url="https://h.example.com"),
            McpServerRef(name="feed", transport="sse", url="https://s.example.com"),
        ]
    )
    try:
        await sess.execute("anything")
    finally:
        await sess.close()

    servers = mock_sdk["options_kwargs"][0]["mcp_servers"]
    assert set(servers.keys()) == {"local", "rest", "feed"}
    assert servers["local"]["type"] == "stdio"
    assert servers["rest"]["type"] == "http"
    assert servers["feed"]["type"] == "sse"
    # And each external server gets a wildcard allowed_tools entry.
    allowed = mock_sdk["options_kwargs"][0]["allowed_tools"]
    assert "mcp__local__*" in allowed
    assert "mcp__rest__*" in allowed
    assert "mcp__feed__*" in allowed


async def test_mcp_servers_coexist_with_tools(mock_sdk: dict[str, Any]) -> None:
    """``tools=`` (in-process) + ``mcp_servers=`` (external) merge into one dict."""
    from airframe import McpServerRef

    mock_sdk["client"].receive_response = _make_final_only_receive()

    rt = ClaudeCodeRuntime()
    sess = rt.session(
        tools=[_build_tool()],
        mcp_servers=[McpServerRef(name="external", transport="stdio", command=["a"])],
    )
    try:
        await sess.execute("anything")
    finally:
        await sess.close()

    opts = mock_sdk["options_kwargs"][0]
    servers = opts["mcp_servers"]
    assert "airframe_tools" in servers  # in-process FunctionTool server
    assert "external" in servers
    # Both the per-tool allowed name and the per-server wildcard appear.
    allowed = opts["allowed_tools"]
    assert "mcp__airframe_tools__add" in allowed
    assert "mcp__external__*" in allowed


async def test_mcp_external_name_colliding_with_in_process_raises(
    mock_sdk: dict[str, Any],
) -> None:
    """Reserving the in-process server name catches the collision early."""
    from airframe import McpServerRef
    from airframe.adapters.claude_code import AIRFRAME_MCP_SERVER_NAME

    mock_sdk["client"].receive_response = _make_final_only_receive()

    rt = ClaudeCodeRuntime()
    sess = rt.session(
        tools=[_build_tool()],
        mcp_servers=[
            McpServerRef(name=AIRFRAME_MCP_SERVER_NAME, transport="stdio", command=["x"])
        ],
    )
    try:
        with pytest.raises(ValueError, match="reserved"):
            await sess.execute("anything")
    finally:
        await sess.close()


async def test_mcp_refs_change_between_sessions_reconnects(
    mock_sdk: dict[str, Any],
) -> None:
    """Switching ``mcp_servers=`` builds a fresh client (mcp_servers is connect-bound)."""
    from airframe import McpServerRef

    mock_sdk["client"].receive_response = _make_final_only_receive()

    rt = ClaudeCodeRuntime()
    sess1 = rt.session(mcp_servers=[McpServerRef(name="one", transport="stdio", command=["x"])])
    try:
        await sess1.execute("turn 1")
    finally:
        await sess1.close()

    factory_after_first = mock_sdk["factory"].call_count

    sess2 = rt.session(mcp_servers=[McpServerRef(name="two", transport="stdio", command=["x"])])
    try:
        await sess2.execute("turn 2")
    finally:
        await sess2.close()

    assert mock_sdk["factory"].call_count > factory_after_first


async def test_mcp_refs_change_within_session_invalidates_cache() -> None:
    """The fingerprint differs when refs change → cache key changes.

    Driven via the helper directly so the test doesn't depend on a
    second session boundary.
    """
    from airframe import McpServerRef
    from airframe.sessions import _mcp_servers_fingerprint

    a = [McpServerRef(name="one", transport="stdio", command=["x"])]
    b = [McpServerRef(name="two", transport="stdio", command=["x"])]
    assert _mcp_servers_fingerprint(a) != _mcp_servers_fingerprint(b)
    # Same name + transport + command → same fingerprint (deterministic).
    assert _mcp_servers_fingerprint(a) == _mcp_servers_fingerprint(
        [McpServerRef(name="one", transport="stdio", command=["x"])]
    )


async def test_mcp_fingerprint_excludes_auth_token_and_header_values() -> None:
    """Rotating an ``auth_token`` or header value must not change the fingerprint."""
    from airframe import McpServerRef
    from airframe.sessions import _mcp_servers_fingerprint

    base = [
        McpServerRef(
            name="r",
            transport="http",
            url="https://h",
            headers={"X-Trace": "v1"},
            auth_token="t1",
        )
    ]
    rotated_token = [
        McpServerRef(
            name="r",
            transport="http",
            url="https://h",
            headers={"X-Trace": "v1"},
            auth_token="t2",
        )
    ]
    rotated_header_value = [
        McpServerRef(
            name="r",
            transport="http",
            url="https://h",
            headers={"X-Trace": "v2"},
            auth_token="t1",
        )
    ]
    assert _mcp_servers_fingerprint(base) == _mcp_servers_fingerprint(rotated_token)
    assert _mcp_servers_fingerprint(base) == _mcp_servers_fingerprint(rotated_header_value)

    # Adding a new header key DOES change the fingerprint (the key set
    # participates, just not the values).
    add_header = [
        McpServerRef(
            name="r",
            transport="http",
            url="https://h",
            headers={"X-Trace": "v1", "X-Extra": "new"},
            auth_token="t1",
        )
    ]
    assert _mcp_servers_fingerprint(base) != _mcp_servers_fingerprint(add_header)


async def test_stream_strips_external_mcp_prefix(mock_sdk: dict[str, Any]) -> None:
    """Tool calls routed through an external MCP server come back with the bare name."""
    from airframe import McpServerRef
    from airframe.events import ToolCallStart

    assistant = _FakeAssistantMessage(
        content=[
            _FakeToolUseBlock(
                id="toolu_99",
                name="mcp__github_remote__search_repos",
                input={"q": "airframe"},
            )
        ]
    )
    final = _FakeResultMessage(result="found")

    async def fake_receive() -> Any:
        yield assistant
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session(
        mcp_servers=[
            McpServerRef(name="github_remote", transport="http", url="https://example.com")
        ]
    )
    starts: list[ToolCallStart] = []
    try:
        async for event in sess.stream("search"):
            if isinstance(event, ToolCallStart):
                starts.append(event)
    finally:
        await sess.close()

    assert len(starts) == 1
    # External server's prefix was stripped just like the in-process server's is.
    assert starts[0].tool_name == "search_repos"


async def test_stream_unknown_mcp_prefix_passes_through(mock_sdk: dict[str, Any]) -> None:
    """Tools from servers not in the registered set keep their raw vendor name.

    The plan's risk-note #6: ``Unrecognised prefixes pass through
    verbatim so consumers can still inspect raw vendor tool names if
    needed.`` This protects consumers who reach into
    :meth:`AgentRuntime.unwrap` and register a server out-of-band.
    """
    from airframe.events import ToolCallStart

    assistant = _FakeAssistantMessage(
        content=[
            _FakeToolUseBlock(
                id="toolu_88",
                name="mcp__not_registered__do_thing",
                input={},
            )
        ]
    )
    final = _FakeResultMessage(result="ok")

    async def fake_receive() -> Any:
        yield assistant
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session()  # no tools=, no mcp_servers=
    starts: list[ToolCallStart] = []
    try:
        async for event in sess.stream("anything"):
            if isinstance(event, ToolCallStart):
                starts.append(event)
    finally:
        await sess.close()

    assert len(starts) == 1
    assert starts[0].tool_name == "mcp__not_registered__do_thing"


def test_mcp_fingerprint_empty_list_is_constant() -> None:
    """Empty / None refs collapse to the same sentinel so no-op sessions cache."""
    from airframe.sessions import _mcp_servers_fingerprint

    assert _mcp_servers_fingerprint([]) == "__no_mcp_servers__"


# ---------------------------------------------------------------------------
# Permission callback (Phase 5 Iteration B)
# ---------------------------------------------------------------------------


class _FakePermissionContext:
    """Minimal stand-in for ToolPermissionContext for unit tests."""

    def __init__(
        self,
        *,
        decision_reason: str | None = None,
        description: str | None = None,
        title: str | None = None,
    ) -> None:
        self.decision_reason = decision_reason
        self.description = description
        self.title = title
        self.signal = None
        self.suggestions: list = []
        self.tool_use_id = None
        self.agent_id = None
        self.blocked_path = None


async def test_permission_callback_wraps_into_can_use_tool(
    mock_sdk: dict[str, Any],
) -> None:
    """on_permission= becomes the SDK's can_use_tool callable on options."""
    from airframe import PermissionDecision, PermissionRequest

    received: list[PermissionRequest] = []

    class _RecordingCallback:
        async def handle(self, request: PermissionRequest) -> PermissionDecision:
            received.append(request)
            return "allow"

    final = _FakeResultMessage(result="ok")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session(on_permission=_RecordingCallback())
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    opts = mock_sdk["options_kwargs"][0]
    assert "can_use_tool" in opts
    # Invoke the wrapped callable directly to verify the airframe
    # PermissionRequest is constructed correctly and the decision
    # maps to PermissionResultAllow.
    from claude_agent_sdk import PermissionResultAllow

    can_use_tool = opts["can_use_tool"]
    result = await can_use_tool(
        "write_file",
        {"path": "/tmp/x"},
        _FakePermissionContext(decision_reason="writes to filesystem"),
    )
    assert isinstance(result, PermissionResultAllow)
    assert len(received) == 1
    req = received[0]
    assert req.tool_name == "write_file"
    assert req.tool_args == {"path": "/tmp/x"}
    assert req.reason == "writes to filesystem"


async def test_permission_callback_deny_returns_permission_result_deny(
    mock_sdk: dict[str, Any],
) -> None:
    """``deny`` decision maps to :class:`PermissionResultDeny`."""
    from airframe import PermissionDecision, PermissionRequest

    class _DenyAll:
        async def handle(self, request: PermissionRequest) -> PermissionDecision:
            return "deny"

    final = _FakeResultMessage(result="ok")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session(on_permission=_DenyAll())
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    from claude_agent_sdk import PermissionResultDeny

    can_use_tool = mock_sdk["options_kwargs"][0]["can_use_tool"]
    result = await can_use_tool("read_file", {}, _FakePermissionContext())
    assert isinstance(result, PermissionResultDeny)
    # Reason falls through into the message when present; otherwise
    # an actionable generic message names the tool.
    assert "read_file" in result.message


async def test_permission_callback_defer_coerces_to_allow_with_debug_log(
    mock_sdk: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``defer`` collapses to :class:`PermissionResultAllow` + debug log.

    Claude's binary result type has no third option; the existing
    ``permission_mode="bypassPermissions"`` default already allows
    everything, so "I don't have an opinion" collapses to "allow"
    with an audit-trail debug log.
    """
    import logging

    from airframe import PermissionDecision, PermissionRequest

    class _Defer:
        async def handle(self, request: PermissionRequest) -> PermissionDecision:
            return "defer"

    final = _FakeResultMessage(result="ok")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session(on_permission=_Defer())
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    from claude_agent_sdk import PermissionResultAllow

    can_use_tool = mock_sdk["options_kwargs"][0]["can_use_tool"]
    with caplog.at_level(logging.DEBUG, logger="airframe.adapters.claude_code"):
        result = await can_use_tool("read_file", {}, _FakePermissionContext())
    assert isinstance(result, PermissionResultAllow)
    assert any("defer" in rec.message and "read_file" in rec.message for rec in caplog.records)


async def test_no_permission_callback_omits_can_use_tool_kwarg(
    mock_sdk: dict[str, Any],
) -> None:
    """Sessions opened without on_permission= don't set can_use_tool on options."""
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

    assert "can_use_tool" not in mock_sdk["options_kwargs"][0]


async def test_permission_callback_change_between_sessions_reconnects(
    mock_sdk: dict[str, Any],
) -> None:
    """Switching ``on_permission=`` invalidates the cached client.

    Callback identity joins the ``_ensure_client`` cache key —
    ``can_use_tool`` is baked at connect time, so a callback swap
    must force reconnect.
    """
    from airframe import PermissionDecision, PermissionRequest

    class _Cb1:
        async def handle(self, request: PermissionRequest) -> PermissionDecision:
            return "allow"

    class _Cb2:
        async def handle(self, request: PermissionRequest) -> PermissionDecision:
            return "deny"

    final = _FakeResultMessage(result="r")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess1 = rt.session(on_permission=_Cb1())
    try:
        await sess1.execute("turn 1")
    finally:
        await sess1.close()
    factory_after_first = mock_sdk["factory"].call_count

    sess2 = rt.session(on_permission=_Cb2())
    try:
        await sess2.execute("turn 2")
    finally:
        await sess2.close()

    assert mock_sdk["factory"].call_count > factory_after_first


def test_permission_fingerprint_distinguishes_callback_identity() -> None:
    """Different callback objects produce different fingerprints; same
    object is stable; None collapses to a sentinel."""
    from airframe import PermissionDecision, PermissionRequest
    from airframe.adapters.claude_code import _permission_fingerprint

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


async def test_on_event_wires_hooks_into_claude_agent_options(
    mock_sdk: dict[str, Any],
) -> None:
    """on_event= becomes a hooks= dict on ClaudeAgentOptions covering
    every native event the SDK exposes."""
    from airframe import HookEvent

    received: list[HookEvent] = []

    final = _FakeResultMessage(result="ok")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session(on_event=received.append)
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    opts = mock_sdk["options_kwargs"][0]
    assert "hooks" in opts
    # The hooks dict carries one entry per native event name we map.
    from airframe.adapters.claude_code import _CLAUDE_HOOK_NAME_TO_KIND

    for sdk_name in _CLAUDE_HOOK_NAME_TO_KIND:
        assert sdk_name in opts["hooks"]


async def test_on_event_synthesises_session_start_at_connect(
    mock_sdk: dict[str, Any],
) -> None:
    """session_start fires on first execute() — Claude has no native event."""
    from airframe import HookEvent

    received: list[HookEvent] = []
    final = _FakeResultMessage(result="ok")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session(on_event=received.append)
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    kinds = [e.kind for e in received]
    # session_start fires first, session_end fires last.
    assert kinds[0] == "session_start"
    assert kinds[-1] == "session_end"
    # session_start payload carries model + resumed flag.
    assert received[0].payload["model"] == "claude-haiku-4-5"
    assert received[0].payload["resumed"] is False


async def test_on_event_native_pre_tool_use_translates_to_hook_event(
    mock_sdk: dict[str, Any],
) -> None:
    """Invoking the SDK's PreToolUse hook fires a pre_tool_use HookEvent.

    The mock fixture captures the registered hooks-dict callable;
    we invoke it directly to verify the translation without
    needing the SDK's CLI subprocess to actually fire a hook.
    """
    from airframe import HookEvent

    received: list[HookEvent] = []
    final = _FakeResultMessage(result="ok")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session(on_event=received.append)
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    # Pull the PreToolUse callback out of the registered hooks-dict
    # and invoke it directly. (The mocked SDK doesn't dispatch
    # automatically.)
    hooks_dict = mock_sdk["options_kwargs"][0]["hooks"]
    pre_tool_matcher = hooks_dict["PreToolUse"][0]
    pre_tool_cb = pre_tool_matcher.hooks[0]
    fake_input = {
        "hook_event_name": "PreToolUse",
        "session_id": "sess-XYZ",
        "tool_name": "Write",
        "tool_input": {"path": "/tmp/x", "content": "..."},
        "tool_use_id": "toolu_42",
    }
    result = await pre_tool_cb(fake_input, "toolu_42", None)
    assert result == {}  # pure observation — no continue=/decision=/etc.

    # Filter to pre_tool_use events.
    pre_events = [e for e in received if e.kind == "pre_tool_use"]
    assert len(pre_events) == 1
    payload = pre_events[0].payload
    assert payload["tool_name"] == "Write"
    assert payload["tool_use_id"] == "toolu_42"
    assert payload["tool_input"] == {"path": "/tmp/x", "content": "..."}


async def test_no_on_event_omits_hooks_kwarg(mock_sdk: dict[str, Any]) -> None:
    """Sessions opened without on_event= don't set hooks on options."""
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

    assert "hooks" not in mock_sdk["options_kwargs"][0]


async def test_close_session_end_idempotent(mock_sdk: dict[str, Any]) -> None:
    """Repeat close() calls don't re-fire session_end."""
    from airframe import HookEvent

    received: list[HookEvent] = []
    final = _FakeResultMessage(result="ok")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session(on_event=received.append)
    await sess.execute("hi")
    await sess.close()
    await sess.close()
    await sess.close()

    end_events = [e for e in received if e.kind == "session_end"]
    assert len(end_events) == 1


async def test_observer_raises_does_not_break_session(
    mock_sdk: dict[str, Any],
) -> None:
    """A raising observer is debug-logged and swallowed."""
    from airframe import HookEvent

    def raising_observer(_event: HookEvent) -> None:
        raise RuntimeError("observer boom")

    final = _FakeResultMessage(result="ok")

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    rt = ClaudeCodeRuntime()
    sess = rt.session(on_event=raising_observer)
    try:
        # Should not propagate the observer's exception.
        result = await sess.execute("hi")
    finally:
        await sess.close()
    assert result.text == "ok"


def test_claude_runtime_declares_lifecycle_hooks() -> None:
    """Claude declares LIFECYCLE_HOOKS + the full 8-kind emittable set."""
    rt = ClaudeCodeRuntime()
    assert rt.supports(Feature.LIFECYCLE_HOOKS)
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
                "rate_limit",
            }
        )
        == ClaudeCodeRuntime.EMITTABLE_HOOK_KINDS
    )


# ---------------------------------------------------------------------------
# Budget caps (Phase 5 Iteration D)
# ---------------------------------------------------------------------------


def test_claude_runtime_declares_budget_caps() -> None:
    """Claude declares both BUDGET_USD_CAP and BUDGET_TURN_CAP after Iteration D."""
    rt = ClaudeCodeRuntime()
    assert rt.supports(Feature.BUDGET_USD_CAP)
    assert rt.supports(Feature.BUDGET_TURN_CAP)


async def test_max_turns_overrides_runtime_default_in_options(
    mock_sdk: dict[str, Any],
) -> None:
    """``max_turns=`` on execute() lands as ``ClaudeAgentOptions.max_turns``,
    overriding the runtime-default DEFAULT_MAX_TURNS at connect time."""
    final = _FakeResultMessage(result="ok", total_cost_usd=0.0)

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive
    rt = ClaudeCodeRuntime()
    sess = rt.session()
    try:
        await sess.execute("hi", max_turns=7)
    finally:
        await sess.close()

    opts = mock_sdk["options_kwargs"][0]
    assert opts["max_turns"] == 7


async def test_max_turns_omitted_uses_runtime_default(mock_sdk: dict[str, Any]) -> None:
    """No ``max_turns=`` → ClaudeAgentOptions.max_turns falls back to
    the runtime's DEFAULT_MAX_TURNS (60)."""
    final = _FakeResultMessage(result="ok", total_cost_usd=0.0)

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
    assert opts["max_turns"] == 60


async def test_max_turns_change_between_turns_forces_reconnect(
    mock_sdk: dict[str, Any],
) -> None:
    """A different ``max_turns=`` value joins the cache key and forces
    a reconnect — the SDK bakes max_turns at connect time."""
    final = _FakeResultMessage(result="r", total_cost_usd=0.0)

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive
    rt = ClaudeCodeRuntime()
    sess = rt.session()
    try:
        await sess.execute("a", max_turns=5)
        await sess.execute("b", max_turns=15)
    finally:
        await sess.close()
    assert mock_sdk["client"].connect.await_count == 2


async def test_max_turns_cap_raises_when_cumulative_count_reached(
    mock_sdk: dict[str, Any],
) -> None:
    """After running max_turns turns, the next execute() raises
    RuntimeBudgetExceededError(kind='turns')."""
    from airframe.errors import RuntimeBudgetExceededError

    final = _FakeResultMessage(result="ok", total_cost_usd=0.0)

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive
    rt = ClaudeCodeRuntime()
    sess = rt.session()
    try:
        # Two turns succeed under the cap.
        await sess.execute("turn 1", max_turns=2)
        await sess.execute("turn 2", max_turns=2)
        # Third turn trips the cap.
        with pytest.raises(RuntimeBudgetExceededError) as exc_info:
            await sess.execute("turn 3", max_turns=2)
    finally:
        await sess.close()
    err = exc_info.value
    assert err.kind == "turns"
    assert err.cap == 2.0
    assert err.current == 2.0


async def test_max_budget_usd_cap_raises_when_cumulative_cost_reached(
    mock_sdk: dict[str, Any],
) -> None:
    """Per-turn cost accumulates on the session; once cumulative
    exceeds max_budget_usd, the next execute() raises
    RuntimeBudgetExceededError(kind='usd')."""
    from airframe.errors import RuntimeBudgetExceededError

    final = _FakeResultMessage(result="ok", total_cost_usd=0.03)

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive
    rt = ClaudeCodeRuntime()
    sess = rt.session()
    try:
        # Two turns at $0.03 each → cumulative $0.06 > cap $0.05.
        await sess.execute("turn 1", max_budget_usd=0.05)
        await sess.execute("turn 2", max_budget_usd=0.05)
        with pytest.raises(RuntimeBudgetExceededError) as exc_info:
            await sess.execute("turn 3", max_budget_usd=0.05)
    finally:
        await sess.close()
    err = exc_info.value
    assert err.kind == "usd"
    assert err.cap == 0.05
    assert err.current >= 0.05


async def test_budget_caps_none_open_cleanly_and_dont_track(
    mock_sdk: dict[str, Any],
) -> None:
    """Sessions without caps still accumulate counters (they're
    cheap), but no enforcement fires."""
    final = _FakeResultMessage(result="ok", total_cost_usd=1.0)

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive
    rt = ClaudeCodeRuntime()
    sess = rt.session()
    try:
        await sess.execute("a")
        await sess.execute("b")
        await sess.execute("c")
    finally:
        await sess.close()
    # Three turns ran to completion despite the high $1 per-turn
    # cost; without max_budget_usd= the cap never fires.


async def test_stream_honours_budget_caps(mock_sdk: dict[str, Any]) -> None:
    """The streaming path runs the same pre-turn enforce — exhausted
    cap raises before yielding any events."""
    from airframe.errors import RuntimeBudgetExceededError

    final = _FakeResultMessage(result="ok", total_cost_usd=0.02)

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive
    rt = ClaudeCodeRuntime()
    sess = rt.session()
    try:
        # First stream turn succeeds and pushes cumulative cost.
        async for _ in sess.stream("a", max_budget_usd=0.01):
            pass
        # Cumulative is now $0.02 > cap $0.01 — next stream raises.
        with pytest.raises(RuntimeBudgetExceededError) as exc_info:
            async for _ in sess.stream("b", max_budget_usd=0.01):
                pass
    finally:
        await sess.close()
    assert exc_info.value.kind == "usd"
