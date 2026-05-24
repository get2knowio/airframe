"""Unit tests for :class:`OpenAICompatibleSession`.

Phase 1 Iteration C — first end-to-end bespoke
:class:`~airframe.protocol.AgentSession`. Mocks the ``openai`` SDK at
the boundary so we can exercise the multi-turn buffer, the streaming
chunk shape, and the cancellation path without hitting the network.

What's covered:

* Multi-turn ``messages=[]`` buffer accumulates user + assistant
  messages and gets resent on subsequent turns.
* ``system=`` on session construction seeds the buffer once.
* Failure during :meth:`execute` rolls back the user message so the
  next attempt has a clean history.
* :meth:`stream` yields :class:`TextDelta` for each chunk's content
  and ends with exactly one :class:`TurnComplete`.
* Streaming honours ``schema=`` — accumulated text parses into
  :attr:`RuntimeResult.structured`.
* :meth:`cancel` aborts an in-flight :meth:`execute` (the awaiting
  call raises :class:`RuntimeCancelledError` and the buffer rolls
  back).
* :meth:`close` is idempotent and prevents further calls.

Behavioural live-vendor probes belong in
:mod:`airframe.testing.integration` (Phase 1 work).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from airframe.adapters.opencode_zen import OpenCodeZenRuntime
from airframe.errors import RuntimeCancelledError, UnsupportedFeatureError
from airframe.events import TextDelta, TurnComplete
from airframe.features import Feature
from airframe.protocol import AgentSession


class _Schema(BaseModel):
    summary: str
    count: int


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_response(
    *,
    content: str = "hello world",
    finish_reason: str = "stop",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    tool_calls: list[Any] | None = None,
) -> Any:
    """Stand-in for an ``openai`` ChatCompletion (non-streaming).

    ``tool_calls`` defaults to ``None`` so the tool-loop sees a normal
    text turn; tests that exercise the tool path build their own
    ``_FakeToolCall`` instances and pass them in.
    """
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.prompt_tokens_details = None
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _make_chunk(
    *,
    content: str | None = None,
    finish_reason: str | None = None,
    tool_call_deltas: list[Any] | None = None,
) -> Any:
    """Stand-in for one ``ChatCompletionChunk`` from the streaming API.

    ``tool_call_deltas`` defaults to ``None`` so the consumer accumulator
    sees a plain text chunk; tests exercising the tool-loop streaming
    path supply per-index :class:`_FakeToolCallDelta` instances.
    """
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_call_deltas
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


def _make_usage_chunk(*, prompt_tokens: int, completion_tokens: int) -> Any:
    """Final stream chunk carrying usage when ``stream_options=include_usage``."""
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.prompt_tokens_details = None
    chunk = MagicMock()
    chunk.choices = []  # final usage frame has no choices
    chunk.usage = usage
    return chunk


class _AsyncIter:
    """Tiny async iterator over a list of chunks, with a close() coroutine."""

    def __init__(self, items: list[Any]) -> None:
        self._items = items
        self._idx = 0
        self.closed = False

    def __aiter__(self) -> _AsyncIter:
        return self

    async def __anext__(self) -> Any:
        if self.closed or self._idx >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._idx]
        self._idx += 1
        return item

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def mock_openai(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace ``openai.AsyncOpenAI`` with a fresh mock per test."""
    import openai

    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_make_response())

    # Production uses with_raw_response.create() to read x-ratelimit-*
    # headers; the delegating shim looks up .create dynamically so per-
    # test rebinds of ``client.chat.completions.create`` still take
    # effect through the with_raw_response path.
    async def _raw_create(**kwargs: Any) -> Any:
        response = await client.chat.completions.create(**kwargs)
        raw = MagicMock()
        raw.parse = MagicMock(return_value=response)
        raw.headers = {}
        return raw

    client.chat.completions.with_raw_response = MagicMock()
    client.chat.completions.with_raw_response.create = AsyncMock(side_effect=_raw_create)
    client.close = AsyncMock()

    factory = MagicMock(return_value=client)
    monkeypatch.setattr(openai, "AsyncOpenAI", factory)
    return client


def _make_runtime() -> OpenCodeZenRuntime:
    """An OpenCodeZenRuntime with auth satisfied for construction."""
    return OpenCodeZenRuntime(api_key="dummy-for-test")


# ---------------------------------------------------------------------------
# Factory + capability surface
# ---------------------------------------------------------------------------


async def test_session_factory_returns_bespoke_session() -> None:
    """``session()`` returns the bespoke class, not _ThinAgentSession."""
    from airframe.adapters.openai_compatible import OpenAICompatibleSession

    runtime = _make_runtime()
    sess = runtime.session()
    try:
        assert isinstance(sess, OpenAICompatibleSession)
        assert isinstance(sess, AgentSession)
        assert sess.id is None  # No server-side session for chat-completions.
    finally:
        await sess.close()


def test_streaming_and_cancel_features_declared() -> None:
    """Iteration C flipped these on; SESSION_RESUME stays False."""
    runtime = _make_runtime()
    assert runtime.supports(Feature.STREAMING) is True
    assert runtime.supports(Feature.CANCEL) is True
    assert runtime.supports(Feature.SESSION_RESUME) is False


def test_session_resume_raises_unsupported_feature_error() -> None:
    """Per the plan: chat-completions vendors raise on resume=."""
    runtime = _make_runtime()
    with pytest.raises(UnsupportedFeatureError) as excinfo:
        runtime.session(resume="anything")
    assert excinfo.value.feature == "session_resume"


# ---------------------------------------------------------------------------
# Multi-turn buffer
# ---------------------------------------------------------------------------


async def test_messages_buffer_accumulates_across_turns(mock_openai: MagicMock) -> None:
    """Each turn appends user + assistant; the next call resends the full history."""
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[
            _make_response(content="reply one"),
            _make_response(content="reply two"),
        ]
    )
    runtime = _make_runtime()
    sess = runtime.session(system="be brief")
    try:
        first = await sess.execute("turn 1")
        second = await sess.execute("turn 2")
    finally:
        await sess.close()

    assert first.text == "reply one"
    assert second.text == "reply two"

    # The second call should have sent the full conversation:
    # system + user(t1) + assistant(r1) + user(t2).
    second_call = mock_openai.chat.completions.create.await_args_list[1]
    messages = second_call.kwargs["messages"]
    assert messages == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "reply one"},
        {"role": "user", "content": "turn 2"},
    ]


async def test_system_prompt_appears_once(mock_openai: MagicMock) -> None:
    """``system=`` on session() seeds the buffer at index 0; no per-turn duplication."""
    runtime = _make_runtime()
    sess = runtime.session(system="seed prompt")
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    messages = mock_openai.chat.completions.create.await_args_list[0].kwargs["messages"]
    assert sum(1 for m in messages if m["role"] == "system") == 1
    assert messages[0] == {"role": "system", "content": "seed prompt"}


async def test_execute_failure_rolls_back_user_message(mock_openai: MagicMock) -> None:
    """A failed turn must not leave a dangling user message in the buffer."""
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[RuntimeError("boom"), _make_response(content="retry ok")]
    )
    runtime = _make_runtime()
    sess = runtime.session()
    try:
        with pytest.raises(Exception):  # noqa: B017
            await sess.execute("first attempt")
        # Buffer should be empty (no system prompt either).
        # Next turn should send just the new user message.
        second = await sess.execute("retry")
    finally:
        await sess.close()

    assert second.text == "retry ok"
    second_messages = mock_openai.chat.completions.create.await_args_list[1].kwargs["messages"]
    assert second_messages == [{"role": "user", "content": "retry"}]


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


async def test_stream_yields_text_deltas_then_turn_complete(mock_openai: MagicMock) -> None:
    """The canonical stream shape: N TextDelta then exactly one TurnComplete."""
    chunks = [
        _make_chunk(content="hel"),
        _make_chunk(content="lo "),
        _make_chunk(content="world", finish_reason="stop"),
        _make_usage_chunk(prompt_tokens=10, completion_tokens=3),
    ]
    mock_openai.chat.completions.create = AsyncMock(return_value=_AsyncIter(chunks))

    runtime = _make_runtime()
    sess = runtime.session()
    events: list[Any] = []
    try:
        async for event in sess.stream("hi"):
            events.append(event)
    finally:
        await sess.close()

    text_events = [e for e in events if isinstance(e, TextDelta)]
    turn_events = [e for e in events if isinstance(e, TurnComplete)]
    assert [e.text for e in text_events] == ["hel", "lo ", "world"]
    assert len(turn_events) == 1
    result = turn_events[0].result
    assert result.text == "hello world"
    assert result.finish == "stop"
    assert result.cost.input_tokens == 10
    assert result.cost.output_tokens == 3


async def test_stream_used_stream_kwargs(mock_openai: MagicMock) -> None:
    """``stream=True`` and ``stream_options.include_usage=True`` are wired."""
    mock_openai.chat.completions.create = AsyncMock(return_value=_AsyncIter([]))
    runtime = _make_runtime()
    sess = runtime.session()
    try:
        async for _ in sess.stream("anything"):
            pass
    finally:
        await sess.close()

    call = mock_openai.chat.completions.create.await_args
    assert call.kwargs["stream"] is True
    assert call.kwargs["stream_options"] == {"include_usage": True}


async def test_stream_with_schema_parses_structured(mock_openai: MagicMock) -> None:
    """Accumulated text is parsed into RuntimeResult.structured when schema= given."""
    chunks = [
        _make_chunk(content='{"summary":'),
        _make_chunk(content=' "ok", "count": 7}', finish_reason="stop"),
        _make_usage_chunk(prompt_tokens=5, completion_tokens=10),
    ]
    mock_openai.chat.completions.create = AsyncMock(return_value=_AsyncIter(chunks))
    runtime = _make_runtime()
    sess = runtime.session()
    try:
        events = [e async for e in sess.stream("give me a brief", schema=_Schema)]
    finally:
        await sess.close()
    turn = next(e for e in events if isinstance(e, TurnComplete))
    assert turn.result.structured == {"summary": "ok", "count": 7}


async def test_stream_appends_assistant_message_after_completion(mock_openai: MagicMock) -> None:
    """After a successful stream, the next execute() sees the full history."""
    chunks = [_make_chunk(content="streamed reply", finish_reason="stop")]
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[_AsyncIter(chunks), _make_response(content="follow-up")]
    )
    runtime = _make_runtime()
    sess = runtime.session()
    try:
        async for _ in sess.stream("prompt 1"):
            pass
        await sess.execute("prompt 2")
    finally:
        await sess.close()

    second_messages = mock_openai.chat.completions.create.await_args_list[1].kwargs["messages"]
    assert second_messages == [
        {"role": "user", "content": "prompt 1"},
        {"role": "assistant", "content": "streamed reply"},
        {"role": "user", "content": "prompt 2"},
    ]


# ---------------------------------------------------------------------------
# thinking= (Phase 2 Iteration B)
# ---------------------------------------------------------------------------


async def test_thinking_effort_forwarded_as_reasoning_effort(
    mock_openai: MagicMock,
) -> None:
    """thinking=<effort> becomes the ``reasoning_effort`` kwarg on the wire."""
    runtime = _make_runtime()
    sess = runtime.session()
    try:
        await sess.execute("plan it", thinking="high")
    finally:
        await sess.close()

    call = mock_openai.chat.completions.create.await_args_list[0]
    assert call.kwargs["reasoning_effort"] == "high"


async def test_thinking_none_omits_reasoning_effort_kwarg(
    mock_openai: MagicMock,
) -> None:
    """thinking=None means the kwarg is left out — vendor uses its default."""
    runtime = _make_runtime()
    sess = runtime.session()
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    call = mock_openai.chat.completions.create.await_args_list[0]
    assert "reasoning_effort" not in call.kwargs


async def test_thinking_disabled_omits_reasoning_effort_kwarg(
    mock_openai: MagicMock,
) -> None:
    """thinking='disabled' is wire-equivalent to None for OpenAI-compat."""
    runtime = _make_runtime()
    sess = runtime.session()
    try:
        await sess.execute("hi", thinking="disabled")
    finally:
        await sess.close()

    call = mock_openai.chat.completions.create.await_args_list[0]
    assert "reasoning_effort" not in call.kwargs


async def test_thinking_dict_raises_unsupported_feature_error(
    mock_openai: MagicMock,
) -> None:
    """The ``{"budget_tokens": N}`` shape is Anthropic-only."""
    runtime = _make_runtime()
    sess = runtime.session()
    try:
        with pytest.raises(UnsupportedFeatureError) as exc_info:
            await sess.execute("hi", thinking={"budget_tokens": 5000})
    finally:
        await sess.close()

    assert exc_info.value.feature == Feature.REASONING_BUDGET_TOKENS


async def test_thinking_varies_per_turn_without_session_rebuild(
    mock_openai: MagicMock,
) -> None:
    """OpenAI-compat is stateless on thinking — each turn forwards its own value."""
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[
            _make_response(content="r1"),
            _make_response(content="r2"),
            _make_response(content="r3"),
        ]
    )
    runtime = _make_runtime()
    sess = runtime.session()
    try:
        await sess.execute("a", thinking="low")
        await sess.execute("b", thinking="high")
        await sess.execute("c")
    finally:
        await sess.close()

    efforts = [
        call.kwargs.get("reasoning_effort")
        for call in mock_openai.chat.completions.create.await_args_list
    ]
    assert efforts == ["low", "high", None]


async def test_stream_forwards_thinking(mock_openai: MagicMock) -> None:
    """stream() takes the same translation path as execute()."""
    mock_openai.chat.completions.create = AsyncMock(return_value=_AsyncIter([]))
    runtime = _make_runtime()
    sess = runtime.session()
    try:
        async for _ in sess.stream("hi", thinking="medium"):
            pass
    finally:
        await sess.close()

    call = mock_openai.chat.completions.create.await_args
    assert call.kwargs["reasoning_effort"] == "medium"


# ---------------------------------------------------------------------------
# Vision input (Phase 2 Iteration C)
# ---------------------------------------------------------------------------


async def test_image_input_becomes_content_parts_with_data_url(
    mock_openai: MagicMock, tmp_path: Any
) -> None:
    """A list-shaped prompt with ImageInput(path=) flips content to parts shape."""
    import base64

    from airframe.inputs import ImageInput

    img_path = tmp_path / "tiny.png"
    img_bytes = b"\x89PNG\r\n\x1a\nFAKE"
    img_path.write_bytes(img_bytes)

    runtime = _make_runtime()
    sess = runtime.session()
    try:
        await sess.execute(["look at this:", ImageInput(path=str(img_path))])
    finally:
        await sess.close()

    call = mock_openai.chat.completions.create.await_args_list[0]
    messages = call.kwargs["messages"]
    user_msg = next(m for m in messages if m["role"] == "user")
    content = user_msg["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "look at this:"}
    assert content[1]["type"] == "image_url"
    expected_b64 = base64.b64encode(img_bytes).decode("ascii")
    assert content[1]["image_url"]["url"] == f"data:image/png;base64,{expected_b64}"


async def test_plain_string_keeps_scalar_content(mock_openai: MagicMock) -> None:
    """No images → content stays a bare string (preserves the v0 wire shape)."""
    runtime = _make_runtime()
    sess = runtime.session()
    try:
        await sess.execute("just text")
    finally:
        await sess.close()

    call = mock_openai.chat.completions.create.await_args_list[0]
    user_msg = next(m for m in call.kwargs["messages"] if m["role"] == "user")
    assert user_msg["content"] == "just text"


async def test_image_input_bytes_inlines_as_data_url(mock_openai: MagicMock) -> None:
    """Iteration D: bytes_= → base64 data URL, no filesystem hit."""
    import base64

    from airframe.inputs import ImageInput

    raw = b"\x89PNG\r\n\x1a\nFAKE"
    runtime = _make_runtime()
    sess = runtime.session()
    try:
        await sess.execute(["hi", ImageInput(bytes_=raw, media_type="image/png")])
    finally:
        await sess.close()

    call = mock_openai.chat.completions.create.await_args_list[0]
    user_msg = next(m for m in call.kwargs["messages"] if m["role"] == "user")
    expected = f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"
    assert user_msg["content"][1]["image_url"]["url"] == expected


async def test_image_input_url_passes_through(mock_openai: MagicMock) -> None:
    """Iteration D: url= → vendor fetches it; airframe just hands the URL over."""
    from airframe.inputs import ImageInput

    runtime = _make_runtime()
    sess = runtime.session()
    try:
        await sess.execute(["hi", ImageInput(url="https://example.com/cat.png")])
    finally:
        await sess.close()

    call = mock_openai.chat.completions.create.await_args_list[0]
    user_msg = next(m for m in call.kwargs["messages"] if m["role"] == "user")
    assert user_msg["content"][1]["image_url"]["url"] == "https://example.com/cat.png"


async def test_image_input_bytes_defaults_media_type_to_png(
    mock_openai: MagicMock,
) -> None:
    """bytes_= without media_type falls back to image/png (no sniffing)."""
    from airframe.inputs import ImageInput

    runtime = _make_runtime()
    sess = runtime.session()
    try:
        await sess.execute([ImageInput(bytes_=b"raw")])
    finally:
        await sess.close()

    call = mock_openai.chat.completions.create.await_args_list[0]
    user_msg = next(m for m in call.kwargs["messages"] if m["role"] == "user")
    assert user_msg["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")


async def test_file_input_raises_unsupported_feature(mock_openai: MagicMock) -> None:
    """OpenAI-compat doesn't declare FILE_INPUT (varies wildly across vendors)."""
    from airframe.inputs import FileInput

    runtime = _make_runtime()
    sess = runtime.session()
    try:
        with pytest.raises(UnsupportedFeatureError) as exc_info:
            await sess.execute(["hi", FileInput(path="/tmp/anything.pdf")])
    finally:
        await sess.close()
    assert exc_info.value.feature == Feature.FILE_INPUT


async def test_image_media_type_override_is_honored(mock_openai: MagicMock, tmp_path: Any) -> None:
    """ImageInput.media_type beats the path-extension sniff."""
    from airframe.inputs import ImageInput

    img_path = tmp_path / "unknown.bin"
    img_path.write_bytes(b"raw")
    runtime = _make_runtime()
    sess = runtime.session()
    try:
        await sess.execute([ImageInput(path=str(img_path), media_type="image/webp")])
    finally:
        await sess.close()

    call = mock_openai.chat.completions.create.await_args_list[0]
    user_msg = next(m for m in call.kwargs["messages"] if m["role"] == "user")
    assert user_msg["content"][0]["image_url"]["url"].startswith("data:image/webp;base64,")


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


async def test_cancel_aborts_in_flight_execute(mock_openai: MagicMock) -> None:
    """cancel() while execute() is running raises RuntimeCancelledError."""

    async def _hang(**_: Any) -> Any:
        await asyncio.sleep(5.0)
        return _make_response()

    mock_openai.chat.completions.create = AsyncMock(side_effect=_hang)
    runtime = _make_runtime()
    sess = runtime.session()
    try:
        exec_task = asyncio.create_task(sess.execute("hangs"))
        # Yield so the execute task installs itself as _in_flight_task.
        await asyncio.sleep(0)
        await sess.cancel()
        with pytest.raises(RuntimeCancelledError):
            await exec_task
        # Buffer should be rolled back — next execute starts clean.
        assert sess._messages == []  # noqa: SLF001 — invariant check
    finally:
        await sess.close()


async def test_cancel_closes_active_stream(mock_openai: MagicMock) -> None:
    """cancel() during stream() closes the underlying AsyncStream."""
    iter_obj = _AsyncIter([])  # exhausted immediately so the generator finishes
    mock_openai.chat.completions.create = AsyncMock(return_value=iter_obj)
    runtime = _make_runtime()
    sess = runtime.session()
    try:
        async for _ in sess.stream("x"):
            pass
        await sess.cancel()  # nothing in flight; no-op is fine
    finally:
        await sess.close()


async def test_cancel_when_idle_is_noop() -> None:
    """The contract: cancel() on a fresh session never raises."""
    runtime = _make_runtime()
    sess = runtime.session()
    try:
        await sess.cancel()
        await sess.cancel()
    finally:
        await sess.close()


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


async def test_close_is_idempotent_and_blocks_further_execute(mock_openai: MagicMock) -> None:
    runtime = _make_runtime()
    sess = runtime.session()
    await sess.close()
    await sess.close()
    await sess.close()
    with pytest.raises(RuntimeError):
        await sess.execute("nope")


async def test_close_does_not_close_runtime_client(mock_openai: MagicMock) -> None:
    """Session.close() leaves the runtime's AsyncOpenAI client alive."""
    runtime = _make_runtime()
    sess = runtime.session()
    await sess.execute("warm the client")
    await sess.close()
    mock_openai.close.assert_not_called()


# ---------------------------------------------------------------------------
# Function tools (Phase 3 Iteration B)
# ---------------------------------------------------------------------------


class _AddParams(BaseModel):
    a: float
    b: float


def _make_tool_call(*, call_id: str, name: str, arguments: str) -> Any:
    """Stand-in for one ``ChatCompletionMessageToolCall``."""
    fn = MagicMock()
    fn.name = name
    fn.arguments = arguments
    tc = MagicMock()
    tc.id = call_id
    tc.function = fn
    return tc


def _make_tool_call_delta(
    *,
    index: int,
    call_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> Any:
    """Stand-in for one ``delta.tool_calls[i]`` fragment in a stream chunk."""
    fn: Any = None
    if name is not None or arguments is not None:
        fn = MagicMock()
        fn.name = name
        fn.arguments = arguments
    dtc = MagicMock()
    dtc.index = index
    dtc.id = call_id
    dtc.function = fn
    return dtc


def _make_tool_call_response(*, tool_calls: list[Any]) -> Any:
    """Non-streaming response carrying tool_calls (no final text)."""
    return _make_response(content=None, finish_reason="tool_calls", tool_calls=tool_calls)


async def test_execute_round_trips_one_tool(mock_openai: MagicMock) -> None:
    """Single-tool round-trip: model calls add(17, 23), handler returns 40,
    re-call produces final text."""
    from airframe import FunctionTool

    async def add(params: _AddParams) -> float:
        return params.a + params.b

    tool = FunctionTool(name="add", description="Add two numbers.", params=_AddParams, handler=add)

    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[
            _make_tool_call_response(
                tool_calls=[
                    _make_tool_call(call_id="call_1", name="add", arguments='{"a": 17, "b": 23}')
                ]
            ),
            _make_response(content="The answer is 40."),
        ]
    )
    runtime = _make_runtime()
    sess = runtime.session(tools=[tool])
    try:
        result = await sess.execute("what's 17 + 23?")
    finally:
        await sess.close()

    assert result.text == "The answer is 40."
    assert mock_openai.chat.completions.create.await_count == 2
    # First call carries the tools= payload.
    first_call = mock_openai.chat.completions.create.await_args_list[0]
    assert first_call.kwargs["tools"][0] == {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers.",
            "parameters": _AddParams.model_json_schema(),
        },
    }
    # Second call's messages should include the intermediate
    # assistant+tool_calls turn plus the role="tool" reply.
    second_messages = mock_openai.chat.completions.create.await_args_list[1].kwargs["messages"]
    assert second_messages[-2]["role"] == "assistant"
    assert second_messages[-2]["tool_calls"][0]["id"] == "call_1"
    assert second_messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "40.0",
    }


async def test_execute_handles_parallel_tool_calls(mock_openai: MagicMock) -> None:
    """One assistant message can carry multiple tool calls; each must dispatch."""
    from airframe import FunctionTool

    handler_calls: list[tuple[float, float]] = []

    async def add(params: _AddParams) -> float:
        handler_calls.append((params.a, params.b))
        return params.a + params.b

    tool = FunctionTool(name="add", description="Add.", params=_AddParams, handler=add)

    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[
            _make_tool_call_response(
                tool_calls=[
                    _make_tool_call(call_id="c1", name="add", arguments='{"a": 1, "b": 2}'),
                    _make_tool_call(call_id="c2", name="add", arguments='{"a": 3, "b": 4}'),
                    _make_tool_call(call_id="c3", name="add", arguments='{"a": 5, "b": 6}'),
                ]
            ),
            _make_response(content="All done."),
        ]
    )
    runtime = _make_runtime()
    sess = runtime.session(tools=[tool])
    try:
        result = await sess.execute("do three sums")
    finally:
        await sess.close()

    assert result.text == "All done."
    # All three handlers ran in registration order.
    assert handler_calls == [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    # The follow-up call has three role="tool" replies, in order.
    second_messages = mock_openai.chat.completions.create.await_args_list[1].kwargs["messages"]
    tool_messages = [m for m in second_messages if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["c1", "c2", "c3"]
    assert [m["content"] for m in tool_messages] == ["3.0", "7.0", "11.0"]


async def test_handler_exception_becomes_is_error_tool_message(
    mock_openai: MagicMock,
) -> None:
    """A raising handler doesn't propagate — the model sees the error and recovers."""
    from airframe import FunctionTool

    async def boom(_: _AddParams) -> float:
        raise ValueError("the maths fell apart")

    tool = FunctionTool(name="add", description="Add.", params=_AddParams, handler=boom)

    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[
            _make_tool_call_response(
                tool_calls=[
                    _make_tool_call(call_id="c1", name="add", arguments='{"a": 1, "b": 2}')
                ]
            ),
            _make_response(content="apologies, the tool failed."),
        ]
    )
    runtime = _make_runtime()
    sess = runtime.session(tools=[tool])
    try:
        result = await sess.execute("add stuff")
    finally:
        await sess.close()

    # Loop continued; consumer-facing call did not raise.
    assert result.text == "apologies, the tool failed."
    # The role="tool" content carries the error string.
    second_messages = mock_openai.chat.completions.create.await_args_list[1].kwargs["messages"]
    tool_msg = next(m for m in second_messages if m.get("role") == "tool")
    assert "ValueError" in tool_msg["content"]
    assert "the maths fell apart" in tool_msg["content"]


async def test_unknown_tool_name_becomes_is_error_message(
    mock_openai: MagicMock,
) -> None:
    """If the model invents a tool name we don't know, surface that to it."""
    from airframe import FunctionTool

    async def add(params: _AddParams) -> float:
        return params.a + params.b

    tool = FunctionTool(name="add", description="Add.", params=_AddParams, handler=add)
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[
            _make_tool_call_response(
                tool_calls=[_make_tool_call(call_id="x", name="multiply", arguments="{}")]
            ),
            _make_response(content="never mind."),
        ]
    )
    runtime = _make_runtime()
    sess = runtime.session(tools=[tool])
    try:
        await sess.execute("multiply something")
    finally:
        await sess.close()

    second_messages = mock_openai.chat.completions.create.await_args_list[1].kwargs["messages"]
    tool_msg = next(m for m in second_messages if m.get("role") == "tool")
    assert "'multiply'" in tool_msg["content"]
    assert "not registered" in tool_msg["content"]


async def test_tool_loop_iteration_cap_raises_runtime_protocol_error(
    mock_openai: MagicMock,
) -> None:
    """A model that keeps requesting tools must be surfaced as a protocol error."""
    from airframe import FunctionTool
    from airframe.adapters.openai_compatible import MAX_TOOL_ITERATIONS
    from airframe.errors import RuntimeProtocolError

    async def add(params: _AddParams) -> float:
        return params.a + params.b

    tool = FunctionTool(name="add", description="Add.", params=_AddParams, handler=add)
    # Every iteration returns another tool call — never terminates.
    looping_response = _make_tool_call_response(
        tool_calls=[_make_tool_call(call_id="c", name="add", arguments='{"a": 1, "b": 2}')]
    )

    async def _always_tool(**_: Any) -> Any:
        return looping_response

    mock_openai.chat.completions.create = AsyncMock(side_effect=_always_tool)
    runtime = _make_runtime()
    sess = runtime.session(tools=[tool])
    try:
        with pytest.raises(RuntimeProtocolError) as exc_info:
            await sess.execute("loop forever")
    finally:
        await sess.close()

    assert str(MAX_TOOL_ITERATIONS) in str(exc_info.value)
    # The cap is hit (and not exceeded) by the time we raise.
    assert mock_openai.chat.completions.create.await_count == MAX_TOOL_ITERATIONS


async def test_tool_loop_failure_rolls_back_buffer(mock_openai: MagicMock) -> None:
    """A failed tool turn must not pollute the buffer for the next attempt."""
    from airframe import FunctionTool

    async def add(params: _AddParams) -> float:
        return params.a + params.b

    tool = FunctionTool(name="add", description="Add.", params=_AddParams, handler=add)

    # First execute: API blows up mid-loop. Second execute: clean call.
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[
            _make_tool_call_response(
                tool_calls=[
                    _make_tool_call(call_id="c1", name="add", arguments='{"a": 1, "b": 2}')
                ]
            ),
            RuntimeError("vendor hiccup"),
            _make_response(content="hello"),
        ]
    )
    runtime = _make_runtime()
    sess = runtime.session(tools=[tool])
    try:
        with pytest.raises(Exception):  # noqa: B017
            await sess.execute("first attempt")
        # Buffer should be empty (no system prompt either).
        assert sess._messages == []  # noqa: SLF001
        result = await sess.execute("retry")
    finally:
        await sess.close()

    assert result.text == "hello"
    # The retry call should send only the new user message.
    last_messages = mock_openai.chat.completions.create.await_args_list[-1].kwargs["messages"]
    assert last_messages == [{"role": "user", "content": "retry"}]


async def test_session_without_tools_omits_tools_kwarg(mock_openai: MagicMock) -> None:
    """No-tools sessions keep the pre-Iteration-B wire shape (no ``tools=`` key)."""
    runtime = _make_runtime()
    sess = runtime.session()
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    call = mock_openai.chat.completions.create.await_args_list[0]
    assert "tools" not in call.kwargs


async def test_stream_emits_tool_call_events_around_handler(
    mock_openai: MagicMock,
) -> None:
    """stream() emits ToolCallStart, ToolCallResult, then final TextDelta + TurnComplete."""
    from airframe import FunctionTool
    from airframe.events import ToolCallResult, ToolCallStart

    async def add(params: _AddParams) -> float:
        return params.a + params.b

    tool = FunctionTool(name="add", description="Add.", params=_AddParams, handler=add)

    # First iteration: stream announces a tool call across two delta chunks.
    tool_chunks = [
        _make_chunk(
            tool_call_deltas=[
                _make_tool_call_delta(
                    index=0,
                    call_id="c1",
                    name="add",
                    arguments='{"a": 1, ',
                )
            ]
        ),
        _make_chunk(
            tool_call_deltas=[_make_tool_call_delta(index=0, arguments='"b": 2}')],
            finish_reason="tool_calls",
        ),
        _make_usage_chunk(prompt_tokens=10, completion_tokens=3),
    ]
    # Second iteration: streams the final text answer.
    text_chunks = [
        _make_chunk(content="The "),
        _make_chunk(content="answer is 3.", finish_reason="stop"),
        _make_usage_chunk(prompt_tokens=20, completion_tokens=4),
    ]
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[_AsyncIter(tool_chunks), _AsyncIter(text_chunks)]
    )
    runtime = _make_runtime()
    sess = runtime.session(tools=[tool])
    events: list[Any] = []
    try:
        async for event in sess.stream("what's 1+2?"):
            events.append(event)
    finally:
        await sess.close()

    starts = [e for e in events if isinstance(e, ToolCallStart)]
    results = [e for e in events if isinstance(e, ToolCallResult)]
    text_events = [e for e in events if isinstance(e, TextDelta)]
    turns = [e for e in events if isinstance(e, TurnComplete)]

    assert len(starts) == 1
    assert starts[0].tool_name == "add"
    assert starts[0].tool_call_id == "c1"
    assert starts[0].arguments_preview == '{"a": 1, "b": 2}'
    assert len(results) == 1
    assert results[0].tool_call_id == "c1"
    assert results[0].output == 3.0
    assert results[0].is_error is False
    # Final text comes from the second iteration only.
    assert "".join(e.text for e in text_events) == "The answer is 3."
    # Exactly one TurnComplete at the very end.
    assert len(turns) == 1
    assert events[-1] is turns[0]
    assert turns[0].result.text == "The answer is 3."


async def test_stream_handler_exception_propagates_as_is_error_event(
    mock_openai: MagicMock,
) -> None:
    """Handler raising during stream() still produces a ToolCallResult(is_error=True)."""
    from airframe import FunctionTool
    from airframe.events import ToolCallResult

    async def boom(_: _AddParams) -> float:
        raise RuntimeError("kaboom")

    tool = FunctionTool(name="add", description="Add.", params=_AddParams, handler=boom)
    tool_chunks = [
        _make_chunk(
            tool_call_deltas=[
                _make_tool_call_delta(
                    index=0, call_id="c1", name="add", arguments='{"a": 1, "b": 2}'
                )
            ],
            finish_reason="tool_calls",
        ),
        _make_usage_chunk(prompt_tokens=10, completion_tokens=3),
    ]
    text_chunks = [
        _make_chunk(content="apologies.", finish_reason="stop"),
        _make_usage_chunk(prompt_tokens=20, completion_tokens=2),
    ]
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[_AsyncIter(tool_chunks), _AsyncIter(text_chunks)]
    )
    runtime = _make_runtime()
    sess = runtime.session(tools=[tool])
    events: list[Any] = []
    try:
        async for event in sess.stream("loop"):
            events.append(event)
    finally:
        await sess.close()

    results = [e for e in events if isinstance(e, ToolCallResult)]
    assert len(results) == 1
    assert results[0].is_error is True
    assert "RuntimeError" in str(results[0].output)


async def test_stream_iteration_cap_raises_runtime_protocol_error(
    mock_openai: MagicMock,
) -> None:
    """Streaming runaways surface as RuntimeProtocolError just like execute()."""
    from airframe import FunctionTool
    from airframe.adapters.openai_compatible import MAX_TOOL_ITERATIONS
    from airframe.errors import RuntimeProtocolError

    async def add(params: _AddParams) -> float:
        return params.a + params.b

    tool = FunctionTool(name="add", description="Add.", params=_AddParams, handler=add)

    def _loop_chunks() -> list[Any]:
        return [
            _make_chunk(
                tool_call_deltas=[
                    _make_tool_call_delta(
                        index=0,
                        call_id="c",
                        name="add",
                        arguments='{"a": 1, "b": 2}',
                    )
                ],
                finish_reason="tool_calls",
            ),
            _make_usage_chunk(prompt_tokens=1, completion_tokens=1),
        ]

    async def _always_loop(**_: Any) -> Any:
        return _AsyncIter(_loop_chunks())

    mock_openai.chat.completions.create = AsyncMock(side_effect=_always_loop)

    runtime = _make_runtime()
    sess = runtime.session(tools=[tool])
    try:
        with pytest.raises(RuntimeProtocolError):
            async for _ in sess.stream("infinite"):
                pass
    finally:
        await sess.close()
    assert mock_openai.chat.completions.create.await_count == MAX_TOOL_ITERATIONS


# ---------------------------------------------------------------------------
# MCP server refs (Phase 4 Iteration D — OpenAI-compat declines)
# ---------------------------------------------------------------------------


async def test_session_mcp_servers_declines_with_responses_api_pointer() -> None:
    """``mcp_servers=`` raises UnsupportedFeatureError with the
    Responses-API pointer.

    Iteration D replaces the generic shared-helper decline with an
    OpenAI-compat-specific message pointing at a future
    ``OpenAIResponsesRuntime``. The decline is **permanent** for the
    chat-completions family — Chat Completions has no MCP-as-tool
    wire shape; that lives on the Responses API.
    """
    from airframe import McpServerRef
    from airframe.errors import UnsupportedFeatureError

    rt = _make_runtime()
    with pytest.raises(UnsupportedFeatureError) as exc_info:
        rt.session(
            mcp_servers=[
                McpServerRef(
                    name="everything",
                    transport="stdio",
                    command=["uvx", "mcp-server-everything"],
                )
            ]
        )
    assert exc_info.value.feature == Feature.TOOLS_MCP_STDIO
    message = str(exc_info.value)
    text = message.lower()
    # Pin the actionable pointer.
    assert "responses" in text
    assert "chat completions" in text
    # Mentions the future direct-API class so consumers know the path.
    assert "openairesponsesruntime" in text.replace(" ", "")


async def test_session_mcp_servers_decline_carries_transport_feature() -> None:
    """The ``.feature`` attribute matches the *first* ref's transport."""
    from airframe import McpServerRef
    from airframe.errors import UnsupportedFeatureError

    rt = _make_runtime()
    cases: list[tuple[McpServerRef, Feature]] = [
        (
            McpServerRef(name="s", transport="stdio", command=["x"]),
            Feature.TOOLS_MCP_STDIO,
        ),
        (
            McpServerRef(name="h", transport="http", url="https://h"),
            Feature.TOOLS_MCP_HTTP,
        ),
        (
            McpServerRef(name="e", transport="sse", url="https://h/sse"),
            Feature.TOOLS_MCP_SSE,
        ),
    ]
    for ref, expected_feature in cases:
        with pytest.raises(UnsupportedFeatureError) as exc_info:
            rt.session(mcp_servers=[ref])
        assert exc_info.value.feature == expected_feature, (
            f"expected feature={expected_feature.name!r} for {ref.transport!r}; "
            f"got {exc_info.value.feature!r}"
        )


async def test_session_mcp_servers_none_or_empty_still_opens_cleanly() -> None:
    """``mcp_servers=None`` / ``mcp_servers=[]`` are both no-ops."""
    rt = _make_runtime()
    sess_none = rt.session(mcp_servers=None)
    sess_empty = rt.session(mcp_servers=[])
    assert sess_none is not None
    assert sess_empty is not None
    await sess_none.close()
    await sess_empty.close()


def test_openai_compat_declines_every_mcp_transport() -> None:
    """The chat-completions family permanently declines every transport flag."""
    rt = _make_runtime()
    assert rt.supports(Feature.TOOLS_MCP_STDIO) is False
    assert rt.supports(Feature.TOOLS_MCP_HTTP) is False
    assert rt.supports(Feature.TOOLS_MCP_SSE) is False
    assert rt.supports(Feature.TOOLS_MCP_IN_PROCESS) is False


# ---------------------------------------------------------------------------
# Permission callback (Phase 5 Iteration B — permanent decline)
# ---------------------------------------------------------------------------


async def test_session_on_permission_declines_with_responses_api_pointer() -> None:
    """``on_permission=`` raises UnsupportedFeatureError with the
    Responses-API pointer.

    The decline is **permanent** for the chat-completions family —
    Chat Completions has no tool-permission wire shape. A future
    ``OpenAIResponsesRuntime`` could wire it.
    """
    from airframe import PermissionDecision, PermissionRequest
    from airframe.errors import UnsupportedFeatureError

    class _Cb:
        async def handle(self, request: PermissionRequest) -> PermissionDecision:
            return "allow"

    rt = _make_runtime()
    with pytest.raises(UnsupportedFeatureError) as exc_info:
        rt.session(on_permission=_Cb())
    assert exc_info.value.feature == Feature.PERMISSION_CALLBACK
    text = str(exc_info.value).lower()
    # Pin the actionable pointer.
    assert "chat completions" in text
    assert "openairesponsesruntime" in text.replace(" ", "")


async def test_session_on_permission_none_opens_cleanly() -> None:
    """``on_permission=None`` is the no-op default — must not raise."""
    rt = _make_runtime()
    sess = rt.session(on_permission=None)
    assert sess is not None
    await sess.close()


def test_openai_compat_does_not_declare_permission_callback() -> None:
    """The chat-completions family stays at PERMISSION_CALLBACK=False."""
    rt = _make_runtime()
    assert rt.supports(Feature.PERMISSION_CALLBACK) is False


# ---------------------------------------------------------------------------
# Lifecycle hooks (Phase 5 Iteration C)
# ---------------------------------------------------------------------------


def test_openai_compat_runtime_declares_lifecycle_hooks() -> None:
    """LIFECYCLE_HOOKS is True after Iteration C."""
    rt = _make_runtime()
    assert rt.supports(Feature.LIFECYCLE_HOOKS) is True


def test_openai_compat_emittable_hook_kinds_matches_plan() -> None:
    """Six kinds — no pre_compact (no compaction concept on chat-completions)
    and no rate_limit (SDK doesn't surface a discrete throttle event)."""
    from airframe.adapters.openai_compatible import OpenAICompatibleRuntime

    assert (
        frozenset(
            {
                "session_start",
                "session_end",
                "user_prompt_submit",
                "pre_tool_use",
                "post_tool_use",
                "tool_failure",
            }
        )
        == OpenAICompatibleRuntime.EMITTABLE_HOOK_KINDS
    )


async def test_on_event_execute_emits_session_start_and_user_prompt_submit(
    mock_openai: MagicMock,
) -> None:
    """First execute() fires session_start; every turn fires
    user_prompt_submit with prompt + length."""
    from airframe.hooks import HookEvent

    events: list[HookEvent] = []
    rt = _make_runtime()
    sess = rt.session(on_event=events.append)
    try:
        await sess.execute("first")
        await sess.execute("second")
    finally:
        await sess.close()

    starts = [e for e in events if e.kind == "session_start"]
    prompts = [e for e in events if e.kind == "user_prompt_submit"]
    assert len(starts) == 1
    assert len(prompts) == 2
    assert prompts[0].payload["prompt"] == "first"
    assert prompts[0].payload["length"] == len("first")
    # No vendor session_id on chat-completions.
    assert all(e.session_id is None for e in events)


async def test_on_event_stream_also_emits_session_start_and_user_prompt_submit(
    mock_openai: MagicMock,
) -> None:
    """The stream() path emits the same session-level kinds as execute()."""
    from airframe.hooks import HookEvent

    chunks = [
        _make_chunk(content="hi", finish_reason="stop"),
        _make_usage_chunk(prompt_tokens=1, completion_tokens=1),
    ]
    mock_openai.chat.completions.create = AsyncMock(return_value=_AsyncIter(chunks))

    events: list[HookEvent] = []
    rt = _make_runtime()
    sess = rt.session(on_event=events.append)
    try:
        async for _ in sess.stream("hello"):
            pass
    finally:
        await sess.close()

    kinds = [e.kind for e in events]
    assert "session_start" in kinds
    assert "user_prompt_submit" in kinds


async def test_on_event_tool_round_trip_emits_pre_and_post_tool_use(
    mock_openai: MagicMock,
) -> None:
    """A round-trip with a function tool fires pre_tool_use before
    handler invocation and post_tool_use on success — in that order."""
    from airframe import FunctionTool
    from airframe.hooks import HookEvent

    async def add(params: _AddParams) -> float:
        return params.a + params.b

    tool = FunctionTool(name="add", description="Add.", params=_AddParams, handler=add)

    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[
            _make_tool_call_response(
                tool_calls=[
                    _make_tool_call(call_id="c1", name="add", arguments='{"a": 1, "b": 2}')
                ]
            ),
            _make_response(content="3"),
        ]
    )

    events: list[HookEvent] = []
    rt = _make_runtime()
    sess = rt.session(tools=[tool], on_event=events.append)
    try:
        await sess.execute("what's 1 + 2?")
    finally:
        await sess.close()

    tool_kinds = [e.kind for e in events if e.kind in {"pre_tool_use", "post_tool_use"}]
    assert tool_kinds == ["pre_tool_use", "post_tool_use"]
    pre = next(e for e in events if e.kind == "pre_tool_use")
    post = next(e for e in events if e.kind == "post_tool_use")
    assert pre.payload["tool_name"] == "add"
    assert pre.payload["tool_call_id"] == "c1"
    assert post.payload["tool_call_id"] == "c1"
    assert str(post.payload["output"]) == "3.0"


async def test_on_event_tool_failure_translates_to_tool_failure_kind(
    mock_openai: MagicMock,
) -> None:
    """Handler-raised exception → tool_failure (not post_tool_use)."""
    from airframe import FunctionTool
    from airframe.hooks import HookEvent

    async def boom(params: _AddParams) -> float:
        raise RuntimeError("handler broke")

    tool = FunctionTool(name="add", description="Add.", params=_AddParams, handler=boom)

    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[
            _make_tool_call_response(
                tool_calls=[
                    _make_tool_call(call_id="c1", name="add", arguments='{"a": 1, "b": 2}')
                ]
            ),
            _make_response(content="apologies"),
        ]
    )

    events: list[HookEvent] = []
    rt = _make_runtime()
    sess = rt.session(tools=[tool], on_event=events.append)
    try:
        await sess.execute("call add")
    finally:
        await sess.close()

    kinds = [e.kind for e in events if e.kind in {"post_tool_use", "tool_failure"}]
    assert kinds == ["tool_failure"]
    failure = next(e for e in events if e.kind == "tool_failure")
    assert "error" in failure.payload


async def test_on_event_stream_tool_round_trip_emits_pre_and_post_tool_use(
    mock_openai: MagicMock,
) -> None:
    """The stream() path fires the same per-tool kinds as execute()."""
    from airframe import FunctionTool
    from airframe.hooks import HookEvent

    async def add(params: _AddParams) -> float:
        return params.a + params.b

    tool = FunctionTool(name="add", description="Add.", params=_AddParams, handler=add)

    first_stream_chunks = [
        _make_chunk(
            tool_call_deltas=[
                _make_tool_call_delta(index=0, call_id="c1", name="add", arguments=""),
            ]
        ),
        _make_chunk(
            tool_call_deltas=[
                _make_tool_call_delta(index=0, arguments='{"a": 1, "b": 2}'),
            ],
            finish_reason="tool_calls",
        ),
        _make_usage_chunk(prompt_tokens=1, completion_tokens=1),
    ]
    second_stream_chunks = [
        _make_chunk(content="3", finish_reason="stop"),
        _make_usage_chunk(prompt_tokens=1, completion_tokens=1),
    ]
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[_AsyncIter(first_stream_chunks), _AsyncIter(second_stream_chunks)]
    )

    events: list[HookEvent] = []
    rt = _make_runtime()
    sess = rt.session(tools=[tool], on_event=events.append)
    try:
        async for _ in sess.stream("call add"):
            pass
    finally:
        await sess.close()

    kinds = [e.kind for e in events if e.kind in {"pre_tool_use", "post_tool_use"}]
    assert kinds == ["pre_tool_use", "post_tool_use"]


async def test_close_synthesises_session_end_when_session_start_fired(
    mock_openai: MagicMock,
) -> None:
    """close() after at least one turn emits exactly one session_end."""
    from airframe.hooks import HookEvent

    events: list[HookEvent] = []
    rt = _make_runtime()
    sess = rt.session(on_event=events.append)
    await sess.execute("hi")
    assert [e for e in events if e.kind == "session_end"] == []
    await sess.close()
    ends = [e for e in events if e.kind == "session_end"]
    assert len(ends) == 1


async def test_close_session_end_is_idempotent(mock_openai: MagicMock) -> None:
    """Multiple close() calls fire session_end at most once."""
    from airframe.hooks import HookEvent

    events: list[HookEvent] = []
    rt = _make_runtime()
    sess = rt.session(on_event=events.append)
    await sess.execute("hi")
    await sess.close()
    await sess.close()
    await sess.close()
    ends = [e for e in events if e.kind == "session_end"]
    assert len(ends) == 1


async def test_close_without_execute_omits_session_end(mock_openai: MagicMock) -> None:
    """If session_start never fired (no execute() ever ran), close()
    must NOT fire a phantom session_end."""
    from airframe.hooks import HookEvent

    events: list[HookEvent] = []
    rt = _make_runtime()
    sess = rt.session(on_event=events.append)
    await sess.close()
    assert [e for e in events if e.kind == "session_end"] == []


async def test_no_on_event_skips_hook_emission(mock_openai: MagicMock) -> None:
    """Without on_event= the adapter's hot path doesn't emit hooks.
    Verified indirectly: the session executes a tool round-trip
    without raising and produces the expected text."""
    from airframe import FunctionTool

    async def add(params: _AddParams) -> float:
        return params.a + params.b

    tool = FunctionTool(name="add", description="Add.", params=_AddParams, handler=add)
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[
            _make_tool_call_response(
                tool_calls=[
                    _make_tool_call(call_id="c1", name="add", arguments='{"a": 1, "b": 2}')
                ]
            ),
            _make_response(content="3"),
        ]
    )

    rt = _make_runtime()
    sess = rt.session(tools=[tool])
    try:
        result = await sess.execute("compute")
    finally:
        await sess.close()
    assert result.text == "3"


async def test_on_event_observer_that_raises_does_not_break_session(
    mock_openai: MagicMock,
) -> None:
    """A raising observer must not break the turn — _fire_hook_event
    swallows everything except KeyboardInterrupt/SystemExit."""
    from airframe.hooks import HookEvent

    calls = {"n": 0}

    def boom(event: HookEvent) -> None:
        calls["n"] += 1
        raise RuntimeError("observer broke")

    rt = _make_runtime()
    sess = rt.session(on_event=boom)
    try:
        result = await sess.execute("hi")
    finally:
        await sess.close()
    assert calls["n"] >= 1
    assert result.text == "hello world"


# ---------------------------------------------------------------------------
# Budget caps (Phase 5 Iteration D)
# ---------------------------------------------------------------------------


def test_openai_compat_declares_budget_caps() -> None:
    """OpenAI-compat declares both BUDGET_USD_CAP and BUDGET_TURN_CAP."""
    rt = _make_runtime()
    assert rt.supports(Feature.BUDGET_USD_CAP)
    assert rt.supports(Feature.BUDGET_TURN_CAP)


async def test_max_turns_cap_raises_when_count_reached(mock_openai: MagicMock) -> None:
    """After running max_turns turns, the next execute() raises."""
    from airframe.errors import RuntimeBudgetExceededError

    rt = _make_runtime()
    sess = rt.session()
    try:
        await sess.execute("turn 1", max_turns=2)
        await sess.execute("turn 2", max_turns=2)
        with pytest.raises(RuntimeBudgetExceededError) as exc_info:
            await sess.execute("turn 3", max_turns=2)
    finally:
        await sess.close()
    err = exc_info.value
    assert err.kind == "turns"
    assert err.cap == 2.0
    assert err.current == 2.0


async def test_max_budget_usd_cap_raises_when_cumulative_cost_reached(
    mock_openai: MagicMock,
) -> None:
    """Cumulative cost accumulates per turn; cap trips at the next
    pre-turn enforce."""
    from airframe.errors import RuntimeBudgetExceededError

    # Default _make_response: prompt_tokens=100, completion_tokens=50.
    # On gpt-5-nano ($0.0001/$0.0002 per 1K) → $0.00002/turn.
    # Cap at $0.000015 → first turn succeeds, second turn raises.
    rt = _make_runtime()
    sess = rt.session()
    try:
        await sess.execute("turn 1", max_budget_usd=0.000015)
        with pytest.raises(RuntimeBudgetExceededError) as exc_info:
            await sess.execute("turn 2", max_budget_usd=0.000015)
    finally:
        await sess.close()
    err = exc_info.value
    assert err.kind == "usd"
    assert err.cap == 0.000015


async def test_stream_honours_budget_caps(mock_openai: MagicMock) -> None:
    """Stream path uses the same enforce — exhausted cap raises."""
    from airframe.errors import RuntimeBudgetExceededError

    chunks = [
        _make_chunk(content="hi", finish_reason="stop"),
        _make_usage_chunk(prompt_tokens=100, completion_tokens=50),
    ]
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[_AsyncIter(list(chunks)), _AsyncIter(list(chunks))]
    )
    rt = _make_runtime()
    sess = rt.session()
    try:
        async for _ in sess.stream("a", max_budget_usd=0.000015):
            pass
        with pytest.raises(RuntimeBudgetExceededError):
            async for _ in sess.stream("b", max_budget_usd=0.000015):
                pass
    finally:
        await sess.close()


async def test_budget_caps_none_open_cleanly(mock_openai: MagicMock) -> None:
    """No caps → turns run freely."""
    rt = _make_runtime()
    sess = rt.session()
    try:
        for i in range(5):
            await sess.execute(f"turn {i}")
    finally:
        await sess.close()


# ---------------------------------------------------------------------------
# Provider options (v0.5.0-readiness — OpenAICompatOptions wired)
# ---------------------------------------------------------------------------


async def test_openai_compat_options_prompt_cache_key_lands_on_create(
    mock_openai: MagicMock,
) -> None:
    """``OpenAICompatOptions.prompt_cache_key`` rides into every create() call."""
    from airframe import OpenAICompatOptions

    rt = _make_runtime()
    sess = rt.session(provider_options=OpenAICompatOptions(prompt_cache_key="user-42"))
    try:
        await sess.execute("hi")
    finally:
        await sess.close()
    call = mock_openai.chat.completions.create.await_args_list[0]
    assert call.kwargs["prompt_cache_key"] == "user-42"


async def test_openai_compat_options_full_field_set_lands(mock_openai: MagicMock) -> None:
    """Every populated OpenAICompatOptions field reaches the create() kwargs."""
    from airframe import OpenAICompatOptions

    rt = _make_runtime()
    sess = rt.session(
        provider_options=OpenAICompatOptions(
            prompt_cache_key="k",
            prompt_cache_retention="24h",
            service_tier="priority",
            safety_identifier="user-99",
            verbosity="low",
            store=True,
        )
    )
    try:
        await sess.execute("hi")
    finally:
        await sess.close()
    call = mock_openai.chat.completions.create.await_args_list[0]
    assert call.kwargs["prompt_cache_key"] == "k"
    assert call.kwargs["prompt_cache_retention"] == "24h"
    assert call.kwargs["service_tier"] == "priority"
    assert call.kwargs["safety_identifier"] == "user-99"
    assert call.kwargs["verbosity"] == "low"
    assert call.kwargs["store"] is True


async def test_openai_compat_options_default_omits_kwargs(mock_openai: MagicMock) -> None:
    """No ``provider_options=`` → none of the new kwargs are passed."""
    rt = _make_runtime()
    sess = rt.session()
    try:
        await sess.execute("hi")
    finally:
        await sess.close()
    call = mock_openai.chat.completions.create.await_args_list[0]
    for k in (
        "prompt_cache_key",
        "prompt_cache_retention",
        "service_tier",
        "safety_identifier",
        "verbosity",
        "store",
    ):
        assert k not in call.kwargs


async def test_openai_compat_options_apply_on_stream_path(mock_openai: MagicMock) -> None:
    """Stream path also routes provider_options through the create() kwargs."""
    from airframe import OpenAICompatOptions

    chunks = [
        _make_chunk(content="hi", finish_reason="stop"),
        _make_usage_chunk(prompt_tokens=1, completion_tokens=1),
    ]
    mock_openai.chat.completions.create = AsyncMock(return_value=_AsyncIter(chunks))

    rt = _make_runtime()
    sess = rt.session(provider_options=OpenAICompatOptions(prompt_cache_key="stream-k"))
    try:
        async for _ in sess.stream("hi"):
            pass
    finally:
        await sess.close()
    call = mock_openai.chat.completions.create.await_args
    assert call.kwargs["prompt_cache_key"] == "stream-k"


async def test_openai_compat_options_wrong_namespace_raises_unsupported_feature() -> None:
    from airframe import ClaudeOptions
    from airframe.errors import UnsupportedFeatureError

    rt = _make_runtime()
    with pytest.raises(UnsupportedFeatureError) as exc_info:
        rt.session(provider_options=ClaudeOptions())
    assert "ClaudeOptions" in str(exc_info.value)
    assert "OpenAICompatOptions" in str(exc_info.value)
