"""Unit tests for :class:`CodexAgentSession`.

Phase 1 Iteration F — fourth and final per-vendor session. Mocks the
``openai_codex_sdk`` SDK at the boundary so we exercise the
:meth:`Thread.run_streamed` event translation, the resume= path
through :meth:`Codex.resume_thread`, and cancellation via
:class:`AbortController` / :attr:`TurnOptions.signal` without spawning
a CLI subprocess.

Live-vendor probes belong in ``airframe.testing.integration``
(Phase 1 work).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from airframe.adapters.codex import CodexAgentSession, CodexRuntime
from airframe.errors import RuntimeCancelledError
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
        input_tokens: int = 50,
        output_tokens: int = 25,
        cached_input_tokens: int = 0,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cached_input_tokens = cached_input_tokens


class _FakeAgentMessageItem:
    def __init__(self, *, id: str, text: str) -> None:
        self.id = id
        self.type = "agent_message"
        self.text = text


class _FakeReasoningItem:
    def __init__(self, *, id: str, text: str) -> None:
        self.id = id
        self.type = "reasoning"
        self.text = text


class _FakeItemUpdated:
    def __init__(self, item: Any) -> None:
        self.type = "item.updated"
        self.item = item


class _FakeItemCompleted:
    def __init__(self, item: Any) -> None:
        self.type = "item.completed"
        self.item = item


class _FakeTurnCompleted:
    def __init__(self, usage: Any) -> None:
        self.type = "turn.completed"
        self.usage = usage


class _FakeTurnFailed:
    def __init__(self, message: str) -> None:
        self.type = "turn.failed"
        self.error = MagicMock()
        self.error.message = message


_UNSET = object()


class _FakeTurn:
    """Stand-in for ``openai_codex_sdk.Turn``."""

    def __init__(
        self,
        *,
        final_response: str = '{"summary": "ok", "count": 42}',
        items: list[Any] | None = None,
        usage: Any = _UNSET,
    ) -> None:
        self.final_response = final_response
        self.items = items or []
        self.usage = _FakeUsage() if usage is _UNSET else usage


class _FakeStreamedTurn:
    """Stand-in for ``StreamedTurn`` — has an `.events` async iterator."""

    def __init__(self, events: list[Any]) -> None:
        self._events = events

    @property
    def events(self) -> Any:
        return self._make_iter()

    async def _make_iter(self) -> Any:
        for e in self._events:
            yield e


# ---------------------------------------------------------------------------
# Fixture: patch the SDK symbols the adapter imports lazily
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sdk(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    import openai_codex_sdk as sdk
    from openai_codex_sdk import abort as abort_mod
    from openai_codex_sdk import types as types_mod

    # --- Thread + Client mocks ----------------------------------------
    mock_thread = MagicMock()
    mock_thread.run = AsyncMock(return_value=_FakeTurn())
    mock_thread.run_streamed = AsyncMock()
    mock_thread.id = None

    mock_client = MagicMock()
    mock_client.start_thread = MagicMock(return_value=mock_thread)
    mock_client.resume_thread = MagicMock(return_value=mock_thread)

    monkeypatch.setattr(sdk, "Codex", lambda options=None: mock_client)

    # --- AbortController + AbortError --------------------------------
    aborts: list[Any] = []

    class _FakeAbortSignal:
        def __init__(self) -> None:
            self.aborted = False

    class _FakeAbortController:
        def __init__(self) -> None:
            self.signal = _FakeAbortSignal()
            aborts.append(self)

        def abort(self, reason: Any = None) -> None:
            self.signal.aborted = True

    class _FakeAbortError(Exception):
        pass

    monkeypatch.setattr(sdk, "AbortController", _FakeAbortController)
    monkeypatch.setattr(abort_mod, "AbortError", _FakeAbortError)

    # --- Event/Item types — substitute with our fakes ----------------
    monkeypatch.setattr(types_mod, "AgentMessageItem", _FakeAgentMessageItem)
    monkeypatch.setattr(types_mod, "ReasoningItem", _FakeReasoningItem)
    monkeypatch.setattr(types_mod, "ItemUpdatedEvent", _FakeItemUpdated)
    monkeypatch.setattr(types_mod, "ItemCompletedEvent", _FakeItemCompleted)
    monkeypatch.setattr(types_mod, "TurnCompletedEvent", _FakeTurnCompleted)
    monkeypatch.setattr(types_mod, "TurnFailedEvent", _FakeTurnFailed)

    return {
        "client": mock_client,
        "thread": mock_thread,
        "aborts": aborts,
        "AbortError": _FakeAbortError,
    }


# ---------------------------------------------------------------------------
# Factory + capability surface
# ---------------------------------------------------------------------------


async def test_session_factory_returns_bespoke_session(mock_sdk: dict[str, Any]) -> None:
    rt = CodexRuntime()
    sess = rt.session()
    try:
        assert isinstance(sess, CodexAgentSession)
        assert isinstance(sess, AgentSession)
        assert sess.id is None
    finally:
        await sess.close()


def test_streaming_resume_cancel_features_declared() -> None:
    rt = CodexRuntime()
    assert rt.supports(Feature.STREAMING)
    assert rt.supports(Feature.SESSION_RESUME)
    assert rt.supports(Feature.CANCEL)


async def test_resume_seeds_id_and_calls_resume_thread(mock_sdk: dict[str, Any]) -> None:
    """resume=<id> seeds session.id immediately and routes to client.resume_thread."""
    mock_sdk["thread"].id = "thread-resumed"
    rt = CodexRuntime()
    sess = rt.session(resume="thread-resumed")
    try:
        assert sess.id == "thread-resumed"
        await sess.execute("continue please")
    finally:
        await sess.close()

    mock_sdk["client"].resume_thread.assert_called_with(
        "thread-resumed",
        {
            "model": "gpt-5-codex",
            "sandboxMode": "read-only",
            "skipGitRepoCheck": True,
        },
    )
    mock_sdk["client"].start_thread.assert_not_called()


async def test_fresh_session_uses_start_thread(mock_sdk: dict[str, Any]) -> None:
    """resume=None → client.start_thread, not resume_thread."""
    rt = CodexRuntime()
    sess = rt.session()
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    mock_sdk["client"].start_thread.assert_called()
    mock_sdk["client"].resume_thread.assert_not_called()


# ---------------------------------------------------------------------------
# Multi-turn lifecycle
# ---------------------------------------------------------------------------


async def test_id_populated_after_first_turn(mock_sdk: dict[str, Any]) -> None:
    """The session surfaces the live Thread.id after the first turn."""

    async def first_run(*args: Any, **kwargs: Any) -> Any:
        # Simulate the SDK setting thread.id from a thread.started event.
        mock_sdk["thread"].id = "thread-XYZ"
        return _FakeTurn(final_response='{"summary": "x", "count": 1}')

    mock_sdk["thread"].run = AsyncMock(side_effect=first_run)
    rt = CodexRuntime()
    sess = rt.session()
    try:
        assert sess.id is None
        await sess.execute("first", schema=_Schema)
        assert sess.id == "thread-XYZ"
    finally:
        await sess.close()


async def test_thread_reused_across_turns(mock_sdk: dict[str, Any]) -> None:
    """Codex Thread is multi-turn-safe — one start_thread call suffices."""
    rt = CodexRuntime()
    sess = rt.session()
    try:
        await sess.execute("a")
        await sess.execute("b")
    finally:
        await sess.close()

    assert mock_sdk["client"].start_thread.call_count == 1


async def test_schema_can_vary_per_turn_without_rebuild(mock_sdk: dict[str, Any]) -> None:
    """outputSchema is per-TurnOptions (not Thread), so schema-change keeps the thread."""
    call_idx = {"n": 0}

    async def fake_run(prompt: str, options: dict[str, Any]) -> Any:
        idx = call_idx["n"]
        call_idx["n"] += 1
        if idx == 0:
            assert "outputSchema" not in options
            return _FakeTurn(final_response="plain text")
        else:
            assert "outputSchema" in options
            return _FakeTurn(final_response='{"summary": "ok", "count": 1}')

    mock_sdk["thread"].run = AsyncMock(side_effect=fake_run)

    rt = CodexRuntime()
    sess = rt.session()
    try:
        await sess.execute("plain turn")
        await sess.execute("structured turn", schema=_Schema)
    finally:
        await sess.close()

    assert mock_sdk["client"].start_thread.call_count == 1


async def test_system_prompt_prepended_to_user_prompt(mock_sdk: dict[str, Any]) -> None:
    """Codex has no system_message setting — system= concatenates onto the prompt."""
    captured: dict[str, Any] = {}

    async def fake_run(prompt: str, options: dict[str, Any]) -> Any:
        captured["prompt"] = prompt
        return _FakeTurn(final_response="ok")

    mock_sdk["thread"].run = AsyncMock(side_effect=fake_run)

    rt = CodexRuntime()
    sess = rt.session(system="be brief")
    try:
        await sess.execute("the question")
    finally:
        await sess.close()

    assert captured["prompt"] == "be brief\n\nthe question"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


async def test_stream_yields_appendable_text_deltas(mock_sdk: dict[str, Any]) -> None:
    """Per-item tail tracking — concatenated TextDeltas reconstruct the message."""
    msg_id = "msg-1"
    events = [
        _FakeItemUpdated(_FakeAgentMessageItem(id=msg_id, text="Hel")),
        _FakeItemUpdated(_FakeAgentMessageItem(id=msg_id, text="Hello ")),
        _FakeItemUpdated(_FakeAgentMessageItem(id=msg_id, text="Hello wor")),
        _FakeItemCompleted(_FakeAgentMessageItem(id=msg_id, text="Hello world")),
        _FakeTurnCompleted(_FakeUsage(input_tokens=10, output_tokens=3)),
    ]
    mock_sdk["thread"].run_streamed = AsyncMock(return_value=_FakeStreamedTurn(events))

    rt = CodexRuntime()
    sess = rt.session()
    yielded: list[Any] = []
    try:
        async for ev in sess.stream("hi"):
            yielded.append(ev)
    finally:
        await sess.close()

    text_events = [e for e in yielded if isinstance(e, TextDelta)]
    turn_events = [e for e in yielded if isinstance(e, TurnComplete)]
    # Tails are appendable: "Hel" + "lo " + "Hello wor"[6:] + "Hello world"[9:]
    assert "".join(e.text for e in text_events) == "Hello world"
    assert len(turn_events) == 1
    assert turn_events[0].result.text == "Hello world"
    assert turn_events[0].result.cost.input_tokens == 10


async def test_stream_yields_reasoning_deltas(mock_sdk: dict[str, Any]) -> None:
    events = [
        _FakeItemUpdated(_FakeReasoningItem(id="r-1", text="let me ")),
        _FakeItemCompleted(_FakeReasoningItem(id="r-1", text="let me think")),
        _FakeItemCompleted(_FakeAgentMessageItem(id="m-1", text="done")),
        _FakeTurnCompleted(_FakeUsage()),
    ]
    mock_sdk["thread"].run_streamed = AsyncMock(return_value=_FakeStreamedTurn(events))

    rt = CodexRuntime()
    sess = rt.session()
    yielded: list[Any] = []
    try:
        async for ev in sess.stream("think hard"):
            yielded.append(ev)
    finally:
        await sess.close()

    reasoning = [e for e in yielded if isinstance(e, ReasoningDelta)]
    assert "".join(e.text for e in reasoning) == "let me think"


async def test_stream_with_schema_parses_structured(mock_sdk: dict[str, Any]) -> None:
    payload = json.dumps({"summary": "ok", "count": 7})
    events = [
        _FakeItemCompleted(_FakeAgentMessageItem(id="m-1", text=payload)),
        _FakeTurnCompleted(_FakeUsage()),
    ]
    mock_sdk["thread"].run_streamed = AsyncMock(return_value=_FakeStreamedTurn(events))

    rt = CodexRuntime()
    sess = rt.session()
    try:
        events_yielded = [e async for e in sess.stream("brief", schema=_Schema)]
    finally:
        await sess.close()

    turn = next(e for e in events_yielded if isinstance(e, TurnComplete))
    assert turn.result.structured == {"summary": "ok", "count": 7}


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


async def test_cancel_aborts_in_flight_execute(mock_sdk: dict[str, Any]) -> None:
    """cancel() during execute() calls controller.abort() and surfaces RuntimeCancelledError."""
    AbortError = mock_sdk["AbortError"]
    hang = asyncio.Event()

    async def hanging_run(prompt: str, options: dict[str, Any]) -> Any:
        # Wait for cancel() to flip the signal then raise AbortError.
        signal = options["signal"]
        while not signal.aborted:
            await asyncio.sleep(0.01)
            if hang.is_set():  # safety unstick
                break
        raise AbortError("cancelled")

    mock_sdk["thread"].run = AsyncMock(side_effect=hanging_run)

    rt = CodexRuntime()
    sess = rt.session()
    try:
        exec_task = asyncio.create_task(sess.execute("hangs"))
        # Wait until the session enters in-flight state.
        for _ in range(100):
            await asyncio.sleep(0)
            if sess._in_flight:  # noqa: SLF001
                break
        assert sess._in_flight  # noqa: SLF001
        await sess.cancel()
        with pytest.raises(RuntimeCancelledError):
            await exec_task
    finally:
        hang.set()
        await sess.close()

    # AbortController was constructed and its abort() flipped the signal.
    assert mock_sdk["aborts"], "no AbortController was instantiated"
    assert mock_sdk["aborts"][-1].signal.aborted is True


async def test_cancel_when_idle_is_noop(mock_sdk: dict[str, Any]) -> None:
    rt = CodexRuntime()
    sess = rt.session()
    try:
        await sess.cancel()
        await sess.cancel()
    finally:
        await sess.close()
    # No AbortController was needed.
    assert mock_sdk["aborts"] == []


async def test_cancel_during_stream(mock_sdk: dict[str, Any]) -> None:
    """Cancelling mid-stream raises AbortError → RuntimeCancelledError."""
    AbortError = mock_sdk["AbortError"]

    async def streamed_events() -> Any:
        yield _FakeItemUpdated(_FakeAgentMessageItem(id="m-1", text="par"))
        # Simulate the next pull raising because abort fired.
        raise AbortError("cancelled mid-stream")

    streamed = MagicMock()
    streamed.events = streamed_events()
    mock_sdk["thread"].run_streamed = AsyncMock(return_value=streamed)

    rt = CodexRuntime()
    sess = rt.session()
    try:
        with pytest.raises(RuntimeCancelledError):
            async for _ in sess.stream("hi"):
                pass
    finally:
        await sess.close()


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


async def test_turn_failed_event_raises(mock_sdk: dict[str, Any]) -> None:
    """TurnFailedEvent during stream raises through the runtime's classifier."""
    events = [
        _FakeItemCompleted(_FakeAgentMessageItem(id="m-1", text="partial")),
        _FakeTurnFailed("rate limit exceeded"),
    ]
    mock_sdk["thread"].run_streamed = AsyncMock(return_value=_FakeStreamedTurn(events))

    rt = CodexRuntime()
    sess = rt.session()
    try:
        with pytest.raises(Exception):  # noqa: B017 — runtime classifies into Runtime*Error subclass
            async for _ in sess.stream("hi"):
                pass
    finally:
        await sess.close()


# ---------------------------------------------------------------------------
# thinking= (Phase 2 Iteration B)
# ---------------------------------------------------------------------------


async def test_thinking_effort_forwarded_to_thread_options(
    mock_sdk: dict[str, Any],
) -> None:
    """thinking=<effort> bakes ``modelReasoningEffort`` into start_thread()."""
    rt = CodexRuntime()
    sess = rt.session()
    try:
        await sess.execute("plan it", thinking="high")
    finally:
        await sess.close()

    mock_sdk["client"].start_thread.assert_called_once_with(
        {
            "model": "gpt-5-codex",
            "sandboxMode": "read-only",
            "skipGitRepoCheck": True,
            "modelReasoningEffort": "high",
        },
    )


async def test_thinking_none_omits_model_reasoning_effort(
    mock_sdk: dict[str, Any],
) -> None:
    """thinking=None leaves ``modelReasoningEffort`` off — Codex picks its default."""
    rt = CodexRuntime()
    sess = rt.session()
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    (call_args,) = mock_sdk["client"].start_thread.call_args_list
    options = call_args.args[0]
    assert "modelReasoningEffort" not in options


async def test_thinking_disabled_omits_model_reasoning_effort(
    mock_sdk: dict[str, Any],
) -> None:
    """thinking='disabled' falls through to None — same wire shape."""
    rt = CodexRuntime()
    sess = rt.session()
    try:
        await sess.execute("hi", thinking="disabled")
    finally:
        await sess.close()

    (call_args,) = mock_sdk["client"].start_thread.call_args_list
    options = call_args.args[0]
    assert "modelReasoningEffort" not in options


async def test_thinking_change_between_turns_rebuilds_thread(
    mock_sdk: dict[str, Any],
) -> None:
    """``modelReasoningEffort`` is per-Thread, so changing thinking rebuilds it."""
    rt = CodexRuntime()
    sess = rt.session()
    try:
        await sess.execute("warm", thinking="low")
        await sess.execute("hot", thinking="high")
    finally:
        await sess.close()

    assert mock_sdk["client"].start_thread.call_count == 2
    first_options = mock_sdk["client"].start_thread.call_args_list[0].args[0]
    second_options = mock_sdk["client"].start_thread.call_args_list[1].args[0]
    assert first_options["modelReasoningEffort"] == "low"
    assert second_options["modelReasoningEffort"] == "high"


async def test_same_thinking_across_turns_reuses_thread(
    mock_sdk: dict[str, Any],
) -> None:
    """Same effort across turns keeps the existing Thread — no rebuild."""
    rt = CodexRuntime()
    sess = rt.session()
    try:
        await sess.execute("a", thinking="medium")
        await sess.execute("b", thinking="medium")
    finally:
        await sess.close()

    assert mock_sdk["client"].start_thread.call_count == 1


async def test_thinking_dict_raises_unsupported_feature_error(
    mock_sdk: dict[str, Any],
) -> None:
    """The ``{"budget_tokens": N}`` shape is Anthropic-only — Codex declines."""
    from airframe.errors import UnsupportedFeatureError
    from airframe.features import Feature

    rt = CodexRuntime()
    sess = rt.session()
    try:
        with pytest.raises(UnsupportedFeatureError) as exc_info:
            await sess.execute("hi", thinking={"budget_tokens": 5000})
    finally:
        await sess.close()

    assert exc_info.value.feature == Feature.REASONING_BUDGET_TOKENS


async def test_stream_forwards_thinking_to_thread_options(
    mock_sdk: dict[str, Any],
) -> None:
    """stream() takes the same path through _ensure_thread as execute()."""
    events = [
        _FakeItemCompleted(_FakeAgentMessageItem(id="m-1", text="done")),
        _FakeTurnCompleted(_FakeUsage()),
    ]
    mock_sdk["thread"].run_streamed = AsyncMock(return_value=_FakeStreamedTurn(events))

    rt = CodexRuntime()
    sess = rt.session()
    try:
        async for _ in sess.stream("hi", thinking="low"):
            pass
    finally:
        await sess.close()

    (call_args,) = mock_sdk["client"].start_thread.call_args_list
    assert call_args.args[0]["modelReasoningEffort"] == "low"


# ---------------------------------------------------------------------------
# Vision / file input (Phase 2 Iteration C)
# ---------------------------------------------------------------------------


async def test_image_input_routes_through_local_image_input(
    mock_sdk: dict[str, Any], tmp_path: Any
) -> None:
    """ImageInput(path=) becomes a LocalImageInput entry in the input list."""
    from openai_codex_sdk.types import LocalImageInput, TextInput

    from airframe.inputs import ImageInput

    captured: dict[str, Any] = {}

    async def fake_run(run_input: Any, options: dict[str, Any]) -> Any:
        captured["input"] = run_input
        return _FakeTurn(final_response="ok")

    mock_sdk["thread"].run = AsyncMock(side_effect=fake_run)

    img_path = tmp_path / "x.png"
    img_path.write_bytes(b"fake")

    rt = CodexRuntime()
    sess = rt.session()
    try:
        await sess.execute(["caption this:", ImageInput(path=str(img_path))])
    finally:
        await sess.close()

    items = captured["input"]
    assert isinstance(items, list)
    assert isinstance(items[0], TextInput) and items[0].text == "caption this:"
    assert isinstance(items[1], LocalImageInput) and items[1].path == str(img_path)


async def test_plain_string_input_unchanged(mock_sdk: dict[str, Any]) -> None:
    """No attachments → input stays a bare string (preserves the v0 wire shape)."""
    captured: dict[str, Any] = {}

    async def fake_run(run_input: Any, options: dict[str, Any]) -> Any:
        captured["input"] = run_input
        return _FakeTurn(final_response="ok")

    mock_sdk["thread"].run = AsyncMock(side_effect=fake_run)

    rt = CodexRuntime()
    sess = rt.session()
    try:
        await sess.execute("just text")
    finally:
        await sess.close()

    assert captured["input"] == "just text"


async def test_file_input_appended_as_text_hint(mock_sdk: dict[str, Any], tmp_path: Any) -> None:
    """FileInput becomes an ``Attached file: <path>`` line in the prompt text."""
    from airframe.inputs import FileInput

    captured: dict[str, Any] = {}

    async def fake_run(run_input: Any, options: dict[str, Any]) -> Any:
        captured["input"] = run_input
        return _FakeTurn(final_response="ok")

    mock_sdk["thread"].run = AsyncMock(side_effect=fake_run)
    file_path = tmp_path / "spec.pdf"
    file_path.write_text("dummy")

    rt = CodexRuntime()
    sess = rt.session()
    try:
        await sess.execute(["summarise:", FileInput(path=str(file_path))])
    finally:
        await sess.close()

    assert isinstance(captured["input"], str)
    assert f"Attached file: {file_path}" in captured["input"]


async def test_image_bytes_raises_with_helpful_message(
    mock_sdk: dict[str, Any],
) -> None:
    """Codex's LocalImageInput is path-only; bytes_= raises with a write-to-disk hint."""
    from airframe.errors import UnsupportedFeatureError
    from airframe.inputs import ImageInput

    rt = CodexRuntime()
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

    rt = CodexRuntime()
    sess = rt.session()
    try:
        with pytest.raises(UnsupportedFeatureError) as exc_info:
            await sess.execute(["x", ImageInput(url="https://example.com/x.png")])
    finally:
        await sess.close()
    assert exc_info.value.feature == Feature.VISION_INPUT
    assert "path=" in str(exc_info.value)


async def test_stream_forwards_image_input(mock_sdk: dict[str, Any], tmp_path: Any) -> None:
    """stream() routes attachments through the same _build_codex_input path."""
    from openai_codex_sdk.types import LocalImageInput

    from airframe.inputs import ImageInput

    captured: dict[str, Any] = {}

    async def fake_streamed(run_input: Any, options: dict[str, Any]) -> Any:
        captured["input"] = run_input
        return _FakeStreamedTurn(
            [
                _FakeItemCompleted(_FakeAgentMessageItem(id="m-1", text="done")),
                _FakeTurnCompleted(_FakeUsage()),
            ]
        )

    mock_sdk["thread"].run_streamed = AsyncMock(side_effect=fake_streamed)

    img_path = tmp_path / "y.png"
    img_path.write_bytes(b"fake")

    rt = CodexRuntime()
    sess = rt.session()
    try:
        async for _ in sess.stream(["look:", ImageInput(path=str(img_path))]):
            pass
    finally:
        await sess.close()

    items = captured["input"]
    assert any(isinstance(p, LocalImageInput) for p in items)


# ---------------------------------------------------------------------------
# close() lifecycle
# ---------------------------------------------------------------------------


async def test_close_blocks_further_execute(mock_sdk: dict[str, Any]) -> None:
    rt = CodexRuntime()
    sess = rt.session()
    await sess.execute("warm")
    await sess.close()
    with pytest.raises(RuntimeError):
        await sess.execute("nope")


async def test_close_is_idempotent(mock_sdk: dict[str, Any]) -> None:
    rt = CodexRuntime()
    sess = rt.session()
    await sess.close()
    await sess.close()
    await sess.close()


# ---------------------------------------------------------------------------
# Function tools (Phase 3 Iteration D — Codex declines)
# ---------------------------------------------------------------------------


async def test_session_tools_kwarg_declines_with_cli_config_pointer(
    mock_sdk: dict[str, Any],
) -> None:
    """tools= raises UnsupportedFeatureError with the CLI-config workaround.

    Iteration D replaces the generic ``_check_tools_supported`` decline
    with a Codex-specific message that points consumers at the
    ``codex`` CLI's config file. The decline is permanent, not the
    "wait for the next iteration" pattern the other three adapters
    used during Iterations B/C.
    """
    from airframe import FunctionTool
    from airframe.errors import UnsupportedFeatureError

    class _NoArgs(BaseModel):
        pass

    async def _noop(_: _NoArgs) -> None:
        return None

    tool = FunctionTool(
        name="noop",
        description="never invoked",
        params=_NoArgs,
        handler=_noop,
    )
    rt = CodexRuntime()
    with pytest.raises(UnsupportedFeatureError) as exc_info:
        rt.session(tools=[tool])
    assert exc_info.value.feature == Feature.TOOLS_FUNCTION
    message = str(exc_info.value)
    # Pin the actionable text — message rot would defeat the whole
    # point of swapping in a Codex-specific message.
    assert "config" in message.lower()
    assert "codex" in message.lower()


async def test_session_tools_none_or_empty_still_opens_cleanly(
    mock_sdk: dict[str, Any],
) -> None:
    """tools=None / tools=[] are both no-ops — neither path triggers the decline."""
    rt = CodexRuntime()
    sess_none = rt.session(tools=None)
    sess_empty = rt.session(tools=[])
    assert sess_none is not None
    assert sess_empty is not None
    await sess_none.close()
    await sess_empty.close()


def test_codex_runtime_does_not_declare_tools_function() -> None:
    """Codex stays on TOOLS_FUNCTION=False — its Python SDK has no
    tool-registration channel."""
    rt = CodexRuntime()
    assert rt.supports(Feature.TOOLS_FUNCTION) is False


# ---------------------------------------------------------------------------
# MCP server refs (Phase 4 Iteration D — Codex declines)
# ---------------------------------------------------------------------------


async def test_session_mcp_servers_declines_with_cli_config_pointer(
    mock_sdk: dict[str, Any],
) -> None:
    """``mcp_servers=`` raises UnsupportedFeatureError with the
    ``~/.codex/config.toml`` workaround.

    Iteration D replaces the generic shared-helper decline with a
    Codex-specific message symmetric with Phase 3 Iteration D's
    ``tools=`` decline. The decline is **permanent** — the Codex
    Python SDK has no programmatic MCP-registration channel.
    """
    from airframe import McpServerRef
    from airframe.errors import UnsupportedFeatureError

    rt = CodexRuntime()
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
    # The first ref's transport surfaces on .feature so consumer code
    # branching on Feature.TOOLS_MCP_STDIO still works.
    assert exc_info.value.feature == Feature.TOOLS_MCP_STDIO
    message = str(exc_info.value).lower()
    # Pin the actionable text — message rot would defeat the whole
    # point of the swap.
    assert "config" in message
    assert "codex" in message
    assert "[[mcp_servers]]" in message


async def test_session_mcp_servers_decline_carries_transport_feature(
    mock_sdk: dict[str, Any],
) -> None:
    """The ``.feature`` attribute matches the *first* ref's transport.

    Plan requirement (Iteration D): "the first ref's transport, since
    the Codex Python SDK declines all transports equally". Verified
    for each transport variant.
    """
    from airframe import McpServerRef
    from airframe.errors import UnsupportedFeatureError

    rt = CodexRuntime()
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


async def test_session_mcp_servers_none_or_empty_still_opens_cleanly(
    mock_sdk: dict[str, Any],
) -> None:
    """``mcp_servers=None`` / ``mcp_servers=[]`` are both no-ops."""
    rt = CodexRuntime()
    sess_none = rt.session(mcp_servers=None)
    sess_empty = rt.session(mcp_servers=[])
    assert sess_none is not None
    assert sess_empty is not None
    await sess_none.close()
    await sess_empty.close()


def test_codex_runtime_declines_every_mcp_transport() -> None:
    """Codex permanently declines every transport flag."""
    rt = CodexRuntime()
    assert rt.supports(Feature.TOOLS_MCP_STDIO) is False
    assert rt.supports(Feature.TOOLS_MCP_HTTP) is False
    assert rt.supports(Feature.TOOLS_MCP_SSE) is False
    assert rt.supports(Feature.TOOLS_MCP_IN_PROCESS) is False


# ---------------------------------------------------------------------------
# Permission callback (Phase 5 Iteration B — session-wide approval_policy)
# ---------------------------------------------------------------------------


async def test_permission_callback_allow_maps_to_never_policy(
    mock_sdk: dict[str, Any],
) -> None:
    """``"allow"`` derives ``approval_policy="never"`` — auto-approve everything.

    Codex's approval channel is session-wide, not per-call. airframe
    calls the user's callback **once** at first execute() with a
    sentinel PermissionRequest and translates the decision into the
    ApprovalMode enum baked into start_thread().
    """
    from airframe import PermissionDecision, PermissionRequest

    captured: list[PermissionRequest] = []

    class _Allow:
        async def handle(self, request: PermissionRequest) -> PermissionDecision:
            captured.append(request)
            return "allow"

    rt = CodexRuntime()
    sess = rt.session(on_permission=_Allow())
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    options = mock_sdk["client"].start_thread.call_args.args[0]
    assert options["approval_policy"] == "never"
    # The callback fired exactly once with the sentinel request.
    assert len(captured) == 1
    from airframe.adapters.codex import CODEX_SESSION_PERMISSION_TOOL

    assert captured[0].tool_name == CODEX_SESSION_PERMISSION_TOOL
    assert captured[0].tool_args == {}
    # The reason explains the session-wide-only limitation.
    assert "session-wide" in (captured[0].reason or "")


async def test_permission_callback_deny_maps_to_untrusted_policy(
    mock_sdk: dict[str, Any],
) -> None:
    """``"deny"`` derives ``approval_policy="untrusted"`` — strictest mode."""
    from airframe import PermissionDecision, PermissionRequest

    class _Deny:
        async def handle(self, request: PermissionRequest) -> PermissionDecision:
            return "deny"

    rt = CodexRuntime()
    sess = rt.session(on_permission=_Deny())
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    options = mock_sdk["client"].start_thread.call_args.args[0]
    assert options["approval_policy"] == "untrusted"


async def test_permission_callback_defer_maps_to_on_request_policy(
    mock_sdk: dict[str, Any],
) -> None:
    """``"defer"`` derives ``approval_policy="on-request"`` — Codex's default
    per-call prompting."""
    from airframe import PermissionDecision, PermissionRequest

    class _Defer:
        async def handle(self, request: PermissionRequest) -> PermissionDecision:
            return "defer"

    rt = CodexRuntime()
    sess = rt.session(on_permission=_Defer())
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    options = mock_sdk["client"].start_thread.call_args.args[0]
    assert options["approval_policy"] == "on-request"


async def test_permission_callback_fires_only_once_per_session(
    mock_sdk: dict[str, Any],
) -> None:
    """The callback is invoked at most once per session — Codex's
    approval_policy is baked at Thread creation, not re-evaluated
    per turn. This is the loud limitation the docstring calls out.
    """
    from airframe import PermissionDecision, PermissionRequest

    call_count = 0

    class _CountingCallback:
        async def handle(self, request: PermissionRequest) -> PermissionDecision:
            nonlocal call_count
            call_count += 1
            return "allow"

    rt = CodexRuntime()
    sess = rt.session(on_permission=_CountingCallback())
    try:
        await sess.execute("turn 1")
        await sess.execute("turn 2")
        await sess.execute("turn 3")
    finally:
        await sess.close()

    assert call_count == 1, (
        "Codex's approval_policy is session-wide; the callback must "
        f"fire exactly once at first execute(), not per turn. Got {call_count}."
    )


async def test_no_permission_callback_omits_approval_policy(
    mock_sdk: dict[str, Any],
) -> None:
    """Sessions opened without on_permission= leave approval_policy off
    — Codex uses its own default."""
    rt = CodexRuntime()
    sess = rt.session()
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    options = mock_sdk["client"].start_thread.call_args.args[0]
    assert "approval_policy" not in options


def test_codex_runtime_declares_permission_callback() -> None:
    """Codex flips PERMISSION_CALLBACK True after Phase 5 Iteration B."""
    rt = CodexRuntime()
    assert rt.supports(Feature.PERMISSION_CALLBACK) is True


# ---------------------------------------------------------------------------
# Lifecycle hooks (Phase 5 Iteration C)
# ---------------------------------------------------------------------------


class _FakeCommandExecutionItem:
    def __init__(
        self,
        *,
        id: str,
        command: str = "echo hi",
        status: str = "completed",
        exit_code: int | None = 0,
        aggregated_output: str = "hi\n",
    ) -> None:
        self.id = id
        self.type = "command_execution"
        self.command = command
        self.status = status
        self.exit_code = exit_code
        self.aggregated_output = aggregated_output


class _FakeMcpToolCallItem:
    def __init__(
        self,
        *,
        id: str,
        server: str = "everything",
        tool: str = "ping",
        status: str = "completed",
        arguments: Any = None,
        result: Any = None,
        error: Any = None,
    ) -> None:
        self.id = id
        self.type = "mcp_tool_call"
        self.server = server
        self.tool = tool
        self.status = status
        self.arguments = arguments
        self.result = result
        self.error = error


class _FakeItemStarted:
    def __init__(self, item: Any) -> None:
        self.type = "item.started"
        self.item = item


def _patch_codex_item_types(monkeypatch: pytest.MonkeyPatch) -> None:
    """Substitute fake CommandExecution/McpToolCall types + ItemStartedEvent
    so isinstance() dispatch in stream() picks them up."""
    from openai_codex_sdk import types as types_mod

    monkeypatch.setattr(types_mod, "CommandExecutionItem", _FakeCommandExecutionItem)
    monkeypatch.setattr(types_mod, "McpToolCallItem", _FakeMcpToolCallItem)
    monkeypatch.setattr(types_mod, "ItemStartedEvent", _FakeItemStarted)


def test_codex_runtime_declares_lifecycle_hooks() -> None:
    """LIFECYCLE_HOOKS is True at the runtime level."""
    rt = CodexRuntime()
    assert rt.supports(Feature.LIFECYCLE_HOOKS)


def test_codex_emittable_hook_kinds_matches_plan() -> None:
    """Codex emits six kinds — no pre_compact (no native compaction
    event) and no rate_limit (SDK has no explicit rate-limit signal)."""
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
        == CodexRuntime.EMITTABLE_HOOK_KINDS
    )


async def test_on_event_execute_emits_session_start_and_user_prompt_submit(
    mock_sdk: dict[str, Any],
) -> None:
    """First execute() fires session_start exactly once, plus
    user_prompt_submit per turn."""
    from airframe.hooks import HookEvent

    events: list[HookEvent] = []

    rt = CodexRuntime()
    sess = rt.session(on_event=events.append)
    try:
        await sess.execute("first")
        await sess.execute("second")
    finally:
        await sess.close()

    starts = [e for e in events if e.kind == "session_start"]
    prompts = [e for e in events if e.kind == "user_prompt_submit"]
    assert len(starts) == 1
    assert starts[0].payload["model"] == "gpt-5-codex"
    assert starts[0].payload["resumed"] is False
    assert len(prompts) == 2
    assert prompts[0].payload["prompt"] == "first"
    assert prompts[0].payload["length"] == len("first")


async def test_on_event_resume_flag_propagates_to_session_start(
    mock_sdk: dict[str, Any],
) -> None:
    """resume=<id> → session_start payload carries resumed=True."""
    from airframe.hooks import HookEvent

    events: list[HookEvent] = []
    mock_sdk["thread"].id = "thread-resumed"

    rt = CodexRuntime()
    sess = rt.session(resume="thread-resumed", on_event=events.append)
    try:
        await sess.execute("hi")
    finally:
        await sess.close()

    starts = [e for e in events if e.kind == "session_start"]
    assert len(starts) == 1
    assert starts[0].payload["resumed"] is True


async def test_on_event_execute_replays_tool_items_post_turn(
    mock_sdk: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """execute() returns a Turn after completion — the adapter replays
    its items at end-of-turn to emit pre/post_tool_use hooks."""
    from airframe.hooks import HookEvent

    _patch_codex_item_types(monkeypatch)

    cmd_item = _FakeCommandExecutionItem(
        id="cmd-1",
        command="ls -la",
        status="completed",
        exit_code=0,
        aggregated_output="total 0\n",
    )
    mcp_item = _FakeMcpToolCallItem(
        id="mcp-1",
        server="everything",
        tool="ping",
        status="completed",
        result={"ok": True},
    )

    async def fake_run(prompt: str, options: dict[str, Any]) -> Any:
        return _FakeTurn(final_response="done", items=[cmd_item, mcp_item])

    mock_sdk["thread"].run = AsyncMock(side_effect=fake_run)

    events: list[HookEvent] = []
    rt = CodexRuntime()
    sess = rt.session(on_event=events.append)
    try:
        await sess.execute("run stuff")
    finally:
        await sess.close()

    # Each item produced exactly one pre + one post hook, in per-item order.
    kinds = [e.kind for e in events if e.kind in {"pre_tool_use", "post_tool_use"}]
    assert kinds == ["pre_tool_use", "post_tool_use", "pre_tool_use", "post_tool_use"]
    pres = [e for e in events if e.kind == "pre_tool_use"]
    posts = [e for e in events if e.kind == "post_tool_use"]
    assert pres[0].payload["tool_name"] == "ls -la"
    assert pres[0].payload["tool_call_id"] == "cmd-1"
    assert pres[1].payload["tool_name"] == "everything/ping"
    assert posts[0].payload["exit_code"] == 0
    assert posts[1].payload["output"] == {"ok": True}


async def test_on_event_execute_replays_failed_command_as_tool_failure(
    mock_sdk: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """status='failed' → tool_failure, not post_tool_use."""
    from airframe.hooks import HookEvent

    _patch_codex_item_types(monkeypatch)

    cmd_item = _FakeCommandExecutionItem(
        id="cmd-1",
        command="false",
        status="failed",
        exit_code=1,
        aggregated_output="",
    )

    async def fake_run(prompt: str, options: dict[str, Any]) -> Any:
        return _FakeTurn(final_response="apologies", items=[cmd_item])

    mock_sdk["thread"].run = AsyncMock(side_effect=fake_run)

    events: list[HookEvent] = []
    rt = CodexRuntime()
    sess = rt.session(on_event=events.append)
    try:
        await sess.execute("run failing")
    finally:
        await sess.close()

    kinds = [e.kind for e in events if e.kind in {"post_tool_use", "tool_failure"}]
    assert kinds == ["tool_failure"]


async def test_on_event_stream_emits_pre_then_post_tool_use_per_item(
    mock_sdk: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Streaming path: pre_tool_use on first ItemStarted/Updated for a
    command item; post_tool_use on ItemCompleted with status=completed."""
    from airframe.hooks import HookEvent

    _patch_codex_item_types(monkeypatch)

    cmd_running = _FakeCommandExecutionItem(
        id="cmd-1",
        command="echo hi",
        status="in_progress",
        exit_code=None,
        aggregated_output="",
    )
    cmd_done = _FakeCommandExecutionItem(
        id="cmd-1",
        command="echo hi",
        status="completed",
        exit_code=0,
        aggregated_output="hi\n",
    )
    msg_done = _FakeAgentMessageItem(id="m-1", text="done")
    events_stream = [
        _FakeItemStarted(cmd_running),
        _FakeItemCompleted(cmd_done),
        _FakeItemCompleted(msg_done),
        _FakeTurnCompleted(_FakeUsage()),
    ]
    mock_sdk["thread"].run_streamed = AsyncMock(return_value=_FakeStreamedTurn(events_stream))

    captured: list[HookEvent] = []
    rt = CodexRuntime()
    sess = rt.session(on_event=captured.append)
    try:
        async for _ in sess.stream("run echo"):
            pass
    finally:
        await sess.close()

    kinds = [e.kind for e in captured if e.kind in {"pre_tool_use", "post_tool_use"}]
    # pre fires on the ItemStartedEvent (running), post fires on the
    # ItemCompletedEvent (completed) — one of each per item id.
    assert kinds == ["pre_tool_use", "post_tool_use"]


async def test_on_event_stream_pre_tool_use_fires_once_per_item_id(
    mock_sdk: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multiple ItemUpdatedEvents for the same item id → still only one
    pre_tool_use HookEvent."""
    from airframe.hooks import HookEvent

    _patch_codex_item_types(monkeypatch)

    cmd_a = _FakeCommandExecutionItem(id="cmd-1", status="in_progress", exit_code=None)
    cmd_b = _FakeCommandExecutionItem(id="cmd-1", status="in_progress", exit_code=None)
    cmd_done = _FakeCommandExecutionItem(id="cmd-1", status="completed", exit_code=0)
    events_stream = [
        _FakeItemUpdated(cmd_a),
        _FakeItemUpdated(cmd_b),
        _FakeItemCompleted(cmd_done),
        _FakeItemCompleted(_FakeAgentMessageItem(id="m-1", text="ok")),
        _FakeTurnCompleted(_FakeUsage()),
    ]
    mock_sdk["thread"].run_streamed = AsyncMock(return_value=_FakeStreamedTurn(events_stream))

    captured: list[HookEvent] = []
    rt = CodexRuntime()
    sess = rt.session(on_event=captured.append)
    try:
        async for _ in sess.stream("hi"):
            pass
    finally:
        await sess.close()

    pres = [e for e in captured if e.kind == "pre_tool_use"]
    assert len(pres) == 1


async def test_close_synthesises_session_end_when_session_start_fired(
    mock_sdk: dict[str, Any],
) -> None:
    """close() after at least one turn emits session_end exactly once."""
    from airframe.hooks import HookEvent

    events: list[HookEvent] = []
    rt = CodexRuntime()
    sess = rt.session(on_event=events.append)
    await sess.execute("hi")
    # No session_end yet before close.
    assert [e for e in events if e.kind == "session_end"] == []
    await sess.close()
    ends = [e for e in events if e.kind == "session_end"]
    assert len(ends) == 1
    assert ends[0].payload["model"] == "gpt-5-codex"


async def test_close_session_end_is_idempotent(mock_sdk: dict[str, Any]) -> None:
    """Multiple close() calls emit session_end at most once."""
    from airframe.hooks import HookEvent

    events: list[HookEvent] = []
    rt = CodexRuntime()
    sess = rt.session(on_event=events.append)
    await sess.execute("hi")
    await sess.close()
    await sess.close()
    await sess.close()
    ends = [e for e in events if e.kind == "session_end"]
    assert len(ends) == 1


async def test_close_without_execute_omits_session_end(
    mock_sdk: dict[str, Any],
) -> None:
    """If session_start never fired (no execute() ever ran), close()
    must NOT fire a phantom session_end."""
    from airframe.hooks import HookEvent

    events: list[HookEvent] = []
    rt = CodexRuntime()
    sess = rt.session(on_event=events.append)
    await sess.close()
    assert [e for e in events if e.kind == "session_end"] == []


async def test_no_on_event_skips_all_hook_emission(
    mock_sdk: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without on_event= the adapter doesn't construct HookEvents (the
    fast path stays fast). Verified indirectly: the session completes
    normally even when no observer is wired."""
    _patch_codex_item_types(monkeypatch)

    cmd_item = _FakeCommandExecutionItem(id="c", status="completed", exit_code=0)

    async def fake_run(prompt: str, options: dict[str, Any]) -> Any:
        return _FakeTurn(final_response="ok", items=[cmd_item])

    mock_sdk["thread"].run = AsyncMock(side_effect=fake_run)

    rt = CodexRuntime()
    sess = rt.session()
    try:
        result = await sess.execute("hi")
    finally:
        await sess.close()
    assert result.text == "ok"


async def test_on_event_observer_that_raises_does_not_break_session(
    mock_sdk: dict[str, Any],
) -> None:
    """A raising observer is caught by _fire_hook_event; the session
    continues to completion."""
    from airframe.hooks import HookEvent

    calls = {"n": 0}

    def boom(event: HookEvent) -> None:
        calls["n"] += 1
        raise RuntimeError("observer broke")

    rt = CodexRuntime()
    sess = rt.session(on_event=boom)
    try:
        result = await sess.execute("hi")
    finally:
        await sess.close()

    # Session ran to completion despite the raising observer.
    assert calls["n"] >= 1
    # _FakeTurn().final_response is a JSON string; without schema= the
    # adapter surfaces it as-is in result.text.
    assert result.text == '{"summary": "ok", "count": 42}'


# ---------------------------------------------------------------------------
# Budget caps (Phase 5 Iteration D)
# ---------------------------------------------------------------------------


def test_codex_runtime_declares_budget_caps() -> None:
    """Codex declares both BUDGET_USD_CAP and BUDGET_TURN_CAP."""
    rt = CodexRuntime()
    assert rt.supports(Feature.BUDGET_USD_CAP)
    assert rt.supports(Feature.BUDGET_TURN_CAP)


async def test_max_turns_cap_raises_when_count_reached(
    mock_sdk: dict[str, Any],
) -> None:
    """After running max_turns turns, the next execute() raises
    RuntimeBudgetExceededError(kind='turns')."""
    from airframe.errors import RuntimeBudgetExceededError

    rt = CodexRuntime()
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
    mock_sdk: dict[str, Any],
) -> None:
    """Cumulative cost from `_FakeUsage` defaults adds up across
    turns; once it crosses the cap the next execute() raises."""
    from airframe.errors import RuntimeBudgetExceededError

    # Default _FakeUsage gives ~$0.000225/turn on gpt-5-codex.
    # Cap at $0.0003 → first turn succeeds ($0.000225),
    # second turn raises (cumulative $0.000225 < $0.0003 still
    # OK, but after second succeeds $0.00045 > $0.0003 → third raises).
    # Use a tighter cap so the trip happens immediately.
    rt = CodexRuntime()
    sess = rt.session()
    try:
        # First turn succeeds → cumulative ~$0.000225.
        await sess.execute("turn 1", max_budget_usd=0.0002)
        # Next turn's pre-enforce trips: $0.000225 >= $0.0002.
        with pytest.raises(RuntimeBudgetExceededError) as exc_info:
            await sess.execute("turn 2", max_budget_usd=0.0002)
    finally:
        await sess.close()
    err = exc_info.value
    assert err.kind == "usd"
    assert err.cap == 0.0002


async def test_stream_honours_budget_caps(mock_sdk: dict[str, Any]) -> None:
    """Stream path uses the same enforce — exhausted cap raises."""
    from airframe.errors import RuntimeBudgetExceededError

    events_stream = [
        _FakeItemCompleted(_FakeAgentMessageItem(id="m-1", text="hi")),
        _FakeTurnCompleted(_FakeUsage()),
    ]
    mock_sdk["thread"].run_streamed = AsyncMock(return_value=_FakeStreamedTurn(events_stream))
    rt = CodexRuntime()
    sess = rt.session()
    try:
        async for _ in sess.stream("a", max_budget_usd=0.0002):
            pass
        with pytest.raises(RuntimeBudgetExceededError):
            async for _ in sess.stream("b", max_budget_usd=0.0002):
                pass
    finally:
        await sess.close()


async def test_budget_caps_none_open_cleanly(mock_sdk: dict[str, Any]) -> None:
    """Without caps, turns run indefinitely."""
    rt = CodexRuntime()
    sess = rt.session()
    try:
        for i in range(5):
            await sess.execute(f"turn {i}")
    finally:
        await sess.close()


# ---------------------------------------------------------------------------
# Provider options (v0.5.0-readiness — CodexOptions wired)
# ---------------------------------------------------------------------------


async def test_codex_options_working_directory_lands_on_thread_options(
    mock_sdk: dict[str, Any],
) -> None:
    """``CodexOptions.working_directory`` rides into ``ThreadOptions.workingDirectory``."""
    from airframe import CodexOptions

    rt = CodexRuntime()
    sess = rt.session(provider_options=CodexOptions(working_directory="/tmp/wd"))
    try:
        await sess.execute("hi")
    finally:
        await sess.close()
    opts = mock_sdk["client"].start_thread.call_args.args[0]
    assert opts["workingDirectory"] == "/tmp/wd"


async def test_codex_options_additional_directories_lands_on_thread_options(
    mock_sdk: dict[str, Any],
) -> None:
    from airframe import CodexOptions

    rt = CodexRuntime()
    sess = rt.session(provider_options=CodexOptions(additional_directories=("/a", "/b")))
    try:
        await sess.execute("hi")
    finally:
        await sess.close()
    opts = mock_sdk["client"].start_thread.call_args.args[0]
    assert opts["additionalDirectories"] == ["/a", "/b"]


async def test_codex_options_network_and_web_search_flags(mock_sdk: dict[str, Any]) -> None:
    from airframe import CodexOptions

    rt = CodexRuntime()
    sess = rt.session(
        provider_options=CodexOptions(network_access_enabled=True, web_search_enabled=True)
    )
    try:
        await sess.execute("hi")
    finally:
        await sess.close()
    opts = mock_sdk["client"].start_thread.call_args.args[0]
    assert opts["networkAccessEnabled"] is True
    assert opts["webSearchEnabled"] is True


async def test_codex_options_default_omits_kwargs(mock_sdk: dict[str, Any]) -> None:
    rt = CodexRuntime()
    sess = rt.session()
    try:
        await sess.execute("hi")
    finally:
        await sess.close()
    opts = mock_sdk["client"].start_thread.call_args.args[0]
    assert "workingDirectory" not in opts
    assert "additionalDirectories" not in opts
    assert "networkAccessEnabled" not in opts
    assert "webSearchEnabled" not in opts


async def test_codex_options_change_between_turns_forces_thread_rebuild(
    mock_sdk: dict[str, Any],
) -> None:
    """A ``CodexOptions`` change across same-session turns forces a rebuild —
    fields bake at start_thread() time and the cache key carries the fingerprint."""
    from airframe import CodexOptions

    rt = CodexRuntime()
    # Same session, two turns with different options — the second turn
    # rebuilds the Thread because the fingerprint changed.
    sess = rt.session(provider_options=CodexOptions(network_access_enabled=False))
    try:
        await sess.execute("a")
        # Manually swap on the live session (private-attr touch is the only
        # path to test cache invalidation without opening a fresh session).
        sess._provider_options = CodexOptions(network_access_enabled=True)  # type: ignore[attr-defined]
        await sess.execute("b")
    finally:
        await sess.close()
    assert mock_sdk["client"].start_thread.call_count == 2


async def test_codex_options_wrong_namespace_raises_unsupported_feature(
    mock_sdk: dict[str, Any],
) -> None:
    from airframe import OpenAICompatOptions
    from airframe.errors import UnsupportedFeatureError

    rt = CodexRuntime()
    with pytest.raises(UnsupportedFeatureError) as exc_info:
        rt.session(provider_options=OpenAICompatOptions())
    assert "OpenAICompatOptions" in str(exc_info.value)
    assert "CodexOptions" in str(exc_info.value)
