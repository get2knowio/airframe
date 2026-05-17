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
