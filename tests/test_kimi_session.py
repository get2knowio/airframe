"""Behavioural unit tests for :class:`KimiSession` — Iterations B + C.

The kimi-agent-sdk package can't be installed in the airframe dev
environment (its ``kimi-cli`` transitive pulls ``fastmcp 2.12.5``
which pins ``mcp<1.17``; ``claude-agent-sdk`` requires ``mcp>=1.23``).
These tests inject lightweight stand-in modules into ``sys.modules``
so airframe's late-imports inside :class:`KimiSession` resolve to the
mocks. ``type(obj).__name__`` matching in
:meth:`KimiSession._classify_wire_message` and
:meth:`KimiSession._classify_sdk_exception` is the deliberate hook
that makes this approach work without depending on the real type
identities.

Coverage:

* ``execute()`` happy path — text aggregation, cost record build,
  RuntimeResult shape.
* ``stream()`` happy path — TextDelta / ReasoningDelta / TurnComplete
  event ordering.
* ``session(resume=…)`` — adapter calls ``Session.resume`` (not
  ``Session.create``) and surfaces the resumed ID.
* ``Session.resume`` returning ``None`` (session not found) →
  :class:`RuntimeProtocolError`.
* ``cancel()`` — calls SDK's ``Session.cancel()`` when in-flight;
  no-op when idle.
* ``close()`` — closes the SDK session if open; idempotent.
* Exception classification — every documented kimi-agent-sdk error
  maps to the airframe ``Runtime*Error`` hierarchy.
* ``KimiOptions.working_directory`` threads into ``Session.create``.
* Iteration C: ``thinking=`` → ``Session.create(thinking: bool)``
  mapping; session rebuild on toggle between turns; reuse when
  unchanged; ``thinking={"budget_tokens": …}`` raises
  :class:`UnsupportedFeatureError`.
* Iteration C: polymorphic prompt — ``ImageInput(url=…)`` /
  ``(bytes_=…)`` / ``(path=…)`` translate to
  :class:`ImageURLPart`; ``FileInput`` declined; system prompt
  prepends to the ``TextPart``.
* Live integration coverage will land alongside Iteration F's
  ``tests/test_kimi_integration.py`` once we have a separate-venv
  story for the kimi-agent-sdk's mcp-version conflict.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

from airframe.errors import (
    RuntimeAuthError,
    RuntimeCancelledError,
    RuntimeProtocolError,
    RuntimeTransientError,
    UnsupportedFeatureError,
)
from airframe.features import Feature
from airframe.inputs import ImageInput

# ---------------------------------------------------------------------------
# Stand-in SDK + KaosPath modules
# ---------------------------------------------------------------------------


class _FakeWire:
    """Minimal stand-in for kimi-agent-sdk wire types.

    ``type(obj).__name__`` is the only thing the adapter looks at, so
    we don't need the real ``ContentPart`` / ``ToolCall`` hierarchy.
    Each instance carries whatever fields its class expects in the
    real SDK.
    """

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


def _wire(class_name: str, **kwargs: Any) -> _FakeWire:
    """Build a wire instance whose ``__name__`` matches ``class_name``."""
    cls = type(class_name, (_FakeWire,), {})
    return cls(**kwargs)


class ApprovalRequest:  # noqa: N801 — must match SDK type name for adapter dispatch
    """Records ``resolve()`` calls so tests can assert on them.

    Class name matches what :meth:`KimiSession._classify_wire_message`
    expects (``type(wire).__name__ == "ApprovalRequest"``).
    """

    def __init__(
        self,
        request_id: str = "req-1",
        *,
        tool_call_id: str = "tc-1",
        sender: str = "agent",
        action: str = "shell",
        description: str = "Run `ls`",
    ) -> None:
        self.id = request_id
        self.tool_call_id = tool_call_id
        self.sender = sender
        self.action = action
        self.description = description
        self.resolved_with: str | None = None
        self.resolved_feedback: str = ""
        self.resolved = False

    def resolve(self, decision: str, feedback: str = "") -> None:
        self.resolved_with = decision
        self.resolved_feedback = feedback
        self.resolved = True


class _FakeSdkSession:
    """Stand-in for ``kimi_agent_sdk.Session``.

    Holds a static list of wire messages to yield, plus simple
    cancel / close tracking. Tests parameterise via the
    ``wire_messages`` / ``raise_on_prompt`` / etc. attributes.
    """

    def __init__(
        self,
        *,
        session_id: str = "sess-fresh",
        wire_messages: list[Any] | None = None,
        raise_on_prompt: BaseException | None = None,
    ) -> None:
        self.id = session_id
        self._wire_messages = wire_messages or []
        self._raise_on_prompt = raise_on_prompt
        self.closed = False
        self.cancel_calls = 0
        # Iteration C: tests assert on the polymorphic user_input the
        # adapter passes through (TextPart + ImageURLPart shape vs
        # plain str). Capture each prompt() call here.
        self.prompt_calls: list[Any] = []

    async def prompt(
        self, user_input: Any, *, merge_wire_messages: bool = False
    ) -> AsyncIterator[Any]:
        del merge_wire_messages
        self.prompt_calls.append(user_input)
        if self._raise_on_prompt is not None:
            raise self._raise_on_prompt
        for wire in self._wire_messages:
            yield wire

    def cancel(self) -> None:
        self.cancel_calls += 1

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def patch_sdk(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Inject fake ``kimi_agent_sdk`` + ``kaos.path`` modules.

    Returns a dict the test can populate before triggering the SDK
    call:

    * ``"sdk_session"``: the :class:`_FakeSdkSession` ``Session.create``
      / ``Session.resume`` should return. Default: empty success.
    * ``"resume_returns_none"``: when ``True``, ``Session.resume``
      returns ``None`` to simulate a missing session.
    * ``"create_raises"`` / ``"resume_raises"``: BaseException to
      raise on the corresponding call site.
    * ``"create_calls"``: list the adapter's ``Session.create`` kwargs
      land in for assertion.
    * ``"resume_calls"``: same for ``Session.resume``.
    """
    state: dict[str, Any] = {
        "sdk_session": None,
        "resume_returns_none": False,
        "create_raises": None,
        "resume_raises": None,
        "create_calls": [],
        "resume_calls": [],
    }

    async def fake_create(**kwargs: Any) -> Any:
        state["create_calls"].append(kwargs)
        if state["create_raises"] is not None:
            raise state["create_raises"]
        return state["sdk_session"] or _FakeSdkSession()

    async def fake_resume(**kwargs: Any) -> Any:
        state["resume_calls"].append(kwargs)
        if state["resume_raises"] is not None:
            raise state["resume_raises"]
        if state["resume_returns_none"]:
            return None
        return state["sdk_session"] or _FakeSdkSession(session_id="sess-resumed")

    fake_session_cls = MagicMock()
    fake_session_cls.create = fake_create
    fake_session_cls.resume = fake_resume

    sdk_module = ModuleType("kimi_agent_sdk")
    sdk_module.Session = fake_session_cls  # type: ignore[attr-defined]
    # Iteration C: content-part shapes the adapter reaches into for
    # polymorphic prompts. Mirror kosong.message's structure closely
    # enough that round-tripping a value through the adapter's
    # ``_build_kimi_user_input`` helper produces the expected types.

    class _FakeTextPart:
        def __init__(self, *, text: str) -> None:
            self.text = text

    class _FakeImageURL:
        def __init__(self, *, url: str, id: str | None = None) -> None:
            self.url = url
            self.id = id

    class _FakeImageURLPart:
        ImageURL = _FakeImageURL

        def __init__(self, *, image_url: _FakeImageURL) -> None:
            self.image_url = image_url

    sdk_module.TextPart = _FakeTextPart  # type: ignore[attr-defined]
    sdk_module.ImageURLPart = _FakeImageURLPart  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kimi_agent_sdk", sdk_module)

    class _FakeKaosPath:
        @staticmethod
        def cwd() -> _FakeKaosPath:
            return _FakeKaosPath()

        def __init__(self, path: str | None = None) -> None:
            self._path = path or "<cwd>"

        def __repr__(self) -> str:
            return f"KaosPath({self._path!r})"

    kaos_path_module = ModuleType("kaos.path")
    kaos_path_module.KaosPath = _FakeKaosPath  # type: ignore[attr-defined]
    kaos_root = ModuleType("kaos")
    monkeypatch.setitem(sys.modules, "kaos", kaos_root)
    monkeypatch.setitem(sys.modules, "kaos.path", kaos_path_module)

    return state


# ---------------------------------------------------------------------------
# execute() — happy path
# ---------------------------------------------------------------------------


def test_execute_aggregates_text_parts_and_returns_runtime_result(
    patch_sdk: dict[str, Any],
) -> None:
    """``execute()`` joins consecutive ``TextPart`` events into the final text."""
    from airframe.adapters.kimi import KimiRuntime

    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[
            _wire("TextPart", text="The capital "),
            _wire("TextPart", text="of France "),
            _wire("TextPart", text="is Paris."),
            _wire("TokenUsage", input_tokens=12, output_tokens=8),
        ]
    )

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session()
    result = asyncio.run(sess.execute("What's the capital of France?"))

    assert result.text == "The capital of France is Paris."
    assert result.structured is None
    assert result.cost.provider_id == "kimi"
    assert result.cost.input_tokens == 12
    assert result.cost.output_tokens == 8
    # Iteration E: cost_usd populates from the in-tree _KIMI_PRICING
    # table for kimi-k2-thinking-turbo (12*1.5 + 8*5.0 per 1M = $0.000058).
    assert result.cost.cost_usd is not None
    assert result.cost.cost_usd > 0
    asyncio.run(sess.close())


def test_execute_translates_thinkpart_to_reasoning_state(
    patch_sdk: dict[str, Any],
) -> None:
    """``ThinkPart`` events surface as reasoning — not folded into ``text``."""
    from airframe.adapters.kimi import KimiRuntime

    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[
            _wire("ThinkPart", text="Reasoning step 1..."),
            _wire("TextPart", text="Final answer."),
        ]
    )

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session()
    result = asyncio.run(sess.execute("Think then answer."))

    # Reasoning is NOT included in the final ``text`` — it lands via
    # ReasoningDelta on stream(), and is captured separately in the
    # cost / event surface. ``text`` is the assistant's visible answer.
    assert result.text == "Final answer."
    asyncio.run(sess.close())


def test_execute_auto_resolves_approval_requests_with_yolo_semantics(
    patch_sdk: dict[str, Any],
) -> None:
    """Iteration B: any ApprovalRequest that slips through gets auto-approved.

    With ``yolo=True`` on the SDK call the SDK shouldn't surface
    these to us, but the defensive resolve makes the adapter robust
    against SDK behaviour changes / missed yolo wiring.
    """
    from airframe.adapters.kimi import KimiRuntime

    approval = ApprovalRequest(request_id="rq-1")
    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[
            _wire("TextPart", text="Hello."),
            approval,
            _wire("TextPart", text=" Done."),
        ]
    )

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session()
    result = asyncio.run(sess.execute("hi"))

    assert approval.resolved is True
    assert approval.resolved_with == "approve"
    assert result.text == "Hello. Done."
    asyncio.run(sess.close())


# ---------------------------------------------------------------------------
# stream() — event ordering
# ---------------------------------------------------------------------------


def test_stream_yields_text_then_reasoning_then_turn_complete(
    patch_sdk: dict[str, Any],
) -> None:
    """``stream()`` emits per-part RuntimeEvents in causal order."""
    from airframe.adapters.kimi import KimiRuntime
    from airframe.events import ReasoningDelta, TextDelta, TurnComplete

    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[
            _wire("TextPart", text="Hello"),
            _wire("ThinkPart", text="(thinking)"),
            _wire("TextPart", text=" world."),
            _wire("TokenUsage", input_tokens=3, output_tokens=2),
        ]
    )

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session()

    async def collect() -> list[Any]:
        events = []
        async for ev in sess.stream("greet"):
            events.append(ev)
        return events

    events = asyncio.run(collect())
    # Three deltas (text + reasoning + text) plus a TurnComplete.
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["TextDelta", "ReasoningDelta", "TextDelta", "TurnComplete"]
    assert isinstance(events[0], TextDelta)
    assert events[0].text == "Hello"
    assert isinstance(events[1], ReasoningDelta)
    assert events[1].text == "(thinking)"
    assert isinstance(events[3], TurnComplete)
    assert events[3].result.text == "Hello world."
    assert events[3].result.cost.input_tokens == 3
    asyncio.run(sess.close())


# ---------------------------------------------------------------------------
# session(resume=…) — Session.resume call shape
# ---------------------------------------------------------------------------


def test_session_resume_calls_sdk_resume_not_create(patch_sdk: dict[str, Any]) -> None:
    from airframe.adapters.kimi import KimiRuntime

    patch_sdk["sdk_session"] = _FakeSdkSession(
        session_id="sess-resumed",
        wire_messages=[_wire("TextPart", text="continuing")],
    )

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session(resume="prior-sess-id")

    # Eagerly visible before any execute call.
    assert sess.id == "prior-sess-id"

    asyncio.run(sess.execute("hi"))

    # After execute, the SDK session is materialised and id matches what
    # Session.resume returned.
    assert sess.id == "sess-resumed"
    # Session.resume called, Session.create not called.
    assert len(patch_sdk["resume_calls"]) == 1
    assert len(patch_sdk["create_calls"]) == 0
    # The resume call carried the prior session_id.
    assert patch_sdk["resume_calls"][0]["session_id"] == "prior-sess-id"
    asyncio.run(sess.close())


def test_session_resume_with_missing_id_raises_protocol_error(
    patch_sdk: dict[str, Any],
) -> None:
    """``Session.resume`` returning ``None`` surfaces as RuntimeProtocolError."""
    from airframe.adapters.kimi import KimiRuntime

    patch_sdk["resume_returns_none"] = True

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session(resume="does-not-exist")
    with pytest.raises(RuntimeProtocolError) as excinfo:
        asyncio.run(sess.execute("hi"))
    assert "does-not-exist" in str(excinfo.value)
    asyncio.run(sess.close())


def test_session_create_called_for_fresh_session(patch_sdk: dict[str, Any]) -> None:
    """Without ``resume=``, the adapter calls ``Session.create``."""
    from airframe.adapters.kimi import KimiRuntime

    patch_sdk["sdk_session"] = _FakeSdkSession(
        session_id="sess-fresh",
        wire_messages=[_wire("TextPart", text="hi")],
    )

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session()
    asyncio.run(sess.execute("hello"))

    assert len(patch_sdk["create_calls"]) == 1
    assert len(patch_sdk["resume_calls"]) == 0
    # Default kwargs: yolo=True, model from runtime default.
    call = patch_sdk["create_calls"][0]
    assert call["yolo"] is True
    assert call["model"] == "kimi-k2-thinking-turbo"
    asyncio.run(sess.close())


# ---------------------------------------------------------------------------
# cancel() / close()
# ---------------------------------------------------------------------------


def test_cancel_when_idle_is_noop(patch_sdk: dict[str, Any]) -> None:
    from airframe.adapters.kimi import KimiRuntime

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session()
    asyncio.run(sess.cancel())  # no SDK session yet — must not raise
    asyncio.run(sess.close())


def test_cancel_during_in_flight_calls_sdk_cancel(patch_sdk: dict[str, Any]) -> None:
    """The adapter forwards cancel() to the SDK's Session.cancel()."""
    from airframe.adapters.kimi import KimiRuntime

    fake = _FakeSdkSession(wire_messages=[_wire("TextPart", text="ok")])
    patch_sdk["sdk_session"] = fake

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session()

    # Drive a turn so the SDK session materialises, then manually flip
    # ``_in_flight`` and call cancel — pre-flight (no in_flight) we'd
    # short-circuit. Behaviour: cancel forwards to fake.cancel().
    asyncio.run(sess.execute("hi"))
    sess._in_flight = True  # type: ignore[attr-defined]
    asyncio.run(sess.cancel())
    assert fake.cancel_calls == 1
    sess._in_flight = False  # type: ignore[attr-defined]
    asyncio.run(sess.close())


def test_close_is_idempotent_and_closes_sdk_session(patch_sdk: dict[str, Any]) -> None:
    from airframe.adapters.kimi import KimiRuntime

    fake = _FakeSdkSession(wire_messages=[_wire("TextPart", text="ok")])
    patch_sdk["sdk_session"] = fake

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session()
    asyncio.run(sess.execute("hi"))  # materialise
    asyncio.run(sess.close())
    asyncio.run(sess.close())  # second call must not raise
    assert fake.closed is True


def test_close_on_never_used_session_is_safe(patch_sdk: dict[str, Any]) -> None:
    """``close()`` on a session that never reached the SDK call must not crash."""
    from airframe.adapters.kimi import KimiRuntime

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session()
    # Never called execute / stream — _sdk_session is None.
    asyncio.run(sess.close())


# ---------------------------------------------------------------------------
# Exception classification
# ---------------------------------------------------------------------------


def _make_exc(name: str, *, status_code: int | None = None, msg: str = "boom") -> BaseException:
    """Build a stand-in exception whose ``type().__name__`` matches ``name``."""
    cls = type(name, (Exception,), {})
    instance = cls(msg)
    if status_code is not None:
        instance.status_code = status_code  # type: ignore[attr-defined]
    return instance


@pytest.mark.parametrize(
    "exc_name,expected_error",
    [
        ("RunCancelled", RuntimeCancelledError),
        ("APIConnectionError", RuntimeTransientError),
        ("APITimeoutError", RuntimeTransientError),
        ("LLMNotSet", RuntimeAuthError),
        ("LLMNotSupported", RuntimeAuthError),
        ("MaxStepsReached", RuntimeProtocolError),
        ("MCPRuntimeError", RuntimeProtocolError),
        ("ConfigError", RuntimeProtocolError),
        ("APIEmptyResponseError", RuntimeProtocolError),
        ("UnknownVendorError", RuntimeProtocolError),  # fall-through
    ],
)
def test_sdk_exception_classification(
    patch_sdk: dict[str, Any], exc_name: str, expected_error: type[BaseException]
) -> None:
    from airframe.adapters.kimi import KimiRuntime

    patch_sdk["sdk_session"] = _FakeSdkSession(raise_on_prompt=_make_exc(exc_name))

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session()
    with pytest.raises(expected_error):
        asyncio.run(sess.execute("hi"))
    asyncio.run(sess.close())


@pytest.mark.parametrize(
    "status_code,expected_error",
    [
        (401, RuntimeAuthError),
        (403, RuntimeAuthError),
        (429, RuntimeTransientError),
        (502, RuntimeTransientError),
        (503, RuntimeTransientError),
        (504, RuntimeTransientError),
        (400, RuntimeProtocolError),
        (500, RuntimeProtocolError),  # unclassified 5xx → protocol
    ],
)
def test_api_status_error_classification_by_code(
    patch_sdk: dict[str, Any], status_code: int, expected_error: type[BaseException]
) -> None:
    from airframe.adapters.kimi import KimiRuntime

    patch_sdk["sdk_session"] = _FakeSdkSession(
        raise_on_prompt=_make_exc("APIStatusError", status_code=status_code)
    )

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session()
    with pytest.raises(expected_error):
        asyncio.run(sess.execute("hi"))
    asyncio.run(sess.close())


# ---------------------------------------------------------------------------
# KimiOptions.working_directory threading
# ---------------------------------------------------------------------------


def test_working_directory_option_threads_to_session_create(
    patch_sdk: dict[str, Any],
) -> None:
    from airframe.adapters.kimi import KimiRuntime
    from airframe.options import KimiOptions

    patch_sdk["sdk_session"] = _FakeSdkSession(wire_messages=[_wire("TextPart", text="ok")])

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session(provider_options=KimiOptions(working_directory="/tmp/kimi-work"))
    asyncio.run(sess.execute("hi"))

    work_dir = patch_sdk["create_calls"][0]["work_dir"]
    # KaosPath stand-in stores the path as ``_path``.
    assert getattr(work_dir, "_path", None) == "/tmp/kimi-work"
    asyncio.run(sess.close())


# ---------------------------------------------------------------------------
# Env var injection: KIMI_API_KEY mutation
# ---------------------------------------------------------------------------


def test_api_key_constructor_arg_sets_env_for_sdk_call(
    monkeypatch: pytest.MonkeyPatch,
    patch_sdk: dict[str, Any],
) -> None:
    """Explicit ``api_key=`` populates ``KIMI_API_KEY`` for the SDK's auth chain."""
    import os

    from airframe.adapters.kimi import KimiRuntime

    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    captured_env: dict[str, str | None] = {}

    class _RecorderSession(_FakeSdkSession):
        async def prompt(self, user_input: str, *, merge_wire_messages: bool = False):  # type: ignore[override]
            captured_env["during_call"] = os.environ.get("KIMI_API_KEY")
            for w in [_wire("TextPart", text="ok")]:
                yield w

    patch_sdk["sdk_session"] = _RecorderSession()

    rt = KimiRuntime(api_key="sk-explicit")
    sess = rt.session()
    asyncio.run(sess.execute("hi"))

    assert captured_env["during_call"] == "sk-explicit"
    # And it's restored after close.
    asyncio.run(sess.close())
    assert os.environ.get("KIMI_API_KEY") is None


# ---------------------------------------------------------------------------
# Iteration C: reasoning (thinking=)
# ---------------------------------------------------------------------------


def test_thinking_disabled_passes_false_to_session_create(
    patch_sdk: dict[str, Any],
) -> None:
    """``thinking=None`` (default) and ``thinking="disabled"`` both
    pass ``thinking=False`` to ``Session.create``."""
    from airframe import KimiRuntime

    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[_wire("TextPart", text="ok")],
    )

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session()
    asyncio.run(sess.execute("hi"))  # thinking= defaults to None

    assert len(patch_sdk["create_calls"]) == 1
    assert patch_sdk["create_calls"][0]["thinking"] is False
    asyncio.run(sess.close())


@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high"])
def test_thinking_effort_literal_passes_true_to_session_create(
    patch_sdk: dict[str, Any], effort: str
) -> None:
    """Every effort literal collapses to ``thinking=True`` — kimi-agent-sdk
    has no effort granularity; the model decides depth itself."""
    from airframe import KimiRuntime

    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[_wire("TextPart", text="ok")],
    )

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session()
    asyncio.run(sess.execute("hi", thinking=effort))

    assert len(patch_sdk["create_calls"]) == 1
    assert patch_sdk["create_calls"][0]["thinking"] is True
    asyncio.run(sess.close())


def test_thinking_dict_shape_raises_unsupported_feature(
    patch_sdk: dict[str, Any],
) -> None:
    """``thinking={"budget_tokens": N}`` is Claude-only — Kimi has no
    token-budget knob. Raise :class:`UnsupportedFeatureError` with
    :data:`Feature.REASONING_BUDGET_TOKENS`."""
    from airframe import KimiRuntime

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session()
    with pytest.raises(UnsupportedFeatureError) as excinfo:
        asyncio.run(sess.execute("hi", thinking={"budget_tokens": 8192}))
    assert excinfo.value.feature is Feature.REASONING_BUDGET_TOKENS
    asyncio.run(sess.close())


def test_thinking_toggle_between_turns_rebuilds_session(
    patch_sdk: dict[str, Any],
) -> None:
    """Switching ``thinking=`` between turns must close the existing
    SDK session and rebuild — :meth:`Session.create` bakes the flag
    at creation and never re-evaluates."""
    from airframe import KimiRuntime

    first = _FakeSdkSession(
        session_id="sess-1",
        wire_messages=[_wire("TextPart", text="a")],
    )
    second = _FakeSdkSession(
        session_id="sess-1",  # same ID — rebuild resumes by ID
        wire_messages=[_wire("TextPart", text="b")],
    )
    sessions = iter([first, second])

    async def fake_create(**kwargs: Any) -> Any:
        patch_sdk["create_calls"].append(kwargs)
        return next(sessions)

    async def fake_resume(**kwargs: Any) -> Any:
        patch_sdk["resume_calls"].append(kwargs)
        return next(sessions)

    sdk_module = sys.modules["kimi_agent_sdk"]
    sdk_module.Session.create = fake_create
    sdk_module.Session.resume = fake_resume

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session()

    # Turn 1: thinking=disabled → Session.create(thinking=False).
    asyncio.run(sess.execute("first", thinking="disabled"))
    assert len(patch_sdk["create_calls"]) == 1
    assert patch_sdk["create_calls"][0]["thinking"] is False
    assert not first.closed

    # Turn 2: thinking=high → first session closed, rebuild via
    # Session.resume (carries the session id from turn 1) with
    # thinking=True.
    asyncio.run(sess.execute("second", thinking="high"))
    assert first.closed, "first SDK session must be closed on rebuild"
    assert len(patch_sdk["resume_calls"]) == 1
    assert patch_sdk["resume_calls"][0]["thinking"] is True
    assert patch_sdk["resume_calls"][0]["session_id"] == "sess-1"
    asyncio.run(sess.close())


def test_thinking_unchanged_between_turns_reuses_session(
    patch_sdk: dict[str, Any],
) -> None:
    """No rebuild when consecutive turns share the same ``thinking=``."""
    from airframe import KimiRuntime

    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[_wire("TextPart", text="ok")],
    )

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session()
    asyncio.run(sess.execute("first", thinking="medium"))
    asyncio.run(sess.execute("second", thinking="medium"))

    # One create call, no resume call — the SDK session was reused.
    assert len(patch_sdk["create_calls"]) == 1
    assert len(patch_sdk["resume_calls"]) == 0
    asyncio.run(sess.close())


# ---------------------------------------------------------------------------
# Iteration C: polymorphic prompt (vision)
# ---------------------------------------------------------------------------


def test_plain_string_prompt_passes_through_as_string(
    patch_sdk: dict[str, Any],
) -> None:
    """No images → user_input is the bare ``str`` (no list-wrap)."""
    from airframe import KimiRuntime

    sdk_session = _FakeSdkSession(wire_messages=[_wire("TextPart", text="ok")])
    patch_sdk["sdk_session"] = sdk_session

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session()
    asyncio.run(sess.execute("hello"))

    assert sdk_session.prompt_calls == ["hello"]
    asyncio.run(sess.close())


def test_image_input_url_passes_through_as_image_url_part(
    patch_sdk: dict[str, Any],
) -> None:
    """``ImageInput(url=…)`` lands as an :class:`ImageURLPart` with
    the URL forwarded verbatim — no data-URI conversion."""
    from airframe import KimiRuntime

    sdk_session = _FakeSdkSession(wire_messages=[_wire("TextPart", text="ok")])
    patch_sdk["sdk_session"] = sdk_session

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session()
    asyncio.run(
        sess.execute(
            [
                "What's in this image?",
                ImageInput(url="https://example.com/cat.png"),
            ]
        )
    )

    assert len(sdk_session.prompt_calls) == 1
    parts = sdk_session.prompt_calls[0]
    # Expect [TextPart, ImageURLPart].
    assert len(parts) == 2
    assert type(parts[0]).__name__ == "_FakeTextPart"
    assert parts[0].text == "What's in this image?"
    assert type(parts[1]).__name__ == "_FakeImageURLPart"
    assert parts[1].image_url.url == "https://example.com/cat.png"
    asyncio.run(sess.close())


def test_image_input_bytes_becomes_data_uri(patch_sdk: dict[str, Any]) -> None:
    """``ImageInput(bytes_=…)`` is base64-encoded into a ``data:`` URI."""
    from airframe import KimiRuntime

    sdk_session = _FakeSdkSession(wire_messages=[_wire("TextPart", text="ok")])
    patch_sdk["sdk_session"] = sdk_session

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session()
    asyncio.run(
        sess.execute(
            [
                "Describe.",
                ImageInput(bytes_=b"\x89PNG fake png bytes", media_type="image/png"),
            ]
        )
    )

    parts = sdk_session.prompt_calls[0]
    url = parts[1].image_url.url
    assert url.startswith("data:image/png;base64,")
    # The base64 portion decodes back to our input bytes.
    import base64

    payload = url.split(",", 1)[1]
    assert base64.b64decode(payload) == b"\x89PNG fake png bytes"
    asyncio.run(sess.close())


def test_image_input_path_becomes_data_uri(patch_sdk: dict[str, Any], tmp_path: Any) -> None:
    """``ImageInput(path=…)`` reads the file, sniffs media type from
    the extension, and encodes as a ``data:`` URI."""
    from airframe import KimiRuntime

    img_bytes = b"PNG-on-disk-content"
    img_path = tmp_path / "screenshot.png"
    img_path.write_bytes(img_bytes)

    sdk_session = _FakeSdkSession(wire_messages=[_wire("TextPart", text="ok")])
    patch_sdk["sdk_session"] = sdk_session

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session()
    asyncio.run(
        sess.execute(["Look:", ImageInput(path=str(img_path))]),
    )

    url = sdk_session.prompt_calls[0][1].image_url.url
    assert url.startswith("data:image/png;base64,")
    import base64

    payload = url.split(",", 1)[1]
    assert base64.b64decode(payload) == img_bytes
    asyncio.run(sess.close())


def test_image_input_missing_file_raises_unsupported_feature(
    patch_sdk: dict[str, Any], tmp_path: Any
) -> None:
    """``ImageInput(path=…)`` where the file doesn't exist surfaces as
    :class:`UnsupportedFeatureError` (a configuration error — better
    than silently bubbling an :class:`OSError` from inside the SDK)."""
    from airframe import KimiRuntime

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session()
    with pytest.raises(UnsupportedFeatureError) as excinfo:
        asyncio.run(
            sess.execute(
                ["x", ImageInput(path=str(tmp_path / "nope.png"))],
            )
        )
    assert excinfo.value.feature is Feature.VISION_INPUT
    assert "file not found" in str(excinfo.value)
    asyncio.run(sess.close())


def test_file_input_declined_via_split_prompt_parts(
    patch_sdk: dict[str, Any], tmp_path: Any
) -> None:
    """``FileInput`` is declined — Kimi has no prompt-side file slot.
    The shared ``_split_prompt_parts`` helper raises
    :class:`UnsupportedFeatureError` with :data:`Feature.FILE_INPUT`."""
    from airframe import KimiRuntime
    from airframe.inputs import FileInput

    f = tmp_path / "doc.md"
    f.write_text("# doc")

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session()
    with pytest.raises(UnsupportedFeatureError) as excinfo:
        asyncio.run(sess.execute(["read this:", FileInput(path=str(f))]))
    assert excinfo.value.feature is Feature.FILE_INPUT
    asyncio.run(sess.close())


def test_system_prompt_prepends_to_textpart_when_images_present(
    patch_sdk: dict[str, Any],
) -> None:
    """The session's ``system`` prefix lands on the ``TextPart.text``,
    not as a separate part. Mirrors the plain-string-prompt behaviour
    (where ``system`` already concatenates onto the prompt text)."""
    from airframe import KimiRuntime

    sdk_session = _FakeSdkSession(wire_messages=[_wire("TextPart", text="ok")])
    patch_sdk["sdk_session"] = sdk_session

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session(system="You are precise.")
    asyncio.run(
        sess.execute(
            [
                "Caption:",
                ImageInput(url="https://example.com/a.png"),
            ]
        )
    )

    parts = sdk_session.prompt_calls[0]
    assert parts[0].text == "You are precise.\n\nCaption:"
    asyncio.run(sess.close())


# ---------------------------------------------------------------------------
# Iteration D: PermissionCallback ↔ ApprovalRequest dispatch
# ---------------------------------------------------------------------------


class _RecordingPermissionCallback:
    """In-test ``PermissionCallback`` that records every dispatched
    :class:`PermissionRequest` and replays a queued decision per call.

    Matching the ``PermissionCallback`` protocol shape:
    ``async def handle(request) -> PermissionDecision``.
    """

    def __init__(self, decisions: list[str]) -> None:
        self._decisions = list(decisions)
        self.requests: list[Any] = []

    async def handle(self, request: Any) -> str:
        self.requests.append(request)
        return self._decisions.pop(0)  # raises if test runs past queue


def test_yolo_true_by_default_when_no_permission_callback(
    patch_sdk: dict[str, Any],
) -> None:
    """No ``on_permission`` → ``yolo=True`` on Session.create (B+C
    behaviour preserved). Approval requests that slip through are
    auto-approved."""
    from airframe import KimiRuntime

    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[_wire("TextPart", text="ok")],
    )

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session()
    asyncio.run(sess.execute("hi"))

    assert patch_sdk["create_calls"][0]["yolo"] is True
    asyncio.run(sess.close())


def test_yolo_false_when_permission_callback_supplied(
    patch_sdk: dict[str, Any],
) -> None:
    """``on_permission=callback`` flips ``yolo=False`` on Session.create
    so the SDK surfaces ApprovalRequests for the adapter to dispatch."""
    from airframe import KimiRuntime

    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[_wire("TextPart", text="ok")],
    )

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session(on_permission=_RecordingPermissionCallback(["allow"]))
    asyncio.run(sess.execute("hi"))

    assert patch_sdk["create_calls"][0]["yolo"] is False
    asyncio.run(sess.close())


def test_approval_request_dispatched_to_callback_with_allow_resolves_approve(
    patch_sdk: dict[str, Any],
) -> None:
    """allow → ``ApprovalRequest.resolve("approve")``."""
    from airframe import KimiRuntime

    approval = ApprovalRequest(
        request_id="req-A",
        tool_call_id="tc-1",
        sender="agent",
        action="shell.run",
        description="Run `ls -la`",
    )
    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[approval, _wire("TextPart", text="ok")],
    )

    callback = _RecordingPermissionCallback(["allow"])
    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session(on_permission=callback)
    asyncio.run(sess.execute("hi"))

    assert approval.resolved
    assert approval.resolved_with == "approve"
    assert approval.resolved_feedback == ""
    # The callback saw a PermissionRequest with the wire's fields lifted.
    assert len(callback.requests) == 1
    req = callback.requests[0]
    assert req.tool_name == "shell.run"
    assert req.tool_args == {"tool_call_id": "tc-1", "sender": "agent"}
    assert req.reason == "Run `ls -la`"
    asyncio.run(sess.close())


def test_approval_request_deny_resolves_reject(
    patch_sdk: dict[str, Any],
) -> None:
    """deny → ``ApprovalRequest.resolve("reject")`` with empty feedback."""
    from airframe import KimiRuntime

    approval = ApprovalRequest(request_id="req-B")
    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[approval, _wire("TextPart", text="x")],
    )

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session(on_permission=_RecordingPermissionCallback(["deny"]))
    asyncio.run(sess.execute("hi"))

    assert approval.resolved_with == "reject"
    assert approval.resolved_feedback == ""
    asyncio.run(sess.close())


def test_approval_request_defer_resolves_reject_with_feedback(
    patch_sdk: dict[str, Any],
) -> None:
    """defer → reject + feedback explaining the SDK has no async
    "ask the human later" channel."""
    from airframe import KimiRuntime

    approval = ApprovalRequest(request_id="req-C")
    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[approval, _wire("TextPart", text="x")],
    )

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session(on_permission=_RecordingPermissionCallback(["defer"]))
    asyncio.run(sess.execute("hi"))

    assert approval.resolved_with == "reject"
    assert "deferred" in approval.resolved_feedback.lower()
    asyncio.run(sess.close())


def test_approval_request_auto_approved_when_no_callback_registered(
    patch_sdk: dict[str, Any],
) -> None:
    """Defensive: an ApprovalRequest reaching the adapter without a
    callback registered (shouldn't happen with yolo=True but the SDK
    surface allows it) gets ``"approve"`` to keep the prompt stream
    moving rather than stalling forever."""
    from airframe import KimiRuntime

    approval = ApprovalRequest(request_id="req-D")
    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[approval, _wire("TextPart", text="ok")],
    )

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session()  # no on_permission
    asyncio.run(sess.execute("hi"))

    assert approval.resolved_with == "approve"
    asyncio.run(sess.close())


def test_permission_callback_dispatched_on_stream_path_too(
    patch_sdk: dict[str, Any],
) -> None:
    """The dispatch is shared by execute() and stream() — verify the
    stream path also calls the callback."""
    from airframe import KimiRuntime

    approval = ApprovalRequest(request_id="req-E")
    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[approval, _wire("TextPart", text="bye")],
    )

    callback = _RecordingPermissionCallback(["allow"])
    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session(on_permission=callback)

    async def drain() -> None:
        async for _ in sess.stream("hi"):
            pass

    asyncio.run(drain())
    assert approval.resolved_with == "approve"
    assert len(callback.requests) == 1
    asyncio.run(sess.close())


# ---------------------------------------------------------------------------
# Iteration D: MCP server refs
# ---------------------------------------------------------------------------


def test_session_without_mcp_servers_omits_mcp_configs(
    patch_sdk: dict[str, Any],
) -> None:
    """No mcp_servers= → no ``mcp_configs`` kwarg on Session.create."""
    from airframe import KimiRuntime

    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[_wire("TextPart", text="ok")],
    )

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session()
    asyncio.run(sess.execute("hi"))

    assert "mcp_configs" not in patch_sdk["create_calls"][0]
    asyncio.run(sess.close())


def test_stdio_mcp_ref_translates_to_session_create_kwarg(
    patch_sdk: dict[str, Any],
) -> None:
    """``McpServerRef(transport="stdio")`` lands as ``mcp_configs=[
    {"mcpServers": {<name>: {"command": ..., "args": [...]}}}]`` on
    Session.create."""
    from airframe import KimiRuntime
    from airframe.tools import McpServerRef

    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[_wire("TextPart", text="ok")],
    )

    ref = McpServerRef(
        name="everything",
        transport="stdio",
        command=["uvx", "mcp-server-everything", "--flag"],
    )
    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session(mcp_servers=[ref])
    asyncio.run(sess.execute("hi"))

    mcp_configs = patch_sdk["create_calls"][0]["mcp_configs"]
    assert mcp_configs == [
        {
            "mcpServers": {
                "everything": {
                    "command": "uvx",
                    "args": ["mcp-server-everything", "--flag"],
                }
            }
        }
    ]
    asyncio.run(sess.close())


def test_http_mcp_ref_translates_with_auth_bearer_header(
    patch_sdk: dict[str, Any],
) -> None:
    """``McpServerRef(transport="http", auth_token=...)`` →
    ``{"url": ..., "transport": "http", "headers": {"Authorization":
    "Bearer ..."}}``."""
    from airframe import KimiRuntime
    from airframe.tools import McpServerRef

    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[_wire("TextPart", text="ok")],
    )

    ref = McpServerRef(
        name="github",
        transport="http",
        url="https://mcp.example.com/v1",
        headers={"X-Trace-Id": "abc"},
        auth_token="ghp_xxx",
    )
    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session(mcp_servers=[ref])
    asyncio.run(sess.execute("hi"))

    server_config = patch_sdk["create_calls"][0]["mcp_configs"][0]["mcpServers"]["github"]
    assert server_config["url"] == "https://mcp.example.com/v1"
    assert server_config["transport"] == "http"
    assert server_config["headers"]["X-Trace-Id"] == "abc"
    assert server_config["headers"]["Authorization"] == "Bearer ghp_xxx"
    asyncio.run(sess.close())


def test_sse_mcp_ref_translates_to_sse_transport(
    patch_sdk: dict[str, Any],
) -> None:
    """``transport="sse"`` round-trips unchanged."""
    from airframe import KimiRuntime
    from airframe.tools import McpServerRef

    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[_wire("TextPart", text="ok")],
    )

    ref = McpServerRef(
        name="livesearch",
        transport="sse",
        url="https://sse.example.com/feed",
    )
    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session(mcp_servers=[ref])
    asyncio.run(sess.execute("hi"))

    cfg = patch_sdk["create_calls"][0]["mcp_configs"][0]["mcpServers"]["livesearch"]
    assert cfg["transport"] == "sse"
    assert cfg["url"] == "https://sse.example.com/feed"
    asyncio.run(sess.close())


def test_caller_supplied_authorization_header_wins_over_auth_token(
    patch_sdk: dict[str, Any],
) -> None:
    """Caller-supplied ``Authorization`` header beats ``auth_token=``
    on collision — same precedence as the other adapters."""
    from airframe import KimiRuntime
    from airframe.tools import McpServerRef

    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[_wire("TextPart", text="ok")],
    )

    ref = McpServerRef(
        name="srv",
        transport="http",
        url="https://x.example.com",
        headers={"Authorization": "Custom abc"},
        auth_token="overridden",
    )
    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session(mcp_servers=[ref])
    asyncio.run(sess.execute("hi"))

    server_config = patch_sdk["create_calls"][0]["mcp_configs"][0]["mcpServers"]["srv"]
    assert server_config["headers"]["Authorization"] == "Custom abc"
    asyncio.run(sess.close())


def test_multiple_mcp_refs_bundle_into_one_mcpconfig(
    patch_sdk: dict[str, Any],
) -> None:
    """Multiple refs land in a single ``MCPConfig.mcpServers`` dict
    keyed by name — the SDK accepts a list of configs but bundling is
    the canonical shape."""
    from airframe import KimiRuntime
    from airframe.tools import McpServerRef

    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[_wire("TextPart", text="ok")],
    )

    refs = [
        McpServerRef(name="a", transport="stdio", command=["a"]),
        McpServerRef(name="b", transport="http", url="https://b.example.com"),
    ]
    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session(mcp_servers=refs)
    asyncio.run(sess.execute("hi"))

    mcp_configs = patch_sdk["create_calls"][0]["mcp_configs"]
    assert len(mcp_configs) == 1  # bundled
    assert set(mcp_configs[0]["mcpServers"].keys()) == {"a", "b"}
    asyncio.run(sess.close())


def test_duplicate_mcp_ref_names_raise_at_session_construction() -> None:
    """Two refs with the same ``name`` would silently overwrite in the
    MCPConfig dict — raise a ValueError synchronously instead."""
    from airframe import KimiRuntime
    from airframe.tools import McpServerRef

    refs = [
        McpServerRef(name="dup", transport="stdio", command=["a"]),
        McpServerRef(name="dup", transport="stdio", command=["b"]),
    ]
    rt = KimiRuntime(api_key="sk-test")
    with pytest.raises(ValueError) as excinfo:
        rt.session(mcp_servers=refs)
    assert "duplicate" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# Iteration D: function-tool permanent decline
# ---------------------------------------------------------------------------


def test_tools_kwarg_raises_unsupported_feature_pointing_at_mcp() -> None:
    """``tools=`` is a permanent decline — the kimi-agent-sdk Python
    surface has no programmatic Python-callable tool channel.
    The error message must point consumers at ``mcp_servers=``."""
    from pydantic import BaseModel

    from airframe import KimiRuntime
    from airframe.tools import FunctionTool

    class _NoArgs(BaseModel):
        pass

    async def my_tool(params: _NoArgs) -> str:
        return "ok"

    tool = FunctionTool(
        name="my_tool",
        description="x",
        params=_NoArgs,
        handler=my_tool,
    )
    rt = KimiRuntime(api_key="sk-test")
    with pytest.raises(UnsupportedFeatureError) as excinfo:
        rt.session(tools=[tool])
    assert excinfo.value.feature is Feature.TOOLS_FUNCTION
    msg = str(excinfo.value)
    assert "mcp_servers" in msg


# ---------------------------------------------------------------------------
# Iteration E: pricing
# ---------------------------------------------------------------------------


def test_cost_usd_populated_from_in_tree_pricing(patch_sdk: dict[str, Any]) -> None:
    """``CostRecord.cost_usd`` populates from :data:`_KIMI_PRICING`
    when the model is in the table. ``cache_read_tokens`` bills at
    the cheaper cache rate, not the fresh-input rate."""
    from airframe import KimiRuntime
    from airframe.adapters.kimi import _KIMI_PRICING

    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[
            _wire("TextPart", text="ok"),
            _wire(
                "TokenUsage",
                input_tokens=1000,
                output_tokens=500,
                cache_read_tokens=200,
            ),
        ]
    )

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session()
    result = asyncio.run(sess.execute("hi"))

    in_rate, out_rate, cache_rate = _KIMI_PRICING["kimi-k2-thinking-turbo"]
    fresh = 1000 - 200
    expected = round(
        (fresh / 1000.0) * in_rate + (200 / 1000.0) * cache_rate + (500 / 1000.0) * out_rate,
        6,
    )
    assert result.cost.cost_usd == expected
    asyncio.run(sess.close())


def test_cost_usd_none_for_models_outside_pricing_table(
    patch_sdk: dict[str, Any],
) -> None:
    """Models not in :data:`_KIMI_PRICING` keep ``cost_usd=None`` so
    consumer code can still trust token counts as a budget proxy."""
    from airframe import KimiRuntime, ProviderModel

    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[
            _wire("TextPart", text="ok"),
            _wire("TokenUsage", input_tokens=100, output_tokens=50),
        ]
    )

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session(model=ProviderModel("kimi", "kimi-future-unreleased"))
    result = asyncio.run(sess.execute("hi"))

    assert result.cost.cost_usd is None
    assert result.cost.input_tokens == 100
    asyncio.run(sess.close())


# ---------------------------------------------------------------------------
# Iteration E: lifecycle hooks
# ---------------------------------------------------------------------------


def _collect_events() -> tuple[list[Any], Any]:
    """Return a (events_list, observer_callable) pair for hook tests."""
    events: list[Any] = []

    def observer(ev: Any) -> None:
        events.append(ev)

    return events, observer


def test_session_start_fires_on_first_execute_only(
    patch_sdk: dict[str, Any],
) -> None:
    """``session_start`` fires once on the first execute() / stream();
    subsequent turns don't re-fire."""
    from airframe import KimiRuntime

    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[_wire("TextPart", text="ok")],
    )

    events, observer = _collect_events()
    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session(on_event=observer)
    asyncio.run(sess.execute("first"))
    asyncio.run(sess.execute("second"))

    start_events = [e for e in events if e.kind == "session_start"]
    assert len(start_events) == 1
    assert start_events[0].payload["model"] == "kimi-k2-thinking-turbo"
    assert start_events[0].payload["resumed"] is False
    asyncio.run(sess.close())


def test_session_end_fires_on_close_with_cumulative_payload(
    patch_sdk: dict[str, Any],
) -> None:
    """``session_end`` carries ``turn_count`` + ``cost_usd``
    cumulative-since-session-start."""
    from airframe import KimiRuntime

    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[
            _wire("TextPart", text="ok"),
            _wire("TokenUsage", input_tokens=100, output_tokens=50),
        ]
    )

    events, observer = _collect_events()
    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session(on_event=observer)
    asyncio.run(sess.execute("hi"))
    asyncio.run(sess.close())

    end_events = [e for e in events if e.kind == "session_end"]
    assert len(end_events) == 1
    payload = end_events[0].payload
    assert payload["turn_count"] == 1
    assert payload["cost_usd"] > 0


def test_session_end_does_not_fire_on_close_without_session_start(
    patch_sdk: dict[str, Any],
) -> None:
    """A session that was opened and immediately closed without ever
    running a turn does NOT emit either lifecycle event."""
    from airframe import KimiRuntime

    events, observer = _collect_events()
    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session(on_event=observer)
    asyncio.run(sess.close())

    assert events == []  # neither start nor end fired


def test_user_prompt_submit_fires_each_turn(patch_sdk: dict[str, Any]) -> None:
    """``user_prompt_submit`` fires once per execute() / stream(),
    carrying ``prompt`` (the post-system-prompt text) and ``length``."""
    from airframe import KimiRuntime

    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[_wire("TextPart", text="x")],
    )

    events, observer = _collect_events()
    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session(on_event=observer)
    asyncio.run(sess.execute("first prompt"))
    asyncio.run(sess.execute("second prompt"))

    submit_events = [e for e in events if e.kind == "user_prompt_submit"]
    assert len(submit_events) == 2
    assert submit_events[0].payload["prompt"] == "first prompt"
    assert submit_events[0].payload["length"] == len("first prompt")
    assert submit_events[1].payload["prompt"] == "second prompt"


def _tool_call(call_id: str, name: str, arguments: str | None = None) -> Any:
    """Build a fake ``ToolCall`` wire whose attribute shape matches kosong."""

    class _FakeFunction:
        def __init__(self, name: str, arguments: str | None) -> None:
            self.name = name
            self.arguments = arguments

    class ToolCall:  # noqa: N801 — must match SDK type name for adapter dispatch
        def __init__(self, id: str, function: Any) -> None:
            self.id = id
            self.function = function

    return ToolCall(id=call_id, function=_FakeFunction(name, arguments))


def _tool_result(
    tool_call_id: str,
    *,
    is_error: bool,
    message: str = "",
    output: Any = None,
) -> Any:
    """Build a fake ``ToolResult`` wire with the kosong shape."""

    class _ReturnValue:
        def __init__(self, is_error: bool, message: str, output: Any) -> None:
            self.is_error = is_error
            self.message = message
            self.output = output

    class ToolResult:  # noqa: N801 — must match SDK type name
        def __init__(self, tool_call_id: str, return_value: Any) -> None:
            self.tool_call_id = tool_call_id
            self.return_value = return_value

    return ToolResult(
        tool_call_id=tool_call_id,
        return_value=_ReturnValue(is_error, message, output),
    )


def test_pre_tool_use_fires_on_toolcall_wire(patch_sdk: dict[str, Any]) -> None:
    """A ``ToolCall`` on the wire stream emits ``pre_tool_use`` with
    name + tool_call_id + arguments."""
    from airframe import KimiRuntime

    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[
            _tool_call("tc-1", "shell", arguments='{"cmd": "ls"}'),
            _wire("TextPart", text="done"),
        ],
    )

    events, observer = _collect_events()
    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session(on_event=observer)
    asyncio.run(sess.execute("hi"))

    pre = [e for e in events if e.kind == "pre_tool_use"]
    assert len(pre) == 1
    assert pre[0].payload["tool_name"] == "shell"
    assert pre[0].payload["tool_call_id"] == "tc-1"
    assert pre[0].payload["arguments"] == '{"cmd": "ls"}'


def test_post_tool_use_fires_on_successful_toolresult(
    patch_sdk: dict[str, Any],
) -> None:
    """A non-error ``ToolResult`` emits ``post_tool_use`` with the
    tool's output / message lifted as ``output``."""
    from airframe import KimiRuntime

    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[
            _tool_result("tc-1", is_error=False, message="ok", output="file list"),
            _wire("TextPart", text="ok"),
        ],
    )

    events, observer = _collect_events()
    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session(on_event=observer)
    asyncio.run(sess.execute("hi"))

    post = [e for e in events if e.kind == "post_tool_use"]
    assert len(post) == 1
    assert post[0].payload["tool_call_id"] == "tc-1"
    assert post[0].payload["output"] == "file list"


def test_tool_failure_fires_on_errored_toolresult(
    patch_sdk: dict[str, Any],
) -> None:
    """An error ``ToolResult`` (``return_value.is_error=True``) emits
    ``tool_failure`` with the message lifted as ``error``."""
    from airframe import KimiRuntime

    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[
            _tool_result("tc-2", is_error=True, message="permission denied"),
            _wire("TextPart", text="x"),
        ],
    )

    events, observer = _collect_events()
    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session(on_event=observer)
    asyncio.run(sess.execute("hi"))

    failures = [e for e in events if e.kind == "tool_failure"]
    assert len(failures) == 1
    assert failures[0].payload["tool_call_id"] == "tc-2"
    assert failures[0].payload["error"] == "permission denied"


def test_pre_compact_fires_on_compaction_begin(
    patch_sdk: dict[str, Any],
) -> None:
    """A ``CompactionBegin`` wire emits ``pre_compact``. The matching
    ``CompactionEnd`` is intentionally silent (no ``post_compact`` kind
    in airframe's taxonomy)."""
    from airframe import KimiRuntime

    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[
            _wire("CompactionBegin"),
            _wire("CompactionEnd"),
            _wire("TextPart", text="x"),
        ],
    )

    events, observer = _collect_events()
    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session(on_event=observer)
    asyncio.run(sess.execute("hi"))

    pre_compact = [e for e in events if e.kind == "pre_compact"]
    assert len(pre_compact) == 1


def test_hook_events_fire_on_stream_path_too(
    patch_sdk: dict[str, Any],
) -> None:
    """The hook emission is shared by execute() and stream()."""
    from airframe import KimiRuntime

    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[
            _tool_call("tc-9", "shell", arguments="{}"),
            _wire("TextPart", text="x"),
        ],
    )

    events, observer = _collect_events()
    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session(on_event=observer)

    async def drain() -> None:
        async for _ in sess.stream("hi"):
            pass

    asyncio.run(drain())
    pre = [e for e in events if e.kind == "pre_tool_use"]
    assert len(pre) == 1


def test_hook_observer_exception_does_not_break_session(
    patch_sdk: dict[str, Any],
) -> None:
    """A raising observer must not propagate — the session must
    continue. :func:`_fire_hook_event` swallows non-system
    exceptions; we just confirm the contract here."""
    from airframe import KimiRuntime

    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[_wire("TextPart", text="ok")],
    )

    def boom(_ev: Any) -> None:
        raise RuntimeError("observer broke")

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session(on_event=boom)
    # Must not raise.
    result = asyncio.run(sess.execute("hi"))
    assert result.text == "ok"
    asyncio.run(sess.close())


# ---------------------------------------------------------------------------
# Iteration E: budget enforcement
# ---------------------------------------------------------------------------


def test_max_turns_cap_trips_after_n_turns(patch_sdk: dict[str, Any]) -> None:
    """``max_turns=2`` lets the first two turns through and aborts the
    third with :class:`RuntimeBudgetExceededError`."""
    from airframe import KimiRuntime
    from airframe.errors import RuntimeBudgetExceededError

    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[_wire("TextPart", text="ok")],
    )

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session()
    asyncio.run(sess.execute("first", max_turns=2))
    asyncio.run(sess.execute("second", max_turns=2))
    with pytest.raises(RuntimeBudgetExceededError) as excinfo:
        asyncio.run(sess.execute("third", max_turns=2))
    assert excinfo.value.kind == "turns"
    assert excinfo.value.cap == 2.0
    asyncio.run(sess.close())


def test_max_budget_usd_cap_trips_when_cumulative_spend_exceeds(
    patch_sdk: dict[str, Any],
) -> None:
    """``max_budget_usd`` aborts the next turn once cumulative spend
    has met the cap. Pre-turn check uses ``cumulative >= cap``, so the
    second turn aborts as soon as the first turn's cost reached the
    cap."""
    from airframe import KimiRuntime
    from airframe.errors import RuntimeBudgetExceededError

    # First turn: ~$0.004 (1000 input @ $0.0015/1k + 500 output @ $0.005/1k
    # = 0.0015 + 0.0025 = $0.004). Cap at $0.003 → second turn trips.
    patch_sdk["sdk_session"] = _FakeSdkSession(
        wire_messages=[
            _wire("TextPart", text="ok"),
            _wire("TokenUsage", input_tokens=1000, output_tokens=500),
        ],
    )

    rt = KimiRuntime(api_key="sk-test")
    sess = rt.session()
    asyncio.run(sess.execute("first", max_budget_usd=0.003))
    # Cumulative now $0.004 >= cap $0.003 → second turn must abort.
    with pytest.raises(RuntimeBudgetExceededError) as excinfo:
        asyncio.run(sess.execute("second", max_budget_usd=0.003))
    assert excinfo.value.kind == "usd"
    asyncio.run(sess.close())
