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

    def __init__(self, request_id: str = "req-1") -> None:
        self.id = request_id
        self.resolved_with: str | None = None
        self.resolved = False

    def resolve(self, decision: str) -> None:
        self.resolved_with = decision
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
    # Iteration B doesn't ship a pricing table — Iteration E lands it.
    assert result.cost.cost_usd is None
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
