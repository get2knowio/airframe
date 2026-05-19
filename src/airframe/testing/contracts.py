"""Conformance contracts every :class:`AgentRuntime` adapter must satisfy.

Each ``test_*`` function in this module is a pytest test that takes
an ``adapter_runtime`` fixture and asserts a property of the airframe
protocol. Third-party adapter authors import the test functions into
their own test module and provide the fixture; pytest collects the
imported tests into that module and runs them against the local
fixture.

This module is the *unit* contract surface: every test here runs in
default CI without network access or vendor credentials. Behavioural
contracts that require live vendor calls (``schema=`` round-trip,
``RuntimeAuthError`` classification on a real 401, populated token
counts on a real response) belong in
:mod:`airframe.testing.integration` (Phase 1 work).

The contracts intentionally never instantiate ``adapter_runtime``
themselves — that's the fixture's job. They also never close it
(closing on a never-used runtime is one of the contracts, but the
fixture should clean up too if construction is expensive).
"""

from __future__ import annotations

from typing import Any

import pytest

from airframe.events import TurnComplete
from airframe.features import Feature
from airframe.protocol import AgentSession, ProviderModel

# ---------------------------------------------------------------------------
# Lifecycle contracts
# ---------------------------------------------------------------------------


async def test_close_is_idempotent(adapter_runtime: Any) -> None:
    """``close()`` may be called multiple times without raising.

    Documented in :doc:`architecture` as a hard rule: ``close()`` runs
    from ``finally`` blocks and ``__aexit__`` paths where shadowing
    the original exception would be catastrophic. Calling it twice
    must therefore be a no-op the second time, not a double-teardown
    error.
    """
    await adapter_runtime.close()
    await adapter_runtime.close()
    await adapter_runtime.close()


async def test_close_on_fresh_runtime(adapter_runtime: Any) -> None:
    """``close()`` on a runtime that's never been used must not raise.

    Caller code commonly wraps construction in a try/finally where
    the body fails before the first ``execute()`` — the ``finally``
    still calls ``close()``. That path must work even though no
    subprocess / HTTP client was ever built.
    """
    await adapter_runtime.close()


# ---------------------------------------------------------------------------
# unwrap() escape-hatch contracts
# ---------------------------------------------------------------------------


def test_unwrap_returns_self(adapter_runtime: Any) -> None:
    """``unwrap(type(self))`` returns ``self``.

    The trivial-but-mandatory base case for JDBC-style ``Wrapper``.
    Keeps the contract consistent across runtimes — a consumer that
    has an ``AgentRuntime`` reference and wants the concrete type
    never has to special-case "this adapter doesn't implement
    unwrap."
    """
    assert adapter_runtime.unwrap(type(adapter_runtime)) is adapter_runtime


def test_unwrap_unrelated_type_raises_typeerror(adapter_runtime: Any) -> None:
    """``unwrap(<unrelated>)`` raises :class:`TypeError`.

    Modelled on JDBC ``Wrapper.unwrap`` raising ``SQLException`` for
    unsupported casts. The behaviour matters: a silently-None result
    would force consumer code to defend against ``unwrap`` returning
    junk, which defeats the type signature.
    """

    class _Unrelated:
        """Stand-in type no adapter should ever unwrap to."""

    with pytest.raises(TypeError):
        adapter_runtime.unwrap(_Unrelated)


# ---------------------------------------------------------------------------
# supports() purity contracts
# ---------------------------------------------------------------------------


def test_supports_returns_bool_for_every_feature(adapter_runtime: Any) -> None:
    """Every :class:`Feature` member maps to a ``bool`` answer.

    Catches an adapter that mistypes ``supports()`` (e.g. returns the
    enum member, or ``None``, or a truthy string). The contract is
    strict: not "truthy", actually ``bool``.
    """
    for feature in Feature:
        result = adapter_runtime.supports(feature)
        assert isinstance(result, bool), (
            f"supports({feature.name}) returned {result!r} ({type(result).__name__}); "
            f"must be a bool"
        )


def test_supports_is_idempotent(adapter_runtime: Any) -> None:
    """Repeated ``supports()`` calls yield the same answer.

    Guards against an accidental "lazy SDK version probe on first
    call" implementation. Consumers will call ``supports()`` in tight
    branching paths; it has to be cheap and stable.
    """
    for feature in Feature:
        first = adapter_runtime.supports(feature)
        second = adapter_runtime.supports(feature)
        third = adapter_runtime.supports(feature)
        assert first == second == third, (
            f"supports({feature.name}) returned different values across calls: "
            f"{first}, {second}, {third}"
        )


def test_supports_structured_output_json_schema_is_true(adapter_runtime: Any) -> None:
    """The one universally-implemented feature at v0.3.0.

    Every airframe adapter must implement ``execute(schema=...)`` —
    that's the v0 / v0.1 / v0.2 / v0.3 contract. The
    ``STRUCTURED_OUTPUT_JSON_SCHEMA`` flag advertises this baseline,
    and every conforming adapter must declare it.

    Phase 1+ may flip additional bits on (STREAMING, SESSION_RESUME,
    etc.) but this one is the floor.
    """
    assert adapter_runtime.supports(Feature.STRUCTURED_OUTPUT_JSON_SCHEMA), (
        f"{type(adapter_runtime).__name__} must declare "
        f"Feature.STRUCTURED_OUTPUT_JSON_SCHEMA — every airframe adapter "
        f"implements execute(schema=...)"
    )


def test_supports_accepts_model_kwarg(adapter_runtime: Any) -> None:
    """The ``model=`` slot is honoured even when unused.

    Per-model differentiation isn't a Phase 0 feature but the slot
    exists in the protocol signature; calling adapters must accept
    the kwarg without raising.
    """
    # Use the adapter's own provider id so we don't trip
    # provider-mismatch logic the binding may eventually carry.
    binding = ProviderModel(adapter_runtime.PROVIDER_ID, "some-model")
    runtime_wide = adapter_runtime.supports(Feature.STRUCTURED_OUTPUT_JSON_SCHEMA)
    with_model = adapter_runtime.supports(Feature.STRUCTURED_OUTPUT_JSON_SCHEMA, model=binding)
    # Both call forms succeed; their answer agrees for the universal
    # feature today. Per-model gating in later phases may diverge them
    # for other features, but the universal one stays consistent.
    assert runtime_wide == with_model


# ---------------------------------------------------------------------------
# Plain-text execute() contract
# ---------------------------------------------------------------------------


def test_plain_text_execute_path_is_wired(adapter_runtime: Any) -> None:
    """The ``schema=None`` path must be wired per the protocol docstring.

    The :meth:`AgentRuntime.execute` docstring promises::

        schema: None means plain text — text answer on RuntimeResult.text,
                structured=None.

    This Phase 0 contract verifies the path is *implemented*, not its
    *behaviour*. The structural checks:

    1. The ``execute()`` signature accepts ``schema=None`` as the
       default — protocol-required.
    2. The implementation does not carry the legacy
       ``"plain-text execute() is not wired in v0"``
       :class:`NotImplementedError` gate. (Three of the four built-in
       adapters historically refused ``schema=None`` with that exact
       message; v0.3.0 wired them.)

    A behavioural contract that validates the *returned* text shape
    requires live vendor credentials and lives in
    :mod:`airframe.testing.integration` (deferred to v0.4.0 along
    with the other network-required contracts called out in the
    v0.3.0 release notes).

    Adapter authors implementing plain-text differently from the
    historical refusal pattern still satisfy this contract — the
    source check is specifically targeting the legacy gate string.
    """
    import inspect

    sig = inspect.signature(type(adapter_runtime).execute)
    schema_param = sig.parameters.get("schema")
    assert schema_param is not None, (
        f"{type(adapter_runtime).__name__}.execute() must accept a `schema=` "
        f"keyword per the AgentRuntime protocol"
    )
    assert schema_param.default is None, (
        f"{type(adapter_runtime).__name__}.execute(schema=...) must default to None "
        f"per the protocol; got default {schema_param.default!r}"
    )

    source = inspect.getsource(type(adapter_runtime).execute)
    legacy_gate = "plain-text execute() is not wired"
    assert legacy_gate not in source, (
        f"{type(adapter_runtime).__name__}.execute() still carries the legacy "
        f"plain-text-not-wired NotImplementedError gate. The protocol docstring "
        f"promises plain-text support; drop the gate and wire the schema=None "
        f"path through the vendor SDK."
    )


# ---------------------------------------------------------------------------
# validate_binding() contracts
# ---------------------------------------------------------------------------


def test_validate_binding_returns_bool(adapter_runtime: Any) -> None:
    """``validate_binding`` is a pure predicate that doesn't raise.

    Documented as "cheap and non-async; suitable for filtering
    bindings before attempting them." Returning anything other than
    a ``bool``, or raising, breaks consumer code that filters lists
    of candidate bindings.
    """
    # Own provider id.
    own = ProviderModel(adapter_runtime.PROVIDER_ID, "any-model-id")
    own_result = adapter_runtime.validate_binding(own)
    assert isinstance(own_result, bool)

    # Foreign provider id — must be False, not raise.
    foreign = ProviderModel("definitely-not-a-real-provider", "model")
    foreign_result = adapter_runtime.validate_binding(foreign)
    assert foreign_result is False, (
        f"{type(adapter_runtime).__name__}.validate_binding rejected a foreign "
        f"provider with {foreign_result!r}; expected False"
    )


# ---------------------------------------------------------------------------
# session() factory contracts — Phase 1 (Iteration B)
# ---------------------------------------------------------------------------


def test_session_factory_returns_agent_session(adapter_runtime: Any) -> None:
    """``runtime.session()`` returns an object satisfying ``AgentSession``.

    Phase 1 Iteration B lands the factory on every adapter; the
    returned object exposes ``id``, ``execute``, ``stream``,
    ``cancel``, ``close``. The :class:`AgentSession` Protocol is
    ``runtime_checkable`` so :func:`isinstance` is the right gate.
    """
    sess = adapter_runtime.session()
    try:
        assert isinstance(sess, AgentSession), (
            f"{type(adapter_runtime).__name__}.session() returned "
            f"{type(sess).__name__}, which doesn't satisfy AgentSession"
        )
        # ``id`` is part of the protocol surface — accessible without raising.
        assert sess.id is None or isinstance(sess.id, str), (
            f"AgentSession.id must be `str | None`; got {type(sess.id).__name__}"
        )
    finally:
        # Eagerly close so a leaked subprocess / HTTP client doesn't
        # outlive the test.
        import asyncio

        asyncio.run(_safe_close(sess))


async def _safe_close(sess: Any) -> None:
    """Helper used by the synchronous session-factory contracts."""
    try:
        await sess.close()
    except Exception:  # noqa: BLE001 — contract test, never raise from teardown
        pass


async def test_session_close_is_idempotent(adapter_runtime: Any) -> None:
    """``session.close()`` is safe to call repeatedly.

    Same discipline as :meth:`AgentRuntime.close` — sessions are
    routinely torn down in ``finally`` blocks and async-context-manager
    ``__aexit__`` paths where shadowing the underlying exception would
    be catastrophic. The contract is checked structurally so adapter
    authors can't accidentally ship a single-shot ``close()``.
    """
    sess = adapter_runtime.session()
    await sess.close()
    await sess.close()
    await sess.close()


async def test_session_close_on_fresh_session_is_safe(adapter_runtime: Any) -> None:
    """Closing a never-used session must not raise.

    Caller code commonly opens a session inside a ``try`` that fails
    before the first :meth:`execute`; the ``finally`` still calls
    :meth:`close`. That path must work without a constructed vendor
    handle.
    """
    sess = adapter_runtime.session()
    await sess.close()


async def test_session_cancel_when_idle_is_noop(adapter_runtime: Any) -> None:
    """``session.cancel()`` when no turn is in flight is a no-op.

    The :class:`AgentSession` docstring is explicit: ``cancel()`` is
    cheap and idempotent; the unsupported-capability branch only fires
    *while a turn is running*. A fresh session has nothing in flight,
    so the call returns without raising — regardless of whether the
    adapter declares :data:`~airframe.features.Feature.CANCEL`.
    """
    sess = adapter_runtime.session()
    try:
        await sess.cancel()
        await sess.cancel()
    finally:
        await sess.close()


def test_session_stream_is_async_generator(adapter_runtime: Any) -> None:
    """``session.stream()`` is an async-generator method.

    Structural check that the implementation uses ``yield`` (i.e.
    returns an :class:`~collections.abc.AsyncGenerator` when called),
    not a coroutine that *returns* an iterator. Catches the easy
    typo of ``async def stream(...) -> ...: return iter(...)`` which
    typechecks but breaks the ``async for event in session.stream():``
    pattern.
    """
    import inspect

    method = type(adapter_runtime.session()).stream
    assert inspect.isasyncgenfunction(method), (
        f"{type(adapter_runtime).__name__}.session().stream must be an "
        f"async generator (async def + yield); got "
        f"{'coroutine' if inspect.iscoroutinefunction(method) else 'function'}"
    )


def test_session_resume_not_implemented_until_feature_flips(adapter_runtime: Any) -> None:
    """Adapters not declaring SESSION_RESUME must refuse ``session(resume=...)``.

    The implementation plan's "no silent fallbacks" principle: a
    capability declined must raise, never quietly drop the request.
    Two acceptable shapes:

    * :class:`NotImplementedError` — Iteration B's
      :func:`~airframe.sessions._open_thin_session` raises this; signals
      "the API exists in the protocol but this adapter hasn't wired it
      yet" (will land in a later iteration).
    * :class:`~airframe.errors.UnsupportedFeatureError` — Iteration C+
      bespoke sessions raise this; signals "this capability will
      *never* land on this adapter" (e.g., chat-completions vendors
      can't resume server-side).

    Adapters declaring :data:`~airframe.features.Feature.SESSION_RESUME`
    opt out of this contract — the resume call should succeed.
    """
    if adapter_runtime.supports(Feature.SESSION_RESUME):
        return  # Adapter wired resume; structural contract doesn't apply.

    import pytest as _pytest

    from airframe.errors import UnsupportedFeatureError

    with _pytest.raises((NotImplementedError, UnsupportedFeatureError, ValueError, TypeError)):
        adapter_runtime.session(resume="any-string")


# ---------------------------------------------------------------------------
# Phase 2 contracts — inputs & reasoning
# ---------------------------------------------------------------------------


def test_session_execute_signature_accepts_thinking_kwarg(adapter_runtime: Any) -> None:
    """:meth:`AgentSession.execute` accepts the ``thinking=`` kwarg.

    Structural check: the kwarg must exist on every adapter's
    :meth:`execute` signature regardless of whether REASONING_EFFORT
    is declared (the protocol requires the signature; the gate
    inside the method enforces capability).
    """
    import inspect

    sess = adapter_runtime.session()
    try:
        sig = inspect.signature(type(sess).execute)
        assert "thinking" in sig.parameters, (
            f"{type(sess).__name__}.execute must accept thinking= per the AgentSession protocol"
        )
    finally:
        import asyncio

        asyncio.run(_safe_close(sess))


async def test_session_thinking_kwarg_declined_when_capability_false(
    adapter_runtime: Any,
) -> None:
    """Adapters declining REASONING_EFFORT raise on ``thinking=`` use.

    Positive case (capability declared) is exercised behaviourally
    in :mod:`airframe.testing.integration` — it requires live
    credentials.
    """
    if adapter_runtime.supports(Feature.REASONING_EFFORT):
        return  # behavioural; live-credentials suite
    from airframe.errors import UnsupportedFeatureError

    sess = adapter_runtime.session()
    try:
        with pytest.raises(UnsupportedFeatureError):
            await sess.execute("hi", thinking="medium")
    finally:
        await sess.close()


async def test_session_polymorphic_prompt_declined_when_vision_false(
    adapter_runtime: Any,
) -> None:
    """Adapters declining VISION_INPUT raise on a list-shaped prompt.

    Positive case (vision-supporting adapters round-tripping an
    :class:`ImageInput`) is behavioural — lives in the integration
    suite where credentials and a real test image are available.
    """
    if adapter_runtime.supports(Feature.VISION_INPUT):
        return
    from airframe.errors import UnsupportedFeatureError
    from airframe.inputs import ImageInput

    sess = adapter_runtime.session()
    try:
        with pytest.raises(UnsupportedFeatureError):
            await sess.execute(
                ["caption:", ImageInput(path="/tmp/__nope__.png")],
            )
    finally:
        await sess.close()


# ---------------------------------------------------------------------------
# Phase 3 contracts — function tools
# ---------------------------------------------------------------------------


def test_session_tools_kwarg_agrees_with_tools_function_capability(
    adapter_runtime: Any,
) -> None:
    """``session(tools=[FunctionTool])`` agrees with :data:`Feature.TOOLS_FUNCTION`.

    Adapters declaring TOOLS_FUNCTION accept the kwarg at session
    construction without raising. Adapters declining must raise
    :class:`UnsupportedFeatureError` with ``feature=TOOLS_FUNCTION``.
    """
    from pydantic import BaseModel

    from airframe.errors import UnsupportedFeatureError
    from airframe.tools import FunctionTool

    class _Params(BaseModel):
        x: int

    async def _handler(p: BaseModel) -> int:
        return p.x  # type: ignore[attr-defined]  # narrowed by params= at runtime

    tool = FunctionTool(name="probe_t", description="t", params=_Params, handler=_handler)

    if adapter_runtime.supports(Feature.TOOLS_FUNCTION):
        sess = adapter_runtime.session(tools=[tool])
        try:
            assert isinstance(sess, AgentSession)
        finally:
            import asyncio

            asyncio.run(_safe_close(sess))
    else:
        with pytest.raises(UnsupportedFeatureError) as exc:
            adapter_runtime.session(tools=[tool])
        assert exc.value.feature == Feature.TOOLS_FUNCTION, (
            f"{type(adapter_runtime).__name__}: tools= decline must carry "
            f"feature=TOOLS_FUNCTION; got {exc.value.feature!r}"
        )


# ---------------------------------------------------------------------------
# Phase 4 contracts — MCP server refs
# ---------------------------------------------------------------------------


def test_session_mcp_servers_kwarg_agrees_with_transport_capabilities(
    adapter_runtime: Any,
) -> None:
    """``session(mcp_servers=[McpServerRef(transport=X)])`` agrees with
    the matching :data:`Feature.TOOLS_MCP_*` flag.

    For each transport the adapter declares, a ref of that transport
    must construct cleanly. For each transport the adapter declines,
    the same ref must raise :class:`UnsupportedFeatureError` with
    the matching feature attribute.
    """
    from airframe.errors import UnsupportedFeatureError
    from airframe.tools import McpServerRef

    matrix = [
        (
            Feature.TOOLS_MCP_STDIO,
            McpServerRef(name="probe_stdio", transport="stdio", command=["echo"]),
        ),
        (
            Feature.TOOLS_MCP_HTTP,
            McpServerRef(name="probe_http", transport="http", url="https://example.com"),
        ),
        (
            Feature.TOOLS_MCP_SSE,
            McpServerRef(name="probe_sse", transport="sse", url="https://example.com"),
        ),
    ]

    for feature, ref in matrix:
        if adapter_runtime.supports(feature):
            sess = adapter_runtime.session(mcp_servers=[ref])
            try:
                assert isinstance(sess, AgentSession)
            finally:
                import asyncio

                asyncio.run(_safe_close(sess))
        else:
            with pytest.raises(UnsupportedFeatureError) as exc:
                adapter_runtime.session(mcp_servers=[ref])
            assert exc.value.feature == feature, (
                f"{type(adapter_runtime).__name__}: mcp_servers=[{ref.transport}] "
                f"decline must carry feature={feature.name}; got {exc.value.feature!r}"
            )


# ---------------------------------------------------------------------------
# Phase 5 contracts — permission, hooks, budget
# ---------------------------------------------------------------------------


def test_session_on_permission_agrees_with_permission_callback_capability(
    adapter_runtime: Any,
) -> None:
    """``session(on_permission=...)`` agrees with :data:`Feature.PERMISSION_CALLBACK`.

    Adapters declaring the flag accept the callback without raising.
    Adapters declining raise :class:`UnsupportedFeatureError` with
    ``feature=PERMISSION_CALLBACK``.
    """
    from airframe.errors import UnsupportedFeatureError
    from airframe.permission import PermissionCallback, PermissionDecision, PermissionRequest

    class _Cb(PermissionCallback):
        async def handle(self, request: PermissionRequest) -> PermissionDecision:
            return "allow"

    cb = _Cb()
    if adapter_runtime.supports(Feature.PERMISSION_CALLBACK):
        sess = adapter_runtime.session(on_permission=cb)
        try:
            assert isinstance(sess, AgentSession)
        finally:
            import asyncio

            asyncio.run(_safe_close(sess))
    else:
        with pytest.raises(UnsupportedFeatureError) as exc:
            adapter_runtime.session(on_permission=cb)
        assert exc.value.feature == Feature.PERMISSION_CALLBACK, (
            f"{type(adapter_runtime).__name__}: on_permission decline must "
            f"carry feature=PERMISSION_CALLBACK; got {exc.value.feature!r}"
        )


def test_session_on_event_agrees_with_lifecycle_hooks_capability(
    adapter_runtime: Any,
) -> None:
    """``session(on_event=...)`` agrees with :data:`Feature.LIFECYCLE_HOOKS`.

    Adapters declaring the flag accept the observer without raising.
    Adapters declining raise :class:`UnsupportedFeatureError` with
    ``feature=LIFECYCLE_HOOKS``.
    """
    from airframe.errors import UnsupportedFeatureError
    from airframe.hooks import HookEvent

    def _observer(_e: HookEvent) -> None:
        pass

    if adapter_runtime.supports(Feature.LIFECYCLE_HOOKS):
        sess = adapter_runtime.session(on_event=_observer)
        try:
            assert isinstance(sess, AgentSession)
        finally:
            import asyncio

            asyncio.run(_safe_close(sess))
    else:
        with pytest.raises(UnsupportedFeatureError) as exc:
            adapter_runtime.session(on_event=_observer)
        assert exc.value.feature == Feature.LIFECYCLE_HOOKS, (
            f"{type(adapter_runtime).__name__}: on_event decline must "
            f"carry feature=LIFECYCLE_HOOKS; got {exc.value.feature!r}"
        )


def test_emittable_hook_kinds_subset_of_eight_literals(adapter_runtime: Any) -> None:
    """Adapters declaring LIFECYCLE_HOOKS expose ``EMITTABLE_HOOK_KINDS``
    that is a subset of the eight canonical :data:`HookEventKind` literals.

    Pins the shape lock: an adapter cannot invent a new ``kind``
    string. Adapters declining LIFECYCLE_HOOKS may omit the ClassVar
    entirely.
    """
    if not adapter_runtime.supports(Feature.LIFECYCLE_HOOKS):
        return
    canonical = {
        "session_start",
        "session_end",
        "user_prompt_submit",
        "pre_tool_use",
        "post_tool_use",
        "tool_failure",
        "pre_compact",
        "rate_limit",
    }
    kinds = getattr(type(adapter_runtime), "EMITTABLE_HOOK_KINDS", None)
    assert kinds is not None, (
        f"{type(adapter_runtime).__name__} declares LIFECYCLE_HOOKS but has "
        f"no EMITTABLE_HOOK_KINDS ClassVar"
    )
    assert set(kinds) <= canonical, (
        f"{type(adapter_runtime).__name__}.EMITTABLE_HOOK_KINDS contains "
        f"non-canonical kinds: {set(kinds) - canonical}"
    )


def test_session_execute_signature_accepts_budget_kwargs(adapter_runtime: Any) -> None:
    """:meth:`AgentSession.execute` accepts ``max_turns=`` and ``max_budget_usd=``.

    Structural check — both kwargs are protocol surface regardless
    of whether the capability is declared.
    """
    import inspect

    sess = adapter_runtime.session()
    try:
        sig = inspect.signature(type(sess).execute)
        assert "max_turns" in sig.parameters
        assert "max_budget_usd" in sig.parameters
    finally:
        import asyncio

        asyncio.run(_safe_close(sess))


async def test_session_max_turns_declined_when_budget_turn_cap_false(
    adapter_runtime: Any,
) -> None:
    """Adapters declining BUDGET_TURN_CAP raise on ``max_turns=`` use.

    Positive case (enforcement at the turn boundary) is exercised
    in :mod:`airframe.testing.integration`.
    """
    if adapter_runtime.supports(Feature.BUDGET_TURN_CAP):
        return
    from airframe.errors import UnsupportedFeatureError

    sess = adapter_runtime.session()
    try:
        with pytest.raises(UnsupportedFeatureError) as exc:
            await sess.execute("hi", max_turns=5)
        assert exc.value.feature == Feature.BUDGET_TURN_CAP, (
            f"{type(adapter_runtime).__name__}: max_turns decline must "
            f"carry feature=BUDGET_TURN_CAP; got {exc.value.feature!r}"
        )
    finally:
        await sess.close()


async def test_session_max_budget_usd_declined_when_budget_usd_cap_false(
    adapter_runtime: Any,
) -> None:
    """Adapters declining BUDGET_USD_CAP raise on ``max_budget_usd=`` use."""
    if adapter_runtime.supports(Feature.BUDGET_USD_CAP):
        return
    from airframe.errors import UnsupportedFeatureError

    sess = adapter_runtime.session()
    try:
        with pytest.raises(UnsupportedFeatureError) as exc:
            await sess.execute("hi", max_budget_usd=0.05)
        assert exc.value.feature == Feature.BUDGET_USD_CAP, (
            f"{type(adapter_runtime).__name__}: max_budget_usd decline must "
            f"carry feature=BUDGET_USD_CAP; got {exc.value.feature!r}"
        )
    finally:
        await sess.close()


# ---------------------------------------------------------------------------
# ProviderOptions cross-namespace rejection
# ---------------------------------------------------------------------------


def test_session_rejects_wrong_provider_options_namespace(adapter_runtime: Any) -> None:
    """Passing the wrong :class:`ProviderOptions` namespace raises.

    Each adapter's ``session(provider_options=)`` must reject options
    of a different vendor's namespace with
    :class:`UnsupportedFeatureError`. The contract uses the
    by-elimination principle: pick a namespace that *isn't* this
    adapter's matching one and verify it raises.
    """
    from airframe.errors import UnsupportedFeatureError
    from airframe.options import (
        BedrockOptions,
        ClaudeOptions,
        CopilotOptions,
        KimiOptions,
        OpenAICompatOptions,
    )

    matching = {
        "claude": ClaudeOptions,
        "github-copilot": CopilotOptions,
        "opencode-zen": OpenAICompatOptions,
        "opencode-go": OpenAICompatOptions,
        "openrouter": OpenAICompatOptions,
        "bedrock": BedrockOptions,
        "kimi": KimiOptions,
    }
    own = matching.get(adapter_runtime.PROVIDER_ID)
    # Pick any namespace that isn't ours.
    all_namespaces = (
        ClaudeOptions,
        CopilotOptions,
        OpenAICompatOptions,
        BedrockOptions,
        KimiOptions,
    )
    others = [c for c in all_namespaces if c is not own]
    assert others, "test fixture must have at least one foreign namespace"
    foreign = others[0]
    with pytest.raises(UnsupportedFeatureError):
        adapter_runtime.session(provider_options=foreign())


# ``TurnComplete`` is exported alongside the contracts so integration-test
# fixtures (Phase 1 Iteration B+) can build the trailing event without
# re-importing it from airframe.events. Re-exporting here keeps the
# contracts module self-contained for adapter authors.
__all__ = [
    "TurnComplete",
    "test_close_is_idempotent",
    "test_close_on_fresh_runtime",
    "test_emittable_hook_kinds_subset_of_eight_literals",
    "test_plain_text_execute_path_is_wired",
    "test_session_cancel_when_idle_is_noop",
    "test_session_close_is_idempotent",
    "test_session_close_on_fresh_session_is_safe",
    "test_session_execute_signature_accepts_budget_kwargs",
    "test_session_execute_signature_accepts_thinking_kwarg",
    "test_session_factory_returns_agent_session",
    "test_session_max_budget_usd_declined_when_budget_usd_cap_false",
    "test_session_max_turns_declined_when_budget_turn_cap_false",
    "test_session_mcp_servers_kwarg_agrees_with_transport_capabilities",
    "test_session_on_event_agrees_with_lifecycle_hooks_capability",
    "test_session_on_permission_agrees_with_permission_callback_capability",
    "test_session_polymorphic_prompt_declined_when_vision_false",
    "test_session_rejects_wrong_provider_options_namespace",
    "test_session_resume_not_implemented_until_feature_flips",
    "test_session_stream_is_async_generator",
    "test_session_thinking_kwarg_declined_when_capability_false",
    "test_session_tools_kwarg_agrees_with_tools_function_capability",
    "test_supports_accepts_model_kwarg",
    "test_supports_is_idempotent",
    "test_supports_returns_bool_for_every_feature",
    "test_supports_structured_output_json_schema_is_true",
    "test_unwrap_returns_self",
    "test_unwrap_unrelated_type_raises_typeerror",
    "test_validate_binding_returns_bool",
]
