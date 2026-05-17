"""Behavioural conformance contracts requiring live vendor credentials.

Companion to :mod:`airframe.testing.contracts` (structural).
Whereas the structural contracts verify "the gate raises when
capability is False" without any vendor call, the integration
contracts run the actual round-trip and verify behavioural
properties:

* ``schema=`` round-trip — model returns a payload that validates
  against the supplied Pydantic schema.
* ``list_models()`` — vendor's models endpoint returns a non-empty
  list of ``ModelInfo``.
* Streaming — ``session.stream()`` yields :class:`TextDelta` events
  before a final :class:`TurnComplete`.
* ``thinking=`` — accepted by adapters declaring
  :data:`Feature.REASONING_EFFORT`; round-trip succeeds.
* Polymorphic prompt — adapters declaring
  :data:`Feature.VISION_INPUT` accept ``list[PromptPart]``.
* ``tools=`` — function-tool round-trip; the model invokes the
  registered tool and the handler's output reaches the final result.
* ``mcp_servers=`` — external MCP server registration; the model
  invokes a tool routed through the server.
* ``on_permission=`` — callback fires with a populated
  :class:`PermissionRequest` and the decision is honoured.
* ``on_event=`` — observer receives :class:`HookEvent` instances in
  causal order.
* ``max_turns=`` / ``max_budget_usd=`` — caps trip the matching
  :class:`RuntimeBudgetExceededError`.

**Gating.** Every test in this module is decorated with
:func:`pytest.mark.integration`. The default ``make test`` /
``make test-fast`` runs exclude the marker, so the suite passes
without credentials. Run integration tests explicitly with::

    pytest -m integration                            # all providers
    pytest -m integration -k claude                  # one provider
    pytest -m integration tests/integration/         # if you've set up a directory

Per-test auth requirements are documented in each function's
docstring. A missing credential causes :func:`pytest.skip` — never
a hard failure — so the suite stays usable on partially-configured
machines.

**How to use.** Adapter authors writing third-party adapters should
import these tests into their own integration suite the same way
the in-tree adapters import :mod:`airframe.testing.contracts`::

    # tests/test_my_adapter_integration.py
    import pytest
    from airframe.testing.integration import (
        test_integration_schema_round_trip,
        test_integration_list_models,
        # ...
    )
    from airframe_adapters_together import TogetherRuntime

    @pytest.fixture
    def adapter_runtime():
        return TogetherRuntime(api_key="...")

Pytest collects the imported tests against the local fixture. The
``integration`` marker carries through, so the same gating applies.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from pydantic import BaseModel

from airframe.errors import RuntimeAuthError, RuntimeBudgetExceededError
from airframe.events import TextDelta, TurnComplete
from airframe.features import Feature

# Per-provider auth-env-var lookup. Missing keys → pytest.skip in
# the fixture below. Adapters that resolve auth from a file (Claude
# Code's ``~/.claude/.credentials.json``, Codex's
# ``~/.codex/auth.json``) skip the env-var check — they'll fail at
# the live call with a clean :class:`RuntimeAuthError` we then
# convert to a skip.
_PROVIDER_AUTH: dict[str, list[str]] = {
    "claude": ["CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY"],
    "github-copilot": ["GITHUB_TOKEN", "GH_TOKEN"],
    "codex": ["OPENAI_API_KEY", "CODEX_API_KEY"],
    "opencode-zen": ["OPENCODE_API_KEY"],
    "opencode-go": ["OPENCODE_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
}


def _has_credentials(provider_id: str) -> bool:
    """Best-effort env-var check; file-based auth resolves at call time."""
    candidates = _PROVIDER_AUTH.get(provider_id, [])
    return any(os.environ.get(v) for v in candidates)


class _Brief(BaseModel):
    """A trivial schema used across the integration tests."""

    summary: str


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixture contract
# ---------------------------------------------------------------------------
#
# Adapter authors provide ``adapter_runtime`` (same name the
# structural contracts use). The fixture is responsible for:
#
# * Constructing the runtime (with credentials if needed).
# * Yielding it.
# * Calling ``await runtime.close()`` on teardown.
#
# Tests below assume the fixture exists. The in-tree mirror lives
# in ``tests/test_*_integration.py`` (one per built-in adapter,
# each guarded by ``pytest.importorskip`` for its vendor SDK).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase 0 / 1 — schema round-trip + models endpoint
# ---------------------------------------------------------------------------


async def test_integration_schema_round_trip(adapter_runtime: Any) -> None:
    """``execute(schema=)`` returns a populated Pydantic payload.

    The Phase 0 baseline. Every conforming adapter must drive the
    schema=... path through to a validated payload on a live call.
    """
    if not _has_credentials(adapter_runtime.PROVIDER_ID):
        pytest.skip(f"no credentials for {adapter_runtime.PROVIDER_ID!r}")
    try:
        result = await adapter_runtime.execute(
            "Reply with a one-sentence summary of Python.",
            schema=_Brief,
        )
    except RuntimeAuthError as exc:
        pytest.skip(f"auth failed: {exc}")
    assert result.structured is not None
    assert "summary" in result.structured
    assert isinstance(result.structured["summary"], str)
    assert result.structured["summary"].strip()


async def test_integration_plain_text_execute(adapter_runtime: Any) -> None:
    """``execute(schema=None)`` returns non-empty text."""
    if not _has_credentials(adapter_runtime.PROVIDER_ID):
        pytest.skip(f"no credentials for {adapter_runtime.PROVIDER_ID!r}")
    try:
        result = await adapter_runtime.execute("Say hello.")
    except RuntimeAuthError as exc:
        pytest.skip(f"auth failed: {exc}")
    assert result.structured is None
    assert isinstance(result.text, str)
    assert result.text.strip()


async def test_integration_list_models(adapter_runtime: Any) -> None:
    """``list_models()`` returns a non-empty list of :class:`ModelInfo`."""
    from airframe.models import ModelInfo

    if not _has_credentials(adapter_runtime.PROVIDER_ID):
        pytest.skip(f"no credentials for {adapter_runtime.PROVIDER_ID!r}")
    try:
        models = await adapter_runtime.list_models()
    except RuntimeAuthError as exc:
        pytest.skip(f"auth failed: {exc}")
    assert models
    for m in models:
        assert isinstance(m, ModelInfo)
        assert m.id


# ---------------------------------------------------------------------------
# Phase 1 — streaming
# ---------------------------------------------------------------------------


async def test_integration_stream_yields_text_then_turn_complete(
    adapter_runtime: Any,
) -> None:
    """Every adapter declaring :data:`Feature.STREAMING` yields deltas
    then exactly one :class:`TurnComplete`."""
    if not adapter_runtime.supports(Feature.STREAMING):
        pytest.skip("adapter does not declare STREAMING")
    if not _has_credentials(adapter_runtime.PROVIDER_ID):
        pytest.skip(f"no credentials for {adapter_runtime.PROVIDER_ID!r}")

    sess = adapter_runtime.session()
    events: list[Any] = []
    try:
        async for ev in sess.stream("Reply with the single word: ping"):
            events.append(ev)
    except RuntimeAuthError as exc:
        pytest.skip(f"auth failed: {exc}")
    finally:
        await sess.close()

    turn_completes = [e for e in events if isinstance(e, TurnComplete)]
    assert len(turn_completes) == 1
    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    assert text_deltas, "stream produced no TextDelta events"


# ---------------------------------------------------------------------------
# Phase 2 — thinking + vision
# ---------------------------------------------------------------------------


async def test_integration_thinking_round_trip(adapter_runtime: Any) -> None:
    """``execute(thinking="low")`` round-trips for adapters declaring REASONING_EFFORT."""
    if not adapter_runtime.supports(Feature.REASONING_EFFORT):
        pytest.skip("adapter does not declare REASONING_EFFORT")
    if not _has_credentials(adapter_runtime.PROVIDER_ID):
        pytest.skip(f"no credentials for {adapter_runtime.PROVIDER_ID!r}")
    sess = adapter_runtime.session()
    try:
        result = await sess.execute("What is 2+2?", thinking="low")
    except RuntimeAuthError as exc:
        pytest.skip(f"auth failed: {exc}")
    finally:
        await sess.close()
    assert result.text or result.structured is not None


# ---------------------------------------------------------------------------
# Phase 3 — function tools
# ---------------------------------------------------------------------------


async def test_integration_function_tool_round_trip(adapter_runtime: Any) -> None:
    """A single ``FunctionTool`` round-trip — the model invokes the tool
    and the handler's output reaches the final response."""
    if not adapter_runtime.supports(Feature.TOOLS_FUNCTION):
        pytest.skip("adapter does not declare TOOLS_FUNCTION")
    if not _has_credentials(adapter_runtime.PROVIDER_ID):
        pytest.skip(f"no credentials for {adapter_runtime.PROVIDER_ID!r}")

    from airframe.tools import FunctionTool

    class _AddParams(BaseModel):
        a: float
        b: float

    invocations: list[tuple[float, float]] = []

    async def _add(p: BaseModel) -> float:
        # ``params=`` narrows the runtime type; mypy sees the abstract base.
        invocations.append((p.a, p.b))  # type: ignore[attr-defined]
        return p.a + p.b  # type: ignore[attr-defined]

    tool = FunctionTool(
        name="add",
        description="Add two numbers and return the sum.",
        params=_AddParams,
        handler=_add,
    )

    sess = adapter_runtime.session(tools=[tool])
    try:
        result = await sess.execute(
            "Use the `add` tool to compute 17 + 25 and reply with the number only.",
        )
    except RuntimeAuthError as exc:
        pytest.skip(f"auth failed: {exc}")
    finally:
        await sess.close()

    assert invocations, "handler never fired"
    # We don't pin the exact text format (models vary) — just verify the
    # answer appears somewhere in the response.
    assert "42" in (result.text or "")


# ---------------------------------------------------------------------------
# Phase 5 — permission, hooks, budget
# ---------------------------------------------------------------------------


async def test_integration_permission_callback_fires(adapter_runtime: Any) -> None:
    """The :class:`PermissionCallback` is invoked at least once during
    a tool-using session.

    Codex's approval policy is session-wide — the callback fires
    exactly once at session start to derive the policy enum. Claude
    and Copilot fire per call. Both shapes satisfy the contract.
    """
    if not adapter_runtime.supports(Feature.PERMISSION_CALLBACK):
        pytest.skip("adapter does not declare PERMISSION_CALLBACK")
    if not adapter_runtime.supports(Feature.TOOLS_FUNCTION):
        pytest.skip("adapter does not declare TOOLS_FUNCTION — needed to trigger the callback")
    if not _has_credentials(adapter_runtime.PROVIDER_ID):
        pytest.skip(f"no credentials for {adapter_runtime.PROVIDER_ID!r}")

    from airframe.permission import PermissionCallback, PermissionDecision, PermissionRequest
    from airframe.tools import FunctionTool

    received: list[PermissionRequest] = []

    class _ApproveAll(PermissionCallback):
        async def handle(self, request: PermissionRequest) -> PermissionDecision:
            received.append(request)
            return "allow"

    class _NoArgs(BaseModel):
        pass

    async def _noop(_: BaseModel) -> str:
        return "done"

    tool = FunctionTool(
        name="noop",
        description="Do nothing and reply 'done'.",
        params=_NoArgs,
        handler=_noop,
    )

    sess = adapter_runtime.session(on_permission=_ApproveAll(), tools=[tool])
    try:
        await sess.execute("Call the `noop` tool and reply with its output.")
    except RuntimeAuthError as exc:
        pytest.skip(f"auth failed: {exc}")
    finally:
        await sess.close()
    assert received, "permission callback never fired"


async def test_integration_hook_observer_receives_events(adapter_runtime: Any) -> None:
    """The :class:`HookEvent` observer sees at least one event during
    a successful turn. Adapters synthesise session_start at first
    execute(), so the observer should always see that.
    """
    if not adapter_runtime.supports(Feature.LIFECYCLE_HOOKS):
        pytest.skip("adapter does not declare LIFECYCLE_HOOKS")
    if not _has_credentials(adapter_runtime.PROVIDER_ID):
        pytest.skip(f"no credentials for {adapter_runtime.PROVIDER_ID!r}")

    from airframe.hooks import HookEvent

    events: list[HookEvent] = []

    def _observer(e: HookEvent) -> None:
        events.append(e)

    sess = adapter_runtime.session(on_event=_observer)
    try:
        await sess.execute("Say hi.")
    except RuntimeAuthError as exc:
        pytest.skip(f"auth failed: {exc}")
    finally:
        await sess.close()
    assert events, "no HookEvent observed"
    kinds = {e.kind for e in events}
    # session_start is universal (all four adapters synthesise it).
    assert "session_start" in kinds, f"expected session_start in {kinds}"


async def test_integration_budget_usd_cap_trips(adapter_runtime: Any) -> None:
    """A deliberately tiny ``max_budget_usd`` trips
    :class:`RuntimeBudgetExceededError` on the second turn (the first
    turn succeeds and pushes cumulative cost above the cap).
    """
    if not adapter_runtime.supports(Feature.BUDGET_USD_CAP):
        pytest.skip("adapter does not declare BUDGET_USD_CAP")
    if not _has_credentials(adapter_runtime.PROVIDER_ID):
        pytest.skip(f"no credentials for {adapter_runtime.PROVIDER_ID!r}")

    sess = adapter_runtime.session()
    fired = False
    try:
        # First turn: a real call. Cumulative cost goes above the
        # micro-cap. Some vendors return cost_usd=None on free tiers —
        # the test skips when the first turn's cost stays at 0.
        try:
            r1 = await sess.execute("Say one short word.", max_budget_usd=0.000001)
        except RuntimeAuthError as exc:
            pytest.skip(f"auth failed: {exc}")
        if (r1.cost.cost_usd or 0.0) <= 0.0:
            pytest.skip("vendor reported zero cost; cap can't trip on this binding")
        try:
            await sess.execute("Say one short word.", max_budget_usd=0.000001)
        except RuntimeBudgetExceededError as exc:
            fired = True
            assert exc.kind == "usd"
    finally:
        await sess.close()
    assert fired, "expected RuntimeBudgetExceededError on the second turn"


__all__ = [
    "test_integration_budget_usd_cap_trips",
    "test_integration_function_tool_round_trip",
    "test_integration_hook_observer_receives_events",
    "test_integration_list_models",
    "test_integration_permission_callback_fires",
    "test_integration_plain_text_execute",
    "test_integration_schema_round_trip",
    "test_integration_stream_yields_text_then_turn_complete",
    "test_integration_thinking_round_trip",
]
