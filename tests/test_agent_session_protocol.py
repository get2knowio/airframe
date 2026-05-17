"""Phase 1 protocol-surface test for :class:`AgentSession`.

Iteration A of Phase 1 lands the ``AgentSession`` Protocol declaration
without yet wiring per-adapter implementations or the
``AgentRuntime.session()`` factory. This file pins the method set on
the Protocol so the public surface can't drift before adapters land.

Behavioural tests (streaming yields ≥1 event ending in TurnComplete,
session-resume round-trip, cancellation within 100 ms) live in
:mod:`airframe.testing.contracts` once Iteration B wires the adapters.
"""

from __future__ import annotations

import inspect

from airframe import AgentSession


def test_agent_session_protocol_methods() -> None:
    """``AgentSession`` declares exactly the locked Phase 1 method set.

    Plus the ``id`` attribute. ADR-004 picks single-active-session per
    runtime, so no ``copy()`` / ``fork()`` methods here today. Adding
    methods later is fine; renaming or removing is breaking.

    Iteration G added ``unwrap()`` since the execute()-as-sugar refactor
    moved session-level vendor types off the runtime onto the session
    (consumers reach `ClaudeSDKClient` via ``session.unwrap(...)``
    instead of `runtime.unwrap(...)`).
    """
    members = {
        name
        for name, value in inspect.getmembers(AgentSession)
        if not name.startswith("_")
        and (inspect.isfunction(value) or inspect.iscoroutinefunction(value))
    }
    assert members == {"execute", "stream", "cancel", "close", "unwrap"}


def test_agent_session_execute_signature() -> None:
    """``execute(prompt, *, schema=None, thinking=None, max_turns=None,
    max_budget_usd=None, timeout=600.0)``.

    Phase 2 Iteration A added ``thinking=``. Phase 5 Iteration A
    added ``max_turns=`` and ``max_budget_usd=`` between ``thinking=``
    and ``timeout=`` so the budget caps sit with the other policy
    knobs. Defaults are all None / 600.0 so consumers can drop a
    single positional ``prompt`` and get the pre-Phase-5 behaviour.
    """
    sig = inspect.signature(AgentSession.execute)
    assert list(sig.parameters) == [
        "self",
        "prompt",
        "schema",
        "thinking",
        "max_turns",
        "max_budget_usd",
        "timeout",
    ]
    assert sig.parameters["schema"].default is None
    assert sig.parameters["thinking"].default is None
    assert sig.parameters["max_turns"].default is None
    assert sig.parameters["max_budget_usd"].default is None
    assert sig.parameters["timeout"].default == 600.0


def test_agent_session_stream_signature() -> None:
    """``stream(prompt, *, schema=None, thinking=None, max_turns=None,
    max_budget_usd=None, timeout=600.0)``.

    Same kwarg shape as ``execute()`` — Phase 5 keeps them in sync.
    """
    sig = inspect.signature(AgentSession.stream)
    assert list(sig.parameters) == [
        "self",
        "prompt",
        "schema",
        "thinking",
        "max_turns",
        "max_budget_usd",
        "timeout",
    ]
    assert sig.parameters["schema"].default is None
    assert sig.parameters["thinking"].default is None
    assert sig.parameters["max_turns"].default is None
    assert sig.parameters["max_budget_usd"].default is None
    assert sig.parameters["timeout"].default == 600.0
