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

from airframe.features import Feature
from airframe.protocol import ProviderModel

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


__all__ = [
    "test_close_is_idempotent",
    "test_close_on_fresh_runtime",
    "test_supports_accepts_model_kwarg",
    "test_supports_is_idempotent",
    "test_supports_returns_bool_for_every_feature",
    "test_supports_structured_output_json_schema_is_true",
    "test_unwrap_returns_self",
    "test_unwrap_unrelated_type_raises_typeerror",
    "test_validate_binding_returns_bool",
]
