"""Unit tests for :class:`Feature` and :meth:`AgentRuntime.supports`.

Phase 0 ships the whole forward-looking enum; later phases flip
``True`` bits on adapter ``SUPPORTED_FEATURES`` sets as their APIs
land. Today only ``STRUCTURED_OUTPUT_JSON_SCHEMA`` is True across
every in-tree adapter because it's the one feature whose API
(``execute(schema=...)``) is already wired.
"""

from __future__ import annotations

import pytest

from airframe import (
    ClaudeCodeRuntime,
    CodexRuntime,
    CopilotRuntime,
    Feature,
    OpenCodeZenRuntime,
)
from airframe.protocol import ProviderModel


def test_feature_string_values_are_stable() -> None:
    """The string values are public surface — locked at v0.3.0.

    Consumer code branches on ``Feature.STREAMING == "streaming"``;
    renaming any of these is a major-version break. This test snapshots
    the current wire values so any later rename is caught at PR time.
    """
    assert Feature.STRUCTURED_OUTPUT_JSON_SCHEMA.value == "structured_output_json_schema"
    assert Feature.STRUCTURED_OUTPUT_STRICT.value == "structured_output_strict"
    assert Feature.STREAMING.value == "streaming"
    assert Feature.SESSION_RESUME.value == "session_resume"
    assert Feature.CANCEL.value == "cancel"
    assert Feature.REASONING_EFFORT.value == "reasoning_effort"
    assert Feature.REASONING_BUDGET_TOKENS.value == "reasoning_budget_tokens"
    assert Feature.VISION_INPUT.value == "vision_input"
    assert Feature.FILE_INPUT.value == "file_input"
    assert Feature.TOOLS_FUNCTION.value == "tools_function"
    assert Feature.TOOLS_MCP_STDIO.value == "tools_mcp_stdio"
    assert Feature.TOOLS_MCP_HTTP.value == "tools_mcp_http"
    assert Feature.TOOLS_MCP_IN_PROCESS.value == "tools_mcp_in_process"
    assert Feature.PERMISSION_CALLBACK.value == "permission_callback"
    assert Feature.LIFECYCLE_HOOKS.value == "lifecycle_hooks"
    assert Feature.BUDGET_USD_CAP.value == "budget_usd_cap"
    assert Feature.BUDGET_TURN_CAP.value == "budget_turn_cap"
    assert Feature.SANDBOX.value == "sandbox"
    assert Feature.SUBAGENTS.value == "subagents"


def test_feature_is_a_str_subclass() -> None:
    """``str`` mixin lets feature values serialise cleanly.

    Structured-log sinks and config files often want a plain string;
    ``Feature.STREAMING`` should equal ``"streaming"`` without an
    explicit cast.
    """
    assert Feature.STREAMING == "streaming"
    # And the inverse: passing the string back into the enum constructor
    # round-trips through the value.
    assert Feature("streaming") is Feature.STREAMING


@pytest.fixture
def adapters() -> list[object]:
    """All four in-tree adapters, instantiated without credentials.

    None of the supports() checks should hit the network — they're all
    pure lookups. Adapter construction defers SDK imports / auth
    until the first execute() call, so this fixture is safe.
    """
    return [
        ClaudeCodeRuntime(),
        CopilotRuntime(),
        CodexRuntime(),
        OpenCodeZenRuntime(api_key="dummy-for-construction"),
    ]


def test_all_adapters_support_structured_output_json_schema(adapters: list) -> None:
    """The one feature universally wired today.

    Every adapter implements ``execute(schema=...)``; the TCK in
    airframe.testing.contracts will assert the round-trip itself. Here
    we just check the capability flag is honest.
    """
    for adapter in adapters:
        assert adapter.supports(Feature.STRUCTURED_OUTPUT_JSON_SCHEMA), (
            f"{type(adapter).__name__} should declare STRUCTURED_OUTPUT_JSON_SCHEMA"
        )


def test_phase_0_only_structured_output_is_true(adapters: list) -> None:
    """Every other Phase 1+ feature returns False on every adapter today.

    This is what gives ``supports()`` a trivial-but-honest contract in
    Phase 0: the only True bit is the feature whose API exists.
    Phase 1+ flips additional bits as it ships the corresponding APIs.
    """
    phase_1_plus = [f for f in Feature if f is not Feature.STRUCTURED_OUTPUT_JSON_SCHEMA]
    for adapter in adapters:
        for feature in phase_1_plus:
            assert not adapter.supports(feature), (
                f"{type(adapter).__name__} should not yet declare {feature.name} "
                f"(Phase 0 ships the flag, later phases wire the API)"
            )


def test_supports_accepts_optional_model_arg(adapters: list) -> None:
    """The ``model=`` kwarg is part of the signature even when unused.

    Phase 0 adapters ignore the model parameter (all features are
    runtime-wide today). The signature reserves the slot so per-model
    differentiation can land later without breaking call sites.
    """
    binding = ProviderModel("claude", "claude-haiku-4-5")
    for adapter in adapters:
        # Both call forms should work; the result for the universal
        # feature is the same.
        assert adapter.supports(Feature.STRUCTURED_OUTPUT_JSON_SCHEMA) == adapter.supports(
            Feature.STRUCTURED_OUTPUT_JSON_SCHEMA, model=binding
        )


def test_supports_is_pure(adapters: list) -> None:
    """Repeated calls yield the same answer; no side effects.

    Capability lookups must be cheap and idempotent — consumers will
    call them in tight branching paths. This guards against an
    accidental "lazy SDK probe on first call" implementation.
    """
    for adapter in adapters:
        first = adapter.supports(Feature.STREAMING)
        second = adapter.supports(Feature.STREAMING)
        third = adapter.supports(Feature.STREAMING)
        assert first == second == third
