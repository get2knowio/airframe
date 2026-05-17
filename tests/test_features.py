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


def test_unwired_features_stay_false(adapters: list) -> None:
    """Features without a wired API still return False on every adapter.

    Phase 0 shipped every :class:`Feature` enum value and adapters
    declared only :data:`Feature.STRUCTURED_OUTPUT_JSON_SCHEMA`. Phase
    1 iteration C flips :data:`Feature.STREAMING` and
    :data:`Feature.CANCEL` on for the OpenAI-compatible family (real
    chunk streaming + ``asyncio.Task.cancel()``). Future iterations
    flip more bits per adapter as their APIs land.

    This contract guards the opposite invariant: features whose API
    hasn't shipped yet must not be declared by *any* adapter. Without
    it, an adapter could silently overdeclare and consumers branching
    on ``supports()`` would hit :class:`UnsupportedFeatureError` at
    the API call site — exactly the surprise the capability gate
    exists to prevent.
    """
    # Anything in ANY_ADAPTER_MAY_SUPPORT can vary per adapter; the
    # remainder must be False everywhere until its corresponding API
    # phase lands.
    any_adapter_may_support = {
        Feature.STRUCTURED_OUTPUT_JSON_SCHEMA,
        Feature.STREAMING,
        Feature.CANCEL,
        Feature.SESSION_RESUME,
        # Phase 2 Iteration B flipped REASONING_EFFORT on all four
        # adapters and REASONING_BUDGET_TOKENS on Claude Code only.
        Feature.REASONING_EFFORT,
        Feature.REASONING_BUDGET_TOKENS,
        # Phase 2 Iteration C flipped VISION_INPUT on all four
        # adapters and FILE_INPUT on three (OpenAI-compat stays False).
        Feature.VISION_INPUT,
        Feature.FILE_INPUT,
        # Phase 3 Iteration B flipped TOOLS_FUNCTION on OpenAI-compat;
        # Iteration C will flip it on Claude + Copilot; Iteration D
        # leaves Codex False (no SDK tool-registration channel).
        Feature.TOOLS_FUNCTION,
    }
    must_be_false = [f for f in Feature if f not in any_adapter_may_support]
    for adapter in adapters:
        for feature in must_be_false:
            assert not adapter.supports(feature), (
                f"{type(adapter).__name__} declares {feature.name} but its API "
                f"hasn't shipped yet — see docs/implementation-plan.md for the "
                f"phase that wires it"
            )


def test_openai_compatible_declares_streaming_and_cancel(adapters: list) -> None:
    """OpenAI-compatible adapters flip STREAMING + CANCEL in Phase 1 / Iteration C.

    Chat-completions has no server-side session, so ``SESSION_RESUME``
    stays False per the plan.
    """
    from airframe import OpenCodeZenRuntime

    for adapter in adapters:
        if not isinstance(adapter, OpenCodeZenRuntime):
            continue
        assert adapter.supports(Feature.STREAMING), (
            "OpenCodeZenRuntime (OpenAICompatibleRuntime base) should declare "
            "Feature.STREAMING after Iteration C"
        )
        assert adapter.supports(Feature.CANCEL), (
            "OpenCodeZenRuntime (OpenAICompatibleRuntime base) should declare "
            "Feature.CANCEL after Iteration C"
        )
        assert not adapter.supports(Feature.SESSION_RESUME), (
            "OpenCodeZenRuntime can't resume — chat-completions has no server-side session"
        )


def test_claude_code_declares_streaming_resume_and_cancel(adapters: list) -> None:
    """ClaudeCodeRuntime flips STREAMING + SESSION_RESUME + CANCEL in Iteration D.

    Wired via ``include_partial_messages=True`` for streaming,
    :attr:`ClaudeAgentOptions.resume` for resume, and
    :meth:`ClaudeSDKClient.interrupt` for cancellation.
    """
    from airframe import ClaudeCodeRuntime

    for adapter in adapters:
        if not isinstance(adapter, ClaudeCodeRuntime):
            continue
        assert adapter.supports(Feature.STREAMING), (
            "ClaudeCodeRuntime should declare Feature.STREAMING after Iteration D"
        )
        assert adapter.supports(Feature.SESSION_RESUME), (
            "ClaudeCodeRuntime should declare Feature.SESSION_RESUME after "
            "Iteration D — wired via ClaudeAgentOptions.resume"
        )
        assert adapter.supports(Feature.CANCEL), (
            "ClaudeCodeRuntime should declare Feature.CANCEL after Iteration D — "
            "wired via ClaudeSDKClient.interrupt()"
        )


def test_copilot_declares_streaming_resume_and_cancel(adapters: list) -> None:
    """CopilotRuntime flips STREAMING + SESSION_RESUME + CANCEL in Iteration E.

    Wired via ``session.on(handler)`` filtering on
    ``ASSISTANT_MESSAGE_DELTA`` / ``ASSISTANT_REASONING_DELTA`` for
    streaming, :meth:`CopilotClient.resume_session` for resume, and
    :meth:`CopilotSession.abort` for cancellation.
    """
    from airframe import CopilotRuntime

    for adapter in adapters:
        if not isinstance(adapter, CopilotRuntime):
            continue
        assert adapter.supports(Feature.STREAMING), (
            "CopilotRuntime should declare Feature.STREAMING after Iteration E"
        )
        assert adapter.supports(Feature.SESSION_RESUME), (
            "CopilotRuntime should declare Feature.SESSION_RESUME after "
            "Iteration E — wired via CopilotClient.resume_session"
        )
        assert adapter.supports(Feature.CANCEL), (
            "CopilotRuntime should declare Feature.CANCEL after Iteration E — "
            "wired via CopilotSession.abort"
        )


def test_codex_declares_streaming_resume_and_cancel(adapters: list) -> None:
    """CodexRuntime flips STREAMING + SESSION_RESUME + CANCEL in Iteration F.

    Wired via :meth:`Thread.run_streamed` for streaming,
    :meth:`Codex.resume_thread` for resume, and :class:`AbortController`
    / :attr:`TurnOptions.signal` for cancellation. With this iteration
    landing, every in-tree adapter exposes the same Phase 1 capability
    set (OpenAI-compat is the only one without `SESSION_RESUME` since
    chat-completions has no server-side session).
    """
    from airframe import CodexRuntime

    for adapter in adapters:
        if not isinstance(adapter, CodexRuntime):
            continue
        assert adapter.supports(Feature.STREAMING), (
            "CodexRuntime should declare Feature.STREAMING after Iteration F"
        )
        assert adapter.supports(Feature.SESSION_RESUME), (
            "CodexRuntime should declare Feature.SESSION_RESUME after Iteration F — "
            "wired via Codex.resume_thread"
        )
        assert adapter.supports(Feature.CANCEL), (
            "CodexRuntime should declare Feature.CANCEL after Iteration F — "
            "wired via AbortController + TurnOptions.signal"
        )


def test_three_sdk_adapters_declare_session_resume(adapters: list) -> None:
    """Phase 1 endgame: every SDK-based adapter declares SESSION_RESUME.

    Three of four in-tree adapters (Claude Code, Copilot, Codex) all
    have server-side session resume via their respective SDKs.
    OpenAI-compat (chat-completions) is the outlier — no server-side
    session. This test pins the matrix at the end of Phase 1's
    per-adapter rollout so a future regression on any of the three
    SDK adapters is caught.
    """
    from airframe import ClaudeCodeRuntime, CodexRuntime, CopilotRuntime, OpenCodeZenRuntime

    for adapter in adapters:
        if isinstance(adapter, ClaudeCodeRuntime | CopilotRuntime | CodexRuntime):
            assert adapter.supports(Feature.SESSION_RESUME), (
                f"{type(adapter).__name__} should declare SESSION_RESUME at end of Phase 1"
            )
        elif isinstance(adapter, OpenCodeZenRuntime):
            assert not adapter.supports(Feature.SESSION_RESUME), (
                "OpenCodeZenRuntime / OpenAICompatibleRuntime can't resume "
                "(chat-completions has no server-side session)"
            )


def test_all_adapters_declare_reasoning_effort(adapters: list) -> None:
    """Phase 2 Iteration B flips REASONING_EFFORT on every adapter.

    All four vendor SDKs have a native "reasoning effort" channel
    (``minimal/low/medium/high`` — Claude/Copilot coerce ``"minimal"``
    to ``"low"`` at debug-log level since their SDKs only expose three
    rungs). With this iteration landing, ``thinking="<effort>"`` is a
    universal capability and consumer code can pass it without
    branching per adapter.
    """
    for adapter in adapters:
        assert adapter.supports(Feature.REASONING_EFFORT), (
            f"{type(adapter).__name__} should declare REASONING_EFFORT after Phase 2 Iteration B"
        )


def test_only_claude_code_declares_reasoning_budget_tokens(adapters: list) -> None:
    """``thinking={"budget_tokens": N}`` is Anthropic-only.

    Claude's SDK exposes a token budget via :class:`ThinkingConfig`;
    the other three vendors expose only the literal effort enum, so
    they decline the dict shape at translation time with
    :class:`UnsupportedFeatureError`.
    """
    from airframe import ClaudeCodeRuntime

    for adapter in adapters:
        if isinstance(adapter, ClaudeCodeRuntime):
            assert adapter.supports(Feature.REASONING_BUDGET_TOKENS), (
                "ClaudeCodeRuntime should declare REASONING_BUDGET_TOKENS after "
                "Phase 2 Iteration B — wired via ClaudeAgentOptions.thinking"
            )
        else:
            assert not adapter.supports(Feature.REASONING_BUDGET_TOKENS), (
                f"{type(adapter).__name__} should NOT declare REASONING_BUDGET_TOKENS — "
                f"only Anthropic exposes a per-call thinking-token budget"
            )


def test_all_adapters_declare_vision_input(adapters: list) -> None:
    """Phase 2 Iteration C flips VISION_INPUT on every adapter.

    All four vendor surfaces have *some* image-input path:

    * Anthropic / Claude Code — prompt-text hint + Read tool.
    * GitHub Copilot — :class:`FileAttachment` on ``send_and_wait``.
    * OpenAI Codex — :class:`LocalImageInput` on ``Thread.run``.
    * OpenAI-compatible HTTP — content-parts ``image_url`` shape.

    Path-only in v0 across the board; bytes / URL is deferred.
    """
    for adapter in adapters:
        assert adapter.supports(Feature.VISION_INPUT), (
            f"{type(adapter).__name__} should declare VISION_INPUT after Phase 2 Iteration C"
        )


def test_three_sdk_adapters_declare_file_input(adapters: list) -> None:
    """FILE_INPUT lands on Claude / Copilot / Codex; OpenAI-compat stays False.

    The roadmap notes file routing "varies wildly across compat
    vendors" — ``client.files.create`` semantics differ from vendor to
    vendor, and some don't support it at all. A future per-vendor
    subclass can opt in; the base stays conservative.
    """
    from airframe import ClaudeCodeRuntime, CodexRuntime, CopilotRuntime, OpenCodeZenRuntime

    for adapter in adapters:
        if isinstance(adapter, ClaudeCodeRuntime | CopilotRuntime | CodexRuntime):
            assert adapter.supports(Feature.FILE_INPUT), (
                f"{type(adapter).__name__} should declare FILE_INPUT after Phase 2 Iteration C"
            )
        elif isinstance(adapter, OpenCodeZenRuntime):
            assert not adapter.supports(Feature.FILE_INPUT), (
                "OpenAI-compatible adapters keep FILE_INPUT False — file routing "
                "varies wildly across compat vendors"
            )


def test_tools_function_universal_except_codex(adapters: list) -> None:
    """Final TOOLS_FUNCTION matrix after Phase 3: Codex is the only adapter
    that declines.

    Iteration B flipped it on OpenAI-compat (client-side tool-loop);
    Iteration C wired Claude (in-process MCP server via
    :func:`claude_agent_sdk.create_sdk_mcp_server`) and Copilot
    (:func:`copilot.define_tool` registrations on the session);
    Iteration D codifies Codex's decline. The Codex Python SDK has no
    tool-registration channel — consumers wire tools through the
    ``codex`` CLI's config file instead.
    """
    from airframe import (
        ClaudeCodeRuntime,
        CodexRuntime,
        CopilotRuntime,
        OpenCodeZenRuntime,
    )

    for adapter in adapters:
        if isinstance(adapter, ClaudeCodeRuntime | CopilotRuntime | OpenCodeZenRuntime):
            assert adapter.supports(Feature.TOOLS_FUNCTION), (
                f"{type(adapter).__name__} should declare TOOLS_FUNCTION after Phase 3"
            )
        elif isinstance(adapter, CodexRuntime):
            assert not adapter.supports(Feature.TOOLS_FUNCTION), (
                "CodexRuntime should NOT declare TOOLS_FUNCTION — the Codex "
                "Python SDK has no tool-registration channel"
            )


def test_session_tools_kwarg_raises_unsupported_feature_on_codex(
    adapters: list,
) -> None:
    """Iteration D: only Codex raises on ``tools=``; the other three accept it.

    Codex's decline carries an actionable CLI-config pointer rather
    than the generic "not wired yet" message from the shared
    capability gate. The other three adapters open cleanly with
    ``tools=`` and dispatch through their respective SDK channels.
    """
    from pydantic import BaseModel

    from airframe import (
        CodexRuntime,
        FunctionTool,
    )
    from airframe.errors import UnsupportedFeatureError

    class _NoArgs(BaseModel):
        pass

    async def _noop(_p: _NoArgs) -> None:
        return None

    tool = FunctionTool(
        name="noop",
        description="Test tool — never invoked.",
        params=_NoArgs,
        handler=_noop,
    )

    for adapter in adapters:
        if isinstance(adapter, CodexRuntime):
            with pytest.raises(UnsupportedFeatureError) as exc_info:
                adapter.session(tools=[tool])  # type: ignore[attr-defined]
            assert exc_info.value.feature == Feature.TOOLS_FUNCTION
            # Iteration D message must point at the workaround.
            text = str(exc_info.value).lower()
            assert "config" in text and "codex" in text
        else:
            # The other three open cleanly with tools=.
            sess = adapter.session(tools=[tool])
            assert sess is not None
            del sess


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
