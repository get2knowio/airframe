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
    assert Feature.TOOLS_MCP_SSE.value == "tools_mcp_sse"
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
        # Phase 4 Iteration B flipped the three MCP transport flags on
        # Claude (broadest transport coverage); Iteration C flips
        # STDIO + HTTP on Copilot (SSE keeps a permanent decline);
        # Iteration D leaves Codex + OpenAI-compat False everywhere.
        Feature.TOOLS_MCP_STDIO,
        Feature.TOOLS_MCP_HTTP,
        Feature.TOOLS_MCP_SSE,
        # Phase 5 Iteration B flipped PERMISSION_CALLBACK on Claude,
        # Copilot, and Codex; OpenAI-compat declines permanently
        # (Chat Completions has no permission wire shape).
        Feature.PERMISSION_CALLBACK,
        # Phase 5 Iteration C flipped LIFECYCLE_HOOKS on all four
        # adapters. Per-adapter emittable-kinds set differs (see
        # each adapter's EMITTABLE_HOOK_KINDS ClassVar).
        Feature.LIFECYCLE_HOOKS,
        # Phase 5 Iteration D flipped BUDGET_USD_CAP on all four
        # (client-side accumulation) and BUDGET_TURN_CAP on three
        # (Claude / Codex / OpenAI-compat). Copilot keeps the
        # turn-cap decline because its vendor caps internally.
        Feature.BUDGET_USD_CAP,
        Feature.BUDGET_TURN_CAP,
    }
    must_be_false = [f for f in Feature if f not in any_adapter_may_support]
    for adapter in adapters:
        for feature in must_be_false:
            assert not adapter.supports(feature), (
                f"{type(adapter).__name__} declares {feature.name} but its API "
                f"hasn't shipped yet — see dev-docs/implementation-plan.md for the "
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


def test_sdk_adapters_declare_session_resume(adapters: list) -> None:
    """SDK-based adapters (Claude Code, Copilot) declare SESSION_RESUME.

    Both have server-side session resume via their respective SDKs.
    OpenAI-compat (chat-completions) is the outlier — no server-side
    session.
    """
    from airframe import ClaudeCodeRuntime, CopilotRuntime, OpenCodeZenRuntime

    for adapter in adapters:
        if isinstance(adapter, ClaudeCodeRuntime | CopilotRuntime):
            assert adapter.supports(Feature.SESSION_RESUME), (
                f"{type(adapter).__name__} should declare SESSION_RESUME"
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


def test_sdk_adapters_declare_file_input(adapters: list) -> None:
    """FILE_INPUT lands on Claude / Copilot; OpenAI-compat stays False.

    The roadmap notes file routing "varies wildly across compat
    vendors" — ``client.files.create`` semantics differ from vendor to
    vendor, and some don't support it at all. A future per-vendor
    subclass can opt in; the base stays conservative.
    """
    from airframe import ClaudeCodeRuntime, CopilotRuntime, OpenCodeZenRuntime

    for adapter in adapters:
        if isinstance(adapter, ClaudeCodeRuntime | CopilotRuntime):
            assert adapter.supports(Feature.FILE_INPUT), (
                f"{type(adapter).__name__} should declare FILE_INPUT after Phase 2 Iteration C"
            )
        elif isinstance(adapter, OpenCodeZenRuntime):
            assert not adapter.supports(Feature.FILE_INPUT), (
                "OpenAI-compatible adapters keep FILE_INPUT False — file routing "
                "varies wildly across compat vendors"
            )


def test_tools_function_universal_across_in_tree_adapters(adapters: list) -> None:
    """Every in-tree adapter in this fixture (Claude, Copilot,
    OpenCode Zen) declares TOOLS_FUNCTION.

    Iteration B flipped it on OpenAI-compat (client-side tool-loop);
    Iteration C wired Claude (in-process MCP server via
    :func:`claude_agent_sdk.create_sdk_mcp_server`) and Copilot
    (:func:`copilot.define_tool` registrations on the session).
    Kimi declines permanently (no Python-callable channel) but isn't
    in the default fixture because the SDK can't be co-installed.
    """
    from airframe import (
        ClaudeCodeRuntime,
        CopilotRuntime,
        OpenCodeZenRuntime,
    )

    for adapter in adapters:
        if isinstance(adapter, ClaudeCodeRuntime | CopilotRuntime | OpenCodeZenRuntime):
            assert adapter.supports(Feature.TOOLS_FUNCTION), (
                f"{type(adapter).__name__} should declare TOOLS_FUNCTION"
            )


def test_claude_declares_all_three_mcp_transports(adapters: list) -> None:
    """Phase 4 Iteration B flips STDIO + HTTP + SSE on ClaudeCodeRuntime.

    Claude's SDK has typed configs for all three transports; the
    adapter translates :class:`McpServerRef` into the matching
    :class:`McpStdioServerConfig` / :class:`McpHttpServerConfig` /
    :class:`McpSSEServerConfig` and passes the keyed dict via
    :attr:`ClaudeAgentOptions.mcp_servers` (merged with the
    in-process tools server). The in-process flag stays False — Phase
    4 doesn't expose ``transport="in_process"`` on
    :class:`McpServerRef`; Claude's in-process MCP server is internal
    plumbing for ``tools=``.
    """
    from airframe import ClaudeCodeRuntime

    for adapter in adapters:
        if not isinstance(adapter, ClaudeCodeRuntime):
            continue
        assert adapter.supports(Feature.TOOLS_MCP_STDIO), (
            "ClaudeCodeRuntime should declare TOOLS_MCP_STDIO after Iteration B"
        )
        assert adapter.supports(Feature.TOOLS_MCP_HTTP), (
            "ClaudeCodeRuntime should declare TOOLS_MCP_HTTP after Iteration B"
        )
        assert adapter.supports(Feature.TOOLS_MCP_SSE), (
            "ClaudeCodeRuntime should declare TOOLS_MCP_SSE after Iteration B"
        )
        assert not adapter.supports(Feature.TOOLS_MCP_IN_PROCESS), (
            "ClaudeCodeRuntime should NOT declare TOOLS_MCP_IN_PROCESS — Phase 4 "
            "doesn't expose an in-process McpServerRef transport"
        )


def test_copilot_declares_stdio_and_http_but_not_sse(adapters: list) -> None:
    """Phase 4 Iteration C flips ``TOOLS_MCP_{STDIO,HTTP}`` on
    :class:`CopilotRuntime`.

    Per the implementation plan, SSE stays False on Copilot; refs of
    that transport surface a specific decline pointing consumers at
    ``http``. ``TOOLS_MCP_IN_PROCESS`` also stays False — Phase 4
    doesn't expose an in-process transport on :class:`McpServerRef`.
    """
    from airframe import CopilotRuntime

    for adapter in adapters:
        if not isinstance(adapter, CopilotRuntime):
            continue
        assert adapter.supports(Feature.TOOLS_MCP_STDIO), (
            "CopilotRuntime should declare TOOLS_MCP_STDIO after Iteration C"
        )
        assert adapter.supports(Feature.TOOLS_MCP_HTTP), (
            "CopilotRuntime should declare TOOLS_MCP_HTTP after Iteration C"
        )
        assert not adapter.supports(Feature.TOOLS_MCP_SSE), (
            "CopilotRuntime should NOT declare TOOLS_MCP_SSE — the plan "
            "declines SSE on Copilot for Phase 4"
        )
        assert not adapter.supports(Feature.TOOLS_MCP_IN_PROCESS), (
            "CopilotRuntime should NOT declare TOOLS_MCP_IN_PROCESS"
        )


def test_openai_compat_declines_all_mcp_transports(adapters: list) -> None:
    """OpenAI-compat declines every MCP transport permanently.

    Chat Completions has no MCP-as-tool slot; the future
    ``OpenAIResponsesRuntime`` could translate to the Responses-API
    ``{"type":"mcp",...}`` tool shape, but the base stays declined.
    """
    from airframe import ClaudeCodeRuntime, CopilotRuntime

    mcp_features = (
        Feature.TOOLS_MCP_STDIO,
        Feature.TOOLS_MCP_HTTP,
        Feature.TOOLS_MCP_SSE,
        Feature.TOOLS_MCP_IN_PROCESS,
    )
    for adapter in adapters:
        if isinstance(adapter, ClaudeCodeRuntime | CopilotRuntime):
            continue
        for feature in mcp_features:
            assert not adapter.supports(feature), (
                f"{type(adapter).__name__} should NOT declare {feature.name} — "
                f"OpenAI-compat declines all MCP transports permanently"
            )


def test_copilot_session_sse_decline_carries_http_hint(adapters: list) -> None:
    """SSE refs on Copilot raise with an actionable pointer at ``http``.

    The plan's Iteration C requirement: ``"SSE decline carries the
    http-transport hint"``. The decline runs *before* the shared
    capability gate so the consumer gets the specific "switch to http"
    advice rather than the generic "not wired" message.
    """
    from airframe import CopilotRuntime, McpServerRef
    from airframe.errors import UnsupportedFeatureError

    for adapter in adapters:
        if not isinstance(adapter, CopilotRuntime):
            continue
        with pytest.raises(UnsupportedFeatureError) as exc_info:
            adapter.session(
                mcp_servers=[
                    McpServerRef(name="remote", transport="sse", url="https://mcp.example.com/sse")
                ]
            )
        assert exc_info.value.feature == Feature.TOOLS_MCP_SSE
        text = str(exc_info.value).lower()
        assert "sse" in text
        # The actionable hint must point at http.
        assert "http" in text


def test_session_mcp_servers_kwarg_raises_on_openai_compat(
    adapters: list,
) -> None:
    """OpenAI-compat raises on every MCP transport.

    Claude opens cleanly (all three transports), Copilot opens cleanly
    for stdio + http (SSE has its own specific test). OpenAI-compat
    surfaces :class:`~airframe.errors.UnsupportedFeatureError` carrying
    the offending transport's :class:`Feature` on the ``.feature``
    attribute.
    """
    from airframe import ClaudeCodeRuntime, CopilotRuntime, McpServerRef
    from airframe.errors import UnsupportedFeatureError

    cases: list[tuple[McpServerRef, Feature]] = [
        (
            McpServerRef(
                name="local", transport="stdio", command=["uvx", "mcp-server-everything"]
            ),
            Feature.TOOLS_MCP_STDIO,
        ),
        (
            McpServerRef(name="remote-http", transport="http", url="https://mcp.example.com"),
            Feature.TOOLS_MCP_HTTP,
        ),
        (
            McpServerRef(name="remote-sse", transport="sse", url="https://mcp.example.com/sse"),
            Feature.TOOLS_MCP_SSE,
        ),
    ]

    for adapter in adapters:
        if isinstance(adapter, ClaudeCodeRuntime | CopilotRuntime):
            # Adapter-specific tests cover the accepting paths and (for
            # Copilot's SSE decline) the vendor-specific message.
            continue
        for ref, expected_feature in cases:
            with pytest.raises(UnsupportedFeatureError) as exc_info:
                adapter.session(mcp_servers=[ref])  # type: ignore[attr-defined]
            assert exc_info.value.feature == expected_feature, (
                f"{type(adapter).__name__} should raise with "
                f"feature={expected_feature.name!r} on a {ref.transport!r} ref; "
                f"got feature={exc_info.value.feature!r}"
            )


def test_claude_session_mcp_servers_kwarg_opens_cleanly(adapters: list) -> None:
    """Phase 4 Iteration B: Claude opens a session with each transport.

    Sanity check at the matrix level — per-transport translation /
    wire-shape assertions live in
    ``tests/test_claude_code_session.py``.
    """
    from airframe import ClaudeCodeRuntime, McpServerRef

    refs: list[McpServerRef] = [
        McpServerRef(name="local", transport="stdio", command=["uvx", "x"]),
        McpServerRef(name="remote-http", transport="http", url="https://mcp.example.com"),
        McpServerRef(name="remote-sse", transport="sse", url="https://mcp.example.com/sse"),
    ]

    for adapter in adapters:
        if not isinstance(adapter, ClaudeCodeRuntime):
            continue
        for ref in refs:
            sess = adapter.session(mcp_servers=[ref])
            assert sess is not None
            # No actual connect yet — _ensure_client is lazy.
            del sess


def test_mcp_transports_final_matrix(adapters: list) -> None:
    """The Phase-4 endgame matrix — pins the table from the
    iteration-breakdown header.

    +-------------------------+ stdio + http + sse + in_process +
    | ClaudeCodeRuntime       |  ✓   |  ✓   |  ✓  |    ✗       |
    | CopilotRuntime          |  ✓   |  ✓   |  ✗  |    ✗       |
    | OpenAICompatibleRuntime |  ✗   |  ✗   |  ✗  |    ✗       |

    ``TOOLS_MCP_IN_PROCESS`` stays False on every adapter — Phase 4
    doesn't expose an in-process transport on :class:`McpServerRef`;
    the Phase 3 in-process MCP path on Claude is internal plumbing
    for ``tools=`` rather than a user-facing capability. Flipping
    it later requires exposing ``transport="in_process"`` on
    :class:`McpServerRef`.
    """
    from airframe import (
        ClaudeCodeRuntime,
        CopilotRuntime,
        OpenCodeZenRuntime,
    )

    expected: dict[type, dict[Feature, bool]] = {
        ClaudeCodeRuntime: {
            Feature.TOOLS_MCP_STDIO: True,
            Feature.TOOLS_MCP_HTTP: True,
            Feature.TOOLS_MCP_SSE: True,
            Feature.TOOLS_MCP_IN_PROCESS: False,
        },
        CopilotRuntime: {
            Feature.TOOLS_MCP_STDIO: True,
            Feature.TOOLS_MCP_HTTP: True,
            Feature.TOOLS_MCP_SSE: False,
            Feature.TOOLS_MCP_IN_PROCESS: False,
        },
        OpenCodeZenRuntime: {
            Feature.TOOLS_MCP_STDIO: False,
            Feature.TOOLS_MCP_HTTP: False,
            Feature.TOOLS_MCP_SSE: False,
            Feature.TOOLS_MCP_IN_PROCESS: False,
        },
    }

    seen_classes: set[type] = set()
    for adapter in adapters:
        cls = type(adapter)
        if cls not in expected:
            continue
        seen_classes.add(cls)
        for feature, want in expected[cls].items():
            assert adapter.supports(feature) is want, (
                f"{cls.__name__}.supports({feature.name}) should be {want}; final-matrix mismatch"
            )
    # Sanity: every expected adapter class actually appeared in the
    # fixture so a typo in the fixture can't silently skip rows.
    assert seen_classes == set(expected.keys()), (
        f"adapters fixture didn't cover the full final-matrix; missing "
        f"{set(expected.keys()) - seen_classes}"
    )


def test_session_mcp_servers_empty_list_is_a_no_op(adapters: list) -> None:
    """An empty / ``None`` list never raises — same default as ``tools=``.

    Iteration A's gate only fires on non-empty input; falsy values
    open a session cleanly so consumers can write
    ``session(mcp_servers=refs or None)`` without branching.
    """
    for adapter in adapters:
        sess = adapter.session(mcp_servers=None)  # type: ignore[attr-defined]
        assert sess is not None
        del sess
        sess = adapter.session(mcp_servers=[])  # type: ignore[attr-defined]
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


# ---------------------------------------------------------------------------
# Phase 5 Iteration A — permission / hooks / budget scaffolding
# ---------------------------------------------------------------------------


def test_phase_5_final_matrix(adapters: list) -> None:
    """Phase 5 Iteration D — the endgame coverage matrix.

    Pins per the plan's Phase 5 coverage table:

    | Adapter | PERMISSION | HOOKS | BUDGET_USD | BUDGET_TURN |
    |---|---|---|---|---|
    | ClaudeCodeRuntime | ✓ | ✓ | ✓ | ✓ |
    | CopilotRuntime    | ✓ | ✓ | ✓ | ✗ |  ← turn cap vendor-internal
    | OpenCodeZenRuntime | ✗ | ✓ | ✓ | ✓ |  ← permission permanently declined

    Regressions in any cell are caught here.
    """
    from airframe import (
        ClaudeCodeRuntime,
        CopilotRuntime,
        OpenCodeZenRuntime,
    )

    matrix: dict[type, dict[Feature, bool]] = {
        ClaudeCodeRuntime: {
            Feature.PERMISSION_CALLBACK: True,
            Feature.LIFECYCLE_HOOKS: True,
            Feature.BUDGET_USD_CAP: True,
            Feature.BUDGET_TURN_CAP: True,
        },
        CopilotRuntime: {
            Feature.PERMISSION_CALLBACK: True,
            Feature.LIFECYCLE_HOOKS: True,
            Feature.BUDGET_USD_CAP: True,
            Feature.BUDGET_TURN_CAP: False,
        },
        OpenCodeZenRuntime: {
            Feature.PERMISSION_CALLBACK: False,
            Feature.LIFECYCLE_HOOKS: True,
            Feature.BUDGET_USD_CAP: True,
            Feature.BUDGET_TURN_CAP: True,
        },
    }
    for adapter in adapters:
        for cls, expected in matrix.items():
            if isinstance(adapter, cls):
                for feature, want in expected.items():
                    got = adapter.supports(feature)
                    assert got == want, (
                        f"{cls.__name__}.supports({feature.name}) should be {want}; got {got}"
                    )


def test_budget_usd_cap_universal(adapters: list) -> None:
    """Phase 5 Iteration D: every adapter declares BUDGET_USD_CAP.

    Client-side accumulation against ``RuntimeResult.cost.cost_usd``
    is universally available — the cap is enforced at turn boundary
    in v0 (mid-turn interrupt is additive later).
    """
    for adapter in adapters:
        assert adapter.supports(Feature.BUDGET_USD_CAP), (
            f"{type(adapter).__name__} should declare BUDGET_USD_CAP after Phase 5 Iteration D"
        )


def test_budget_turn_cap_universal_except_copilot(adapters: list) -> None:
    """Phase 5 Iteration D: Claude / OpenAI-compat flip
    BUDGET_TURN_CAP True; Copilot declines.

    Copilot's vendor SDK caps internal turns at the CLI level via the
    runtime's ``--max-turns`` config, so a user-facing ``max_turns=``
    surface would be misleading — the decline is the honest signal.
    """
    from airframe import CopilotRuntime

    for adapter in adapters:
        if isinstance(adapter, CopilotRuntime):
            assert not adapter.supports(Feature.BUDGET_TURN_CAP), (
                "CopilotRuntime should NOT declare BUDGET_TURN_CAP — "
                "vendor enforces turn caps internally"
            )
        else:
            assert adapter.supports(Feature.BUDGET_TURN_CAP), (
                f"{type(adapter).__name__} should declare BUDGET_TURN_CAP "
                f"after Phase 5 Iteration D"
            )


def test_lifecycle_hooks_universal(adapters: list) -> None:
    """Phase 5 Iteration C: every adapter declares LIFECYCLE_HOOKS.

    The per-adapter *emittable kinds set* differs — each adapter
    advertises an ``EMITTABLE_HOOK_KINDS`` ClassVar with the
    :class:`~airframe.hooks.HookEventKind` literals it can honestly
    produce. See :func:`test_emittable_hook_kinds_matrix` below for
    the per-adapter pin.
    """
    for adapter in adapters:
        assert adapter.supports(Feature.LIFECYCLE_HOOKS), (
            f"{type(adapter).__name__} should declare LIFECYCLE_HOOKS after Phase 5 Iteration C"
        )


def test_emittable_hook_kinds_matrix(adapters: list) -> None:
    """The per-adapter ``EMITTABLE_HOOK_KINDS`` matrix.

    Pinned so a regression where (say) Codex drops ``post_tool_use``
    from its emittable set is caught at PR time. ``session_start``
    and ``session_end`` are universal (every adapter synthesises
    them); ``pre_compact`` / ``rate_limit`` are vendor-specific.
    """
    from airframe import (
        ClaudeCodeRuntime,
        CopilotRuntime,
        OpenCodeZenRuntime,
    )

    expected: dict[type, set[str]] = {
        ClaudeCodeRuntime: {
            "session_start",
            "session_end",
            "user_prompt_submit",
            "pre_tool_use",
            "post_tool_use",
            "tool_failure",
            "pre_compact",
            "rate_limit",
        },
        CopilotRuntime: {
            "session_start",
            "session_end",
            "user_prompt_submit",
            "pre_tool_use",
            "post_tool_use",
            "tool_failure",
            "pre_compact",
        },
        OpenCodeZenRuntime: {
            "session_start",
            "session_end",
            "user_prompt_submit",
            "pre_tool_use",
            "post_tool_use",
            "tool_failure",
        },
    }
    for adapter in adapters:
        kinds = getattr(type(adapter), "EMITTABLE_HOOK_KINDS", None)
        assert kinds is not None, (
            f"{type(adapter).__name__} should expose an EMITTABLE_HOOK_KINDS "
            f"ClassVar after Phase 5 Iteration C"
        )
        for cls, want in expected.items():
            if isinstance(adapter, cls):
                assert set(kinds) == want, (
                    f"{cls.__name__}.EMITTABLE_HOOK_KINDS should be {want}; got {set(kinds)}"
                )


def test_permission_callback_universal_except_openai_compat(adapters: list) -> None:
    """Phase 5 Iteration B: Claude / Copilot flip PERMISSION_CALLBACK
    True; OpenAI-compat (chat-completions) permanently declines.
    """
    from airframe import (
        ClaudeCodeRuntime,
        CopilotRuntime,
        OpenCodeZenRuntime,
    )

    for adapter in adapters:
        if isinstance(adapter, ClaudeCodeRuntime | CopilotRuntime):
            assert adapter.supports(Feature.PERMISSION_CALLBACK), (
                f"{type(adapter).__name__} should declare PERMISSION_CALLBACK "
                f"after Phase 5 Iteration B"
            )
        elif isinstance(adapter, OpenCodeZenRuntime):
            assert not adapter.supports(Feature.PERMISSION_CALLBACK), (
                "OpenCodeZenRuntime should NOT declare PERMISSION_CALLBACK — "
                "Chat Completions has no tool-permission wire shape"
            )


def test_session_on_permission_kwarg_raises_only_on_openai_compat(adapters: list) -> None:
    """Phase 5 Iteration B: only OpenAI-compat raises on ``on_permission=``.

    The accepting adapters in the default fixture (Claude / Copilot)
    open cleanly with the callback supplied. OpenAI-compat raises with
    an actionable decline message pointing at the future
    ``OpenAIResponsesRuntime``.
    """
    from airframe import (
        ClaudeCodeRuntime,
        CopilotRuntime,
        OpenCodeZenRuntime,
        PermissionDecision,
        PermissionRequest,
    )
    from airframe.errors import UnsupportedFeatureError

    class _StubCallback:
        async def handle(self, request: PermissionRequest) -> PermissionDecision:
            return "allow"

    cb = _StubCallback()
    for adapter in adapters:
        if isinstance(adapter, OpenCodeZenRuntime):
            with pytest.raises(UnsupportedFeatureError) as exc_info:
                adapter.session(on_permission=cb)  # type: ignore[attr-defined]
            assert exc_info.value.feature == Feature.PERMISSION_CALLBACK
            text = str(exc_info.value).lower()
            # Pin the actionable pointer.
            assert "chat completions" in text
            assert "openairesponsesruntime" in text.replace(" ", "")
        elif isinstance(adapter, ClaudeCodeRuntime | CopilotRuntime):
            sess = adapter.session(on_permission=cb)  # type: ignore[attr-defined]
            assert sess is not None
            del sess


def test_session_on_event_kwarg_opens_cleanly_on_every_adapter(adapters: list) -> None:
    """Phase 5 Iteration C: every adapter accepts ``on_event=``.

    Per-adapter emission shape lives in the corresponding
    ``test_*_session.py`` file; here we just check the four
    adapters open the session cleanly with an observer registered.
    """
    from airframe import HookEvent

    def _observer(_event: HookEvent) -> None:
        pass

    for adapter in adapters:
        sess = adapter.session(on_event=_observer)  # type: ignore[attr-defined]
        assert sess is not None
        del sess


def test_session_on_permission_or_event_none_opens_cleanly(adapters: list) -> None:
    """``on_permission=None`` / ``on_event=None`` are both no-ops.

    The default-None values must continue to open sessions cleanly
    on every adapter — same shape as Phase 4's
    ``mcp_servers=None`` no-op contract.
    """
    for adapter in adapters:
        sess = adapter.session(on_permission=None, on_event=None)  # type: ignore[attr-defined]
        assert sess is not None
        del sess


async def test_execute_max_turns_kwarg_raises_only_on_copilot(adapters: list) -> None:
    """Phase 5 Iteration D: ``max_turns=`` raises only on Copilot.

    Claude / Codex / OpenAI-compat all flip ``BUDGET_TURN_CAP`` True
    and honour the kwarg. Copilot keeps the decline because the
    vendor SDK caps internal turns at the CLI level — exposing a
    user-facing ``max_turns=`` would be misleading.
    """
    from airframe import CopilotRuntime
    from airframe.errors import UnsupportedFeatureError

    for adapter in adapters:
        sess = adapter.session()  # type: ignore[attr-defined]
        try:
            if isinstance(adapter, CopilotRuntime):
                with pytest.raises(UnsupportedFeatureError) as exc_info:
                    await sess.execute("hi", max_turns=5)
                assert exc_info.value.feature == Feature.BUDGET_TURN_CAP, (
                    f"CopilotRuntime should raise with feature=BUDGET_TURN_CAP; "
                    f"got {exc_info.value.feature!r}"
                )
            # The accepting three need real vendor I/O to run a turn,
            # so we don't actually call execute() here. The per-adapter
            # session tests cover the enforcement path with mocks.
        finally:
            await sess.close()


async def test_execute_max_budget_usd_kwarg_no_longer_raises(adapters: list) -> None:
    """Phase 5 Iteration D: every adapter honours ``max_budget_usd=``.

    Iteration D flipped ``BUDGET_USD_CAP`` True on all four
    adapters; the gate no longer raises on any adapter. Behavioural
    coverage (cap firing, error attributes) lives in the per-adapter
    session tests with mocked vendor I/O.

    Verified at the gate layer: opening a session and calling the
    gate with ``max_budget_usd=`` must not raise
    :class:`UnsupportedFeatureError`. The actual ``execute()`` call
    needs live vendors and is out of scope here.
    """
    from airframe.sessions import _check_budget_supported

    for adapter in adapters:
        # The gate must accept non-None ``max_budget_usd=`` on every
        # adapter post-Iteration-D — no UnsupportedFeatureError.
        _check_budget_supported(
            max_turns=None,
            max_budget_usd=0.05,
            adapter_label=adapter.label,  # type: ignore[attr-defined]
            supports=adapter.supports,  # type: ignore[attr-defined]
        )


async def test_execute_budget_kwargs_none_opens_cleanly(adapters: list) -> None:
    """``max_turns=None`` / ``max_budget_usd=None`` are no-ops — must
    not raise.

    The gate only fires on non-None values; the default-None
    branches continue to flow through to per-adapter behaviour.
    Verified at the session-construction layer; the actual execute
    call needs live vendors and lives in the per-adapter session
    tests.
    """
    for adapter in adapters:
        sess = adapter.session()  # type: ignore[attr-defined]
        try:
            # Just constructing the session and exercising the gate
            # with the defaults — _check_budget_supported is called
            # at the top of execute() and must short-circuit when
            # both kwargs are None.
            from airframe.sessions import _check_budget_supported

            _check_budget_supported(
                max_turns=None,
                max_budget_usd=None,
                adapter_label=adapter.label,  # type: ignore[attr-defined]
                supports=adapter.supports,  # type: ignore[attr-defined]
            )
        finally:
            await sess.close()
