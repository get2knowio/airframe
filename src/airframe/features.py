""":class:`Feature` — capability negotiation for :class:`AgentRuntime`.

Borrowed directly from JDBC's ``DatabaseMetaData.supportsXxx()`` predicates
and SQLAlchemy's ``Dialect.supports_*`` flags: a typed enumeration of
capabilities a runtime *may* expose, plus a single
:meth:`AgentRuntime.supports` method that returns whether *this*
runtime exposes it.

The whole forward-looking set ships in v0.3.0 (Phase 0 of the
implementation plan). Later phases turn ``True`` bits on as their
respective APIs land — they do not add new enum members. This matters
because the string values here are public surface: consumer code that
does ``if runtime.supports(Feature.STREAMING): ...`` should keep
working unmodified across versions.

The associated APIs:

* :data:`Feature.STRUCTURED_OUTPUT_JSON_SCHEMA` — the ``schema=`` kwarg
  on :meth:`AgentRuntime.execute`. Universally supported today.
* :data:`Feature.STRUCTURED_OUTPUT_STRICT` — strict JSON-Schema mode
  (no ``oneOf``, all properties required, ``additionalProperties:false``,
  restricted type set). No API knob in Phase 0; Phase 2 may add a
  ``strict=`` kwarg. Currently every adapter returns ``False``.
* :data:`Feature.STREAMING` — token-level streaming via a future
  ``runtime.stream()`` / ``session.stream()`` method. Phase 1.
* :data:`Feature.SESSION_RESUME` — resume a prior conversation by ID.
  Phase 1, via ``runtime.session(resume=...)``.
* :data:`Feature.CANCEL` — cancel an in-flight call. Phase 1, via
  ``session.cancel()`` or :class:`asyncio.Task` cancellation.
* :data:`Feature.REASONING_EFFORT` — ``thinking=`` kwarg accepting
  literal effort levels (``"minimal"``, ``"low"``, ``"medium"``,
  ``"high"``). Phase 2.
* :data:`Feature.REASONING_BUDGET_TOKENS` — ``thinking={"budget_tokens":N}``
  form. Claude-only today. Phase 2.
* :data:`Feature.VISION_INPUT` — image content parts on
  ``prompt=``. Phase 2.
* :data:`Feature.FILE_INPUT` — document / PDF content parts on
  ``prompt=``. Phase 2.
* :data:`Feature.TOOLS_FUNCTION` — caller-defined function tools via
  ``session(tools=[...])``. Phase 3.
* :data:`Feature.TOOLS_MCP_STDIO` / ``_HTTP`` / ``_IN_PROCESS`` — MCP
  server registration variants. Phase 4.
* :data:`Feature.PERMISSION_CALLBACK` — pre-tool-execution permission
  callback. Phase 5.
* :data:`Feature.LIFECYCLE_HOOKS` — typed event observation. Phase 5.
* :data:`Feature.BUDGET_USD_CAP` / ``BUDGET_TURN_CAP`` — budget caps
  on :meth:`session.execute`. Phase 5.
* :data:`Feature.SANDBOX` — sandboxed tool execution. Phase 6.
* :data:`Feature.SUBAGENTS` — programmatic subagent definitions.
  Phase 6.

The contract :meth:`AgentRuntime.supports` honours:

1. **Cheap.** No network, no SDK version sniffing, no subprocess
   probe. A static lookup table on the adapter class.
2. **Agrees with execute().** If ``supports(F)`` returns ``True``,
   calling the API associated with ``F`` must not raise
   :class:`UnsupportedBindingError` purely on capability grounds.
3. **False is the safe default.** Adapters declare what they *do*
   support; everything else is False. Consumers branching on
   ``supports()`` get correct behaviour even when running against a
   future runtime that adds new enum members.
"""

from __future__ import annotations

from enum import StrEnum


class Feature(StrEnum):
    """Capability flags consumers branch on for portable behaviour.

    :class:`enum.StrEnum` so members compare equal to their wire value
    and serialise cleanly in structured logs / config files
    (``Feature.STREAMING == "streaming"``).
    """

    # --- Structured output (Phase 0 — universally supported today) ---
    STRUCTURED_OUTPUT_JSON_SCHEMA = "structured_output_json_schema"
    """``execute(..., schema=PydanticModel)`` round-trips a typed payload."""

    STRUCTURED_OUTPUT_STRICT = "structured_output_strict"
    """Strict JSON-Schema enforcement (no fallback to best-effort JSON)."""

    # --- Phase 1 — streaming, session lifecycle, cancellation ---
    STREAMING = "streaming"
    """Token-level streaming via ``runtime.stream()`` / ``session.stream()``."""

    SESSION_RESUME = "session_resume"
    """Resume a prior conversation by id via ``runtime.session(resume=...)``."""

    CANCEL = "cancel"
    """Mid-call cancellation via ``session.cancel()`` or task cancel."""

    # --- Phase 2 — inputs and reasoning ---
    REASONING_EFFORT = "reasoning_effort"
    """``thinking=`` accepts ``"minimal"``/``"low"``/``"medium"``/``"high"``."""

    REASONING_BUDGET_TOKENS = "reasoning_budget_tokens"
    """``thinking={"budget_tokens": N}`` form for explicit reasoning budget."""

    VISION_INPUT = "vision_input"
    """``prompt=`` accepts image content parts (path / bytes / URL)."""

    FILE_INPUT = "file_input"
    """``prompt=`` accepts document / PDF content parts."""

    # --- Phase 3 — function tools ---
    TOOLS_FUNCTION = "tools_function"
    """``session(tools=[FunctionTool(...)])`` registers caller-defined tools."""

    # --- Phase 4 — MCP server references ---
    TOOLS_MCP_STDIO = "tools_mcp_stdio"
    """Registers stdio-transport MCP servers via ``session(mcp_servers=...)``."""

    TOOLS_MCP_HTTP = "tools_mcp_http"
    """Registers HTTP-transport MCP servers via ``session(mcp_servers=...)``."""

    TOOLS_MCP_IN_PROCESS = "tools_mcp_in_process"
    """Registers in-process MCP servers (zero IPC overhead)."""

    # --- Phase 5 — permission, hooks, budget ---
    PERMISSION_CALLBACK = "permission_callback"
    """``session(on_permission=...)`` gates tool execution."""

    LIFECYCLE_HOOKS = "lifecycle_hooks"
    """``session(on_event=...)`` receives typed :class:`HookEvent` observations."""

    BUDGET_USD_CAP = "budget_usd_cap"
    """``execute(max_budget_usd=...)`` aborts above a USD threshold."""

    BUDGET_TURN_CAP = "budget_turn_cap"
    """``execute(max_turns=...)`` aborts after a turn count."""

    # --- Phase 6 — sandbox, subagents ---
    SANDBOX = "sandbox"
    """``session(sandbox=...)`` constrains tool filesystem / network access."""

    SUBAGENTS = "subagents"
    """``session(agents=...)`` defines programmatic subagents."""


__all__ = ["Feature"]
