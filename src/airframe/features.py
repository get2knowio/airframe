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
* :data:`Feature.REASONING_OUTPUT` — adapter surfaces the model's
  reasoning trace on :attr:`RuntimeResult.reasoning`. Phase 6.
* :data:`Feature.REQUEST_METADATA` — adapter forwards
  :class:`~airframe.metadata.RequestMetadata` to a native vendor
  channel (abuse-detection / attribution tag). Soft contract:
  non-supporting adapters silently drop the kwarg rather than
  raising. Phase 6.
* :data:`Feature.COUNT_TOKENS` — adapter exposes
  :meth:`AgentRuntime.count_tokens`. Phase 6.
* :data:`Feature.PROMPT_CACHE_CONTROL` — adapter forwards
  :class:`~airframe.cache.CacheConfig` to the vendor's explicit
  cache-key channel. Soft contract. Phase 6.
* :data:`Feature.SLASH_COMMANDS` — adapter discovers + invokes
  user-triggered slash commands from the filesystem. Scaffolding
  only today; namespace locked for forward compat. Phase 6.
* :data:`Feature.VISION_INPUT` — image content parts on
  ``prompt=``. Phase 2.
* :data:`Feature.FILE_INPUT` — document / PDF content parts on
  ``prompt=``. Phase 2.
* :data:`Feature.TOOLS_FUNCTION` — caller-defined function tools via
  ``session(tools=[...])``. Phase 3.
* :data:`Feature.TOOLS_MCP_STDIO` / ``_HTTP`` / ``_SSE`` / ``_IN_PROCESS``
  — MCP server registration variants. Phase 4.
* :data:`Feature.PERMISSION_CALLBACK` — pre-tool-execution permission
  callback. Phase 5.
* :data:`Feature.LIFECYCLE_HOOKS` — typed event observation. Phase 5.
* :data:`Feature.BUDGET_USD_CAP` / ``BUDGET_TURN_CAP`` — budget caps
  on :meth:`session.execute`. Phase 5.
* :data:`Feature.RATE_LIMIT_TELEMETRY` — typed
  :class:`~airframe.rate_limit.RateLimitInfo` on
  :class:`~airframe.protocol.RuntimeResult` and
  :class:`~airframe.errors.RuntimeTransientError`. Phase 6.
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

Polyfill rings — how far an adapter may go to satisfy a feature
--------------------------------------------------------------

Some backends are full agent runtimes; others are bare chat surfaces.
When a feature is native on some and absent on others, airframe holds
a line documented in
:doc:`dev-docs/capability-polyfill-boundary` — *translate and
dispatch, never originate*. Each :class:`Feature` is classified by the
**highest ring** an adapter may use to satisfy it, in
:data:`POLYFILL_RING`:

* :attr:`Ring.TRANSPORT` (0) — the adapter may satisfy the feature
  only by shaping the vendor request/response or by a pure local
  computation (tokeniser, filesystem discovery). It must be backed by
  the vendor's *own* capability; where the vendor lacks it the adapter
  returns ``False``. Airframe never fakes a Transport feature —
  synthesising :data:`TOOLS_NATIVE`, :data:`SUBAGENTS`,
  :data:`STREAMING`, or :data:`SANDBOX` would mean running a search /
  an agent loop / a sandbox of our own, which is origination.
* :attr:`Ring.DISPATCH` (1) — the adapter may *synthesise* the feature
  with a bounded, capped, mechanical loop that hands control to a
  capability the **caller or vendor** supplies (a
  :class:`~airframe.tools.FunctionTool` handler; a vendor-hosted tool).
  The canonical case is the client-side function-tool loop on
  :class:`~airframe.adapters.openai_compatible.OpenAICompatibleRuntime`,
  bounded by ``max_turns``. Airframe orchestrates round-trips but
  originates nothing.
* :attr:`Ring.ORIGINATION` (2) — airframe supplies capability or
  judgment of its own (a built-in web search / RAG / planner /
  agent-to-agent router / conversation memory / a prompt the caller
  didn't write). **No :class:`Feature` is ever classified here** — it
  is the Hermes/PI space airframe is not trying to occupy, and the
  same set the README assigns to the consumer. A proposed feature that
  can only be honoured by origination is the review tripwire: gate it
  behind ``supports()`` and let the consumer bring the capability (as a
  ``FunctionTool``) or pick a stronger backend, or spin it out as a
  separate project above the protocol — never fold it into an adapter.

The enforceable invariants (see ``tests/test_features.py``):
:data:`POLYFILL_RING` covers every :class:`Feature`, and
:attr:`Ring.ORIGINATION` never appears among its values.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum


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

    REASONING_OUTPUT = "reasoning_output"
    """Adapter surfaces the model's reasoning trace on
    :attr:`RuntimeResult.reasoning` (and concatenates streamed
    :class:`~airframe.events.ReasoningDelta` payloads into the trailing
    :class:`~airframe.events.TurnComplete`). Distinct from
    :data:`REASONING_EFFORT`: the latter is "I can ask the model to
    think harder"; this is "I can show you what it thought."
    """

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

    TOOLS_MCP_SSE = "tools_mcp_sse"
    """Registers SSE-transport MCP servers via ``session(mcp_servers=...)``."""

    TOOLS_MCP_IN_PROCESS = "tools_mcp_in_process"
    """Registers in-process MCP servers (zero IPC overhead)."""

    TOOLS_NATIVE = "tools_native"
    """``session(native_tools=[NativeTool(...)])`` enables vendor-hosted
    built-in tools — ones the wrapped SDK both describes to the model and
    executes itself (Claude ``WebSearch`` / ``WebFetch``, OpenAI ``web_search``,
    Copilot ``fetch_webpage``). Distinct from
    :data:`TOOLS_FUNCTION` (caller supplies a Python handler) and the
    :data:`TOOLS_MCP_STDIO` family (external server): native tools carry no
    handler and no server ref — the consumer references a capability and the
    vendor owns description + execution.

    Structural gate (hard contract, like :data:`TOOLS_FUNCTION`): an adapter
    returning ``False`` raises :class:`~airframe.errors.UnsupportedFeatureError`
    when ``native_tools=`` carries any tool addressed to it. *Which*
    :class:`~airframe.native_tools.NativeCapability` values a supporting adapter
    can serve is a finer question answered by
    :meth:`AgentRuntime.supported_native_tools`; requesting a semantic
    capability the adapter doesn't serve also raises (no silent fallback)."""

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
    RATE_LIMIT_TELEMETRY = "rate_limit_telemetry"
    """Adapters populate :class:`~airframe.rate_limit.RateLimitInfo` on
    :attr:`RuntimeResult.rate_limit` (success) and
    :attr:`RuntimeTransientError.rate_limit` (throttle)."""

    REQUEST_METADATA = "request_metadata"
    """Adapter forwards :class:`~airframe.metadata.RequestMetadata` to
    a native vendor channel (OpenAI ``user=`` / ``metadata=``,
    Anthropic ``metadata={"user_id": ...}``). Soft contract — adapters
    that return ``False`` silently drop the kwarg rather than
    raising; consumers who care branch on this flag."""

    COUNT_TOKENS = "count_tokens"
    """Adapter exposes a working :meth:`AgentRuntime.count_tokens`
    method that returns the model's tokeniser-accurate count for a
    prompt, *without* paying for a turn. Adapters that return
    ``False`` raise :class:`UnsupportedFeatureError` when called."""

    PROMPT_CACHE_CONTROL = "prompt_cache_control"
    """Adapter forwards :class:`~airframe.cache.CacheConfig` to a
    native vendor cache-key channel (OpenAI's ``prompt_cache_key`` /
    ``prompt_cache_retention``). Soft contract — adapters that return
    ``False`` silently drop the kwarg; the call still succeeds, just
    without the speed-up / cost reduction explicit caching would
    have provided. Consumers who care branch on this flag."""

    SLASH_COMMANDS = "slash_commands"
    """Adapter discovers user-invokable slash commands from the
    filesystem and lets the consumer enumerate / trigger them through
    :class:`~airframe.slash_commands.SlashCommandsConfig`. Sibling of
    Agent Skills — same YAML-frontmatter convention, different
    trigger semantics (user-invoked vs model-invoked). Scaffolding
    only today; no adapter currently returns ``True``."""

    SANDBOX = "sandbox"
    """``session(sandbox=...)`` constrains tool filesystem / network access."""

    SUBAGENTS = "subagents"
    """``session(agents=...)`` defines programmatic subagents."""


class Ring(IntEnum):
    """The highest degree of intervention airframe permits itself when
    satisfying a :class:`Feature` on a backend that lacks it natively.

    See the module docstring's "Polyfill rings" section and
    :doc:`dev-docs/capability-polyfill-boundary`. Ordered by increasing
    intervention; :attr:`ORIGINATION` is the out-of-scope marker no
    :class:`Feature` may be assigned.
    """

    TRANSPORT = 0
    """Shape the vendor request/response (or a pure local computation).
    Native passthrough only — never synthesised."""

    DISPATCH = 1
    """A bounded, capped, mechanical loop dispatching to a capability the
    caller or vendor supplies. Synthesis permitted; must stay legible,
    capped, and delegate to the native loop wherever the vendor runs
    one."""

    ORIGINATION = 2
    """Airframe supplies capability or judgment of its own. Out of scope
    — no :class:`Feature` is ever classified here."""


#: The highest :class:`Ring` an adapter may use to satisfy each
#: :class:`Feature`. Transport features must be backed by the vendor's
#: own capability (or a pure local computation); Dispatch features may
#: additionally be synthesised via a capped caller/vendor-dispatched
#: loop. No feature maps to :attr:`Ring.ORIGINATION` — synthesising one
#: of these would make airframe an agent rather than a wrapper. Adding a
#: feature that could *only* be honoured by origination is the design
#: tripwire; gate it instead. Kept exhaustive over :class:`Feature` and
#: asserted in ``tests/test_features.py``.
POLYFILL_RING: dict[Feature, Ring] = {
    # Structured output — JSON-schema may be synthesised via a forced
    # tool / a bounded two-call "reason then structure" pass (Copilot
    # already does the forced-tool form). Strict enforcement can't be
    # honestly faked — the vendor either constrains decoding or it
    # doesn't — so it stays native-only.
    Feature.STRUCTURED_OUTPUT_JSON_SCHEMA: Ring.DISPATCH,
    Feature.STRUCTURED_OUTPUT_STRICT: Ring.TRANSPORT,
    # Streaming / sessions / cancellation — all pure wire shaping. A
    # non-streaming endpoint is not made to "stream"; resume on a
    # stateless backend is a client-side message buffer (bounded state,
    # no model-controlled loop, nothing originated).
    Feature.STREAMING: Ring.TRANSPORT,
    Feature.SESSION_RESUME: Ring.TRANSPORT,
    Feature.CANCEL: Ring.TRANSPORT,
    # Reasoning / inputs — forward a knob, parse a field, encode a
    # content part.
    Feature.REASONING_EFFORT: Ring.TRANSPORT,
    Feature.REASONING_BUDGET_TOKENS: Ring.TRANSPORT,
    Feature.REASONING_OUTPUT: Ring.TRANSPORT,
    Feature.VISION_INPUT: Ring.TRANSPORT,
    Feature.FILE_INPUT: Ring.TRANSPORT,
    # Function tools — the canonical Dispatch case: airframe runs the
    # capped client-side loop but dispatches only to the caller's
    # handler.
    Feature.TOOLS_FUNCTION: Ring.DISPATCH,
    # MCP / native tools — referenced and forwarded to the vendor's own
    # entry point; the vendor hosts and executes. Airframe references,
    # never runs. Synthesising these would mean running a tool server /
    # a web search of our own (origination) — so native-only.
    Feature.TOOLS_MCP_STDIO: Ring.TRANSPORT,
    Feature.TOOLS_MCP_HTTP: Ring.TRANSPORT,
    Feature.TOOLS_MCP_SSE: Ring.TRANSPORT,
    Feature.TOOLS_MCP_IN_PROCESS: Ring.TRANSPORT,
    Feature.TOOLS_NATIVE: Ring.TRANSPORT,
    # Permission / observation — wire a callback or an event tap into
    # the vendor's channel.
    Feature.PERMISSION_CALLBACK: Ring.TRANSPORT,
    Feature.LIFECYCLE_HOOKS: Ring.TRANSPORT,
    # Budget caps — airframe may enforce these itself by accounting
    # across the round-trips it already runs and aborting; that is
    # bounded, mechanical, and originates nothing (the turn cap is
    # literally the ceiling on the Dispatch loop).
    Feature.BUDGET_USD_CAP: Ring.DISPATCH,
    Feature.BUDGET_TURN_CAP: Ring.DISPATCH,
    # Telemetry / metadata / offline computation — parse headers, forward
    # a tag, run a tokeniser, discover files. Never originated.
    Feature.RATE_LIMIT_TELEMETRY: Ring.TRANSPORT,
    Feature.REQUEST_METADATA: Ring.TRANSPORT,
    Feature.COUNT_TOKENS: Ring.TRANSPORT,
    Feature.PROMPT_CACHE_CONTROL: Ring.TRANSPORT,
    Feature.SLASH_COMMANDS: Ring.TRANSPORT,
    # Sandbox / subagents — forwarded to vendors that have them; gated
    # elsewhere. Synthesising a sandbox or a subagent orchestrator would
    # be origination, so native-only.
    Feature.SANDBOX: Ring.TRANSPORT,
    Feature.SUBAGENTS: Ring.TRANSPORT,
}


__all__ = ["POLYFILL_RING", "Feature", "Ring"]
