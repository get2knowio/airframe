# Changelog

All notable changes to airframe are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

**Phase 4 of the [implementation plan](docs/implementation-plan.md) is
complete — MCP server references (Tier 2).** Iteration A landed the
protocol-surface shape locks (the :class:`McpServerRef` dataclass,
the new :data:`Feature.TOOLS_MCP_SSE` enum member, and the
``mcp_servers=`` kwarg on every adapter's :meth:`session` signature
gated by the shared capability helper). Iteration B wired
:class:`ClaudeCodeRuntime` — all three transports (stdio, http, sse).
Iteration C wired :class:`CopilotRuntime` — stdio + http natively;
SSE declined per the plan with an actionable "switch to http"
message. Iteration D codifies the Codex + OpenAI-compat permanent
declines with vendor-specific actionable pointers (``~/.codex/config.toml``
for Codex, a future ``OpenAIResponsesRuntime`` for OpenAI-compat),
ships the multi-provider probe, and pins the final transport matrix.
Ready for release as ``v0.7.0`` or to roll straight into Phase 5
(permission, hooks, budget).

### Added (Phase 4, Iteration D — declines, probe, wrap-up)

- `CodexRuntime.session(mcp_servers=<non-empty>)` raises
  :class:`~airframe.errors.UnsupportedFeatureError` with a
  Codex-specific message pointing consumers at the
  ``[[mcp_servers]]`` block in ``~/.codex/config.toml``. Symmetric
  with Phase 3 Iteration D's ``tools=`` decline — the decline is
  **permanent**, not a "wait for the next iteration" gate. The
  :attr:`~airframe.errors.UnsupportedFeatureError.feature` attribute
  carries the *first ref's transport* so consumer code branching on
  ``Feature.TOOLS_MCP_STDIO`` / ``TOOLS_MCP_HTTP`` / ``TOOLS_MCP_SSE``
  still works.
- `OpenAICompatibleRuntime.session(mcp_servers=<non-empty>)` raises
  with an OpenAI-compat-specific message: ``"Chat Completions has no
  MCP-as-tool wire shape; that lives on the Responses API. A future
  ``OpenAIResponsesRuntime`` (separate from this compat family) could
  translate to the Responses-API ``{"type": "mcp", ...}`` tool
  shape."`` Pointer to the future direct-API option. Same
  first-ref-transport on ``.feature`` as Codex.
- `examples/probe_mcp.py` — multi-provider live probe. Registers
  one external stdio MCP server (default:
  ``@modelcontextprotocol/server-everything`` launched via
  ``npx``), prompts the model to list the server's tools (and call
  ``echo`` if available), and prints the
  :class:`~airframe.events.ToolCallStart` /
  :class:`~airframe.events.ToolCallResult` /
  :class:`~airframe.events.TextDelta` /
  :class:`~airframe.events.TurnComplete` sequence from
  :meth:`AgentSession.stream`. Defaults to ``claude`` (broadest
  transport coverage); accepts
  ``--provider claude|github-copilot|codex|opencode`` and
  ``--transport stdio|http|sse``. Adapters that decline surface
  their message verbatim — probe-as-docs, same pattern Phase 3
  used.
- 8 new unit tests in `tests/test_codex_session.py` and
  `tests/test_openai_compatible_session.py` covering per-adapter
  decline messages (both content *and* ``feature=`` attribute),
  per-transport feature mapping, ``mcp_servers=None`` /
  ``mcp_servers=[]`` no-op paths, and the
  ``supports(TOOLS_MCP_*) == False`` capability matrix on both
  adapters.
- `tests/test_features.py::test_mcp_transports_final_matrix` pins
  the Phase-4 endgame table: Claude all three transports, Copilot
  stdio + http, Codex + OpenAI-compat none. The
  ``TOOLS_MCP_IN_PROCESS`` flag stays False on every adapter — Phase
  4 doesn't expose ``transport="in_process"`` on
  :class:`McpServerRef`; the Phase 3 in-process MCP path on Claude
  is internal plumbing for ``tools=`` rather than a user-facing
  capability.

### Changed (Phase 4, Iteration D)

- `CodexRuntime` and `OpenAICompatibleRuntime` no longer call
  :func:`airframe.sessions._check_mcp_servers_supported`. Both now
  raise their vendor-specific declines inline (mirroring the way
  Phase 3 Iteration D unhooked Codex from
  :func:`~airframe.sessions._check_tools_supported`). The shared
  helper still services Claude and Copilot (their transport flags
  flipped True in Iterations B and C mean the helper short-circuits
  to a no-op).
- `docs/implementation-plan.md` — Phase 4 marked ✅ complete; each
  iteration heading marked ✅; dependency-graph annotation updated
  from ``[gated on divergence signal]`` to ``[✅ shipped]``.

### Added (Phase 4, Iteration C — Copilot MCP wiring)

- `_translate_mcp_servers_for_copilot(refs)` builds the dict shape
  :meth:`CopilotClient.create_session`'s ``mcp_servers=`` slot
  expects. **Stdio** maps to ``{"type": "local", "command": <head>,
  "args": <tail>}`` — Copilot's wire enum is ``"local"``, not
  ``"stdio"``, and ``args`` is always emitted (the schema requires
  it). **Http** maps to ``{"type": "http", "url": <url>, "headers":
  <merged>}``. **Sse** raises defensively — the SSE decline is
  enforced at the :meth:`CopilotRuntime.session` boundary so the
  translator should never see one.
- `CopilotAgentSession(mcp_servers=...)` — accepted on the session
  factory; translated lazily at :meth:`_ensure_session` and passed
  via :meth:`CopilotClient.create_session`'s ``mcp_servers=`` kwarg.
  Co-exists with ``tools=`` (FunctionTool) and the forced
  ``submit_result`` tool — the three-way combination passes through
  separate ``create_session`` kwargs, no shadowing.
- `_ensure_session` cache key gains an ``mcp=<fingerprint>``
  fragment (Copilot bakes ``mcp_servers`` at create-session time,
  same as ``tools=`` and ``reasoning_effort``). Refs change → session
  destroy + rebuild.
- `Feature.TOOLS_MCP_STDIO` / `TOOLS_MCP_HTTP` flipped True on
  :attr:`CopilotRuntime.SUPPORTED_FEATURES`. ``TOOLS_MCP_SSE`` stays
  **False per the plan**; refs of that transport surface a
  Copilot-specific :class:`~airframe.errors.UnsupportedFeatureError`
  pointing consumers at the ``http`` transport. The decline runs
  *before* the shared capability gate so the consumer gets the
  actionable message rather than the generic one.
- 12 new unit tests in `tests/test_copilot_session.py` covering
  stdio→``local`` translation (with the always-emit-``args``
  invariant), http with bearer-auth + header merging precedence,
  the SSE-rejection-with-http-hint path, mixed-transport sessions,
  ``tools=`` + ``mcp_servers=`` coexistence, the three-way
  ``submit_result`` + custom tools + MCP combination, the
  no-``mcp_servers=`` ⇒ kwarg-omitted path, cache invalidation on
  refs-change, and the per-flag capability matrix.

### Changed (Phase 4, Iteration C)

- **Shared helpers extracted to `airframe.sessions`.**
  `_mcp_servers_fingerprint(refs)` and `_compose_mcp_headers(ref)`
  moved out of `claude_code.py` so Copilot reuses the same
  secret-free fingerprint logic and the same
  ``Authorization: Bearer`` shorthand-with-caller-override merge.
  Previous Claude callers updated; tests updated to import from the
  new home.
- `tests/test_features.py`:
  `test_non_claude_adapters_still_decline_mcp_transports` and
  `test_session_mcp_servers_kwarg_raises_on_non_claude_adapters`
  evolved into per-adapter expectations now that Copilot also
  declares stdio + http True. New tests
  `test_copilot_declares_stdio_and_http_but_not_sse`,
  `test_codex_and_openai_compat_still_decline_all_mcp_transports`,
  `test_copilot_session_sse_decline_carries_http_hint`, and
  `test_session_mcp_servers_kwarg_raises_on_codex_and_openai_compat`
  pin the Iteration-C matrix.

### Note: SDK observation

While wiring this, we noticed Copilot's
:class:`MCPServerConfigHTTPType` enum exposes both
``HTTP`` and ``SSE`` as valid remote transports — i.e. the Copilot
SDK does have an SSE channel today. The plan was written when this
was not the case and explicitly declines SSE on Copilot. This
iteration follows the plan as written; flipping
:data:`Feature.TOOLS_MCP_SSE` True on Copilot would be a small
follow-up if a downstream consumer needs it (the translator change
is one line: extend the ``"http"`` arm to also accept ``"sse"``;
the gate change is removing the pre-helper SSE decline).

### Added (Phase 4, Iteration B — Claude MCP wiring)

- `_translate_mcp_servers_for_claude(refs)` builds a dict keyed by
  :attr:`McpServerRef.name` where each value is the matching SDK
  TypedDict — :class:`McpStdioServerConfig` /
  :class:`McpHttpServerConfig` / :class:`McpSSEServerConfig`.
  Stdio splits :attr:`McpServerRef.command` (an argv list) into the
  SDK's ``command: str`` + ``args: list[str]``; ``args`` is omitted
  for a one-element command. Http / sse pass :attr:`McpServerRef.url`
  through verbatim. :attr:`McpServerRef.auth_token` becomes
  ``Authorization: Bearer <token>`` (merged with
  :attr:`McpServerRef.headers`; caller-supplied ``Authorization``
  wins on collision).
- `ClaudeCodeSession(mcp_servers=...)` — accepted on the session
  factory. Translated lazily at connect time; the translated dict is
  **merged** with the in-process tools server (Phase 3) into
  :attr:`ClaudeAgentOptions.mcp_servers`. Tool-name collisions with
  the reserved ``airframe_tools`` slot raise
  :class:`ValueError` at connect — no silent shadowing.
- `_mcp_servers_fingerprint(refs)` — deterministic, secret-free
  fingerprint added to :meth:`ClaudeCodeSession._ensure_client`'s
  cache key. Participates from ``name``, ``transport``, ``command``,
  ``url``, and the *sorted keys* of ``headers``; never includes
  header values or ``auth_token``, so rotating a bearer token doesn't
  accidentally invalidate the cache (and sensitive material doesn't
  enter cache identity).
- `_strip_mcp_prefix` generalised from the Phase-3 single-server form
  to a multi-server set the session tracks: the in-process
  ``airframe_tools`` (when ``tools=`` is set) plus every
  :attr:`McpServerRef.name`. Tool calls routed through a recognised
  server come back with the bare name on
  :class:`~airframe.events.ToolCallStart`; unrecognised prefixes pass
  through verbatim per Phase 4 risk note #6.
- Per-server wildcard allowed-tools entry: each external server adds
  ``mcp__<server>__*`` to :attr:`ClaudeAgentOptions.allowed_tools`
  so every tool the server exposes is auto-allowed (parallels the
  per-tool entries the in-process tools server adds).
- `Feature.TOOLS_MCP_STDIO` / `TOOLS_MCP_HTTP` / `TOOLS_MCP_SSE`
  flipped True on :attr:`ClaudeCodeRuntime.SUPPORTED_FEATURES`.
  `TOOLS_MCP_IN_PROCESS` stays False — Phase 4 doesn't expose an
  in-process :class:`McpServerRef` transport; the in-process MCP
  server is internal plumbing for ``tools=``.
- 14 new unit tests in `tests/test_claude_code_session.py` covering
  per-transport translation, bearer/header merging precedence,
  mixed-transport sessions, ``tools=`` + ``mcp_servers=`` coexistence,
  name collisions, cache invalidation on refs-change, fingerprint
  determinism + secret exclusion, external-server prefix stripping,
  and unknown-prefix pass-through.

### Changed (Phase 4, Iteration B)

- `tests/test_features.py`:
  `test_mcp_transport_features_stay_false_iteration_a` and
  `test_session_mcp_servers_kwarg_raises_on_every_adapter` evolved
  into per-adapter expectations now that Claude declares the three
  transport flags True. New tests
  `test_claude_declares_all_three_mcp_transports`,
  `test_non_claude_adapters_still_decline_mcp_transports`,
  `test_session_mcp_servers_kwarg_raises_on_non_claude_adapters`, and
  `test_claude_session_mcp_servers_kwarg_opens_cleanly` pin the
  Iteration-B matrix.
- `test_unwired_features_stay_false` admits
  ``TOOLS_MCP_{STDIO,HTTP,SSE}`` into the
  ``any_adapter_may_support`` set (Claude declares them; the other
  three still decline).

Phase 3 — function tools (Tier 1) — is complete. Iteration A landed
the protocol-surface shape locks; Iteration B wired the OpenAI-
compatible family; Iteration C wired Claude + Copilot through their
respective SDK tool-registration channels; Iteration D codifies
Codex's permanent decline and ships the multi-provider probe.

### Added (Phase 4, Iteration A — protocol scaffolding for MCP refs)

- `airframe.tools.McpServerRef` — frozen+slots dataclass with
  ``name``, ``transport`` (``Literal["stdio", "http", "sse"]``),
  ``command``, ``url``, ``headers``, ``auth_token``.
  ``__post_init__`` validates the transport/field combo: stdio
  requires ``command`` and rejects ``url``; http and sse require
  ``url`` and reject ``command``. Re-exported at the top level as
  ``from airframe import McpServerRef``.
- `Feature.TOOLS_MCP_SSE = "tools_mcp_sse"` — new enum member
  alongside the existing ``TOOLS_MCP_STDIO`` / ``TOOLS_MCP_HTTP`` /
  ``TOOLS_MCP_IN_PROCESS``. Snapshotted in
  `test_feature_string_values_are_stable` from Iteration A onwards
  so a future rename is caught at PR time.
- `airframe.sessions._check_mcp_servers_supported(refs, *,
  adapter_label, supports)` — shared capability gate, symmetric with
  `_check_tools_supported`. Iterates the list, looks up the matching
  ``Feature.TOOLS_MCP_{STDIO,HTTP,SSE}`` per ref, and raises
  :class:`~airframe.errors.UnsupportedFeatureError` on the first
  decline with the specific feature attached. ``supports`` is a
  callable so the helper doesn't need a runtime reference.
- ``mcp_servers: list[McpServerRef] | None = None`` kwarg on
  :meth:`AgentRuntime.session` and every adapter's :meth:`session`.
  Each adapter calls ``_check_mcp_servers_supported`` at the top;
  Iteration A's stopping point is "non-empty list raises immediately,
  empty list / ``None`` opens cleanly." Iterations B–D flip the True
  bits and replace the generic decline messages with vendor-specific
  ones where the decline is *permanent* (Codex's CLI-config pointer,
  OpenAI-compat's "Responses-API only" pointer).
- 11 new unit tests in `tests/test_tools.py` pinning the
  :class:`McpServerRef` shape — frozen+slots, field order, the four
  validation paths (stdio missing command, stdio with url, http/sse
  missing url, http/sse with command), bearer/header pass-through,
  and the top-level export.
- 3 new unit tests in `tests/test_features.py` pinning the matrix at
  the start of Phase 4: every transport flag is False on every
  adapter; ``session(mcp_servers=[<ref>])`` raises with the
  expected ``feature=`` attribute on every adapter for each
  transport; ``mcp_servers=None`` / ``mcp_servers=[]`` opens
  cleanly everywhere.

### Added (Phase 3, Iteration D — Codex decline + probe + docs)

- `CodexRuntime.session(tools=<non-empty>)` raises
  :class:`~airframe.errors.UnsupportedFeatureError` with a
  Codex-specific message pointing consumers at the ``codex`` CLI's
  config file (``~/.codex/config.toml``). Replaces the generic
  shared-helper decline used during Iterations A–C — Codex's
  ``TOOLS_FUNCTION=False`` is **permanent**, not a "wait for the next
  iteration" gate, and the error text now says so explicitly.
  ``tools=None`` / ``tools=[]`` are no-op and don't trigger the
  decline (consistent with the other three adapters' "empty list is
  the same as omitted" handling).
- `examples/probe_tools.py` — multi-provider live probe. Registers a
  tiny ``add(a, b: float) -> float`` :class:`FunctionTool`, prompts
  "what is 17 times 23?", and prints the resulting
  :class:`~airframe.events.ToolCallStart` /
  :class:`~airframe.events.ToolCallResult` /
  :class:`~airframe.events.TextDelta` /
  :class:`~airframe.events.TurnComplete` sequence from
  :meth:`AgentSession.stream`. Defaults to ``opencode`` (simplest
  auth; deterministic client-side tool-loop); accepts
  ``--provider claude|github-copilot|opencode|codex``. The Codex
  branch surfaces the decline message verbatim so the probe doubles
  as documentation for the workaround.
- 3 new unit tests in `tests/test_codex_session.py` covering the
  CLI-config decline message, the ``tools=None`` / ``tools=[]``
  no-op paths, and the capability flag staying False.

### Changed (Phase 3, Iteration D)

- `tests/test_features.py`:
  `test_three_adapters_declare_tools_function` renamed to
  `test_tools_function_universal_except_codex` to pin the final
  Phase 3 matrix: Claude + Copilot + OpenAI-compat all declare
  ``TOOLS_FUNCTION=True``; Codex alone stays False. The matching
  `…_kwarg_raises_…` test now also asserts the decline message
  carries both ``codex`` and ``config`` so message rot doesn't
  silently regress the workaround pointer.
- `CodexRuntime` no longer imports
  :func:`airframe.sessions._check_tools_supported`. The
  Codex-specific decline lives inline in
  :meth:`CodexRuntime.session`; the shared helper still services
  Claude / Copilot / OpenAI-compat (their ``TOOLS_FUNCTION`` flips
  to ``True`` mean the helper short-circuits cleanly).

### Added (Phase 3, Iteration C — Claude + Copilot tool dispatch)

- **Claude:** `_translate_tools_for_claude(tools)` builds an in-process
  MCP server via `claude_agent_sdk.create_sdk_mcp_server(...)`. Each
  `FunctionTool` becomes one `@tool`-decorated coroutine that
  validates incoming args against the user's Pydantic schema, awaits
  the handler, and wraps the return in the SDK's
  `{"content": [{"type":"text","text":...}]}` envelope. Handler
  exceptions / argument-validation failures come back as
  `isError=True` so the model can recover.
- `ClaudeCodeSession(tools=...)` — accepted on the session factory.
  Tools are baked into `ClaudeAgentOptions.mcp_servers` at connect
  time, with matching `mcp__airframe_tools__<name>` entries appended
  to `allowed_tools`. The `_ensure_client` cache key gains a
  `tools=<fingerprint>` fragment so a tools-change between calls
  forces a reconnect.
- `ClaudeCodeSession.stream()` translates `ToolUseBlock` on
  `AssistantMessage` content into `ToolCallStart` and the matching
  `ToolResultBlock` on `UserMessage` content into `ToolCallResult`.
  The `mcp__airframe_tools__` prefix is stripped from `tool_name`
  so consumers see the same `FunctionTool.name` they registered.
- Helpers exported from `airframe.adapters.claude_code`:
  `AIRFRAME_MCP_SERVER_NAME = "airframe_tools"`.
- `Feature.TOOLS_FUNCTION` flipped **True** on
  `ClaudeCodeRuntime.SUPPORTED_FEATURES`.
- **Copilot:** `_translate_one_copilot_tool(ft)` wraps each
  `FunctionTool` as a `copilot.define_tool(...)` registration. The
  `(params, invocation_context) -> result` SDK handler signature is
  adapted to the airframe `(BaseModel) -> Awaitable[Any]` shape by
  ignoring the invocation context. `skip_permission=True` matches the
  existing session-wide `PermissionHandler.approve_all` policy.
- `CopilotAgentSession(tools=...)` — accepted on the session factory.
  Tools are passed via `CopilotClient.create_session(tools=...)` at
  session-creation time. The session cache key (already keyed on
  schema + reasoning-effort) gains a `tools=<fingerprint>` fragment
  so a tools-change forces a rebuild.
- **`submit_result` + custom tools coexistence:** when both `schema=`
  and `tools=` are passed to `CopilotRuntime.session(...)`, the
  adapter prepends the forced `submit_result` tool to the user's
  list. The model sees the structured-output gate in slot 0, then
  the user's tools. Streaming events for `submit_result` are
  filtered out (it's structured-output plumbing, not a user-visible
  tool call).
- `CopilotAgentSession.stream()` translates
  `ToolExecutionStartData` → `ToolCallStart` and
  `ToolExecutionCompleteData` → `ToolCallResult`. Failure responses
  (`success=False`) come back with `is_error=True` and the
  `<code>: <message>` text from `ToolExecutionCompleteError`.
- `Feature.TOOLS_FUNCTION` flipped **True** on
  `CopilotRuntime.SUPPORTED_FEATURES`.
- 8 new unit tests in `tests/test_claude_code_session.py` covering:
  mcp_servers + allowed_tools wiring, the @tool wrapper's validation
  + envelope behaviour, handler-exception → `isError=True`,
  invalid-args → validation error, streaming `ToolCallStart` /
  `ToolCallResult` shape, `ToolResultBlock(is_error=True)`
  propagation, tools-change reconnects, no-tools omits the kwargs.
- 8 new unit tests in `tests/test_copilot_session.py` covering:
  `tools=` reaches `create_session`, the `(params, invocation)`
  adapter signature, `submit_result` + custom-tools coexistence,
  no-tools omits the kwarg, streaming `ToolCallStart` /
  `ToolCallResult` shape, `submit_result` filtered from stream,
  `success=False` → `is_error=True`, tools-change rebuilds.

### Changed (Phase 3, Iteration C)

- `tests/test_features.py` updated for the new asymmetry:
  `test_openai_compatible_declares_tools_function` replaced with
  `test_three_adapters_declare_tools_function` pinning that Claude,
  Copilot, and OpenAI-compat all declare TOOLS_FUNCTION while Codex
  alone stays False; the matching `…_kwarg_raises_…` test now only
  asserts the rejection contract on Codex.

### Added (Phase 3, Iteration B — OpenAI-compat tool round-trip)

- `_translate_tools_for_openai(tools) → list[dict]` — translates
  each `FunctionTool` to the OpenAI wire shape
  (`{"type":"function","function":{"name":…,"description":…,
  "parameters":<json_schema>}}`). Schemas come from
  `FunctionTool.params.model_json_schema()`; the result is computed
  once at session construction and reused on every
  `chat.completions.create()` call.
- `OpenAICompatibleSession(tools=…)` — accepts and caches a
  `list[FunctionTool]` for the session's lifetime. The wire payload
  is precomputed; the per-name lookup table drives dispatch.
- Client-side tool-loop in `_do_execute`: when the response carries
  `tool_calls`, dispatch each handler, append the assistant message
  (with the original `tool_calls` payload) and one `role="tool"`
  reply per call, then re-call. Loops until the model produces a
  final text response or `MAX_TOOL_ITERATIONS=20` round-trips
  elapse — runaway loops surface as
  `RuntimeProtocolError` with a "model kept requesting tools without
  producing a final response" message. Handler exceptions, unknown
  tool names, and Pydantic validation failures all flow back to the
  model as `role="tool"` replies (with `is_error=True` on the
  matching streaming event), so the model can recover on its next
  turn.
- Tool-loop in `stream()`: accumulates `delta.tool_calls` fragments
  by `index` across chunks, then for each call emits `ToolCallStart`
  (with the accumulated `arguments` as `arguments_preview`), invokes
  the handler, emits `ToolCallResult`, and appends the `role="tool"`
  message before the next iteration. Exactly one `TurnComplete` at
  the very end of the user turn — intermediate model turns produce
  tool events but no `TurnComplete` (consistent with the docstring
  clarified in Iteration A).
- `_serialize_tool_output(output)` — JSON-encodes a handler return
  value for the `role="tool"` content field. Strings pass through;
  everything else round-trips via `json.dumps(default=str)`;
  unserialisable types fall back to `repr()`.
- `MAX_TOOL_ITERATIONS = 20` exported from
  `airframe.adapters.openai_compatible`.
- `Feature.TOOLS_FUNCTION` flipped **True** on
  `OpenAICompatibleRuntime.SUPPORTED_FEATURES`. The other three
  adapters keep their Iteration A capability — non-None `tools=`
  still raises `UnsupportedFeatureError` until Iterations C and D
  land.
- 13 new unit tests in `tests/test_openai_compatible_session.py`
  covering: single-tool round-trip, parallel tool calls (multiple
  per assistant message), handler-exception → `is_error` recovery,
  unknown-tool-name → `is_error` message, iteration cap raising
  `RuntimeProtocolError`, buffer rollback on mid-loop failure,
  no-tools sessions omit the `tools=` kwarg, streaming
  `ToolCallStart` / `ToolCallResult` event shape, streaming handler
  exception → `is_error=True`, and streaming iteration cap.

### Changed (Phase 3, Iteration B)

- `tests/test_features.py::test_unwired_features_stay_false` now
  permits `TOOLS_FUNCTION` to vary per adapter (the OpenAI-compat
  family flipped it True; the other three are still False).
- `test_no_adapter_declares_tools_function_yet` replaced with
  `test_openai_compatible_declares_tools_function` pinning the new
  asymmetry; `test_session_tools_kwarg_raises_unsupported_feature`
  replaced with `…_on_unwired_adapters` which exempts OpenAI-compat
  from the rejection contract.
- `OpenAICompatibleSession.execute()` and `.stream()` now snapshot
  the message buffer length up front and roll back to that length
  on any failure, replacing the older "pop one user message if it
  was the last entry" logic. The tool-loop may append multiple
  intermediate messages, so the single-pop rollback would have left
  intermediate state behind.

### Added (Phase 3, Iteration A — protocol scaffolding)

- `airframe.tools.FunctionTool` — frozen+slots dataclass with
  ``name``, ``description``, ``params: type[BaseModel]``, and
  ``handler: Callable[[BaseModel], Awaitable[Any]]``. The
  ``handler`` signature is the **shape lock** for Phase 3: parsed
  Pydantic in, JSON-serialisable Any out. Matches Copilot's
  ``define_tool`` and LangChain's ``@tool`` decorator.
- `tools: list[FunctionTool] | None = None` kwarg on
  :meth:`AgentRuntime.session` Protocol and every adapter's
  ``session()`` (Claude / Copilot / Codex / OpenAI-compat).
- `airframe.sessions._check_tools_supported(tools, *, adapter_label,
  feature_supported)` — shared helper every adapter calls at the
  top of ``session()``. Raises
  :class:`~airframe.errors.UnsupportedFeatureError` with
  ``feature=Feature.TOOLS_FUNCTION`` on a non-None list while the
  adapter's capability flag is False — the same "no silent
  fallback" pattern the prompt-parts helper uses.
- Top-level export: `FunctionTool`.
- 6 new tests in `tests/test_tools.py` pinning the dataclass shape
  (frozen+slots, field order, handler signature, public export
  path).
- 2 new feature-matrix tests: ``TOOLS_FUNCTION`` stays False on every
  adapter; ``session(tools=...)`` raises ``UnsupportedFeatureError``
  on every adapter.

### Changed (Phase 3, Iteration A)

- `TurnComplete` docstring clarified: one ``TurnComplete`` per
  *user turn*, not per *model turn*. Tool round-trips (Phase 3)
  produce intermediate ``ToolCallStart`` / ``ToolCallResult`` pairs
  but only one trailing ``TurnComplete`` with the final model
  turn's result.


---

## [0.5.0] — 2026-05-17

The biggest jump since v0.3.0: bundles both **Phase 1**
(`AgentSession` + streaming + cancel + session resume) and **Phase 2**
(`thinking=` + vision / file input) of the
[implementation plan](docs/implementation-plan.md). No 0.4.x was cut
— Phase 1 + Phase 2 land together to mark a clean break before
Phase 3 (function tools).

Headline additions:

- **`AgentSession`** — the new hinge surface.
  `runtime.session(...)` returns a bespoke per-vendor session that
  owns its own lifecycle; `execute()` becomes documented sugar for
  `runtime.session().execute() + close()`.
- **`session.stream()`** — `AsyncIterator[RuntimeEvent]` with
  `TextDelta` / `ReasoningDelta` / `TurnComplete` events across
  every adapter.
- **`session.cancel()`** — vendor-native interrupt on every adapter.
- **`session(resume=<id>)`** — server-side session resume on the
  three SDK adapters (Claude / Copilot / Codex).
- **`thinking=`** — universal `REASONING_EFFORT` kwarg on every
  adapter; Claude additionally honours
  `thinking={"budget_tokens": N}` via `REASONING_BUDGET_TOKENS`.
- **`prompt: list[PromptPart]`** — polymorphic prompt with
  `ImageInput` / `FileInput` parts. Vision wired on every adapter;
  files wired on Claude / Copilot / Codex (OpenAI-compat stays
  False — varies wildly across compat vendors).
- **Reasoning-token telemetry** on `CostRecord.reasoning_tokens`
  for every adapter that exposes the counter.

Iteration-by-iteration detail follows. Phase 2 iterations land
first (most recent), then Phase 1 retrospective.

### Phase 2 — Inputs and reasoning

### Added (Phase 2, Iteration A — protocol scaffolding)

- `airframe.thinking.ThinkingMode` — union value type for the
  ``thinking=`` kwarg. Variants: ``None`` (default), literal
  ``ReasoningEffort`` (``"minimal"|"low"|"medium"|"high"``), the
  Claude-style ``dict`` (currently documented as
  ``{"budget_tokens": int}``), and ``"disabled"`` (explicit-off
  sentinel). The shape is the **ADR-006 lock** — dict literal for
  ``budget_tokens`` rather than a dedicated dataclass matches
  Anthropic's wire format and what consumers already write.
- `airframe.thinking.ReasoningEffort` — exported `Literal` for the
  portable effort levels.
- `airframe.inputs.ImageInput` — frozen+slots dataclass with
  ``path | bytes_ | url`` + optional ``media_type``. Post-init
  enforces at-least-one-source.
- `airframe.inputs.FileInput` — frozen+slots dataclass with
  ``path`` + optional ``media_type``.
- `airframe.inputs.PromptPart` = ``str | ImageInput | FileInput``.
- `airframe.inputs.Prompt` = ``str | list[PromptPart]`` — back-compat
  with the v0–Phase-1 bare-``str`` shape.
- `AgentRuntime.execute()`, `AgentSession.execute()`, and
  `AgentSession.stream()` all accept the new ``thinking=`` kwarg and
  the polymorphic ``prompt: Prompt`` parameter on the Protocol.
- Top-level exports: `ThinkingMode`, `ReasoningEffort`, `ImageInput`,
  `FileInput`, `Prompt`, `PromptPart`.
- 19 new unit tests across `tests/test_thinking.py` and
  `tests/test_inputs.py` pinning the union shape and the dataclass
  field sets.

### Changed (Phase 2, Iteration A)

- All four adapters' `execute()` / `session.execute()` /
  `session.stream()` signatures accept the new kwargs:
  - ``thinking=`` is accepted but ignored — Iteration B wires it
    per-adapter (Claude → ``ClaudeAgentOptions.thinking``; Copilot →
    ``reasoning_effort`` on session; Codex →
    ``ThreadOptions.model_reasoning_effort``; OpenAI-compat →
    ``chat.completions.create(reasoning_effort=...)``).
  - List-shaped prompts raise `UnsupportedFeatureError` via the
    shared `airframe.sessions._coerce_prompt_or_raise` helper —
    Iteration C wires per-adapter image / file routing and flips
    `Feature.VISION_INPUT` / `Feature.FILE_INPUT`.
- `tests/test_agent_session_protocol.py` updated for the new
  parameter order (`schema`, `thinking`, `timeout`).

### Added (Phase 2, Iteration B — `thinking=` wiring)

- `_translate_thinking_for_openai`, `_translate_thinking_for_claude`,
  `_translate_thinking_for_copilot`, `_translate_thinking_for_codex`
  — per-adapter helpers that map `ThinkingMode` onto each vendor's
  native channel:
  - OpenAI-compat → `chat.completions.create(reasoning_effort=...)`.
  - Claude Code → `ClaudeAgentOptions.effort` for the literal
    string, `ClaudeAgentOptions.thinking={"type": "enabled",
    "budget_tokens": N}` for the dict shape, and
    `thinking={"type": "disabled"}` for explicit-off.
  - Copilot → `CopilotClient.create_session(reasoning_effort=...)`
    / `resume_session(reasoning_effort=...)`.
  - Codex → `ThreadOptions.modelReasoningEffort`.
- `Feature.REASONING_EFFORT` flipped True on **all four adapters**.
- `Feature.REASONING_BUDGET_TOKENS` flipped True on `ClaudeCodeRuntime`
  only — the dict shape is Anthropic-only. The other three adapters
  raise `UnsupportedFeatureError` at translation time when handed a
  dict.
- `CostRecord.reasoning_tokens` populated from each vendor's counter:
  Anthropic `thinking_tokens`, OpenAI
  `completion_tokens_details.reasoning_tokens`, Copilot
  `usage.reasoning_tokens`. (Codex's `Usage` doesn't expose a
  reasoning counter — stays 0.)
- 25 new session-level unit tests across the four
  `tests/test_*_session.py` files pinning per-adapter wire shape, the
  no-coercion / coerce-minimal-to-low behaviors, the dict-on-non-Claude
  decline path, and the cache-invalidation-on-thinking-change semantics
  for the three SDK adapters that bake effort at session-creation time.
- 2 new feature-matrix tests pin `REASONING_EFFORT` as universal and
  `REASONING_BUDGET_TOKENS` as Claude-only.

### Changed (Phase 2, Iteration B)

- The Claude / Copilot / Codex session caches now key on
  `(schema, thinking)` (Codex: `(model, thinking)`) — a thinking
  change between turns rebuilds the underlying vendor session /
  thread because each vendor bakes the effort at connect /
  create-session / start-thread time. OpenAI-compat is per-call, so
  its session is stateless on this axis.
- `"minimal"` coerces to `"low"` on Claude and Copilot at debug-log
  level (no `"minimal"` rung in either vendor's enum). Not silent —
  the log records the coercion. OpenAI-compat and Codex accept
  `"minimal"` natively.

### Added (Phase 2, Iteration C — vision + file input)

- `airframe.sessions._split_prompt_parts(prompt, *, adapter_label,
  supports_vision, supports_file)` — shared helper returning
  `(text, images, files)`. Path-only in v0;
  `ImageInput.bytes_`/`url` raise `UnsupportedFeatureError` with a
  "materialise to disk and pass path=" message; unsupported parts
  raise with the corresponding `Feature` attribute.
- `_build_user_content` (OpenAI-compat) — promotes a user message's
  `content` from a bare string to the content-parts list shape
  (`[{"type":"text",...},{"type":"image_url","image_url":{"url":"data:..;base64,..."}}]`)
  when images are present. Path bytes are base64-encoded inline; the
  `media_type` is taken from `ImageInput.media_type` or sniffed via
  `mimetypes.guess_type` (defaults to `image/png`).
- `_build_codex_input` (Codex) — switches the `Thread.run` /
  `Thread.run_streamed` input from a string to a
  `list[TextInput | LocalImageInput]` when images are present.
  `FileInput` parts append `Attached file: <path>` hints to the
  prompt text (Codex's working-directory sandbox lets the agent open
  them with its built-in shell tools).
- `_build_copilot_attachments` (Copilot) — translates both
  `ImageInput` and `FileInput` into Copilot's
  `FileAttachment` TypedDict (`{"type":"file","path":...}`), passed
  via `session.send_and_wait(prompt, attachments=...)`.
- `_build_claude_prompt` (Claude Code) — appends an `Attached files
  (use the Read tool to access):` block to the prompt text and adds
  `"Read"` to `ClaudeAgentOptions.allowed_tools`. The
  `_ensure_client` cache key grows a `has_attachments` fragment so a
  no-attachment ↔ with-attachment switch reconnects with the right
  `allowed_tools`.
- `Feature.VISION_INPUT` flipped True on **all four adapters**.
- `Feature.FILE_INPUT` flipped True on **Claude / Copilot / Codex**;
  stays False on OpenAI-compat (the roadmap calls out that file
  routing varies wildly across compat vendors).
- 22 new session-level unit tests across the four
  `tests/test_*_session.py` files pinning the per-adapter wire shape,
  the path-only constraint (bytes_/url decline), the
  `FileInput`-decline path on OpenAI-compat, and Claude's
  cache-invalidation-on-attachments-change semantics.
- 2 new feature-matrix tests pin `VISION_INPUT` as universal and
  `FILE_INPUT` as Claude/Copilot/Codex-only.

### Changed (Phase 2, Iteration C)

- All four adapters now call `_split_prompt_parts` instead of
  `_coerce_prompt_or_raise`; the old helper is kept exported for
  third-party adapters that haven't wired any input capability yet
  but is no longer used in-tree.
- `ClaudeCodeSession._ensure_client` takes a new
  `has_attachments: bool` kwarg and uses it both in the cache key
  and to set `ClaudeAgentOptions.allowed_tools=["Read"]`.
- `CopilotAgentSession.execute` / `.stream` now pass
  `attachments=...` to `CopilotSession.send_and_wait` (previously
  unused).
- `CodexAgentSession.execute` / `.stream` now build their `Thread.run`
  input via `_build_codex_input` rather than passing the prompt
  string directly.

### Added (Phase 2, Iteration D — wrap-up: bytes/URL + probes)

- `ImageInput(bytes_=)` and `ImageInput(url=)` support on
  **OpenAI-compat** (native). `_build_user_content` now handles all
  three variants: ``path=`` reads + base64-encodes, ``bytes_=``
  base64-encodes inline (defaults media type to ``image/png`` when
  omitted), ``url=`` passes through as-is for the vendor to fetch.
- `ImageInput(bytes_=)` support on **Copilot** via
  :class:`BlobAttachment` (``{"type":"blob","data":<b64>,
  "mimeType":...}``). `_build_copilot_attachments` now routes
  ``path=`` → `FileAttachment` and ``bytes_=`` → `BlobAttachment`.
  ``url=`` still raises — Copilot's SDK has no URL channel; the
  error message recommends fetching the image and passing
  ``bytes_=`` instead.
- Sharper bytes/URL decline message on **Claude** and **Codex**:
  both now point the consumer to ``tempfile.NamedTemporaryFile`` +
  ``path=`` as the workaround. Neither vendor's SDK has a native
  bytes/URL channel for vision (Claude uses the Read tool which
  reads from disk; Codex uses `LocalImageInput` which is path-only).
- `examples/probe_thinking.py` — live-vendor probe for ``thinking=``.
  Multi-provider, default ``--effort high``, reports
  ``reasoning_tokens`` from the resulting `CostRecord`.
- `examples/probe_vision.py` — live-vendor probe for `ImageInput`.
  Defaults to OpenAI-compat (the only adapter that natively handles
  every variant); takes ``--variant path|bytes|url`` and bundles a
  67-byte inline 1×1 PNG so the probe runs without external assets.
- 8 new session-level unit tests across `test_openai_compatible_session.py`
  and `test_copilot_session.py` covering the bytes / URL paths.

### Changed (Phase 2, Iteration D)

- `_split_prompt_parts` no longer rejects `ImageInput(bytes_=)` /
  `ImageInput(url=)`. Per-vendor variant support differs, so the
  shared helper only gates the feature *category* (vision vs file);
  each adapter's content builder decides which variants it can
  natively serve.
- Iteration C's "path-only in v0" tests on OpenAI-compat and Copilot
  were replaced with positive tests for the new variants. The
  remaining "path-only" assertions on Claude and Codex now also
  assert the error message includes the ``tempfile`` /  ``path=``
  workaround hint.

### Phase 1 — AgentSession + streaming + cancel

The "hinge" phase from the
[implementation plan](docs/implementation-plan.md) — introduces
`AgentSession`, streaming, cancellation, and session resume. Shipped
in iterations A–H over the development cycle; bundled here as part
of the v0.5.0 release.

### Added (Phase 1, Iteration H — wrap-up: probes + architecture refresh)

The last Phase 1 deliverables. No more SDK glue, no more API
changes; just docs + examples that exercise everything the prior
iterations shipped.

- `examples/probe_streaming.py` — live-vendor probe for
  `session.stream()`. Iterates `async for event in
  sess.stream(prompt)`, prints `TextDelta` content live,
  `ReasoningDelta` on a dedicated line, and a summary of the
  trailing `TurnComplete`. Defaults to `claude` but accepts
  `--provider {claude,github-copilot,codex,opencode}`.
- `examples/probe_session_resume.py` — two-turn probe for the
  resume path. Opens a session, captures `session.id` after the
  first turn, closes; reopens with `resume=<id>` and verifies the
  second turn shows continuity (the model recalls a "secret word"
  from the prior turn). Rejects OpenAI-compat with a clear error
  since chat-completions has no server-side session.
- `docs/architecture.md` refreshed for the runtime/session split:
  new staircase diagram showing both protocols + adapter layer,
  runtime-vs-session ownership table, streaming `RuntimeEvent`
  taxonomy, end-of-Phase-1 capability matrix, per-adapter
  operational notes updated to reflect the bespoke sessions, and
  the new `unwrap()` split (runtime-level vs session-level types).

### Changed (Phase 1, Iteration G — `execute()` is sugar over `session()`)

**Breaking** — but breaking pre-1.0 in service of the cleaner shape
the plan called for in Phase 1.

- `AgentRuntime.execute(prompt, ...)` is now documented sugar for
  `runtime.session(system=..., model=...).execute(prompt, schema=...,
  timeout=...)` + `close()`. Single-turn, ephemeral. Consumers wanting
  context warmth across calls open a session explicitly and reuse it.
- **Behaviour change:** the runtime no longer caches per-conversation
  state across consecutive `execute()` calls. Each call opens a fresh
  session, runs one turn, and tears it down. For SDK-based adapters
  that spawn a CLI subprocess per session (Claude Code, Copilot,
  Codex), this means a fresh subprocess per `execute()`. Consumers
  that relied on the previous warm-cache behaviour should migrate to
  the session API:

  ```python
  # before — execute() reused state across calls
  for prompt in prompts:
      result = await runtime.execute(prompt, system=PROMPT)
  ```

  ```python
  # after — explicit session reuses the underlying SDK handle
  sess = runtime.session(system=PROMPT)
  try:
      for prompt in prompts:
          result = await sess.execute(prompt)
  finally:
      await sess.close()
  ```

- `AgentSession.unwrap(cls)` added to the protocol. Vendor session
  types moved off the runtime onto the session:
  - `ClaudeCodeSession.unwrap(ClaudeSDKClient)` — was
    `ClaudeCodeRuntime.unwrap(ClaudeSDKClient)`.
  - `CopilotAgentSession.unwrap(CopilotSession)` — was
    `CopilotRuntime.unwrap(CopilotSession)`.
  - `CodexAgentSession.unwrap(Thread)` — was
    `CodexRuntime.unwrap(Thread)`.
  Runtime-level vendor types (`CopilotClient`, `Codex`, `AsyncOpenAI`)
  stay on the runtime — those are runtime-wide, long-lived.
- `runtime.unwrap(SessionType)` now raises with a redirect message
  pointing callers to `session.unwrap(SessionType)`.
- `runtime.reset()` is a no-op on every adapter — there's nothing
  scope-bound to drop on the runtime any more. Kept for protocol
  completeness and back-compat.
- `ClaudeCodeRuntime` is now genuinely sessionless: `close()` is also
  a no-op (no SDK client, no HTTP client, no subprocess to release).
- `CopilotRuntime.close()` still releases the long-lived
  `CopilotClient` (runtime-owned).
- `CodexRuntime.close()` still drops the cached `Codex` reference
  (cheap, but back-compat with pre-G behaviour).

### Removed (Phase 1, Iteration G)

- `ClaudeCodeRuntime._ensure_client`, `_query_and_drain` — the per-
  conversation logic moved into `ClaudeCodeSession`.
- `CopilotRuntime._ensure_session`, `_on_event`, runtime-level
  capture slots (`_captured_payload`, `_captured_usage`,
  `_captured_error`, `_last_assistant_message`, `_session`,
  `_session_key`) — all moved into `CopilotAgentSession`.
- `CodexRuntime._ensure_thread`, `_thread`, `_thread_key` — all
  moved into `CodexAgentSession`.

### Added (Phase 1, Iteration G)

- `AgentSession.unwrap(cls)` on the protocol with full docstring
  covering per-adapter mappings.
- `_ThinAgentSession.unwrap(cls)` — identity-cast only (no vendor
  state); raises with a helpful message pointing callers to
  `runtime.unwrap()` for vendor objects.
- `tests/test_unwrap.py` extended with three new session-level
  pre-execute tests covering the new vendor-type unwrap path
  through the session.

### Added (Phase 1, Iteration F — Codex end-to-end)

Fourth and final per-vendor session. All four in-tree adapters now
have bespoke `AgentSession` implementations; the
`_ThinAgentSession` placeholder is no longer reachable from any
adapter's `session()` factory (kept in `airframe.sessions` for
third-party adapters that want a quick start).

- `airframe.adapters.codex.CodexAgentSession` — bespoke
  `AgentSession` owning one `Thread` for its lifetime.
  `system` / `model` / `resume` are session-fixed and baked into
  `Codex.start_thread` (or `resume_thread`) at first use. Schema can
  vary per turn — the Codex SDK puts `outputSchema` on `TurnOptions`
  (not `ThreadOptions`), so the same thread serves both plain-text
  and structured turns without rebuild.
- Real streaming via `Thread.run_streamed`. Per-item tail tracking
  on `ItemUpdatedEvent` / `ItemCompletedEvent` keeps `TextDelta`
  instances appendable (concatenated deltas reconstruct the full
  message text, matching the contract OpenAI-compat's chunk-stream
  sets). `AgentMessageItem` → `TextDelta`; `ReasoningItem` →
  `ReasoningDelta`; `TurnCompletedEvent` → trailing `TurnComplete`
  with cost from `Usage`; `TurnFailedEvent` raises through the
  runtime's classifier.
- Native session resume — ``session(resume=<thread_id>)`` forwards
  the ID to `Codex.resume_thread`; `AgentSession.id` is seeded from
  `resume=` and updated from `Thread.id` after the first turn (live
  `thread.started` event surfaces the id on fresh threads).
- Native cancellation via per-turn `AbortController` plumbed into
  `TurnOptions.signal`. `cancel()` calls `controller.abort()`; the
  awaiting turn raises `AbortError`, surfaced as
  `RuntimeCancelledError`. Honours no-op-when-idle via `_in_flight`.
- `Feature.STREAMING`, `Feature.SESSION_RESUME`, and `Feature.CANCEL`
  flipped to `True` on `CodexRuntime.SUPPORTED_FEATURES`. With this,
  every in-tree adapter declares streaming + cancel; the three
  SDK-based adapters (Claude Code, Copilot, Codex) all declare
  session resume; OpenAI-compat is the only adapter without
  `SESSION_RESUME` (chat-completions has no server-side session).
- 17 new unit tests in `tests/test_codex_session.py` covering the
  factory, capability surface, resume-id seeding + `resume_thread`
  routing, fresh thread via `start_thread`, thread reuse across
  turns, schema-can-vary-without-rebuild, system-prompt
  concatenation, appendable streaming deltas, reasoning deltas,
  structured-output parsing during streaming, cancellation in
  execute + mid-stream, `TurnFailedEvent` raising, and `close()`
  lifecycle.

### Changed (Phase 1, Iteration F)

- `tests/test_features.py` extended: replaced
  `test_codex_still_has_only_structured_output` with
  `test_codex_declares_streaming_resume_and_cancel`, and added
  `test_three_sdk_adapters_declare_session_resume` to pin the
  end-of-Phase-1 capability matrix (the three SDK adapters declare
  resume; OpenAI-compat doesn't).

### Added (Phase 1, Iteration E — Copilot end-to-end)

Third per-vendor session. Copilot now flips the same three feature
bits Claude does — streaming, native server-side resume, and
cooperative cancellation are all native to the SDK.

- `airframe.adapters.copilot.CopilotAgentSession` — bespoke
  `AgentSession` owning a `CopilotSession` for its lifetime.
  `system` / `model` / `resume` are session-fixed and baked into
  `CopilotClient.create_session` (or `resume_session`) at creation;
  schema can vary per turn — the session is destroyed and rebuilt
  when the schema fingerprint changes since the `submit_result` tool
  is baked in at session-creation time.
- Real streaming via per-session `session.on(handler)` subscription
  pushing deltas through an `asyncio.Queue` that the generator
  drains:
  - `AssistantMessageDeltaData` → `TextDelta`.
  - `AssistantReasoningDeltaData` → `ReasoningDelta`.
  - Uses `loop.call_soon_threadsafe` since the SDK dispatches
    handlers off its own thread.
- Native session resume — ``session(resume=<session_id>)`` forwards
  the ID to `CopilotClient.resume_session`; `AgentSession.id` is
  seeded with the resume ID and updated to the live
  `CopilotSession.session_id` once the underlying session is built.
- Native cancellation via `CopilotSession.abort()`; honours the
  no-op-when-idle contract via an `_in_flight` flag.
- `Feature.STREAMING`, `Feature.SESSION_RESUME`, and `Feature.CANCEL`
  flipped to `True` on `CopilotRuntime.SUPPORTED_FEATURES`. Only
  `CodexRuntime` is left on the Phase 0 single-bit declaration.
- 15 new unit tests in `tests/test_copilot_session.py` covering the
  factory, capability surface, resume-id seeding + `resume_session`
  routing, `create_session` path for fresh sessions, plain-text +
  structured execute, client reuse across same-schema turns, rebuild
  on schema change, streaming TextDelta / ReasoningDelta extraction,
  `abort()` invocation on cancel, idle no-op, and `close()` lifecycle
  (destroys the vendor session but leaves the runtime's client alive).

### Changed (Phase 1, Iteration E)

- `tests/test_features.py` extended: added
  `test_copilot_declares_streaming_resume_and_cancel` pinning the new
  flips; renamed `test_copilot_and_codex_still_have_only_structured_output`
  to `test_codex_still_has_only_structured_output` since only Codex
  remains on the Phase 0 declaration.

### Added (Phase 1, Iteration D — Claude Code end-to-end)

Second per-vendor session. `ClaudeCodeRuntime` now flips three feature
bits at once — Claude is the most fully-featured of the four families
for session-shaped APIs (streaming, server-side resume, cooperative
cancellation are all native to the SDK).

- `airframe.adapters.claude_code.ClaudeCodeSession` — bespoke
  `AgentSession` owning a `ClaudeSDKClient` for its lifetime. ``system``
  / ``model`` / ``resume`` are session-fixed and baked into
  `ClaudeAgentOptions` at connect; ``schema`` can vary per turn — the
  client reconnects when the schema fingerprint changes since
  ``output_format`` is connect-time-bound.
- Real streaming via ``include_partial_messages=True``. The session
  translates Anthropic stream events into airframe events:
  - ``content_block_delta`` with ``text_delta`` → `TextDelta`.
  - ``content_block_delta`` with ``thinking_delta`` → `ReasoningDelta`.
  - Fallback: `TextBlock` content on `AssistantMessage` emits a
    `TextDelta` when StreamEvents didn't deliver text (older CLI
    versions / non-streaming content paths).
- Native session resume — ``session(resume=<session_id>)`` forwards
  the ID into `ClaudeAgentOptions.resume`; the SDK materialises the
  prior conversation from local-disk session store on connect.
  `AgentSession.id` is seeded with the resume ID immediately and is
  updated from each `ResultMessage.session_id` after every turn.
- Native cancellation via `ClaudeSDKClient.interrupt()` plus
  `asyncio.Task.cancel()` on the wrapping execute task. `cancel()`
  honours the no-op-when-idle contract by tracking an `_in_flight`
  flag.
- `Feature.STREAMING`, `Feature.SESSION_RESUME`, and `Feature.CANCEL`
  flipped to `True` on `ClaudeCodeRuntime.SUPPORTED_FEATURES`. Copilot
  + Codex still report `False` on all three until their iterations.
- 14 new unit tests in `tests/test_claude_code_session.py` covering
  the factory, capability surface, resume-id seeding, options forwarding,
  client reuse across same-schema turns, reconnect on schema change,
  streaming TextDelta / ReasoningDelta extraction, AssistantMessage
  fallback, cancellation (with `interrupt()` awaited), idle no-op,
  and `close()` lifecycle.

### Changed (Phase 1, Iteration D)

- `tests/test_features.py` divergence assertions extended: the
  `any_adapter_may_support` set now includes `SESSION_RESUME`;
  added `test_claude_code_declares_streaming_resume_and_cancel` and
  `test_copilot_and_codex_still_have_only_structured_output` to pin
  exactly which adapter declares what at the end of Iteration D.

### Added (Phase 1, Iteration C — OpenAI-compat end-to-end)

First per-vendor session implementation replacing the Iteration B
placeholder. The OpenAI-compatible family (`OpenCodeZenRuntime` today;
future Together / Groq / Fireworks / OpenRouter subclasses
automatically inherit) now serves real multi-turn conversations,
streaming, and cancellation.

- `airframe.adapters.openai_compatible.OpenAICompatibleSession` —
  bespoke `AgentSession` with a client-side `messages=[]` buffer
  accumulating user + assistant turns across calls. Failures
  (including cancellation) roll back the user message so retries
  send clean history.
- Real streaming via `stream=True` on `chat.completions.create()` plus
  `stream_options={"include_usage": True}` so the trailing
  `TurnComplete` carries a populated `CostRecord`. Per-chunk text
  becomes `TextDelta`; the model's `finish_reason` lands on the
  result.
- Cancellation: `session.cancel()` aborts both `execute()` (wrapping
  `asyncio.Task.cancel()`) and `stream()` (closes the underlying
  `AsyncStream` + sets a flag the generator checks between yields).
  In both cases the message buffer rolls back.
- `Feature.STREAMING` and `Feature.CANCEL` flipped to `True` on
  `OpenAICompatibleRuntime.SUPPORTED_FEATURES`. The other three adapter
  families still report `False` until each lands its own iteration.
- `session(resume=...)` raises `UnsupportedFeatureError` on
  chat-completions vendors per the plan (no server-side session).
  Subclasses backed by the Responses API can override.
- 15 new unit tests in `tests/test_openai_compatible_session.py`
  exercising multi-turn buffer, streaming chunk shape, structured-
  output round-trip during streaming, cancellation, and `close()`
  semantics — all with a mocked `openai` SDK.

### Changed (Phase 1, Iteration C)

- `tests/test_features.py::test_phase_0_only_structured_output_is_true`
  renamed to `test_unwired_features_stay_false` — the assertion now
  permits per-adapter divergence on features whose APIs have landed
  (STRUCTURED_OUTPUT_JSON_SCHEMA, STREAMING, CANCEL) while still
  guarding against any adapter declaring a not-yet-wired capability.
  Adds a companion `test_openai_compatible_declares_streaming_and_cancel`
  pinning the new flips for this family and asserting the other three
  still report `False`.
- `OpenAICompatibleRuntime.execute()` (legacy single-call entrypoint)
  now shares the `_build_response_format()` helper with the new
  session — same JSON-schema wire shape, no behavioural change.
- TCK contract `test_session_resume_not_implemented_until_feature_flips`
  now also accepts `UnsupportedFeatureError` (Iteration C's bespoke
  signal for "this capability will *never* land here") in addition to
  Iteration B's `NotImplementedError` ("hasn't been wired yet").

### Added (Phase 1, Iteration B — `session()` factory + thin sessions)

- `AgentRuntime.session(*, resume=None, system=None, model=None,
  provider_options=None) -> AgentSession` factory on the protocol.
  Every built-in adapter wires it. Iteration B picks
  single-active-session per runtime per ADR-004; concurrent sessions
  can be added later additively.
- `airframe.sessions._ThinAgentSession` — Phase 1 Iteration B's shared
  thin wrapper: forwards `execute()` to the underlying runtime,
  synthesises `stream()` as a single `TurnComplete`, no-ops `cancel()`
  when no turn is in flight, idempotent `close()`. Marked private —
  third-party adapters target `AgentSession` directly. Per-adapter
  bespoke sessions (with real per-vendor streaming, native session
  resume, native cancellation) replace this in Iteration C+ as each
  adapter's `Feature` bits flip on.
- `airframe.UnsupportedFeatureError` — capability-decline signal.
  Phase 1+ companion to `UnsupportedBindingError`: a binding mismatch
  is "this adapter doesn't serve this `(provider, model)`"; an
  unsupported feature is "this adapter serves the binding, but the
  capability you asked for is not wired." Honours the implementation
  plan's "no silent fallbacks" principle. Carries an optional
  `feature` attribute naming the declined capability.
- Six new conformance contracts in `airframe.testing.contracts`:
  `test_session_factory_returns_agent_session`,
  `test_session_close_is_idempotent`,
  `test_session_close_on_fresh_session_is_safe`,
  `test_session_cancel_when_idle_is_noop`,
  `test_session_stream_is_async_generator`,
  `test_session_resume_not_implemented_until_feature_flips`. Every
  in-tree conformance file imports them; third-party adapters get the
  same coverage automatically.

### Changed (Phase 1, Iteration B)

- `AgentSession.cancel()` docstring tightened: when a turn is in
  flight, adapters not declaring `Feature.CANCEL` raise
  `UnsupportedFeatureError` (was: "raise `RuntimeProtocolError`",
  which conflated capability gaps with adapter bugs). When no turn is
  in flight, the call remains a no-op regardless of capability.

### Phase 1 status

Phase 1 is done. Per the [implementation
plan](docs/implementation-plan.md):

- Protocol surface (`AgentSession`, `RuntimeEvent`,
  `runtime.session()`, `session.unwrap()`) — ✓
- Per-adapter bespoke sessions (Claude Code, Copilot, Codex,
  OpenAI-compat) with real streaming, resume, and cancellation
  wired per-vendor — ✓
- `Feature.STREAMING` / `Feature.SESSION_RESUME` / `Feature.CANCEL`
  flipped on every adapter that supports them — ✓
- `AgentRuntime.execute()` refactored into sugar over `session()` — ✓
- TCK conformance contracts extended with the session-level surface — ✓
- Live-vendor probes (`probe_streaming.py`, `probe_session_resume.py`)
  + architecture-doc refresh — ✓

Next: Phase 2 (inputs & reasoning) — `thinking=` kwarg on
`session.execute()`, polymorphic `prompt` for vision / file inputs,
populated `CostRecord.reasoning_tokens`.

### Added (Phase 1, Iteration A — protocol scaffolding)

- `airframe.events` module — five frozen dataclasses (`TextDelta`,
  `ReasoningDelta`, `ToolCallStart`, `ToolCallResult`, `TurnComplete`)
  and the `RuntimeEvent = TextDelta | ReasoningDelta | ToolCallStart
  | ToolCallResult | TurnComplete` discriminated union. ADR-003 shape
  lock — the variant set and field shapes are public surface and
  snapshotted in `tests/test_events.py`. Adding a variant later is
  safe (consumers branch with a wildcard); renaming or removing is
  a major-version break.
- `AgentSession` Protocol in `airframe.protocol` — the multi-turn
  conversation handle scoped to one runtime. Declares
  `id: str | None`, `execute()`, `stream()`, `cancel()`, `close()`.
  ADR-004 picks single-active-session per runtime: ``runtime.session()``
  returning a second handle before the first is ``close()``'d
  invalidates the first. Going concurrent later is additive; going
  from concurrent to single is breaking, so Phase 1 picks single.
- New top-level exports: `AgentSession`, `RuntimeEvent`, `TextDelta`,
  `ReasoningDelta`, `ToolCallStart`, `ToolCallResult`,
  `TurnComplete`.

### Deferred (Iteration A — landed or moved out in Iteration B)

The factory, the structural TCK additions, and the `_ThinAgentSession`
landed in Iteration B above. Real per-vendor streaming, cancellation,
session-resume, the `execute()`-as-sugar refactor, and the two new
probe examples all moved to Iteration C+; the architecture-doc
refresh follows when those land.

## [0.3.0] — 2026-05-16

Phase 0 of the [implementation plan](docs/implementation-plan.md):
foundations that lock the shape of every extension point Phases 1–6
will use, plus a v0 contract-gap fix that wires plain-text
`execute(schema=None)` across every built-in adapter.

### Fixed

- Plain-text `execute(schema=None)` is honoured on every built-in
  adapter, closing a v0 contract gap. The
  `AgentRuntime.execute()` docstring has always promised
  "`None` means plain text — text answer on `RuntimeResult.text`,
  `structured=None`," but `ClaudeCodeRuntime`, `CopilotRuntime`, and
  `CodexRuntime` refused with `NotImplementedError`.
  (`OpenAICompatibleRuntime` / `OpenCodeZenRuntime` were already
  correct.) Per-adapter wiring:
  - `ClaudeCodeRuntime` — omits `output_format` from
    `ClaudeAgentOptions` when `schema is None`;
    `ResultMessage.result` carries the final assistant text. Cache
    key uses a `"__plain_text__"` sentinel so plain-text and
    structured sessions don't collide on `(model, system)`.
  - `CopilotRuntime` — doesn't register the `submit_result` tool
    when `schema is None`; doesn't prepend the forced-tool prefix
    to the system message; caller-supplied `system=` passes
    through verbatim.
  - `CodexRuntime` — omits `outputSchema` from `TurnOptions` when
    `schema is None`; `turn.final_response` is the free-form text.
    Empty `final_response` is a legitimate outcome rather than a
    structured-output violation.

  Motivation: a downstream consumer codebase (Maverick) just
  migrated five long-running personas onto airframe, each of
  which grew a single-field Pydantic schema purely to satisfy the
  `schema is None` gate. With the gate gone, those wrappers
  vanish and personas call
  `runtime.execute(prompt, system=PERSONA_SYSTEM_PROMPT)` directly.

### Added

- `airframe.Feature` enum — capability-negotiation predicates
  modelled on JDBC `DatabaseMetaData.supportsXxx()` and SQLAlchemy
  `Dialect.supports_*`. Ships the whole forward-looking set (19
  members covering Phases 1–6); later phases flip `True` bits onto
  each adapter without adding new members. Enum string values are
  public surface and snapshotted in tests.
- `AgentRuntime.supports(feature, model=None) -> bool` — pure,
  cheap lookup against an adapter's `SUPPORTED_FEATURES` class set.
  Today every in-tree adapter declares only
  `Feature.STRUCTURED_OUTPUT_JSON_SCHEMA`; Phase 1+ adds more.
- `AgentRuntime.unwrap(cls) -> cls` — JDBC-`Wrapper`-style escape
  hatch to the native SDK object. Each adapter accepts
  `unwrap(type(self))` plus its native types
  (`ClaudeCodeRuntime.unwrap(ClaudeSDKClient)`,
  `CopilotRuntime.unwrap(CopilotClient | CopilotSession)`,
  `CodexRuntime.unwrap(Codex | Thread)`,
  `OpenAICompatibleRuntime.unwrap(AsyncOpenAI)`). Unsupported types
  raise `TypeError`.
- `airframe.options` module with four empty frozen dataclasses
  (`ClaudeOptions`, `CopilotOptions`, `CodexOptions`,
  `OpenAICompatOptions`) and a `ProviderOptions` tagged-union alias.
  Vercel-style namespaced extension points for the
  `ClaudeAgentOptions`-sprawl problem; Phase 2+ populates each as
  features land. Not wired anywhere in v0.3.0 — Phase 1 attaches them
  to `runtime.session(provider_options=...)`.
- `CostRecord.reasoning_tokens: int` (default `0`) — canonicalises
  hidden reasoning / extended-thinking tokens across vendors. Phase 2
  wires real population from each SDK's native counter.
- Third-party adapter discovery via the `airframe.adapters`
  entry-point group. Third-party packages declare their runtime in
  `pyproject.toml` and `airframe.list_providers()` picks it up
  automatically. Modelled on SLF4J's `ServiceLoader` binding
  discovery; same pip-extras filtering applies. Built-ins shadow
  third-party entries with the same `PROVIDER_ID`. Malformed plugins
  (import errors, missing `PROVIDER_ID`, collisions) log a warning
  and are skipped — discovery never crashes.
- `discovery.ENTRY_POINT_GROUP = "airframe.adapters"` constant —
  public surface, snapshotted in tests.
- `airframe.testing` submodule with shared conformance contracts.
  Adapter authors import test functions from
  `airframe.testing.contracts` and provide an `adapter_runtime`
  pytest fixture; pytest collects the imported tests under the local
  fixture. SQLAlchemy `testing.suite` pattern. Ten structural
  contracts covering `close()` idempotency,
  `unwrap(type(self))` returning self, `supports()` purity,
  `validate_binding` behaviour, and the plain-text `execute()`
  path being wired
  (`test_plain_text_execute_path_is_wired`).
- New `[testing]` pyproject extra brings pytest + pytest-asyncio for
  adapter authors running the conformance suite. The main package
  does not depend on pytest.
- Per-adapter conformance test files (`tests/test_*_conformance.py`)
  for all four in-tree adapters — canonical examples for third-party
  authors.
- `examples/probe_supports.py` — capability matrix probe printing
  the live Feature × adapter truth table.
- `docs/feature-roadmap.md` — feature audit of the four native SDKs
  airframe wraps, prioritised list of cross-vendor features, and
  abstraction patterns borrowed from JDBC / SLF4J / OpenTelemetry /
  SQLAlchemy / JAAS.
- `docs/implementation-plan.md` — phased plan (Phase 0–6) ordered
  by dependency, with version targets, irreversible-shape-lock
  gating points, and adapter migration tables.

### Changed

- `AgentRuntime` protocol now requires two additional methods:
  `supports()` and `unwrap()`. Built-in adapters all implement them;
  external implementations of the protocol (none known) would need
  to add them.
- `CostRecord` constructor accepts an additional optional kwarg
  (`reasoning_tokens=0`); the structured-log dict via `to_dict()`
  always includes the key.

### Deferred

- Behavioural conformance contracts (401 → `RuntimeAuthError`,
  schema= round-trip, `input_tokens > 0` on a successful call,
  plain-text round-trip returns non-empty `text`) require live
  vendor credentials and naturally co-locate with Phase 1
  streaming/multi-turn integration test infrastructure. The
  existing `examples/probe_*.py` scripts already exercise these
  end-to-end against real vendors. Will land as
  `airframe.testing.integration` in v0.4.0.

## [0.2.0] — 2026-05-16

Model discovery API + an OpenAI-compatible base class. The headline
addition is `list_models()`: every adapter can now enumerate the
models a UI consumer can pick from, against the user's resolved
credentials. The headline change is install-state gating: which
providers `list_providers()` returns now reflects which pip extras
the consumer installed, so menus stay honest.

### Added

- `AgentRuntime.list_models() -> list[ModelInfo]` — live model
  discovery against the vendor's models endpoint. Driving UI menus
  is the expected use case; the call requires auth + network, so a
  failure is the consumer's signal to surface the issue before
  letting the user pick a model that would later fail to execute.
- `ModelInfo` — `id`, `display_name`, `provider_id`, optional
  `context_window`, `pricing_input_per_1k_usd`,
  `pricing_output_per_1k_usd`, capability flags, vendor-raw payload.
- Capability constants: `CAPABILITY_VISION`, `CAPABILITY_TOOLS`,
  `CAPABILITY_STRUCTURED_OUTPUT`, `CAPABILITY_STREAMING`,
  `CAPABILITY_REASONING_EFFORT`.
- `airframe.list_providers(installed_only=True)` — sorted list of
  provider IDs. Default filters to providers whose vendor SDK is
  importable (so `pip install airframe-agents[copilot]` users see
  `["github-copilot"]`, not the full menu). Pass
  `installed_only=False` for the full registry.
- `airframe.runtime_for(provider_id)` — returns the adapter class
  for a canonical provider ID. Raises `ValueError` for unknown IDs;
  raises `ImportError` (with a `pip install airframe-agents[...]`
  hint) when the adapter exists but its SDK isn't installed.
- `OpenAICompatibleRuntime` base — shared HTTP / `response_format` /
  `list_models` / error classification for OpenAI-compatible
  vendors. Subclasses (Zen today; Together / Groq / Fireworks /
  OpenRouter possible) ship a `PROVIDER_ID`, default base URL,
  default model, per-model metadata table, and a vendor-specific
  `_resolve_api_key()` hook — typically ~30 lines.
- `examples/probe_list_models.py` (committed as
  `tests/probe_list_models.py`) — end-to-end menu probe against every
  installed adapter; tabular output with context window / pricing /
  capabilities per model.
- `ClassVar` declarations on every adapter: `PROVIDER_ID`,
  `REQUIRES_PACKAGE`, `EXTRA_NAME` — drive discovery / install
  gating / error messages.

### Changed (breaking)

- **Provider IDs are strict.** Aliases dropped; each adapter has one
  canonical `PROVIDER_ID`. Update consumer code that referenced the
  old strings:
  - `ClaudeCodeRuntime`: was `{"anthropic", "claude", "claude-code"}`,
    now `"claude"`. The string `"anthropic"` is reserved for a
    future direct-API `AnthropicRuntime`.
  - `CopilotRuntime`: was `{"github-copilot", "copilot", "github"}`,
    now `"github-copilot"`.
  - `CodexRuntime`: was `{"openai", "codex"}`, now `"codex"`. The
    string `"openai"` is reserved for a future direct-API
    `OpenAIRuntime`.
  - `OpenCodeZenRuntime`: was `{"opencode-zen", "opencode"}`, now
    `"opencode"`.
- **The `[opencode-zen]` extra was renamed to `[openai-compat]`** —
  one extra for the whole OpenAI-compatible family (Zen today; room
  for siblings without exploding the extras matrix). All siblings
  share the `openai>=1.50,<3` SDK, so per-child extras would be a
  fiction. Update install commands:
  `pip install airframe-agents[opencode-zen]` →
  `pip install airframe-agents[openai-compat]`.
- `ClaudeCodeRuntime` now uses the Claude Agent SDK's native
  `output_format={"type": "json_schema", ...}` instead of the
  forced `submit_result` MCP tool. The SDK enforces the schema
  server-side; the validated payload lands on
  `ResultMessage.structured_output`. (Already shipped in 0.1.x via
  `479b576`; called out here because consumers using the lower-level
  capture surface would notice.)
- `CostRecord.provider_id` now reflects the canonical `PROVIDER_ID`
  (e.g. `"claude"`, `"codex"`, `"github-copilot"`) rather than
  vendor-name strings (`"anthropic"`, `"openai"`).

### Migration

```python
# Before (0.1.x)
ProviderModel("anthropic", "claude-haiku-4-5")
ProviderModel("openai", "gpt-5-codex")
ProviderModel("copilot", "gpt-5-mini")
ProviderModel("opencode-zen", "gpt-5-nano")

# After (0.2.0)
ProviderModel("claude", "claude-haiku-4-5")
ProviderModel("codex", "gpt-5-codex")
ProviderModel("github-copilot", "gpt-5-mini")
ProviderModel("opencode", "gpt-5-nano")
```

## [0.1.0] — 2026-05-15

Initial release. Extracted from the Maverick project after stabilising
through five migration phases.

### Added

- `AgentRuntime` protocol with `execute / reset / close /
  validate_binding`.
- `RuntimeResult`, `CostRecord`, `ProviderModel` data types.
- Vendor-agnostic error hierarchy: `AgentRuntimeError` base plus
  `RuntimeAuthError`, `RuntimeModelNotFoundError`,
  `RuntimeTransientError`, `RuntimeStructuredOutputError`,
  `RuntimeContextOverflowError`, `RuntimeProtocolError`,
  `RuntimeServerStartError`, `RuntimeCancelledError`.
- Four adapters:
  - `ClaudeCodeRuntime` via `claude-agent-sdk`. Claude Max
    subscription / OAuth / API-key auth. Forced `submit_result`
    MCP tool for structured output.
  - `CopilotRuntime` via `github-copilot-sdk`. GitHub Copilot
    subscription. Rejects `claude-*` model IDs (Phase 0 finding:
    Claude on Copilot can't honour tool calls).
  - `CodexRuntime` via `openai-codex-sdk`. ChatGPT Plus
    subscription / `OPENAI_API_KEY`. Native JSON-schema mode (no
    tool-forcing required).
  - `OpenCodeZenRuntime` via the `openai` SDK pointed at
    `https://opencode.ai/zen/v1`. Direct HTTP; no subprocess.
- Optional dependency extras: `[claude]`, `[copilot]`, `[codex]`,
  `[opencode-zen]`, `[all]`.
- 115 unit tests; 4 end-to-end probe scripts under `examples/`.
