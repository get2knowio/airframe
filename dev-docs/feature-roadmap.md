# Shared feature roadmap

A survey of what each native SDK behind an airframe adapter actually
ships, what airframe currently surfaces, and where the protocol can
grow without losing its "five-method, no opaque handles" shape.

Scope: the four SDKs airframe wraps today and the OpenAI-compatible
HTTP surface that `OpenAICompatibleRuntime` targets. Findings come
from inspecting the installed packages and from each vendor's
official docs (URLs collected in §6).

| SDK | Package | Status | Style | Adapter |
| --- | --- | --- | --- | --- |
| Claude Agent SDK | `claude-agent-sdk` 0.2.82 | GA (Agent SDK credit gating from 2026-06-15) | subprocess + JSON-RPC | `ClaudeCodeRuntime` |
| GitHub Copilot SDK | `github-copilot-sdk` 0.3.0 (imports as `copilot`) | Public preview since 2026-04-02 | subprocess + JSON-RPC | `CopilotRuntime` |
| OpenAI Codex SDK | `openai-codex-sdk` 0.1.11 (2026-01-19) | Python SDK pinned to Codex CLI versions | subprocess per turn (JSONL events) | `CodexRuntime` |
| OpenAI Python SDK | `openai` 2.37.0 | GA | direct HTTP (Chat Completions / Responses) | `OpenCodeZenRuntime` / `OpenAICompatibleRuntime` base |

What airframe surfaces today (v0.2.0):

* `execute(prompt, *, schema, system, persona, model, timeout) → RuntimeResult(text, structured, cost, finish, raw)`
* `reset() / close() / validate_binding() / list_models()`
* Cost telemetry — tokens, cache read/write, USD, finish reason
* `ModelInfo` capability flags — `vision`, `tools`, `structured_output`, `streaming`, `reasoning_effort`

Five capability flags are advertised but only one (`structured_output`)
is actually drivable through the protocol. The other four are read-only
metadata: there is no way for a consumer to *use* streaming, tools, or
reasoning today.

---

## 1. Provider-by-provider feature matrix

Legend: ● native first-class · ◐ available via vendor's CLI / config
but not surfaced cleanly in the Python SDK · ○ not supported.
"airframe today" tracks the *protocol*, not what each adapter could
plumb.

| Feature | Claude Agent SDK | Copilot SDK | Codex SDK | OpenAI Python (Chat Completions / OAI-compat) | OpenAI Python (Responses API, OpenAI-only) | airframe today |
| --- | --- | --- | --- | --- | --- | --- |
| **Session identity / resume** | ● `session_id`, `resume`, `continue_conversation`, `fork_session`, `list_sessions`, `delete_session`, `tag_session` | ● `session_id`, `client.resume_session`, `list_sessions`, `delete_session`, `get_last_session_id`, `set_foreground_session_id` | ● `start_thread() / resume_thread(thread_id)`; thread id arrives in `thread.started` event | ○ caller manages `messages=[]` | ● `previous_response_id`, server-managed `conversation` object | ○ implicit; hidden behind `reset()` |
| **Multi-turn within one session** | ● `client.query(prompt)` then `receive_response()`, repeatable | ● `session.send_and_wait()` repeatable; `session.get_messages()` returns history | ● `thread.run(prompt)` repeatable | ◐ caller appends `assistant` + `user` to `messages` list | ● implicit on `previous_response_id` or `conversation` | ○ |
| **Token-level streaming** | ● `include_partial_messages=True` → `StreamEvent` (raw Anthropic SSE chunks) | ● `streaming=True` on `create_session`; emits `assistant.message_delta` / `assistant.reasoning_delta` / `assistant.streaming_delta` | ● `thread.run_streamed()` returns `AsyncIterator[ThreadEvent]` | ● `stream=True` → `AsyncIterator[ChatCompletionChunk]` | ● `stream=True` | ○ |
| **Message-level event stream** | ● `receive_messages()` yields `AssistantMessage` / `UserMessage` / `SystemMessage` / `ResultMessage` / `HookEventMessage` / `TaskNotificationMessage` / `RateLimitEvent` / `MirrorErrorMessage` | ● `session.on(handler)` — **75+ event types** in `SessionEventType` enum, covering session/turn/tool/subagent/skill/MCP/permission/elicitation lifecycle | ● `ItemStartedEvent` / `ItemUpdatedEvent` / `ItemCompletedEvent` / `TurnStartedEvent` / `TurnCompletedEvent` / `TurnFailedEvent` / `ThreadErrorEvent` | ○ | ◐ Responses API has typed stream events | ○ |
| **Custom function tools** | ● `@tool(...)` decorator + `create_sdk_mcp_server()` (in-process MCP) | ● `define_tool(name, description, handler, params_type, skip_permission)` | ○ (no Python tool-registration API; tools live behind the CLI) | ● `tools=[{"type":"function","function":{...}}]`; `tool_choice`; `parallel_tool_calls`; `strict` | ● `tools=[FunctionToolParam]` + 14 other tool types | ○ (used internally for forced structured-output on Copilot only) |
| **Built-in tools** | ● Bash / Read / Edit / Write / Glob / Grep / WebFetch / WebSearch / Skill / Task / TodoWrite etc. via `tools=` / `allowed_tools=` / `disallowed_tools=` | ● Allowlist via `available_tools` / `excluded_tools` on `create_session` | ● Codex's shell/edit/web-search live behind the CLI; toggled via `network_access_enabled`, `web_search_enabled` on `ThreadOptions` | ○ | ● Built-in tools: `file_search`, `web_search`, `code_interpreter`, `image_generation`, `local_shell`, `computer_use`, `apply_patch` | ○ |
| **MCP server registration** | ● `mcp_servers={...}` accepts stdio, SSE, HTTP, in-process SDK servers; `strict_mcp_config`; reconnect/toggle/status APIs | ● `mcp_servers={...}` on `create_session`; `enable_config_discovery` auto-loads `.mcp.json` / `.vscode/mcp.json`; oauth events `mcp.oauth_required` / `mcp.oauth_completed` | ◐ Codex CLI supports MCP via its own config file; Python SDK doesn't expose registration but `McpToolCallItem` shows MCP results | ○ | ● `tools=[Mcp(type="mcp", server_label=, server_url=, ...)]` — remote MCP, including service connectors (Gmail, Drive, etc.) | ○ |
| **Subagents / orchestration** | ● `agents={name: AgentDefinition}` programmatic subagents; `list_subagents`, `get_subagent_messages`; Task tool | ● `custom_agents=[CustomAgentConfig]`, `subagent.started/completed/failed/selected/deselected` events | ○ at SDK level | ○ | ○ | ○ |
| **Vision (image) inputs** | ◐ Read tool reads images from disk; no direct image-in-prompt API | ● `attachments=[FileAttachment {type:"file", path:...}]` on `send` | ● `input=[TextInput, LocalImageInput {path:...}]` | ● content parts `[{type:"image_url", image_url:{url}}]`; widely supported by compat vendors with vision models | ● same content-part shape | ○ |
| **File / PDF / document inputs** | ◐ via Read tool | ● `attachments=[FileAttachment / DirectoryAttachment / BlobAttachment / SelectionAttachment]` | ◐ via working-directory + Read | ◐ `client.files.create` + reference; varies wildly across compat vendors | ● `response_input_file` part | ○ |
| **Audio inputs / TTS / STT** | ○ | ○ | ○ | ● `audio.transcriptions`, `audio.speech`; chat audio modality on `gpt-4o-audio` | ● | ○ |
| **Structured output (JSON Schema)** | ● `output_format={"type":"json_schema","schema":...}` (native, server-enforced) | ◐ no native mode — forced via the `submit_result` tool pattern airframe ships | ● `TurnOptions.output_schema={...}` (passed to CLI as `--output-schema`) | ● `response_format={"type":"json_schema","json_schema":{...,"strict":true}}`; `client.chat.completions.parse(..., response_format=PydanticModel)` | ● `text={"format":{"type":"json_schema",...}}` | ● (every adapter) |
| **Reasoning / thinking control** | ● `thinking={"type":"adaptive"\|"enabled"\|"disabled","budget_tokens":N}` + `effort: low/medium/high/xhigh/max` | ● `reasoning_effort` on `create_session`; `assistant.reasoning` / `assistant.reasoning_delta` events | ● `model_reasoning_effort: minimal/low/medium/high` on `ThreadOptions`; `ReasoningItem` in turn output | ● `reasoning_effort: minimal/low/medium/high` on Chat Completions; `verbosity` | ● `reasoning={"effort":...}` | ○ (capability flag exists, no knob) |
| **Prompt caching — read stats** | ● `usage.cache_read_input_tokens` / `cache_creation_input_tokens` | ● `AssistantUsageData.cache_read_tokens` / `cache_write_tokens` | ● `Usage.cached_input_tokens` | ● `usage.prompt_tokens_details.cached_tokens` | ● same | ● (surfaced in `CostRecord`) |
| **Prompt caching — controls** | ◐ implicit (session warmth); messages-API `cache_control` markers not in agent SDK | ○ implicit | ○ implicit | ● `prompt_cache_key`, `prompt_cache_retention: "in_memory" \| "24h"` | ● same | ○ |
| **Permission / approval callback** | ● `can_use_tool` callback + `permission_mode: default/acceptEdits/bypassPermissions/plan/dontAsk/auto` + `PermissionRequest` hook | ● mandatory `on_permission_request`; `permission.requested` / `permission.completed` events; per-tool `skip_permission` | ● `approval_policy: never/on-request/on-failure/untrusted` on `ThreadOptions` | ○ | ○ (built-in tools have `require_approval` on MCP) | ○ |
| **Lifecycle hooks** | ● `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `UserPromptSubmit`, `Stop`, `SubagentStop`, `SubagentStart`, `PreCompact`, `Notification`, `PermissionRequest` — with input/output shapes per event | ● `SessionHooks` typed dict — `pre_tool_use`, `post_tool_use`, `user_prompt_submitted`, `session_start`, `session_end`, `error_occurred` | ○ | ○ | ○ | ○ |
| **Working-directory / sandbox** | ● `cwd`, `add_dirs`, `sandbox: SandboxSettings` (network/fs allow rules) | ● `working_directory` on `create_session` | ● `working_directory`, `additional_directories`, `sandbox_mode: read-only/workspace-write/danger-full-access`, `network_access_enabled` | ○ | ◐ container sandbox via code_interpreter | ○ |
| **Cancellation / interrupt** | ● `client.interrupt()` + `stop_task(task_id)` | ● `session.abort()` | ● `AbortController` / `AbortSignal` on `TurnOptions.signal` | ◐ asyncio task cancel propagates to httpx | ● also `background=True` + cancel | ○ |
| **Web search built-in** | ● `WebSearch` / `WebFetch` tools | ● built-in tool (allowlist via `available_tools`) | ● `web_search_enabled` on `ThreadOptions`; `WebSearchItem` in turn items | ● `web_search_options` on Chat Completions; some compat vendors support | ● `tools=[WebSearchToolParam]` | ○ |
| **Cost reporting** | ● `total_cost_usd` (vendor-computed) | ● `AssistantUsageData.cost` (vendor-computed) | ○ tokens only; airframe computes from pricing table | ● tokens only | ● tokens only | ● `CostRecord` |
| **Live model listing** | ● `/v1/models` (api-key only — OAuth bearer doesn't work) | ● `client.list_models()` with rich capability metadata | ● `/v1/models` via `AsyncOpenAI` (codex CLI subset filtered) | ● `client.models.list()` | ● same | ● `list_models()` |
| **Budget / turn caps** | ● `max_turns`, `max_budget_usd`, `task_budget` | ○ | ○ | ● `max_tokens` / `max_completion_tokens` | ● `max_output_tokens`, `max_tool_calls` | ○ |
| **Batch API** | ○ | ○ | ○ | ● `client.batches.create` (OpenAI only; compat vendors ~never) | ● | ○ |
| **Fine-tuning** | ○ | ○ | ○ | ● `client.fine_tuning.jobs` (OpenAI only) | ● | ○ |
| **Skills / plugins** | ● `skills=[...]`, `plugins=[SdkPluginConfig]` | ● `skill_directories`, `disabled_skills`, `skill.invoked` event | ○ | ○ | ○ | ○ |
| **File checkpointing / rewind** | ● `enable_file_checkpointing=True` + `client.rewind_files(message_id)` | ◐ `session.snapshot_rewind` event implies support | ○ | ○ | ○ | ○ |
| **Compaction / context summarisation** | ● `PreCompact` hook + `SystemMessage(subtype="compact_boundary")` | ● `session.compaction_start` / `session.compaction_complete` events | ○ | ○ | ● `context_management` param | ○ |
| **Rate-limit telemetry** | ● `RateLimitEvent` / `RateLimitInfo` (`five_hour`, `seven_day`, `seven_day_opus`, etc.) | ◐ surfaced as `SessionErrorData` with status_code | ○ at SDK level | ◐ HTTP 429 → `RateLimitError` | ◐ same | ◐ via `RuntimeTransientError` |
| **External session storage / mirror** | ● `session_store: SessionStore`, `session_store_flush`, `InMemorySessionStore`, `fold_session_summary`, `import_session_to_store` | ○ at SDK level | ○ | ○ | ○ (Responses API server stores) | ○ |

---

## 2. Per-SDK notes

### 2.1 Claude Agent SDK — `claude-agent-sdk` 0.2.82

The widest surface of the four. Anthropic [positions this as "the
same tools, agent loop, and context management that power Claude
Code, programmable in Python and TypeScript"](https://code.claude.com/docs/en/agent-sdk/overview).
A one-shot `query(prompt, options)` async iterator is offered
alongside the long-lived `ClaudeSDKClient` we wrap today. Key types
from `claude_agent_sdk`:

* `ClaudeSDKClient(options=ClaudeAgentOptions(...))` + `async with` lifecycle.
* `ClaudeAgentOptions` exposes ~60 fields: `tools`, `allowed_tools`,
  `disallowed_tools`, `system_prompt`, `mcp_servers`, `strict_mcp_config`,
  `permission_mode`, `continue_conversation`, `resume`, `session_id`,
  `max_turns`, `max_budget_usd`, `model`, `fallback_model`, `betas`,
  `cwd`, `add_dirs`, `env`, `extra_args`, `can_use_tool`, `hooks`,
  `include_partial_messages`, `include_hook_events`, `fork_session`,
  `agents` (subagents), `setting_sources`, `skills`, `sandbox`,
  `plugins`, `thinking`, `effort`, `output_format`,
  `enable_file_checkpointing`, `session_store`, `task_budget`, more.
* Messages on the receive stream: `UserMessage`, `AssistantMessage`,
  `SystemMessage`, `ResultMessage`, `StreamEvent`,
  `TaskStartedMessage` / `TaskProgressMessage` / `TaskNotificationMessage`,
  `HookEventMessage`, `RateLimitEvent`, `MirrorErrorMessage`.
* `ResultMessage` carries `subtype`, `duration_ms`, `duration_api_ms`,
  `is_error`, `num_turns`, `session_id`, `stop_reason`,
  `total_cost_usd`, `usage`, `result`, `structured_output`,
  `model_usage`, `permission_denials`, `deferred_tool_use`, `errors`,
  `api_error_status`.
* Runtime mutation: `set_permission_mode(mode)`, `set_model(model)`,
  `interrupt()`, `stop_task(task_id)`, `rewind_files(message_id)`,
  `reconnect_mcp_server(name)`, `toggle_mcp_server(name, enabled)`,
  `get_mcp_status()`, `get_context_usage()`.
* Session lifecycle outside the client: `list_sessions`,
  `get_session_info`, `get_session_messages`, `list_subagents`,
  `fork_session`, `rename_session`, `tag_session`, `delete_session`.
* MCP server types: `McpStdioServerConfig`, `McpSSEServerConfig`,
  `McpHttpServerConfig`, `McpSdkServerConfig`,
  `McpClaudeAIProxyServerConfig`. In-process SDK servers built via
  `create_sdk_mcp_server(name, version, tools=[...])` + `@tool` decorator.
* Hooks taxonomy: `HookEvent = PreToolUse | PostToolUse |
  PostToolUseFailure | UserPromptSubmit | Stop | SubagentStop |
  PreCompact | Notification | SubagentStart | PermissionRequest`.
  (The user-facing docs also surface `SessionStart` / `SessionEnd` as
  hook names — those are emitted via the same plumbing.)

Notable 2025–2026 additions (from the
[Python SDK CHANGELOG](https://github.com/anthropics/claude-agent-sdk-python/blob/main/CHANGELOG.md)):
native structured outputs (v0.1.7); `max_budget_usd` (v0.1.6) +
`task_budget` (v0.1.51); explicit `ThinkingConfig` adaptive / enabled
/ disabled (v0.1.36); `effort` levels including `"xhigh"` for Opus
4.7 (v0.1.74); `fork_session` + `rewind_files` (v0.1.0 / v0.1.15);
`SessionStore` protocol with eager-vs-batched flush (v0.1.64 /
v0.1.73); `@tool` decorator + `create_sdk_mcp_server()` (v0.0.22+
with `ToolAnnotations` enhancements at v0.1.31); strict-MCP config
(v0.1.74); `include_partial_messages` fine-grained streaming
(v0.1.36 / v0.1.48); `auto` permission mode (v0.1.57); `skills=`
field (v0.1.62); subagent transcript listing (v0.1.60); W3C
trace-context propagation for the CLI subprocess (v0.1.60);
`api_error_status` on `ResultMessage` (v0.1.76).

Auth caveat to note for airframe: Anthropic [does not permit
third-party use of `claude.ai` login or rate limits for products
built on the Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview).
The subscription path is "the developer of the agent is the
Claude-Max user," not "your end-user supplies their own Claude
account." Airframe's auth chain already reflects this (Max OAuth
goes through the SDK's bundled CLI; API-key auth is the documented
production path).

### 2.2 GitHub Copilot SDK — `github-copilot-sdk` 0.3.0 (`import copilot`)

JSON-RPC wrapper around the `copilot` CLI. Entered [technical preview
in January 2026 and public preview on 2026-04-02](https://aitoolsbee.com/news/github-copilot-sdk-enables-agent-orchestration-via-mcp-registry-and-cli/),
released alongside SDKs for TypeScript, Go, .NET and Java that share
the same JSON-RPC protocol. Even larger event surface than Claude —
75+ entries in `SessionEventType` enum at
`copilot/generated/session_events.py`.

* `CopilotClient(SubprocessConfig(cli_path=, github_token=, use_logged_in_user=))`
  with `start()`, `stop()`, `force_stop()`, `ping()`, `get_status()`,
  `get_auth_status()`, `list_models()`, `list_sessions(filter)`,
  `get_session_metadata(id)`, `delete_session(id)`,
  `get_last_session_id()`, `get_foreground_session_id()`,
  `set_foreground_session_id(id)`, client-level `on()` for lifecycle
  events, plus `create_session(...)` and `resume_session(...)`.
* `create_session` accepts (selected): `on_permission_request`, `model`,
  `session_id`, `client_name`, `reasoning_effort`, `tools`,
  `system_message`, `available_tools`, `excluded_tools`,
  `on_user_input_request`, `hooks: SessionHooks`, `working_directory`,
  `provider: ProviderConfig` (Azure / custom), `model_capabilities`,
  `streaming`, `include_sub_agent_streaming_events`, `mcp_servers`,
  `custom_agents`, `default_agent`, `agent`, `config_dir`,
  `enable_config_discovery`, `skill_directories`, `disabled_skills`,
  `infinite_sessions`, `on_event`, `commands`,
  `on_elicitation_request`, `create_session_fs_handler`, `github_token`.
* `CopilotSession`: `send`, `send_and_wait`, `on(handler)`,
  `get_messages()`, `disconnect()`, `destroy()`, `abort()`,
  `set_model(model, reasoning_effort=)`, `log(...)`, plus the `ui`
  helper (`elicitation`, `confirm`, `select`, `input`).
* Attachments: `FileAttachment`, `DirectoryAttachment`,
  `SelectionAttachment`, `BlobAttachment`.
* Hooks (`SessionHooks` TypedDict): `pre_tool_use`, `post_tool_use`,
  `user_prompt_submitted`, `session_start`, `session_end`,
  `error_occurred`.
* No native JSON-schema output — airframe forces structured output
  via `submit_result` tool. `validate_binding` rejects `claude-*`
  on Copilot because Claude served through Chat Completions emits
  markdown-fenced JSON instead of calling tools.
* MCP server configuration documented at
  [GitHub Docs — Using MCP servers with the Copilot SDK](https://docs.github.com/en/copilot/how-tos/copilot-sdk/use-copilot-sdk/mcp-servers).
  Transport types are `stdio` and `http`; `enable_config_discovery`
  auto-loads `.mcp.json` / `.vscode/mcp.json` from the working
  directory. OAuth-required servers surface
  `mcp.oauth_required` / `mcp.oauth_completed` events on the session
  event stream.

### 2.3 OpenAI Codex SDK — `openai-codex-sdk` 0.1.11

The narrowest of the agentic SDKs — TypeScript-port shape (camelCase
aliases on every option). [Official docs](https://developers.openai.com/codex/sdk)
describe the same shape across the TypeScript SDK
(`@openai/codex-sdk`, Node.js 18+) and the Python port (Python 3.10+).
The Python package on PyPI [is published from the openai/codex
monorepo with Codex-pinned versioning](https://github.com/openai/codex/pull/18996) —
Python SDK version tracks the underlying Codex CLI binary, not a
separate semver. Key types in `openai_codex_sdk.types`:

* `Codex({apiKey, baseUrl, codexPathOverride, env})`.
* `client.start_thread(ThreadOptions)` / `client.resume_thread(thread_id, ThreadOptions)`.
* `ThreadOptions`: `model`, `sandbox_mode` (`read-only` /
  `workspace-write` / `danger-full-access`), `working_directory`,
  `skip_git_repo_check`, `model_reasoning_effort`
  (`minimal/low/medium/high`), `network_access_enabled`,
  `web_search_enabled`, `approval_policy`
  (`never/on-request/on-failure/untrusted`), `additional_directories`.
* `TurnOptions`: `output_schema`, `signal: AbortSignal`.
* `Input`: `str` *or* `list[TextInput | LocalImageInput]` —
  `LocalImageInput(type="local_image", path=...)`.
* `thread.run(input, options) → Turn(items, final_response, usage)` or
  `thread.run_streamed(...) → StreamedTurn(events: AsyncIterator)`.
* Thread items: `AgentMessageItem`, `ReasoningItem`,
  `CommandExecutionItem`, `FileChangeItem`, `McpToolCallItem`,
  `WebSearchItem`, `TodoListItem`, `ErrorItem`.
* Events: `ThreadStartedEvent`, `TurnStartedEvent`,
  `TurnCompletedEvent` (carries `Usage`), `TurnFailedEvent`,
  `ItemStartedEvent`, `ItemUpdatedEvent`, `ItemCompletedEvent`,
  `ThreadErrorEvent`.
* `Usage(input_tokens, cached_input_tokens, output_tokens)`. No
  reasoning_tokens, no cache_write_tokens, no `total_cost_usd`.
* No Python tool-registration API. MCP tool calls appear in the event
  stream (`McpToolCallItem`) but must be configured at the CLI level.
* Lifecycle: one subprocess per `thread.run()`. The `Thread` Python
  object is a thin reference holding `thread_id`.

Notable 2026 additions ([Codex CLI changelog](https://developers.openai.com/codex/changelog)):
Python app-server SDK publishing under Codex-pinned versioning;
thread pagination with summary / full turn views over a remote
app-server; Unix socket transport; profile-backed permission /
sandbox configuration replacing the legacy `approval_policy` /
`sandbox_mode` knobs; storage of MCP tool calls in rollouts.
Airframe's adapter pins `approval_policy` and `sandbox_mode` today
and should track the profile-backed replacement.

### 2.4 OpenAI Python SDK — `openai` 2.37.0

Used by `OpenAICompatibleRuntime` against any vendor that speaks
OpenAI's HTTP shape (opencode Zen today; Together / Groq / Fireworks /
OpenRouter as planned siblings). Two surfaces matter:

**Chat Completions** (`client.chat.completions.create`) — the surface
all OAI-compat vendors implement. Notable params:
`messages`, `model`, `tools`, `tool_choice`, `parallel_tool_calls`,
`response_format`, `stream`, `stream_options`, `temperature`, `top_p`,
`max_completion_tokens` / `max_tokens`, `reasoning_effort`, `verbosity`,
`prompt_cache_key`, `prompt_cache_retention`, `seed`, `service_tier`,
`store`, `audio`, `modalities`, `prediction`, `web_search_options`,
`metadata`, `safety_identifier`, `n`, `logit_bias`, `logprobs`,
`top_logprobs`, `frequency_penalty`, `presence_penalty`, `stop`,
plus `extra_headers / extra_query / extra_body`.

Response: `response.choices[0].message` carries `content`,
`tool_calls[]`, `refusal`, `audio`, `reasoning_content`, etc.
`usage` carries `prompt_tokens`, `completion_tokens`, and
`prompt_tokens_details.cached_tokens` /
`completion_tokens_details.reasoning_tokens`.

**Responses API** (`client.responses.create`) — OpenAI-proprietary,
not a compat-vendor surface. Adds: server-managed `conversation`,
`previous_response_id`, server-side `tools` (function, web_search,
file_search, code_interpreter, image_generation, computer_use,
local_shell, apply_patch, MCP), `background=True` async jobs,
`context_management`, `include` flags for hidden fields, `instructions`
override, typed input items, granular streaming events.

MCP via Responses ([Connectors and MCP servers guide](https://developers.openai.com/api/docs/guides/tools-connectors-mcp)):
`tools=[{"type":"mcp", "server_label":..., "server_url":..., "connector_id":..., "allowed_tools":..., "require_approval":..., "authorization":..., "server_description":..., "defer_loading":...}]`.
The named-connector list ships eight built-ins:
`connector_dropbox`, `connector_gmail`, `connector_googlecalendar`,
`connector_googledrive`, `connector_microsoftteams`,
`connector_outlookcalendar`, `connector_outlookemail`,
`connector_sharepoint`. OpenAI's docs are explicit that **MCP-as-tool
is Responses-only — there is no Chat Completions support**, which
means the entire `OpenAICompatibleRuntime` family (Chat Completions
shape) cannot serve MCP regardless of vendor.

Structured outputs ([guide](https://developers.openai.com/api/docs/guides/structured-outputs)):
on Chat Completions, `response_format={"type":"json_schema","json_schema":{"name":...,"strict":true,"schema":...}}`;
on Responses, `text={"format":{"type":"json_schema","strict":true,"schema":...}}`;
plus the `client.chat.completions.parse(response_format=PydanticModel)`
helper. `strict: true` constrains the schema (no `oneOf`, all
properties required, `additionalProperties: false`, restricted type
set) — `OpenAICompatibleRuntime` currently passes `strict: false`
for maximum compat-vendor portability.

Compat-vendor coverage of these features is highly uneven —
`reasoning_effort` and `prompt_cache_key` are OpenAI-only;
`response_format=json_schema` is widely but not universally supported;
Responses API and MCP-as-tool are OpenAI-only as of writing.

---

## 3. Prioritized list of features airframe should add

Each item is a *protocol-level* addition (or refinement) — not a list
of vendor knobs to forward. The bar: every adapter should be able to
implement a meaningful version of the feature, and consumers should
not need to switch on `provider_id` to use it.

Ordered by ratio of (consumer pain today) × (cross-vendor coverage) ÷
(abstraction cost).

### Already closed in v0.3.0 — plain-text `execute(schema=None)`

Not a feature addition; a v0 contract-gap fix folded into Phase 0.
The `AgentRuntime.execute()` docstring promised
"`None` means plain text — text answer on `RuntimeResult.text`,
`structured=None`," but three of the four built-in adapters
(`ClaudeCodeRuntime`, `CopilotRuntime`, `CodexRuntime`) refused
with `NotImplementedError`. The OpenAI-compatible family was
correct already. v0.3.0 wires the missing three and adds a
conformance check (`test_plain_text_execute_path_is_wired` in
`airframe.testing.contracts`) that every adapter must pass.

**Motivation.** A downstream consumer codebase (Maverick) just
migrated five long-running personas onto airframe. Each grew a
single-field Pydantic schema (`Payload(text: str)`) purely to
satisfy the `schema is None` gate — markdown summaries, free-form
analyses, agents that write files via tools and only need a "done"
signal all paid the schema-wrapper tax. With the gate gone, those
wrappers vanish and personas call
`runtime.execute(prompt, system=PERSONA_SYSTEM_PROMPT)` directly.

See [implementation-plan.md §"Phase 0 — Foundations" #7](./implementation-plan.md)
for the per-adapter wiring notes.

### P0 — Streaming responses

**Why now.** Three of four SDKs are streaming-native. Without a
streaming protocol method, consumers can't render partial output, see
tool calls fire, or display reasoning while it happens — they just
block on `execute()` until the full result lands. The capability flag
already exists in `ModelInfo`; only the call site is missing.

**Sketch.**

```python
async def stream(
    self,
    prompt: str,
    *,
    schema: type[BaseModel] | None = None,
    system: str | None = None,
    model: ProviderModel | None = None,
    timeout: float = 600.0,
) -> AsyncIterator[RuntimeEvent]: ...
```

Where `RuntimeEvent` is a vendor-agnostic discriminated union:

```python
@dataclass(frozen=True, slots=True)
class TextDelta:
    """Partial assistant text."""
    text: str

@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    """Partial reasoning/thinking trace (when the model exposes one)."""
    text: str

@dataclass(frozen=True, slots=True)
class ToolCallStart:
    tool_name: str
    tool_call_id: str
    arguments_preview: str   # partial JSON

@dataclass(frozen=True, slots=True)
class ToolCallResult:
    tool_call_id: str
    output: Any
    is_error: bool

@dataclass(frozen=True, slots=True)
class TurnComplete:
    """Terminal event; mirrors RuntimeResult."""
    result: RuntimeResult

RuntimeEvent = TextDelta | ReasoningDelta | ToolCallStart | ToolCallResult | TurnComplete
```

Adapter mapping:
* Claude → `include_partial_messages=True` on `ClaudeAgentOptions`,
  fold raw `StreamEvent` chunks + `AssistantMessage` blocks.
* Copilot → `streaming=True` on `create_session`, subscribe via
  `session.on(handler)` to `assistant.message_delta` /
  `assistant.reasoning_delta` / `tool.execution_*`.
* Codex → `thread.run_streamed()` → fold `ItemStartedEvent` /
  `ItemUpdatedEvent` / `ItemCompletedEvent`.
* OpenAI-compat → `stream=True` on Chat Completions, walk
  `ChatCompletionChunk.choices[0].delta`.

`execute()` becomes sugar over `stream()` — drain to the terminal
`TurnComplete` and return its `result`.

### P1 — Explicit multi-turn / session continuation

**Status:** Largely shipped in v0.3.0 (the `AgentSession` retrofit)
and v0.4.0+ (`runtime.session(resume=...)` on adapters declaring
`Feature.SESSION_RESUME`). One remaining gap surfaced by an
Earlybird adoption attempt: **stateless backend + frontend-owned
transcript** — the fourth multi-turn shape that none of the
existing three mechanisms (in-memory session, `resume=`,
single-call multi-turn loop) serve. Spec'd in
[`history-seed-plan.md`](./history-seed-plan.md) as a new
`prior_messages=` kwarg on `session()` + `Feature.HISTORY_SEED`
flag. v0.9.0 candidate.

**Why.** Today the protocol is "send prompt → get answer; if you
`reset()` you've nuked the conversation." A consumer that wants to do
follow-up turns has to keep the runtime instance alive and just call
`execute()` again — which works on Claude/Copilot/Codex but is silently
single-turn on OpenAI-compat (no conversation state in chat completions).

The README is explicit that "Anything *above* the protocol — retry
policy, fallback across vendors, conversation memory, multi-agent
orchestration — is left to the consumer." Conversation memory is the
one entry on that list that almost every consumer needs.

**Sketch.** Add a thin session abstraction that adapters with native
sessions can map onto, and that the stateless `OpenAICompatibleRuntime`
satisfies with a client-side message buffer:

```python
class AgentSession(Protocol):
    """One conversation. Closes when no longer referenced; cheap to recreate."""
    id: str | None    # vendor-issued session/thread/response id, when known

    async def execute(self, prompt: str, *, schema=None, ...) -> RuntimeResult: ...
    async def stream(self, prompt: str, *, schema=None, ...) -> AsyncIterator[RuntimeEvent]: ...
    async def close(self) -> None: ...

class AgentRuntime(Protocol):
    ...
    def session(self, *, resume: str | None = None, system: str | None = None,
                model: ProviderModel | None = None) -> AgentSession: ...
```

Existing `runtime.execute(prompt)` becomes shorthand for
`runtime.session().execute(prompt)` — a single-turn session. The
multi-turn case is `s = runtime.session(); await s.execute(a); await
s.execute(b); await s.close()`.

This keeps the "no opaque handles in the call site" principle (the
caller threads a typed `AgentSession`, not a session-id string) while
making conversation state explicit.

Resume semantics: when `resume=` is passed, adapters that support it
restore state (`continue_conversation` / `resume_session` /
`resume_thread` / `previous_response_id`); compat HTTP adapters
either reject the resume token or use it as a local-buffer key.

### P1 — Cancellation / interrupt

**Why.** All four SDKs expose it; airframe has no surface today.
Once streaming lands, mid-stream cancel is the natural pair.

**Sketch.** Either:
1. `RuntimeResult` is yielded from an awaitable; `asyncio.Task.cancel()`
   on the surrounding task propagates through the adapter (this works
   for OpenAI-compat already; for the subprocess SDKs, the adapter
   wires `CancelledError` to the vendor's native interrupt).
2. Or a more explicit `RuntimeHandle` returned alongside the awaitable:
   `handle, fut = runtime.execute_handle(...)`; `await handle.cancel()`.

Option 1 fits Python idioms; option 2 maps more cleanly to the
SDKs' native `interrupt()` / `abort()` / `AbortController.abort()`.
Recommendation: option 1, plus an explicit `await session.cancel()`
method on `AgentSession` when there's no `asyncio.Task` to cancel.

### P2 — Reasoning / thinking effort

**Why.** Every SDK has the knob; airframe has the capability flag
(`CAPABILITY_REASONING_EFFORT`) but no way to actually set effort.
Modelling it now avoids three different per-vendor kwarg shapes
leaking into consumer code later.

**Sketch.** Promote a parameter onto `execute()` / `stream()`:

```python
ReasoningEffort = Literal["minimal", "low", "medium", "high"]
ThinkingMode = (
    None                                       # default for the model
    | ReasoningEffort                          # effort-style knob
    | {"budget_tokens": int}                   # explicit budget (Claude)
    | "disabled"                               # turn it off
)

async def execute(
    self,
    prompt: str,
    *,
    schema: type[BaseModel] | None = None,
    thinking: ThinkingMode = None,
    ...
) -> RuntimeResult: ...
```

Per-adapter mapping:
* Claude → `ClaudeAgentOptions.effort` for the literal cases; `thinking={"type":"enabled","budget_tokens":N}` for explicit budget.
* Copilot → `reasoning_effort` on `create_session` (effort literals only).
* Codex → `model_reasoning_effort` on `ThreadOptions`.
* OpenAI-compat → `reasoning_effort` kwarg on Chat Completions.

Adapters that get an unsupported shape (e.g. `budget_tokens` to
Copilot) coerce to the nearest equivalent or raise
`UnsupportedBindingError`.

Add `reasoning_tokens` to `CostRecord` (every SDK reports it under a
different name; canonicalise once).

### P2 — Vision / file inputs

**Why.** Different shape per vendor, same intent. Once present, a
consumer can write image-aware agents without switching on `runtime`.

**Sketch.** Make `prompt` polymorphic on top of `str`:

```python
@dataclass(frozen=True, slots=True)
class ImageInput:
    """Image content — by path, by bytes, or by URL."""
    path: str | None = None
    bytes_: bytes | None = None
    url: str | None = None
    media_type: str | None = None  # "image/png", etc.

@dataclass(frozen=True, slots=True)
class FileInput:
    """File / document — adapters either upload or attach by path."""
    path: str
    media_type: str | None = None

PromptPart = str | ImageInput | FileInput
Prompt = str | list[PromptPart]

async def execute(self, prompt: Prompt, *, ...) -> RuntimeResult: ...
```

Per-adapter:
* Claude → fall back to the Read tool (auto-allow on the prompt's
  image/file paths). Bytes/URL → temp file. Confirm a future
  `claude-agent-sdk` version surfaces direct image input.
* Copilot → translate `ImageInput(path=)` to
  `FileAttachment(type="file", path=...)`; bytes → `BlobAttachment`.
* Codex → `ImageInput(path=)` → `LocalImageInput(type="local_image", path=...)`; bytes/URL → temp file (Codex is path-only).
* OpenAI-compat → content parts:
  `{"type":"image_url","image_url":{"url": "data:image/png;base64,..."}}`.

Adapters reject inputs they can't serve with a clear
`UnsupportedBindingError` (e.g. Codex on a URL with no network access).

### P2 — Tool / MCP server registration

**Why.** Three of four SDKs surface a tool-registration API; the
fourth (compat HTTP) supports it via the standard `tools=[]` shape.
MCP is the cross-vendor standard for *remote* tools and has native
support on three of four (Claude in-process + remote; Copilot remote;
Codex via CLI config; OpenAI-Responses remote). Today airframe uses
the tool slot internally for one purpose only (forced structured
output on Copilot) and exposes nothing.

This is the largest-surface item and the most likely to crack the
"narrow protocol" principle. Two-tier proposal:

* **Tier 1 — function tools** (`P2`). One Pydantic-typed callable
  registered at session construction:

  ```python
  @dataclass(frozen=True, slots=True)
  class FunctionTool:
      name: str
      description: str
      params: type[BaseModel]              # → JSON schema
      handler: Callable[[BaseModel], Awaitable[Any]]

  runtime.session(tools=[my_tool])
  ```

  Adapter mapping: Claude → in-process MCP via
  `create_sdk_mcp_server`; Copilot → `define_tool`; Codex → not
  supported, adapter rejects (or surfaces a `tools` capability gate);
  OpenAI-compat → `tools=[{"type":"function",...}]` + tool-result
  round-trip.

* **Tier 2 — MCP server references** (`P3`). A typed config object
  the adapter forwards to whichever native MCP entry point exists:

  ```python
  @dataclass(frozen=True, slots=True)
  class McpServerRef:
      name: str
      transport: Literal["stdio", "http", "sse"]
      command: list[str] | None = None     # stdio
      url: str | None = None               # http / sse
      headers: dict[str, str] | None = None
      auth_token: str | None = None
  ```

  Adapter mapping: Claude → `mcp_servers={name: McpStdioServerConfig | McpHttpServerConfig | McpSSEServerConfig}`; Copilot → `mcp_servers={name: MCPStdioServerConfig | MCPHTTPServerConfig}`; Codex → reject (not in Python SDK surface); OpenAI-compat → reject unless backed by Responses API (which OAI-compat vendors don't expose).

### P3 — Permission / approval callback

**Why.** Subprocess SDKs all expose it (Claude `can_use_tool` +
permission modes; Copilot mandatory `on_permission_request`; Codex
`approval_policy`). It's a real consumer concern for agentic flows
(does this delete files?). But it doesn't translate to OpenAI-compat
HTTP at all, so it has to ride alongside tool registration as a
session-level concept.

**Sketch.** Optional callback on `runtime.session(...)`:

```python
PermissionDecision = Literal["allow", "deny", "ask_user"]
PermissionCallback = Callable[[PermissionRequest], Awaitable[PermissionDecision]]

@dataclass(frozen=True, slots=True)
class PermissionRequest:
    tool_name: str
    tool_args: dict[str, Any]
    reason: str | None
```

Adapters without a permission concept ignore the callback. Adapters
with one wire it to their native handler.

### P3 — Budget caps

**Why.** Claude exposes `max_turns` *and* `max_budget_usd`; the
others have only token-level caps. A unified surface lets consumers
write budget-aware agents portably.

**Sketch.** Optional kwargs on `execute()`:

```python
async def execute(
    self,
    prompt: Prompt,
    *,
    max_turns: int | None = None,
    max_budget_usd: float | None = None,
    max_output_tokens: int | None = None,
    ...
) -> RuntimeResult: ...
```

Adapters honour what they can; ignore what they can't. Document the
trade-off (Codex / Copilot won't enforce `max_budget_usd` natively —
the adapter could enforce it by tracking cumulative cost across turns
inside the session and aborting; or it could just no-op).

### P3 — Lifecycle observability hooks

**Why.** Claude and Copilot both expose first-class hook taxonomies
(`PreToolUse`, `PostToolUse`, `UserPromptSubmit`, `SessionStart`,
etc.). Consumers building dashboards / cost tracking / audit logs
re-invent the same plumbing per adapter today.

**Sketch.** Single observer registration with a discriminated event
union — narrower than the streaming events because these are
*observation*, not *control*:

```python
@dataclass(frozen=True, slots=True)
class HookEvent:
    kind: Literal[
        "session_start", "session_end",
        "user_prompt_submit",
        "pre_tool_use", "post_tool_use", "tool_failure",
        "pre_compact", "rate_limit",
    ]
    session_id: str | None
    payload: dict[str, Any]

runtime.session(on_event=lambda e: log(e))
```

Adapters that don't have a native hook (Codex, OpenAI-compat) can
synthesise the obvious ones from the message/event stream. Adapters
that do (Claude, Copilot) wire it through.

### P3 — Agent Skills

**Why.** Since this roadmap was first written, "Skills" has
converged into a **cross-vendor filesystem convention** — a folder
containing a `SKILL.md` (YAML frontmatter + Markdown body) describing
a capability the model can autonomously invoke. Four of airframe's
shipped adapters now implement it the same way:

* **Claude Agent SDK** — `ClaudeAgentOptions(setting_sources=["user",
  "project"], skills="all" | [...names])`. Discovers
  `.claude/skills/*/SKILL.md` and `~/.claude/skills/*/SKILL.md`.
* **GitHub Copilot SDK** — `skill_directories=`, `disabled_skills=`
  on `create_session`; `skill.invoked` event. SKILL.md support added
  April 2026, deliberately cross-compatible with Claude/Cursor/Codex
  skill bundles.
* **Kimi Agent SDK** — "knowledge-based extensions discovered from
  layered directories, grouped by scope and injected into the system
  prompt." Reuses Kimi Code config; no Python-level skills API.
* **OpenCode (`opencode serve`)** — first-party auto-discovery from
  `.opencode/skills/`, `~/.config/opencode/skills/`, plus deliberately
  also `.claude/skills/` and `.agents/skills/` so a single bundle
  works cross-vendor. Access gated via `opencode.json`
  `permission.skill` (`allow` / `deny` / `ask`).

`bedrock` (Converse), `opencode-zen`, `opencode-go`, and `openrouter`
have no skill concept. The reserved `bedrock-agents` adapter has its
own action-groups / Knowledge-Bases surface that is *not* SKILL.md
and shouldn't be folded in.

The non-obvious bit is that Skills are a **filesystem contract, not
an API one** — every vendor's Python surface is just a *filter* over
what was already discovered on disk; nobody exposes
`register_skill(...)`. That keeps airframe's abstraction small: pipe
"where to look" + "which to enable"; don't try to abstract the
SKILL.md authoring format (it's already cross-vendor).

**Sketch.** A capability flag plus a typed `ProviderOptions` namespace,
forwarded at session construction:

```python
class Feature(IntEnum):
    ...
    AGENT_SKILLS = auto()

@dataclass(frozen=True, slots=True)
class SkillsConfig:
    enabled: Literal["all"] | list[str] | None = "all"
    search_paths: list[Path] | None = None    # extra roots beyond defaults
    include_user_global: bool = True          # ~/.claude/skills, etc.

runtime.session(skills=SkillsConfig(enabled=["pdf", "docx"]))
```

Per-adapter mapping:
* Claude → `setting_sources=["user", "project"]` + `skills=...`;
  `search_paths` extend `cwd` / `add_dirs`.
* Copilot → `skill_directories=` + `disabled_skills=` on
  `create_session`.
* Kimi → write/symlink `search_paths` into the layered discovery path
  Kimi Code already walks; filter via Kimi Code config.
* OpenCode → translate `enabled=` to `opencode.json`
  `permission.skill` allow/deny rules; honour the server's
  auto-discovery for `search_paths`.
* `bedrock` / `opencode-zen` / `opencode-go` / `openrouter` →
  `supports(AGENT_SKILLS)` returns `False`; passing a `SkillsConfig`
  raises `UnsupportedBindingError`.

Phase-0 scaffold first — add the `Feature.AGENT_SKILLS` enum entry
and an empty `SkillsConfig` namespace returning `False` from every
adapter — then a substantive phase that wires the four supporting
adapters and adds a conformance test (fixture dir containing a known
SKILL.md ⇒ the model can name and invoke it).

**Don't.** Don't fake skills on non-supporting adapters by inlining
SKILL.md content into the system prompt. `supports()` exists so
consumers can fall back themselves; vendor shimming is the kind of
"helpful" abstraction the codebase deliberately avoids.

### P3 — Typed rate-limit telemetry

**Status:** Shipped on `main` (commit `f995d35`). `Feature.RATE_LIMIT_TELEMETRY`
declared by `ClaudeCodeRuntime` (consumes `RateLimitEvent` on the SDK
message stream, accumulates per-window state) and the
`OpenAICompatibleRuntime` base (`with_raw_response.create` reads the
`x-ratelimit-*` + `retry-after` headers). `RuntimeResult.rate_limit`
and `RuntimeTransientError.rate_limit` carry the typed
`RateLimitInfo`. Streaming-path header capture on OAI-compat is a
follow-up — `rate_limit=None` on that path today.

**Why.** Every vendor surfaces structured quota data and airframe
drops it on the floor. Claude's `RateLimitInfo` exposes typed
windows (`five_hour`, `seven_day`, `seven_day_opus`,
`seven_day_sonnet`, `overage`) with `utilization`, `resets_at`, and
`status` (`allowed` / `allowed_warning` / `rejected`); OpenAI returns
the standard `x-ratelimit-limit-{requests,tokens}` /
`x-ratelimit-remaining-{requests,tokens}` /
`x-ratelimit-reset-{requests,tokens}` / `retry-after` header set;
Copilot exposes per-error `SessionErrorData` with status_code;
Bedrock surfaces throttling exceptions with retry-after metadata.
Today airframe collapses all of this into an opaque
`RuntimeTransientError`, so consumers can't write budget-aware
retry, build "you have X requests left this hour" UX, or feed
quota dashboards without `unwrap()`ing per adapter.

The matrix already flags this at line 70 (`◐ via
RuntimeTransientError`). The HookEvent surface (Phase 5) has a
`rate_limit` event kind with `retry_after_seconds` in the payload,
but it's an *observation* of throttling, not a *measurement* of
remaining quota — and the typed `RateLimitInfo` Claude already
provides gets squashed into an untyped dict on the way out.

**Sketch.** Lift the data to a typed object that rides on both
`RuntimeResult` (when the vendor returns quota data on a successful
call) and `RuntimeTransientError` (when we actually got throttled):

```python
@dataclass(frozen=True, slots=True)
class RateLimitWindow:
    """One quota window."""
    name: str                              # vendor's name: "rpm", "tpm",
                                           # "five_hour", "seven_day", ...
    remaining: int | None = None           # OpenAI-style
    limit: int | None = None               # OpenAI-style
    utilization: float | None = None       # Claude-style (0.0-1.0)
    reset_at: datetime | None = None
    retry_after_seconds: float | None = None
    status: Literal["allowed", "allowed_warning", "rejected"] | None = None

@dataclass(frozen=True, slots=True)
class RateLimitInfo:
    windows: tuple[RateLimitWindow, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)

# Additive: RuntimeResult.rate_limit: RateLimitInfo | None = None
# Additive: RuntimeTransientError.rate_limit: RateLimitInfo | None
```

Per-adapter mapping:
* Claude → translate `RateLimitInfo` from the SDK's
  `RateLimitEvent`; one `RateLimitWindow` per active window. Already
  observed via the `rate_limit` hook — stash the most recent on the
  session and attach to the next `RuntimeResult`.
* OpenAI-compat → switch the create() call to
  `client.chat.completions.with_raw_response.create(...)`, parse the
  six `x-ratelimit-*` headers + `retry-after`, emit two windows
  (requests + tokens). On 429 → `RateLimitError`, extract from
  `exc.response.headers`.
* Kimi → 429s arrive as exceptions; populate `retry_after_seconds`
  from `Retry-After` when present, leave `windows=()` otherwise.
* Bedrock → `ThrottlingException` carries retry-after; surface as a
  single-window `RateLimitInfo` plus headers when present.
* Copilot, OpenCode-server → `RateLimitInfo` on the corresponding
  error event when the SDK exposes it; otherwise omit.

Add `Feature.RATE_LIMIT_TELEMETRY` — adapters returning `True` must
populate either `RuntimeResult.rate_limit` (when quota data was on
the wire) or `RuntimeTransientError.rate_limit` (on throttle).
Adapters returning `False` leave both `None`. Conformance: a unit
test asserts that adapters declaring support never raise a
`RuntimeTransientError` with rate-limit text in the message but
`rate_limit=None`.

### P3 — Reasoning trace on `RuntimeResult`

**Status:** Shipped on `main` (commit `4541eec`). `Feature.REASONING_OUTPUT`
declared by `ClaudeCodeRuntime` (consumes `ThinkingBlock` content
from `AssistantMessage` on the non-streaming path; accumulates
streamed `ReasoningDelta` events on streaming) and the
`OpenAICompatibleRuntime` base (defensively reads
`message.reasoning_content` / `delta.reasoning_content` for
DeepSeek-R1 derivatives — vendors that don't surface reasoning leave
the field `None`). `RuntimeResult.reasoning` carries the snapshot;
streaming `TurnComplete.result.reasoning` holds the concatenation of
all `ReasoningDelta` payloads.

**Why.** P2 streaming defines `ReasoningDelta`, but the
non-streaming `RuntimeResult` has no place to land a finalised
reasoning trace — so consumers calling `execute()` instead of
`stream()` lose it entirely. Claude (`ReasoningItem` /
`thinking` blocks), Copilot (`assistant.reasoning_delta` collapsing
to a final string), Codex (`ReasoningItem` in turn output), and the
OpenAI Responses API all expose it. The capability flag
`REASONING_EFFORT` exists; the *output surface* doesn't.

**Sketch.** Add a sibling field to `RuntimeResult.text`:

```python
@dataclass(frozen=True, slots=True)
class RuntimeResult:
    text: str
    structured: Any
    cost: CostRecord
    finish: str | None
    reasoning: str | None = None       # NEW — final reasoning trace
    rate_limit: RateLimitInfo | None = None
    raw: Any = field(default=None, repr=False)
```

Adapters that declare `Feature.REASONING_EFFORT` and saw a
reasoning block populate `reasoning`; adapters that didn't (or
whose model didn't emit one) leave it `None`. Streaming pairs
naturally — `TurnComplete.result.reasoning` is the concatenation of
all `ReasoningDelta` payloads.

Add `Feature.REASONING_OUTPUT` (distinct from `REASONING_EFFORT`:
the latter is "I can ask the model to think harder"; this is "I
can show you what it thought"). Adapters whose vendor SDK exposes
the reasoning text declare it; the rest leave the field `None`.

### P3 — Per-request metadata / user-id

**Status:** Shipped on `main` (commit `1ce2e76`). `Feature.REQUEST_METADATA`
declared by `ClaudeCodeRuntime` (`user_id` → `ClaudeAgentOptions.user`)
and the `OpenAICompatibleRuntime` base (`user_id` → `user=`, `tags` →
`metadata=` dict on chat.completions, `request_id` → `extra_headers
{"X-Request-ID": ...}`). The other adapters accept `metadata=` and
silently drop the tag — an intentional *soft contract* departure from
the codebase's usual "raise on capability decline" convention, because
the call's correctness doesn't depend on the tag reaching the vendor.
Consumers who care branch on `supports(Feature.REQUEST_METADATA)`.

**Why.** Every vendor SDK accepts a free-form per-request tag for
abuse detection, per-tenant usage attribution, and audit trails:
OpenAI's `user=`, Anthropic's `metadata={"user_id": ...}`, Bedrock's
`clientRequestToken`, Copilot's `headers={"X-User-Id": ...}` pattern.
Consumers running multi-tenant agent UIs all reinvent this per
adapter today (or skip it and weaken their abuse-detection story).

**Sketch.** Optional kwarg on `execute()` / `session()`:

```python
@dataclass(frozen=True, slots=True)
class RequestMetadata:
    user_id: str | None = None         # per-end-user identifier
    request_id: str | None = None      # caller-side correlation id
    tags: dict[str, str] | None = None # arbitrary string→string labels

runtime.session(metadata=RequestMetadata(user_id="acct_1234"))
await session.execute(prompt, metadata=RequestMetadata(request_id="req-abc"))
```

Per-adapter mapping:
* OpenAI-compat → `user=` on `chat.completions.create`; tags →
  best-effort header injection or no-op.
* Claude → `metadata={"user_id": ...}` on the messages call.
* Bedrock → `clientRequestToken` from `request_id`; `user_id` as
  an inference-config metadata tag where supported.
* Adapters with no metadata channel ignore the kwarg silently
  (it's observation only, not behaviour-changing — silent ignore
  is appropriate, unlike capability gates).

Add `Feature.REQUEST_METADATA`. Zero-risk addition: dropping the
field on a non-supporting adapter changes no observable behaviour.

### P3 — Pre-flight token counting

**Status:** Shipped on `main` (this commit). `Feature.COUNT_TOKENS`
flipped on `ClaudeCodeRuntime` (delegates to
`anthropic.AsyncAnthropic.messages.count_tokens` — same auth-resolution
dance as `list_models()`) and the `OpenAICompatibleRuntime` base
(`tiktoken.encoding_for_model` with a fall-back to `o200k_base` for
compat vendors whose model IDs aren't in tiktoken's registry —
approximate but useful). The other four adapters raise
`UnsupportedFeatureError` per the strict capability-gate convention.
v1 supports plain-text prompts only; image/file attachments deferred
until a consumer asks (base64 expansion is non-trivial).

**Why.** Consumers want "is this prompt going to blow the context
window / break my budget" *before* paying for a turn. Anthropic
ships a `count_tokens` endpoint, OpenAI ships `tiktoken`, Bedrock
provides per-model encoders, the Claude Agent SDK exposes
`ContextUsageCategory`. Today consumers reach for `tiktoken` and
hope its tokenisation matches whatever model they're actually
calling — sometimes it doesn't.

**Sketch.** A new protocol method:

```python
async def count_tokens(
    self,
    prompt: Prompt,
    *,
    system: str | None = None,
    model: ProviderModel | None = None,
) -> int:
    """Pre-flight token count for the given prompt against this
    model. Adapters use the vendor's native tokeniser /
    count-tokens endpoint where one exists, an offline encoder
    where one is bundled, or raise UnsupportedFeatureError."""
```

Per-adapter:
* Claude → `anthropic.AsyncAnthropic.messages.count_tokens(...)`
  via the underlying SDK (per CLAUDE.md: prefer the official SDK
  over hand-rolled HTTP).
* OpenAI-compat → vendor-bundled `tiktoken` encoding for the
  model when available; raise `UnsupportedFeatureError` for compat
  vendors with no encoder.
* Bedrock → per-family encoder (Anthropic / Llama / Nova) from the
  Bedrock service helpers.
* Kimi / Copilot / OpenCode → defer until the vendor SDK ships a
  counter; declare `False`.

Add `Feature.COUNT_TOKENS`. The return type is deliberately a flat
`int` (not a structured breakdown of system + prompt + reserved
output) — adapters that want richer detail can additionally
surface it on `ContextUsageCategory`-style telemetry.

### P3 — Prompt-cache controls (keys + retention)

**Status:** Shipped on `main` (this commit). `Feature.PROMPT_CACHE_CONTROL`
declared by the `OpenAICompatibleRuntime` base (`CacheConfig.key` →
`prompt_cache_key=`; `retention="short"`/`"long"` →
`"in_memory"`/`"24h"`). The portable `cache=` value takes precedence
over the OpenAI-specific `OpenAICompatOptions.prompt_cache_key` so
consumers setting both get the cross-vendor surface through. All
other adapters accept `cache=` and silently drop it per the soft
contract — same shape as `metadata=`.

**Why.** §1's matrix at lines 55–56 splits prompt caching into
*read stats* (✓ shipped — surfaced in `CostRecord`) and *controls*
(✗ — no airframe surface). OpenAI exposes `prompt_cache_key` and
`prompt_cache_retention: "in_memory" | "24h"`; Anthropic exposes
`cache_control` markers on the messages-API content blocks
(`{"type": "ephemeral"}` / `{"type": "persistent"}` with a TTL).
For long-running agentic workflows the consumer knows the cache key
better than the vendor's heuristic ("this same system prompt +
codebase context across every turn of this session" — a stable
key buys 90%+ cache hit rates instead of the 30-50% the heuristic
manages).

**Sketch.** Session-level config that adapters thread into their
native cache channel:

```python
@dataclass(frozen=True, slots=True)
class CacheConfig:
    key: str | None = None                     # stable across calls
    retention: Literal["short", "long"] | None = None
    # "short" ≈ in-memory / 5min; "long" ≈ persistent / 1h+
    # adapter maps to vendor's nearest equivalent

runtime.session(cache=CacheConfig(key="agent-foo:session-42", retention="long"))
```

Per-adapter:
* OpenAI-compat → `prompt_cache_key=` + `prompt_cache_retention=`
  ("short" → `"in_memory"`, "long" → `"24h"`).
* Claude Agent SDK → not directly exposed; the agent SDK manages
  caching via session warmth. Stash the key as session metadata
  for future use; declare `False` until the SDK opens the surface.
* Bedrock → `Anthropic` family on Bedrock honours
  `cache_control`; map at the message-construction site.
* Others → `False`.

Add `Feature.PROMPT_CACHE_CONTROL`. The retention literal is
deliberately coarse (`short` / `long`) because vendor windows
vary; consumers wanting precise control should `unwrap()`.

### P3 — Slash commands as portable assets

**Why.** Exact analog to the [Agent Skills](#p3--agent-skills)
entry above. Claude (`.claude/commands/*.md`), OpenCode
(`.opencode/command/*.md`), Copilot (`.github/copilot/commands/`),
and Kimi all expose filesystem-defined, user-invoked commands
following the same YAML-frontmatter-plus-Markdown convention
SKILL.md uses. Same "filesystem contract, not API" pattern; same
"airframe pipes where/which, doesn't abstract authoring" stance.

Slash commands differ from skills in *who triggers them*: skills
are model-invoked autonomously; slash commands are user-invoked
explicitly (typed into a UI or selected from a palette). That
matters for the API surface — slash commands need an
*invocation* surface, not just a *configuration* surface, so the
consumer can wire "user typed `/refactor`" through.

**Sketch.**

```python
class Feature(IntEnum):
    ...
    SLASH_COMMANDS = auto()

@dataclass(frozen=True, slots=True)
class SlashCommandsConfig:
    enabled: Literal["all"] | list[str] | None = "all"
    search_paths: list[Path] | None = None
    include_user_global: bool = True

# Configuration: identical pattern to SkillsConfig.
runtime.session(slash_commands=SlashCommandsConfig(enabled=["refactor", "explain"]))

# Discovery — for surfacing them in the consumer's UI palette:
@dataclass(frozen=True, slots=True)
class SlashCommand:
    name: str
    description: str
    arg_schema: type[BaseModel] | None    # None for arg-less commands

async def list_slash_commands(self) -> list[SlashCommand]: ...

# Invocation — equivalent to execute() with a special prompt shape:
await session.execute(SlashCommandInvocation(name="refactor",
                                              args={"target": "foo.py"}))
```

Same per-adapter mapping pattern as Skills (Claude /
github-copilot / kimi / opencode wire it; bedrock /
opencode-zen / opencode-go / openrouter return `False`). Spec the
config now, defer the runtime invocation surface to a real phase.

### P4 — Compaction control (not just observation)

**Why.** The lifecycle hooks proposal lets consumers *observe*
compaction (`pre_compact` event); it doesn't let them *configure*
it. Claude has `PreCompact` hooks + `fold_session_summary` +
`session_store`; Copilot has `session.compaction_start` /
`compaction_complete`; OpenAI Responses has a typed
`context_management` param. Long-running sessions (research
agents, daemons, anything that lives > ~100 turns) need a knob —
"compact at 80% of window," "use this summariser prompt,"
"compact now" — not just a notification.

**Sketch.** Session-level config plus an explicit method:

```python
@dataclass(frozen=True, slots=True)
class CompactionConfig:
    trigger: Literal["auto", "manual"] | None = "auto"
    threshold_ratio: float | None = None      # 0.0-1.0 of context window
    summary_prompt: str | None = None         # caller-supplied summariser

runtime.session(compaction=CompactionConfig(threshold_ratio=0.8))
await session.compact()  # force now
```

Per-adapter:
* Claude → forward `threshold_ratio` to the agent SDK's
  compaction-trigger knob; `summary_prompt` overrides the
  default summariser; `session.compact()` calls
  `fold_session_summary`.
* OpenAI Responses → `context_management={...}`.
* OpenAI-compat (Chat Completions) → declines: no server-side
  state, no compaction concept.
* Others → `False` until the vendor surfaces controls.

Add `Feature.COMPACTION_CONTROL`. Lower priority because the
*observation* side (`pre_compact` event) already gives most
consumers what they need; control is for the long-tail of
multi-hour sessions.

### Briefly considered, not yet a section

These cross-cutting gaps exist but don't warrant a full §3 entry
yet — either they belong in §5 sequencing, or they're awaiting a
concrete consumer ask, or they're trivial enough to bolt on
opportunistically.

* **HTTP client / `base_url` / proxy injection.** Every vendor SDK
  accepts a custom `httpx.AsyncClient` and `base_url`. Belongs in
  the `ProviderOptions` scaffolding (§6.5) — add fields as
  consumers ask, rather than promoting to a §3 protocol concept.
* **Citations / source attribution.** OpenAI's `file_search`
  annotations and Anthropic Citations both surface
  "where did this claim come from." Real consumer value for
  RAG-style flows, but the shapes diverge enough (per-claim
  fragments vs. per-document references) that abstracting now is
  premature. Hold until a concrete consumer ask pins the shape.
* **Stop sequences.** Universally supported (`stop=` /
  `stop_sequences=`), trivially addable as a kwarg on `execute()`.
  Almost never used in agentic flows (the model decides when to
  stop based on the conversation, not a string match), so low
  priority. Land opportunistically when something else touches
  the `execute()` signature.

### P4 — Working-directory / sandbox

**Why.** Three of four agentic SDKs expose CWD + sandbox toggles.
OpenAI-compat doesn't. A protocol concept here lets consumers write
"run this agent against `/tmp/repo-clone`" portably for the SDKs that
support it.

**Sketch.** Constructor or session-level kwarg:

```python
runtime.session(
    cwd="/tmp/repo-clone",
    sandbox="read-only" | "workspace-write" | "danger-full-access" | None,
    network_access=False,
)
```

Adapters that can't enforce a sandbox raise `UnsupportedBindingError`
if the caller asks for one; ignore the toggle when `None`.

### P4 — Subagents

**Why.** Claude (`agents=`) and Copilot (`custom_agents=`) both
expose programmatic subagents. Codex / OpenAI-compat don't. Low
priority because cross-vendor coverage is half; consumers wanting
multi-agent orchestration today should use a higher-level library.

### Out of scope (probably forever)

* **Batch APIs** — OpenAI-only, doesn't fit the
  "synchronous agent call" mental model. Build a separate
  `airframe.batch` if needed.
* **Fine-tuning** — provider-API surface, not agent-runtime surface.
* **Audio / TTS / STT** — niche enough that the OpenAI SDK's resources
  can be used directly when needed.
* **File checkpointing / rewind** — Claude-specific; one consumer
  doesn't justify a protocol concept.
* **External session storage / mirroring** — Claude-specific. Useful
  but only for one adapter.

---

## 4. Where divergence makes clean abstraction hard

### 4.1 Conversation-state semantics

* **Claude** persists conversations to disk (`~/.claude/projects/...`)
  and can `resume` by UUID across processes.
* **Copilot** persists sessions in its CLI's session store; resumable
  across processes; has `infinite_sessions` config.
* **Codex** has a `thread_id` issued by the CLI on first event;
  `resume_thread(thread_id)` works but no persistent listing surface
  in Python.
* **OpenAI-compat Chat Completions** has no server-side state at all —
  multi-turn lives entirely in the caller's `messages=[]` buffer.
* **OpenAI Responses API** has server-side state via
  `previous_response_id` + `conversation`, but only for OpenAI itself,
  not compat vendors.

Implication: `AgentSession` is easy to *define* but adapters will be
materially different inside. Claude/Copilot/Codex sessions are
references to vendor-managed state; OAI-compat sessions are
client-side rolling message buffers. The protocol can hide this, but
a `session.id` is meaningful only sometimes.

### 4.2 Tool / MCP semantics

* Claude can register **in-process MCP tools** (zero IPC overhead) or
  external MCP servers via stdio/SSE/HTTP.
* Copilot can register custom Python tools via `define_tool` (handler
  is called via JSON-RPC from the CLI) or MCP servers via config.
* Codex has *no* Python tool-registration. Custom tools must be wired
  into the codex CLI's MCP config externally.
* OpenAI Chat Completions takes `tools=[{...}]` per request — every
  tool call is a round-trip: model returns `tool_calls`, caller
  executes, caller sends result as `role="tool"` message, model
  continues. Compat vendors all support this.
* OpenAI Responses API supports the same plus 14 built-in tool types
  and remote MCP via `tools=[{"type":"mcp",...}]`. Compat vendors do
  not.

Implication: there is no single "register a tool" call that works
identically across all four. The proposed Tier 1 / Tier 2 split is the
cleanest carve; Codex will refuse Tier 1, OpenAI-compat will refuse
Tier 2 (unless backed by Responses, which compat vendors won't be).

### 4.3 Built-in tools

Each agentic SDK ships its own catalogue (Claude: Bash, Read, Edit,
Write, Grep, Glob, WebFetch, WebSearch, Task, Skill, TodoWrite, ...;
Codex: shell, edit, web search, todo; Copilot: similar plus skills;
OpenAI Responses: web_search, file_search, code_interpreter, etc.).
Tool *names* don't match across vendors. Tool *semantics* don't match
either — Claude's Read works on local files, Copilot's file ops go
through the session-fs provider, Codex's shell respects `sandbox_mode`.

Airframe should not try to canonicalise these. Two options:

1. Treat the built-in catalogue as opaque vendor metadata —
   `ModelInfo.capabilities` plus an opaque
   `runtime.list_builtin_tools() → list[str]` for UIs.
2. Surface a *minimal* portable subset (`{"shell", "read", "edit", "web_search"}`)
   and let the consumer enable/disable those by family, with the
   adapter mapping each family to whichever native tool fills the role.

Option 1 is honest and doesn't grow the protocol. Option 2 is
seductive but every model will surprise the consumer in some way (a
"read" tool that's actually a sandboxed file proxy is a different
beast from one that hits raw disk).

### 4.4 Permission UX

Subprocess SDKs all have a permission concept because they're running
in a workspace and can do destructive things. The OpenAI-compat HTTP
surface has none — the model returns tool calls and the *caller* runs
them; whether to ask the user is the caller's concern.

The proposed P3 permission callback is consistent across the
subprocess adapters. For OpenAI-compat, the equivalent is the caller
deciding whether to actually execute a returned `tool_call` — which
isn't an adapter concern. So the callback is documented as "honoured
when the runtime has a built-in tool model; consumer-side otherwise."

### 4.5 Reasoning effort encoding

* Claude: `low/medium/high/xhigh/max` + `thinking.budget_tokens`.
* Copilot: implicit set (`reasoning_effort` is a `ReasoningEffort`
  literal — schema not exported as enum here).
* Codex: `minimal/low/medium/high`.
* OpenAI Chat Completions: `minimal/low/medium/high`.

The intersection is `low/medium/high`. `minimal` is on Codex/OpenAI
only; `xhigh/max` is on Claude only. Recommendation: protocol surface
exposes `minimal/low/medium/high`, and adapters that don't have
`minimal` map it to `low` with a debug log; Claude's `xhigh/max` is
not reachable through the protocol (use vendor-specific options if you
need it).

### 4.6 Cost reporting

Only Claude reports `total_cost_usd` directly. Codex/Copilot/OpenAI
return tokens, and adapters compute USD from a per-model pricing
table. This is fine and already lives in airframe. As long as new
features (reasoning tokens, cache writes) are folded into
`CostRecord` rather than leaking out, the abstraction holds.

Add: `reasoning_tokens` field on `CostRecord`. Every SDK reports it
under a different name; canonicalise once.

---

## 5. Migration sequencing (suggested)

> **See also: [implementation-plan.md](./implementation-plan.md)** for
> a phased plan that orders this list by dependency (not just
> priority), adds version targets, calls out the shape-lock gating
> points, and incorporates the §6 pattern recommendations.


1. **`reasoning_tokens` on `CostRecord`** — additive, no breaking
   change. Cheap. Lands as part of any P2 reasoning work.
2. **Streaming (`stream()` method + `RuntimeEvent` union)** — biggest
   user-visible win; doesn't require an `AgentSession` rework. Define
   the event taxonomy first; pick it carefully because changing later
   is painful.
3. **`AgentSession` protocol + `runtime.session()` factory** —
   refactors `execute()` into sugar over a single-turn session. Once
   this lands, multi-turn, cancellation, tool registration, and
   permission callbacks all attach to `AgentSession` rather than
   bloating the runtime-level API.
4. **Cancellation** — once `AgentSession` exists, `await session.cancel()`
   is a one-method addition.
5. **Reasoning effort / thinking** — additive kwargs on
   `execute()` / `stream()`.
6. **Vision / file inputs** — additive; `prompt` becomes polymorphic.
7. **Tool registration (Tier 1: function tools)** — session-level
   kwarg.
8. **MCP server refs (Tier 2)** — session-level kwarg.
9. **Permission callback** — session-level kwarg.
10. **Budget caps, hooks, sandbox, subagents** — additive.

Each step is independently shippable. The `AgentSession` step is the
hinge — most of the later items are easier to model once it exists.

### Adapter expansion candidates

Adapter additions run on a parallel track to the protocol-feature
phases above. Each one is signal-gated (concrete consumer or
capability-gap demand triggers prioritisation) and has its own
dedicated dev-doc; the list here is the index.

**Shipped:**

* **`BedrockRuntime`** (v0.6.0) — wraps AWS Bedrock's Converse API;
  the enterprise / IAM-rooted access path. Multi-vendor model catalog
  (Anthropic / Meta / Mistral / Cohere / Amazon Nova) behind one AWS
  auth scheme. Retrospective plan retained for reference at
  [`bedrock-adapter-plan.md`](./bedrock-adapter-plan.md).
* **`KimiRuntime`** (v0.7.0) — wraps Moonshot AI's `kimi-agent-sdk`
  (subprocess-class, thin wrapper around `kimi-cli`). First adapter
  to deliver agent-SDK-class agentics over an open-weight model line
  (Kimi K2 / K2.6 / K2-thinking-turbo). Plan:
  [`kimi-adapter-plan.md`](./kimi-adapter-plan.md).
* **`OpenCodeServerRuntime`** — wraps the bespoke OpenCode HTTP
  agent server (`sst/opencode` — distinct from the `OpenCodeZenRuntime`
  / `OpenCodeGoRuntime` gateway adapters that share the brand). The
  lineup's first **model-agnostic agent adapter**: fronts whichever
  upstream `opencode auth login` has configured (Anthropic / OpenAI /
  OpenRouter / Ollama / vLLM / Together / Groq / MoonshotAI),
  including ChatGPT-OAuth subscriptions. Server-side sessions, native
  SSE streaming, lifecycle hooks. Function tools / MCP / permission-
  reply currently decline pending SDK surface — opencode-ai 0.1.0a36
  has no `client.mcp` / `client.permission` resources; flags will
  flip True once the SDK catches up. Plan:
  [`opencode-adapter-plan.md`](./opencode-adapter-plan.md).

**Phase 1 candidates** (open-weight or new-shape agentics; each
mergeable in parallel — disjoint files):

* **`MistralRuntime`** — wraps Mistral AI's Agents API via
  `mistralai[agents]`. New shape in the lineup: hosted multi-tenant
  agent service with server-managed conversations, multi-agent
  handoffs (server/client execution), agent versioning + aliases, and
  connectors (Mistral's MCP equivalent). Architecturally prep for
  the future `BedrockAgentsRuntime` sibling. Full plan:
  [`mistral-adapter-plan.md`](./mistral-adapter-plan.md).

**Signal-gated post-1.0:**

* **`GeminiRuntime`** — direct wrapper around `google-genai`
  (Developer API + Vertex). Fills the obvious gap in the
  model-house-bound matrix. Full plan:
  [`google-genai-adapter-plan.md`](./google-genai-adapter-plan.md).

These adapters share a common iteration shape (ABCDEF — scaffold,
execute/stream/cancel, polymorphic prompt + reasoning, tools +
permission, hooks + budget, wrap-up) and can land mergeable in
parallel since they touch disjoint files.

---

## 6. Patterns from mature abstraction frameworks

The 60-field `ClaudeAgentOptions` makes a deeper question concrete:
how have other "one API, many vendor backends" frameworks resolved
the tension between portable surface and vendor-specific sprawl?
Airframe is JDBC for agent SDKs by stated intent; it's worth being
explicit about which JDBC-era and modern patterns transfer and which
are anti-patterns to avoid.

### 6.1 API / SDK split (SLF4J, OpenTelemetry)

[SLF4J](https://www.slf4j.org/manual.html) and
[OpenTelemetry](https://opentelemetry.io/docs/specs/otel/overview/)
ship two layers: a tiny *API* package that instrumented code depends
on, and an *SDK / binding / provider* that's installed separately at
deployment time. The OpenTelemetry spec is explicit:
"Instrumentation authors MUST NOT directly reference any SDK package
of any kind, only the API." No-op default implementations let
API-only code run even with no SDK present.

**Why it works.** It collapses the library/application dependency
problem. A reusable helper library can do
`from airframe import AgentRuntime` without forcing its consumer
onto any particular adapter — the *consumer* picks one at deployment
by `pip install airframe-agents[claude]`.

**Translated to airframe.** Today `airframe` and the adapters ship in
one distribution gated by pip extras. That works while the protocol
is still moving, but as it stabilises consider splitting:

* `airframe-spec` — protocol types, `AgentRuntime`, `RuntimeResult`,
  `ProviderModel`, `ModelInfo`, `CostRecord`, error hierarchy.
  Zero runtime deps. A no-op `NullRuntime` that raises a clear "no
  SDK installed" error.
* `airframe-adapters-claude` / `-copilot` / `-codex` / `-openai-compat`
  — actual implementations, depending on `airframe-spec` and their
  respective vendor SDK.

The current entry-point discovery (`list_providers()` filtering by
installed extras) is already the SLF4J ServiceLoader pattern in
miniature; this just makes the boundary visible at the dist level.

### 6.2 Driver registration via SPI / entry points (JDBC, SLF4J)

JDBC 4.0 moved from `Class.forName("com.vendor.Driver")` to
[ServiceLoader auto-registration](https://en.wikipedia.org/wiki/Java_Database_Connectivity)
via `META-INF/services/java.sql.Driver`. SLF4J's 2.0 binding
discovery works the same way. Each driver jar on the classpath
registers itself; `DriverManager.getConnection(url)` picks the first
driver that says "yes" to the URL.

**Airframe today** does this in spirit through `discovery.py` —
adapters live under `airframe.adapters.*` and `list_providers()`
returns the installed ones. Worth formalising the contract: each
adapter declares `PROVIDER_ID` / `REQUIRES_PACKAGE` / `EXTRA_NAME` as
ClassVars (already true). Third-party adapters could register via
[Python entry points](https://packaging.python.org/en/latest/specifications/entry-points/)
under a `airframe.adapters` group, so a hypothetical
`airframe-adapters-together` package needs no patch to core.

**Anti-pattern.** JDBC's `jdbc:postgresql://...` vs
`jdbc:oracle:thin:@host:port/sid` URL syntax is wildly inconsistent
because each driver invented its own scheme. Airframe's typed
`ProviderModel(provider_id, model_id)` already avoids this — keep
it. Don't introduce a "runtime URL" string format.

### 6.3 Capability negotiation predicates (JDBC, SQLAlchemy, Vercel)

JDBC's [`DatabaseMetaData`](https://docs.oracle.com/javase/8/docs/api/java/sql/DatabaseMetaData.html)
has ~150 `supportsXxx()` predicates: `supportsSavepoints()`,
`supportsBatchUpdates()`, `supportsResultSetType(int)`. SQLAlchemy's
`Dialect` ships flags like `supports_returning`,
`supports_native_enum`, `supports_native_uuid`,
`supports_server_side_cursors`, `supports_statement_cache`,
`supports_multivalues_insert`, `supports_identity_columns`. Vercel
AI SDK exposes a `model.profile` dict with `tool_calling`,
`reasoning_output`, `image_inputs`, etc.

**Airframe today** has a single coarse predicate (`validate_binding`)
and five capability strings on `ModelInfo`. That's fine as a first
pass but inverts cleanly into a typed feature gate:

```python
class CapabilityProbe(Protocol):
    def supports(self, feature: Feature, model: ProviderModel | None = None) -> bool: ...

class Feature(Enum):
    STRUCTURED_OUTPUT_JSON_SCHEMA = "structured_output_json_schema"
    STRUCTURED_OUTPUT_STRICT = "structured_output_strict"
    VISION_INPUT = "vision_input"
    FILE_INPUT = "file_input"
    STREAMING = "streaming"
    TOOLS_FUNCTION = "tools_function"
    TOOLS_MCP_REMOTE = "tools_mcp_remote"
    TOOLS_MCP_IN_PROCESS = "tools_mcp_in_process"
    REASONING_EFFORT = "reasoning_effort"
    REASONING_BUDGET_TOKENS = "reasoning_budget_tokens"
    SESSION_RESUME = "session_resume"
    PERMISSION_CALLBACK = "permission_callback"
    SANDBOX = "sandbox"
    CANCEL = "cancel"
```

Consumers can branch (`if runtime.supports(Feature.TOOLS_MCP_REMOTE,
model): ...`) without sniffing `provider_id`. Adapters answer locally
— no network call. This is exactly the right shape for the
divergences §4 calls out (reasoning-effort encoding intersection;
tool registration tiers; MCP-only-on-Responses).

**Anti-pattern.** Don't let capability probes degrade into a sea of
silent fallbacks. If a consumer asks for a feature on a model that
doesn't support it, the call should fail fast with
`UnsupportedFeatureError`, not "best effort succeed." JDBC's
`getObject(int)` problem — type erasure that punts everything to
runtime — is the version of this to avoid.

### 6.4 Unwrap / escape hatch (JDBC `Wrapper`, OTel)

[JDBC 4.0's `Wrapper` interface](https://docs.oracle.com/javase/8/docs/api/java/sql/Wrapper.html)
gives every JDBC object an `unwrap(Class<T>)` method that yields the
vendor-specific underlying object. `Connection` is portable;
`((OracleConnection) conn.unwrap(OracleConnection.class)).setEndToEndMetrics(...)`
is the documented escape hatch when you need an Oracle-only feature.

**Airframe today** has this informally: `RuntimeResult.raw` carries
the vendor object. Formalising on the runtime itself helps:

```python
class AgentRuntime(Protocol):
    ...
    def unwrap(self, cls: type[T]) -> T:
        """Return the underlying vendor object cast to ``cls``,
        or raise ``TypeError`` if this runtime can't satisfy it."""
```

So `runtime.unwrap(ClaudeSDKClient)` returns the live
`ClaudeSDKClient` for callers that need a Claude-only knob; same
runtime in a different deployment refuses cleanly. Pairs naturally
with `isinstance` checks the consumer can do *before* unwrapping.

### 6.5 Vendor properties via typed namespaces (JDBC, Spring Cloud Stream, Vercel)

JDBC handles vendor knobs through
[`Properties`](https://docs.oracle.com/javase/8/docs/api/java/sql/DriverManager.html#getConnection-java.lang.String-java.util.Properties-)
passed to `getConnection(url, props)` — each driver documents which
keys it honours.
[Spring Cloud Stream](https://docs.spring.io/spring-cloud-stream/reference/spring-cloud-stream/binders.html)
namespaces them: `spring.cloud.stream.kafka.*` vs
`spring.cloud.stream.rabbit.*` — the application code is
binder-neutral, the YAML knows which binder it's tuning.
[Vercel AI SDK 5](https://ai-sdk.dev/docs/ai-sdk-core/middleware)
adds `providerOptions` on every call: portable kwargs at the top,
`providerOptions.anthropic.*` / `providerOptions.openai.*` for
vendor-specific tuning.

**This is the answer to `ClaudeAgentOptions` sprawl.** Don't try to
mirror 60 fields on `AgentRuntime.execute()`. Don't bury everything
in a `dict[str, Any]` either. Do the Vercel split:

```python
async def execute(
    self,
    prompt: Prompt,
    *,
    schema: type[BaseModel] | None = None,
    system: str | None = None,
    model: ProviderModel | None = None,
    timeout: float = 600.0,
    # Portable, P0-P2 from the roadmap above:
    thinking: ThinkingMode | None = None,
    tools: list[FunctionTool] | None = None,
    # Per-vendor namespaces — typed but optional:
    provider_options: ProviderOptions | None = None,
) -> RuntimeResult: ...
```

Where `ProviderOptions` is a tagged-union dataclass:

```python
@dataclass(frozen=True, slots=True)
class ClaudeOptions:
    max_budget_usd: float | None = None
    fork_session: bool = False
    enable_file_checkpointing: bool = False
    sandbox: dict | None = None
    plugins: list | None = None
    # ...whatever Claude-specific knobs we deem worth typing

@dataclass(frozen=True, slots=True)
class CodexOptions:
    sandbox_mode: Literal["read-only", "workspace-write", "danger-full-access"] | None = None
    approval_policy: Literal["never", "on-request", "on-failure", "untrusted"] | None = None
    network_access_enabled: bool | None = None
    # ...

ProviderOptions = ClaudeOptions | CopilotOptions | CodexOptions | OpenAICompatOptions
```

The adapter pattern-matches: a `ClaudeCodeRuntime` accepts
`ClaudeOptions`, ignores or rejects the others. The consumer paying
the abstraction tax (portability) doesn't see the noise; the
consumer paying for vendor power gets typed access without an
escape hatch.

**Anti-pattern.** JMS shipped only `setStringProperty(name, value)` /
`setIntProperty` on `Message` — vendor-specific message properties
silently broke portability and there was no way to know at compile
time whether a consumer understood your property. Typed
`ProviderOptions` classes avoid this: the consumer can't pass
`ClaudeOptions` to `CopilotRuntime` accidentally — mypy catches it.

### 6.6 Layered hierarchy: Factory → Session → Operation (JDBC, JPA, Hibernate)

JDBC's [`DataSource` → `Connection` → `Statement` → `ResultSet`](https://en.wikipedia.org/wiki/Java_Database_Connectivity)
is the canonical staircase. Each layer is a distinct lifecycle
scope: data source lives for the app, connection for a session,
statement for a query, result set for a cursor. JPA splits
`EntityManagerFactory` → `EntityManager` → `Query`. The pattern is
older than JDBC (it's just *factory + handle + operation*) but JDBC
made it the industry default.

**Airframe today** mostly hides the middle. The proposed
`AgentSession` (roadmap §3, P1) makes this hierarchy explicit:

* `AgentRuntime` — long-lived; owns auth, subprocess pool, HTTP client.
* `AgentSession` — task-scoped; owns conversation state, tool
  registry, permission callback, working directory.
* `execute()` / `stream()` — turn-scoped; owns the prompt, the
  schema, the budget for this one call.

The mapping to existing JDBC analogues:

| JDBC | airframe (proposed) |
| --- | --- |
| `DataSource` | `AgentRuntime` |
| `Connection` | `AgentSession` |
| `PreparedStatement` | a session with `schema=` pre-bound — Claude/Copilot already cache on `(model, system, schema)` |
| `ResultSet` | `AsyncIterator[RuntimeEvent]` from `stream()` |
| `DatabaseMetaData` | `runtime.supports(Feature)` + `list_models()` |
| `Connection.unwrap(...)` | `runtime.unwrap(NativeClient)` |

**Anti-pattern.** JDBC also shipped `CallableStatement` and the
infamous `getObject(int)` that returns `Object` and forces casts.
The pattern translated naively would be "one method that returns
`Any`." Keep the typed-result staircase; let structured output and
tool calls be different methods, not different overloads of one.

### 6.7 CallbackHandler pattern (JAAS)

[`javax.security.auth.callback.CallbackHandler`](https://docs.oracle.com/javase/8/docs/api/javax/security/auth/callback/CallbackHandler.html)
is the cleanest answer to "how does a portable auth provider ask the
caller for information without coupling to UI." The
`LoginModule` builds an array of typed `Callback` objects
(`NameCallback`, `PasswordCallback`, `ConfirmationCallback`,
`TextOutputCallback`); the application's `CallbackHandler.handle()`
inspects each one's type and fills it in. GUI app pops a dialog;
CLI prompts stdin; server reads from config. The auth module
doesn't know or care.

**This is exactly the right shape for airframe's permission UX**
(roadmap §3, P3). Subprocess-based adapters (Claude, Copilot, Codex)
all have a "the agent wants to run tool X, can it?" moment. The
handler shape:

```python
class PermissionCallback(Protocol):
    async def handle(self, request: PermissionRequest) -> PermissionDecision: ...

@dataclass(frozen=True, slots=True)
class PermissionRequest:
    tool_name: str
    tool_args: dict[str, Any]
    reason: str | None
    # Extension point: new fields added here are no-ops for old handlers,
    # not breaking changes — same forward-compat property JAAS relies on.

PermissionDecision = Literal["allow", "deny", "defer"]
```

Adapters without a permission concept (`OpenAICompatibleRuntime`)
silently ignore the callback. Adapters with one wire it through.
JAAS's forward-compatibility property — `UnsupportedCallbackException`
for callback types a handler doesn't recognise, rather than a hard
break — is worth preserving.

### 6.8 Strict spec + reference impl + conformance suite (JSR, OCI, CSI)

JDBC ships with a JCK / TCK; the OCI runtime spec ships with
[conformance tests](https://github.com/opencontainers/runtime-tools)
that runc, crun, kata, and others must pass to claim conformance;
[CSI](https://github.com/container-storage-interface/spec) does the
same for storage drivers. The pattern: tight spec + reference
implementation + executable conformance test suite. Third-party
implementers can land confidently.

**For airframe.** A `airframe-tck` package containing parameterised
pytest fixtures that exercise every adapter against the canonical
contract:

* Every adapter must classify a 401 as `RuntimeAuthError`.
* Every adapter must populate `CostRecord.input_tokens > 0` after a
  successful call.
* Every adapter that claims `Feature.STRUCTURED_OUTPUT_JSON_SCHEMA`
  must round-trip a Pydantic model end-to-end.
* `close()` must be idempotent.
* `validate_binding` must agree with `execute` (no
  "validate says yes, execute raises UnsupportedBindingError").

A third-party `airframe-adapters-together` author runs
`pytest --airframe-tck` against their adapter and ships when green.
This is the missing piece between "airframe has a Protocol" and
"third-party adapters land safely."

### 6.9 Middleware / wrapping (Vercel AI SDK 5)

[Vercel's `wrapLanguageModel(model, middleware)`](https://ai-sdk.dev/docs/ai-sdk-core/middleware)
lets you compose cross-cutting behaviour — logging, caching,
guardrails, RAG, retries, reasoning extraction — that works against
any provider. The middleware shape is a small set of async hooks
(`wrapGenerate`, `wrapStream`, `transformParams`); it does not see
vendor-specific kwargs unless it asks for them via
`providerOptions`.

**Airframe today** keeps "retry policy, fallback across vendors,
conversation memory" out of the protocol on purpose (`README.md`,
"Anything *above* the protocol ... is left to the consumer").
That's a coherent stance, but a documented middleware *shape* would
let cross-cutting code be portable across adapters even when it's
not in the core. A reusable retry policy that knows it can react to
`RuntimeAuthError` / `RuntimeTransientError` is the value-add; one
that knows it can also wrap `stream()` is more so.

The minimum-viable contract:

```python
class RuntimeMiddleware(Protocol):
    async def before_execute(self, ctx: ExecuteContext) -> None: ...
    async def after_execute(self, ctx: ExecuteContext, result: RuntimeResult) -> RuntimeResult: ...
    async def on_error(self, ctx: ExecuteContext, exc: Exception) -> Exception | None: ...
    # stream variants similar
```

`runtime = with_middleware(ClaudeCodeRuntime(), [TracingMiddleware(), RetryMiddleware()])`
returns something that still satisfies `AgentRuntime`. Composable;
no inheritance gymnastics.

### 6.10 Standardised metadata vs vendor-specific (LangChain)

[LangChain `AIMessage`](https://docs.langchain.com/oss/python/langchain/models)
splits its metadata cleanly: a `usage_metadata` dict with canonical
`input_tokens`, `output_tokens`, `total_tokens` (works on every
provider); a `response_metadata` blob with whatever the vendor
shipped (logprobs on OpenAI, billing tier on Anthropic, etc.).

**Airframe today** already does the right thing here:
`CostRecord` is canonical, `RuntimeResult.raw` is the opaque vendor
blob. Worth keeping as a hard rule: *no vendor-specific field gets
promoted onto `CostRecord` or `RuntimeResult` unless every adapter
populates it.* The discipline that kept `CostRecord` clean (no
"reasoning_tokens" silently shaped like one vendor's) is the
discipline to keep.

The flip side — and a thing to add — is a typed *vendor extension*
slot. Roadmap §3 already proposes `reasoning_tokens` on `CostRecord`
because every SDK reports it under a different name; canonicalise
once. Same logic will apply later for "cache_hit_ratio,"
"safety_filter_triggered," etc.

### 6.11 Init-string factories (LangChain, SQLAlchemy)

[`init_chat_model("openai:gpt-5.4")`](https://docs.langchain.com/oss/python/langchain/models)
and `create_engine("postgresql+psycopg2://...")` use a string to
pick provider + model in one go. Airframe's
`runtime_for(provider_id)` is the typed equivalent — strictly
better. Don't introduce a string mini-language; the typed pair
`ProviderModel("claude", "claude-haiku-4-5")` is the right shape.

### 6.12 Resisting the LiteLLM trap

[LiteLLM](https://docs.litellm.ai/docs/) takes a different stance:
"every provider is OpenAI Chat Completions shape underneath." That
works for the 80% case (text in, text out, optional tools) and is
extremely successful — but the abstraction visibly leaks once you
need claude-specific cache breakpoints, or Codex's
`approval_policy`, or Copilot's `on_permission_request`. LiteLLM's
answer is `extra_body` / `drop_params` — i.e., an opaque dict that
shifts the type-safety problem to runtime.

**Airframe's positioning is deliberately different.** From the
README: "The protocol is intentionally narrow ... vendor-specific
behaviour ... lives inside each adapter, where it belongs." That's
the right call for an "agentic SDKs" target where vendor surfaces
diverge much more than chat-completions vendors do. The Vercel-style
typed `ProviderOptions` namespaces (§6.5) preserve that stance
without forcing every consumer to drop into `unwrap()`.

### 6.13 What *not* to import from JDBC and friends

Patterns that would actively hurt:

* **JDBC `getObject(int columnIndex)` style untyped accessors.**
  Returning `Any` from anywhere should require a typed
  `runtime.unwrap(...)` ceremony, not be the path of least
  resistance.
* **Hibernate's leaky SQL `Dialect`.** Dialect classes are full of
  SQL fragments because portable HQL still has to emit real SQL.
  Airframe's prompts aren't translated; don't introduce a
  pseudo-prompt-dialect concept that would force one.
* **JMS vendor-specific message properties** (`setStringProperty`).
  A `Map<String, Object>` extension point with no typing is the
  shape that bit-rots fastest. Use the typed `ProviderOptions`
  namespaces instead.
* **JNDI for runtime lookup.** Service-locator patterns made JEE
  app-server-dependent. Don't introduce a runtime registry the
  consumer has to populate at startup; let entry-point discovery
  and explicit `runtime_for(...)` do the work.
* **JDBC URL syntax.** Different per driver, eternal source of
  pain. Stick with `ProviderModel(provider_id, model_id)`.
* **The "lowest common denominator" trap.** If a feature is on three
  of four adapters and meaningful to consumers, surface it with
  graceful refusal on the fourth (`UnsupportedFeatureError`), not
  by hiding it from the API entirely. JDBC's capability predicates
  exist precisely because the alternative — banning savepoints
  because not every DB supports them — would have killed adoption.

### 6.14 Mapping back to the roadmap's flagged divergences

| Divergence (§4) | Pattern that resolves it |
| --- | --- |
| Conversation-state semantics (server vs client-buffered) | §6.6 Factory→Session staircase: `AgentSession` is server-backed where possible, client-buffered for OAI-compat. `session.id` is `Optional[str]`. |
| Tool registration mechanics (in-process / JSON-RPC / not-supported / HTTP round-trip) | §6.3 capability gates (`Feature.TOOLS_FUNCTION`, `Feature.TOOLS_MCP_*`) + §6.5 typed `ProviderOptions` for the bits that don't generalise. |
| Built-in tool catalogues that don't align | §6.3 capability flag (`Feature.WEB_SEARCH`, `Feature.SHELL_EXEC`) for the few that recur; everything else stays vendor-specific via §6.4 unwrap. |
| Permission UX (subprocess-only) | §6.7 JAAS-style `PermissionCallback`. Adapters without a permission concept silently ignore. |
| Reasoning-effort encoding (intersection is `low/medium/high`) | §6.3 capability gate (`Feature.REASONING_EFFORT` for the three-level common case, `Feature.REASONING_BUDGET_TOKENS` for Claude's explicit budget); §6.5 `ProviderOptions` for `xhigh` / `minimal` outliers. |
| `ClaudeAgentOptions` 60-field sprawl | §6.5 typed namespaces — small portable core + `ClaudeOptions` typed dataclass for the rest. |

### 6.15 Recommended next steps (additive to §5 sequencing)

These are *patterns*, not features — they shape how the §5 feature
work lands.

1. **Adopt typed `ProviderOptions` namespaces** *before* adding the
   next per-vendor knob. Once vision / tools / reasoning land, the
   shape of "where does the vendor-specific bit go" is set. Pick
   well now.
2. **Promote `RuntimeResult.raw` into a typed `runtime.unwrap(cls)`**
   on the runtime itself. Documents the escape hatch.
3. **Add `Feature` enum + `runtime.supports(feature, model=None)`**
   as `validate_binding`'s richer sibling. Roadmap §3 already
   gestures at this; making it a named protocol method gives every
   adapter one place to declare its truth.
4. **Publish `airframe-tck`** as soon as the protocol has one more
   stable release after `AgentSession` lands. The bar for
   third-party adapters is high until then; conformance tests lower
   it.
5. **Document a `RuntimeMiddleware` shape** even if airframe core
   doesn't ship batteries. Retry / tracing / cost-aggregation
   middleware that works across adapters is a community win and
   doesn't bloat core.
6. **Keep the API/SDK split implicit until it hurts.** The pip-extra
   model is doing SLF4J's job at smaller scale. Promote to a true
   `airframe-spec` package only if a real library consumer is
   bottlenecked on transitively pulling in subprocess-SDK deps.

---

## 7. Sources

Ground-truth from inspecting installed packages
(`.venv/lib/python3.12/site-packages/`):

- `claude_agent_sdk/types.py`, `claude_agent_sdk/client.py`,
  `claude_agent_sdk/__init__.py` — `claude-agent-sdk` 0.2.82
- `copilot/__init__.py`, `copilot/client.py`, `copilot/session.py`,
  `copilot/generated/session_events.py`,
  `copilot/generated/rpc.py` — `github-copilot-sdk` 0.3.0
- `openai_codex_sdk/types.py`, `openai_codex_sdk/codex.py`,
  `openai_codex_sdk/thread.py`, `openai_codex_sdk/__init__.py` —
  `openai-codex-sdk` 0.1.11
- `openai/resources/chat/completions/completions.py`,
  `openai/resources/responses/responses.py`,
  `openai/types/responses/tool_param.py` — `openai` 2.37.0

Vendor documentation consulted:

**Claude Agent SDK**
- [Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview)
- [`claude-agent-sdk-python` README](https://github.com/anthropics/claude-agent-sdk-python)
- [`claude-agent-sdk-python` CHANGELOG](https://github.com/anthropics/claude-agent-sdk-python/blob/main/CHANGELOG.md)
- [Use the Claude Agent SDK with your Claude plan](https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan) (Agent SDK credit gating from 2026-06-15)

**GitHub Copilot SDK**
- [`github-copilot-sdk` on PyPI](https://pypi.org/project/github-copilot-sdk/)
- [`github/copilot-sdk` getting-started](https://github.com/github/copilot-sdk/blob/main/docs/getting-started.md)
- [GitHub Docs — Using MCP servers with the Copilot SDK](https://docs.github.com/en/copilot/how-tos/copilot-sdk/use-copilot-sdk/mcp-servers)
- [GitHub Docs — Adding MCP servers for GitHub Copilot CLI](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/add-mcp-servers)
- [Microsoft Community Hub — Getting started with the GitHub Copilot SDK](https://techcommunity.microsoft.com/blog/microsoftmissioncriticalblog/getting-started-with-github-copilot-sdk/4510059)
- [AItoolsbee — GitHub Copilot SDK enables agent orchestration via MCP registry and CLI](https://aitoolsbee.com/news/github-copilot-sdk-enables-agent-orchestration-via-mcp-registry-and-cli/) (preview-status dates)

**OpenAI Codex SDK**
- [Codex SDK docs](https://developers.openai.com/codex/sdk)
- [Codex changelog](https://developers.openai.com/codex/changelog)
- [`openai-codex-sdk` on PyPI](https://pypi.org/project/openai-codex-sdk/)
- [`openai/codex` GitHub repo](https://github.com/openai/codex)
- [`openai/codex` PR #18996 — Publish Python SDK with Codex-pinned versioning](https://github.com/openai/codex/pull/18996)
- [`openai/codex/sdk/python`](https://github.com/openai/codex/tree/main/sdk/python)

**OpenAI Python SDK / Responses / Chat Completions**
- [Structured outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs)
- [Connectors and MCP servers (Responses)](https://developers.openai.com/api/docs/guides/tools-connectors-mcp)
- [Responses API reference](https://platform.openai.com/docs/api-reference/responses) (consulted via SDK source — the public-facing URL requires authenticated access)
- [Remote MCP server tool guide](https://platform.openai.com/docs/guides/tools-remote-mcp)
- [`openai/openai-python` GitHub repo](https://github.com/openai/openai-python)

**Abstraction patterns (§6)**
- [JDBC overview tutorial](https://docs.oracle.com/javase/tutorial/jdbc/overview/index.html) and [Wikipedia: JDBC](https://en.wikipedia.org/wiki/Java_Database_Connectivity)
- [`java.sql.Wrapper` API](https://docs.oracle.com/javase/8/docs/api/java/sql/Wrapper.html)
- [`java.sql.DatabaseMetaData` API](https://docs.oracle.com/javase/8/docs/api/java/sql/DatabaseMetaData.html)
- [`DriverManager.getConnection(url, Properties)`](https://docs.oracle.com/javase/8/docs/api/java/sql/DriverManager.html#getConnection-java.lang.String-java.util.Properties-)
- [SLF4J manual](https://www.slf4j.org/manual.html)
- [OpenTelemetry specification overview](https://opentelemetry.io/docs/specs/otel/overview/)
- [SQLAlchemy dialects](https://docs.sqlalchemy.org/en/20/dialects/index.html) and [Dialect internals](https://docs.sqlalchemy.org/en/20/core/internals.html)
- [Spring Cloud Stream binders](https://docs.spring.io/spring-cloud-stream/reference/spring-cloud-stream/binders.html)
- [JAAS CallbackHandler API](https://docs.oracle.com/javase/8/docs/api/javax/security/auth/callback/CallbackHandler.html)
- [Vercel AI SDK — providers and models](https://ai-sdk.dev/docs/foundations/providers-and-models)
- [Vercel AI SDK — language model middleware](https://ai-sdk.dev/docs/ai-sdk-core/middleware)
- [AI SDK 5 introduces LanguageModelV2 spec](https://x.com/aisdk/status/1929836760609292594)
- [LangChain — Chat Models](https://docs.langchain.com/oss/python/langchain/models)
- [LiteLLM docs](https://docs.litellm.ai/docs/)
- [OCI runtime-tools (conformance)](https://github.com/opencontainers/runtime-tools)
- [Container Storage Interface spec](https://github.com/container-storage-interface/spec)
- [Python packaging — entry points specification](https://packaging.python.org/en/latest/specifications/entry-points/)
- [Docker plugin API](https://docs.docker.com/engine/extend/plugin_api/)
