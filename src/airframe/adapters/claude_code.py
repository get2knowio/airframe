"""``ClaudeCodeRuntime`` — :class:`AgentRuntime` over the Claude Agent SDK.

Wraps :class:`claude_agent_sdk.ClaudeSDKClient` to expose Claude's
agent family via the official ``claude-agent-sdk`` package. The SDK
spawns and manages the ``claude`` CLI subprocess; airframe doesn't
allocate ports, juggle passwords, validate model IDs at startup, or
maintain any client code.

**Auth.** Three options, checked in order:

1. ``CLAUDE_CODE_OAUTH_TOKEN`` env var — a long-lived OAuth token
   minted by ``claude setup-token``. Best for CI / non-interactive
   contexts.
2. ``~/.claude/.credentials.json`` — the interactive Claude Code
   OAuth flow's stored token. What you get when you've logged in
   via the ``claude`` CLI on this machine.
3. ``ANTHROPIC_API_KEY`` env var — pay-per-token API access. Useful
   for production deployments without a Max subscription.

**Structured output.** Uses the SDK's native
:attr:`ClaudeAgentOptions.output_format` —
``{"type": "json_schema", "schema": schema.model_json_schema()}``.
The CLI enforces the schema server-side and the validated payload
lands on :attr:`ResultMessage.structured_output`. No tool-forcing,
no MCP shim, no system-prompt prefix.

**Lifecycle.** Phase 1 Iteration G moved per-conversation state off
the runtime onto :class:`ClaudeCodeSession`. The runtime is now
**sessionless** — it holds only the long-lived configuration
(model id, OAuth token override, ``max_turns`` default). Open a
session with :meth:`session`; each session lazily constructs its
own :class:`ClaudeSDKClient` keyed by
``(schema, thinking, attachments, tools, mcp_servers, on_permission,
max_turns)``. Cache-key changes within one session force a
reconnect (the SDK bakes most of those into
:class:`ClaudeAgentOptions` at connect time); separate sessions
never share a client. ``runtime.reset()`` and ``runtime.close()``
are no-ops; ``session.close()`` disconnects the underlying SDK
client and is idempotent.

``runtime.execute(...)`` is sugar for
``runtime.session(...).execute(...) + close()`` — single-turn,
ephemeral subprocess; the same path as the bespoke session but
torn down per call.

**Cost.** The SDK exposes ``total_cost_usd`` on the
``ResultMessage`` — populated directly into the
:class:`CostRecord`. Token counts come from
``ResultMessage.usage``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from pydantic import BaseModel

from airframe.cost import CostRecord
from airframe.errors import (
    AgentRuntimeError,
    RuntimeAuthError,
    RuntimeCancelledError,
    RuntimeProtocolError,
    RuntimeServerStartError,
    RuntimeStructuredOutputError,
    RuntimeTransientError,
    UnsupportedFeatureError,
)
from airframe.events import (
    ReasoningDelta,
    RuntimeEvent,
    TextDelta,
    ToolCallResult,
    ToolCallStart,
    TurnComplete,
)
from airframe.features import Feature
from airframe.inputs import Prompt
from airframe.models import ModelInfo
from airframe.options import ClaudeOptions
from airframe.protocol import (
    AgentRuntime,
    AgentSession,
    ProviderModel,
    RuntimeResult,
    UnsupportedBindingError,
)
from airframe.sessions import (
    _check_budget_supported,
    _check_hooks_supported,
    _check_mcp_servers_supported,
    _check_permission_supported,
    _check_provider_options,
    _check_tools_supported,
    _compose_mcp_headers,
    _enforce_budget_pre_turn,
    _fire_hook_event,
    _mcp_servers_fingerprint,
    _split_prompt_parts,
)
from airframe.thinking import ThinkingMode
from airframe.tools import FunctionTool, McpServerRef

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from airframe.hooks import HookEvent
    from airframe.options import ProviderOptions
    from airframe.permission import PermissionCallback

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Default Claude model when no binding is specified.
DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5"

#: Maximum number of agent turns the SDK is allowed to take inside one
#: ``execute()`` call before the loop is terminated. Briefings / outlines
#: can take 20-40 turns when the model reads files first; we leave room.
DEFAULT_MAX_TURNS = 60


@dataclass(frozen=True, slots=True)
class _ModelMeta:
    """Per-model enrichment for the live ``/v1/models`` response."""

    display_name: str
    context_window: int | None = None
    input_per_1k: float | None = None
    output_per_1k: float | None = None
    capabilities: frozenset[str] = field(default_factory=frozenset)


#: Curated metadata for known Claude models. The live API returns IDs
#: and display names; this table layers context window + pricing on
#: top. Unknown IDs come back without enrichment.
_METADATA: dict[str, _ModelMeta] = {
    "claude-haiku-4-5": _ModelMeta(
        "Claude Haiku 4.5", context_window=200_000, input_per_1k=0.0010, output_per_1k=0.0050
    ),
    "claude-sonnet-4-6": _ModelMeta(
        "Claude Sonnet 4.6", context_window=200_000, input_per_1k=0.0030, output_per_1k=0.0150
    ),
    "claude-opus-4-7": _ModelMeta(
        "Claude Opus 4.7", context_window=200_000, input_per_1k=0.0150, output_per_1k=0.0750
    ),
}


class ClaudeCodeRuntime(AgentRuntime):
    """One Claude Agent SDK client per runtime instance.

    Args:
        model: Default Claude model identifier used when ``execute()``
            is called without a ``ProviderModel`` override. Honours
            ``CLAUDE_MODEL_OVERRIDE`` env var if set for testing.
        max_turns: Hard cap on agent turns within one ``execute()``.
        api_key: Optional explicit Anthropic API key. When ``None``
            (default), auth resolves via the SDK's normal flow:
            ``CLAUDE_CODE_OAUTH_TOKEN`` env var → ``~/.claude/.credentials.json``
            → ``ANTHROPIC_API_KEY`` env var.
    """

    label = "claude_code"

    #: Canonical provider ID this adapter serves.
    PROVIDER_ID: ClassVar[str] = "claude"

    #: Vendor SDK that must be importable for this adapter to work.
    REQUIRES_PACKAGE: ClassVar[str] = "claude_agent_sdk"

    #: pip extra that brings the vendor SDK in.
    EXTRA_NAME: ClassVar[str] = "claude"

    #: Features this runtime exposes today.
    #:
    #: * ``STRUCTURED_OUTPUT_JSON_SCHEMA`` — wired via
    #:   ``ClaudeAgentOptions.output_format`` (Phase 0).
    #: * ``STREAMING`` — wired via :class:`ClaudeCodeSession` using
    #:   ``include_partial_messages=True`` + filtering for
    #:   ``content_block_delta`` / ``thinking_delta`` events on
    #:   :meth:`ClaudeSDKClient.receive_response` (Phase 1, Iteration D).
    #: * ``SESSION_RESUME`` — wired via ``ClaudeAgentOptions.resume``;
    #:   :meth:`AgentRuntime.session` accepts ``resume=<session_id>``
    #:   (Phase 1, Iteration D). The session ID surfaces on
    #:   :attr:`AgentSession.id` after the first turn.
    #: * ``CANCEL`` — wired via :meth:`ClaudeSDKClient.interrupt`
    #:   plus :func:`asyncio.Task.cancel` on the in-flight execute task
    #:   (Phase 1, Iteration D).
    #: * ``REASONING_EFFORT`` — wired via ``ClaudeAgentOptions.effort``
    #:   (``"low" | "medium" | "high"``). The Anthropic SDK has a
    #:   richer enum (``"xhigh"``, ``"max"``); airframe stays on the
    #:   portable intersection. ``"minimal"`` is coerced to ``"low"``
    #:   with a debug-level log since Anthropic has no equivalent
    #:   (Phase 2, Iteration B).
    #: * ``REASONING_BUDGET_TOKENS`` — wired via
    #:   ``ClaudeAgentOptions.thinking = {"type": "enabled",
    #:   "budget_tokens": N}`` when ``thinking={"budget_tokens": N}``
    #:   is passed. Claude-only — no other adapter supports a token
    #:   budget (Phase 2, Iteration B).
    #: * ``VISION_INPUT`` / ``FILE_INPUT`` — both wired via the SDK's
    #:   Read tool (auto-allowed for prompt-attached paths). The
    #:   adapter appends an ``Attached files (use the Read tool):``
    #:   block to the prompt text and adds ``"Read"`` to
    #:   :attr:`ClaudeAgentOptions.allowed_tools`. Cache key includes
    #:   whether attachments are present so a no-attachment → with-
    #:   attachment switch reconnects with the right tools list.
    #:   Path-only in v0; bytes/URL raise (Phase 2, Iteration C).
    #: * ``TOOLS_FUNCTION`` — wired via an in-process MCP server built
    #:   with :func:`claude_agent_sdk.create_sdk_mcp_server` + the
    #:   :func:`claude_agent_sdk.tool` decorator. The SDK dispatches
    #:   tool calls inside the CLI subprocess and we surface
    #:   :class:`~airframe.events.ToolCallStart` /
    #:   :class:`~airframe.events.ToolCallResult` from
    #:   :class:`ToolUseBlock` / :class:`ToolResultBlock` on the
    #:   message stream. Tools join the existing ``_ensure_client``
    #:   cache key (Phase 3, Iteration C).
    #: * ``TOOLS_MCP_STDIO`` / ``TOOLS_MCP_HTTP`` / ``TOOLS_MCP_SSE``
    #:   — wired by translating each :class:`McpServerRef` to the
    #:   matching :class:`McpStdioServerConfig` /
    #:   :class:`McpHttpServerConfig` / :class:`McpSSEServerConfig`
    #:   TypedDict and passing the keyed dict via
    #:   :attr:`ClaudeAgentOptions.mcp_servers`, merged with the
    #:   in-process tools server. ``auth_token=`` becomes an
    #:   ``Authorization: Bearer …`` header on the network
    #:   transports; ``headers=`` passes through verbatim. The
    #:   refs fingerprint joins the cache key — but only
    #:   ``name``, ``transport``, ``command``, ``url``, and the
    #:   sorted header *keys* participate; header values and
    #:   ``auth_token`` never enter the fingerprint (Phase 4,
    #:   Iteration B).
    #: * ``PERMISSION_CALLBACK`` — wired by translating
    #:   :class:`~airframe.permission.PermissionCallback` into the
    #:   SDK's :attr:`ClaudeAgentOptions.can_use_tool` callable. The
    #:   adapter awaits the user's callback per tool-use request,
    #:   maps ``"allow"`` / ``"deny"`` to
    #:   :class:`PermissionResultAllow` / :class:`PermissionResultDeny`,
    #:   and treats ``"defer"`` as ``"allow"`` (with a debug log)
    #:   since the existing ``permission_mode="bypassPermissions"``
    #:   default is already the no-gate behaviour. Callback identity
    #:   joins the ``_ensure_client`` cache key so a callback swap
    #:   forces reconnect (Phase 5, Iteration B).
    #: * ``LIFECYCLE_HOOKS`` — wired by translating each native
    #:   :attr:`ClaudeAgentOptions.hooks` event into a
    #:   :class:`~airframe.hooks.HookEvent` and fanning it out to the
    #:   user's ``on_event=`` callback. Claude is the richest source
    #:   here: every one of airframe's 8 :class:`HookEventKind`
    #:   literals is emittable (``session_start`` is synthesised at
    #:   connect since Claude has no native event for that; the
    #:   other 7 map 1:1 from SDK hooks). Observer raises are
    #:   debug-logged and swallowed so a buggy observer can't break
    #:   the session (Phase 5, Iteration C).
    SUPPORTED_FEATURES: ClassVar[frozenset[Feature]] = frozenset(
        {
            Feature.STRUCTURED_OUTPUT_JSON_SCHEMA,
            Feature.STREAMING,
            Feature.SESSION_RESUME,
            Feature.CANCEL,
            Feature.REASONING_EFFORT,
            Feature.REASONING_BUDGET_TOKENS,
            Feature.VISION_INPUT,
            Feature.FILE_INPUT,
            Feature.TOOLS_FUNCTION,
            Feature.TOOLS_MCP_STDIO,
            Feature.TOOLS_MCP_HTTP,
            Feature.TOOLS_MCP_SSE,
            Feature.PERMISSION_CALLBACK,
            Feature.LIFECYCLE_HOOKS,
            Feature.BUDGET_USD_CAP,
            Feature.BUDGET_TURN_CAP,
        }
    )

    #: The set of :class:`~airframe.hooks.HookEventKind` literals
    #: this adapter can emit through ``on_event=``. Claude is the
    #: richest source — every kind in airframe's enumeration is
    #: supported (``session_start`` is synthesised at connect time
    #: since Claude has no native event for it).
    EMITTABLE_HOOK_KINDS: ClassVar[frozenset[str]] = frozenset(
        {
            "session_start",
            "session_end",
            "user_prompt_submit",
            "pre_tool_use",
            "post_tool_use",
            "tool_failure",
            "pre_compact",
            "rate_limit",
        }
    )

    def __init__(
        self,
        *,
        model: str | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        api_key: str | None = None,
    ) -> None:
        self._default_model = (
            model or os.environ.get("CLAUDE_MODEL_OVERRIDE") or DEFAULT_CLAUDE_MODEL
        )
        self._max_turns = max_turns
        # When the caller explicitly passes a key, plumb it into the
        # SDK's env so its auth resolution picks it up over the OAuth
        # paths. We don't mutate os.environ — we set it per-spawn via
        # ClaudeAgentOptions.env.
        self._api_key_override = api_key
        # Phase 1 Iteration G: the runtime no longer caches a
        # ClaudeSDKClient — sessions own it. The runtime is genuinely
        # sessionless, holding only config (model / max_turns / api key).

    # --- AgentRuntime interface ---------------------------------------------

    async def execute(
        self,
        prompt: Prompt,
        *,
        schema: type[BaseModel] | None = None,
        system: str | None = None,
        persona: str | None = None,
        model: ProviderModel | None = None,
        thinking: ThinkingMode = None,
        timeout: float = 600.0,
    ) -> RuntimeResult:
        # Phase 1 Iteration G: ``execute()`` is documented sugar for
        # ``runtime.session(...).execute(...) + close()``. Single-turn,
        # ephemeral — the underlying ClaudeSDKClient is spawned and
        # disconnected per call. Consumers wanting context warmth across
        # calls open a session explicitly and reuse it.
        del persona  # accepted in the protocol but not consumed by Claude
        sess = self.session(system=system, model=model)
        try:
            return await sess.execute(prompt, schema=schema, thinking=thinking, timeout=timeout)
        finally:
            await sess.close()

    async def reset(self) -> None:
        # Phase 1 Iteration G: the runtime no longer caches a session-
        # scoped client. ``execute()`` opens and closes its session
        # per call, so there is nothing scope-bound to drop here.
        # Kept as a no-op for protocol completeness and back-compat.
        return None

    async def close(self) -> None:
        # Phase 1 Iteration G: the runtime is sessionless — no
        # subprocess, no HTTP client, no long-lived vendor handle to
        # release. Kept as a no-op for protocol completeness.
        return None

    def validate_binding(self, binding: ProviderModel) -> bool:
        return binding.provider_id == self.PROVIDER_ID

    def supports(self, feature: Feature, model: ProviderModel | None = None) -> bool:
        return feature in self.SUPPORTED_FEATURES

    def unwrap(self, cls: type[T]) -> T:
        # Late-import to avoid pulling claude-agent-sdk during module
        # load. Users calling unwrap() have already accepted the SDK
        # dependency by instantiating the runtime.
        from claude_agent_sdk import ClaudeSDKClient

        if isinstance(self, cls):
            return self  # type: ignore[return-value]
        if cls is ClaudeSDKClient:
            # Phase 1 Iteration G moved the SDK client off the runtime
            # onto the session. The runtime is now sessionless, so this
            # type is reachable only via ``session.unwrap(ClaudeSDKClient)``.
            raise TypeError(
                "ClaudeCodeRuntime no longer owns a ClaudeSDKClient — "
                "sessions do. Open a session with `sess = runtime.session(...)`, "
                "run a turn, then call `sess.unwrap(ClaudeSDKClient)`."
            )
        raise TypeError(
            f"ClaudeCodeRuntime cannot unwrap to {cls!r}; only "
            f"ClaudeCodeRuntime is supported on the runtime today. "
            f"Vendor session objects live on AgentSession — use "
            f"session.unwrap(NativeType)."
        )

    def session(
        self,
        *,
        resume: str | None = None,
        system: str | None = None,
        model: ProviderModel | None = None,
        tools: list[FunctionTool] | None = None,
        mcp_servers: list[McpServerRef] | None = None,
        on_permission: PermissionCallback | None = None,
        on_event: Callable[[HookEvent], None] | None = None,
        provider_options: ProviderOptions | None = None,
    ) -> AgentSession:
        """Open a bespoke :class:`ClaudeCodeSession`.

        Phase 1 Iteration D replaces the
        :class:`~airframe.sessions._ThinAgentSession` placeholder with
        a session that owns its own :class:`ClaudeSDKClient` lifecycle:
        streaming via ``include_partial_messages=True``, native
        session resume via :attr:`ClaudeAgentOptions.resume`, and
        cancellation via :meth:`ClaudeSDKClient.interrupt`.

        Args:
            resume: Vendor-assigned Claude session ID to resume — the
                value surfaced on a prior :class:`ResultMessage` /
                :class:`AssistantMessage` ``session_id`` field. ``None``
                opens a fresh session.
            system: System prompt baked into
                :attr:`ClaudeAgentOptions.system_prompt` at connect.
            model: Default :class:`ProviderModel` for every turn in
                the session.
            tools: List of :class:`~airframe.tools.FunctionTool` the
                model may invoke. Translated to an in-process MCP server
                via :func:`claude_agent_sdk.create_sdk_mcp_server` and
                attached via :attr:`ClaudeAgentOptions.mcp_servers`. The
                SDK dispatches; the session surfaces tool events from
                :class:`ToolUseBlock` / :class:`ToolResultBlock` on the
                message stream.
            mcp_servers: List of :class:`~airframe.tools.McpServerRef`
                identifying external MCP servers the model may invoke.
                Each ref is translated to the matching
                :class:`McpStdioServerConfig` /
                :class:`McpHttpServerConfig` /
                :class:`McpSSEServerConfig` TypedDict and merged with
                the in-process tools server (if any) into
                :attr:`ClaudeAgentOptions.mcp_servers`. Tool calls
                routed through external servers surface as the same
                :class:`~airframe.events.ToolCallStart` /
                :class:`~airframe.events.ToolCallResult` events with
                the ``mcp__<server>__`` prefix stripped. Phase 4
                Iteration B.
            on_permission: Phase 5 scaffolding accepted by the
                signature; non-None raises
                :class:`~airframe.errors.UnsupportedFeatureError`
                until Phase 5 Iteration B wires
                :attr:`ClaudeAgentOptions.can_use_tool`.
            on_event: Phase 5 scaffolding accepted by the signature;
                non-None raises until Phase 5 Iteration C wires
                :attr:`ClaudeAgentOptions.hooks` +
                ``include_hook_events=True``.
            provider_options: Optional :class:`ClaudeOptions` namespace
                carrying Claude-only knobs. Three populated fields
                as of v0.5.0:

                * ``append_system_prompt`` — text appended to the
                  resolved system prompt (vs. ``system=`` replacing
                  it). Lands on
                  :attr:`ClaudeAgentOptions.append_system_prompt`.
                * ``fork_session`` — when combined with ``resume=``,
                  forks instead of resuming. Lands on
                  :attr:`ClaudeAgentOptions.fork_session`.
                * ``strict_mcp_config`` — strict MCP tool allowlisting.
                  Lands on
                  :attr:`ClaudeAgentOptions.strict_mcp_config`.

                Passing :class:`CopilotOptions` /
                :class:`CodexOptions` / :class:`OpenAICompatOptions`
                here raises :class:`UnsupportedFeatureError`.
        """
        _check_tools_supported(
            tools,
            adapter_label=self.label,
            feature_supported=self.supports(Feature.TOOLS_FUNCTION),
        )
        _check_mcp_servers_supported(
            mcp_servers,
            adapter_label=self.label,
            supports=self.supports,
        )
        _check_permission_supported(
            on_permission,
            adapter_label=self.label,
            supports=self.supports,
        )
        _check_hooks_supported(
            on_event,
            adapter_label=self.label,
            supports=self.supports,
        )
        _check_provider_options(
            provider_options,
            expected_type=ClaudeOptions,
            adapter_label=self.label,
        )
        # _check_provider_options narrowed the type; the cast keeps mypy
        # happy without a runtime isinstance second-check.
        claude_options = provider_options if isinstance(provider_options, ClaudeOptions) else None
        model_id = self._resolve_model(model) if model is not None else self._default_model
        return ClaudeCodeSession(
            self,
            resume=resume,
            system=system,
            model_id=model_id,
            tools=tools,
            mcp_servers=mcp_servers,
            on_permission=on_permission,
            on_event=on_event,
            provider_options=claude_options,
        )

    async def list_models(self) -> list[ModelInfo]:
        """Return live Claude models from Anthropic's ``/v1/models``.

        Hits ``GET https://api.anthropic.com/v1/models`` directly via
        :mod:`httpx`. The Claude Agent SDK doesn't surface this; we use
        the API key the SDK would have used (``ANTHROPIC_API_KEY``).
        OAuth bearer tokens (Claude Max subscription) work for the
        Messages API but not for ``/v1/models``, so we require an API
        key here.
        """
        import httpx

        api_key = self._api_key_override or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeAuthError(
                "ClaudeCodeRuntime.list_models() needs ANTHROPIC_API_KEY. "
                "OAuth subscription tokens don't work for /v1/models."
            )
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get("https://api.anthropic.com/v1/models", headers=headers)
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in (401, 403):
                raise RuntimeAuthError(f"claude_code: auth: {exc}") from exc
            if status in (429, 502, 503, 504):
                raise RuntimeTransientError(f"claude_code: transient {status}") from exc
            raise RuntimeProtocolError(
                f"claude_code: /v1/models returned {status}", body=exc.response.text[:500]
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeTransientError(f"claude_code: network: {exc}") from exc

        from airframe.models import (
            CAPABILITY_STREAMING,
            CAPABILITY_STRUCTURED_OUTPUT,
            CAPABILITY_TOOLS,
            CAPABILITY_VISION,
        )

        out: list[ModelInfo] = []
        for entry in payload.get("data", []):
            model_id = entry["id"]
            meta = _METADATA.get(model_id, _ModelMeta(entry.get("display_name", model_id)))
            out.append(
                ModelInfo(
                    id=model_id,
                    display_name=meta.display_name,
                    provider_id=self.PROVIDER_ID,
                    context_window=meta.context_window,
                    pricing_input_per_1k_usd=meta.input_per_1k,
                    pricing_output_per_1k_usd=meta.output_per_1k,
                    capabilities=meta.capabilities
                    | frozenset(
                        {
                            CAPABILITY_TOOLS,
                            CAPABILITY_STRUCTURED_OUTPUT,
                            CAPABILITY_STREAMING,
                            CAPABILITY_VISION,
                        }
                    ),
                    raw=entry,
                )
            )
        return out

    # --- Internals ---------------------------------------------------------

    def _resolve_model(self, model: ProviderModel | None) -> str:
        if model is None:
            return self._default_model
        if not self.validate_binding(model):
            raise UnsupportedBindingError(
                f"ClaudeCodeRuntime cannot serve {model.label!r}; "
                f"provider must be {self.PROVIDER_ID!r}"
            )
        return model.model_id

    def _cost_from_result(self, result_msg: Any, *, model_id: str) -> CostRecord:
        usage = result_msg.usage or {}
        # Phase 2 Iteration B: Anthropic's Messages API surfaces hidden
        # reasoning under the ``thinking_tokens`` key when extended
        # thinking is enabled. The field is absent on non-thinking
        # turns; falls through to 0.
        return CostRecord(
            provider_id=self.PROVIDER_ID,
            model_id=model_id,
            cost_usd=result_msg.total_cost_usd,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cache_read_tokens=int(usage.get("cache_read_input_tokens") or 0),
            cache_write_tokens=int(usage.get("cache_creation_input_tokens") or 0),
            finish=result_msg.stop_reason,
            reasoning_tokens=int(usage.get("thinking_tokens") or 0),
        )

    def _classify_exception(self, exc: BaseException) -> Exception:
        """Map Claude SDK exceptions onto airframe's runtime hierarchy."""
        from claude_agent_sdk import (
            ClaudeSDKError,
            CLIConnectionError,
            CLIJSONDecodeError,
            CLINotFoundError,
        )

        if isinstance(exc, CLINotFoundError):
            return RuntimeServerStartError(f"claude_code: CLI not found: {exc}")
        if isinstance(exc, CLIConnectionError):
            return RuntimeTransientError(f"claude_code: connection lost: {exc}")
        if isinstance(exc, CLIJSONDecodeError):
            return RuntimeProtocolError(f"claude_code: bad JSON on stream: {exc}", body=None)
        if isinstance(exc, ClaudeSDKError):
            msg = str(exc).lower()
            if "auth" in msg or "credentials" in msg or "401" in msg:
                return RuntimeAuthError(f"claude_code: auth failure: {exc}")
            if "rate" in msg or "429" in msg or "503" in msg:
                return RuntimeTransientError(f"claude_code: transient: {exc}")
            return AgentRuntimeError(f"claude_code: {exc}")
        return AgentRuntimeError(f"claude_code: unexpected {type(exc).__name__}: {exc}")


class ClaudeCodeSession:
    """Bespoke :class:`~airframe.protocol.AgentSession` for the Claude Agent SDK.

    Phase 1 Iteration D — replaces the
    :class:`~airframe.sessions._ThinAgentSession` placeholder for this
    adapter. Owns one :class:`ClaudeSDKClient` for its lifetime;
    ``system`` / ``model`` / ``resume`` are session-fixed and baked
    into :class:`ClaudeAgentOptions` at connect. Schema can vary per
    :meth:`execute` / :meth:`stream` call — the client reconnects when
    a different schema fingerprint is requested because
    ``output_format`` is connect-time-bound.

    **Streaming.** :meth:`stream` sets
    ``include_partial_messages=True`` on the client and translates
    Anthropic stream events into
    :class:`~airframe.events.TextDelta` / :class:`ReasoningDelta`
    on the fly. The trailing :class:`~airframe.events.TurnComplete`
    carries the canonical :class:`~airframe.cost.CostRecord` built
    from the SDK's :class:`ResultMessage`.

    **Cancellation.** :meth:`cancel` calls
    :meth:`ClaudeSDKClient.interrupt` to abort the in-flight CLI turn
    AND cancels the wrapping :class:`asyncio.Task` for :meth:`execute`.
    The awaiting call raises
    :class:`~airframe.errors.RuntimeCancelledError`.

    **Resume.** ``session(resume=<session_id>)`` forwards the ID into
    :attr:`ClaudeAgentOptions.resume`; the SDK materialises the prior
    conversation from local-disk session store on connect. :attr:`id`
    reflects the live session ID after the first turn — either the
    resumed ID (when ``resume=`` was given) or a fresh one Claude
    assigned.
    """

    def __init__(
        self,
        runtime: ClaudeCodeRuntime,
        *,
        resume: str | None,
        system: str | None,
        model_id: str,
        tools: list[FunctionTool] | None = None,
        mcp_servers: list[McpServerRef] | None = None,
        on_permission: PermissionCallback | None = None,
        on_event: Callable[[HookEvent], None] | None = None,
        provider_options: ClaudeOptions | None = None,
    ) -> None:
        self._runtime = runtime
        self._resume = resume
        self._system = system
        self._model_id = model_id
        # Session-owned client + the cache key that produced it. We
        # reconnect when the requested schema fingerprint changes
        # because ``output_format`` is baked at connect time.
        self._client: Any | None = None
        self._client_key: str | None = None
        self._closed = False
        self._in_flight_task: asyncio.Task[Any] | None = None
        # True from the moment execute()/stream() starts work until it
        # finishes (or is cancelled). Drives cancel()'s no-op-when-idle
        # contract per the AgentSession docstring.
        self._in_flight = False
        self._stream_cancelled = False
        # Populated from the first ResultMessage / AssistantMessage.
        # ``resume=`` callers see this seeded with their resume ID
        # before the first turn; absent that, ``None`` until a turn
        # completes.
        self.id: str | None = resume
        # Phase 3 Iteration C: tools are fixed for the session's
        # lifetime — the in-process MCP server is built at connect
        # time and baked into ClaudeAgentOptions.mcp_servers. The
        # fingerprint joins the cache key so consumers that
        # accidentally pass different tools per call get a clear
        # reconnect rather than silent staleness.
        self._tools: list[FunctionTool] = list(tools or [])
        self._tools_fingerprint = _tools_fingerprint(self._tools)
        # Phase 4 Iteration B: external MCP servers. Translated lazily
        # at connect time and merged with the in-process tools server
        # into ClaudeAgentOptions.mcp_servers. The fingerprint joins
        # the cache key so a refs-change forces reconnect; the set of
        # known server names drives _strip_mcp_prefix on the stream
        # path so consumers see bare tool names regardless of which
        # server routed the call.
        self._mcp_servers: list[McpServerRef] = list(mcp_servers or [])
        self._mcp_servers_fingerprint = _mcp_servers_fingerprint(self._mcp_servers)
        names: set[str] = {ref.name for ref in self._mcp_servers}
        if self._tools:
            names.add(AIRFRAME_MCP_SERVER_NAME)
        self._known_mcp_servers: frozenset[str] = frozenset(names)
        # Phase 5 Iteration B: permission callback baked at connect
        # time via ClaudeAgentOptions.can_use_tool. Callback identity
        # joins the cache key so a callback swap forces reconnect.
        self._on_permission: PermissionCallback | None = on_permission
        self._permission_fingerprint = _permission_fingerprint(on_permission)
        # Phase 5 Iteration C: lifecycle-hook observer. Stored
        # directly on the session — hook callbacks are pure
        # observation and don't affect SDK flow, so we don't need
        # to fingerprint the callback into the cache key (the
        # observer's identity never changes the request semantics).
        self._on_event: Callable[[HookEvent], None] | None = on_event
        # Track whether session_start has fired so re-entrant
        # connects (cache-key invalidations during a session)
        # don't double-fire.
        self._session_start_fired = False
        # ProviderOptions namespace — Claude-specific knobs that don't
        # fit the portable surface. All three populated fields
        # (append_system_prompt, fork_session, strict_mcp_config) are
        # baked at connect time, so the namespace identity joins the
        # cache key fragment.
        self._provider_options: ClaudeOptions | None = provider_options
        # Phase 5 Iteration D: per-session running totals for budget
        # enforcement. ``_cumulative_cost_usd`` accumulates the cost
        # field of every :class:`RuntimeResult` the session produces;
        # ``_turn_count`` counts user-visible turns (one per
        # :meth:`execute` / :meth:`stream` call). Both reset on
        # :meth:`close` but persist across multiple turns of one
        # session — the budget cap is *session-wide*, not per-call.
        self._cumulative_cost_usd: float = 0.0
        self._turn_count: int = 0

    async def execute(
        self,
        prompt: Prompt,
        *,
        schema: type[BaseModel] | None = None,
        thinking: ThinkingMode = None,
        max_turns: int | None = None,
        max_budget_usd: float | None = None,
        timeout: float = 600.0,
    ) -> RuntimeResult:
        if self._closed:
            raise RuntimeError("session is closed")
        _check_budget_supported(
            max_turns=max_turns,
            max_budget_usd=max_budget_usd,
            adapter_label=self._runtime.label,
            supports=self._runtime.supports,
        )
        _enforce_budget_pre_turn(
            max_turns=max_turns,
            max_budget_usd=max_budget_usd,
            cumulative_cost_usd=self._cumulative_cost_usd,
            turn_count=self._turn_count,
            adapter_label=self._runtime.label,
        )
        text, images, files = _split_prompt_parts(
            prompt,
            adapter_label=self._runtime.label,
            supports_vision=True,
            supports_file=True,
        )
        has_attachments = bool(images or files)
        prompt_str = _build_claude_prompt(text, images, files)
        task = asyncio.create_task(
            self._do_execute(
                prompt_str,
                schema=schema,
                thinking=thinking,
                has_attachments=has_attachments,
                max_turns=max_turns,
                timeout=timeout,
            )
        )
        self._in_flight_task = task
        try:
            result = await task
        except asyncio.CancelledError as exc:
            raise RuntimeCancelledError(f"{self._runtime.label}: cancelled") from exc
        finally:
            self._in_flight_task = None
        # Accumulate post-turn so the *next* execute() sees the
        # updated running total. Cost can be None on some result
        # shapes — coerce to 0.0 so the cap math stays well-defined.
        self._turn_count += 1
        self._cumulative_cost_usd += result.cost.cost_usd or 0.0
        return result

    async def _do_execute(
        self,
        prompt: str,
        *,
        schema: type[BaseModel] | None,
        thinking: ThinkingMode,
        has_attachments: bool,
        max_turns: int | None,
        timeout: float,
    ) -> RuntimeResult:
        client = await self._ensure_client(
            schema=schema,
            thinking=thinking,
            has_attachments=has_attachments,
            max_turns=max_turns,
        )
        self._in_flight = True
        try:
            result_msg = await asyncio.wait_for(
                self._query_and_drain(client, prompt),
                timeout=timeout,
            )
        except TimeoutError as exc:
            raise RuntimeTransientError(
                f"{self._runtime.label}: execute timed out after {timeout}s"
            ) from exc
        except Exception as exc:
            raise self._runtime._classify_exception(exc) from exc
        finally:
            self._in_flight = False
        return self._build_result(result_msg, schema=schema)

    async def stream(
        self,
        prompt: Prompt,
        *,
        schema: type[BaseModel] | None = None,
        thinking: ThinkingMode = None,
        max_turns: int | None = None,
        max_budget_usd: float | None = None,
        timeout: float = 600.0,
    ) -> AsyncIterator[RuntimeEvent]:
        if self._closed:
            raise RuntimeError("session is closed")
        _check_budget_supported(
            max_turns=max_turns,
            max_budget_usd=max_budget_usd,
            adapter_label=self._runtime.label,
            supports=self._runtime.supports,
        )
        _enforce_budget_pre_turn(
            max_turns=max_turns,
            max_budget_usd=max_budget_usd,
            cumulative_cost_usd=self._cumulative_cost_usd,
            turn_count=self._turn_count,
            adapter_label=self._runtime.label,
        )
        text, images, files = _split_prompt_parts(
            prompt,
            adapter_label=self._runtime.label,
            supports_vision=True,
            supports_file=True,
        )
        has_attachments = bool(images or files)
        prompt_str = _build_claude_prompt(text, images, files)
        self._stream_cancelled = False
        try:
            client = await self._ensure_client(
                schema=schema,
                thinking=thinking,
                has_attachments=has_attachments,
                max_turns=max_turns,
            )
        except Exception as exc:
            raise self._runtime._classify_exception(exc) from exc

        try:
            await client.query(prompt_str)
        except Exception as exc:
            raise self._runtime._classify_exception(exc) from exc

        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            StreamEvent,
            TextBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
        )

        result_msg: Any = None
        self._in_flight = True
        try:
            async for msg in client.receive_response():
                if self._stream_cancelled:
                    raise RuntimeCancelledError(f"{self._runtime.label}: stream cancelled")
                if isinstance(msg, StreamEvent):
                    for event in _events_from_stream_event(msg):
                        yield event
                    continue
                if isinstance(msg, AssistantMessage):
                    # Fallback when include_partial_messages did not deliver
                    # text via StreamEvent (older CLI versions, or content
                    # blocks that arrived intact). Emit a TextDelta per
                    # TextBlock so consumers always see the assistant text
                    # at least once before TurnComplete.
                    for block in msg.content:
                        if isinstance(block, TextBlock) and block.text:
                            yield TextDelta(text=block.text)
                        elif isinstance(block, ToolUseBlock):
                            yield ToolCallStart(
                                tool_name=_strip_mcp_prefix(block.name, self._known_mcp_servers),
                                tool_call_id=block.id,
                                arguments_preview=_serialize_tool_arguments(block.input),
                            )
                    continue
                if isinstance(msg, UserMessage) and isinstance(msg.content, list):
                    # The SDK turns MCP tool results into UserMessage
                    # turns carrying ToolResultBlock content (Anthropic's
                    # protocol: tool results are user-side). Translate
                    # each into the matching ToolCallResult event.
                    for block in msg.content:
                        if isinstance(block, ToolResultBlock):
                            yield ToolCallResult(
                                tool_call_id=block.tool_use_id,
                                output=_tool_result_output(block),
                                is_error=bool(block.is_error),
                            )
                    continue
                if isinstance(msg, ResultMessage):
                    result_msg = msg
                    break
        except Exception as exc:
            self._in_flight = False
            if isinstance(exc, RuntimeCancelledError):
                raise
            raise self._runtime._classify_exception(exc) from exc

        self._in_flight = False
        result = self._build_result(result_msg, schema=schema)
        # Phase 5 Iteration D: bump session-wide budget trackers
        # before yielding TurnComplete so a consumer observing the
        # event sees the updated cumulative state via subsequent
        # execute() / stream() calls.
        self._turn_count += 1
        self._cumulative_cost_usd += result.cost.cost_usd or 0.0
        yield TurnComplete(result=result)

    async def cancel(self) -> None:
        # No-op when no turn is in flight — per the AgentSession contract.
        if not self._in_flight:
            return
        # Signal the stream() generator to raise on its next yield boundary.
        self._stream_cancelled = True
        # Abort the in-flight execute() task, if any.
        task = self._in_flight_task
        if task is not None and not task.done():
            task.cancel()
        # Ask the SDK to interrupt the CLI turn.
        client = self._client
        if client is not None:
            try:
                await client.interrupt()
            except Exception as exc:  # noqa: BLE001 — cancellation never raises
                logger.debug("%s.session_interrupt_failed error=%s", self._runtime.label, exc)

    async def close(self) -> None:
        # Phase 5 Iteration C: fire session_end before tearing down
        # if we ever connected (so consumer observers see a clean
        # start→end pair). Don't fire if the session never connected,
        # and don't re-fire on repeat close() calls (close is
        # idempotent).
        if self._session_start_fired and not self._closed:
            _fire_hook_event(
                self._on_event,
                "session_end",
                session_id=self.id,
                payload={"model": self._model_id},
            )
        self._closed = True
        client = self._client
        self._client = None
        self._client_key = None
        if client is None:
            return
        try:
            await client.disconnect()
        except Exception as exc:  # noqa: BLE001 — teardown never raises
            logger.debug("%s.session_close_failed error=%s", self._runtime.label, exc)

    def unwrap(self, cls: type[T]) -> T:
        from claude_agent_sdk import ClaudeSDKClient

        if isinstance(self, cls):
            return self  # type: ignore[return-value]
        if cls is ClaudeSDKClient:
            if self._client is None:
                raise TypeError(
                    "ClaudeCodeSession.unwrap(ClaudeSDKClient): no client "
                    "exists yet — call execute() or stream() first."
                )
            return self._client  # type: ignore[return-value]
        raise TypeError(
            f"ClaudeCodeSession cannot unwrap to {cls!r}; supported types are "
            f"ClaudeCodeSession and claude_agent_sdk.ClaudeSDKClient."
        )

    async def _ensure_client(
        self,
        *,
        schema: type[BaseModel] | None,
        thinking: ThinkingMode = None,
        has_attachments: bool = False,
        max_turns: int | None = None,
    ) -> Any:
        # Reconnect when the schema OR thinking OR attachments OR tools
        # OR mcp_servers OR max_turns fingerprint changes — ``output_format``,
        # ``effort`` / ``thinking``, ``allowed_tools``, ``mcp_servers``,
        # and ``max_turns`` are all baked into ClaudeAgentOptions at
        # connect time. (model, system, resume) are fixed for the
        # session and don't contribute to the key.
        schema_fragment = (
            f"{schema.__name__}|{schema.model_json_schema()}"
            if schema is not None
            else "__plain_text__"
        )
        effort, thinking_config = _translate_thinking_for_claude(thinking)
        thinking_fragment = f"effort={effort}|thinking={thinking_config}"
        attachments_fragment = f"attachments={has_attachments}"
        tools_fragment = f"tools={self._tools_fingerprint}"
        mcp_fragment = f"mcp={self._mcp_servers_fingerprint}"
        permission_fragment = f"perm={self._permission_fingerprint}"
        max_turns_fragment = f"max_turns={max_turns}"
        provider_options_fragment = f"po={_claude_options_fingerprint(self._provider_options)}"
        cache_key = (
            f"{schema_fragment}|{thinking_fragment}|{attachments_fragment}|"
            f"{tools_fragment}|{mcp_fragment}|{permission_fragment}|"
            f"{max_turns_fragment}|{provider_options_fragment}"
        )
        if self._client is not None and self._client_key == cache_key:
            return self._client

        # Tear down any stale client before rebuilding so we don't leak
        # the previous subprocess.
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "%s.session_reconnect_disconnect_failed error=%s",
                    self._runtime.label,
                    exc,
                )
            self._client = None
            self._client_key = None

        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

        env_override: dict[str, str] = {}
        if self._runtime._api_key_override is not None:
            env_override["ANTHROPIC_API_KEY"] = self._runtime._api_key_override

        options_kwargs: dict[str, Any] = {
            "model": self._model_id,
            # Phase 5 Iteration D: ``max_turns=`` on execute()/stream()
            # overrides the runtime-default DEFAULT_MAX_TURNS by riding
            # into ClaudeAgentOptions.max_turns at connect time. Cache
            # key carries the value so a turn-cap change forces
            # reconnect (same pattern as schema= / thinking=).
            "max_turns": max_turns if max_turns is not None else self._runtime._max_turns,
            "permission_mode": "bypassPermissions",
            "env": env_override or {},
            # Always-on so stream() gets fine-grained deltas. execute()
            # ignores StreamEvents — receive_response drains them
            # alongside the ResultMessage and we only act on the latter.
            "include_partial_messages": True,
        }
        if schema is not None:
            options_kwargs["output_format"] = {
                "type": "json_schema",
                "schema": schema.model_json_schema(),
            }
        if self._system is not None:
            options_kwargs["system_prompt"] = self._system
        if self._resume is not None:
            options_kwargs["resume"] = self._resume
        if effort is not None:
            options_kwargs["effort"] = effort
        if thinking_config is not None:
            options_kwargs["thinking"] = thinking_config
        if self._on_permission is not None:
            options_kwargs["can_use_tool"] = _translate_permission_for_claude(self._on_permission)
        if self._on_event is not None:
            options_kwargs["hooks"] = _build_claude_hooks_config(self._on_event, session=self)
        po = self._provider_options
        if po is not None:
            if po.append_system_prompt is not None:
                options_kwargs["append_system_prompt"] = po.append_system_prompt
            if po.fork_session:
                options_kwargs["fork_session"] = True
            if po.strict_mcp_config:
                options_kwargs["strict_mcp_config"] = True

        # Bring in the auto-allowed Read tool for attachments, the
        # in-process MCP server for FunctionTools, and the per-name
        # external MCP servers. The CLI uses ``mcp__<server>__<tool>``
        # naming for MCP tools, so the allowed_tools entries must
        # match that pattern; we let _translate_tools_for_claude build
        # both the in-process server config and the matching
        # per-tool allowed-tools names so the two stay in sync. For
        # external MCP servers we don't know tool names ahead of time,
        # so we add the ``mcp__<server>__*`` wildcard form so every
        # tool the server exposes is auto-allowed.
        allowed_tools: list[str] = []
        if has_attachments:
            allowed_tools.append("Read")
        mcp_servers_config: dict[str, Any] = {}
        if self._tools:
            server_config, allowed_tool_names = _translate_tools_for_claude(self._tools)
            mcp_servers_config[AIRFRAME_MCP_SERVER_NAME] = server_config
            allowed_tools.extend(allowed_tool_names)
        if self._mcp_servers:
            external = _translate_mcp_servers_for_claude(self._mcp_servers)
            # Collision check: surface name overlaps loudly. The
            # in-process server's name is the only "reserved" value
            # ``mcp_servers=`` callers must avoid.
            for name in external:
                if name in mcp_servers_config:
                    raise ValueError(
                        f"{self._runtime.label}: McpServerRef.name {name!r} collides "
                        f"with airframe's in-process tools server name. Rename the "
                        f"external server (the in-process name "
                        f"{AIRFRAME_MCP_SERVER_NAME!r} is reserved)."
                    )
            mcp_servers_config.update(external)
            for ref in self._mcp_servers:
                allowed_tools.append(f"mcp__{ref.name}__*")
        if mcp_servers_config:
            options_kwargs["mcp_servers"] = mcp_servers_config
        if allowed_tools:
            options_kwargs["allowed_tools"] = allowed_tools

        options = ClaudeAgentOptions(**options_kwargs)
        try:
            client = ClaudeSDKClient(options=options)
            await client.connect()
        except Exception as exc:
            raise self._runtime._classify_exception(exc) from exc
        self._client = client
        self._client_key = cache_key
        # Phase 5 Iteration C: synthesise session_start on first
        # connect. Claude has no native event for this (its
        # ``SessionStart`` hook fires inside the CLI process before
        # the SDK client wires up), so airframe emits it from the
        # adapter layer. Subsequent reconnects (cache-key changes
        # mid-session) don't re-fire.
        if not self._session_start_fired:
            _fire_hook_event(
                self._on_event,
                "session_start",
                session_id=self.id,
                payload={"model": self._model_id, "resumed": self._resume is not None},
            )
            self._session_start_fired = True
        return client

    async def _query_and_drain(self, client: Any, prompt: str) -> Any:
        from claude_agent_sdk import ResultMessage

        await client.query(prompt)
        final: Any = None
        async for msg in client.receive_response():
            if isinstance(msg, ResultMessage):
                final = msg
        return final

    def _build_result(self, result_msg: Any, *, schema: type[BaseModel] | None) -> RuntimeResult:
        if result_msg is None:
            raise RuntimeProtocolError(
                f"{self._runtime.label}: stream closed without a ResultMessage"
            )
        # Surface the live session ID for resume.
        sess_id = getattr(result_msg, "session_id", None)
        if sess_id:
            self.id = sess_id

        if getattr(result_msg, "is_error", False):
            err_text = (result_msg.errors or [])[:1] or [result_msg.subtype or "unknown"]
            raise AgentRuntimeError(
                f"{self._runtime.label}: is_error subtype={result_msg.subtype} {err_text}"
            )

        text = result_msg.result or ""
        cost = self._runtime._cost_from_result(result_msg, model_id=self._model_id)

        if schema is None:
            return RuntimeResult(
                text=text,
                structured=None,
                cost=cost,
                finish=result_msg.stop_reason,
                raw=result_msg,
            )

        structured = getattr(result_msg, "structured_output", None)
        if structured is None:
            raise RuntimeStructuredOutputError(
                f"{self._runtime.label}: structured_output was empty "
                f"(stop_reason={result_msg.stop_reason}, "
                f"subtype={getattr(result_msg, 'subtype', None)})",
                body={
                    "stop_reason": result_msg.stop_reason,
                    "subtype": getattr(result_msg, "subtype", None),
                    "result": result_msg.result,
                },
            )
        return RuntimeResult(
            text=text,
            structured=structured,
            cost=cost,
            finish=result_msg.stop_reason,
            raw=result_msg,
        )


def _events_from_stream_event(stream_event: Any) -> list[RuntimeEvent]:
    """Translate one :class:`StreamEvent` into airframe :class:`RuntimeEvent` instances.

    Anthropic's wire format emits a stream of ``content_block_delta`` events
    each carrying a typed ``delta``:

    * ``{"type": "text_delta", "text": "..."}`` → :class:`TextDelta`
    * ``{"type": "thinking_delta", "thinking": "..."}`` → :class:`ReasoningDelta`
    * ``{"type": "input_json_delta", "partial_json": "..."}`` → tool args
      (no event surfaced today — Phase 3 wires :class:`ToolCallStart`
      from the matching ``content_block_start``).

    Other event kinds (``message_start``, ``message_stop``, etc.) are
    ignored at this layer; the SDK already tracks the lifecycle and
    delivers a :class:`ResultMessage` at the end of the turn.
    """
    raw = getattr(stream_event, "event", None) or {}
    if raw.get("type") != "content_block_delta":
        return []
    delta = raw.get("delta") or {}
    kind = delta.get("type")
    if kind == "text_delta":
        text = delta.get("text") or ""
        if text:
            return [TextDelta(text=text)]
    elif kind == "thinking_delta":
        thinking = delta.get("thinking") or ""
        if thinking:
            return [ReasoningDelta(text=thinking)]
    return []


def _build_claude_prompt(text: str, images: list[Any], files: list[Any]) -> str:
    """Append a Read-tool hint block for image / file attachments.

    Claude Agent SDK has no direct image-in-prompt API today; the
    portable workaround is to surface attachment paths in the prompt
    text and let the model open them with the Read tool (caller adds
    ``allowed_tools=["Read"]``). The hint is a plain list — short
    enough that it doesn't crowd long user prompts but unambiguous
    enough that the model picks it up reliably.

    **Path-only.** The Read tool reads from the filesystem;
    :class:`ImageInput(bytes_=)` and :class:`ImageInput(url=)` raise
    :class:`UnsupportedFeatureError`. Consumer should write bytes to
    disk (e.g. ``tempfile.NamedTemporaryFile``) and pass ``path=``
    instead.

    Returns the original ``text`` unchanged when no attachments are
    present, so the no-vision path stays untouched.
    """
    for img in images:
        if img.path is None:
            kind = "bytes_" if img.bytes_ is not None else "url"
            raise UnsupportedFeatureError(
                f"claude_code: ImageInput({kind}=...) has no Claude Agent SDK "
                f"channel — the Read tool fallback only opens filesystem paths. "
                f"Write the bytes to disk (tempfile.NamedTemporaryFile) and "
                f"pass path= instead.",
                feature=Feature.VISION_INPUT,
            )
    if not images and not files:
        return text
    lines = ["Attached files (use the Read tool to access):"]
    for img in images:
        lines.append(f"- {img.path}")
    for file in files:
        suffix = f" ({file.media_type})" if file.media_type else ""
        lines.append(f"- {file.path}{suffix}")
    block = "\n".join(lines)
    return f"{text}\n\n{block}" if text else block


def _translate_thinking_for_claude(
    thinking: ThinkingMode,
) -> tuple[str | None, dict[str, Any] | None]:
    """Translate :data:`ThinkingMode` to ``(effort, thinking_config)``.

    Claude exposes two related knobs on :class:`ClaudeAgentOptions`:

    * ``effort: Literal["low" | "medium" | "high" | "xhigh" | "max"]``
      — coarse, portable.
    * ``thinking: ThinkingConfig`` — fine-grained
      ``{"type": "adaptive" | "enabled" | "disabled",
      "budget_tokens"?: int}``.

    This helper returns ``(effort, thinking_config)`` — either or both
    may be ``None``. ``None, None`` means "send neither" (vendor
    default).

    Mappings:

    * ``None`` → no override (vendor default).
    * ``"disabled"`` → ``thinking = {"type": "disabled"}``.
    * ``"low" | "medium" | "high"`` → ``effort = <same>``.
    * ``"minimal"`` → ``effort = "low"`` with a debug log (Anthropic
      has no minimal tier).
    * ``{"budget_tokens": N}`` → ``thinking = {"type": "enabled",
      "budget_tokens": N}``.

    Raises:
        UnsupportedFeatureError: when the dict shape is unrecognised
            (no ``budget_tokens`` key) or the literal isn't one of the
            documented values.
    """
    if thinking is None:
        return None, None
    if thinking == "disabled":
        return None, {"type": "disabled"}
    if isinstance(thinking, str):
        if thinking == "minimal":
            logger.debug(
                "claude_code: thinking='minimal' has no Anthropic equivalent; coercing to 'low'"
            )
            return "low", None
        if thinking in ("low", "medium", "high"):
            return thinking, None
        raise UnsupportedFeatureError(
            f"claude_code: unrecognised thinking effort {thinking!r}; "
            f"supported: 'minimal' (→'low'), 'low', 'medium', 'high', 'disabled'.",
            feature="reasoning_effort",
        )
    if isinstance(thinking, dict):
        budget = thinking.get("budget_tokens")
        if budget is None or not isinstance(budget, int):
            raise UnsupportedFeatureError(
                f"claude_code: dict-shaped thinking must include integer "
                f"'budget_tokens'; got keys={list(thinking)}",
                feature="reasoning_budget_tokens",
            )
        return None, {"type": "enabled", "budget_tokens": int(budget)}
    raise UnsupportedFeatureError(
        f"claude_code: unrecognised thinking mode {thinking!r}",
        feature="reasoning_effort",
    )


#: Name of the in-process MCP server airframe registers when
#: ``tools=`` is set on :meth:`ClaudeCodeRuntime.session`. The CLI
#: addresses tools served by this server as ``mcp__{name}__{tool}``
#: in ``allowed_tools``; keeping the server name stable per session
#: lets the cache key fingerprint stay tied to the user-facing tool
#: list rather than a random server-instance ID.
AIRFRAME_MCP_SERVER_NAME = "airframe_tools"


def _tools_fingerprint(tools: list[FunctionTool]) -> str:
    """Build a deterministic fingerprint for a ``tools=`` list.

    Used in the :meth:`_ensure_client` cache key so a tools-change
    between turns rebuilds the underlying CLI subprocess (since
    ``mcp_servers`` is baked at connect time). Includes each tool's
    ``name``, ``description``, and its Pydantic schema so a change to
    the input shape — even a documentation tweak — invalidates the
    cached client.
    """
    if not tools:
        return "__no_tools__"
    parts: list[str] = []
    for t in tools:
        parts.append(f"{t.name}|{t.description}|{t.params.model_json_schema()}")
    return "||".join(parts)


def _translate_tools_for_claude(
    tools: list[FunctionTool],
) -> tuple[Any, list[str]]:
    """Build an in-process MCP server + the matching allowed-tools names.

    Each :class:`FunctionTool` becomes one SDK ``@tool``-decorated
    coroutine that:

    1. Validates the incoming ``args`` dict against the tool's
       :attr:`FunctionTool.params` Pydantic schema (the SDK passes raw
       dicts even when ``input_schema`` is a :class:`BaseModel`).
    2. Awaits the user-supplied handler with the typed instance.
    3. Wraps the return value in the SDK's
       ``{"content": [{"type": "text", "text": <output>}]}`` envelope.
       Handler exceptions are caught and surfaced as ``is_error=True``
       so the model can recover on its next turn.

    Returns ``(server_config, allowed_tool_names)``. The allowed-tools
    names use the ``mcp__<server>__<tool>`` shape the CLI expects when
    permission-gating MCP-routed tools.
    """
    from claude_agent_sdk import create_sdk_mcp_server, tool

    sdk_tools = []
    allowed_tool_names: list[str] = []
    for ft in tools:
        # Capture each tool in the closure explicitly — ``ft`` is the
        # iterator variable, so the inner function must bind the
        # specific FunctionTool, not the loop's last value.
        sdk_tools.append(_build_sdk_tool(ft, tool_decorator=tool))
        allowed_tool_names.append(f"mcp__{AIRFRAME_MCP_SERVER_NAME}__{ft.name}")
    server_config = create_sdk_mcp_server(
        name=AIRFRAME_MCP_SERVER_NAME,
        tools=sdk_tools,
    )
    return server_config, allowed_tool_names


def _build_sdk_tool(ft: FunctionTool, *, tool_decorator: Any) -> Any:
    """Wrap one :class:`FunctionTool` as a Claude SDK ``@tool`` coroutine.

    Split out so the closure binds ``ft`` (and the user's handler)
    explicitly, free of the surrounding loop variable.
    """

    @tool_decorator(ft.name, ft.description, ft.params)
    async def _wrapper(args: dict[str, Any]) -> dict[str, Any]:
        try:
            params = ft.params.model_validate(args)
        except Exception as exc:  # noqa: BLE001 — surface to model
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Tool arguments did not match the {ft.params.__name__} schema: {exc}"
                        ),
                    }
                ],
                "isError": True,
            }
        try:
            output = await ft.handler(params)
        except Exception as exc:  # noqa: BLE001 — surface to model
            return {
                "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                "isError": True,
            }
        return {"content": [{"type": "text", "text": _stringify_tool_output(output)}]}

    return _wrapper


def _stringify_tool_output(output: Any) -> str:
    """JSON-encode a tool handler's return value for MCP transport.

    Strings pass through verbatim (so already-formatted markdown stays
    legible in the model's view); everything else round-trips through
    :func:`json.dumps` with ``default=str``; unserialisable types fall
    back to :func:`repr` so the model still sees *something*.
    """
    import json

    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, default=str)
    except (TypeError, ValueError):
        return repr(output)


def _serialize_tool_arguments(input_dict: dict[str, Any]) -> str:
    """Serialise a :attr:`ToolUseBlock.input` dict for the
    ``arguments_preview`` event field."""
    import json

    try:
        return json.dumps(input_dict, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return repr(input_dict)


def _strip_mcp_prefix(tool_name: str, server_names: frozenset[str]) -> str:
    """Trim the ``mcp__<server>__`` prefix the SDK adds to MCP tool names.

    Phase 4 Iteration B generalised this from the single-server form
    (Phase 3 only knew about ``mcp__airframe_tools__``) to the
    multi-server set the session tracks: the in-process
    :data:`AIRFRAME_MCP_SERVER_NAME` whenever ``tools=`` is set, plus
    every :attr:`McpServerRef.name` for refs in ``mcp_servers=``.
    Tools delivered through a recognised server come back with their
    bare name; unrecognised prefixes pass through verbatim so
    consumers can still inspect raw vendor tool names if they
    register a server via direct
    :func:`~ClaudeCodeRuntime.unwrap` access.
    """
    for name in server_names:
        prefix = f"mcp__{name}__"
        if tool_name.startswith(prefix):
            return tool_name[len(prefix) :]
    return tool_name


def _translate_mcp_servers_for_claude(
    refs: list[McpServerRef],
) -> dict[str, dict[str, Any]]:
    """Translate :class:`McpServerRef` list into Claude's
    :attr:`ClaudeAgentOptions.mcp_servers` dict shape.

    The SDK accepts a dict keyed by server name where each value is a
    :class:`McpStdioServerConfig` / :class:`McpHttpServerConfig` /
    :class:`McpSSEServerConfig` TypedDict. Mapping:

    * **stdio** — splits :attr:`McpServerRef.command` (an argv list)
      into the SDK's ``command: str`` (head) + ``args: list[str]``
      (tail). ``args`` is omitted when the argv has only one element.
    * **http** / **sse** — :attr:`McpServerRef.url` passes through.
      :attr:`McpServerRef.auth_token`, if set, becomes
      ``Authorization: Bearer <token>``; caller-supplied
      :attr:`McpServerRef.headers` are merged on top so an explicit
      ``Authorization`` in ``headers=`` overrides the shorthand.

    Returns a fresh dict each call — never mutates the caller's refs.
    """
    out: dict[str, dict[str, Any]] = {}
    for ref in refs:
        if ref.transport == "stdio":
            # Guaranteed by McpServerRef.__post_init__: command is a
            # non-empty list when transport is stdio.
            assert ref.command
            cfg: dict[str, Any] = {"type": "stdio", "command": ref.command[0]}
            if len(ref.command) > 1:
                cfg["args"] = list(ref.command[1:])
            out[ref.name] = cfg
        else:  # http or sse — also guaranteed by __post_init__
            assert ref.url
            cfg = {"type": ref.transport, "url": ref.url}
            merged = _compose_mcp_headers(ref)
            if merged:
                cfg["headers"] = merged
            out[ref.name] = cfg
    return out


def _translate_permission_for_claude(callback: PermissionCallback) -> Any:
    """Wrap an airframe :class:`PermissionCallback` as Claude's
    :attr:`ClaudeAgentOptions.can_use_tool`.

    The SDK calls the returned coroutine with
    ``(tool_name, tool_args, ToolPermissionContext)`` and expects a
    :class:`PermissionResultAllow` / :class:`PermissionResultDeny`
    back. We build a :class:`PermissionRequest`, await the user's
    callback, and translate the :data:`PermissionDecision`:

    * ``"allow"`` → :class:`PermissionResultAllow`.
    * ``"deny"`` → :class:`PermissionResultDeny` with the
      :attr:`PermissionRequest.reason` (or a generic message) so the
      model sees actionable feedback.
    * ``"defer"`` → :class:`PermissionResultAllow` with a debug log.
      Claude's binary result type has no third option; the existing
      ``permission_mode="bypassPermissions"`` default already
      allows everything, so ``"defer"`` matches that semantics —
      "I don't have an opinion; let the SDK / default policy
      decide" collapses to "allow" on this adapter.

    The :attr:`PermissionRequest.reason` is built from the SDK
    context's :attr:`ToolPermissionContext.decision_reason` /
    :attr:`description` / :attr:`title` (first non-empty); the
    :attr:`tool_args` come from the SDK's raw ``tool_args`` dict.
    """
    from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

    from airframe.permission import PermissionRequest

    async def _can_use_tool(
        tool_name: str,
        tool_args: dict[str, Any],
        context: Any,
    ) -> Any:
        reason = (
            getattr(context, "decision_reason", None)
            or getattr(context, "description", None)
            or getattr(context, "title", None)
        )
        request = PermissionRequest(
            tool_name=tool_name,
            tool_args=tool_args,
            reason=reason,
        )
        decision = await callback.handle(request)
        if decision == "allow":
            return PermissionResultAllow()
        if decision == "deny":
            return PermissionResultDeny(
                message=reason or f"airframe PermissionCallback denied {tool_name!r}",
            )
        # "defer" — Claude's binary result type has no third option;
        # treat as allow to match the existing bypassPermissions
        # default. Debug-log so consumers can audit the deferral.
        logger.debug(
            "claude_code: PermissionCallback returned 'defer' for tool=%r; "
            "coercing to 'allow' (Claude's can_use_tool result is binary; "
            "the existing permission_mode='bypassPermissions' default "
            "matches the 'defer' intent)",
            tool_name,
        )
        return PermissionResultAllow()

    return _can_use_tool


def _permission_fingerprint(callback: PermissionCallback | None) -> str:
    """Deterministic fingerprint of the permission-callback identity.

    Used in the :meth:`ClaudeCodeSession._ensure_client` cache key so
    a callback swap between turns forces reconnect (Claude's
    ``can_use_tool`` is baked at connect time). Identity-based —
    we don't try to introspect the callable's behaviour, just
    detect "same object" vs "different object" so consumers who pass
    a fresh lambda per call see a clear reconnect rather than silent
    staleness.
    """
    if callback is None:
        return "__no_permission__"
    return f"id={id(callback)}|type={type(callback).__name__}"


def _claude_options_fingerprint(po: ClaudeOptions | None) -> str:
    """Deterministic fingerprint of the :class:`ClaudeOptions` value.

    All three populated fields (``append_system_prompt``,
    ``fork_session``, ``strict_mcp_config``) bake into
    :class:`ClaudeAgentOptions` at connect time, so a change
    between turns must force reconnect (same pattern as ``schema=``,
    ``tools=``, ``on_permission=``). Value-based — dataclasses are
    frozen, so a structurally-equal instance fingerprints
    identically.
    """
    if po is None:
        return "__no_provider_options__"
    return (
        f"asp={po.append_system_prompt!r}|fork={po.fork_session}|strict_mcp={po.strict_mcp_config}"
    )


#: Map Claude's native ``hook_event_name`` values onto airframe's
#: :data:`HookEventKind` literals. SDK events not in this map
#: (``SubagentStop``, ``Notification``, ``SubagentStart``,
#: ``PermissionRequest``) don't have a direct airframe equivalent —
#: ``PermissionRequest`` is its own callback channel, the others fall
#: outside the 8 lifecycle moments airframe normalises today.
_CLAUDE_HOOK_NAME_TO_KIND: dict[str, str] = {
    "PreToolUse": "pre_tool_use",
    "PostToolUse": "post_tool_use",
    "PostToolUseFailure": "tool_failure",
    "UserPromptSubmit": "user_prompt_submit",
    "PreCompact": "pre_compact",
    "Stop": "session_end",
}


def _build_claude_hooks_config(
    on_event: Callable[[HookEvent], None],
    *,
    session: ClaudeCodeSession,
) -> dict[str, list[Any]]:
    """Build :attr:`ClaudeAgentOptions.hooks` that fans every native
    Claude hook into the user's ``on_event=`` callback.

    The SDK dispatches each hook with
    ``(HookInput, tool_use_id, HookContext)`` and expects a JSON
    output back. airframe's observer contract is pure observation —
    we never block or modify the SDK flow — so the returned
    :class:`SyncHookJSONOutput` is always empty (``{}``). The session
    keeps the latest known ``session_id`` so subsequent
    synthesised events (``session_end`` at close) carry the right
    value.

    Per-kind payload schema (airframe normalises across vendors):

    * ``pre_tool_use`` / ``post_tool_use`` / ``tool_failure`` —
      ``{"tool_name", "tool_use_id", "tool_input"}``; success path
      additionally carries ``"tool_response"``; failure path carries
      ``"error"`` when the SDK exposes one.
    * ``user_prompt_submit`` — ``{"prompt", "length"}`` (prompt is
      the raw user-supplied string).
    * ``pre_compact`` — ``{"trigger", "custom_instructions"}``.
    * ``session_end`` — ``{"stop_hook_active"}``; vendor-specific
      shutdown semantics live in :attr:`payload` for consumers that
      need to branch.
    """
    from claude_agent_sdk import HookMatcher

    async def _make_handler(kind: str, hook_input: Any) -> dict[str, Any]:
        # Hook inputs are TypedDicts at runtime (plain dicts). Pull
        # the session_id off the input when present and forward to
        # the session so synthesised events stay consistent.
        if isinstance(hook_input, dict):
            sid = hook_input.get("session_id")
            if sid:
                session.id = sid
        payload = _extract_claude_hook_payload(kind, hook_input)
        _fire_hook_event(
            on_event,
            kind,
            session_id=session.id,
            payload=payload,
        )
        # Pure observation — no continue=/decision=/etc. signals.
        return {}

    def _make_callback(kind: str) -> Any:
        async def _cb(hook_input: Any, _tool_use_id: str | None, _ctx: Any) -> dict[str, Any]:
            return await _make_handler(kind, hook_input)

        return _cb

    hooks_config: dict[str, list[Any]] = {}
    for sdk_event_name, kind in _CLAUDE_HOOK_NAME_TO_KIND.items():
        hooks_config[sdk_event_name] = [HookMatcher(matcher=None, hooks=[_make_callback(kind)])]
    return hooks_config


def _extract_claude_hook_payload(kind: str, hook_input: Any) -> dict[str, Any]:
    """Pull airframe's normalised payload fields out of a Claude
    hook-input TypedDict.

    Each kind exposes a different field set; this helper picks the
    portable subset so consumer code can branch on
    ``event.kind == "pre_tool_use"`` without writing per-vendor
    glue. Unknown fields stay accessible via the raw SDK type behind
    ``runtime.unwrap()`` if a consumer needs them.
    """
    if not isinstance(hook_input, dict):
        return {}
    if kind in ("pre_tool_use", "post_tool_use", "tool_failure"):
        payload: dict[str, Any] = {
            "tool_name": hook_input.get("tool_name", ""),
            "tool_use_id": hook_input.get("tool_use_id", ""),
            "tool_input": hook_input.get("tool_input", {}),
        }
        if "tool_response" in hook_input:
            payload["tool_response"] = hook_input["tool_response"]
        if "error" in hook_input:
            payload["error"] = hook_input["error"]
        return payload
    if kind == "user_prompt_submit":
        prompt = hook_input.get("prompt", "")
        return {"prompt": prompt, "length": len(prompt) if isinstance(prompt, str) else 0}
    if kind == "pre_compact":
        return {
            "trigger": hook_input.get("trigger"),
            "custom_instructions": hook_input.get("custom_instructions"),
        }
    if kind == "session_end":
        return {"stop_hook_active": hook_input.get("stop_hook_active", False)}
    return {}


def _tool_result_output(block: Any) -> Any:
    """Extract a user-friendly :attr:`ToolCallResult.output` from a
    :class:`ToolResultBlock`.

    The SDK delivers tool results as either a bare string or a list of
    content blocks (mirroring Anthropic's wire shape). We collapse
    the list of ``{"type":"text","text":"…"}`` parts into a single
    string when that's all the model sent; otherwise pass the raw
    structure through so consumers don't lose typed metadata.
    """
    content = block.content
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
            else:
                # Mixed structure — return the raw list so consumers
                # can inspect non-text parts (rare).
                return content
        return "".join(text_parts)
    return content


__all__ = [
    "AIRFRAME_MCP_SERVER_NAME",
    "DEFAULT_CLAUDE_MODEL",
    "DEFAULT_MAX_TURNS",
    "ClaudeCodeRuntime",
    "ClaudeCodeSession",
]
