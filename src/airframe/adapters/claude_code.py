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
3. ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN`` env var —
   pay-per-token API access. Useful for production deployments
   without a Max subscription.

Setting ``ANTHROPIC_BASE_URL`` aims the adapter at an
Anthropic-compatible third-party endpoint. In that mode options 1
and 2 are skipped: they are Anthropic subscription credentials and
airframe will not forward them off-site. Supply the endpoint's own
credential via ``ANTHROPIC_AUTH_TOKEN``. Note this scoping applies
to airframe's own direct calls (:meth:`list_models`,
:meth:`count_tokens`); the ``claude`` CLI subprocess does its own
auth resolution from the inherited environment.

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
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel

from airframe.cache import CacheConfig
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
from airframe.metadata import RequestMetadata
from airframe.models import ModelInfo
from airframe.native_tools import NativeCapability, NativeTool
from airframe.options import ClaudeOptions
from airframe.protocol import (
    AgentRuntime,
    AgentSession,
    ProviderModel,
    RuntimeResult,
    UnsupportedBindingError,
)
from airframe.rate_limit import RateLimitInfo, RateLimitWindow
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
    _native_tools_fingerprint,
    _resolve_native_tools,
    _split_prompt_parts,
)
from airframe.slash_commands import SlashCommand, SlashCommandsConfig
from airframe.thinking import ThinkingMode
from airframe.tools import FunctionTool, McpServerRef

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

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


#: Path to the interactive Claude Code OAuth credentials file. Set
#: per the documented auth chain — overridden by ``CLAUDE_CODE_OAUTH_TOKEN``
#: env var on machines where the file isn't reachable.
DEFAULT_CLAUDE_CREDENTIALS_PATH = "~/.claude/.credentials.json"


#: Hosts that count as "Anthropic's own API" when scoping credentials.
#: Subscription OAuth tokens are only ever sent to these.
_ANTHROPIC_API_HOSTS = frozenset({"api.anthropic.com"})


def _is_anthropic_endpoint(base_url: str | None) -> bool:
    """``True`` when ``base_url`` is unset or points at Anthropic's own API.

    ``ANTHROPIC_BASE_URL`` is the documented way to aim the Claude
    Agent SDK — and therefore this adapter — at an Anthropic-compatible
    third-party endpoint (a vendor gateway, a corporate proxy, a local
    mock). When it points off-site, airframe must withhold
    Anthropic-minted subscription credentials: those bearer tokens
    authenticate *the user's Anthropic account* and are meaningless to
    any other vendor except as a stolen secret.

    A ``base_url`` that fails to parse into a hostname is treated as
    third-party — the conservative direction, since the cost of a false
    negative is a withheld token and the cost of a false positive is a
    leaked one.

    Args:
        base_url: Raw ``ANTHROPIC_BASE_URL`` value, or ``None`` when unset.

    Returns:
        ``True`` when subscription OAuth tokens may safely be sent.
    """
    if not base_url or not base_url.strip():
        return True
    host = (urlparse(base_url).hostname or "").lower()
    if not host:
        return False
    return host in _ANTHROPIC_API_HOSTS or host.endswith(".anthropic.com")


def _read_claude_credentials_oauth_token() -> str | None:
    """Extract the OAuth bearer token from ``~/.claude/.credentials.json``.

    The interactive Claude Code login writes a file of the shape::

        {
          "claudeAiOauth": {
            "accessToken": "sk-ant-oat...",
            "refreshToken": "...",
            "expiresAt": 1234567890,
            "scopes": [...],
            "subscriptionType": "max"
          }
        }

    This helper reads the file, validates the expected shape, and
    returns the access token. Returns ``None`` for every "this isn't
    a usable token" case (missing file, malformed JSON, missing keys,
    empty string) so the caller can fall through cleanly to the
    "no credentials anywhere" branch. The file path is overridable
    via the ``CLAUDE_CREDENTIALS_PATH`` env var (useful for tests).

    Refresh-token handling lives in the Anthropic SDK once the access
    token is passed via ``auth_token=`` — when the token expires
    mid-call the SDK refreshes against ``/v1/oauth/token`` using the
    refresh token from its own config store, not this file.
    """
    import json

    path_str = os.environ.get("CLAUDE_CREDENTIALS_PATH") or DEFAULT_CLAUDE_CREDENTIALS_PATH
    path = os.path.expanduser(path_str)
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (FileNotFoundError, PermissionError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    oauth = payload.get("claudeAiOauth")
    if not isinstance(oauth, dict):
        return None
    token = oauth.get("accessToken")
    if not isinstance(token, str) or not token:
        return None
    return token


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
    #:   :class:`~airframe.permission.PermissionCallback` into a
    #:   native ``PreToolUse`` hook that returns
    #:   ``permissionDecision: "deny"`` when the callback denies, and
    #:   an empty response for ``"allow"`` / ``"defer"``. The
    #:   ``PreToolUse`` channel is used rather than
    #:   :attr:`ClaudeAgentOptions.can_use_tool` because the SDK only
    #:   invokes ``can_use_tool`` for calls that would otherwise
    #:   prompt — under the adapter's
    #:   ``permission_mode="bypassPermissions"`` nothing ever reaches
    #:   it, whereas ``PreToolUse`` fires for every tool call
    #:   regardless of permission mode. ``can_use_tool`` stays wired
    #:   as a no-op belt-and-braces for non-bypass configurations.
    #:   Callback raises are debug-logged and treated as ``"defer"``.
    #:   Callback identity joins the ``_ensure_client`` cache key so a
    #:   callback swap forces reconnect (Phase 5, Iteration B; hook
    #:   gating added in 0.9.2).
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
            Feature.TOOLS_NATIVE,
            Feature.PERMISSION_CALLBACK,
            Feature.LIFECYCLE_HOOKS,
            Feature.BUDGET_USD_CAP,
            Feature.BUDGET_TURN_CAP,
            Feature.RATE_LIMIT_TELEMETRY,
            Feature.REASONING_OUTPUT,
            Feature.REQUEST_METADATA,
            Feature.COUNT_TOKENS,
            Feature.SLASH_COMMANDS,
        }
    )

    #: Vendor-hosted built-in tools this adapter can enable via
    #: ``native_tools=``. Claude's CLI runs ``WebSearch`` / ``WebFetch``
    #: on Anthropic's infrastructure; airframe maps the portable
    #: capabilities onto those tool names in ``allowed_tools``.
    SUPPORTED_NATIVE_TOOLS: ClassVar[frozenset[NativeCapability]] = frozenset(
        {
            NativeCapability.WEB_SEARCH,
            NativeCapability.WEB_FETCH,
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
        metadata: RequestMetadata | None = None,
        cache: CacheConfig | None = None,
    ) -> RuntimeResult:
        # Phase 1 Iteration G: ``execute()`` is documented sugar for
        # ``runtime.session(...).execute(...) + close()``. Single-turn,
        # ephemeral — the underlying ClaudeSDKClient is spawned and
        # disconnected per call. Consumers wanting context warmth across
        # calls open a session explicitly and reuse it.
        del persona  # accepted in the protocol but not consumed by Claude
        sess = self.session(system=system, model=model, metadata=metadata, cache=cache)
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

    def supported_native_tools(
        self, model: ProviderModel | None = None
    ) -> frozenset[NativeCapability]:
        return self.SUPPORTED_NATIVE_TOOLS

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
        native_tools: list[NativeTool] | None = None,
        on_permission: PermissionCallback | None = None,
        on_event: Callable[[HookEvent], None] | None = None,
        provider_options: ProviderOptions | None = None,
        metadata: RequestMetadata | None = None,
        cache: CacheConfig | None = None,
        slash_commands: SlashCommandsConfig | None = None,
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
            on_permission: Optional gate consulted before every tool
                call. Wired through a native ``PreToolUse`` hook (see
                :func:`_build_pre_tool_use_gate`): ``"deny"`` blocks
                the call and hands the model a tool failure carrying
                the reason; ``"allow"`` / ``"defer"`` fall through to
                the session's permission posture. A callback that
                raises is debug-logged and treated as ``"defer"``.
            on_event: Optional observer called for each lifecycle
                event. Wired through :attr:`ClaudeAgentOptions.hooks`;
                observation only — the return value is discarded and
                cannot alter SDK flow (use ``on_permission=`` to gate).
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
                :class:`OpenAICompatOptions` here raises
                :class:`UnsupportedFeatureError`.
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
        resolved_native_tools = _resolve_native_tools(
            native_tools,
            adapter_label=self.label,
            provider_id=self.PROVIDER_ID,
            feature_supported=self.supports(Feature.TOOLS_NATIVE),
            supported_capabilities=self.supported_native_tools(model),
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
        # Phase 6 — PROMPT_CACHE_CONTROL soft contract: Claude Agent SDK
        # manages caching via session warmth, not an explicit key channel,
        # so the airframe cache= value is silently dropped here. The
        # decline matches the soft contract metadata= follows.
        del cache
        return ClaudeCodeSession(
            self,
            resume=resume,
            system=system,
            model_id=model_id,
            tools=tools,
            mcp_servers=mcp_servers,
            native_tools=resolved_native_tools,
            on_permission=on_permission,
            on_event=on_event,
            provider_options=claude_options,
            metadata=metadata,
            slash_commands=slash_commands,
        )

    def _resolve_anthropic_auth(self, *, caller: str) -> dict[str, Any]:
        """Pick auth kwargs for a direct :class:`AsyncAnthropic` call.

        Shared by :meth:`list_models` and :meth:`count_tokens` — the two
        places this adapter talks to an Anthropic-shaped HTTP endpoint
        itself instead of going through the ``claude`` CLI subprocess.

        Resolution order:

        1. Explicit ``api_key=`` constructor arg → ``api_key=``.
        2. ``CLAUDE_CODE_OAUTH_TOKEN`` env → ``auth_token=``.
        3. ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN`` env → no
           kwargs at all; the SDK reads either one itself.
        4. ``~/.claude/.credentials.json`` → ``auth_token=``.

        Rungs 2 and 4 are **Anthropic-only**. Both carry subscription
        OAuth tokens minted for the user's own Anthropic account, so
        they are skipped when ``ANTHROPIC_BASE_URL`` aims at a
        third-party endpoint — otherwise pointing the adapter at an
        Anthropic-compatible vendor would silently forward the user's
        subscription credential to that vendor. Rungs 1 and 3 are
        deliberately *not* scoped: an explicit constructor argument and
        the API-key-shaped env vars are exactly what compatible vendors
        reuse to carry their own credentials.

        Args:
            caller: Public method name, used in the error message.

        Returns:
            Kwargs to splat into :class:`AsyncAnthropic`. Empty when the
            SDK should resolve auth from the environment on its own.

        Raises:
            RuntimeAuthError: No credential resolved through any rung.
        """
        if self._api_key_override:
            return {"api_key": self._api_key_override}

        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        first_party = _is_anthropic_endpoint(base_url)

        if first_party and (oauth := os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")):
            return {"auth_token": oauth}
        if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            # Let the SDK pick whichever is set — it reads both from env.
            return {}
        if first_party:
            # Last-resort: the interactive Claude Code credentials file.
            # The SDK doesn't know about this path, so airframe extracts
            # the token and passes it as auth_token=.
            oauth_from_file = _read_claude_credentials_oauth_token()
            if oauth_from_file:
                return {"auth_token": oauth_from_file}
            raise RuntimeAuthError(
                f"ClaudeCodeRuntime.{caller}(): no credentials found. "
                "Set CLAUDE_CODE_OAUTH_TOKEN (Claude Max subscription), "
                "ANTHROPIC_API_KEY (pay-per-token API), or run "
                "`claude setup-token` to populate ~/.claude/.credentials.json."
            )
        raise RuntimeAuthError(
            f"ClaudeCodeRuntime.{caller}(): ANTHROPIC_BASE_URL is set to "
            f"{base_url!r}, which is not Anthropic's API, and no credential "
            "for that endpoint was found. Anthropic subscription tokens "
            "(CLAUDE_CODE_OAUTH_TOKEN, ~/.claude/.credentials.json) are "
            "deliberately withheld from third-party endpoints — they "
            "authenticate your Anthropic account and must not be sent "
            "elsewhere. Set ANTHROPIC_AUTH_TOKEN (or ANTHROPIC_API_KEY) to "
            "this endpoint's own credential."
        )

    async def count_tokens(
        self,
        prompt: Prompt,
        *,
        system: str | None = None,
        model: ProviderModel | None = None,
    ) -> int:
        """Tokeniser-accurate count via ``anthropic.messages.count_tokens``.

        Delegates to the official ``anthropic`` Python SDK rather than
        rolling an HTTP call ourselves — same pattern :meth:`list_models`
        uses, and the same shared :meth:`_resolve_anthropic_auth` ladder.

        v1 supports plain-text and string-only multi-part prompts.
        Image / file attachments would require base64-encoding into
        the messages payload; that round-trip is non-trivial and
        deferred until a consumer asks. Lists carrying image/file
        parts raise :class:`UnsupportedFeatureError` pointing at the
        gap.
        """
        from anthropic import (
            APIConnectionError as _AnthropicConnError,
        )
        from anthropic import (
            APIStatusError as _AnthropicAPIError,
        )
        from anthropic import AsyncAnthropic

        text, images, files = _split_prompt_parts(
            prompt,
            adapter_label=self.label,
            supports_vision=True,
            supports_file=True,
        )
        if images or files:
            raise UnsupportedFeatureError(
                f"{self.label}: count_tokens() does not yet support image / file "
                f"attachments — only plain-text prompts. The tokeniser endpoint "
                f"counts attachments accurately on a real turn; airframe needs "
                f"to translate ImageInput / FileInput into the base64 messages "
                f"shape before forwarding.",
                feature=Feature.COUNT_TOKENS,
            )

        model_id = self._resolve_model(model) if model is not None else self._default_model

        kwargs = self._resolve_anthropic_auth(caller="count_tokens")

        messages = [{"role": "user", "content": text}]
        count_kwargs: dict[str, Any] = {"model": model_id, "messages": messages}
        if system is not None:
            count_kwargs["system"] = system

        try:
            async with AsyncAnthropic(**kwargs) as client:
                result = await client.messages.count_tokens(**count_kwargs)
        except _AnthropicAPIError as exc:
            status = exc.status_code
            if status in (401, 403):
                raise RuntimeAuthError(f"claude_code: count_tokens auth: {exc}") from exc
            if status in (429, 502, 503, 504):
                raise RuntimeTransientError(
                    f"claude_code: count_tokens transient {status}"
                ) from exc
            raise RuntimeProtocolError(
                f"claude_code: count_tokens returned {status}",
                body=str(exc)[:500],
            ) from exc
        except _AnthropicConnError as exc:
            raise RuntimeTransientError(f"claude_code: count_tokens network: {exc}") from exc

        return int(result.input_tokens)

    async def list_models(self) -> list[ModelInfo]:
        """Return live Claude models from Anthropic's ``/v1/models``.

        Delegates to the official ``anthropic`` Python SDK's
        :meth:`AsyncAnthropic.models.list` rather than rolling an
        :mod:`httpx` GET ourselves. The SDK handles two auth modes
        airframe would otherwise have to track separately:

        * **API key** (``x-api-key`` header) — historical default.
          Pay-per-token API access.
        * **OAuth Bearer** (``Authorization: Bearer …`` +
          ``anthropic-beta: oauth-2025-04-20``) — Claude Max
          subscription. The Bearer path is gated by the beta header;
          omitting it makes ``/v1/models`` reject the token. Earlier
          versions of this adapter hand-rolled the request and missed
          the header, so subscription users saw 401s.

        Auth resolution lives in :meth:`_resolve_anthropic_auth`, which
        also scopes subscription OAuth tokens to Anthropic's own API —
        see that method for the full ladder.

        Raises:
            RuntimeAuthError: no credential resolved through any layer.
            RuntimeTransientError: SDK reported a 429 / 5xx / connection
                error.
            RuntimeProtocolError: SDK reported an unexpected status.
        """
        # Late import to avoid pulling `anthropic` during module load.
        # Surfaced as a clear ImportError when the [claude] extra
        # isn't installed (mirrors the SDK-not-installed pattern the
        # rest of the adapter uses).
        from anthropic import (
            APIConnectionError as _AnthropicConnError,
        )
        from anthropic import (
            APIStatusError as _AnthropicAPIError,
        )
        from anthropic import AsyncAnthropic

        kwargs = self._resolve_anthropic_auth(caller="list_models")

        try:
            async with AsyncAnthropic(**kwargs) as client:
                page = await client.models.list()
                entries = list(page.data)
        except _AnthropicAPIError as exc:
            status = exc.status_code
            if status in (401, 403):
                raise RuntimeAuthError(f"claude_code: auth: {exc}") from exc
            if status in (429, 502, 503, 504):
                raise RuntimeTransientError(f"claude_code: transient {status}") from exc
            raise RuntimeProtocolError(
                f"claude_code: /v1/models returned {status}",
                body=str(exc)[:500],
            ) from exc
        except _AnthropicConnError as exc:
            raise RuntimeTransientError(f"claude_code: network: {exc}") from exc

        from airframe.models import (
            CAPABILITY_STREAMING,
            CAPABILITY_STRUCTURED_OUTPUT,
            CAPABILITY_TOOLS,
            CAPABILITY_VISION,
        )

        out: list[ModelInfo] = []
        for entry in entries:
            model_id = entry.id
            display = getattr(entry, "display_name", model_id) or model_id
            meta = _METADATA.get(model_id, _ModelMeta(display))
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
        native_tools: list[NativeTool] | None = None,
        on_permission: PermissionCallback | None = None,
        on_event: Callable[[HookEvent], None] | None = None,
        provider_options: ClaudeOptions | None = None,
        metadata: RequestMetadata | None = None,
        slash_commands: SlashCommandsConfig | None = None,
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
        # Native (vendor-hosted) tools — WebSearch / WebFetch etc. The
        # session()-level _resolve_native_tools already filtered to the
        # subset this adapter serves and validated capabilities, so here
        # we only translate names + fingerprint for the cache key. Like
        # tools/mcp_servers, the set is baked into allowed_tools at connect
        # time, so a change forces a reconnect via the fingerprint.
        self._native_tools: list[NativeTool] = list(native_tools or [])
        self._native_tools_fingerprint = _native_tools_fingerprint(self._native_tools)
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
        # Phase 6 — RATE_LIMIT_TELEMETRY. The Claude SDK emits
        # :class:`RateLimitEvent` instances on the message stream as
        # the server's quota state changes. Each event carries one
        # window's state (``rate_limit_type``); we accumulate the
        # most-recent state per window in a dict keyed by window name
        # and snapshot to a :class:`RateLimitInfo` at result-build
        # time so :attr:`RuntimeResult.rate_limit` reflects the latest
        # quota across all windows the session has heard from.
        self._rate_limit_windows: dict[str, RateLimitWindow] = {}
        # Phase 6 — REASONING_OUTPUT. Claude emits the model's
        # extended-thinking trace as :class:`ThinkingBlock` content
        # blocks on :class:`AssistantMessage` (non-streaming) and as
        # ``thinking_delta`` events on the streaming wire (already
        # surfaced as :class:`ReasoningDelta`). The buffer accumulates
        # both shapes per turn; ``_build_result`` snapshots it onto
        # :attr:`RuntimeResult.reasoning` and resets at the start of
        # the next turn via :meth:`_reset_reasoning_buffer`.
        self._reasoning_buffer: list[str] = []
        # Phase 6 — REQUEST_METADATA. Only the ``user_id`` field maps
        # to a real Claude Agent SDK channel (``ClaudeAgentOptions.user``);
        # ``request_id`` / ``tags`` are silently dropped (no agent-SDK
        # channel). The ``user`` value joins the cache-key fragment so
        # changing it forces a reconnect — Claude bakes ``user`` at
        # connect time.
        self._metadata: RequestMetadata | None = metadata
        self._metadata_fingerprint = metadata.user_id if metadata else None
        # Phase 6 — SLASH_COMMANDS. Stashed for ``list_slash_commands``;
        # discovery is filesystem-only and adapter-agnostic. The Claude
        # Agent SDK additionally auto-expands ``/cmd`` invocations
        # natively when the consumer passes ``/cmd args`` through
        # ``execute()``, so airframe's role is just to surface the
        # palette metadata for UI use.
        self._slash_commands: SlashCommandsConfig | None = slash_commands

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
        self._reset_reasoning_buffer()
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
            RateLimitEvent,
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
                if isinstance(msg, RateLimitEvent):
                    self._absorb_rate_limit_event(msg)
                    continue
                if isinstance(msg, StreamEvent):
                    for event in _events_from_stream_event(msg):
                        if isinstance(event, ReasoningDelta):
                            self._reasoning_buffer.append(event.text)
                        yield event
                    continue
                if isinstance(msg, AssistantMessage):
                    # Fallback when include_partial_messages did not deliver
                    # text via StreamEvent (older CLI versions, or content
                    # blocks that arrived intact). Emit a TextDelta per
                    # TextBlock so consumers always see the assistant text
                    # at least once before TurnComplete. Reasoning gets
                    # the same fallback treatment — absorb ThinkingBlock
                    # content only if streaming didn't surface any.
                    if not self._reasoning_buffer:
                        self._absorb_reasoning_from_assistant_message(msg)
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

    async def list_slash_commands(self) -> list[SlashCommand]:
        # Filesystem-only discovery — adapter-agnostic. The Claude
        # Agent SDK additionally auto-expands ``/cmd`` invocations
        # natively when the consumer passes them through execute(),
        # so consumers get both the palette metadata (here) and the
        # automatic dispatch (via execute("/cmd args")).
        from airframe.slash_commands import discover

        return discover(self._slash_commands)

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
        native_fragment = f"native={self._native_tools_fingerprint}"
        permission_fragment = f"perm={self._permission_fingerprint}"
        max_turns_fragment = f"max_turns={max_turns}"
        provider_options_fragment = f"po={_claude_options_fingerprint(self._provider_options)}"
        metadata_fragment = f"md={self._metadata_fingerprint}"
        cache_key = (
            f"{schema_fragment}|{thinking_fragment}|{attachments_fragment}|"
            f"{tools_fragment}|{mcp_fragment}|{native_fragment}|{permission_fragment}|"
            f"{max_turns_fragment}|{provider_options_fragment}|{metadata_fragment}"
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
        # Hooks carry two jobs: the ``on_event=`` observer fan-out and
        # the ``on_permission=`` gate. ``can_use_tool`` above is never
        # invoked under ``permission_mode="bypassPermissions"`` (the SDK
        # only calls it for tool calls that would otherwise prompt), so
        # the PreToolUse hook is the channel that actually blocks. Build
        # the config whenever either callback is present.
        if self._on_event is not None or self._on_permission is not None:
            options_kwargs["hooks"] = _build_claude_hooks_config(
                self._on_event,
                session=self,
                on_permission=self._on_permission,
            )
        if self._metadata is not None and self._metadata.user_id:
            options_kwargs["user"] = self._metadata.user_id
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
        # Native (vendor-hosted) tools enter allowed_tools by their plain
        # Claude CLI name (``WebSearch``/``WebFetch``) — the SDK already
        # knows + executes them, so unlike FunctionTools there's no MCP
        # server to register, just the name in the allowlist.
        for native_name in _translate_native_tools_for_claude(self._native_tools):
            if native_name not in allowed_tools:
                allowed_tools.append(native_name)
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
        from claude_agent_sdk import AssistantMessage, RateLimitEvent, ResultMessage

        self._reset_reasoning_buffer()
        await client.query(prompt)
        final: Any = None
        async for msg in client.receive_response():
            if isinstance(msg, RateLimitEvent):
                self._absorb_rate_limit_event(msg)
                continue
            if isinstance(msg, AssistantMessage):
                self._absorb_reasoning_from_assistant_message(msg)
                continue
            if isinstance(msg, ResultMessage):
                final = msg
        return final

    def _reset_reasoning_buffer(self) -> None:
        """Drop any reasoning text captured during the previous turn."""
        self._reasoning_buffer.clear()

    def _absorb_reasoning_from_assistant_message(self, msg: Any) -> None:
        """Collect ``ThinkingBlock.thinking`` text from an AssistantMessage.

        Claude packages the model's extended-thinking trace as
        :class:`ThinkingBlock` content blocks. The blocks may be
        interleaved with :class:`TextBlock` / :class:`ToolUseBlock`
        entries; we only consume the thinking text and ignore the
        rest (other blocks are handled by the streaming path or by
        :meth:`_build_result`).
        """
        from claude_agent_sdk import ThinkingBlock

        for block in getattr(msg, "content", None) or []:
            if isinstance(block, ThinkingBlock):
                text = getattr(block, "thinking", None)
                if text:
                    self._reasoning_buffer.append(text)

    def _reasoning_snapshot(self) -> str | None:
        """Snapshot the per-turn reasoning trace, or ``None`` if empty."""
        if not self._reasoning_buffer:
            return None
        return "".join(self._reasoning_buffer)

    def _absorb_rate_limit_event(self, event: Any) -> None:
        """Update ``_rate_limit_windows`` from a Claude ``RateLimitEvent``.

        Each SDK event carries one window's state via
        ``event.rate_limit_info.rate_limit_type``. We key by that
        type name so multiple events (different windows) accumulate;
        re-emission of the same window overwrites the prior entry
        with the freshest state.
        """
        info = getattr(event, "rate_limit_info", None)
        if info is None:
            return
        window_name = getattr(info, "rate_limit_type", None) or "default"
        resets_at_unix = getattr(info, "resets_at", None)
        reset_at: datetime | None
        if resets_at_unix is not None:
            try:
                reset_at = datetime.fromtimestamp(int(resets_at_unix), tz=UTC)
            except (TypeError, ValueError, OverflowError, OSError):
                reset_at = None
        else:
            reset_at = None
        self._rate_limit_windows[window_name] = RateLimitWindow(
            name=window_name,
            utilization=getattr(info, "utilization", None),
            reset_at=reset_at,
            status=getattr(info, "status", None),
        )
        # Overage signal: surface as a separate "overage" window so
        # consumers can see both the per-window utilisation and the
        # overage state without merging fields onto one shape.
        overage_status = getattr(info, "overage_status", None)
        if overage_status is not None:
            overage_resets_at = getattr(info, "overage_resets_at", None)
            overage_reset: datetime | None
            if overage_resets_at is not None:
                try:
                    overage_reset = datetime.fromtimestamp(int(overage_resets_at), tz=UTC)
                except (TypeError, ValueError, OverflowError, OSError):
                    overage_reset = None
            else:
                overage_reset = None
            self._rate_limit_windows["overage"] = RateLimitWindow(
                name="overage",
                reset_at=overage_reset,
                status=overage_status,
            )

    def _rate_limit_snapshot(self) -> RateLimitInfo | None:
        """Snapshot the per-window state for attachment to a result."""
        if not self._rate_limit_windows:
            return None
        return RateLimitInfo(windows=tuple(self._rate_limit_windows.values()))

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

        rate_limit = self._rate_limit_snapshot()
        reasoning = self._reasoning_snapshot()

        if schema is None:
            return RuntimeResult(
                text=text,
                structured=None,
                cost=cost,
                finish=result_msg.stop_reason,
                reasoning=reasoning,
                rate_limit=rate_limit,
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
            reasoning=reasoning,
            rate_limit=rate_limit,
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


#: Maps portable :class:`~airframe.native_tools.NativeCapability`
#: members to the Claude CLI built-in tool names that serve them.
#: Only hosted (Anthropic-executed) tools belong here — local-
#: execution built-ins (Bash/Read/Write) are out of scope for the
#: native-tools abstraction.
_NATIVE_CAPABILITY_TO_CLAUDE_TOOL: dict[NativeCapability, str] = {
    NativeCapability.WEB_SEARCH: "WebSearch",
    NativeCapability.WEB_FETCH: "WebFetch",
}


def _translate_native_tools_for_claude(native_tools: list[NativeTool]) -> list[str]:
    """Translate resolved :class:`NativeTool` items into Claude tool names.

    Semantic tools map through :data:`_NATIVE_CAPABILITY_TO_CLAUDE_TOOL`; raw
    tools (already filtered to ``provider_id == "claude"`` by
    :func:`_resolve_native_tools`) pass their ``name`` through verbatim. The
    list is pre-validated at ``session()`` time — every semantic capability here
    is in :attr:`ClaudeCodeRuntime.SUPPORTED_NATIVE_TOOLS` — so a missing map
    entry is an internal invariant breach, not a user error.

    ``NativeTool.options`` is accepted on the type but not yet wired into the
    Claude CLI here (it has no per-tool option channel on ``allowed_tools``);
    options still participate in the session cache fingerprint so a change forces
    a reconnect once a future option channel lands.
    """
    names: list[str] = []
    for t in native_tools:
        if t.capability is not None:
            name = _NATIVE_CAPABILITY_TO_CLAUDE_TOOL.get(t.capability)
            if name is None:  # pragma: no cover — guarded by _resolve_native_tools
                raise UnsupportedFeatureError(
                    f"claude_code: native capability {t.capability.value!r} has no "
                    f"Claude tool mapping.",
                    feature=Feature.TOOLS_NATIVE,
                )
            names.append(name)
        elif t.name is not None:
            names.append(t.name)
    return names


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

    .. note::

       This channel is **inert** under the adapter's
       ``permission_mode="bypassPermissions"`` — the SDK only invokes
       ``can_use_tool`` for calls that would otherwise raise an
       interactive prompt, and bypass mode means none do. The gate
       that actually blocks lives on the ``PreToolUse`` hook
       (:func:`_build_pre_tool_use_gate`); this wrapper stays wired so
       the callback is still honoured if a consumer reaches through
       :meth:`~ClaudeCodeRuntime.unwrap` to run under a stricter
       permission mode.

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
    on_event: Callable[[HookEvent], None] | None,
    *,
    session: ClaudeCodeSession,
    on_permission: PermissionCallback | None = None,
) -> dict[str, list[Any]]:
    """Build :attr:`ClaudeAgentOptions.hooks` that fans every native
    Claude hook into the user's ``on_event=`` callback and — when
    ``on_permission=`` is set — gates tool calls on the ``PreToolUse``
    hook.

    The SDK dispatches each hook with
    ``(HookInput, tool_use_id, HookContext)`` and expects a JSON
    output back. The ``on_event=`` observer contract is pure
    observation — it never blocks or modifies the SDK flow — so
    without ``on_permission=`` the returned
    :class:`SyncHookJSONOutput` is always empty (``{}``). The session
    keeps the latest known ``session_id`` so subsequent
    synthesised events (``session_end`` at close) carry the right
    value.

    ``on_permission=`` rides the same ``PreToolUse`` hook rather than
    :attr:`ClaudeAgentOptions.can_use_tool`, which the SDK only invokes
    for calls that would otherwise prompt — under the adapter's
    ``permission_mode="bypassPermissions"`` nothing ever reaches that
    channel. ``PreToolUse`` fires for *every* tool call regardless of
    permission mode, so it is the only place a ``"deny"`` can actually
    block. See :func:`_build_pre_tool_use_gate`.

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

    gate = (
        _build_pre_tool_use_gate(on_permission, server_names=session._known_mcp_servers)
        if on_permission is not None
        else None
    )

    async def _make_handler(kind: str, hook_input: Any) -> dict[str, Any]:
        # Hook inputs are TypedDicts at runtime (plain dicts). Pull
        # the session_id off the input when present and forward to
        # the session so synthesised events stay consistent.
        if isinstance(hook_input, dict):
            sid = hook_input.get("session_id")
            if sid:
                session.id = sid
        if on_event is not None:
            payload = _extract_claude_hook_payload(kind, hook_input)
            _fire_hook_event(
                on_event,
                kind,
                session_id=session.id,
                payload=payload,
            )
        if gate is not None and kind == "pre_tool_use":
            return await gate(hook_input)
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


def _build_pre_tool_use_gate(
    callback: PermissionCallback,
    *,
    server_names: frozenset[str],
) -> Callable[[Any], Awaitable[dict[str, Any]]]:
    """Wrap a :class:`PermissionCallback` as a ``PreToolUse`` hook body.

    The returned coroutine takes the raw ``PreToolUseHookInput`` dict,
    builds an airframe :class:`PermissionRequest` from it, awaits the
    user's callback, and translates the :data:`PermissionDecision` into
    the native hook response:

    * ``"deny"`` → ``{"hookSpecificOutput": {"hookEventName":
      "PreToolUse", "permissionDecision": "deny",
      "permissionDecisionReason": …}}``. The CLI blocks the call and
      feeds the reason back to the model as a tool failure, so the model
      can recover and keep working.
    * ``"allow"`` / ``"defer"`` → ``{}``. The adapter deliberately does
      *not* emit ``permissionDecision: "allow"`` for ``"allow"``: an
      empty response falls through to the session's existing posture
      (``permission_mode="bypassPermissions"``), which already permits
      the call, and avoids widening permissions for anything a
      downstream hook or setting would otherwise gate.

    Unlike the ``can_use_tool`` channel, ``PreToolUse`` carries no
    vendor-supplied rationale, so :attr:`PermissionRequest.reason` is
    always ``None`` here. ``tool_name`` gets the same
    ``mcp__<server>__`` stripping as the event stream so callbacks
    behave identically across providers.

    A callback that raises is logged and treated as ``"defer"`` — a
    buggy gate must never kill the session. Consumers needing
    fail-closed semantics implement that inside their own callback.
    """
    from airframe.permission import PermissionRequest

    async def _gate(hook_input: Any) -> dict[str, Any]:
        if not isinstance(hook_input, dict):
            return {}
        raw_name = str(hook_input.get("tool_name", ""))
        tool_name = _strip_mcp_prefix(raw_name, server_names)
        tool_input = hook_input.get("tool_input", {})
        request = PermissionRequest(
            tool_name=tool_name,
            tool_args=tool_input if isinstance(tool_input, dict) else {},
            reason=None,
        )
        try:
            decision = await callback.handle(request)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "claude_code: PermissionCallback raised for tool=%r error=%s; "
                "treating as 'defer' (vendor default)",
                tool_name,
                exc,
            )
            return {}
        if decision == "deny":
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        f"airframe PermissionCallback denied {tool_name!r}"
                    ),
                }
            }
        if decision == "defer":
            logger.debug(
                "claude_code: PermissionCallback returned 'defer' for tool=%r; "
                "falling through to the vendor default policy",
                tool_name,
            )
        return {}

    return _gate


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
