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

**Lifecycle.** ``execute()`` lazily constructs a
:class:`ClaudeSDKClient` keyed by ``(schema, system, model)`` — any
change to that triple forces a reconnect because ``output_format``
is baked into ``ClaudeAgentOptions`` at connect time. Subsequent
``execute()`` calls reuse the subprocess (warm cache accrues).
``reset()`` disconnects; the next ``execute()`` reconnects.
``close()`` is equivalent to ``reset()`` here.

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
from airframe.protocol import (
    AgentRuntime,
    AgentSession,
    ProviderModel,
    RuntimeResult,
    UnsupportedBindingError,
)
from airframe.sessions import _check_tools_supported, _split_prompt_parts
from airframe.thinking import ThinkingMode
from airframe.tools import FunctionTool

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

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
        provider_options: Any | None = None,
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
            provider_options: Reserved for Phase 2+ (currently unused).
        """
        _check_tools_supported(
            tools,
            adapter_label=self.label,
            feature_supported=self.supports(Feature.TOOLS_FUNCTION),
        )
        # provider_options accepted but unused — Phase 2+ fills each
        # ProviderOptions dataclass as the corresponding feature lands.
        del provider_options
        model_id = self._resolve_model(model) if model is not None else self._default_model
        return ClaudeCodeSession(
            self,
            resume=resume,
            system=system,
            model_id=model_id,
            tools=tools,
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

    async def execute(
        self,
        prompt: Prompt,
        *,
        schema: type[BaseModel] | None = None,
        thinking: ThinkingMode = None,
        timeout: float = 600.0,
    ) -> RuntimeResult:
        if self._closed:
            raise RuntimeError("session is closed")
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
                timeout=timeout,
            )
        )
        self._in_flight_task = task
        try:
            return await task
        except asyncio.CancelledError as exc:
            raise RuntimeCancelledError(f"{self._runtime.label}: cancelled") from exc
        finally:
            self._in_flight_task = None

    async def _do_execute(
        self,
        prompt: str,
        *,
        schema: type[BaseModel] | None,
        thinking: ThinkingMode,
        has_attachments: bool,
        timeout: float,
    ) -> RuntimeResult:
        client = await self._ensure_client(
            schema=schema, thinking=thinking, has_attachments=has_attachments
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
        timeout: float = 600.0,
    ) -> AsyncIterator[RuntimeEvent]:
        if self._closed:
            raise RuntimeError("session is closed")
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
                schema=schema, thinking=thinking, has_attachments=has_attachments
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
                                tool_name=_strip_mcp_prefix(block.name),
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
    ) -> Any:
        # Reconnect when the schema OR thinking OR attachments OR tools
        # fingerprint changes — ``output_format``, ``effort`` /
        # ``thinking``, ``allowed_tools``, and ``mcp_servers`` are all
        # baked into ClaudeAgentOptions at connect time. (model, system,
        # resume) are fixed for the session and don't contribute to the
        # key.
        schema_fragment = (
            f"{schema.__name__}|{schema.model_json_schema()}"
            if schema is not None
            else "__plain_text__"
        )
        effort, thinking_config = _translate_thinking_for_claude(thinking)
        thinking_fragment = f"effort={effort}|thinking={thinking_config}"
        attachments_fragment = f"attachments={has_attachments}"
        tools_fragment = f"tools={self._tools_fingerprint}"
        cache_key = (
            f"{schema_fragment}|{thinking_fragment}|{attachments_fragment}|{tools_fragment}"
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
            "max_turns": self._runtime._max_turns,
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

        # Bring in the auto-allowed Read tool for attachments and any
        # MCP-routed FunctionTools. The CLI uses ``mcp__<server>__<tool>``
        # naming for MCP tools, so the allowed_tools entries must match
        # that pattern; we let _translate_tools_for_claude build both
        # the server config and the allowed-tools names so the two
        # stay in sync.
        allowed_tools: list[str] = []
        if has_attachments:
            allowed_tools.append("Read")
        if self._tools:
            server_config, allowed_tool_names = _translate_tools_for_claude(self._tools)
            options_kwargs["mcp_servers"] = {AIRFRAME_MCP_SERVER_NAME: server_config}
            allowed_tools.extend(allowed_tool_names)
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


def _strip_mcp_prefix(tool_name: str) -> str:
    """Trim the ``mcp__<server>__`` prefix the SDK adds to MCP tool
    names so consumer code sees the same :attr:`FunctionTool.name` it
    registered."""
    prefix = f"mcp__{AIRFRAME_MCP_SERVER_NAME}__"
    if tool_name.startswith(prefix):
        return tool_name[len(prefix) :]
    return tool_name


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
