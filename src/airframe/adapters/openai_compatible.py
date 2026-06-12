"""``OpenAICompatibleRuntime`` — shared base for OpenAI-compatible HTTP vendors.

Many vendors speak OpenAI's Chat Completions wire format: OpenAI itself,
Together, Groq, Fireworks, Anyscale, OpenRouter, vLLM, LM Studio,
Anthropic's ``/v1/messages/openai`` proxy, and (in airframe's case
right now) the opencode-go Zen gateway. They share:

* AsyncOpenAI HTTP client (base URL + API key).
* ``response_format={"type": "json_schema", ...}`` for structured output.
* ``/v1/models`` listing for the model menu.
* The same exception taxonomy on the ``openai`` SDK.

This base captures all of that. Each vendor-specific subclass is
~30 lines: declares its ``PROVIDER_ID``, default base URL, default
model, auth resolver, and per-model metadata table. See
``opencode_zen.py`` for the canonical example.

**What lives in the subclass**:

- ``PROVIDER_ID`` — canonical provider name (drives discovery / dispatch).
- ``EXTRA_NAME`` — pip extra users install for this family.
- ``DEFAULT_BASE_URL`` — vendor's HTTP endpoint.
- ``DEFAULT_MODEL`` — fallback when no binding is specified.
- ``_resolve_api_key(api_key)`` — vendor-specific auth chain
  (env vars, credentials files, etc.).
- ``_METADATA`` — per-model display name / context window / pricing /
  capabilities. Joined against the live ``list_models()`` response.

**What lives in the base**:

- ``execute()`` — prompt → ``RuntimeResult`` with structured-output.
- ``list_models()`` — live menu enriched from ``_METADATA``.
- ``reset()`` / ``close()`` — stateless HTTP teardown.
- ``validate_binding()`` — canonical provider check.
- Error classification — maps ``openai.APIError`` subclasses onto
  airframe's ``Runtime*Error`` hierarchy.
- Envelope unwrap — handles single-key JSON wrappers some models emit.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from pydantic import BaseModel

from airframe.cache import CacheConfig
from airframe.cost import CostRecord
from airframe.errors import (
    AgentRuntimeError,
    RuntimeAuthError,
    RuntimeCancelledError,
    RuntimeModelNotFoundError,
    RuntimeProtocolError,
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
from airframe.options import OpenAICompatOptions
from airframe.protocol import (
    AgentRuntime,
    AgentSession,
    ProviderModel,
    RuntimeResult,
    UnsupportedBindingError,
)
from airframe.rate_limit import RateLimitInfo, RateLimitWindow
from airframe.sessions import (
    _MCP_TRANSPORT_TO_FEATURE,
    _check_budget_supported,
    _check_hooks_supported,
    _check_provider_options,
    _check_tools_supported,
    _enforce_budget_pre_turn,
    _fire_hook_event,
    _resolve_native_tools,
    _split_prompt_parts,
)
from airframe.slash_commands import SlashCommand, SlashCommandsConfig
from airframe.thinking import ThinkingMode
from airframe.tools import FunctionTool, McpServerRef

if TYPE_CHECKING:
    from collections.abc import Callable

    from airframe.hooks import HookEvent
    from airframe.options import ProviderOptions
    from airframe.permission import PermissionCallback

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Hard cap on tool-loop iterations within one user turn. A model that
#: keeps requesting tool calls indefinitely is a real failure mode;
#: capping it surfaces the runaway via :class:`RuntimeProtocolError`
#: instead of hanging the call. Twenty round-trips is roughly an order
#: of magnitude above any sane agent loop and well below most vendor
#: timeouts; consumers who genuinely need more can override later via
#: a future :class:`OpenAICompatOptions` field.
MAX_TOOL_ITERATIONS = 20


@dataclass(frozen=True, slots=True)
class ModelMeta:
    """Per-model metadata enrichment for OpenAI-compatible adapters.

    The vendor's ``/v1/models`` endpoint typically returns just IDs.
    Subclasses ship a ``_METADATA: dict[str, ModelMeta]`` table to
    annotate known IDs with display name / context window / pricing /
    capability flags. Unknown IDs (new vendor releases we haven't
    catalogued yet) come back with sensible defaults.
    """

    display_name: str
    context_window: int | None = None
    input_per_1k: float | None = None
    output_per_1k: float | None = None
    capabilities: frozenset[str] = field(default_factory=frozenset)


class OpenAICompatibleRuntime(AgentRuntime):
    """Base class for OpenAI-compatible HTTP-backed adapters.

    Subclasses configure the vendor-specific bits via class attributes
    and the ``_resolve_api_key()`` hook. The :class:`AgentRuntime`
    contract is satisfied entirely here; subclasses rarely need to
    override methods.
    """

    # --- Subclass configuration (override these) ----------------------------

    #: Canonical provider ID this adapter serves. Must be unique across
    #: all built-in adapters (driving :func:`airframe.list_providers`).
    PROVIDER_ID: ClassVar[str] = ""

    #: pip extra that brings the vendor SDK in. All OpenAI-compatible
    #: subclasses share the ``openai-compat`` extra.
    EXTRA_NAME: ClassVar[str] = "openai-compat"

    #: Underlying SDK that must be importable. Shared across the family.
    REQUIRES_PACKAGE: ClassVar[str] = "openai"

    #: Vendor HTTP endpoint. Subclasses set this; consumers can override
    #: at construction time.
    DEFAULT_BASE_URL: ClassVar[str] = ""

    #: Fallback model ID when ``execute()`` is called without a binding.
    DEFAULT_MODEL: ClassVar[str] = ""

    #: Per-model metadata table — keys are model IDs, values are
    #: :class:`ModelMeta`. Drives ``list_models()`` enrichment and
    #: ``cost_usd`` computation.
    METADATA: ClassVar[dict[str, ModelMeta]] = {}

    #: Features this runtime family exposes today.
    #:
    #: * ``STRUCTURED_OUTPUT_JSON_SCHEMA`` — wired via
    #:   ``response_format={"type":"json_schema",...}`` (Phase 0).
    #: * ``STREAMING`` — wired via the bespoke
    #:   :class:`OpenAICompatibleSession` using ``stream=True`` on
    #:   ``chat.completions.create()`` (Phase 1, Iteration C).
    #: * ``CANCEL`` — wired via :func:`asyncio.Task.cancel` on the
    #:   in-flight :meth:`execute` task, and via closing the in-flight
    #:   :class:`AsyncStream` for :meth:`stream` (Phase 1, Iteration C).
    #: * ``REASONING_EFFORT`` — wired via the ``reasoning_effort``
    #:   kwarg on ``chat.completions.create()`` (Phase 2, Iteration B).
    #:   The vendor (or specific model) rejects effort levels it
    #:   doesn't support; airframe forwards verbatim.
    #: * ``VISION_INPUT`` — wired via OpenAI's content-parts shape
    #:   (``[{"type":"image_url","image_url":{"url":"data:image/..;base64,.."}}]``)
    #:   (Phase 2, Iteration C). Path-only in v0; ``ImageInput.bytes_``
    #:   / ``url`` raise — Iteration D adds those.
    #: * ``TOOLS_FUNCTION`` — wired via the
    #:   ``tools=[{"type":"function","function":{...}}]`` shape on
    #:   ``chat.completions.create()`` plus a client-side tool-loop in
    #:   :class:`OpenAICompatibleSession`. Capped at
    #:   :data:`MAX_TOOL_ITERATIONS` round-trips per user turn
    #:   (Phase 3, Iteration B).
    #:
    #: ``FILE_INPUT`` stays False: file routing varies wildly across
    #: OpenAI-compatible vendors (``client.files.create`` semantics
    #: differ; some vendors don't support it at all). A future
    #: per-vendor opt-in subclass can flip it.
    #:
    #: ``SESSION_RESUME`` stays False: chat-completions has no
    #: server-side session, and the plan calls for raising
    #: :class:`~airframe.errors.UnsupportedFeatureError` on
    #: ``session(resume=...)``. Subclasses backed by the Responses API
    #: can override and wire it.
    #:
    #: ``REASONING_BUDGET_TOKENS`` stays False: only Anthropic's
    #: Messages API exposes a token-budget shape; OpenAI-compatible
    #: vendors use the literal effort enum.
    #:
    #: MCP-as-tool is OpenAI Responses-only; the Chat Completions
    #: surface this base targets does NOT support it, and that won't
    #: change. ``STRUCTURED_OUTPUT_STRICT`` stays False even though the
    #: SDK accepts ``strict: True`` — the base passes ``strict: False``
    #: for compat-vendor portability (Together / Groq / Fireworks /
    #: OpenRouter coverage is uneven). Phase 2 may add an explicit
    #: opt-in.
    SUPPORTED_FEATURES: ClassVar[frozenset[Feature]] = frozenset(
        {
            Feature.STRUCTURED_OUTPUT_JSON_SCHEMA,
            Feature.STREAMING,
            Feature.CANCEL,
            Feature.REASONING_EFFORT,
            Feature.VISION_INPUT,
            Feature.TOOLS_FUNCTION,
            Feature.LIFECYCLE_HOOKS,
            # Phase 5 Iteration D: client-side accumulation per
            # session. Both caps enforced at turn boundary in v0
            # (mid-turn interrupt is additive later). Note: ``max_turns``
            # is a *user-facing per-execute() budget*, distinct from
            # ``MAX_TOOL_ITERATIONS`` (the internal runaway guard
            # in the tool loop) — see the class docstring.
            Feature.BUDGET_USD_CAP,
            Feature.BUDGET_TURN_CAP,
            Feature.RATE_LIMIT_TELEMETRY,
            Feature.REASONING_OUTPUT,
            Feature.REQUEST_METADATA,
            Feature.COUNT_TOKENS,
            Feature.PROMPT_CACHE_CONTROL,
            Feature.SLASH_COMMANDS,
        }
    )

    #: The :class:`~airframe.hooks.HookEventKind` literals this
    #: adapter can emit through ``on_event=``. Synthesised from the
    #: client-side tool-loop in :class:`OpenAICompatibleSession`.
    #: ``pre_compact`` / ``rate_limit`` are **not emittable** —
    #: chat-completions has no compaction concept and the SDK
    #: doesn't surface rate-limit signals as discrete events.
    EMITTABLE_HOOK_KINDS: ClassVar[frozenset[str]] = frozenset(
        {
            "session_start",
            "session_end",
            "user_prompt_submit",
            "pre_tool_use",
            "post_tool_use",
            "tool_failure",
        }
    )

    #: Tag used in structured-log rows (``runtime=opencode_zen`` etc.).
    label: str = "openai_compatible"

    # --- Construction -------------------------------------------------------

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 600.0,
    ) -> None:
        self._default_model = model or self.DEFAULT_MODEL
        self._base_url = base_url or self.DEFAULT_BASE_URL
        self._api_key_override = api_key
        self._timeout = timeout
        self._client: Any | None = None  # AsyncOpenAI; lazy

    # --- Subclass hooks -----------------------------------------------------

    def _resolve_api_key(self, api_key: str | None) -> str:
        """Resolve the vendor API key.

        Default: explicit arg → ``{PROVIDER_ID}_API_KEY`` env var.
        Subclasses override to add vendor-specific lookups (credentials
        files, OAuth tokens, etc.).
        """
        if api_key:
            return api_key
        env_key = f"{self.PROVIDER_ID.upper().replace('-', '_')}_API_KEY"
        env = os.environ.get(env_key)
        if env:
            return env
        raise RuntimeAuthError(
            f"{type(self).__name__}: no API key found. Set {env_key} or pass api_key= explicitly."
        )

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
        # ephemeral. Consumers wanting context warmth across calls open
        # a session explicitly and reuse it.
        del persona  # accepted in the protocol but not consumed by this family
        sess = self.session(system=system, model=model, metadata=metadata, cache=cache)
        try:
            return await sess.execute(prompt, schema=schema, thinking=thinking, timeout=timeout)
        finally:
            await sess.close()

    async def reset(self) -> None:
        """No-op: OpenAI-compatible calls are stateless HTTP."""
        return None

    async def close(self) -> None:
        client = self._client
        self._client = None
        if client is None:
            return
        try:
            await client.close()
        except Exception as exc:  # noqa: BLE001 — teardown never raises
            logger.debug("%s.close_failed error=%s", self.label, exc)

    def validate_binding(self, binding: ProviderModel) -> bool:
        return binding.provider_id == self.PROVIDER_ID

    def supports(self, feature: Feature, model: ProviderModel | None = None) -> bool:
        return feature in self.SUPPORTED_FEATURES

    def supported_native_tools(
        self, model: ProviderModel | None = None
    ) -> frozenset[NativeCapability]:
        # The Chat Completions wire shape this base wraps has no hosted-tool
        # slot — OpenAI's web_search / code_interpreter / file_search are
        # Responses-API tools. A future OpenAIResponsesRuntime is where native
        # tools would land; this family permanently declines.
        return frozenset()

    def unwrap(self, cls: type[T]) -> T:
        from openai import AsyncOpenAI

        if isinstance(self, cls):
            return self  # type: ignore[return-value]
        if cls is AsyncOpenAI:
            # Build lazily — same pattern execute() / list_models() use.
            return self._ensure_client()  # type: ignore[return-value]
        raise TypeError(
            f"{type(self).__name__} cannot unwrap to {cls!r}; supported types are "
            f"{type(self).__name__} and openai.AsyncOpenAI."
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
        """Open a bespoke :class:`OpenAICompatibleSession`.

        Iteration C of Phase 1 replaces the
        :class:`~airframe.sessions._ThinAgentSession` placeholder with a
        real client-side ``messages=[]`` buffer, ``stream=True``-backed
        :meth:`AgentSession.stream`, and
        :func:`asyncio.Task.cancel`-driven
        :meth:`AgentSession.cancel`.

        Args:
            tools: List of :class:`~airframe.tools.FunctionTool` to
                expose to the model for the session's lifetime.
                Translated to ``tools=[{"type":"function",...}]`` on
                every ``chat.completions.create()`` call. The session
                drives the client-side tool-loop: parse
                ``response.choices[0].message.tool_calls``, dispatch
                each handler, append ``role="tool"`` messages with the
                JSON-serialised result, and re-call until the model
                stops requesting tools (or
                :data:`MAX_TOOL_ITERATIONS` is hit, at which point the
                runaway is surfaced as
                :class:`~airframe.errors.RuntimeProtocolError`).
                Phase 3 Iteration B.
            mcp_servers: Accepted for protocol parity but the Chat
                Completions wire shape this base wraps has no MCP-as-
                tool slot — that translation is Responses-API only.
                Non-empty list raises
                :class:`~airframe.errors.UnsupportedFeatureError`
                pointing at the (future) ``OpenAIResponsesRuntime``,
                which would be the path for MCP on the OpenAI side.
                The decline is **permanent** for the chat-completions
                family (Phase 4 Iteration D); the
                :attr:`~airframe.errors.UnsupportedFeatureError.feature`
                attribute carries the first ref's transport so
                consumer code branching on
                :data:`~airframe.features.Feature.TOOLS_MCP_STDIO` /
                :data:`~airframe.features.Feature.TOOLS_MCP_HTTP` /
                :data:`~airframe.features.Feature.TOOLS_MCP_SSE`
                still works.
            on_permission: Phase 5 scaffolding accepted by the
                signature; non-None raises
                :class:`~airframe.errors.UnsupportedFeatureError`.
                Chat Completions has no permission wire shape — the
                decline is **permanent** for this compat family
                (Phase 5 Iteration B); a future
                ``OpenAIResponsesRuntime`` could wire it.
            on_event: Phase 5 scaffolding accepted by the signature;
                non-None raises until Phase 5 Iteration C synthesises
                :class:`~airframe.hooks.HookEvent` from the client-
                side tool-loop in
                :class:`OpenAICompatibleSession`.

        Raises:
            UnsupportedFeatureError: when ``resume`` is non-None.
                Chat-completions vendors have no server-side session;
                subclasses backed by the Responses API can override.
        """
        if resume is not None:
            raise UnsupportedFeatureError(
                f"{type(self).__name__}.session(resume=...) is not supported — "
                "chat-completions vendors have no server-side session. "
                "Check runtime.supports(Feature.SESSION_RESUME) first.",
                feature="session_resume",
            )
        _check_tools_supported(
            tools,
            adapter_label=self.label,
            feature_supported=self.supports(Feature.TOOLS_FUNCTION),
        )
        _resolve_native_tools(
            native_tools,
            adapter_label=self.label,
            provider_id=self.PROVIDER_ID,
            feature_supported=self.supports(Feature.TOOLS_NATIVE),
            supported_capabilities=self.supported_native_tools(model),
        )
        if on_permission is not None:
            # Phase 5 Iteration B — Chat Completions has no permission
            # wire shape. Decline is permanent for this compat family;
            # point consumers at the future OpenAIResponsesRuntime
            # path (Responses API exposes a tool-permission concept).
            raise UnsupportedFeatureError(
                f"{type(self).__name__}.session(on_permission=...) is not "
                f"supported — Chat Completions has no tool-permission wire "
                f"shape. A future ``OpenAIResponsesRuntime`` (separate "
                f"from this compat family) could wire it. Check "
                f"runtime.supports(Feature.PERMISSION_CALLBACK) before "
                f"passing on_permission=.",
                feature=Feature.PERMISSION_CALLBACK,
            )
        _check_hooks_supported(
            on_event,
            adapter_label=self.label,
            supports=self.supports,
        )
        if mcp_servers:
            # Phase 4 Iteration D — the chat-completions wire shape
            # this base wraps has no MCP-as-tool slot. The Responses
            # API does, but that's a separate adapter family. Surface
            # an OpenAI-compat-specific decline pointing at the
            # future direct-API option instead of the generic
            # shared-helper message.
            first = mcp_servers[0]
            feature = _MCP_TRANSPORT_TO_FEATURE.get(first.transport, Feature.TOOLS_MCP_STDIO)
            raise UnsupportedFeatureError(
                f"{type(self).__name__}.session(mcp_servers=...) is not "
                f"supported — Chat Completions has no MCP-as-tool wire "
                f"shape; that lives on the Responses API. A future "
                f"``OpenAIResponsesRuntime`` (separate from this compat "
                f"family) could translate to the Responses-API "
                f'``{{"type": "mcp", ...}}`` tool shape. Check '
                f"runtime.supports(Feature.TOOLS_MCP_STDIO) before "
                f"passing mcp_servers=.",
                feature=feature,
            )
        _check_provider_options(
            provider_options,
            expected_type=OpenAICompatOptions,
            adapter_label=self.label,
        )
        compat_options = (
            provider_options if isinstance(provider_options, OpenAICompatOptions) else None
        )
        return OpenAICompatibleSession(
            self,
            system=system,
            model=model,
            tools=tools,
            on_event=on_event,
            provider_options=compat_options,
            metadata=metadata,
            cache=cache,
            slash_commands=slash_commands,
        )

    async def count_tokens(
        self,
        prompt: Prompt,
        *,
        system: str | None = None,
        model: ProviderModel | None = None,
    ) -> int:
        """Count prompt tokens via ``tiktoken``.

        Uses ``tiktoken.encoding_for_model(model_id)`` when the model
        is one tiktoken recognises (every OpenAI GPT-* family member);
        falls back to ``o200k_base`` (GPT-4o tokeniser) as a
        best-effort approximation for compat-vendor models that use
        OpenAI-compatible tokenisers. **Caveat:** Vendors using
        non-OpenAI tokenisers (DeepSeek's tokeniser, Llama's, etc.)
        get an *approximate* count — typically within 5–10% but not
        exact. Consumers who need tokeniser-accurate counts for
        non-OpenAI models should ``unwrap()`` and use the vendor's
        own counter.

        v1 supports plain-text and string-only multi-part prompts.
        Image / file attachments would require base64 expansion and
        per-vendor counting heuristics; deferred.
        """
        try:
            import tiktoken
        except ImportError as exc:
            raise UnsupportedFeatureError(
                f"{self.label}: count_tokens() requires the 'tiktoken' package. "
                f"Install via `pip install airframe-agents[openai-compat]` or "
                f"`pip install tiktoken`.",
                feature=Feature.COUNT_TOKENS,
            ) from exc

        text, images, files = _split_prompt_parts(
            prompt,
            adapter_label=self.label,
            supports_vision=True,
            supports_file=False,
        )
        if images or files:
            raise UnsupportedFeatureError(
                f"{self.label}: count_tokens() does not yet support image / "
                f"file attachments — only plain-text prompts.",
                feature=Feature.COUNT_TOKENS,
            )

        model_id = self._resolve_model(model) if model is not None else self._default_model
        try:
            encoding = tiktoken.encoding_for_model(model_id)
        except KeyError:
            # Compat vendors expose model IDs tiktoken doesn't know
            # ("gpt-5-nano" via OpenCode Zen, "qwen3" via OpenRouter,
            # etc.). Fall back to the GPT-4o tokeniser as a reasonable
            # approximation — every modern OpenAI-compatible vendor
            # ships a tokeniser at least similar to o200k_base.
            encoding = tiktoken.get_encoding("o200k_base")

        # Per OpenAI's tokens-cookbook formula: every message adds 3
        # tokens of overhead plus 1 token for the role. The conversation
        # gets a trailing 3-token "assistant priming" overhead.
        total = 0
        if system:
            total += 3 + 1 + len(encoding.encode(system))
        total += 3 + 1 + len(encoding.encode(text))
        total += 3  # assistant priming
        return total

    async def list_models(self) -> list[ModelInfo]:
        """Return the live model menu from the vendor.

        Hits ``GET <base_url>/models`` via :class:`AsyncOpenAI` and
        enriches each entry from the subclass's :attr:`METADATA` table.
        Unknown IDs come back with ``display_name=id`` and ``None`` /
        empty for the rest.
        """
        client = self._ensure_client()
        try:
            page = await client.models.list()
        except Exception as exc:
            raise self._classify_exception(exc) from exc

        out: list[ModelInfo] = []
        for entry in page.data:
            meta = self.METADATA.get(entry.id)
            out.append(
                ModelInfo(
                    id=entry.id,
                    display_name=meta.display_name if meta else entry.id,
                    provider_id=self.PROVIDER_ID,
                    context_window=meta.context_window if meta else None,
                    pricing_input_per_1k_usd=meta.input_per_1k if meta else None,
                    pricing_output_per_1k_usd=meta.output_per_1k if meta else None,
                    capabilities=meta.capabilities if meta else frozenset(),
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
                f"{type(self).__name__} cannot serve {model.label!r}; "
                f"provider must be {self.PROVIDER_ID!r}"
            )
        return model.model_id

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        from openai import AsyncOpenAI

        api_key = self._resolve_api_key(self._api_key_override)
        self._client = AsyncOpenAI(base_url=self._base_url, api_key=api_key, timeout=self._timeout)
        return self._client

    def _compute_cost_usd(
        self, model_id: str, *, input_tokens: int, output_tokens: int
    ) -> float | None:
        """Look up per-1K-token pricing from :attr:`METADATA`."""
        meta = self.METADATA.get(model_id)
        if meta is None or meta.input_per_1k is None or meta.output_per_1k is None:
            return None
        return round(
            (input_tokens / 1000.0) * meta.input_per_1k
            + (output_tokens / 1000.0) * meta.output_per_1k,
            6,
        )

    def _build_result(
        self,
        response: Any,
        *,
        model_id: str,
        schema: type[BaseModel] | None,
        rate_limit: RateLimitInfo | None = None,
        reasoning: str | None = None,
    ) -> RuntimeResult:
        if not response.choices:
            raise RuntimeProtocolError(
                f"{self.label}: response had no choices",
                body=str(response)[:500],
            )
        choice = response.choices[0]
        message = choice.message
        text = message.content or ""
        finish = choice.finish_reason

        structured: Any = None
        if schema is not None:
            structured = self._parse_structured(text, schema=schema)

        # DeepSeek-R1 and several compat reasoning models surface the
        # chain-of-thought trace on ``message.reasoning_content``; a few
        # vendors use ``message.reasoning``. Caller-supplied
        # ``reasoning`` (from the streaming accumulator) wins when
        # present — the response object will be empty in that path.
        if reasoning is None:
            reasoning = _extract_message_reasoning(message)

        usage = response.usage
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        details = getattr(usage, "prompt_tokens_details", None)
        cache_read = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
        # Phase 2 Iteration B: surface reasoning tokens when the
        # vendor reports them (GPT-5 / o-series via
        # ``completion_tokens_details.reasoning_tokens``).
        completion_details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = (
            int(getattr(completion_details, "reasoning_tokens", 0) or 0)
            if completion_details
            else 0
        )

        cost = CostRecord(
            provider_id=self.PROVIDER_ID,
            model_id=model_id,
            cost_usd=self._compute_cost_usd(
                model_id, input_tokens=input_tokens, output_tokens=output_tokens
            ),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=0,  # OpenAI Chat Completions doesn't expose write counts.
            finish=finish,
            reasoning_tokens=reasoning_tokens,
        )

        return RuntimeResult(
            text=text,
            structured=structured,
            cost=cost,
            finish=finish,
            reasoning=reasoning,
            rate_limit=rate_limit,
            raw=response,
        )

    def _parse_structured(self, text: str, *, schema: type[BaseModel]) -> dict[str, Any]:
        """Parse JSON content with light envelope-unwrapping.

        Most vendors honour ``response_format`` cleanly. A few wrap the
        payload in a single ``{"input": ...}`` / ``{"content": ...}``
        envelope (a quirk we've seen on multiple gateways). We unwrap
        one level when we see that exact shape; otherwise the validator
        catches it.
        """
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeStructuredOutputError(
                f"{self.label}: structured payload was not valid JSON: {exc}",
                body=text[:500],
            ) from exc
        return _unwrap_envelope(data)

    def _classify_exception(self, exc: BaseException) -> Exception:
        """Map openai SDK exceptions onto airframe's runtime hierarchy."""
        from openai import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            AuthenticationError,
            BadRequestError,
            NotFoundError,
            PermissionDeniedError,
            RateLimitError,
        )

        if isinstance(exc, AuthenticationError | PermissionDeniedError):
            return RuntimeAuthError(f"{self.label}: auth: {exc}")
        if isinstance(exc, NotFoundError):
            return RuntimeModelNotFoundError(f"{self.label}: model not found: {exc}")
        if isinstance(exc, BadRequestError):
            return RuntimeStructuredOutputError(
                f"{self.label}: bad request: {exc}",
                body=getattr(exc, "body", None),
            )
        if isinstance(exc, RateLimitError | APITimeoutError | APIConnectionError):
            rate_limit: RateLimitInfo | None = None
            if isinstance(exc, RateLimitError):
                headers = getattr(getattr(exc, "response", None), "headers", None)
                rate_limit = _parse_openai_rate_limit_headers(headers)
            return RuntimeTransientError(f"{self.label}: transient: {exc}", rate_limit=rate_limit)
        if isinstance(exc, APIStatusError):
            status = getattr(exc, "status_code", None)
            if status is not None and 500 <= status < 600:
                return RuntimeTransientError(f"{self.label}: 5xx: {exc}")
            return AgentRuntimeError(f"{self.label}: api error: {exc}")
        return AgentRuntimeError(f"{self.label}: unexpected {type(exc).__name__}: {exc}")


class OpenAICompatibleSession:
    """Bespoke :class:`~airframe.protocol.AgentSession` for OpenAI-compatible HTTP.

    Phase 1 Iteration C — the first per-vendor session that replaces
    the shared :class:`~airframe.sessions._ThinAgentSession` placeholder.

    The chat-completions wire format has no server-side session, so
    multi-turn conversation lives in a client-side
    :attr:`_messages` buffer: each :meth:`execute` /
    :meth:`stream` appends the user message before the call and the
    assistant response after success. Failures (including cancellation)
    pop the user message so a retry sends a clean history.

    **Streaming.** :meth:`stream` opens an
    :class:`openai.AsyncStream` with ``stream=True`` and
    ``stream_options={"include_usage": True}`` so the trailing
    :class:`~airframe.events.TurnComplete` carries a populated
    :class:`~airframe.cost.CostRecord`. Per-chunk text becomes
    :class:`~airframe.events.TextDelta`; the model's
    ``finish_reason`` lands on the final result.

    **Cancellation.** :meth:`cancel` aborts the in-flight turn:

    * For :meth:`execute`, cancels the wrapping :class:`asyncio.Task`;
      the awaiting call raises
      :class:`~airframe.errors.RuntimeCancelledError`.
    * For :meth:`stream`, sets a flag the generator checks between
      yields and closes the underlying :class:`AsyncStream` so the
      in-flight HTTP read unblocks. The generator raises
      :class:`RuntimeCancelledError` on its next yield boundary; the
      message buffer is rolled back to its pre-turn state.

    :attr:`id` is always ``None`` — there's no vendor-side session ID
    to surface. Consumer code branching on ``session.id is None`` can
    treat that as the "stateless HTTP" signal.
    """

    id: str | None = None

    def __init__(
        self,
        runtime: OpenAICompatibleRuntime,
        *,
        system: str | None = None,
        model: ProviderModel | None = None,
        tools: list[FunctionTool] | None = None,
        on_event: Callable[[HookEvent], None] | None = None,
        provider_options: OpenAICompatOptions | None = None,
        metadata: RequestMetadata | None = None,
        cache: CacheConfig | None = None,
        slash_commands: SlashCommandsConfig | None = None,
    ) -> None:
        self._runtime = runtime
        self._model = model
        self._messages: list[dict[str, Any]] = []
        if system:
            self._messages.append({"role": "system", "content": system})
        self._closed = False
        self._in_flight_task: asyncio.Task[Any] | None = None
        self._active_stream: Any | None = None
        self._stream_cancelled = False
        # Phase 5 Iteration C: lifecycle-hook observer. Chat
        # Completions has no native event channel; the adapter
        # synthesises events from the existing client-side tool-loop.
        # session_start fires on first execute(); session_end fires
        # on close(); per-tool events fire around handler invocation.
        self._on_event: Callable[[HookEvent], None] | None = on_event
        self._session_start_fired = False
        self._session_end_fired = False
        # Phase 5 Iteration D: per-session running budget. Both
        # caps enforced at turn boundary in v0. Distinct from the
        # tool-loop's MAX_TOOL_ITERATIONS runaway guard.
        self._cumulative_cost_usd: float = 0.0
        self._turn_count: int = 0
        # ProviderOptions — OpenAI-only knobs merged into every
        # chat.completions.create() call. Compat vendors silently
        # ignore unrecognised kwargs in their server-side validation,
        # so passing these to non-OpenAI compat endpoints is a no-op
        # rather than an error.
        self._provider_options: OpenAICompatOptions | None = provider_options
        # Phase 6 — REQUEST_METADATA. ``user_id`` → ``user=`` kwarg on
        # chat.completions.create (abuse-detection tag). ``tags`` →
        # ``metadata=`` kwarg (typed ``Dict[str, str]`` on the OpenAI
        # SDK). ``request_id`` → ``extra_headers={"X-Request-ID": ...}``
        # — defensive, not all compat vendors echo it.
        self._metadata: RequestMetadata | None = metadata
        # Phase 6 — PROMPT_CACHE_CONTROL. ``key`` → ``prompt_cache_key=``;
        # ``retention="short"`` → ``"in_memory"`` (~5min);
        # ``retention="long"`` → ``"24h"`` on the OpenAI SDK. The
        # cross-vendor cache= value takes precedence over the OpenAI-
        # specific OpenAICompatOptions.prompt_cache_key (consumers who
        # set both get the portable value through).
        self._cache: CacheConfig | None = cache
        # Phase 6 — SLASH_COMMANDS. Filesystem-only discovery. No
        # native channel on Chat Completions; consumers expand the
        # body themselves before calling execute().
        self._slash_commands: SlashCommandsConfig | None = slash_commands
        # Tools are fixed for the session's lifetime — translated once
        # and reused on every chat.completions.create() call. ``None``
        # / ``[]`` both mean "don't send the kwarg" so the wire shape
        # for tool-free sessions is unchanged.
        self._tools_by_name: dict[str, FunctionTool] = {t.name: t for t in (tools or [])}
        self._tools_wire: list[dict[str, Any]] | None = (
            _translate_tools_for_openai(tools) if tools else None
        )

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
        text, images, _files = _split_prompt_parts(
            prompt,
            adapter_label=self._runtime.label,
            supports_vision=True,
            supports_file=False,
        )
        reasoning_effort = _translate_thinking_for_openai(thinking, label=self._runtime.label)
        pre_len = len(self._messages)
        self._messages.append({"role": "user", "content": _build_user_content(text, images)})
        self._fire_session_start_if_needed()
        _fire_hook_event(
            self._on_event,
            "user_prompt_submit",
            session_id=None,
            payload={"prompt": text, "length": len(text)},
        )
        task = asyncio.create_task(
            self._do_execute(schema=schema, reasoning_effort=reasoning_effort, timeout=timeout)
        )
        self._in_flight_task = task
        try:
            result = await task
        except asyncio.CancelledError as exc:
            del self._messages[pre_len:]
            raise RuntimeCancelledError(f"{self._runtime.label}: cancelled") from exc
        except BaseException:
            del self._messages[pre_len:]
            raise
        finally:
            self._in_flight_task = None
        self._turn_count += 1
        self._cumulative_cost_usd += result.cost.cost_usd or 0.0
        return result

    async def _do_execute(
        self,
        *,
        schema: type[BaseModel] | None,
        reasoning_effort: str | None,
        timeout: float,
    ) -> RuntimeResult:
        """Drive the client-side tool-loop for one user turn.

        Sends the current ``messages`` buffer to
        :meth:`chat.completions.create`; if the response carries
        ``tool_calls``, dispatch each handler, append the assistant
        message (with ``tool_calls``) and one ``role="tool"`` reply per
        call, then re-call. Loops until the model emits a final text
        response or :data:`MAX_TOOL_ITERATIONS` round-trips elapse.
        Appends both intermediate tool round-trips and the final
        assistant message to ``self._messages`` so a follow-up turn
        sees the full history.
        """
        client = self._runtime._ensure_client()
        model_id = self._runtime._resolve_model(self._model)
        response_format = _build_response_format(schema)
        for _ in range(MAX_TOOL_ITERATIONS):
            create_kwargs: dict[str, Any] = {
                "model": model_id,
                "messages": list(self._messages),
                "response_format": response_format,
                "timeout": timeout,
            }
            if reasoning_effort is not None:
                create_kwargs["reasoning_effort"] = reasoning_effort
            if self._tools_wire is not None:
                create_kwargs["tools"] = self._tools_wire
            self._apply_provider_options(create_kwargs)
            self._apply_cache_config(create_kwargs)
            self._apply_request_metadata(create_kwargs)
            try:
                raw = await client.chat.completions.with_raw_response.create(**create_kwargs)
                response = raw.parse()
            except Exception as exc:
                raise self._runtime._classify_exception(exc) from exc
            rate_limit = _parse_openai_rate_limit_headers(getattr(raw, "headers", None))

            if not response.choices:
                raise RuntimeProtocolError(
                    f"{self._runtime.label}: response had no choices",
                    body=str(response)[:500],
                )
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                result = self._runtime._build_result(
                    response,
                    model_id=model_id,
                    schema=schema,
                    rate_limit=rate_limit,
                )
                self._messages.append({"role": "assistant", "content": result.text})
                return result

            # Intermediate tool round-trip — append the assistant
            # message carrying the tool_calls (the API requires this
            # before the matching role="tool" messages), then dispatch.
            self._messages.append(_assistant_tool_call_message(message, tool_calls))
            for tc in tool_calls:
                _fire_hook_event(
                    self._on_event,
                    "pre_tool_use",
                    session_id=None,
                    payload={
                        "tool_name": tc.function.name,
                        "tool_call_id": tc.id,
                        "arguments": tc.function.arguments or "",
                    },
                )
                output, is_error = await self._invoke_tool(
                    tool_name=tc.function.name,
                    arguments_json=tc.function.arguments or "",
                )
                _fire_hook_event(
                    self._on_event,
                    "tool_failure" if is_error else "post_tool_use",
                    session_id=None,
                    payload={
                        "tool_name": tc.function.name,
                        "tool_call_id": tc.id,
                        ("error" if is_error else "output"): output,
                    },
                )
                self._messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": _serialize_tool_output(output),
                    }
                )
                del is_error  # encoded into the message via the content; loop continues

        raise RuntimeProtocolError(
            f"{self._runtime.label}: tool loop exceeded "
            f"{MAX_TOOL_ITERATIONS} iterations — the model kept requesting "
            f"tools without producing a final response. This usually points "
            f"to a tool handler returning an output the model can't act on, "
            f"or a system prompt that doesn't tell the model how to stop."
        )

    async def _invoke_tool(self, *, tool_name: str, arguments_json: str) -> tuple[Any, bool]:
        """Run one tool handler. Returns ``(output, is_error)``.

        Both unknown-tool and parse/validation/handler failures come
        back as ``is_error=True`` with a human-readable string
        ``output`` so the model can see what happened and recover on
        its next turn. The model deciding "I should have called X
        differently" is one of the failure modes tool-loops are
        designed to survive — silently failing the whole turn would
        be hostile to that recovery path.
        """
        tool = self._tools_by_name.get(tool_name)
        if tool is None:
            return f"Tool {tool_name!r} is not registered on this session.", True
        try:
            args = json.loads(arguments_json) if arguments_json else {}
        except json.JSONDecodeError as exc:
            return f"Failed to parse tool arguments as JSON: {exc}", True
        try:
            params = tool.params.model_validate(args)
        except Exception as exc:  # noqa: BLE001 — surface Pydantic errors to the model
            return (f"Tool arguments did not match the {tool.params.__name__} schema: {exc}"), True
        try:
            output = await tool.handler(params)
        except Exception as exc:  # noqa: BLE001 — handler errors flow back to the model
            return f"{type(exc).__name__}: {exc}", True
        return output, False

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
        text, images, _files = _split_prompt_parts(
            prompt,
            adapter_label=self._runtime.label,
            supports_vision=True,
            supports_file=False,
        )
        reasoning_effort = _translate_thinking_for_openai(thinking, label=self._runtime.label)
        pre_len = len(self._messages)
        self._messages.append({"role": "user", "content": _build_user_content(text, images)})
        self._stream_cancelled = False
        self._fire_session_start_if_needed()
        _fire_hook_event(
            self._on_event,
            "user_prompt_submit",
            session_id=None,
            payload={"prompt": text, "length": len(text)},
        )
        # Accumulators that span the whole user turn (multiple model
        # turns under the tool loop). Assistant text is appended across
        # iterations because the final visible response may be split
        # by a mid-turn tool call ("First I'll look this up... [tool
        # call] ... The answer is 42").
        text_chunks_user_turn: list[str] = []
        reasoning_chunks_user_turn: list[str] = []
        usage: Any = None
        committed = False
        try:
            client = self._runtime._ensure_client()
            model_id = self._runtime._resolve_model(self._model)
            response_format = _build_response_format(schema)

            for _ in range(MAX_TOOL_ITERATIONS):
                stream_kwargs: dict[str, Any] = {
                    "model": model_id,
                    "messages": list(self._messages),
                    "response_format": response_format,
                    "timeout": timeout,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                if reasoning_effort is not None:
                    stream_kwargs["reasoning_effort"] = reasoning_effort
                if self._tools_wire is not None:
                    stream_kwargs["tools"] = self._tools_wire
                self._apply_provider_options(stream_kwargs)
                self._apply_cache_config(stream_kwargs)
                self._apply_request_metadata(stream_kwargs)
                try:
                    stream = await client.chat.completions.create(**stream_kwargs)
                except Exception as exc:
                    raise self._runtime._classify_exception(exc) from exc

                # Per-model-turn accumulators
                model_turn_text: list[str] = []
                tool_calls_by_index: dict[int, dict[str, Any]] = {}
                this_finish: str | None = None
                self._active_stream = stream
                try:
                    async for chunk in stream:
                        if self._stream_cancelled:
                            raise RuntimeCancelledError(f"{self._runtime.label}: stream cancelled")
                        choices = getattr(chunk, "choices", None) or []
                        if choices:
                            choice = choices[0]
                            delta = getattr(choice, "delta", None)
                            if delta is not None:
                                content = getattr(delta, "content", None)
                                if content:
                                    model_turn_text.append(content)
                                    text_chunks_user_turn.append(content)
                                    yield TextDelta(text=content)
                                reasoning_delta_text = _extract_delta_reasoning(delta)
                                if reasoning_delta_text:
                                    reasoning_chunks_user_turn.append(reasoning_delta_text)
                                    yield ReasoningDelta(text=reasoning_delta_text)
                                delta_tool_calls = getattr(delta, "tool_calls", None) or []
                                for dtc in delta_tool_calls:
                                    _accumulate_tool_call_delta(tool_calls_by_index, dtc)
                            chunk_finish = getattr(choice, "finish_reason", None)
                            if chunk_finish:
                                this_finish = chunk_finish
                        chunk_usage = getattr(chunk, "usage", None)
                        if chunk_usage is not None:
                            usage = chunk_usage
                finally:
                    self._active_stream = None

                if not tool_calls_by_index:
                    # Final model turn — build result, commit, yield.
                    full_text = "".join(text_chunks_user_turn)
                    structured: Any = None
                    if schema is not None:
                        structured = self._runtime._parse_structured(full_text, schema=schema)
                    cost = _build_cost_from_stream_usage(
                        usage,
                        runtime=self._runtime,
                        model_id=model_id,
                        finish=this_finish,
                    )
                    full_reasoning = (
                        "".join(reasoning_chunks_user_turn) if reasoning_chunks_user_turn else None
                    )
                    result = RuntimeResult(
                        text=full_text,
                        structured=structured,
                        cost=cost,
                        finish=this_finish,
                        reasoning=full_reasoning,
                        raw=None,
                    )
                    self._messages.append({"role": "assistant", "content": full_text})
                    committed = True
                    self._turn_count += 1
                    self._cumulative_cost_usd += result.cost.cost_usd or 0.0
                    yield TurnComplete(result=result)
                    return

                # Intermediate tool round-trip. Append the assistant
                # message carrying the tool_calls payload, then for each
                # tool call: emit ToolCallStart with the accumulated
                # arguments, invoke the handler, emit ToolCallResult,
                # and append the matching role="tool" message.
                ordered = sorted(tool_calls_by_index.items())
                synthetic_message = _synthesize_assistant_tool_message(
                    "".join(model_turn_text), ordered
                )
                self._messages.append(synthetic_message)
                for _idx, entry in ordered:
                    tc_id: str = entry["id"]
                    name: str = entry["name"]
                    args_str: str = entry["arguments"]
                    yield ToolCallStart(
                        tool_name=name,
                        tool_call_id=tc_id,
                        arguments_preview=args_str,
                    )
                    _fire_hook_event(
                        self._on_event,
                        "pre_tool_use",
                        session_id=None,
                        payload={
                            "tool_name": name,
                            "tool_call_id": tc_id,
                            "arguments": args_str,
                        },
                    )
                    output, is_error = await self._invoke_tool(
                        tool_name=name, arguments_json=args_str
                    )
                    yield ToolCallResult(tool_call_id=tc_id, output=output, is_error=is_error)
                    _fire_hook_event(
                        self._on_event,
                        "tool_failure" if is_error else "post_tool_use",
                        session_id=None,
                        payload={
                            "tool_name": name,
                            "tool_call_id": tc_id,
                            ("error" if is_error else "output"): output,
                        },
                    )
                    self._messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": _serialize_tool_output(output),
                        }
                    )
                # next iteration re-calls with the tool results in scope

            raise RuntimeProtocolError(
                f"{self._runtime.label}: tool loop exceeded "
                f"{MAX_TOOL_ITERATIONS} iterations — the model kept requesting "
                f"tools without producing a final response."
            )
        except BaseException:
            # Roll back any uncommitted turn state so the next attempt
            # sends a clean history. After the trailing TurnComplete
            # has yielded, ``committed`` is True and the buffer keeps
            # the new user/assistant/tool entries.
            if not committed:
                del self._messages[pre_len:]
            raise

    async def list_slash_commands(self) -> list[SlashCommand]:
        from airframe.slash_commands import discover

        return discover(self._slash_commands)

    async def cancel(self) -> None:
        # Signal stream() to raise on its next yield boundary.
        self._stream_cancelled = True
        # Abort the in-flight execute() task, if any.
        task = self._in_flight_task
        if task is not None and not task.done():
            task.cancel()
        # Close the in-flight openai AsyncStream, if any. Closing
        # aborts the underlying HTTP read so the generator's
        # ``async for`` doesn't block waiting for a chunk that will
        # never arrive.
        stream = self._active_stream
        if stream is not None:
            try:
                close = getattr(stream, "close", None)
                if close is not None:
                    res = close()
                    if asyncio.iscoroutine(res):
                        await res
            except Exception as exc:  # noqa: BLE001 — cancellation never raises
                logger.debug("%s.stream_close_failed error=%s", self._runtime.label, exc)

    async def close(self) -> None:
        # Phase 5 Iteration C: synthesise session_end at close if
        # session_start ever fired. Repeat close() calls are
        # idempotent — gated on both flags.
        already_closed = self._closed
        self._closed = True
        if (
            not already_closed
            and self._on_event is not None
            and self._session_start_fired
            and not self._session_end_fired
        ):
            self._session_end_fired = True
            _fire_hook_event(
                self._on_event,
                "session_end",
                session_id=None,
                payload={},
            )
        # Runtime owns the AsyncOpenAI client; never tear it down here.
        # Cancel any in-flight work as a courtesy.
        await self.cancel()

    def _apply_cache_config(self, kwargs: dict[str, Any]) -> None:
        """Merge :class:`CacheConfig` fields into a create() kwargs dict.

        Maps the cross-vendor ``CacheConfig`` onto OpenAI's two
        prompt-cache channels:

        * ``key`` → ``prompt_cache_key=`` (the explicit cache key).
        * ``retention="short"`` → ``prompt_cache_retention="in_memory"``
          (5-minute in-process cache).
        * ``retention="long"`` → ``prompt_cache_retention="24h"``
          (persistent cache).

        The cross-vendor ``cache=`` value takes precedence over the
        OpenAI-specific :attr:`OpenAICompatOptions.prompt_cache_key` —
        a consumer setting both reasonably expects the portable
        surface to win. Compat vendors that don't honour these kwargs
        silently drop them server-side.
        """
        cache = self._cache
        if cache is None:
            return
        if cache.key is not None:
            kwargs["prompt_cache_key"] = cache.key
        if cache.retention is not None:
            kwargs["prompt_cache_retention"] = "in_memory" if cache.retention == "short" else "24h"

    def _apply_request_metadata(self, kwargs: dict[str, Any]) -> None:
        """Merge :class:`RequestMetadata` fields into a create() kwargs dict.

        Maps the cross-vendor metadata namespace onto OpenAI's three
        request-level channels:

        * ``user_id`` → ``user=`` (abuse-detection tag).
        * ``tags`` → ``metadata=`` (typed ``Dict[str, str]``).
        * ``request_id`` → ``extra_headers={"X-Request-ID": ...}``.

        Compat vendors that don't honour one of these silently drop
        it server-side — the call still succeeds. Per the soft
        contract, this is not a feature gate.
        """
        md = self._metadata
        if md is None:
            return
        if md.user_id:
            kwargs["user"] = md.user_id
        if md.tags:
            existing = kwargs.get("metadata")
            kwargs["metadata"] = (
                {**existing, **md.tags} if isinstance(existing, dict) else dict(md.tags)
            )
        if md.request_id:
            extra = kwargs.get("extra_headers")
            header = {"X-Request-ID": md.request_id}
            kwargs["extra_headers"] = {**extra, **header} if isinstance(extra, dict) else header

    def _apply_provider_options(self, kwargs: dict[str, Any]) -> None:
        """Merge :class:`OpenAICompatOptions` fields into a create() kwargs dict.

        Called from both :meth:`_do_execute` and :meth:`stream` before
        the ``chat.completions.create()`` call so the same fields
        reach both code paths. Each field is only set when non-None
        — the OpenAI SDK rejects ``None`` for some of these (e.g.
        ``service_tier``) so we omit the kwarg entirely rather than
        passing a sentinel.

        Compat vendors that don't recognise a field silently ignore
        it in their server-side validation — passing OpenAI-only
        knobs to Together / Groq / Fireworks / OpenCodeZen is a
        no-op rather than an error.
        """
        po = self._provider_options
        if po is None:
            return
        if po.prompt_cache_key is not None:
            kwargs["prompt_cache_key"] = po.prompt_cache_key
        if po.prompt_cache_retention is not None:
            kwargs["prompt_cache_retention"] = po.prompt_cache_retention
        if po.service_tier is not None:
            kwargs["service_tier"] = po.service_tier
        if po.safety_identifier is not None:
            kwargs["safety_identifier"] = po.safety_identifier
        if po.verbosity is not None:
            kwargs["verbosity"] = po.verbosity
        if po.store is not None:
            kwargs["store"] = po.store

    def _fire_session_start_if_needed(self) -> None:
        """Emit ``session_start`` once per session at first
        ``execute()`` / ``stream()`` call.

        Chat Completions has no native session-start event; the
        adapter synthesises one at first use so consumer observers
        see a clean start → end pair.
        """
        if self._on_event is None or self._session_start_fired:
            return
        self._session_start_fired = True
        model_id = self._runtime._resolve_model(self._model) if self._model else None
        _fire_hook_event(
            self._on_event,
            "session_start",
            session_id=None,
            payload={"model": model_id} if model_id else {},
        )

    def unwrap(self, cls: type[T]) -> T:
        # Stateless HTTP — the only "session state" is the messages
        # buffer (no vendor object to expose). Identity-cast supported;
        # AsyncOpenAI lives on the runtime, reach it there.
        if isinstance(self, cls):
            return self  # type: ignore[return-value]
        raise TypeError(
            f"OpenAICompatibleSession has no vendor session object to unwrap "
            f"to {cls!r}; the AsyncOpenAI client lives on the runtime — "
            f"call runtime.unwrap(AsyncOpenAI) instead."
        )


def _extract_message_reasoning(message: Any) -> str | None:
    """Pull the model's reasoning trace off a non-streaming response message.

    DeepSeek-R1 and several derivative reasoning models surface the
    chain-of-thought as ``message.reasoning_content``. A few vendors
    use ``message.reasoning`` instead. Returns the first non-empty
    field found; ``None`` when neither is present (the standard
    OpenAI Chat Completions shape doesn't expose reasoning text on
    Chat Completions — only the token count via
    ``usage.completion_tokens_details.reasoning_tokens``).
    """
    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content:
        return str(reasoning_content)
    reasoning = getattr(message, "reasoning", None)
    if reasoning:
        return str(reasoning)
    return None


def _extract_delta_reasoning(delta: Any) -> str | None:
    """Pull reasoning text off one streaming ``ChatCompletionChunk`` delta.

    Mirrors :func:`_extract_message_reasoning` for the streaming wire:
    the same two field names (``reasoning_content`` /
    ``reasoning``) arrive piecewise on the ``delta`` object. Returns
    ``None`` when the delta carried no reasoning content.
    """
    reasoning_content = getattr(delta, "reasoning_content", None)
    if reasoning_content:
        return str(reasoning_content)
    reasoning = getattr(delta, "reasoning", None)
    if reasoning:
        return str(reasoning)
    return None


def _parse_openai_rate_limit_headers(headers: Any) -> RateLimitInfo | None:
    """Translate OpenAI-style ``x-ratelimit-*`` headers into :class:`RateLimitInfo`.

    OpenAI (and most compat vendors that mimic its surface) emit:

    * ``x-ratelimit-limit-requests`` / ``x-ratelimit-limit-tokens`` — window total.
    * ``x-ratelimit-remaining-requests`` / ``x-ratelimit-remaining-tokens`` — left in window.
    * ``x-ratelimit-reset-requests`` / ``x-ratelimit-reset-tokens`` — duration string
      until the window resets (``"1s"`` / ``"6m0s"`` / ``"1h2m3s"`` / ``"42ms"``).
    * ``retry-after`` — server-suggested wait in seconds (also a duration string
      on some vendors). Typically only set on 429 responses.

    Returns ``None`` when ``headers`` is falsy or carries no recognisable
    rate-limit data — adapters surface ``rate_limit=None`` on calls where
    the vendor stayed quiet, distinct from "the field exists but is empty."
    """
    if not headers:
        return None
    requests_window = _build_openai_window(headers, kind="requests")
    tokens_window = _build_openai_window(headers, kind="tokens")
    retry_after = _parse_retry_after(headers.get("retry-after"))
    windows: list[RateLimitWindow] = []
    if requests_window is not None:
        windows.append(_with_retry_after(requests_window, retry_after))
    if tokens_window is not None:
        windows.append(_with_retry_after(tokens_window, retry_after))
    if not windows:
        return None
    return RateLimitInfo(windows=tuple(windows))


def _build_openai_window(headers: Any, *, kind: str) -> RateLimitWindow | None:
    """One ``RateLimitWindow`` from the three ``x-ratelimit-*-{kind}`` headers."""
    limit_raw = headers.get(f"x-ratelimit-limit-{kind}")
    remaining_raw = headers.get(f"x-ratelimit-remaining-{kind}")
    reset_raw = headers.get(f"x-ratelimit-reset-{kind}")
    if limit_raw is None and remaining_raw is None and reset_raw is None:
        return None
    return RateLimitWindow(
        name=kind,
        remaining=_parse_int(remaining_raw),
        limit=_parse_int(limit_raw),
        reset_at=_reset_at_from_duration(reset_raw),
    )


def _with_retry_after(
    window: RateLimitWindow, retry_after_seconds: float | None
) -> RateLimitWindow:
    """Return a copy of ``window`` with ``retry_after_seconds`` set."""
    if retry_after_seconds is None:
        return window
    return RateLimitWindow(
        name=window.name,
        remaining=window.remaining,
        limit=window.limit,
        utilization=window.utilization,
        reset_at=window.reset_at,
        retry_after_seconds=retry_after_seconds,
        status=window.status,
    )


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_retry_after(value: Any) -> float | None:
    """OpenAI emits ``retry-after`` as an integer-seconds string."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return _duration_string_to_seconds(value)


def _reset_at_from_duration(value: Any) -> datetime | None:
    seconds = _duration_string_to_seconds(value)
    if seconds is None:
        return None
    return datetime.now(tz=UTC) + timedelta(seconds=seconds)


def _duration_string_to_seconds(value: Any) -> float | None:
    """Parse OpenAI's ``"1h2m3s"`` / ``"42ms"`` / ``"6.5s"`` duration strings."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    total = 0.0
    num = ""
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isdigit() or ch == ".":
            num += ch
            i += 1
            continue
        # Unit — one of ms / s / m / h. Order matters: check 'ms' before 's'.
        if text[i : i + 2] == "ms":
            unit = "ms"
            i += 2
        else:
            unit = ch
            i += 1
        if not num:
            return None
        amount = float(num)
        num = ""
        if unit == "ms":
            total += amount / 1000.0
        elif unit == "s":
            total += amount
        elif unit == "m":
            total += amount * 60.0
        elif unit == "h":
            total += amount * 3600.0
        else:
            return None
    if num:
        # Trailing bare number — treat as seconds (some vendors send "10").
        total += float(num)
    return total


def _build_response_format(schema: type[BaseModel] | None) -> dict[str, Any] | None:
    """Build the ``response_format`` kwarg for chat.completions.create()."""
    if schema is None:
        return None
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "strict": False,
            "schema": schema.model_json_schema(),
        },
    }


def _build_user_content(text: str, images: list[Any]) -> str | list[dict[str, Any]]:
    """Build the ``content`` field for a user message.

    Plain string when no images are present (keeps fixtures and wire
    dumps compact for the 99% case). Content-parts list when one or
    more :class:`~airframe.inputs.ImageInput` are attached, since the
    chat-completions content-parts shape doesn't accept a bare string
    alongside ``image_url`` entries.

    All three :class:`ImageInput` variants reach the vendor as an
    ``image_url`` entry — OpenAI's chat-completions API treats them
    uniformly:

    * ``path=`` → read the file, base64-encode, emit as a ``data:``
      URL with ``media_type`` (taken from the dataclass or sniffed
      from the file extension).
    * ``bytes_=`` → same data-URL shape, no filesystem read.
      ``media_type`` defaults to ``image/png`` when omitted.
    * ``url=`` → pass-through. The vendor fetches the remote image
      itself (every OpenAI-compatible vision model supports remote
      URLs).
    """
    if not images:
        return text
    import base64
    import mimetypes

    parts: list[dict[str, Any]] = []
    if text:
        parts.append({"type": "text", "text": text})
    for img in images:
        if img.url is not None:
            url = img.url
        else:
            if img.path is not None:
                media_type = img.media_type or mimetypes.guess_type(img.path)[0] or "image/png"
                with open(img.path, "rb") as fh:
                    raw = fh.read()
            else:
                media_type = img.media_type or "image/png"
                raw = img.bytes_ or b""
            b64 = base64.b64encode(raw).decode("ascii")
            url = f"data:{media_type};base64,{b64}"
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


def _translate_thinking_for_openai(thinking: ThinkingMode, *, label: str) -> str | None:
    """Translate :data:`ThinkingMode` to the ``reasoning_effort`` kwarg.

    The OpenAI-compatible wire format accepts a literal effort enum.
    Returns ``None`` to mean "don't send the kwarg" (default model
    behaviour); a literal string ``"minimal" | "low" | "medium" |
    "high"`` otherwise.

    Raises:
        UnsupportedFeatureError: when ``thinking`` is a dict
            (Claude-only ``budget_tokens`` shape).
    """
    if thinking is None or thinking == "disabled":
        # "disabled" means: don't enable reasoning. For OpenAI-compat
        # that's the same as omitting reasoning_effort (vendor default
        # for non-reasoning models is no reasoning anyway). This is
        # NOT a silent fallback — the consumer asked to turn it off,
        # and that's exactly what we do.
        return None
    if isinstance(thinking, str):
        # Literal effort. "minimal" is GPT-5-only; the vendor will
        # reject it on older models. We forward verbatim.
        return thinking
    if isinstance(thinking, dict):
        raise UnsupportedFeatureError(
            f"{label}: dict-shaped thinking ({{'budget_tokens': N}}) is "
            f"Claude-only; OpenAI-compatible vendors use a literal effort "
            f"level. Pass 'low' | 'medium' | 'high' instead.",
            feature="reasoning_budget_tokens",
        )
    raise UnsupportedFeatureError(
        f"{label}: unrecognised thinking mode {thinking!r}",
        feature="reasoning_effort",
    )


def _accumulate_tool_call_delta(by_index: dict[int, dict[str, Any]], delta: Any) -> None:
    """Fold one streamed ``delta.tool_calls`` entry into the per-turn buffer.

    The Chat Completions stream surfaces tool calls as fragments
    indexed by position in the eventual ``tool_calls`` array. The
    ``id`` and ``function.name`` typically arrive in the first delta
    for an index; ``function.arguments`` arrives across many chunks
    as a partial-JSON string that we concatenate. The result is
    keyed by ``index`` because that's the only identifier guaranteed
    to be present on every chunk.
    """
    idx = getattr(delta, "index", None)
    if idx is None:
        return
    entry = by_index.setdefault(idx, {"id": "", "name": "", "arguments": ""})
    tc_id = getattr(delta, "id", None)
    if tc_id:
        entry["id"] = tc_id
    fn = getattr(delta, "function", None)
    if fn is not None:
        name = getattr(fn, "name", None)
        if name:
            entry["name"] = name
        args_chunk = getattr(fn, "arguments", None)
        if args_chunk:
            entry["arguments"] += args_chunk


def _synthesize_assistant_tool_message(
    content: str, ordered_tool_calls: list[tuple[int, dict[str, Any]]]
) -> dict[str, Any]:
    """Build the assistant-with-tool_calls buffer entry from streamed deltas.

    Mirrors :func:`_assistant_tool_call_message` but constructed from
    the per-index dicts we accumulate while reading the stream. Empty
    ``content`` becomes ``None`` to match the wire shape OpenAI
    returns when the assistant only requested tools without saying
    anything. ``id`` falls back to a synthetic ``call_<index>`` when
    the vendor didn't supply one — every vendor we've tested does,
    but the fallback keeps the buffer well-formed regardless.
    """
    return {
        "role": "assistant",
        "content": content or None,
        "tool_calls": [
            {
                "id": entry["id"] or f"call_{idx}",
                "type": "function",
                "function": {
                    "name": entry["name"],
                    "arguments": entry["arguments"],
                },
            }
            for idx, entry in ordered_tool_calls
        ],
    }


def _build_cost_from_stream_usage(
    usage: Any,
    *,
    runtime: OpenAICompatibleRuntime,
    model_id: str,
    finish: str | None,
) -> CostRecord:
    """Compute a :class:`CostRecord` from the streaming usage frame.

    Mirrors the non-streaming path in :meth:`OpenAICompatibleRuntime._build_result`
    but works off the standalone ``usage`` frame the streaming API emits
    when ``stream_options.include_usage=True`` is set.
    """
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    details = getattr(usage, "prompt_tokens_details", None) if usage else None
    cache_read = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
    completion_details = getattr(usage, "completion_tokens_details", None) if usage else None
    reasoning_tokens = (
        int(getattr(completion_details, "reasoning_tokens", 0) or 0) if completion_details else 0
    )
    return CostRecord(
        provider_id=runtime.PROVIDER_ID,
        model_id=model_id,
        cost_usd=runtime._compute_cost_usd(
            model_id, input_tokens=input_tokens, output_tokens=output_tokens
        ),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=0,
        finish=finish,
        reasoning_tokens=reasoning_tokens,
    )


def _assistant_tool_call_message(message: Any, tool_calls: list[Any]) -> dict[str, Any]:
    """Build the ``role="assistant"`` buffer entry for a tool-call turn.

    The OpenAI API expects intermediate assistant turns that requested
    tools to carry both their ``content`` (often empty / ``None``) and
    the original ``tool_calls`` payload, so subsequent
    ``role="tool"`` messages can reference each ``tool_call_id``. The
    shape is symmetric to what the API returned.
    """
    return {
        "role": "assistant",
        "content": getattr(message, "content", None),
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "",
                },
            }
            for tc in tool_calls
        ],
    }


def _translate_tools_for_openai(tools: list[FunctionTool]) -> list[dict[str, Any]]:
    """Translate :class:`FunctionTool` instances to the OpenAI wire shape.

    Each tool becomes one ``{"type": "function", "function": {...}}``
    entry suitable for the ``tools=`` kwarg on
    :meth:`AsyncOpenAI.chat.completions.create`. The parameter schema
    comes from the tool's :attr:`FunctionTool.params` Pydantic model via
    :meth:`BaseModel.model_json_schema`.

    The result is computed once at session construction and reused on
    every API call — :class:`FunctionTool` is frozen and the schema
    payload is large enough that re-serialising every turn would be
    visible at scale.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.params.model_json_schema(),
            },
        }
        for tool in tools
    ]


def _serialize_tool_output(output: Any) -> str:
    """JSON-encode a tool handler's return value for the wire.

    The OpenAI tool-result message carries a string ``content`` field.
    Strings pass through verbatim (keeps already-formatted text legible
    in transcripts); everything else round-trips through
    :func:`json.dumps` with ``default=str`` so non-JSON types
    (datetimes, decimals, dataclasses) don't crash the loop. If
    serialisation fails outright the fallback is ``repr(output)`` —
    not perfect, but the model sees *something* and the consumer's
    handler bug surfaces in the trace.
    """
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, default=str)
    except (TypeError, ValueError):
        return repr(output)


_ENVELOPE_KEYS = frozenset(
    {
        "input",
        "output",
        "parameter",
        "parameters",
        "arguments",
        "content",
        "data",
        "result",
        "value",
    }
)


def _unwrap_envelope(payload: Any) -> Any:
    """Strip a single-key wrapper around the typed payload.

    Some OpenAI-compatible providers emit ``{"input": {...}}`` or
    ``{"content": "<json-string>"}`` instead of the bare payload. We
    unwrap one level when we see that exact shape; if the wrapper's
    value is a JSON string, decode it.
    """
    if not isinstance(payload, dict):
        return payload
    if len(payload) == 1:
        ((k, v),) = payload.items()
        if k in _ENVELOPE_KEYS:
            if isinstance(v, str):
                try:
                    parsed = json.loads(v)
                    return _unwrap_envelope(parsed)
                except json.JSONDecodeError:
                    return payload
            return _unwrap_envelope(v)
    return payload


__all__ = [
    "MAX_TOOL_ITERATIONS",
    "ModelMeta",
    "OpenAICompatibleRuntime",
    "OpenAICompatibleSession",
]
