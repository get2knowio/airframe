"""``KimiRuntime`` — :class:`AgentRuntime` over Moonshot AI's Kimi Agent SDK.

Wraps the official ``kimi-agent-sdk`` Python package (first-party from
the ``MoonshotAI`` org on GitHub; Apache-2.0) which itself is a thin
Python surface around the ``kimi-cli`` subprocess. Architecturally
this adapter is the closest analogue to :class:`ClaudeCodeRuntime` in
the lineup — both are subprocess-class agent SDKs with sessions,
streaming, approvals, and MCP.

**Iteration A — scaffolding only.** This module ships the protocol
surface: identity, auth resolution, ``validate_binding``,
``supports``, ``unwrap``, an offline-fallback ``list_models()``, and
a bespoke :class:`KimiSession` whose execute / stream signatures
match the protocol exactly (including the Phase 5 ``max_turns`` /
``max_budget_usd`` kwargs the conformance contracts pin). Every
per-feature kwarg is gated against the corresponding
:class:`Feature` flag and raises :class:`UnsupportedFeatureError`
when the capability is declined. ``execute()`` raises
:class:`NotImplementedError` *after* the gates pass — the SDK-backed
implementation lands in Iteration B. ``SUPPORTED_FEATURES`` declares
only :data:`Feature.STRUCTURED_OUTPUT_JSON_SCHEMA` (the universal
floor); the rest flip on in Iterations B–F per
``dev-docs/kimi-adapter-plan.md``.

**Auth.** Three options, checked in order:

1. Explicit ``api_key=`` / ``base_url=`` / ``model=`` constructor
   arguments. Highest precedence; forwarded into the SDK's
   :class:`Config`.
2. ``KIMI_API_KEY`` env var (and ``KIMI_BASE_URL`` / ``KIMI_MODEL_NAME``
   companions) — the SDK's native env-derived defaults.
3. The SDK's own resolution if neither of the above is set — the
   adapter doesn't override this layer; whatever the
   ``kimi-agent-sdk`` Config object picks up wins.

If no API key resolves through any layer, the first network call
raises :class:`RuntimeAuthError` pointing at
``https://platform.moonshot.ai/console/api-keys``.

**Python version.** ``kimi-agent-sdk`` requires Python ≥ 3.12 (a
stricter floor than airframe's ≥ 3.11). Users on 3.11 can install
``airframe-agents`` and use every other adapter, but
``pip install airframe-agents[kimi]`` will fail loudly with a clear
message from pip. Documented in
:doc:`/adapters/kimi`.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from pydantic import BaseModel

from airframe.errors import RuntimeAuthError, UnsupportedFeatureError
from airframe.events import RuntimeEvent, TurnComplete
from airframe.features import Feature
from airframe.inputs import Prompt
from airframe.models import ModelInfo
from airframe.options import KimiOptions
from airframe.protocol import (
    AgentRuntime,
    AgentSession,
    ProviderModel,
    RuntimeResult,
)
from airframe.sessions import (
    _check_budget_supported,
    _check_hooks_supported,
    _check_mcp_servers_supported,
    _check_permission_supported,
    _check_provider_options,
    _check_tools_supported,
    _split_prompt_parts,
)
from airframe.thinking import ThinkingMode

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from airframe.hooks import HookEvent
    from airframe.options import ProviderOptions
    from airframe.permission import PermissionCallback
    from airframe.tools import FunctionTool, McpServerRef

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Default Kimi model when no binding is specified. Kimi K2-thinking-turbo
#: is the SDK's documented default and the most capable agentic model in
#: the line as of early 2026. Override via :class:`ProviderModel` or the
#: ``KIMI_MODEL_NAME`` env var.
DEFAULT_KIMI_MODEL = "kimi-k2-thinking-turbo"

#: Default Moonshot API endpoint. Matches the OpenAI-compatible base URL
#: ``kimi-cli`` ships with; the Agent SDK reads the same env var.
DEFAULT_KIMI_BASE_URL = "https://api.moonshot.ai/v1"


#: Curated fallback catalogue for :meth:`KimiRuntime.list_models` when no
#: credential is available (or when the live ``/v1/models`` endpoint is
#: unreachable). Real catalogue surfaces from Moonshot's API when called
#: with a valid key — see :meth:`list_models`. Pricing left as ``None``
#: in Iteration A; populated alongside the in-tree ``_KIMI_PRICING``
#: table in Iteration E.
_FALLBACK_MODELS: tuple[ModelInfo, ...] = (
    ModelInfo(
        id="kimi-k2-thinking-turbo",
        display_name="Kimi K2 (Thinking, Turbo)",
        provider_id="kimi",
        context_window=256_000,
        pricing_input_per_1k_usd=None,
        pricing_output_per_1k_usd=None,
        capabilities=frozenset(),
    ),
    ModelInfo(
        id="kimi-k2-thinking",
        display_name="Kimi K2 (Thinking)",
        provider_id="kimi",
        context_window=256_000,
        pricing_input_per_1k_usd=None,
        pricing_output_per_1k_usd=None,
        capabilities=frozenset(),
    ),
)


class KimiRuntime(AgentRuntime):
    """``AgentRuntime`` over Moonshot AI's Kimi Agent SDK.

    Args:
        model: Default Kimi model identifier used when ``execute()`` is
            called without a :class:`ProviderModel` override. Resolution
            chain: this argument → ``KIMI_MODEL_NAME`` env var →
            :data:`DEFAULT_KIMI_MODEL`.
        base_url: Override the Moonshot API base URL. Resolution chain:
            this argument → ``KIMI_BASE_URL`` env var →
            :data:`DEFAULT_KIMI_BASE_URL`.
        api_key: Optional explicit Moonshot API key. Resolution chain:
            this argument → ``KIMI_API_KEY`` env var → SDK's own
            resolution. Mutation of ``os.environ`` is avoided; explicit
            keys are forwarded through the SDK's :class:`Config`.

    Iteration A: ``execute()`` raises :class:`NotImplementedError`
    after feature-gate checks pass. ``session()`` returns a
    :class:`KimiSession` whose protocol surface is complete and
    correctly gated, but whose terminal SDK call is the same
    :class:`NotImplementedError`. Iteration B replaces the SDK call
    sites with the real :class:`kimi_agent_sdk.Session` lifecycle.
    """

    label = "kimi"

    #: Canonical provider ID this adapter serves. Distinct from
    #: ``"moonshot"`` (reserved for a future OpenAI-compat sibling
    #: wrapping the ``api.moonshot.ai/v1`` chat-completions endpoint).
    PROVIDER_ID: ClassVar[str] = "kimi"

    #: Vendor SDK that must be importable for this adapter to work.
    REQUIRES_PACKAGE: ClassVar[str] = "kimi_agent_sdk"

    #: pip extra that brings the vendor SDK in.
    EXTRA_NAME: ClassVar[str] = "kimi"

    #: Features this runtime exposes today.
    #:
    #: Iteration A declares only :data:`~airframe.features.Feature.STRUCTURED_OUTPUT_JSON_SCHEMA`
    #: — the universal "every airframe adapter ships `execute(schema=...)`"
    #: floor that the conformance contract enforces. The actual SDK-
    #: backed implementation lands in Iteration B; until then,
    #: ``execute()`` raises :class:`NotImplementedError` with a clear
    #: message pointing at the iteration. Iterations B–F flip the
    #: remaining flags on as each feature lands per
    #: ``dev-docs/kimi-adapter-plan.md``.
    SUPPORTED_FEATURES: ClassVar[frozenset[Feature]] = frozenset(
        {Feature.STRUCTURED_OUTPUT_JSON_SCHEMA}
    )

    #: The set of :class:`~airframe.hooks.HookEventKind` literals this
    #: adapter can emit through ``on_event=``. Empty in Iteration A;
    #: Iteration E adds the six kinds the SDK surfaces natively.
    EMITTABLE_HOOK_KINDS: ClassVar[frozenset[str]] = frozenset()

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._default_model = model or os.environ.get("KIMI_MODEL_NAME") or DEFAULT_KIMI_MODEL
        self._base_url = base_url or os.environ.get("KIMI_BASE_URL") or DEFAULT_KIMI_BASE_URL
        # Explicit api_key wins; otherwise we defer resolution to the
        # SDK at first call (which reads KIMI_API_KEY). Storing the
        # explicit override per-instance keeps os.environ unmutated.
        self._api_key_override = api_key

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
        # Iteration A scaffold: the protocol surface is in place but no
        # behaviour is wired. Iteration B replaces this body with the
        # real kimi-agent-sdk Session-driven implementation.
        del prompt, schema, system, persona, model, thinking, timeout
        raise NotImplementedError(
            "KimiRuntime.execute() is not yet wired — Iteration B of the "
            "Kimi adapter plan (dev-docs/kimi-adapter-plan.md) lands the "
            "kimi-agent-sdk Session-backed execute / stream / cancel slice."
        )

    async def reset(self) -> None:
        # Sessionless at the runtime level — sessions own their own state.
        return None

    async def close(self) -> None:
        # Iteration A: no long-lived vendor handle to release. Iteration B
        # may add an HTTP client or similar; close() stays idempotent.
        return None

    def validate_binding(self, binding: ProviderModel) -> bool:
        # Kimi serves only ``kimi-*`` model IDs; reject anything else
        # (analogous to how CopilotRuntime rejects ``claude-*``). Foreign
        # provider IDs return False rather than raise — validate_binding
        # is meant for filtering candidate bindings.
        if binding.provider_id != self.PROVIDER_ID:
            return False
        return binding.model_id.startswith("kimi-") if binding.model_id else False

    def supports(self, feature: Feature, model: ProviderModel | None = None) -> bool:
        del model  # static per-adapter declaration in Iteration A
        return feature in self.SUPPORTED_FEATURES

    def unwrap(self, cls: type[T]) -> T:
        if isinstance(self, cls):
            return self  # type: ignore[return-value]
        raise TypeError(
            f"KimiRuntime cannot unwrap to {cls!r}; only KimiRuntime is "
            f"supported on the runtime today. Iteration B adds session-"
            f"level unwrap to the underlying kimi_agent_sdk.Session."
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
        """Open a :class:`KimiSession`.

        Iteration A returns a session whose protocol surface is fully
        in place — execute/stream signatures match the protocol; every
        feature kwarg (``tools``, ``mcp_servers``, ``on_permission``,
        ``on_event``, ``thinking``, ``max_turns``, ``max_budget_usd``,
        polymorphic ``prompt``) is gated against the corresponding
        :class:`Feature` flag and raises
        :class:`UnsupportedFeatureError` when the capability is
        declined. The *actual turn execution* path raises
        :class:`NotImplementedError` until Iteration B wires the
        ``kimi-agent-sdk`` ``Session``-backed implementation.

        ``resume=`` raises :class:`NotImplementedError` because
        :data:`~airframe.features.Feature.SESSION_RESUME` is not yet
        declared. The conformance contract checks this gate.
        """
        if resume is not None:
            raise NotImplementedError(
                "session(resume=...) is not wired yet — Iteration B of the "
                "Kimi adapter plan adds resume via the kimi-agent-sdk "
                "Session API. Check runtime.supports(Feature.SESSION_RESUME) first."
            )
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
            expected_type=KimiOptions,
            adapter_label=self.label,
        )
        kimi_options = provider_options if isinstance(provider_options, KimiOptions) else None
        return KimiSession(
            self,
            system=system,
            model=model,
            provider_options=kimi_options,
        )

    async def list_models(self) -> list[ModelInfo]:
        """Return Kimi models — live when credentialed, fallback otherwise.

        Iteration A: returns the curated :data:`_FALLBACK_MODELS` list
        unconditionally. Iteration B / E enriches this with live
        ``GET /v1/models`` results from Moonshot's OpenAI-compatible
        endpoint (Kimi shares the auth scheme with the chat-completions
        surface — same ``KIMI_API_KEY``).
        """
        return list(_FALLBACK_MODELS)

    # --- internal helpers ---------------------------------------------------

    def _resolve_api_key(self, api_key: str | None) -> str:
        """Resolve the Moonshot API key per the documented chain.

        Order:

        1. Explicit ``api_key`` argument (already passed through if not None).
        2. The instance-level override captured at ``__init__`` time.
        3. The ``KIMI_API_KEY`` env var.

        Raises:
            RuntimeAuthError: no key resolves through any layer.
        """
        if api_key:
            return api_key
        if self._api_key_override:
            return self._api_key_override
        env = os.environ.get("KIMI_API_KEY")
        if env:
            return env
        raise RuntimeAuthError(
            "KimiRuntime: no API key found. Set KIMI_API_KEY, "
            "pass api_key= explicitly, or mint one at "
            "https://platform.moonshot.ai/console/api-keys."
        )


class KimiSession(AgentSession):
    """Bespoke :class:`AgentSession` for :class:`KimiRuntime`.

    Iteration A protocol-correct stub. The full execute / stream
    signatures match the :class:`AgentSession` protocol (including the
    Phase 5 ``max_turns`` / ``max_budget_usd`` kwargs that the
    structural conformance contract checks), every per-feature kwarg
    is gated against the corresponding :class:`Feature` flag, and the
    failure mode for unsupported features is
    :class:`UnsupportedFeatureError` (not :class:`NotImplementedError`)
    — the conformance contracts distinguish the two.

    The terminal SDK call inside :meth:`execute` /
    :meth:`stream` raises :class:`NotImplementedError` until
    Iteration B replaces this class with the real ``kimi-agent-sdk``
    ``Session``-backed implementation.
    """

    def __init__(
        self,
        runtime: KimiRuntime,
        *,
        system: str | None = None,
        model: ProviderModel | None = None,
        provider_options: KimiOptions | None = None,
    ) -> None:
        self._runtime = runtime
        self._system = system
        self._model = model
        self._provider_options = provider_options
        self._closed = False
        self._in_flight = False
        # ``id`` populates from the kimi-agent-sdk Session in
        # Iteration B; ``None`` until then.
        self.id: str | None = None

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
        # Gate every Phase 2+ feature kwarg against capabilities BEFORE
        # entering the SDK path so the failure mode is
        # UnsupportedFeatureError (not NotImplementedError) — the
        # conformance contracts care about the distinction.
        if thinking is not None and not self._runtime.supports(Feature.REASONING_EFFORT):
            raise UnsupportedFeatureError(
                f"{self._runtime.label}: thinking= is not wired on this "
                f"adapter yet. Check runtime.supports(Feature.REASONING_EFFORT) "
                f"before passing thinking=.",
                feature=Feature.REASONING_EFFORT,
            )
        # Polymorphic prompts gate against VISION_INPUT / FILE_INPUT
        # via the shared helper. Plain ``str`` prompts pass through.
        _split_prompt_parts(
            prompt,
            adapter_label=self._runtime.label,
            supports_vision=self._runtime.supports(Feature.VISION_INPUT),
            supports_file=self._runtime.supports(Feature.FILE_INPUT),
        )
        _check_budget_supported(
            max_turns=max_turns,
            max_budget_usd=max_budget_usd,
            adapter_label=self._runtime.label,
            supports=self._runtime.supports,
        )
        # All gates passed — Iteration A stops here. Iteration B
        # replaces this raise with the real ``kimi-agent-sdk`` call.
        del schema, timeout
        self._in_flight = True
        try:
            raise NotImplementedError(
                "KimiSession.execute() is not yet wired — Iteration B of the "
                "Kimi adapter plan (dev-docs/kimi-adapter-plan.md) lands the "
                "kimi-agent-sdk Session-backed turn execution."
            )
        finally:
            self._in_flight = False

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
        # ``execute`` raises before returning a RuntimeResult in
        # Iteration A; the ``yield`` keeps Python recognising this as
        # an async generator function (the conformance contract checks
        # ``inspect.isasyncgenfunction``). Iteration B replaces this
        # with the SDK's typed event stream.
        result = await self.execute(
            prompt,
            schema=schema,
            thinking=thinking,
            max_turns=max_turns,
            max_budget_usd=max_budget_usd,
            timeout=timeout,
        )
        yield TurnComplete(result=result)

    async def cancel(self) -> None:
        if not self._in_flight:
            # No-op when idle — matches the conformance contract
            # ``test_session_cancel_when_idle_is_noop``.
            return
        # Mid-turn cancel isn't wired yet — Iteration B adds it via
        # the SDK's interrupt path.
        raise UnsupportedFeatureError(
            "KimiSession.cancel() is not wired yet; check "
            "runtime.supports(Feature.CANCEL) before calling.",
            feature=Feature.CANCEL,
        )

    async def close(self) -> None:
        self._closed = True

    def unwrap(self, cls: type[Any]) -> Any:
        if isinstance(self, cls):
            return self
        raise TypeError(
            f"{type(self).__name__} has no vendor-specific session object to "
            f"unwrap to {cls!r} yet — Iteration B exposes the underlying "
            f"kimi_agent_sdk.Session via this method."
        )


__all__ = [
    "DEFAULT_KIMI_BASE_URL",
    "DEFAULT_KIMI_MODEL",
    "KimiRuntime",
    "KimiSession",
]
