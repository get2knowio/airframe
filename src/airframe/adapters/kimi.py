"""``KimiRuntime`` — :class:`AgentRuntime` over Moonshot AI's Kimi Agent SDK.

Wraps the official ``kimi-agent-sdk`` Python package (first-party from
the ``MoonshotAI`` org on GitHub; Apache-2.0) which itself is a thin
Python surface around the ``kimi-cli`` subprocess. Architecturally
this adapter is the closest analogue to :class:`ClaudeCodeRuntime` in
the lineup — both are subprocess-class agent SDKs with sessions,
streaming, approvals, and MCP.

**Iteration B.** The protocol surface from Iteration A is now SDK-
backed: :class:`KimiSession` lazily creates / resumes a
``kimi_agent_sdk.Session`` on first :meth:`execute` / :meth:`stream`,
drives a turn through ``session.prompt()``, translates the SDK's
``WireMessage`` stream into airframe's :class:`RuntimeEvent` union,
and surfaces cost telemetry from ``TokenUsage`` events.
:data:`Feature.STREAMING`, :data:`Feature.CANCEL`, and
:data:`Feature.SESSION_RESUME` flip True. Structured output (the
``schema=`` kwarg on :meth:`execute`) still raises
:class:`NotImplementedError` — kimi-agent-sdk exposes no JSON-schema
constraint knob, and the wrap-don't-rewrite principle in
``CLAUDE.md`` rules out prompt-engineering it. Iteration D will
wire structured output via an in-process MCP forced-tool, the same
pattern :class:`CopilotRuntime` uses.

**Iteration C.** Polymorphic prompt + reasoning. :class:`KimiSession`
now translates :class:`~airframe.inputs.ImageInput` parts into the
SDK's :class:`ImageURLPart` shape (URL → pass-through, bytes / path →
base64 data URI), and threads ``thinking=`` through
:meth:`Session.create`'s boolean ``thinking`` kwarg.
:data:`Feature.REASONING_EFFORT` and :data:`Feature.VISION_INPUT`
flip True; :data:`Feature.FILE_INPUT` stays False (the SDK has no
prompt-side file slot — files reach Kimi tools via the session's
``work_dir``). ``thinking=`` is session-scoped at the SDK boundary
(baked at :meth:`Session.create` time, not per-prompt) so a toggle
between turns rebuilds the SDK session, mirroring how Codex rebuilds
its :class:`Thread` on a reasoning-effort change.

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

import base64
import logging
import mimetypes
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from pydantic import BaseModel

from airframe.cost import CostRecord
from airframe.errors import (
    RuntimeAuthError,
    RuntimeCancelledError,
    RuntimeProtocolError,
    RuntimeTransientError,
    UnsupportedFeatureError,
)
from airframe.events import (
    ReasoningDelta,
    RuntimeEvent,
    TextDelta,
    TurnComplete,
)
from airframe.features import Feature
from airframe.inputs import ImageInput, Prompt
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


def _translate_thinking_for_kimi(thinking: ThinkingMode) -> bool:
    """Translate airframe's :data:`ThinkingMode` onto the SDK's
    boolean ``thinking`` kwarg.

    The kimi-agent-sdk's :meth:`Session.create` accepts ``thinking:
    bool = False`` — there's no effort literal; the model itself
    decides reasoning depth based on input complexity once thinking
    is enabled. Mapping:

    * ``None`` / ``"disabled"`` → ``False``.
    * Any effort literal (``"minimal"``, ``"low"``, ``"medium"``,
      ``"high"``) → ``True``. Effort granularity is lost on the SDK
      boundary — the model decides depth itself.
    * ``{"budget_tokens": N}`` raises :class:`UnsupportedFeatureError`
      with :data:`Feature.REASONING_BUDGET_TOKENS` — Kimi has no
      token-budget channel for reasoning (Claude-only shape).

    Returns:
        ``True`` if thinking should be enabled on the SDK session,
        ``False`` otherwise.

    Raises:
        UnsupportedFeatureError: ``thinking`` is a ``dict`` shape, or
        an unrecognised value the literal type doesn't cover.
    """
    if thinking is None or thinking == "disabled":
        return False
    if isinstance(thinking, str):
        return True
    if isinstance(thinking, dict):
        raise UnsupportedFeatureError(
            "kimi: thinking=<dict> (budget_tokens shape) is Claude-only; "
            "kimi-agent-sdk exposes a boolean thinking knob — pass a literal "
            "effort string ('minimal'|'low'|'medium'|'high') instead, or omit "
            "thinking= to use the SDK's default (False).",
            feature=Feature.REASONING_BUDGET_TOKENS,
        )
    raise UnsupportedFeatureError(
        f"kimi: unsupported thinking= value {thinking!r}",
        feature=Feature.REASONING_EFFORT,
    )


def _image_to_data_uri(img: ImageInput) -> str:
    """Convert an :class:`ImageInput` to a value suitable for
    :class:`ImageURLPart.ImageURL.url`.

    Kosong's :class:`ImageURLPart` accepts both real URLs and
    ``data:`` URIs. The mapping:

    * ``url=`` → returned verbatim (HTTPS pass-through to the model).
    * ``bytes_=`` → ``data:{media_type};base64,{b64}``. ``media_type``
      defaults to ``image/png`` when omitted; the SDK doesn't sniff
      bytes itself, so a portable default beats raising.
    * ``path=`` → file read, base64-encode, build a ``data:`` URI.
      ``media_type`` resolves from :mod:`mimetypes` against the path
      extension when omitted.

    Raises:
        UnsupportedFeatureError: ``ImageInput.path`` points at a file
        that doesn't exist. This is a configuration error (we don't
        want a silent ``OSError`` from inside the SDK call).
    """
    if img.url is not None:
        return img.url
    if img.bytes_ is not None:
        media_type = img.media_type or "image/png"
        b64 = base64.b64encode(img.bytes_).decode("ascii")
        return f"data:{media_type};base64,{b64}"
    assert img.path is not None  # ImageInput.__post_init__ enforces one-of
    path = Path(img.path)
    if not path.is_file():
        raise UnsupportedFeatureError(
            f"kimi: ImageInput(path={img.path!r}) — file not found. "
            f"Pass an existing path, or use bytes_= / url= instead.",
            feature=Feature.VISION_INPUT,
        )
    if img.media_type is not None:
        path_media_type = img.media_type
    else:
        guessed, _ = mimetypes.guess_type(path.name)
        path_media_type = guessed or "application/octet-stream"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{path_media_type};base64,{b64}"


def _build_kimi_user_input(
    text: str,
    images: list[ImageInput],
    *,
    sdk_module: Any,
) -> Any:
    """Build the ``user_input`` value for :meth:`Session.prompt`.

    Returns ``text`` verbatim (a ``str``) when no images are present —
    the SDK accepts plain strings as the common case, no need to wrap
    in a single-element :class:`TextPart` list. When images are
    present, returns ``list[ContentPart]`` with one :class:`TextPart`
    leading and one :class:`ImageURLPart` per image.

    Args:
        text: The prompt text (already system-prompt-prepended by the
            caller).
        images: Image parts to attach. Empty list → plain-text path.
        sdk_module: The lazily-imported ``kimi_agent_sdk`` module
            (passed in so the helper doesn't import at module load —
            kimi-agent-sdk has hostile transitive deps that conflict
            with claude-agent-sdk; lazy imports are mandatory).

    Returns:
        Either a ``str`` (no images) or a ``list`` of :class:`TextPart`
        + :class:`ImageURLPart` instances ready to pass to
        :meth:`Session.prompt`.
    """
    if not images:
        return text
    text_part = sdk_module.TextPart(text=text)
    image_url_cls = sdk_module.ImageURLPart.ImageURL
    parts: list[Any] = [text_part]
    parts.extend(
        sdk_module.ImageURLPart(image_url=image_url_cls(url=_image_to_data_uri(img)))
        for img in images
    )
    return parts


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
            resolution. When set, the session injects the key into
            ``os.environ["KIMI_API_KEY"]`` for the duration of the
            SDK call (and restores the prior value on close) since
            ``Session.create`` doesn't accept an ``api_key=`` kwarg
            directly. Iteration C may switch to a typed ``Config``
            object once that surface is better understood.

    Iteration B: :class:`KimiSession` is fully SDK-backed —
    ``execute`` / ``stream`` drive ``Session.create`` /
    ``Session.resume`` lazily, translate the ``WireMessage`` stream
    into airframe's :class:`RuntimeEvent` union, and surface cost
    telemetry. ``execute(schema=…)`` still raises
    :class:`NotImplementedError` pending Iteration D's MCP-based
    forced-tool path for structured output.
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
    #: Iteration B adds :data:`Feature.STREAMING`,
    #: :data:`Feature.CANCEL`, and :data:`Feature.SESSION_RESUME` —
    #: the SDK exposes the corresponding surface natively:
    #: ``session.prompt()`` is the streaming async generator,
    #: ``session.cancel()`` sets the SDK's cancel event, and
    #: ``Session.resume(work_dir, session_id)`` resumes a prior
    #: session by ID. Iteration C adds :data:`Feature.REASONING_EFFORT`
    #: (mapped to the SDK's boolean ``thinking`` kwarg on
    #: :meth:`Session.create`) and :data:`Feature.VISION_INPUT` (image
    #: prompt parts translated to :class:`ImageURLPart`; URLs pass
    #: through, bytes / paths land as ``data:`` URIs).
    #: :data:`Feature.FILE_INPUT` stays False — the SDK has no
    #: prompt-side file slot; files reach Kimi tools via the session's
    #: ``work_dir`` instead. Structured output stays at the conformance
    #: floor only — :data:`Feature.STRUCTURED_OUTPUT_JSON_SCHEMA`
    #: declared True (every airframe adapter must declare it), but
    #: ``execute(schema=…)`` raises :class:`NotImplementedError`
    #: until Iteration D wires the MCP-based forced-tool pattern.
    SUPPORTED_FEATURES: ClassVar[frozenset[Feature]] = frozenset(
        {
            Feature.STRUCTURED_OUTPUT_JSON_SCHEMA,
            Feature.STREAMING,
            Feature.CANCEL,
            Feature.SESSION_RESUME,
            Feature.REASONING_EFFORT,
            Feature.VISION_INPUT,
        }
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

        The session's protocol surface is fully in place — execute /
        stream signatures match the protocol; every feature kwarg
        (``tools``, ``mcp_servers``, ``on_permission``, ``on_event``,
        ``thinking``, ``max_turns``, ``max_budget_usd``, polymorphic
        ``prompt``) is gated against the corresponding
        :class:`Feature` flag and raises
        :class:`UnsupportedFeatureError` when the capability is
        declined.

        Iteration B wires the SDK call sites. ``resume=`` resumes via
        ``kimi_agent_sdk.Session.resume`` (the SDK looks up
        ``session_id`` under ``work_dir``; ``None`` from the SDK on a
        missing ID surfaces as :class:`RuntimeProtocolError` at first
        :meth:`execute`/:meth:`stream`). ``schema=`` on
        :meth:`KimiSession.execute` still raises
        :class:`NotImplementedError` until Iteration D's MCP-based
        forced-tool path lands.
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
            expected_type=KimiOptions,
            adapter_label=self.label,
        )
        kimi_options = provider_options if isinstance(provider_options, KimiOptions) else None
        return KimiSession(
            self,
            resume=resume,
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

    Wraps a lazy-created ``kimi_agent_sdk.Session``. The SDK session
    is constructed on first :meth:`execute`/`stream` so the synchronous
    ``runtime.session()`` factory stays compatible with the async
    ``Session.create``/`Session.resume` calls underneath.

    **Auth.** The Kimi Agent SDK reads ``KIMI_API_KEY`` / ``KIMI_BASE_URL``
    / ``KIMI_MODEL_NAME`` from the environment via its ``Config`` layer.
    If :class:`KimiRuntime` was constructed with an explicit ``api_key=``
    or ``base_url=``, the session installs them into ``os.environ`` for
    the duration of the underlying SDK call so the SDK's auth chain
    picks them up. The env mutation is scoped to one call boundary and
    restored on close.

    **Approvals.** Iteration B hard-codes ``yolo=True`` on the SDK call
    (auto-approve every tool / shell invocation). Iteration D wires
    ``PermissionCallback`` properly via the SDK's
    ``approval_handler_fn`` bridge.

    **Structured output.** ``schema=`` raises
    :class:`NotImplementedError` — kimi-agent-sdk exposes no
    JSON-schema constraint knob. Iteration D adds it via the
    MCP-based forced-tool pattern.
    """

    def __init__(
        self,
        runtime: KimiRuntime,
        *,
        resume: str | None = None,
        system: str | None = None,
        model: ProviderModel | None = None,
        provider_options: KimiOptions | None = None,
    ) -> None:
        self._runtime = runtime
        self._resume_id = resume
        self._system = system
        self._model = model
        self._provider_options = provider_options
        self._sdk_session: Any = None  # lazy-created on first execute/stream
        # Pre-initialised so ``close()`` can run even if
        # ``_ensure_sdk_session`` was never called (e.g. a session
        # opened-and-immediately-closed never reaches the env-mutation
        # step).
        self._env_overrides: dict[str, str | None] = {}
        self._closed = False
        self._in_flight = False
        # ``id`` is populated from the SDK session once it materialises.
        # When ``resume=`` was passed we surface it eagerly so callers
        # can read it before driving a turn.
        self.id: str | None = resume
        # Iteration C: the SDK bakes ``thinking`` at session-create time;
        # tracking what's baked lets us rebuild when a later turn passes
        # a different ``thinking=``. ``None`` means "no session
        # materialised yet" — first turn populates this and the SDK
        # session together.
        self._sdk_thinking_baked: bool | None = None

    # --- SDK lifecycle ------------------------------------------------------

    async def _ensure_sdk_session(self, *, thinking: bool) -> Any:
        """Lazily create / resume / rebuild the underlying ``kimi_agent_sdk.Session``.

        Args:
            thinking: The desired ``thinking`` flag for this turn. When
                a session already exists with a different flag baked
                in, it's closed and rebuilt — the SDK's
                :meth:`Session.create` bakes ``thinking`` once and
                never re-evaluates. A prior session ID (whether from
                ``resume=`` or from a previous turn) is preserved
                across the rebuild: the new SDK session resumes by
                that ID so multi-turn state survives the toggle.
        """
        if self._sdk_session is not None and self._sdk_thinking_baked == thinking:
            return self._sdk_session

        if self._sdk_session is not None:
            # Thinking toggled between turns. Close the existing SDK
            # session and re-resume by its ID so the conversation
            # carries over.
            prior_id = self.id
            try:
                await self._sdk_session.close()
            except Exception:  # noqa: BLE001 — close never raises
                logger.debug("kimi: prior SDK session close raised", exc_info=True)
            self._sdk_session = None
            if prior_id is not None:
                self._resume_id = prior_id

        # Late imports — the ``[kimi]`` extra installs these. The
        # ImportError surfaces clearly when the extra isn't present.
        from kaos.path import KaosPath
        from kimi_agent_sdk import Session

        # Resolve work_dir: KimiOptions.working_directory → KaosPath.cwd().
        work_dir_str = self._provider_options.working_directory if self._provider_options else None
        work_dir = KaosPath(work_dir_str) if work_dir_str else KaosPath.cwd()

        # Resolve model id from the binding override or the runtime default.
        model_id = (
            self._model.model_id if self._model is not None else self._runtime._default_model
        )

        # Auth: the SDK reads KIMI_API_KEY / KIMI_BASE_URL from env.
        # Mutate os.environ to inject explicit constructor args; restore
        # at session close so we don't leak across runtimes. Iteration C+
        # may switch to building a ``Config`` object explicitly once we
        # have a clearer picture of the Config surface — env mutation is
        # the pragmatic Iteration B move.
        if self._runtime._api_key_override:
            self._env_overrides["KIMI_API_KEY"] = os.environ.get("KIMI_API_KEY")
            os.environ["KIMI_API_KEY"] = self._runtime._api_key_override
        if self._runtime._base_url and self._runtime._base_url != DEFAULT_KIMI_BASE_URL:
            self._env_overrides["KIMI_BASE_URL"] = os.environ.get("KIMI_BASE_URL")
            os.environ["KIMI_BASE_URL"] = self._runtime._base_url

        try:
            if self._resume_id is not None:
                sdk = await Session.resume(
                    work_dir=work_dir,
                    session_id=self._resume_id,
                    model=model_id,
                    yolo=True,
                    thinking=thinking,
                )
                if sdk is None:
                    raise RuntimeProtocolError(
                        f"{self._runtime.label}: session "
                        f"{self._resume_id!r} not found under {work_dir}. "
                        "Verify the session ID and the work_dir match a "
                        "prior `Session.create` / `Session.resume` call."
                    )
            else:
                sdk = await Session.create(
                    work_dir=work_dir,
                    model=model_id,
                    yolo=True,
                    thinking=thinking,
                )
        except RuntimeProtocolError:
            raise
        except Exception as exc:
            self._restore_env()
            self._classify_sdk_exception(exc)

        self._sdk_session = sdk
        self._sdk_thinking_baked = thinking
        self.id = sdk.id
        return sdk

    def _restore_env(self) -> None:
        """Undo any environment mutations made by ``_ensure_sdk_session``."""
        for key, prior in self._env_overrides.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior
        self._env_overrides = {}

    # --- AgentSession interface --------------------------------------------

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
        text, images, thinking_bool = self._gate_and_coerce_prompt(
            prompt,
            schema=schema,
            thinking=thinking,
            max_turns=max_turns,
            max_budget_usd=max_budget_usd,
        )
        del timeout  # Iteration B doesn't wire a per-call deadline; the
        # SDK's internal step caps are the de-facto upper bound.

        text_buffer: list[str] = []
        reasoning_buffer: list[str] = []
        last_usage: Any = None
        sdk = await self._ensure_sdk_session(thinking=thinking_bool)
        import kimi_agent_sdk as kimi_sdk  # late import — see module docstring

        user_input = _build_kimi_user_input(text, images, sdk_module=kimi_sdk)

        self._in_flight = True
        try:
            async for wire in self._iter_wire_messages(sdk, user_input):
                kind = self._classify_wire_message(wire)
                if kind == "text":
                    text_buffer.append(wire.text)
                elif kind == "reasoning":
                    reasoning_buffer.append(wire.text)
                elif kind == "approval":
                    # Iteration B: yolo=True is set on Session.create, so
                    # the SDK won't surface ApprovalRequest objects to us
                    # — but defensively resolve any that slip through.
                    wire.resolve("approve")
                elif kind == "usage":
                    last_usage = wire
                # Other wire-types (TurnBegin / TurnEnd / StepBegin / etc.)
                # are observed for their side-effect on stream events
                # (when stream() drives the same loop); execute() doesn't
                # need to act on them.
        finally:
            self._in_flight = False

        text = "".join(text_buffer)
        cost = self._build_cost_record(model_id=self._resolved_model_id(), usage=last_usage)
        return RuntimeResult(
            text=text,
            structured=None,
            cost=cost,
            finish="stop",
            raw=None,
        )

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
        text, images, thinking_bool = self._gate_and_coerce_prompt(
            prompt,
            schema=schema,
            thinking=thinking,
            max_turns=max_turns,
            max_budget_usd=max_budget_usd,
        )
        del timeout

        text_buffer: list[str] = []
        last_usage: Any = None
        sdk = await self._ensure_sdk_session(thinking=thinking_bool)
        import kimi_agent_sdk as kimi_sdk  # late import — see module docstring

        user_input = _build_kimi_user_input(text, images, sdk_module=kimi_sdk)

        self._in_flight = True
        try:
            async for wire in self._iter_wire_messages(sdk, user_input):
                kind = self._classify_wire_message(wire)
                if kind == "text":
                    text_buffer.append(wire.text)
                    yield TextDelta(text=wire.text)
                elif kind == "reasoning":
                    yield ReasoningDelta(text=wire.text)
                elif kind == "approval":
                    wire.resolve("approve")
                elif kind == "usage":
                    last_usage = wire
        finally:
            self._in_flight = False

        text = "".join(text_buffer)
        cost = self._build_cost_record(model_id=self._resolved_model_id(), usage=last_usage)
        yield TurnComplete(
            result=RuntimeResult(text=text, structured=None, cost=cost, finish="stop", raw=None)
        )

    async def cancel(self) -> None:
        if not self._in_flight:
            # No-op when idle — matches the conformance contract
            # ``test_session_cancel_when_idle_is_noop``.
            return
        if self._sdk_session is not None:
            # ``Session.cancel()`` sets the underlying cancel event; the
            # running ``session.prompt()`` raises ``RunCancelled`` which
            # we classify as :class:`RuntimeCancelledError`.
            self._sdk_session.cancel()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._sdk_session is not None:
            try:
                await self._sdk_session.close()
            except Exception:  # noqa: BLE001 — close never raises
                logger.debug("kimi: SDK session close raised", exc_info=True)
            self._sdk_session = None
        self._restore_env()

    def unwrap(self, cls: type[Any]) -> Any:
        if isinstance(self, cls):
            return self
        # Expose the underlying kimi_agent_sdk.Session for callers that
        # want vendor-specific access (e.g. status snapshot, model name).
        if self._sdk_session is not None and isinstance(self._sdk_session, cls):
            return self._sdk_session
        raise TypeError(
            f"{type(self).__name__} cannot unwrap to {cls!r}. Use "
            f"``runtime.unwrap(KimiRuntime)`` for runtime-level access, or "
            f"``session.unwrap(kimi_agent_sdk.Session)`` once the session "
            f"has materialised (after the first execute/stream call)."
        )

    # --- Internals ----------------------------------------------------------

    def _gate_and_coerce_prompt(
        self,
        prompt: Prompt,
        *,
        schema: type[BaseModel] | None,
        thinking: ThinkingMode,
        max_turns: int | None,
        max_budget_usd: float | None,
    ) -> tuple[str, list[ImageInput], bool]:
        """Run all per-call gates; return ``(text, images, thinking_bool)``.

        Capability gates raise :class:`UnsupportedFeatureError` (not
        :class:`NotImplementedError`) so the conformance contracts that
        distinguish "declined" from "not-yet-wired" stay happy.

        Returns:
            ``(text, images, thinking_bool)``. ``text`` already has the
            session's ``system`` prefix prepended; ``images`` is the
            list of attached :class:`ImageInput` parts to forward to
            :func:`_build_kimi_user_input`; ``thinking_bool`` is the
            value to pass to :meth:`Session.create` /
            :meth:`Session.resume`.
        """
        if schema is not None:
            raise NotImplementedError(
                f"{self._runtime.label}: execute(schema=...) is not yet "
                f"wired — Iteration D of the Kimi adapter plan adds "
                f"structured output via an in-process MCP forced-tool, "
                f"mirroring CopilotRuntime's pattern. Until then, request "
                f"JSON via prompt-engineering in your application and parse "
                f"the response yourself, OR set runtime.supports("
                f"Feature.STRUCTURED_OUTPUT_JSON_SCHEMA) expectations "
                f"accordingly."
            )
        # Polymorphic prompts gate against VISION_INPUT / FILE_INPUT
        # via the shared helper. Plain ``str`` prompts pass through and
        # come back as ``(prompt, [], [])``. Iteration C: vision is on,
        # files remain declined (no SDK channel for prompt-side files —
        # use the session's work_dir + tool reads instead).
        text, images, _files = _split_prompt_parts(
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
        if self._system:
            # The SDK's ``Session.create(config=...)`` is the canonical
            # system-prompt slot. Iteration C still prepends to the
            # prompt text as a lightweight stand-in (the Config shape
            # is opaque without installing kimi-cli, which conflicts
            # with claude-agent-sdk); a future iteration may route
            # through Config properly.
            text = f"{self._system}\n\n{text}"
        thinking_bool = _translate_thinking_for_kimi(thinking)
        return text, images, thinking_bool

    async def _iter_wire_messages(self, sdk: Any, user_input: Any) -> AsyncIterator[Any]:
        """Yield :class:`WireMessage` instances from the SDK, classifying errors."""
        try:
            async for wire in sdk.prompt(user_input):
                yield wire
        except Exception as exc:
            self._classify_sdk_exception(exc)

    def _classify_wire_message(self, wire: Any) -> str:
        """Return a coarse category for ``wire`` — text/reasoning/approval/usage/other.

        The match-by-type-name approach avoids importing every Wire
        type at module load (kimi_agent_sdk pulls in fastmcp /
        kimi-cli / kaos / kosong transitively, all of which are
        co-installation hazards). Tests substitute lightweight fake
        types whose ``__name__`` matches.
        """
        name = type(wire).__name__
        if name == "TextPart":
            return "text"
        if name == "ThinkPart":
            return "reasoning"
        if name == "ApprovalRequest":
            return "approval"
        if name == "TokenUsage":
            return "usage"
        return "other"

    def _classify_sdk_exception(self, exc: BaseException) -> None:
        """Translate kimi-agent-sdk exceptions to airframe's ``Runtime*Error``.

        Match on ``type(exc).__name__`` (rather than ``isinstance``)
        for the same reason :meth:`_classify_wire_message` does — keeps
        the test surface free of transitive-dep entanglement.
        """
        name = type(exc).__name__
        msg = f"{self._runtime.label}: {name}: {exc}"
        if name == "RunCancelled":
            raise RuntimeCancelledError(msg) from exc
        if name in {"APIConnectionError", "APITimeoutError"}:
            raise RuntimeTransientError(msg) from exc
        if name == "APIStatusError":
            status = getattr(exc, "status_code", None)
            if status in (401, 403):
                raise RuntimeAuthError(msg) from exc
            if status in (429, 502, 503, 504):
                raise RuntimeTransientError(msg) from exc
            raise RuntimeProtocolError(msg) from exc
        if name == "APIEmptyResponseError":
            raise RuntimeProtocolError(msg) from exc
        if name in {"LLMNotSet", "LLMNotSupported"}:
            raise RuntimeAuthError(msg) from exc
        if name in {
            "ConfigError",
            "AgentSpecError",
            "InvalidToolError",
            "MCPConfigError",
            "MCPRuntimeError",
            "SystemPromptTemplateError",
            "PromptValidationError",
            "MaxStepsReached",
        }:
            raise RuntimeProtocolError(msg) from exc
        if name == "SessionStateError":
            raise RuntimeError(msg) from exc
        # Unknown — surface as protocol error rather than swallowing.
        raise RuntimeProtocolError(msg) from exc

    def _resolved_model_id(self) -> str:
        return self._model.model_id if self._model is not None else self._runtime._default_model

    def _build_cost_record(self, *, model_id: str, usage: Any) -> CostRecord:
        """Build a :class:`CostRecord` from a ``TokenUsage`` wire message.

        Iteration B leaves USD cost as ``None`` — the pricing table
        lands in Iteration E alongside ``BUDGET_USD_CAP``. Token
        counts populate from ``usage.input_tokens`` /
        ``usage.output_tokens`` when present; otherwise zero.
        """
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0) if usage is not None else 0
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage is not None else 0
        cache_read = int(getattr(usage, "cache_read_tokens", 0) or 0) if usage is not None else 0
        cache_write = int(getattr(usage, "cache_write_tokens", 0) or 0) if usage is not None else 0
        return CostRecord(
            provider_id=self._runtime.PROVIDER_ID,
            model_id=model_id,
            cost_usd=None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            finish="stop",
        )


__all__ = [
    "DEFAULT_KIMI_BASE_URL",
    "DEFAULT_KIMI_MODEL",
    "KimiRuntime",
    "KimiSession",
]
