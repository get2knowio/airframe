"""``CopilotRuntime`` — :class:`AgentRuntime` over the GitHub Copilot SDK.

Wraps :class:`copilot.CopilotClient` to expose Maverick's agent layer
to OpenAI / GPT-family / xAI models routed through the user's GitHub
Copilot subscription. The ``github-copilot-sdk`` package spawns and
manages the ``copilot`` CLI subprocess; Maverick doesn't allocate
ports, juggle passwords, validate model IDs at startup, or maintain
any client code.

**Auth.** Three options, checked in order:

1. Explicit ``github_token=`` constructor argument.
2. ``GITHUB_TOKEN`` env var (or ``GH_TOKEN``).
3. ``use_logged_in_user=True`` — the SDK picks up the OAuth credentials
   stored by ``gh auth login``. The interactive path for developer
   machines.

**Claude is intentionally not routed here.** Phase 0 of the migration
spike found that Claude models served via Copilot Chat Completions
emit markdown-fenced JSON instead of honouring tool calls — the
structured-output forcing pattern does not work. :meth:`validate_binding`
rejects any ``model_id`` that starts with ``claude-``; those route
through :class:`ClaudeCodeRuntime` (subscription / OAuth) or the
``AnthropicRuntime`` (API key) instead.

**Structured output.** Implemented via a hidden ``submit_result``
tool registered with the agent's schema via :func:`copilot.define_tool`.
The model is forced (via system-message append) to call
``submit_result`` exactly once with a typed payload; the runtime
captures the validated Pydantic instance and returns its dict form
as :attr:`RuntimeResult.structured`.

**Lifecycle.** ``execute()`` lazily constructs a
:class:`CopilotClient` and a :class:`CopilotSession` keyed by
``(schema, system, model)`` — any change to that triple forces a
session recreation because the tool list, model, and system message
are baked into ``create_session()`` at session-creation time.
Subsequent ``execute()`` calls on the same triple reuse the session.
``reset()`` destroys the session; the next ``execute()`` creates a
fresh one. ``close()`` destroys the session and disconnects the
underlying client.

**Cost.** The SDK emits one ``AssistantUsageData`` event per model
turn on the session event stream. We subscribe via
:meth:`CopilotSession.on` and accumulate the final event's
``cost`` / token fields into the :class:`CostRecord`.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from pydantic import BaseModel, ValidationError

from airframe.cost import CostRecord
from airframe.errors import (
    AgentRuntimeError,
    RuntimeAuthError,
    RuntimeCancelledError,
    RuntimeModelNotFoundError,
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

#: Default Copilot model when no binding is specified. GPT-5 mini is
#: the v0 default because it's the cheapest stable tier on the
#: subscription. Selected per-call via ``ProviderModel.model_id``.
DEFAULT_COPILOT_MODEL = "gpt-5-mini"

#: Canonical name for the hidden structured-output tool.
SUBMIT_RESULT_TOOL = "submit_result"

#: Canonical provider ID this adapter serves.
PROVIDER_ID = "github-copilot"


class CopilotRuntime(AgentRuntime):
    """One Copilot SDK client per runtime instance.

    Args:
        model: Default Copilot model identifier used when ``execute()``
            is called without a ``ProviderModel`` override. Honours
            ``COPILOT_MODEL_OVERRIDE`` env var if set for testing.
        github_token: Optional explicit GitHub token. When ``None``
            (default), auth resolves via ``GITHUB_TOKEN`` / ``GH_TOKEN``
            env vars, then falls back to ``use_logged_in_user=True``
            so the SDK reads ``gh auth`` credentials.
        cli_path: Optional override for the ``copilot`` CLI path.
    """

    label = "copilot"

    #: Canonical provider ID for this adapter.
    PROVIDER_ID: ClassVar[str] = PROVIDER_ID

    #: Vendor SDK that must be importable for this adapter to work.
    REQUIRES_PACKAGE: ClassVar[str] = "copilot"

    #: pip extra that brings the vendor SDK in.
    EXTRA_NAME: ClassVar[str] = "copilot"

    #: Features this runtime exposes today.
    #:
    #: * ``STRUCTURED_OUTPUT_JSON_SCHEMA`` — wired via the forced
    #:   ``submit_result`` tool pattern (Phase 0).
    #: * ``STREAMING`` — wired via :class:`CopilotAgentSession` using
    #:   ``session.on(handler)`` filtering for
    #:   ``ASSISTANT_MESSAGE_DELTA`` / ``ASSISTANT_REASONING_DELTA``
    #:   (Phase 1, Iteration E).
    #: * ``SESSION_RESUME`` — wired via
    #:   :meth:`CopilotClient.resume_session`;
    #:   :meth:`AgentRuntime.session` accepts ``resume=<session_id>``
    #:   (Phase 1, Iteration E). The session ID surfaces on
    #:   :attr:`AgentSession.id` once the underlying CopilotSession is
    #:   built.
    #: * ``CANCEL`` — wired via :meth:`CopilotSession.abort` (Phase 1,
    #:   Iteration E).
    #: * ``REASONING_EFFORT`` — wired via the ``reasoning_effort``
    #:   kwarg on ``create_session`` / ``resume_session``. Baked at
    #:   session-create time, so a ``thinking=`` change between turns
    #:   triggers a rebuild (same pattern as schema). Copilot's enum is
    #:   ``"low" | "medium" | "high" | "xhigh"``; airframe stays on
    #:   the portable intersection. ``"minimal"`` is coerced to
    #:   ``"low"`` with a debug-level log (Phase 2, Iteration B).
    #:
    #: * ``VISION_INPUT`` / ``FILE_INPUT`` — wired via Copilot's
    #:   attachment slot on :meth:`CopilotSession.send_and_wait`.
    #:   ``ImageInput(path=)`` and ``FileInput(path=)`` use
    #:   :class:`FileAttachment` (``{"type":"file","path":...}``);
    #:   ``ImageInput(bytes_=)`` uses :class:`BlobAttachment`
    #:   (``{"type":"blob","data":<b64>,"mimeType":...}``).
    #:   ``ImageInput(url=)`` raises — Copilot's SDK has no URL channel
    #:   (Phase 2, Iterations C + D).
    #:
    #: ``REASONING_BUDGET_TOKENS`` stays False — Copilot uses the
    #: enum, not a token budget. Pass a literal effort string instead.
    #:
    #: * ``TOOLS_FUNCTION`` — wired via :func:`copilot.define_tool`
    #:   registrations passed through :meth:`CopilotClient.create_session`
    #:   ``tools=`` slot. The SDK dispatches; airframe surfaces
    #:   :class:`~airframe.events.ToolCallStart` /
    #:   :class:`~airframe.events.ToolCallResult` from
    #:   :class:`ToolExecutionStartData` /
    #:   :class:`ToolExecutionCompleteData` session events.
    #:   When ``schema=`` is also set, the adapter prepends the existing
    #:   forced ``submit_result`` tool so structured-output coexists
    #:   (Phase 3, Iteration C).
    SUPPORTED_FEATURES: ClassVar[frozenset[Feature]] = frozenset(
        {
            Feature.STRUCTURED_OUTPUT_JSON_SCHEMA,
            Feature.STREAMING,
            Feature.SESSION_RESUME,
            Feature.CANCEL,
            Feature.REASONING_EFFORT,
            Feature.VISION_INPUT,
            Feature.FILE_INPUT,
            Feature.TOOLS_FUNCTION,
        }
    )

    def __init__(
        self,
        *,
        model: str | None = None,
        github_token: str | None = None,
        cli_path: str | None = None,
    ) -> None:
        self._default_model = (
            model or os.environ.get("COPILOT_MODEL_OVERRIDE") or DEFAULT_COPILOT_MODEL
        )
        self._github_token = (
            github_token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        )
        self._cli_path = cli_path or os.environ.get("COPILOT_CLI_PATH")

        self._client: Any | None = None  # copilot.CopilotClient
        # Phase 1 Iteration G: per-conversation state (CopilotSession,
        # captured payload/usage/error/message) moved off the runtime
        # onto CopilotAgentSession. The runtime keeps only the
        # runtime-wide CopilotClient.

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
        # ephemeral — the CopilotSession is destroyed per call. The
        # CopilotClient (runtime-owned, long-lived) is reused across
        # calls. Consumers wanting session warmth across turns open a
        # session explicitly.
        del persona  # accepted in the protocol but not consumed by Copilot
        sess = self.session(system=system, model=model)
        try:
            return await sess.execute(prompt, schema=schema, thinking=thinking, timeout=timeout)
        finally:
            await sess.close()

    async def reset(self) -> None:
        # Phase 1 Iteration G: the runtime no longer caches a CopilotSession.
        # The CopilotClient (long-lived CLI handle) survives reset() — only
        # close() releases it. With no scope-bound state to drop, reset is
        # a no-op; kept for protocol completeness.
        return None

    async def close(self) -> None:
        # Phase 1 Iteration G: only the runtime-wide CopilotClient needs
        # tearing down here. The per-conversation CopilotSession lives
        # on AgentSession instances and is closed by session.close().
        client = self._client
        self._client = None
        if client is None:
            return
        try:
            await client.stop()
        except Exception as exc:  # noqa: BLE001 — teardown never raises
            logger.debug("copilot_runtime.close_failed error=%s", exc)

    def validate_binding(self, binding: ProviderModel) -> bool:
        if binding.provider_id != self.PROVIDER_ID:
            return False
        # Phase 0 spike finding: Claude served via Copilot Chat Completions
        # emits markdown-fenced JSON instead of calling tools. Route Claude
        # bindings through ClaudeCodeRuntime / AnthropicRuntime instead.
        return not binding.model_id.startswith("claude-")

    def supports(self, feature: Feature, model: ProviderModel | None = None) -> bool:
        return feature in self.SUPPORTED_FEATURES

    def unwrap(self, cls: type[T]) -> T:
        from copilot import CopilotClient
        from copilot.session import CopilotSession

        if isinstance(self, cls):
            return self  # type: ignore[return-value]
        if cls is CopilotClient:
            if self._client is None:
                raise TypeError(
                    "CopilotRuntime.unwrap(CopilotClient): no client exists yet "
                    "— call execute() first."
                )
            return self._client  # type: ignore[return-value]
        if cls is CopilotSession:
            # Phase 1 Iteration G moved the per-conversation
            # CopilotSession off the runtime onto CopilotAgentSession.
            raise TypeError(
                "CopilotRuntime no longer owns a CopilotSession — sessions do. "
                "Open a session with `sess = runtime.session(...)`, run a turn, "
                "then call `sess.unwrap(CopilotSession)`."
            )
        raise TypeError(
            f"CopilotRuntime cannot unwrap to {cls!r}; supported types are "
            f"CopilotRuntime and copilot.CopilotClient. Vendor session objects "
            f"live on AgentSession — use session.unwrap(NativeType)."
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
        """Open a bespoke :class:`CopilotAgentSession`.

        Phase 1 Iteration E replaces the
        :class:`~airframe.sessions._ThinAgentSession` placeholder with
        a session that owns its own :class:`CopilotSession` lifecycle:
        streaming via per-session
        :meth:`CopilotSession.on` subscriptions for
        ``ASSISTANT_MESSAGE_DELTA`` / ``ASSISTANT_REASONING_DELTA``,
        native session resume via
        :meth:`CopilotClient.resume_session`, and cancellation via
        :meth:`CopilotSession.abort`.

        Args:
            resume: Vendor-assigned session ID to resume — Copilot
                surfaces this on :attr:`CopilotSession.session_id`.
                ``None`` opens a fresh session.
            system: System message appended via the
                :attr:`SystemMessageConfig` shape at session create /
                resume.
            model: Default :class:`ProviderModel` for every turn.
                Claude bindings are rejected here too — same reason as
                ``execute()``: Copilot routes Claude as markdown JSON,
                not via tool calls.
            tools: List of :class:`~airframe.tools.FunctionTool` the
                model may invoke. Translated to
                :func:`copilot.define_tool` registrations and passed
                via :meth:`CopilotClient.create_session(tools=...)`.
                When ``schema=`` is also set, the forced
                ``submit_result`` tool is prepended to the user's
                list so structured-output coexists with custom tools.
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
        return CopilotAgentSession(
            self,
            resume=resume,
            system=system,
            model_id=model_id,
            tools=tools,
        )

    async def list_models(self) -> list[ModelInfo]:
        """Return the live model menu from Copilot's CLI.

        Copilot's SDK exposes :meth:`CopilotClient.list_models` natively
        with rich metadata: display name, context window, vision /
        reasoning-effort support, and a billing multiplier. We surface
        everything as :class:`ModelInfo` and skip the metadata-table
        fallback (Copilot tells us everything we need).
        """
        client = await self._ensure_client()
        try:
            sdk_models = await client.list_models()
        except Exception as exc:
            raise self._classify_exception(exc) from exc

        from airframe.models import (
            CAPABILITY_REASONING_EFFORT,
            CAPABILITY_STREAMING,
            CAPABILITY_STRUCTURED_OUTPUT,
            CAPABILITY_TOOLS,
            CAPABILITY_VISION,
        )

        out: list[ModelInfo] = []
        for m in sdk_models:
            caps: set[str] = set()
            if m.capabilities.supports.vision:
                caps.add(CAPABILITY_VISION)
            if m.capabilities.supports.reasoning_effort:
                caps.add(CAPABILITY_REASONING_EFFORT)
            # Copilot CLI runtimes all support tools and structured output
            # via the SDK's define_tool / forced-tool pattern; streaming is
            # also universal. Declare these statically.
            caps.update({CAPABILITY_TOOLS, CAPABILITY_STRUCTURED_OUTPUT, CAPABILITY_STREAMING})
            out.append(
                ModelInfo(
                    id=m.id,
                    display_name=m.name,
                    provider_id=self.PROVIDER_ID,
                    context_window=m.capabilities.limits.max_context_window_tokens,
                    pricing_input_per_1k_usd=None,  # Copilot is subscription-priced
                    pricing_output_per_1k_usd=None,
                    capabilities=frozenset(caps),
                    raw=m,
                )
            )
        return out

    # --- Internals ---------------------------------------------------------

    def _resolve_model(self, model: ProviderModel | None) -> str:
        if model is None:
            return self._default_model
        if not self.validate_binding(model):
            raise UnsupportedBindingError(
                f"CopilotRuntime cannot serve {model.label!r}; "
                f"provider must be {self.PROVIDER_ID!r} "
                f"and the model_id must not start with 'claude-'"
            )
        return model.model_id

    async def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client

        from copilot import CopilotClient, SubprocessConfig

        config_kwargs: dict[str, Any] = {}
        if self._cli_path is not None:
            config_kwargs["cli_path"] = self._cli_path
        if self._github_token is not None:
            config_kwargs["github_token"] = self._github_token
        else:
            # Fall back to the user's gh CLI credentials.
            config_kwargs["use_logged_in_user"] = True

        try:
            client = CopilotClient(SubprocessConfig(**config_kwargs))
        except Exception as exc:
            raise self._classify_exception(exc) from exc
        self._client = client
        return client

    def _cost_from_usage(self, usage: Any, *, model_id: str) -> CostRecord:
        if usage is None:
            return CostRecord(
                provider_id=self.PROVIDER_ID,
                model_id=model_id,
                cost_usd=None,
                input_tokens=0,
                output_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
                finish="stop",
            )
        # Phase 2 Iteration B: Copilot's AssistantUsageData surfaces
        # reasoning tokens directly when a reasoning-effort model is
        # in play. Falls through to 0 for non-reasoning turns.
        reasoning_tokens = int(getattr(usage, "reasoning_tokens", 0) or 0)
        return CostRecord(
            provider_id=self.PROVIDER_ID,
            model_id=model_id,
            cost_usd=float(usage.cost) if usage.cost is not None else None,
            input_tokens=int(usage.input_tokens or 0),
            output_tokens=int(usage.output_tokens or 0),
            cache_read_tokens=int(usage.cache_read_tokens or 0),
            cache_write_tokens=int(usage.cache_write_tokens or 0),
            finish="stop",
            reasoning_tokens=reasoning_tokens,
        )

    def _error_from_session_error(self, error_data: Any) -> Exception:
        status = getattr(error_data, "status_code", None)
        message = getattr(error_data, "message", "") or ""
        error_type = getattr(error_data, "error_type", "") or ""
        body = {
            "error_type": error_type,
            "status_code": status,
            "message": message,
        }
        if status in (401, 403):
            return RuntimeAuthError(f"copilot: auth: {message}")
        if status == 404 or ("model" in error_type.lower() and "not" in error_type.lower()):
            return RuntimeModelNotFoundError(f"copilot: model not found: {message}")
        if status in (429, 502, 503, 504):
            return RuntimeTransientError(f"copilot: transient {status}: {message}")
        if status is not None and 500 <= status < 600:
            return RuntimeTransientError(f"copilot: 5xx: {message}")
        return RuntimeProtocolError(f"copilot: session error: {message}", body=body)

    def _classify_exception(self, exc: BaseException) -> Exception:
        """Map Copilot SDK exceptions onto Maverick's runtime hierarchy."""
        if isinstance(exc, UnsupportedBindingError):
            return exc
        if isinstance(exc, ValidationError):
            # The submit_result handler got a payload that didn't validate
            # against our schema — that's a structured-output failure, not
            # a transient.
            return RuntimeStructuredOutputError(
                f"copilot: payload failed schema validation: {exc}",
                body=str(exc)[:1000],
            )
        if isinstance(exc, FileNotFoundError):
            return RuntimeServerStartError(f"copilot: CLI not found: {exc}")

        msg = str(exc).lower()
        if "auth" in msg or "401" in msg or "403" in msg or "unauthorized" in msg:
            return RuntimeAuthError(f"copilot: auth: {exc}")
        if "not found" in msg and "model" in msg:
            return RuntimeModelNotFoundError(f"copilot: model not found: {exc}")
        if "rate" in msg or "429" in msg or "503" in msg or "timeout" in msg:
            return RuntimeTransientError(f"copilot: transient: {exc}")
        return AgentRuntimeError(f"copilot: unexpected {type(exc).__name__}: {exc}")


class CopilotAgentSession:
    """Bespoke :class:`~airframe.protocol.AgentSession` for the Copilot SDK.

    Phase 1 Iteration E — third per-vendor session (after OpenAI-compat
    and Claude Code). Owns one :class:`CopilotSession` for its lifetime;
    ``system`` / ``model`` / ``resume`` are session-fixed and baked into
    :meth:`CopilotClient.create_session` (or :meth:`resume_session`) at
    creation time. Schema can vary per :meth:`execute` / :meth:`stream`
    call — the session is destroyed and re-created when the schema
    fingerprint changes since the ``submit_result`` tool is baked into
    ``create_session()`` at session-creation time.

    **Streaming.** :meth:`stream` subscribes a delta-collecting handler
    via :meth:`CopilotSession.on`, runs
    :meth:`CopilotSession.send_and_wait` as a background task, and
    drains an :class:`asyncio.Queue` yielding airframe events:

    * ``AssistantMessageDeltaData`` → :class:`TextDelta`
    * ``AssistantReasoningDeltaData`` → :class:`ReasoningDelta`

    The trailing :class:`~airframe.events.TurnComplete` carries the
    canonical :class:`~airframe.cost.CostRecord` built from the
    ``AssistantUsageData`` event captured during the turn.

    **Cancellation.** :meth:`cancel` calls
    :meth:`CopilotSession.abort` to abort the CLI side of the turn; the
    awaiting :meth:`send_and_wait` raises, surfaced as
    :class:`~airframe.errors.RuntimeCancelledError`.

    **Resume.** ``session(resume=<session_id>)`` forwards the ID into
    :meth:`CopilotClient.resume_session`; :attr:`id` is seeded with the
    resume ID and updated to the live :attr:`CopilotSession.session_id`
    after the underlying session is built.
    """

    def __init__(
        self,
        runtime: CopilotRuntime,
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
        # Session-owned vendor session + the schema-fingerprint key it
        # was built with. Schema-fingerprint changes force a destroy +
        # rebuild because the submit_result tool is baked in at
        # create_session() time.
        self._session: Any | None = None  # CopilotSession
        self._session_key: str | None = None
        self._unsubscribe_capture: Any | None = None  # Callable[[], None]
        self._closed = False
        self._in_flight = False
        # Per-turn capture slots, refreshed on each execute() / stream().
        self._captured_payload: BaseModel | None = None
        self._captured_usage: Any | None = None
        self._captured_error: Any | None = None
        self._last_assistant_message: Any | None = None
        # Seeded from resume= so consumer code that branches on
        # session.id before the first turn sees the right value.
        self.id: str | None = resume
        # Phase 3 Iteration C: tools are session-fixed and baked at
        # create_session() time. The fingerprint joins the cache key
        # so a tools-change forces a session rebuild (same pattern
        # schema and reasoning-effort follow).
        self._tools: list[FunctionTool] = list(tools or [])
        self._tools_fingerprint = _copilot_tools_fingerprint(self._tools)

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
        attachments = _build_copilot_attachments(images, files)
        session = await self._ensure_session(schema=schema, thinking=thinking)
        self._reset_capture_slots()
        self._in_flight = True
        try:
            await asyncio.wait_for(
                session.send_and_wait(text, attachments=attachments, timeout=timeout),
                timeout=timeout,
            )
        except asyncio.CancelledError as exc:
            raise RuntimeCancelledError(f"{self._runtime.label}: cancelled") from exc
        except TimeoutError as exc:
            raise RuntimeTransientError(
                f"{self._runtime.label}: execute timed out after {timeout}s"
            ) from exc
        except Exception as exc:
            raise self._runtime._classify_exception(exc) from exc
        finally:
            self._in_flight = False

        return self._build_result(schema=schema)

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
        attachments = _build_copilot_attachments(images, files)
        session = await self._ensure_session(schema=schema, thinking=thinking)
        self._reset_capture_slots()

        # Per-stream queue + delta handler. We subscribe a lightweight
        # callback that pushes deltas into the queue; the generator
        # below drains until the send_and_wait task completes.
        queue: asyncio.Queue[RuntimeEvent] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        from copilot.generated.session_events import (
            AssistantMessageDeltaData,
            AssistantReasoningDeltaData,
            ToolExecutionCompleteData,
            ToolExecutionStartData,
        )

        # The forced ``submit_result`` tool is structured-output
        # plumbing, not a user-visible tool call. Filter it out of the
        # streaming events so consumers don't have to special-case it.
        suppress_tool_call_ids: set[str] = set()

        def _on_delta(event: Any) -> None:
            # Runs synchronously off the SDK dispatch thread — keep it
            # cheap. Use call_soon_threadsafe to enqueue from the SDK's
            # thread back onto our event loop.
            data = getattr(event, "data", None)
            if isinstance(data, AssistantMessageDeltaData):
                text = data.delta_content or ""
                if text:
                    loop.call_soon_threadsafe(queue.put_nowait, TextDelta(text=text))
            elif isinstance(data, AssistantReasoningDeltaData):
                text = data.delta_content or ""
                if text:
                    loop.call_soon_threadsafe(queue.put_nowait, ReasoningDelta(text=text))
            elif isinstance(data, ToolExecutionStartData):
                if data.tool_name == SUBMIT_RESULT_TOOL:
                    suppress_tool_call_ids.add(data.tool_call_id)
                    return
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    ToolCallStart(
                        tool_name=data.tool_name,
                        tool_call_id=data.tool_call_id,
                        arguments_preview=_serialize_copilot_tool_arguments(data.arguments),
                    ),
                )
            elif isinstance(data, ToolExecutionCompleteData):
                if data.tool_call_id in suppress_tool_call_ids:
                    suppress_tool_call_ids.discard(data.tool_call_id)
                    return
                output = _copilot_tool_result_output(data)
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    ToolCallResult(
                        tool_call_id=data.tool_call_id,
                        output=output,
                        is_error=not data.success,
                    ),
                )

        unsubscribe = session.on(_on_delta)
        self._in_flight = True
        send_task = asyncio.create_task(
            asyncio.wait_for(
                session.send_and_wait(text, attachments=attachments, timeout=timeout),
                timeout=timeout,
            )
        )

        try:
            while True:
                # Wait for either the next delta or the send task to finish.
                getter = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait(
                    {getter, send_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if getter in done:
                    yield getter.result()
                else:
                    # send_task finished — cancel the queue getter and
                    # drain any final events the SDK may have enqueued
                    # before send_and_wait returned.
                    getter.cancel()
                    while not queue.empty():
                        yield queue.get_nowait()
                    break
            try:
                await send_task
            except asyncio.CancelledError as exc:
                raise RuntimeCancelledError(f"{self._runtime.label}: cancelled") from exc
            except TimeoutError as exc:
                raise RuntimeTransientError(
                    f"{self._runtime.label}: stream timed out after {timeout}s"
                ) from exc
            except Exception as exc:
                raise self._runtime._classify_exception(exc) from exc
        finally:
            unsubscribe()
            self._in_flight = False

        result = self._build_result(schema=schema)
        yield TurnComplete(result=result)

    async def cancel(self) -> None:
        # No-op when no turn is in flight — per the AgentSession contract.
        if not self._in_flight:
            return
        session = self._session
        if session is None:
            return
        try:
            await session.abort()
        except Exception as exc:  # noqa: BLE001 — cancellation never raises
            logger.debug("%s.session_abort_failed error=%s", self._runtime.label, exc)

    async def close(self) -> None:
        self._closed = True
        # Tear down the per-session capture subscription.
        unsubscribe = self._unsubscribe_capture
        self._unsubscribe_capture = None
        if unsubscribe is not None:
            try:
                unsubscribe()
            except Exception as exc:  # noqa: BLE001
                logger.debug("%s.session_unsubscribe_failed error=%s", self._runtime.label, exc)
        # Destroy the vendor session. The runtime owns the CopilotClient
        # and lives independently — don't tear it down here.
        session = self._session
        self._session = None
        self._session_key = None
        if session is None:
            return
        try:
            await session.destroy()
        except Exception as exc:  # noqa: BLE001 — teardown never raises
            logger.debug("%s.session_close_failed error=%s", self._runtime.label, exc)

    def unwrap(self, cls: type[T]) -> T:
        from copilot.session import CopilotSession

        if isinstance(self, cls):
            return self  # type: ignore[return-value]
        if cls is CopilotSession:
            if self._session is None:
                raise TypeError(
                    "CopilotAgentSession.unwrap(CopilotSession): no session "
                    "exists yet — call execute() or stream() first."
                )
            return self._session  # type: ignore[return-value]
        raise TypeError(
            f"CopilotAgentSession cannot unwrap to {cls!r}; supported types are "
            f"CopilotAgentSession and copilot.session.CopilotSession. "
            f"(CopilotClient lives on the runtime — "
            f"call runtime.unwrap(CopilotClient).)"
        )

    # --- Internals ---------------------------------------------------------

    def _reset_capture_slots(self) -> None:
        self._captured_payload = None
        self._captured_usage = None
        self._captured_error = None
        self._last_assistant_message = None

    async def _ensure_session(
        self,
        *,
        schema: type[BaseModel] | None,
        thinking: ThinkingMode = None,
    ) -> Any:
        schema_fragment = (
            f"{schema.__name__}|{schema.model_json_schema()}"
            if schema is not None
            else "__plain_text__"
        )
        reasoning_effort = _translate_thinking_for_copilot(thinking)
        tools_fragment = f"tools={self._tools_fingerprint}"
        cache_key = f"{schema_fragment}|effort={reasoning_effort}|{tools_fragment}"
        if self._session is not None and self._session_key == cache_key:
            return self._session

        # Schema OR thinking OR tools fingerprint changed (or first
        # turn) — tear down any stale session before rebuilding so we
        # don't leak it. ``reasoning_effort`` and ``tools=`` are baked
        # at create_session time, so both join schema in the cache key.
        await self._tear_down_session()

        client = await self._runtime._ensure_client()

        from copilot.session import PermissionHandler

        create_kwargs: dict[str, Any] = {
            "on_permission_request": PermissionHandler.approve_all,
            "model": self._model_id,
        }
        if reasoning_effort is not None:
            create_kwargs["reasoning_effort"] = reasoning_effort

        # Assemble the tools list: forced ``submit_result`` first (when
        # schema= is set) so the model sees the structured-output gate
        # in slot zero, then any user-supplied :class:`FunctionTool`
        # registrations. Either bucket can be empty; ``tools=[]`` /
        # missing kwarg are both fine.
        session_tools: list[Any] = []
        if schema is not None:
            from copilot import define_tool

            captured_schema = schema  # bind for closure stability

            async def _submit_handler(params: schema) -> dict[str, Any]:  # type: ignore[valid-type]
                self._captured_payload = params
                return {"ok": True}

            submit_tool = define_tool(
                SUBMIT_RESULT_TOOL,
                description=(
                    f"Submit the final typed payload as a {captured_schema.__name__}. "
                    "Call this exactly once with all required fields filled in."
                ),
                handler=lambda params, inv: _submit_handler(params),
                params_type=captured_schema,
                skip_permission=True,
            )
            session_tools.append(submit_tool)

            forced_prefix = (
                "When you are ready to answer, call the "
                f"`{SUBMIT_RESULT_TOOL}` tool with the typed payload. "
                "Do not emit a final assistant message; the tool call is your answer.\n\n"
            )
            create_kwargs["system_message"] = {
                "mode": "append",
                "content": forced_prefix + (self._system or ""),
            }
        elif self._system is not None:
            create_kwargs["system_message"] = {"mode": "append", "content": self._system}

        for ft in self._tools:
            session_tools.append(_translate_one_copilot_tool(ft))

        if session_tools:
            create_kwargs["tools"] = session_tools

        try:
            if self._resume is not None:
                session = await client.resume_session(self._resume, **create_kwargs)
            else:
                session = await client.create_session(**create_kwargs)
        except Exception as exc:
            raise self._runtime._classify_exception(exc) from exc

        # Always-on capture subscription for usage / error / final message.
        self._unsubscribe_capture = session.on(self._on_capture_event)
        self._session = session
        self._session_key = cache_key
        # Surface the live session ID. resume= callers may see a
        # different value here if Copilot forked the session.
        live_id = getattr(session, "session_id", None)
        if live_id:
            self.id = live_id
        return session

    async def _tear_down_session(self) -> None:
        unsubscribe = self._unsubscribe_capture
        self._unsubscribe_capture = None
        if unsubscribe is not None:
            try:
                unsubscribe()
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "%s.session_unsubscribe_failed error=%s",
                    self._runtime.label,
                    exc,
                )
        session = self._session
        self._session = None
        self._session_key = None
        if session is None:
            return
        try:
            await session.destroy()
        except Exception as exc:  # noqa: BLE001
            logger.debug("%s.session_teardown_failed error=%s", self._runtime.label, exc)

    def _on_capture_event(self, event: Any) -> None:
        """Capture usage / error / final-assistant-message events.

        Runs synchronously off the SDK dispatch — keep it cheap. Mirrors
        the runtime's ``_on_event`` but scoped to this session.
        """
        from copilot.generated.session_events import (
            AssistantMessageData,
            AssistantUsageData,
            SessionErrorData,
        )

        data = getattr(event, "data", None)
        if isinstance(data, AssistantUsageData):
            self._captured_usage = data
        elif isinstance(data, SessionErrorData):
            self._captured_error = data
        elif isinstance(data, AssistantMessageData):
            self._last_assistant_message = event

    def _build_result(self, *, schema: type[BaseModel] | None) -> RuntimeResult:
        if self._captured_error is not None:
            raise self._runtime._error_from_session_error(self._captured_error)

        text = ""
        if self._last_assistant_message is not None and hasattr(
            self._last_assistant_message, "data"
        ):
            text = getattr(self._last_assistant_message.data, "content", "") or ""

        cost = self._runtime._cost_from_usage(self._captured_usage, model_id=self._model_id)

        if schema is None:
            return RuntimeResult(
                text=text,
                structured=None,
                cost=cost,
                finish="stop",
                raw={
                    "usage": self._captured_usage,
                    "message": self._last_assistant_message,
                },
            )

        captured = self._captured_payload
        if captured is None:
            preview_text = text[:300]
            raise RuntimeStructuredOutputError(
                f"{self._runtime.label}: {SUBMIT_RESULT_TOOL} was never called",
                body={"assistant_message_preview": preview_text},
            )

        return RuntimeResult(
            text=text,
            structured=captured.model_dump(),
            cost=cost,
            finish="stop",
            raw={
                "usage": self._captured_usage,
                "message": self._last_assistant_message,
            },
        )


def _build_copilot_attachments(images: list[Any], files: list[Any]) -> list[dict[str, Any]] | None:
    """Build the ``attachments=`` list for ``send_and_wait``.

    Returns ``None`` (so the SDK keeps its own default behaviour) when
    no parts were attached; otherwise a list of TypedDicts ready to
    pass to the SDK. Per-variant routing:

    * ``ImageInput(path=)`` → :class:`FileAttachment`
      (``{"type":"file","path":str}``).
    * ``ImageInput(bytes_=)`` → :class:`BlobAttachment`
      (``{"type":"blob","data":<b64>,"mimeType":...}``). ``media_type``
      defaults to ``image/png`` when omitted.
    * ``ImageInput(url=)`` → raises :class:`UnsupportedFeatureError`.
      Copilot's SDK has no URL channel; the consumer should fetch the
      image and pass ``bytes_=`` or ``path=``.
    * ``FileInput(path=)`` → :class:`FileAttachment`. Copilot handles
      documents and images uniformly through the attachment slot.
    """
    if not images and not files:
        return None
    import base64

    attachments: list[dict[str, Any]] = []
    for img in images:
        if img.path is not None:
            attachments.append({"type": "file", "path": img.path})
        elif img.bytes_ is not None:
            attachments.append(
                {
                    "type": "blob",
                    "data": base64.b64encode(img.bytes_).decode("ascii"),
                    "mimeType": img.media_type or "image/png",
                }
            )
        else:
            raise UnsupportedFeatureError(
                "copilot: ImageInput(url=...) has no Copilot SDK channel; "
                "fetch the image and pass bytes_= or path= instead.",
                feature=Feature.VISION_INPUT,
            )
    for file in files:
        attachments.append({"type": "file", "path": file.path})
    return attachments


def _translate_thinking_for_copilot(thinking: ThinkingMode) -> str | None:
    """Translate :data:`ThinkingMode` to the ``reasoning_effort`` kwarg.

    Copilot exposes ``Literal["low" | "medium" | "high" | "xhigh"]``
    on :meth:`CopilotClient.create_session` / ``resume_session``.
    Returns ``None`` to mean "don't send the kwarg" (vendor default
    for the chosen model).

    Mappings:

    * ``None`` → ``None`` (vendor default).
    * ``"disabled"`` → ``None`` (Copilot has no explicit-off; omitting
      the kwarg picks the model's non-reasoning default for non-
      reasoning models, which is what the user asked for).
    * ``"low" | "medium" | "high"`` → pass through.
    * ``"minimal"`` → ``"low"`` with a debug log (no Copilot
      equivalent).

    Raises:
        UnsupportedFeatureError: when a dict shape is passed (Claude-
            only ``budget_tokens``).
    """
    if thinking is None or thinking == "disabled":
        return None
    if isinstance(thinking, str):
        if thinking == "minimal":
            logger.debug(
                "copilot: thinking='minimal' has no Copilot equivalent; coercing to 'low'"
            )
            return "low"
        if thinking in ("low", "medium", "high"):
            return thinking
        raise UnsupportedFeatureError(
            f"copilot: unrecognised thinking effort {thinking!r}; "
            f"supported: 'minimal' (→'low'), 'low', 'medium', 'high', 'disabled'.",
            feature="reasoning_effort",
        )
    if isinstance(thinking, dict):
        raise UnsupportedFeatureError(
            "copilot: dict-shaped thinking ({'budget_tokens': N}) is Claude-only; "
            "use a literal effort level instead.",
            feature="reasoning_budget_tokens",
        )
    raise UnsupportedFeatureError(
        f"copilot: unrecognised thinking mode {thinking!r}",
        feature="reasoning_effort",
    )


def _copilot_tools_fingerprint(tools: list[FunctionTool]) -> str:
    """Deterministic fingerprint for the session's ``tools=`` list.

    The fingerprint goes into the :meth:`_ensure_session` cache key so
    a tools-change forces a session rebuild (Copilot bakes the tool
    list at ``create_session()`` time — there's no after-the-fact
    registration). Includes each tool's ``name``, ``description``, and
    Pydantic schema so a doc tweak invalidates the cache.
    """
    if not tools:
        return "__no_tools__"
    parts: list[str] = []
    for t in tools:
        parts.append(f"{t.name}|{t.description}|{t.params.model_json_schema()}")
    return "||".join(parts)


def _translate_one_copilot_tool(ft: FunctionTool) -> Any:
    """Wrap one :class:`FunctionTool` as a Copilot SDK ``Tool``.

    The SDK's :func:`copilot.define_tool` takes the handler in its
    ``(params, invocation_context) -> Any`` shape and the params
    Pydantic model as ``params_type=``. We adapt the airframe handler
    signature (``(BaseModel) -> Awaitable[Any]``) by ignoring the
    ``invocation_context`` and awaiting the user's coroutine. Handler
    exceptions propagate to the SDK, which surfaces them via the
    matching :class:`ToolExecutionCompleteData(success=False)` — the
    airframe stream emits :class:`ToolCallResult(is_error=True)` in
    response so the model can recover.

    ``skip_permission=True`` because :meth:`CopilotAgentSession._ensure_session`
    already installs :meth:`PermissionHandler.approve_all` at session
    construction — keeping every airframe-registered tool consistent
    with that policy avoids accidental per-tool prompts that would
    block automation.
    """
    from copilot import define_tool

    captured = ft  # bind for closure stability

    async def _handler(params: Any, _invocation: Any) -> Any:
        return await captured.handler(params)

    return define_tool(
        captured.name,
        description=captured.description,
        handler=_handler,
        params_type=captured.params,
        skip_permission=True,
    )


def _serialize_copilot_tool_arguments(arguments: Any) -> str:
    """Serialise a :attr:`ToolExecutionStartData.arguments` value for
    the airframe ``arguments_preview`` event field."""
    import json

    if arguments is None:
        return ""
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return repr(arguments)


def _copilot_tool_result_output(data: Any) -> Any:
    """Extract :attr:`ToolCallResult.output` from a
    :class:`ToolExecutionCompleteData` payload.

    Success path: the SDK's ``ToolExecutionCompleteResult.content``
    field is the canonical user-visible string. Falls back through
    ``detailed_content`` then ``contents`` (a typed multi-part list)
    when ``content`` is empty.

    Failure path (``success=False``): surface
    :attr:`ToolExecutionCompleteError.message` so the consumer sees
    the same string the model would.
    """
    if not data.success:
        err = data.error
        if err is None:
            return "tool execution failed"
        message = getattr(err, "message", None) or "tool execution failed"
        code = getattr(err, "code", None)
        return f"{code}: {message}" if code else message
    result = data.result
    if result is None:
        return ""
    if result.content:
        return result.content
    if result.detailed_content:
        return result.detailed_content
    contents = result.contents or []
    text_parts: list[str] = []
    for part in contents:
        text = getattr(part, "data", None) or getattr(part, "description", None)
        if text:
            text_parts.append(str(text))
    return "".join(text_parts)


__all__ = [
    "DEFAULT_COPILOT_MODEL",
    "CopilotAgentSession",
    "CopilotRuntime",
    "SUBMIT_RESULT_TOOL",
]
