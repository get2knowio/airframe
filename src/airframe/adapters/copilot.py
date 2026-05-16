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
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel, ValidationError

from airframe.cost import CostRecord
from airframe.errors import (
    AgentRuntimeError,
    RuntimeAuthError,
    RuntimeModelNotFoundError,
    RuntimeProtocolError,
    RuntimeServerStartError,
    RuntimeStructuredOutputError,
    RuntimeTransientError,
)
from airframe.features import Feature
from airframe.models import ModelInfo
from airframe.protocol import (
    AgentRuntime,
    ProviderModel,
    RuntimeResult,
    UnsupportedBindingError,
)

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

    #: Features this runtime exposes today. Phase 0 declares only
    #: structured output — wired via the forced ``submit_result`` tool
    #: pattern. Later phases flip more bits as their APIs land.
    SUPPORTED_FEATURES: ClassVar[frozenset[Feature]] = frozenset(
        {Feature.STRUCTURED_OUTPUT_JSON_SCHEMA}
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
        self._session: Any | None = None  # copilot.CopilotSession
        self._session_key: str | None = None

        # Per-execute state captured via the tool handler + event subscriber.
        self._captured_payload: BaseModel | None = None
        self._captured_usage: Any | None = None  # AssistantUsageData
        self._captured_error: Any | None = None  # SessionErrorData
        self._last_assistant_message: Any | None = None

    # --- AgentRuntime interface ---------------------------------------------

    async def execute(
        self,
        prompt: str,
        *,
        schema: type[BaseModel] | None = None,
        system: str | None = None,
        persona: str | None = None,
        model: ProviderModel | None = None,
        timeout: float = 600.0,
    ) -> RuntimeResult:
        model_id = self._resolve_model(model)
        session = await self._ensure_session(schema=schema, system=system, model=model_id)

        # Reset per-execute capture slots.
        self._captured_payload = None
        self._captured_usage = None
        self._captured_error = None
        self._last_assistant_message = None

        try:
            await asyncio.wait_for(
                session.send_and_wait(prompt, timeout=timeout),
                timeout=timeout,
            )
        except TimeoutError as exc:
            raise RuntimeTransientError(f"copilot: execute timed out after {timeout}s") from exc
        except Exception as exc:
            raise self._classify_exception(exc) from exc

        if self._captured_error is not None:
            raise self._error_from_session_error(self._captured_error)

        text = ""
        if self._last_assistant_message is not None and hasattr(
            self._last_assistant_message, "data"
        ):
            text = getattr(self._last_assistant_message.data, "content", "") or ""

        if schema is None:
            # Plain-text mode: no submit_result tool was registered, so
            # no payload to capture. The final assistant message
            # content is the result.
            return RuntimeResult(
                text=text,
                structured=None,
                cost=self._cost_from_usage(self._captured_usage, model_id=model_id),
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
                f"copilot: {SUBMIT_RESULT_TOOL} was never called",
                body={"assistant_message_preview": preview_text},
            )

        return RuntimeResult(
            text=text,
            structured=captured.model_dump(),
            cost=self._cost_from_usage(self._captured_usage, model_id=model_id),
            finish="stop",
            raw={
                "usage": self._captured_usage,
                "message": self._last_assistant_message,
            },
        )

    async def reset(self) -> None:
        session = self._session
        self._session = None
        self._session_key = None
        if session is None:
            return
        try:
            await session.destroy()
        except Exception as exc:  # noqa: BLE001 — teardown never raises
            logger.debug("copilot_runtime.reset_failed error=%s", exc)

    async def close(self) -> None:
        await self.reset()
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
            if self._session is None:
                raise TypeError(
                    "CopilotRuntime.unwrap(CopilotSession): no session exists yet "
                    "— call execute() first."
                )
            return self._session  # type: ignore[return-value]
        raise TypeError(
            f"CopilotRuntime cannot unwrap to {cls!r}; supported types are "
            f"CopilotRuntime, copilot.CopilotClient, and copilot.session.CopilotSession."
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

    async def _ensure_session(
        self,
        *,
        schema: type[BaseModel] | None,
        system: str | None,
        model: str,
    ) -> Any:
        schema_fragment = (
            f"{schema.__name__}|{schema.model_json_schema()}"
            if schema is not None
            else "__plain_text__"
        )
        key = f"{model}|{system or ''}|{schema_fragment}"
        if self._session is not None and self._session_key == key:
            return self._session
        await self.reset()
        client = await self._ensure_client()

        from copilot.session import PermissionHandler

        # Two session shapes share the path:
        # * schema=None — plain text. No submit_result tool registered;
        #   no forced-tool prefix on the system message; whatever the
        #   caller passed as ``system`` reaches the model verbatim.
        # * schema=Pydantic — structured. submit_result tool registered
        #   with the schema's JSON shape; system message gets the
        #   forced-tool prefix telling the model to call it.
        create_kwargs: dict[str, Any] = {
            "on_permission_request": PermissionHandler.approve_all,
            "model": model,
        }

        if schema is not None:
            from copilot import define_tool

            async def _submit_handler(params: schema) -> dict[str, Any]:  # type: ignore[valid-type]
                self._captured_payload = params
                return {"ok": True}

            submit_tool = define_tool(
                SUBMIT_RESULT_TOOL,
                description=(
                    f"Submit the final typed payload as a {schema.__name__}. "
                    "Call this exactly once with all required fields filled in."
                ),
                handler=lambda params, inv: _submit_handler(params),
                params_type=schema,
                skip_permission=True,
            )
            create_kwargs["tools"] = [submit_tool]

            forced_prefix = (
                "When you are ready to answer, call the "
                f"`{SUBMIT_RESULT_TOOL}` tool with the typed payload. "
                "Do not emit a final assistant message; the tool call is your answer.\n\n"
            )
            create_kwargs["system_message"] = {
                "mode": "append",
                "content": forced_prefix + (system or ""),
            }
        elif system is not None:
            # Plain text + caller-supplied system prompt: append it
            # without any forced-tool framing.
            create_kwargs["system_message"] = {"mode": "append", "content": system}

        try:
            session = await client.create_session(**create_kwargs)
        except Exception as exc:
            raise self._classify_exception(exc) from exc

        # Subscribe to usage + error events.
        session.on(self._on_event)

        self._session = session
        self._session_key = key
        return session

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

    def _on_event(self, event: Any) -> None:
        """Capture usage + error + final-assistant-message events.

        Runs synchronously off the SDK's event dispatch — keep it cheap.
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
        return CostRecord(
            provider_id=self.PROVIDER_ID,
            model_id=model_id,
            cost_usd=float(usage.cost) if usage.cost is not None else None,
            input_tokens=int(usage.input_tokens or 0),
            output_tokens=int(usage.output_tokens or 0),
            cache_read_tokens=int(usage.cache_read_tokens or 0),
            cache_write_tokens=int(usage.cache_write_tokens or 0),
            finish="stop",
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


__all__ = [
    "DEFAULT_COPILOT_MODEL",
    "CopilotRuntime",
    "SUBMIT_RESULT_TOOL",
]
