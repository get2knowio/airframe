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
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel

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
from airframe.events import RuntimeEvent, TextDelta, TurnComplete
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
from airframe.sessions import _split_prompt_parts
from airframe.thinking import ThinkingMode

logger = logging.getLogger(__name__)

T = TypeVar("T")


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
    ) -> RuntimeResult:
        # Phase 1 Iteration G: ``execute()`` is documented sugar for
        # ``runtime.session(...).execute(...) + close()``. Single-turn,
        # ephemeral. Consumers wanting context warmth across calls open
        # a session explicitly and reuse it.
        del persona  # accepted in the protocol but not consumed by this family
        sess = self.session(system=system, model=model)
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
        provider_options: Any | None = None,
    ) -> AgentSession:
        """Open a bespoke :class:`OpenAICompatibleSession`.

        Iteration C of Phase 1 replaces the
        :class:`~airframe.sessions._ThinAgentSession` placeholder with a
        real client-side ``messages=[]`` buffer, ``stream=True``-backed
        :meth:`AgentSession.stream`, and
        :func:`asyncio.Task.cancel`-driven
        :meth:`AgentSession.cancel`.

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
        # provider_options accepted but unused — Phase 2+ fills each
        # ProviderOptions dataclass as the corresponding feature lands.
        del provider_options
        return OpenAICompatibleSession(self, system=system, model=model)

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
            return RuntimeTransientError(f"{self.label}: transient: {exc}")
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
        text, images, _files = _split_prompt_parts(
            prompt,
            adapter_label=self._runtime.label,
            supports_vision=True,
            supports_file=False,
        )
        reasoning_effort = _translate_thinking_for_openai(thinking, label=self._runtime.label)
        self._messages.append({"role": "user", "content": _build_user_content(text, images)})
        task = asyncio.create_task(
            self._do_execute(schema=schema, reasoning_effort=reasoning_effort, timeout=timeout)
        )
        self._in_flight_task = task
        try:
            result = await task
        except asyncio.CancelledError as exc:
            self._messages.pop()
            raise RuntimeCancelledError(f"{self._runtime.label}: cancelled") from exc
        except BaseException:
            self._messages.pop()
            raise
        finally:
            self._in_flight_task = None
        self._messages.append({"role": "assistant", "content": result.text})
        return result

    async def _do_execute(
        self,
        *,
        schema: type[BaseModel] | None,
        reasoning_effort: str | None,
        timeout: float,
    ) -> RuntimeResult:
        client = self._runtime._ensure_client()
        model_id = self._runtime._resolve_model(self._model)
        response_format = _build_response_format(schema)
        create_kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": list(self._messages),
            "response_format": response_format,
            "timeout": timeout,
        }
        if reasoning_effort is not None:
            create_kwargs["reasoning_effort"] = reasoning_effort
        try:
            response = await client.chat.completions.create(**create_kwargs)
        except Exception as exc:
            raise self._runtime._classify_exception(exc) from exc
        return self._runtime._build_result(response, model_id=model_id, schema=schema)

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
        text, images, _files = _split_prompt_parts(
            prompt,
            adapter_label=self._runtime.label,
            supports_vision=True,
            supports_file=False,
        )
        reasoning_effort = _translate_thinking_for_openai(thinking, label=self._runtime.label)
        self._messages.append({"role": "user", "content": _build_user_content(text, images)})
        self._stream_cancelled = False
        text_chunks: list[str] = []
        finish: str | None = None
        usage: Any = None
        try:
            client = self._runtime._ensure_client()
            model_id = self._runtime._resolve_model(self._model)
            response_format = _build_response_format(schema)
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
            try:
                stream = await client.chat.completions.create(**stream_kwargs)
            except Exception as exc:
                raise self._runtime._classify_exception(exc) from exc

            self._active_stream = stream
            try:
                async for chunk in stream:
                    if self._stream_cancelled:
                        raise RuntimeCancelledError(f"{self._runtime.label}: stream cancelled")
                    choices = getattr(chunk, "choices", None) or []
                    if choices:
                        choice = choices[0]
                        delta = getattr(choice, "delta", None)
                        content = getattr(delta, "content", None) if delta else None
                        if content:
                            text_chunks.append(content)
                            yield TextDelta(text=content)
                        chunk_finish = getattr(choice, "finish_reason", None)
                        if chunk_finish:
                            finish = chunk_finish
                    chunk_usage = getattr(chunk, "usage", None)
                    if chunk_usage is not None:
                        usage = chunk_usage
            finally:
                self._active_stream = None

            full_text = "".join(text_chunks)
            structured: Any = None
            if schema is not None:
                structured = self._runtime._parse_structured(full_text, schema=schema)

            input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
            output_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
            details = getattr(usage, "prompt_tokens_details", None) if usage else None
            cache_read = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
            completion_details = (
                getattr(usage, "completion_tokens_details", None) if usage else None
            )
            reasoning_tokens = (
                int(getattr(completion_details, "reasoning_tokens", 0) or 0)
                if completion_details
                else 0
            )
            cost = CostRecord(
                provider_id=self._runtime.PROVIDER_ID,
                model_id=model_id,
                cost_usd=self._runtime._compute_cost_usd(
                    model_id, input_tokens=input_tokens, output_tokens=output_tokens
                ),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read,
                cache_write_tokens=0,
                finish=finish,
                reasoning_tokens=reasoning_tokens,
            )
            result = RuntimeResult(
                text=full_text,
                structured=structured,
                cost=cost,
                finish=finish,
                raw=None,
            )
            self._messages.append({"role": "assistant", "content": full_text})
            yield TurnComplete(result=result)
        except BaseException:
            # Roll back the user message we appended at the top so the
            # next attempt sends a clean history. The TurnComplete
            # branch above has already extended the buffer with the
            # assistant message before yielding — failures past that
            # point leave the buffer in its committed state.
            if self._messages and self._messages[-1].get("role") == "user":
                self._messages.pop()
            raise

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
        self._closed = True
        # Runtime owns the AsyncOpenAI client; never tear it down here.
        # Cancel any in-flight work as a courtesy.
        await self.cancel()

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


__all__ = ["ModelMeta", "OpenAICompatibleRuntime", "OpenAICompatibleSession"]
