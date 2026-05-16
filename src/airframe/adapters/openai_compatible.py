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

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel

from airframe.cost import CostRecord
from airframe.errors import (
    AgentRuntimeError,
    RuntimeAuthError,
    RuntimeModelNotFoundError,
    RuntimeProtocolError,
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

    #: Features this runtime family exposes today. Phase 0 declares
    #: only structured output (already wired via
    #: ``response_format={"type":"json_schema",...}``). MCP-as-tool is
    #: OpenAI Responses-only; the Chat Completions surface this base
    #: targets does NOT support it, and that won't change.
    #: ``STRUCTURED_OUTPUT_STRICT`` stays False even though the SDK
    #: accepts ``strict: True`` — the base passes ``strict: False`` for
    #: compat-vendor portability (Together / Groq / Fireworks /
    #: OpenRouter coverage is uneven). Phase 2 may add an explicit
    #: opt-in.
    SUPPORTED_FEATURES: ClassVar[frozenset[Feature]] = frozenset(
        {Feature.STRUCTURED_OUTPUT_JSON_SCHEMA}
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
        prompt: str,
        *,
        schema: type[BaseModel] | None = None,
        system: str | None = None,
        persona: str | None = None,
        model: ProviderModel | None = None,
        timeout: float = 600.0,
    ) -> RuntimeResult:
        model_id = self._resolve_model(model)
        client = self._ensure_client()

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response_format: dict[str, Any] | None = None
        if schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": False,
                    "schema": schema.model_json_schema(),
                },
            }

        try:
            response = await client.chat.completions.create(
                model=model_id,
                messages=messages,
                response_format=response_format,
                timeout=timeout,
            )
        except Exception as exc:
            raise self._classify_exception(exc) from exc

        return self._build_result(response, model_id=model_id, schema=schema)

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


__all__ = ["ModelMeta", "OpenAICompatibleRuntime"]
