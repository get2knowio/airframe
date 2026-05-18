"""``BedrockRuntime`` — :class:`AgentRuntime` over AWS Bedrock's Converse API.

Wraps the Converse API surface served by ``aioboto3``'s
``bedrock-runtime`` client. Converse is Bedrock's vendor-normalised
chat envelope (``messages`` + ``system`` + ``toolConfig`` +
``inferenceConfig`` + ``additionalModelRequestFields``), fronting
Anthropic Claude, Meta Llama, Mistral, Cohere, Amazon Nova, and
AI21 Jamba behind one AWS-billed endpoint with IAM-rooted auth and
region pinning.

**Iteration A scope.** This commit lands the adapter's protocol
scaffolding only — discovery, capability predicates, the four-step
AWS credential chain, region resolution, ``unwrap()`` self-cast,
``close()`` / ``reset()`` lifecycle, and ``list_models()`` backed by
``bedrock.list_foundation_models()``. Behaviour-bearing methods
(``execute``, ``session``, ``stream``, ``cancel``) raise
:class:`NotImplementedError` pointing at the iteration that will
wire them (B for execute/stream/cancel + structured output, C for
vision/files + reasoning, D for tools + permission, E for hooks +
budget, F for ``BedrockOptions`` wrap-up). ``SUPPORTED_FEATURES``
is the empty set until those iterations flip flags on.

**Auth.** Resolved at first network call via the boto3 four-step
chain — see :doc:`auth` and ``_resolve_aws_credentials``:

1. Explicit ``aws_access_key_id`` / ``aws_secret_access_key``
   (+ optional ``aws_session_token``) constructor args.
2. ``AWS_ACCESS_KEY_ID`` + ``AWS_SECRET_ACCESS_KEY``
   (+ optional ``AWS_SESSION_TOKEN``) env vars.
3. ``AWS_PROFILE`` env var → ``~/.aws/credentials`` /
   ``~/.aws/config`` profile resolution.
4. Default credential chain — IAM instance profile (EC2), ECS task
   role, Lambda execution role, IRSA (EKS).

**Region.** Independent from credentials:

1. Explicit ``region_name=`` constructor arg.
2. ``AWS_REGION`` / ``AWS_DEFAULT_REGION`` env var.
3. ``~/.aws/config`` ``region`` for the resolved profile (boto3
   handles this natively when the session is constructed without
   an explicit region).

If no region resolves *and* the user did not configure one in
``~/.aws/config``, the first network call raises
:class:`RuntimeAuthError` — Bedrock is region-pinned and silent
fallback to a default region would route traffic to a different
model catalog than the user expects.

**``validate_binding``.** Accepts any non-empty ``model_id`` when
``provider_id == "bedrock"``. The Bedrock catalog is too dynamic
to gate by prefix — inference profiles
(``us.anthropic.claude-3-5-sonnet-20241022-v2:0``), provisioned
throughput ARNs, and per-region model variants all coexist; an
allowlist would lag the catalog.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from airframe.errors import (
    RuntimeAuthError,
    RuntimeProtocolError,
    RuntimeTransientError,
)
from airframe.features import Feature
from airframe.models import (
    CAPABILITY_STREAMING,
    CAPABILITY_STRUCTURED_OUTPUT,
    CAPABILITY_TOOLS,
    CAPABILITY_VISION,
    ModelInfo,
)
from airframe.protocol import (
    AgentRuntime,
    AgentSession,
    ProviderModel,
    RuntimeResult,
    UnsupportedBindingError,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import BaseModel

    from airframe.hooks import HookEvent
    from airframe.inputs import Prompt
    from airframe.options import ProviderOptions
    from airframe.permission import PermissionCallback
    from airframe.thinking import ThinkingMode
    from airframe.tools import FunctionTool, McpServerRef

logger = logging.getLogger(__name__)

T = TypeVar("T")


#: Default Bedrock model when no binding is specified. Claude 3.5 Haiku
#: is the cheapest broadly-available Anthropic-on-Bedrock variant.
#: Override per-call via :class:`ProviderModel` or
#: ``BEDROCK_DEFAULT_MODEL``.
DEFAULT_BEDROCK_MODEL = "anthropic.claude-3-5-haiku-20241022-v1:0"


@dataclass(frozen=True, slots=True)
class _ModelMeta:
    """Per-model enrichment for the live ``list_foundation_models`` response."""

    display_name: str
    context_window: int | None = None
    input_per_1k: float | None = None
    output_per_1k: float | None = None
    capabilities: frozenset[str] = field(default_factory=frozenset)


#: Curated subset of Bedrock model IDs with stable display names +
#: context windows. The live ``list_foundation_models`` response carries
#: ``modelId`` / ``modelName`` / ``providerName``; this table layers
#: airframe's normalised display label and context window on top.
#: Pricing is intentionally absent at Iteration A — the cost table
#: (``_BEDROCK_PRICING``) lands with the budget-cap work in Iteration E.
#:
#: Inference-profile IDs (e.g. ``us.anthropic.claude-...``) are not
#: keyed here — they enrich with ``None`` and fall through to the raw
#: ``modelName`` / ``modelId`` from the API response. Bedrock's
#: cross-region inference profiles are too region-specific to maintain
#: a static table for.
_BEDROCK_METADATA: dict[str, _ModelMeta] = {
    # Anthropic on Bedrock
    "anthropic.claude-3-5-haiku-20241022-v1:0": _ModelMeta(
        "Claude 3.5 Haiku (via Bedrock)", context_window=200_000
    ),
    "anthropic.claude-3-5-sonnet-20241022-v2:0": _ModelMeta(
        "Claude 3.5 Sonnet v2 (via Bedrock)", context_window=200_000
    ),
    "anthropic.claude-3-opus-20240229-v1:0": _ModelMeta(
        "Claude 3 Opus (via Bedrock)", context_window=200_000
    ),
    # Amazon Nova
    "amazon.nova-micro-v1:0": _ModelMeta("Amazon Nova Micro", context_window=128_000),
    "amazon.nova-lite-v1:0": _ModelMeta("Amazon Nova Lite", context_window=300_000),
    "amazon.nova-pro-v1:0": _ModelMeta("Amazon Nova Pro", context_window=300_000),
    # Meta Llama 3.x Instruct
    "meta.llama3-1-8b-instruct-v1:0": _ModelMeta(
        "Llama 3.1 8B Instruct (via Bedrock)", context_window=128_000
    ),
    "meta.llama3-1-70b-instruct-v1:0": _ModelMeta(
        "Llama 3.1 70B Instruct (via Bedrock)", context_window=128_000
    ),
    "meta.llama3-1-405b-instruct-v1:0": _ModelMeta(
        "Llama 3.1 405B Instruct (via Bedrock)", context_window=128_000
    ),
    # Mistral
    "mistral.mistral-large-2407-v1:0": _ModelMeta(
        "Mistral Large 2 (via Bedrock)", context_window=128_000
    ),
    # Cohere
    "cohere.command-r-plus-v1:0": _ModelMeta(
        "Cohere Command R+ (via Bedrock)", context_window=128_000
    ),
}


class BedrockRuntime(AgentRuntime):
    """AWS Bedrock Converse API as an :class:`AgentRuntime`.

    Args:
        model: Default Bedrock model identifier used when ``execute()``
            is called without a :class:`ProviderModel` override. Honours
            ``BEDROCK_DEFAULT_MODEL`` env var if set for testing.
        region_name: AWS region the Bedrock-runtime client targets.
            Falls back to ``AWS_REGION`` / ``AWS_DEFAULT_REGION`` env
            var, then to whatever boto3 resolves from
            ``~/.aws/config``.
        aws_access_key_id: Explicit AWS access key. When ``None``
            (default), auth resolves via env / profile / instance-role
            chain.
        aws_secret_access_key: Companion secret to ``aws_access_key_id``.
            Required when the key id is set explicitly.
        aws_session_token: Temporary STS session token (e.g. from
            ``aws sts assume-role``). Optional even when the access
            key + secret are explicit.
        profile_name: ``~/.aws/credentials`` profile name. Equivalent
            to setting ``AWS_PROFILE`` for the lifetime of this
            runtime instance.
    """

    label = "bedrock"

    #: Canonical provider ID this adapter serves.
    PROVIDER_ID: ClassVar[str] = "bedrock"

    #: Vendor SDK that must be importable for this adapter to work.
    REQUIRES_PACKAGE: ClassVar[str] = "aioboto3"

    #: pip extra that brings the vendor SDK in.
    EXTRA_NAME: ClassVar[str] = "bedrock"

    #: Features this runtime exposes today.
    #:
    #: Iteration A intentionally ships an empty set — discovery,
    #: capability predicates, and ``list_models()`` work, but no
    #: behaviour-bearing capability is wired yet. Iteration B flips
    #: ``STRUCTURED_OUTPUT_JSON_SCHEMA``, ``STREAMING``, and ``CANCEL``
    #: True once ``execute()`` / ``stream()`` / ``cancel()`` land.
    SUPPORTED_FEATURES: ClassVar[frozenset[Feature]] = frozenset()

    def __init__(
        self,
        *,
        model: str | None = None,
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        profile_name: str | None = None,
    ) -> None:
        self._default_model = (
            model or os.environ.get("BEDROCK_DEFAULT_MODEL") or DEFAULT_BEDROCK_MODEL
        )
        self._region_override = region_name
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._aws_session_token = aws_session_token
        self._profile_name = profile_name
        self._closed = False

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
        # Iteration A scaffolding: the Converse API call lands in
        # Iteration B alongside the bespoke ``BedrockSession``. Until
        # then the entry point exists so ``supports()`` / discovery
        # work, but a real call raises so the gap is loud.
        raise NotImplementedError(
            "BedrockRuntime.execute() is wired in Iteration B of the bedrock-adapter "
            "plan. Iteration A ships discovery + capability scaffolding only."
        )

    async def reset(self) -> None:
        # Sessionless runtime — the per-conversation buffer lives on
        # ``BedrockSession`` (Iteration B). Nothing scope-bound to drop.
        return None

    async def close(self) -> None:
        # Idempotent + never raises (runs from ``finally`` / ``__aexit__``).
        # The aioboto3 client is opened lazily inside ``list_models()`` /
        # ``execute()`` as an ``async with`` block, so nothing long-lived
        # outlives those calls at Iteration A. ``_closed`` is set so
        # later iterations can refuse calls after teardown without
        # changing the public lifecycle contract.
        self._closed = True

    def validate_binding(self, binding: ProviderModel) -> bool:
        # Accept any non-empty model_id: Bedrock's catalog mixes raw
        # model IDs, inference-profile IDs, and PT ARNs. A prefix
        # allowlist would silently reject valid bindings every time AWS
        # adds a new model family or inference profile.
        return binding.provider_id == self.PROVIDER_ID and bool(binding.model_id)

    def supports(self, feature: Feature, model: ProviderModel | None = None) -> bool:
        return feature in self.SUPPORTED_FEATURES

    def unwrap(self, cls: type[T]) -> T:
        if isinstance(self, cls):
            return self  # type: ignore[return-value]
        # Iteration A doesn't materialise a vendor client — the
        # bedrock-runtime aioboto3 client is built per ``list_models()``
        # call as an async-context-manager and torn down on exit. The
        # persistent client mapping (``unwrap(BedrockRuntimeClient)``)
        # lands in Iteration B once execute()/session() own the client.
        raise TypeError(
            f"BedrockRuntime cannot unwrap to {cls!r}; only "
            f"BedrockRuntime is supported on the runtime today. The "
            f"aioboto3 bedrock-runtime client becomes reachable via "
            f"``unwrap(BedrockRuntimeClient)`` in Iteration B once the "
            f"session class owns it."
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
        # Iteration A scaffolding. ``BedrockSession`` (with
        # ``messages=[]`` buffer + per-turn ``client.converse()``) lands
        # in Iteration B. Raising here keeps the protocol method present
        # so ``runtime.session`` is callable for type-checking / mocking
        # purposes, but the method is not yet behavioural.
        del resume, system, model, tools, mcp_servers
        del on_permission, on_event, provider_options
        raise NotImplementedError(
            "BedrockRuntime.session() is wired in Iteration B of the bedrock-adapter "
            "plan. Iteration A ships discovery + capability scaffolding only."
        )

    async def list_models(self) -> list[ModelInfo]:
        """Return text-output models the resolved AWS identity can see.

        Hits ``bedrock.list_foundation_models(byOutputModality="TEXT")``
        on the resolved AWS region and identity. Embedding-only models
        are filtered out server-side by the modality argument.

        Raises:
            RuntimeAuthError: When AWS credentials are missing /
                invalid, or when no region resolves through the boto3
                chain (Bedrock is region-pinned; falling back to
                ``us-east-1`` silently would route to a different
                catalog).
            RuntimeTransientError: When the Bedrock API returns a
                throttling / 5xx response.
            RuntimeProtocolError: When the response shape is
                unparseable.
        """
        # Lazy-import: ``import airframe`` shouldn't pull aioboto3 in.
        # Consumers calling list_models() have already accepted the
        # dependency by instantiating BedrockRuntime under the
        # [bedrock] extra.
        try:
            import aioboto3
        except ImportError as exc:
            raise ImportError(
                "BedrockRuntime requires the 'aioboto3' package. "
                "Install with: pip install airframe-agents[bedrock]"
            ) from exc

        region = self._resolve_region()
        if not region:
            raise RuntimeAuthError(
                "BedrockRuntime: no AWS region resolved. Bedrock is region-pinned; "
                "set AWS_REGION (or pass region_name= explicitly) so list_models() "
                "knows which catalog to query."
            )
        session_kwargs = self._resolve_aws_credentials()
        # ``bedrock`` (the control-plane / catalog client) is distinct
        # from ``bedrock-runtime`` (the model-invocation client). Only
        # ``bedrock`` exposes ``list_foundation_models``; ``execute()``
        # in Iteration B will open ``bedrock-runtime`` instead.
        try:
            session = aioboto3.Session(**session_kwargs)
            async with session.client("bedrock", region_name=region) as client:
                payload = await client.list_foundation_models(byOutputModality="TEXT")
        except Exception as exc:  # noqa: BLE001 — classify at boundary
            raise _classify_bedrock_error(exc) from exc

        summaries = payload.get("modelSummaries", []) if isinstance(payload, dict) else []
        out: list[ModelInfo] = []
        for entry in summaries:
            model_id = entry.get("modelId")
            if not isinstance(model_id, str) or not model_id:
                continue
            meta = _BEDROCK_METADATA.get(model_id)
            display_name = (
                meta.display_name if meta is not None else entry.get("modelName") or model_id
            )
            context_window = meta.context_window if meta is not None else None
            capabilities = _infer_capabilities(entry)
            if meta is not None:
                capabilities = meta.capabilities | capabilities
            out.append(
                ModelInfo(
                    id=model_id,
                    display_name=display_name,
                    provider_id=self.PROVIDER_ID,
                    context_window=context_window,
                    pricing_input_per_1k_usd=meta.input_per_1k if meta is not None else None,
                    pricing_output_per_1k_usd=meta.output_per_1k if meta is not None else None,
                    capabilities=capabilities,
                    raw=entry,
                )
            )
        return out

    # --- Internals ----------------------------------------------------------

    def _resolve_aws_credentials(self) -> dict[str, str | None]:
        """Return kwargs for :class:`aioboto3.Session` per the auth chain.

        Returns the kwargs ``aioboto3.Session(...)`` understands —
        ``aws_access_key_id``, ``aws_secret_access_key``,
        ``aws_session_token``, ``profile_name``. ``None`` values are
        omitted so boto3's own resolution (env vars → profile →
        instance role) keeps working.

        Order of precedence:

        1. Explicit constructor args (``aws_access_key_id=`` etc.).
        2. ``AWS_ACCESS_KEY_ID`` + ``AWS_SECRET_ACCESS_KEY``
           (+ optional ``AWS_SESSION_TOKEN``) env vars.
        3. ``AWS_PROFILE`` env var → profile resolution (boto3
           handles this when ``profile_name=`` is passed; otherwise
           boto3 picks ``AWS_PROFILE`` up natively from env).
        4. Default credential chain (instance profile / ECS task role
           / Lambda role / IRSA) — boto3 handles when no explicit
           kwargs are given.
        """
        kwargs: dict[str, str | None] = {}
        # Step 1: explicit constructor args win.
        if self._aws_access_key_id is not None:
            kwargs["aws_access_key_id"] = self._aws_access_key_id
            kwargs["aws_secret_access_key"] = self._aws_secret_access_key
            if self._aws_session_token is not None:
                kwargs["aws_session_token"] = self._aws_session_token
            if self._profile_name is not None:
                kwargs["profile_name"] = self._profile_name
            return kwargs
        # Step 2: env-var key pair. We don't read these explicitly —
        # boto3 picks them up natively when no profile_name is set.
        # Setting profile_name here would *override* the env-var path,
        # so we only set it if the user asked for it.
        if self._profile_name is not None:
            kwargs["profile_name"] = self._profile_name
            return kwargs
        # Step 3: AWS_PROFILE env var. boto3 reads this natively when
        # the session is constructed with no profile_name, so we leave
        # kwargs empty.
        # Step 4: default chain. Same — empty kwargs let boto3 walk it.
        return kwargs

    def _resolve_region(self) -> str | None:
        """Return the AWS region per the resolution chain, or ``None``.

        1. Explicit ``region_name=`` constructor arg.
        2. ``AWS_REGION`` env var.
        3. ``AWS_DEFAULT_REGION`` env var.

        Returns ``None`` when none resolves. The caller (``list_models``,
        future ``execute``) raises :class:`RuntimeAuthError` rather than
        letting boto3 fall through to the SDK default region — Bedrock
        catalogs are per-region and silent fallback misroutes traffic.
        boto3's own ``~/.aws/config`` ``region = ...`` resolution still
        works when callers pass ``region_name=None`` through to
        :class:`aioboto3.Session`; that path is honoured at the session
        layer, not here.
        """
        if self._region_override:
            return self._region_override
        env_region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        if env_region:
            return env_region
        return None

    def _resolve_model(self, model: ProviderModel | None) -> str:
        if model is None:
            return self._default_model
        if not self.validate_binding(model):
            raise UnsupportedBindingError(
                f"BedrockRuntime cannot serve {model.label!r}; "
                f"provider must be {self.PROVIDER_ID!r} and model_id must be non-empty"
            )
        return model.model_id


def _infer_capabilities(entry: dict[str, Any]) -> frozenset[str]:
    """Derive airframe capability flags from a Bedrock model summary.

    The ``modelSummaries`` entries carry ``inputModalities`` (list of
    ``"TEXT"`` / ``"IMAGE"`` / ``"DOCUMENT"``) and
    ``responseStreamingSupported`` (bool). Map those onto the
    airframe capability strings so menus can render badges without
    consulting a separate per-model lookup.

    Structured output + tools are universally on under Converse, so
    every text-output model gets those flags too.
    """
    caps: set[str] = {CAPABILITY_STRUCTURED_OUTPUT, CAPABILITY_TOOLS}
    if entry.get("responseStreamingSupported"):
        caps.add(CAPABILITY_STREAMING)
    input_modalities = entry.get("inputModalities") or []
    if isinstance(input_modalities, list) and "IMAGE" in input_modalities:
        caps.add(CAPABILITY_VISION)
    return frozenset(caps)


def _classify_bedrock_error(exc: Exception) -> Exception:
    """Map a boto3 / aioboto3 exception onto the airframe error hierarchy.

    Iteration A only needs auth / transient / protocol classification
    for the catalog endpoint; the per-model execute() classification
    (``ValidationException`` → :class:`RuntimeModelNotFoundError`,
    throttling → :class:`RuntimeTransientError`) lands in Iteration B
    alongside ``execute()`` itself.
    """
    # Late-import so we never force botocore at module-import time.
    try:
        from botocore.exceptions import (
            ClientError,
            EndpointConnectionError,
            NoCredentialsError,
            NoRegionError,
            PartialCredentialsError,
        )
    except ImportError:
        # botocore not installed (i.e. the consumer skipped the extra
        # entirely). Surface the original exception unchanged.
        return exc

    if isinstance(exc, NoCredentialsError | PartialCredentialsError):
        return RuntimeAuthError(f"bedrock: no usable AWS credentials: {exc}")
    if isinstance(exc, NoRegionError):
        return RuntimeAuthError(
            "bedrock: no AWS region resolved. Set AWS_REGION (or pass "
            f"region_name=) — Bedrock is region-pinned. Underlying: {exc}"
        )
    if isinstance(exc, ClientError):
        # ``response["Error"]["Code"]`` is the canonical error name.
        code = ""
        status: int | None = None
        try:
            err = exc.response.get("Error", {})  # type: ignore[attr-defined]
            code = err.get("Code", "") or ""
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        if code in {
            "UnrecognizedClientException",
            "InvalidSignatureException",
            "AccessDeniedException",
            "ExpiredTokenException",
        }:
            return RuntimeAuthError(f"bedrock: auth: {exc}", status=status)
        if code in {"ThrottlingException", "ServiceUnavailableException"}:
            return RuntimeTransientError(f"bedrock: transient {code}: {exc}", status=status)
        if isinstance(status, int) and 500 <= status < 600:
            return RuntimeTransientError(f"bedrock: transient {status}: {exc}", status=status)
        return RuntimeProtocolError(f"bedrock: {code or 'ClientError'}: {exc}", status=status)
    if isinstance(exc, EndpointConnectionError):
        return RuntimeTransientError(f"bedrock: endpoint unreachable: {exc}")
    return exc


__all__ = [
    "DEFAULT_BEDROCK_MODEL",
    "BedrockRuntime",
]
