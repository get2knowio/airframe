"""``BedrockRuntime`` — :class:`AgentRuntime` over AWS Bedrock's Converse API.

Wraps the Converse API surface served by ``aioboto3``'s
``bedrock-runtime`` client. Converse is Bedrock's vendor-normalised
chat envelope (``messages`` + ``system`` + ``toolConfig`` +
``inferenceConfig`` + ``additionalModelRequestFields``), fronting
Anthropic Claude, Meta Llama, Mistral, Cohere, Amazon Nova, and
AI21 Jamba behind one AWS-billed endpoint with IAM-rooted auth and
region pinning.

**Iteration D scope** (the current commit). Layers function tools
+ permission gating onto the Iteration C baseline. The forced
``submit_result`` tool used for ``schema=`` coexists with
user-registered :class:`~airframe.tools.FunctionTool` entries —
Converse's ``toolConfig`` slot accepts both — so structured output
and function-calling are not mutually exclusive. ``Feature.TOOLS_FUNCTION``
and ``Feature.PERMISSION_CALLBACK`` flip True.

The client-side tool loop in :meth:`BedrockSession.execute` /
:meth:`BedrockSession.stream` parses ``toolUse`` content blocks
from each ``converse()`` response, dispatches the registered
handlers (with a :class:`~airframe.permission.PermissionCallback`
gate when set), appends ``toolResult`` blocks back to the
conversation, and re-calls ``converse()`` until the model emits a
final text response or the ``MAX_TOOL_ITERATIONS`` cap fires.
Mirrors the OpenAI-compatible base's loop shape.

**MCP non-goal.** Bedrock Converse has no MCP slot.
``TOOLS_MCP_STDIO`` / ``TOOLS_MCP_HTTP`` / ``TOOLS_MCP_SSE`` stay
False permanently; ``session(mcp_servers=...)`` raises a decline
that points users at ``runtime.unwrap(BedrockRuntimeClient)`` if
they want to hand-craft an MCP shim themselves.

**Earlier-iteration scope.** Polymorphic prompts + thinking landed
in Iteration C; structured output + streaming + cancel in B;
discovery + auth in A. Hooks + budget land in E.

Layered onto Iteration C's surface:

* :class:`~airframe.inputs.ImageInput` parts translate into
  Converse ``{"image": {"format": "...", "source": {"bytes": ...}}}``
  content blocks. Format is detected from the bytes header or from
  the ``path`` extension. Anthropic on Bedrock, Amazon Nova, and
  Meta Llama 3.2 vision models all honour these.
* :class:`~airframe.inputs.FileInput` parts translate into
  ``{"document": {"format": "pdf|md|txt|...", "name": "...",
  "source": {"bytes": ...}}}`` content blocks. Anthropic-only
  today; other vendors silently ignore.
* ``thinking=`` is translated to Anthropic's extended-thinking
  config and sent under ``additionalModelRequestFields={"thinking":
  {"type": "enabled", "budget_tokens": N}}``. Mapping mirrors
  :class:`ClaudeCodeRuntime`: ``"low"`` → 1024, ``"medium"`` →
  8192, ``"high"`` → 32768, ``{"budget_tokens": N}`` → N,
  ``"disabled"`` omits the field, ``"minimal"`` coerces to
  ``"low"`` with a debug-level log. Non-Anthropic models log at
  debug and pass through — Bedrock ignores unknown
  ``additionalModelRequestFields`` keys per vendor.
* Flags flipped True: ``REASONING_EFFORT``,
  ``REASONING_BUDGET_TOKENS``, ``VISION_INPUT``, ``FILE_INPUT``.

Tools + permission land in Iteration D; hooks + budget in E.

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
3. ``~/.aws/config`` ``region`` for the resolved profile.

If no region resolves, the first network call raises
:class:`RuntimeAuthError`. Bedrock is region-pinned and silent
fallback to a default region would route traffic to a different
model catalog than the user expects.

**Structured output.** Implemented via Bedrock's native ``toolConfig``
with a forced ``submit_result`` tool whose ``inputSchema`` is the
user-supplied Pydantic schema serialised to JSON Schema. The model
must call the tool exactly once; the adapter extracts the validated
payload from the resulting ``toolUse`` content block. Mirrors
:class:`CopilotRuntime`'s forced-tool pattern exactly.

**Lifecycle.** The bedrock-runtime client is lazily built on first
``execute()`` / ``stream()`` (and on ``list_models()`` for the
control-plane client), and torn down by :meth:`close`. Sibling
sessions reuse one client per runtime — opening a session never
spawns a new client.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from pydantic import BaseModel, ValidationError

from airframe.cost import CostRecord
from airframe.errors import (
    RuntimeAuthError,
    RuntimeCancelledError,
    RuntimeContextOverflowError,
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
from airframe.inputs import FileInput, ImageInput, Prompt
from airframe.models import (
    CAPABILITY_STREAMING,
    CAPABILITY_STRUCTURED_OUTPUT,
    CAPABILITY_TOOLS,
    CAPABILITY_VISION,
    ModelInfo,
)
from airframe.permission import PermissionRequest
from airframe.protocol import (
    AgentRuntime,
    AgentSession,
    ProviderModel,
    RuntimeResult,
    UnsupportedBindingError,
)
from airframe.sessions import _split_prompt_parts

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from airframe.hooks import HookEvent
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


#: Canonical name for the hidden structured-output tool. Same constant
#: shape as :class:`CopilotRuntime`'s ``SUBMIT_RESULT_TOOL`` — Bedrock
#: Converse's ``toolConfig`` slot honours the same forced-tool pattern.
SUBMIT_RESULT_TOOL = "submit_result"


#: Hard cap on client-side tool-loop iterations within one user turn.
#: A model that keeps requesting tool calls indefinitely is a real
#: failure mode; matches :data:`MAX_TOOL_ITERATIONS` in the
#: OpenAI-compatible base so the portable bound is the same regardless
#: of which adapter happens to drive the loop.
MAX_TOOL_ITERATIONS = 20


@dataclass(slots=True)
class _StreamTurnState:
    """Mutable accumulator for one ``converse_stream`` turn.

    Replaces what would otherwise be a clutch of nonlocal vars across
    the streaming generator and its outer driver.
    :class:`BedrockSession.stream` reads each field after each turn
    to decide whether to loop or terminate.
    """

    text_parts: list[str] = field(default_factory=list)
    assistant_blocks: list[dict[str, Any]] = field(default_factory=list)
    user_tool_uses: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)
    submit_input: dict[str, Any] | None = None
    stop_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


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
#: Pricing is intentionally absent at Iteration B — the cost table
#: (``_BEDROCK_PRICING``) lands with the budget-cap work in Iteration E.
#:
#: Inference-profile IDs (e.g. ``us.anthropic.claude-...``) are not
#: keyed here — they enrich with ``None`` and fall through to the raw
#: ``modelName`` / ``modelId`` from the API response.
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
    #: * ``STRUCTURED_OUTPUT_JSON_SCHEMA`` — wired via the forced
    #:   ``submit_result`` tool in Converse's ``toolConfig`` slot
    #:   (Iteration B).
    #: * ``STREAMING`` — wired via :class:`BedrockSession.stream` over
    #:   ``client.converse_stream`` (Iteration B).
    #: * ``CANCEL`` — wired via :func:`asyncio.Task.cancel` for
    #:   :meth:`BedrockSession.execute` + a stream-iterator close for
    #:   :meth:`BedrockSession.stream` (Iteration B).
    #: * ``REASONING_EFFORT`` / ``REASONING_BUDGET_TOKENS`` — wired via
    #:   ``additionalModelRequestFields={"thinking": {...}}`` for
    #:   Anthropic-on-Bedrock variants. Non-Anthropic models silently
    #:   ignore the field (Iteration C). Per-vendor: Bedrock declines
    #:   the field rather than rejecting it, so airframe lets the
    #:   request through with a debug-log when the model isn't
    #:   ``anthropic.*``.
    #: * ``VISION_INPUT`` — wired via Converse ``{"image": ...}``
    #:   content blocks. ``path=`` reads the file; ``bytes_=`` passes
    #:   through directly; ``url=`` raises (Converse needs the bytes
    #:   locally) (Iteration C).
    #: * ``FILE_INPUT`` — wired via Converse ``{"document": ...}``
    #:   content blocks. Anthropic-only today; other vendors silently
    #:   ignore (Iteration C).
    #: * ``TOOLS_FUNCTION`` — wired via Converse ``toolConfig`` +
    #:   client-side tool loop in :meth:`BedrockSession._do_execute`
    #:   / :meth:`BedrockSession._stream_one_turn`. The forced
    #:   ``submit_result`` tool used for ``schema=`` coexists with
    #:   user tools — both ride the same ``toolConfig.tools`` list
    #:   (Iteration D).
    #: * ``PERMISSION_CALLBACK`` — wired by firing
    #:   :class:`~airframe.permission.PermissionCallback` around each
    #:   tool handler invocation. ``"allow"`` → run handler;
    #:   ``"deny"`` → return an error message the model sees on its
    #:   next turn; ``"defer"`` → treat as allow with a debug log
    #:   (Bedrock has no native fallback policy to defer to)
    #:   (Iteration D).
    #:
    #: ``SESSION_RESUME`` stays False — Converse is stateless from the
    #: client's perspective. ``TOOLS_MCP_*`` stay False permanently —
    #: Bedrock Converse has no MCP slot. Hooks + budget land in E.
    SUPPORTED_FEATURES: ClassVar[frozenset[Feature]] = frozenset(
        {
            Feature.STRUCTURED_OUTPUT_JSON_SCHEMA,
            Feature.STREAMING,
            Feature.CANCEL,
            Feature.REASONING_EFFORT,
            Feature.REASONING_BUDGET_TOKENS,
            Feature.VISION_INPUT,
            Feature.FILE_INPUT,
            Feature.TOOLS_FUNCTION,
            Feature.PERMISSION_CALLBACK,
        }
    )

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
        # Lazy bedrock-runtime client + its enclosing async context.
        # Built on first execute()/stream(); torn down by close().
        self._aws_session: Any = None
        self._runtime_client_ctx: Any = None
        self._runtime_client: Any = None
        self._client_lock: asyncio.Lock | None = None
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
        # Documented sugar for ``runtime.session(...).execute(...) + close()``.
        # Single-turn, ephemeral — the runtime client is shared, so the
        # only per-call setup is the BedrockSession itself.
        del persona  # accepted in the protocol but not consumed by Bedrock
        sess = self.session(system=system, model=model)
        try:
            return await sess.execute(prompt, schema=schema, thinking=thinking, timeout=timeout)
        finally:
            await sess.close()

    async def reset(self) -> None:
        # Sessionless runtime — the per-conversation buffer lives on
        # ``BedrockSession``. Nothing scope-bound to drop. The shared
        # bedrock-runtime client stays alive for sibling sessions.
        return None

    async def close(self) -> None:
        # Idempotent + never raises (runs from ``finally`` / ``__aexit__``).
        # Tears down the lazily-built bedrock-runtime client if any.
        ctx = self._runtime_client_ctx
        self._runtime_client_ctx = None
        self._runtime_client = None
        self._aws_session = None
        if ctx is not None:
            try:
                await ctx.__aexit__(None, None, None)
            except Exception as exc:  # noqa: BLE001 — teardown never raises
                logger.debug("bedrock.client_teardown_failed error=%s", exc)
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
        # The aioboto3 bedrock-runtime client class is dynamically
        # generated by botocore; users reach it by passing the type
        # they imported (`from types_aiobotocore_bedrock_runtime import
        # BedrockRuntimeClient`) or by the runtime client they already
        # hold. ``isinstance`` is the only honest check.
        if self._runtime_client is not None and isinstance(self._runtime_client, cls):
            return self._runtime_client  # type: ignore[return-value]
        raise TypeError(
            f"BedrockRuntime cannot unwrap to {cls!r}; only "
            f"BedrockRuntime or the live aioboto3 bedrock-runtime client "
            f"(once execute()/stream() has built it) are supported. "
            f"Call execute()/stream() first if you need the client."
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
        # Iteration B accepts the structural kwargs but only ``system`` /
        # ``model`` are honoured; later iterations wire tools/MCP/hooks/
        # permission/options. Per the "no silent fallbacks" principle,
        # passing a non-None decline-target raises.
        if resume is not None:
            raise UnsupportedFeatureError(
                "BedrockSession: SESSION_RESUME is not supported — Converse is "
                "stateless from the client's perspective. The messages buffer "
                "doesn't survive process restart.",
                feature=Feature.SESSION_RESUME,
            )
        if mcp_servers is not None and mcp_servers:
            raise UnsupportedFeatureError(
                "BedrockSession: Bedrock Converse has no MCP slot — "
                "TOOLS_MCP_* are permanent declines on this adapter. If you "
                "need to bridge to an MCP server, reach the live aioboto3 "
                "client via runtime.unwrap(BedrockRuntimeClient) and "
                "hand-craft the shim.",
                feature=Feature.TOOLS_MCP_STDIO,
            )
        if on_event is not None:
            raise UnsupportedFeatureError(
                "BedrockSession: LIFECYCLE_HOOKS lands in Iteration E.",
                feature=Feature.LIFECYCLE_HOOKS,
            )
        if provider_options is not None:
            raise UnsupportedFeatureError(
                "BedrockSession: BedrockOptions namespace lands in Iteration F. "
                "Pass provider_options=None for now."
            )
        model_id = self._resolve_model(model) if model is not None else self._default_model
        return BedrockSession(
            self,
            system=system,
            model_id=model_id,
            tools=tools,
            on_permission=on_permission,
        )

    async def list_models(self) -> list[ModelInfo]:
        """Return text-output models the resolved AWS identity can see.

        Hits ``bedrock.list_foundation_models(byOutputModality="TEXT")``
        on the resolved AWS region and identity. Embedding-only models
        are filtered out server-side by the modality argument.

        Note: ``bedrock`` (the control-plane / catalog client) is
        distinct from ``bedrock-runtime`` (the model-invocation client)
        — only the former exposes ``list_foundation_models``. This call
        does *not* reuse the runtime-level bedrock-runtime client.

        Raises:
            RuntimeAuthError: When AWS credentials are missing /
                invalid, or when no region resolves through the boto3
                chain.
            RuntimeTransientError: When the Bedrock API returns a
                throttling / 5xx response.
            RuntimeProtocolError: When the response shape is
                unparseable.
        """
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

    async def _get_runtime_client(self) -> Any:
        """Return the live aioboto3 bedrock-runtime client, lazy-built.

        Sibling :class:`BedrockSession` instances all share one client —
        Bedrock's stateless wire model makes per-session clients pure
        overhead. The client is built under a lock so concurrent first-
        ``execute()`` calls don't race.
        """
        if self._closed:
            raise RuntimeError("BedrockRuntime is closed")
        if self._runtime_client is not None:
            return self._runtime_client
        # The asyncio.Lock has to be created inside a running loop;
        # constructing it lazily here keeps __init__ usable from sync
        # code (matches how tests construct the runtime).
        if self._client_lock is None:
            self._client_lock = asyncio.Lock()
        async with self._client_lock:
            if self._runtime_client is not None:
                return self._runtime_client
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
                    "set AWS_REGION (or pass region_name= explicitly)."
                )
            try:
                self._aws_session = aioboto3.Session(**self._resolve_aws_credentials())
                self._runtime_client_ctx = self._aws_session.client(
                    "bedrock-runtime", region_name=region
                )
                self._runtime_client = await self._runtime_client_ctx.__aenter__()
            except Exception as exc:
                self._runtime_client_ctx = None
                self._runtime_client = None
                self._aws_session = None
                raise _classify_bedrock_error(exc) from exc
            return self._runtime_client

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
        # Step 1.5: explicit profile_name without keys.
        if self._profile_name is not None:
            kwargs["profile_name"] = self._profile_name
            return kwargs
        # Steps 2-4: empty kwargs let boto3 walk env vars → AWS_PROFILE
        # → default credential chain natively.
        return kwargs

    def _resolve_region(self) -> str | None:
        """Return the AWS region per the resolution chain, or ``None``.

        1. Explicit ``region_name=`` constructor arg.
        2. ``AWS_REGION`` env var.
        3. ``AWS_DEFAULT_REGION`` env var.
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


class BedrockSession:
    """Per-conversation handle for Bedrock Converse.

    Owns a client-side ``messages=[]`` buffer; each :meth:`execute` /
    :meth:`stream` appends the user message before the call and the
    assistant response after success. Failures (including cancellation)
    pop the user message so a retry sends a clean history. Mirrors
    :class:`OpenAICompatibleSession`'s buffer discipline — Converse is
    stateless from the client side, same as Chat Completions.

    :attr:`id` is always ``None``; Converse has no server-side session
    identifier. Consumer code branching on ``session.id is None`` can
    treat that as the "stateless wire" signal.

    The bedrock-runtime client lives on the parent
    :class:`BedrockRuntime` and is shared across sibling sessions.
    Opening a session never spawns a vendor client; closing one
    leaves the runtime client untouched.
    """

    id: str | None = None

    def __init__(
        self,
        runtime: BedrockRuntime,
        *,
        system: str | None = None,
        model_id: str,
        tools: list[FunctionTool] | None = None,
        on_permission: PermissionCallback | None = None,
    ) -> None:
        self._runtime = runtime
        self._model_id = model_id
        self._system = system
        self._messages: list[dict[str, Any]] = []
        self._closed = False
        self._in_flight_task: asyncio.Task[Any] | None = None
        self._active_stream: Any | None = None
        self._stream_cancelled = False
        # Tools are fixed for the session's lifetime. ``None`` / ``[]``
        # both mean "no user tools"; the schema= forced submit_result
        # tool is added per-call in _build_call_kwargs.
        self._tools_by_name: dict[str, FunctionTool] = {t.name: t for t in (tools or [])}
        self._tools_wire: list[dict[str, Any]] = (
            [_translate_one_tool_for_bedrock(t) for t in tools] if tools else []
        )
        self._on_permission = on_permission

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
        _enforce_budget_gates(max_turns=max_turns, max_budget_usd=max_budget_usd)

        content = _build_user_content(prompt, adapter_label=self._runtime.label)
        thinking_field = _translate_thinking_for_bedrock(
            thinking, model_id=self._model_id, label=self._runtime.label
        )
        pre_len = len(self._messages)
        self._messages.append({"role": "user", "content": content})

        task = asyncio.create_task(
            self._do_execute(schema=schema, thinking_field=thinking_field, timeout=timeout)
        )
        self._in_flight_task = task
        try:
            return await task
        except asyncio.CancelledError as exc:
            del self._messages[pre_len:]
            raise RuntimeCancelledError(f"{self._runtime.label}: cancelled") from exc
        except BaseException:
            del self._messages[pre_len:]
            raise
        finally:
            self._in_flight_task = None

    async def _do_execute(
        self,
        *,
        schema: type[BaseModel] | None,
        thinking_field: dict[str, Any] | None,
        timeout: float,
    ) -> RuntimeResult:
        """Drive the client-side tool-loop for one user turn.

        Sends the current ``messages`` buffer to ``client.converse``;
        if the response carries user-tool ``toolUse`` blocks, dispatch
        each handler (under the permission gate if one is set),
        append the assistant message + the matching
        ``{"role": "user", "content": [{"toolResult": ...}]}`` block,
        then re-call. Loops until the model emits a final text
        response (no user-tool calls) or :data:`MAX_TOOL_ITERATIONS`
        round-trips elapse. The forced ``submit_result`` tool used
        for ``schema=`` is captured along the way; the loop only
        terminates when no more *user* tools fire.
        """
        client = await self._runtime._get_runtime_client()
        captured_submit_input: dict[str, Any] | None = None
        for _ in range(MAX_TOOL_ITERATIONS):
            call_kwargs = self._build_call_kwargs(schema=schema, thinking_field=thinking_field)
            try:
                response = await asyncio.wait_for(client.converse(**call_kwargs), timeout=timeout)
            except TimeoutError as exc:
                raise RuntimeTransientError(
                    f"{self._runtime.label}: timeout after {timeout}s"
                ) from exc
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise _classify_bedrock_error(exc) from exc

            output_msg = _require_output_message(response, label=self._runtime.label)
            self._messages.append(output_msg)
            text_parts, user_tool_uses, submit_input = _split_assistant_blocks(
                output_msg, user_tool_names=set(self._tools_by_name)
            )
            if submit_input is not None:
                captured_submit_input = submit_input

            if not user_tool_uses:
                stop_reason = response.get("stopReason")
                structured = None
                if schema is not None:
                    structured = _validate_tool_payload(
                        captured_submit_input,
                        schema=schema,
                        label=self._runtime.label,
                    )
                cost = _build_cost_record(
                    self._runtime.PROVIDER_ID,
                    self._model_id,
                    response.get("usage", {}) or {},
                    finish=stop_reason,
                )
                return RuntimeResult(
                    text="".join(text_parts),
                    structured=structured,
                    cost=cost,
                    finish=stop_reason,
                    raw=response,
                )

            # Dispatch user-tool calls, append toolResult blocks, loop.
            result_blocks: list[dict[str, Any]] = []
            for tool_use_id, name, args in user_tool_uses:
                output, is_error = await self._invoke_tool_with_permission(
                    tool_name=name, tool_args=args
                )
                result_blocks.append(
                    _build_tool_result_block(
                        tool_use_id=tool_use_id, output=output, is_error=is_error
                    )
                )
            self._messages.append({"role": "user", "content": result_blocks})

        raise RuntimeProtocolError(
            f"{self._runtime.label}: tool loop exceeded {MAX_TOOL_ITERATIONS} iterations — "
            f"the model kept requesting tools without producing a final response. This "
            f"usually points to a tool handler returning an output the model can't act on, "
            f"or a system prompt that doesn't tell the model how to stop."
        )

    async def _invoke_tool_with_permission(
        self, *, tool_name: str, tool_args: dict[str, Any]
    ) -> tuple[Any, bool]:
        """Run one tool handler under the optional permission gate.

        Returns ``(output, is_error)``. Permission-deny, unknown tool,
        argument-parse failure, and handler exceptions all surface as
        ``is_error=True`` with a human-readable string output so the
        model can see what happened and recover on its next turn.

        Permission semantics:

        * ``"allow"`` → invoke the handler.
        * ``"deny"`` → skip the handler; return a denial message.
        * ``"defer"`` → log at debug and fall through to allow.
          Bedrock has no native permission fallback policy to defer
          to (unlike Claude / Codex), so silently allowing keeps the
          contract symmetric with the OpenAI-compat path.
        """
        if self._on_permission is not None:
            try:
                decision = await self._on_permission.handle(
                    PermissionRequest(tool_name=tool_name, tool_args=tool_args)
                )
            except Exception as exc:  # noqa: BLE001 — surface to model
                return (
                    f"Permission callback raised {type(exc).__name__}: {exc}; refusing tool call.",
                    True,
                )
            if decision == "deny":
                return (
                    f"Tool {tool_name!r} call was denied by the permission callback.",
                    True,
                )
            if decision == "defer":
                logger.debug(
                    "%s: permission='defer' on tool=%s; Bedrock has no native "
                    "fallback policy — treating as allow.",
                    self._runtime.label,
                    tool_name,
                )
            # "allow" + "defer" → fall through to handler invocation.

        tool = self._tools_by_name.get(tool_name)
        if tool is None:
            return f"Tool {tool_name!r} is not registered on this session.", True
        try:
            params = tool.params.model_validate(tool_args)
        except Exception as exc:  # noqa: BLE001 — Pydantic errors flow back
            return (
                f"Tool arguments did not match the {tool.params.__name__} schema: {exc}",
                True,
            )
        try:
            output = await tool.handler(params)
        except Exception as exc:  # noqa: BLE001 — handler errors flow back
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
        _enforce_budget_gates(max_turns=max_turns, max_budget_usd=max_budget_usd)

        content = _build_user_content(prompt, adapter_label=self._runtime.label)
        thinking_field = _translate_thinking_for_bedrock(
            thinking, model_id=self._model_id, label=self._runtime.label
        )
        pre_len = len(self._messages)
        self._messages.append({"role": "user", "content": content})
        self._stream_cancelled = False

        client = await self._runtime._get_runtime_client()
        full_text: list[str] = []
        captured_submit: dict[str, Any] | None = None
        final_stop: str | None = None
        final_usage: dict[str, Any] = {}

        try:
            for _ in range(MAX_TOOL_ITERATIONS):
                state = _StreamTurnState()
                async for event in self._stream_one_turn(
                    client=client,
                    schema=schema,
                    thinking_field=thinking_field,
                    state=state,
                ):
                    yield event
                if self._stream_cancelled:
                    raise RuntimeCancelledError(f"{self._runtime.label}: cancelled")

                # Snapshot the turn's outputs.
                full_text.extend(state.text_parts)
                if state.submit_input is not None:
                    captured_submit = state.submit_input
                if state.stop_reason is not None:
                    final_stop = state.stop_reason
                if state.usage:
                    final_usage = state.usage

                # Build the assistant message exactly as Converse would
                # have returned it (text + every toolUse) so subsequent
                # turns see history that round-trips back to the API.
                if state.assistant_blocks:
                    self._messages.append({"role": "assistant", "content": state.assistant_blocks})

                if not state.user_tool_uses:
                    # No user tools requested — final turn.
                    structured = None
                    if schema is not None:
                        structured = _validate_tool_payload(
                            captured_submit,
                            schema=schema,
                            label=self._runtime.label,
                        )
                    cost = _build_cost_record(
                        self._runtime.PROVIDER_ID,
                        self._model_id,
                        final_usage,
                        finish=final_stop,
                    )
                    result = RuntimeResult(
                        text="".join(full_text),
                        structured=structured,
                        cost=cost,
                        finish=final_stop,
                    )
                    yield TurnComplete(result=result)
                    return

                # Dispatch user-tool calls, emit events, append toolResult.
                result_blocks: list[dict[str, Any]] = []
                for tool_use_id, name, args in state.user_tool_uses:
                    yield ToolCallStart(
                        tool_name=name,
                        tool_call_id=tool_use_id,
                        arguments_preview=_arguments_preview(args),
                    )
                    output, is_error = await self._invoke_tool_with_permission(
                        tool_name=name, tool_args=args
                    )
                    yield ToolCallResult(
                        tool_call_id=tool_use_id,
                        output=output,
                        is_error=is_error,
                    )
                    result_blocks.append(
                        _build_tool_result_block(
                            tool_use_id=tool_use_id, output=output, is_error=is_error
                        )
                    )
                self._messages.append({"role": "user", "content": result_blocks})
            raise RuntimeProtocolError(
                f"{self._runtime.label}: tool loop exceeded {MAX_TOOL_ITERATIONS} "
                f"iterations during stream() — the model kept requesting tools "
                f"without producing a final response."
            )
        except BaseException:
            # Pop *all* turn-related messages on any failure path so a
            # retry sends a clean history.
            del self._messages[pre_len:]
            raise

    async def _stream_one_turn(
        self,
        *,
        client: Any,
        schema: type[BaseModel] | None,
        thinking_field: dict[str, Any] | None,
        state: _StreamTurnState,
    ) -> AsyncIterator[RuntimeEvent]:
        """Drive one ``converse_stream()`` call and yield deltas.

        Mutates ``state`` with the accumulated text, user-tool uses,
        captured submit_result input, stop reason, and usage. The
        outer :meth:`stream` reads state after each call to decide
        whether to loop or terminate.
        """
        call_kwargs = self._build_call_kwargs(schema=schema, thinking_field=thinking_field)
        try:
            response = await client.converse_stream(**call_kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise _classify_bedrock_error(exc) from exc
        stream_iter = response.get("stream") if isinstance(response, dict) else None
        if stream_iter is None:
            raise RuntimeProtocolError(
                f"{self._runtime.label}: converse_stream returned no 'stream' field"
            )
        self._active_stream = stream_iter

        # Per-block-index buffers — Converse interleaves block deltas
        # but each block carries an explicit index so we can demux.
        text_block_buf: dict[int, list[str]] = {}
        tool_block_buf: dict[int, dict[str, Any]] = {}
        # The most recent block index we saw — used as the fallback
        # when a delta chunk doesn't carry an explicit index.
        current_idx: int = -1
        try:
            async for chunk in stream_iter:
                if not isinstance(chunk, dict):
                    continue
                if "contentBlockStart" in chunk:
                    cbs = chunk["contentBlockStart"]
                    idx = int(cbs.get("contentBlockIndex", -1))
                    current_idx = idx
                    start = cbs.get("start", {}) or {}
                    if "toolUse" in start:
                        tu = start["toolUse"]
                        tool_block_buf[idx] = {
                            "id": tu.get("toolUseId") or "",
                            "name": tu.get("name") or "",
                            "input": "",
                        }
                    else:
                        text_block_buf[idx] = []
                elif "contentBlockDelta" in chunk:
                    cbd = chunk["contentBlockDelta"]
                    idx = int(cbd.get("contentBlockIndex", current_idx))
                    current_idx = idx
                    delta = cbd.get("delta", {}) or {}
                    if "text" in delta:
                        piece = delta["text"]
                        text_block_buf.setdefault(idx, []).append(piece)
                        state.text_parts.append(piece)
                        yield TextDelta(text=piece)
                    elif "reasoningContent" in delta:
                        rc = delta["reasoningContent"]
                        # Skip ``redactedContent`` chunks (no text).
                        if "text" in rc:
                            yield ReasoningDelta(text=rc["text"])
                    elif "toolUse" in delta:
                        td = delta["toolUse"]
                        if "input" in td and idx in tool_block_buf:
                            tool_block_buf[idx]["input"] += td["input"]
                elif "contentBlockStop" in chunk:
                    idx = int(chunk["contentBlockStop"].get("contentBlockIndex", current_idx))
                    if idx in text_block_buf:
                        state.assistant_blocks.append({"text": "".join(text_block_buf.pop(idx))})
                    elif idx in tool_block_buf:
                        tb = tool_block_buf.pop(idx)
                        try:
                            args = json.loads(tb["input"]) if tb["input"] else {}
                        except json.JSONDecodeError:
                            args = {}
                        state.assistant_blocks.append(
                            {
                                "toolUse": {
                                    "toolUseId": tb["id"],
                                    "name": tb["name"],
                                    "input": args,
                                }
                            }
                        )
                        if tb["name"] == SUBMIT_RESULT_TOOL:
                            state.submit_input = args if isinstance(args, dict) else None
                        else:
                            args_dict = args if isinstance(args, dict) else {}
                            state.user_tool_uses.append((tb["id"], tb["name"], args_dict))
                elif "messageStop" in chunk:
                    state.stop_reason = chunk["messageStop"].get("stopReason")
                    # Flush any open text blocks — some Bedrock vendors
                    # omit contentBlockStart/Stop on simple text turns
                    # and only emit deltas. Treat that as one logical
                    # block per accumulated index.
                    for idx in sorted(text_block_buf):
                        chunks_text = text_block_buf.pop(idx)
                        if chunks_text:
                            state.assistant_blocks.append({"text": "".join(chunks_text)})
                elif "metadata" in chunk:
                    metadata = chunk.get("metadata", {})
                    if isinstance(metadata, dict):
                        state.usage = metadata.get("usage", {}) or {}
        except asyncio.CancelledError:
            raise RuntimeCancelledError(f"{self._runtime.label}: cancelled") from None
        except Exception as exc:
            raise _classify_bedrock_error(exc) from exc
        finally:
            self._active_stream = None

    async def cancel(self) -> None:
        # Cooperative cancellation: cancel the in-flight execute task if
        # one is running; otherwise mark the stream so the generator
        # raises on its next yield boundary. Idempotent — calling on an
        # idle session is a no-op.
        task = self._in_flight_task
        if task is not None and not task.done():
            task.cancel()
            return
        if self._active_stream is not None:
            self._stream_cancelled = True
            close = getattr(self._active_stream, "close", None)
            if close is not None:
                try:
                    result = close()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as exc:  # noqa: BLE001 — teardown
                    logger.debug("%s.stream_close_failed error=%s", self._runtime.label, exc)

    async def close(self) -> None:
        # Idempotent + never raises. Drops the messages buffer; the
        # shared runtime client stays alive for sibling sessions.
        self._closed = True
        self._messages.clear()

    def unwrap(self, cls: type[T]) -> T:
        if isinstance(self, cls):
            return self  # type: ignore[return-value]
        raise TypeError(
            f"BedrockSession cannot unwrap to {cls!r}; the aioboto3 "
            f"bedrock-runtime client lives on the runtime — reach it via "
            f"``runtime.unwrap(BedrockRuntimeClient)`` instead."
        )

    # --- Internals ----------------------------------------------------------

    def _build_call_kwargs(
        self,
        *,
        schema: type[BaseModel] | None,
        thinking_field: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Pass a shallow copy so the assistant-message append we do
        # after the call doesn't retroactively mutate what the API saw.
        call_kwargs: dict[str, Any] = {
            "modelId": self._model_id,
            "messages": list(self._messages),
        }
        if self._system:
            call_kwargs["system"] = [{"text": self._system}]
        # toolConfig.tools merges the forced submit_result tool (when
        # schema= is set) with the user-registered FunctionTool entries.
        # Either bucket may be empty.
        tool_specs: list[dict[str, Any]] = []
        if schema is not None:
            tool_specs.append(_submit_result_tool_spec(schema))
        tool_specs.extend(self._tools_wire)
        if tool_specs:
            tool_config: dict[str, Any] = {"tools": tool_specs}
            if schema is not None:
                # Pin toolChoice so the model can't skip submit_result.
                # User tools still surface via the same loop because
                # the model is allowed to call both within one turn.
                tool_config["toolChoice"] = {"tool": {"name": SUBMIT_RESULT_TOOL}}
            call_kwargs["toolConfig"] = tool_config
        if thinking_field is not None:
            call_kwargs["additionalModelRequestFields"] = {"thinking": thinking_field}
        return call_kwargs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _enforce_budget_gates(
    *,
    max_turns: int | None,
    max_budget_usd: float | None,
) -> None:
    """Decline budget kwargs until Iteration E wires the cost table.

    Per the "no silent fallbacks" principle: a capability declined
    must raise, never quietly drop the request.
    """
    if max_turns is not None:
        raise UnsupportedFeatureError(
            "BedrockSession: BUDGET_TURN_CAP lands in Iteration E.",
            feature=Feature.BUDGET_TURN_CAP,
        )
    if max_budget_usd is not None:
        raise UnsupportedFeatureError(
            "BedrockSession: BUDGET_USD_CAP lands in Iteration E.",
            feature=Feature.BUDGET_USD_CAP,
        )


#: Bedrock Converse's recognised image formats (the ``format`` field on
#: the image content block). ``gif`` / ``webp`` honoured on
#: Anthropic-on-Bedrock; others are silently dropped per vendor.
_BEDROCK_IMAGE_FORMATS: frozenset[str] = frozenset({"png", "jpeg", "gif", "webp"})

#: Document formats Converse accepts on the document content block.
#: ``pdf`` is the broad case (Anthropic); the text variants
#: (``txt`` / ``md`` / ``html`` / ``csv``) ride along for the same
#: vendor's native document understanding. Anything outside the set is
#: rejected at translate time rather than surfacing as a vendor 400.
_BEDROCK_DOCUMENT_FORMATS: frozenset[str] = frozenset(
    {"pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md"}
)

#: Magic-number → image format. Used when the caller passes
#: ``ImageInput(bytes_=...)`` without ``media_type=`` so we can pick
#: the right ``format`` string without guessing from the byte stream
#: structure beyond the four-or-twelve-byte header.
_IMAGE_MAGIC_NUMBERS: list[tuple[bytes, str]] = [
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"RIFF", "webp"),  # WebP is RIFF/WEBP; we accept the RIFF prefix
]


def _infer_image_format(*, path: str | None, bytes_: bytes | None) -> str:
    """Return the Bedrock-recognised image ``format`` string.

    Tries the file-extension first when ``path=`` is set, then sniffs
    the bytes magic number. Raises :class:`UnsupportedFeatureError`
    when neither is conclusive — Bedrock requires a literal format,
    so silent fallback would produce a vendor 400.
    """
    if path is not None:
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        if ext == "jpg":
            ext = "jpeg"
        if ext in _BEDROCK_IMAGE_FORMATS:
            return ext
    if bytes_ is not None:
        for magic, fmt in _IMAGE_MAGIC_NUMBERS:
            if bytes_.startswith(magic):
                return fmt
    raise UnsupportedFeatureError(
        "BedrockSession: could not infer image format. Bedrock Converse "
        "needs one of png|jpeg|gif|webp; pass a path with a recognised "
        "extension, or use a different image.",
        feature=Feature.VISION_INPUT,
    )


def _infer_document_format(*, path: str, media_type: str | None) -> str:
    """Return the Bedrock-recognised document ``format`` string."""
    if media_type is not None:
        # Strip the prefix: 'application/pdf' → 'pdf', 'text/markdown' → 'md'.
        if "/" in media_type:
            mt = media_type.split("/", 1)[1].lower()
            mt = {"markdown": "md", "plain": "txt", "html": "html"}.get(mt, mt)
            if mt in _BEDROCK_DOCUMENT_FORMATS:
                return mt
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    ext = {"markdown": "md", "htm": "html", "text": "txt"}.get(ext, ext)
    if ext in _BEDROCK_DOCUMENT_FORMATS:
        return ext
    raise UnsupportedFeatureError(
        f"BedrockSession: unrecognised document format for {path!r}. "
        f"Bedrock Converse accepts: {sorted(_BEDROCK_DOCUMENT_FORMATS)}.",
        feature=Feature.FILE_INPUT,
    )


def _document_name_from_path(path: str) -> str:
    """Build a Converse-legal ``document.name`` from a filesystem path.

    Bedrock requires the name to be alphanumeric + whitespace + a small
    set of punctuation; underscores and hyphens get through. Returns a
    sanitised basename without the extension.
    """
    base = os.path.basename(path)
    base = os.path.splitext(base)[0]
    sanitised = "".join(c if c.isalnum() or c in " _-" else "_" for c in base) or "document"
    return sanitised[:64]


def _build_image_block(img: ImageInput) -> dict[str, Any]:
    """Translate one :class:`ImageInput` to a Converse image content block."""
    if img.url is not None and img.bytes_ is None and img.path is None:
        raise UnsupportedFeatureError(
            "BedrockSession: ImageInput(url=) is not supported — Bedrock "
            "Converse needs the bytes locally. Pass path= or bytes_=.",
            feature=Feature.VISION_INPUT,
        )
    if img.bytes_ is not None:
        data = img.bytes_
    elif img.path is not None:
        with open(img.path, "rb") as fh:
            data = fh.read()
    else:  # pragma: no cover — guarded by ImageInput.__post_init__
        raise UnsupportedFeatureError(
            "BedrockSession: ImageInput must set path= or bytes_=.",
            feature=Feature.VISION_INPUT,
        )
    fmt = _infer_image_format(path=img.path, bytes_=data)
    return {"image": {"format": fmt, "source": {"bytes": data}}}


def _build_document_block(file: FileInput) -> dict[str, Any]:
    """Translate one :class:`FileInput` to a Converse document content block."""
    with open(file.path, "rb") as fh:
        data = fh.read()
    fmt = _infer_document_format(path=file.path, media_type=file.media_type)
    return {
        "document": {
            "format": fmt,
            "name": _document_name_from_path(file.path),
            "source": {"bytes": data},
        }
    }


def _build_user_content(prompt: Prompt, *, adapter_label: str) -> list[dict[str, Any]]:
    """Build the Converse ``content`` list for a user message.

    Routes the prompt through :func:`_split_prompt_parts` and turns
    each :class:`ImageInput` / :class:`FileInput` into its native
    Converse content block. The text portion lands as a single
    leading ``{"text": ...}`` block (Converse uses one block per
    text segment but the conventional shape is one combined text
    block followed by attachments).
    """
    text, images, files = _split_prompt_parts(
        prompt,
        adapter_label=adapter_label,
        supports_vision=True,
        supports_file=True,
    )
    blocks: list[dict[str, Any]] = []
    if text:
        blocks.append({"text": text})
    for img in images:
        blocks.append(_build_image_block(img))
    for file in files:
        blocks.append(_build_document_block(file))
    if not blocks:
        # The model spec wants at least one block; default to an
        # empty-string text block so the wire shape stays valid.
        blocks.append({"text": ""})
    return blocks


#: ``thinking`` effort → budget-tokens mapping for Anthropic-on-Bedrock.
#: Mirrors :class:`ClaudeCodeRuntime` to keep the airframe portable
#: surface stable across both routes to Claude.
_BEDROCK_THINKING_BUDGETS: dict[str, int] = {
    "low": 1024,
    "medium": 8192,
    "high": 32768,
}


def _is_anthropic_on_bedrock(model_id: str) -> bool:
    """Return ``True`` when the model id targets an Anthropic Claude variant.

    Covers raw IDs (``anthropic.claude-...``) and region-prefixed
    inference profiles (``us.anthropic.claude-...``). PT ARNs are
    opaque to vendor identity here — they pass through with thinking
    silently dropped (the field rides ``additionalModelRequestFields``
    which Bedrock per-vendor validates).
    """
    head = model_id.split(":", 1)[0].lower()
    return ".anthropic." in f".{head}." or head.startswith("anthropic.")


def _translate_thinking_for_bedrock(
    thinking: ThinkingMode,
    *,
    model_id: str,
    label: str,
) -> dict[str, Any] | None:
    """Translate :data:`ThinkingMode` into the Converse thinking field.

    The result lands in ``additionalModelRequestFields={"thinking":
    <returned>}`` when non-None. ``None`` skips the field entirely.

    Mappings:

    * ``None`` → ``None`` (no override).
    * ``"disabled"`` → ``None`` (omit; Bedrock has no explicit
      disable shape — sending nothing is the disable).
    * ``"minimal"`` → ``{"type": "enabled", "budget_tokens": 1024}``
      with a debug log (Anthropic has no "minimal" tier — coerced
      to "low" same as :class:`ClaudeCodeRuntime`).
    * ``"low" | "medium" | "high"`` → ``{"type": "enabled",
      "budget_tokens": 1024|8192|32768}``.
    * ``{"budget_tokens": N}`` → ``{"type": "enabled",
      "budget_tokens": N}``.

    Non-Anthropic models receive ``None`` with a debug log — Bedrock
    silently ignores ``additionalModelRequestFields`` keys the
    vendor doesn't understand, so sending the field anyway is a
    no-op, but we save the extra bytes.
    """
    if thinking is None:
        return None
    if thinking == "disabled":
        return None
    if isinstance(thinking, str):
        if thinking == "minimal":
            logger.debug(
                "%s: thinking='minimal' has no Anthropic equivalent; coercing to 'low'",
                label,
            )
            budget = _BEDROCK_THINKING_BUDGETS["low"]
        elif thinking in _BEDROCK_THINKING_BUDGETS:
            budget = _BEDROCK_THINKING_BUDGETS[thinking]
        else:
            raise UnsupportedFeatureError(
                f"{label}: unrecognised thinking effort {thinking!r}; "
                f"supported: 'minimal' (→'low'), 'low', 'medium', 'high', 'disabled'.",
                feature=Feature.REASONING_EFFORT,
            )
    elif isinstance(thinking, dict):
        raw_budget = thinking.get("budget_tokens")
        if raw_budget is None or not isinstance(raw_budget, int):
            raise UnsupportedFeatureError(
                f"{label}: dict-shaped thinking must include integer "
                f"'budget_tokens'; got keys={list(thinking)}",
                feature=Feature.REASONING_BUDGET_TOKENS,
            )
        budget = int(raw_budget)
    else:
        raise UnsupportedFeatureError(
            f"{label}: unrecognised thinking mode {thinking!r}",
            feature=Feature.REASONING_EFFORT,
        )

    if not _is_anthropic_on_bedrock(model_id):
        logger.debug(
            "%s: thinking= forwarded to non-Anthropic model %s; "
            "Bedrock will silently ignore additionalModelRequestFields "
            "this vendor doesn't honour.",
            label,
            model_id,
        )
        return None
    return {"type": "enabled", "budget_tokens": budget}


def _submit_result_tool_spec(schema: type[BaseModel]) -> dict[str, Any]:
    """Build the single ``toolSpec`` entry that gates structured output.

    Returned dict shape: ``{"toolSpec": {"name", "description",
    "inputSchema": {"json": ...}}}`` — ready to drop into the
    ``toolConfig.tools`` list. The :class:`BedrockSession` caller
    pins ``toolChoice`` to force this tool when ``schema=`` is set.
    """
    return {
        "toolSpec": {
            "name": SUBMIT_RESULT_TOOL,
            "description": (
                f"Submit the final typed payload as a {schema.__name__}. "
                "Call this exactly once with all required fields filled in."
            ),
            "inputSchema": {"json": schema.model_json_schema()},
        }
    }


def _build_submit_result_tool_config(schema: type[BaseModel]) -> dict[str, Any]:
    """Build a stand-alone ``toolConfig`` carrying only the forced submit_result tool.

    Retained as a convenience for tests / consumers that want the
    structured-output ``toolConfig`` without going through a
    :class:`BedrockSession`. The session itself uses
    :func:`_submit_result_tool_spec` directly so user tools can ride
    in the same ``toolConfig.tools`` list.
    """
    return {
        "tools": [_submit_result_tool_spec(schema)],
        "toolChoice": {"tool": {"name": SUBMIT_RESULT_TOOL}},
    }


def _translate_one_tool_for_bedrock(tool: FunctionTool) -> dict[str, Any]:
    """Translate one :class:`FunctionTool` into a Converse ``toolSpec`` entry.

    Bedrock's tool API mirrors Anthropic's: name, description, and a
    JSON Schema describing the call arguments. The Pydantic schema
    serialised via ``model_json_schema()`` round-trips cleanly.
    """
    return {
        "toolSpec": {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": {"json": tool.params.model_json_schema()},
        }
    }


def _require_output_message(response: Any, *, label: str) -> dict[str, Any]:
    """Defensive accessor for ``response.output.message`` shape.

    Bedrock's wire shape is stable, but defending here means a
    malformed response surfaces as :class:`RuntimeProtocolError`
    rather than a confusing :class:`AttributeError`.
    """
    if not isinstance(response, dict):
        raise RuntimeProtocolError(f"{label}: converse response is not a dict")
    output_msg = response.get("output", {}).get("message")
    if not isinstance(output_msg, dict):
        raise RuntimeProtocolError(f"{label}: converse response missing output.message")
    return output_msg


def _split_assistant_blocks(
    output_msg: dict[str, Any],
    *,
    user_tool_names: set[str],
) -> tuple[list[str], list[tuple[str, str, dict[str, Any]]], dict[str, Any] | None]:
    """Pull text + tool uses out of a Converse assistant message.

    Returns ``(text_parts, user_tool_uses, submit_input)``:

    * ``text_parts`` — every ``{"text": ...}`` block's content, in
      order. Caller ``"".join``-s for the final text.
    * ``user_tool_uses`` — list of ``(toolUseId, name, input)`` for
      every ``{"toolUse": ...}`` block whose name appears in
      ``user_tool_names``. Unknown tools surface here too (with the
      same shape) so the loop can return a tool-not-registered
      error message back to the model rather than silently dropping
      the request.
    * ``submit_input`` — the ``input`` dict from the
      ``submit_result`` ``toolUse`` block when present, else
      ``None``.
    """
    text_parts: list[str] = []
    user_uses: list[tuple[str, str, dict[str, Any]]] = []
    submit_input: dict[str, Any] | None = None
    for block in output_msg.get("content", []) or []:
        if not isinstance(block, dict):
            continue
        if "text" in block:
            text_parts.append(block["text"])
        elif "toolUse" in block:
            tu = block["toolUse"]
            if not isinstance(tu, dict):
                continue
            name = tu.get("name")
            if name == SUBMIT_RESULT_TOOL:
                submit_input = tu.get("input")
            else:
                use_id = tu.get("toolUseId") or ""
                args = tu.get("input") or {}
                if not isinstance(args, dict):
                    args = {}
                user_uses.append((use_id, str(name or ""), args))
                # ``user_tool_names`` is not used to filter — unknown
                # tool calls still flow through so the loop can return
                # a "tool not registered" error to the model. The
                # parameter is reserved for future per-call routing
                # decisions (e.g. emitting different events for known
                # vs unknown tools).
    del user_tool_names  # accepted in the signature for future use
    return text_parts, user_uses, submit_input


def _arguments_preview(args: dict[str, Any], *, max_len: int = 200) -> str:
    """Render ``args`` as a short JSON string for :class:`ToolCallStart`.

    Bedrock streams tool input as JSON chunks; we surface the parsed
    dict back as a compact JSON snippet so consumers see a readable
    preview. Truncated at ``max_len`` to keep event payloads tight.
    """
    try:
        text = json.dumps(args, default=str)
    except (TypeError, ValueError):
        text = str(args)
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _build_tool_result_block(*, tool_use_id: str, output: Any, is_error: bool) -> dict[str, Any]:
    """Build one Converse ``{"toolResult": {...}}`` content block.

    ``output`` becomes a single content entry under the toolResult:
    dict / list outputs ride a ``{"json": ...}`` slot; everything
    else is coerced to a string under ``{"text": ...}``. ``status``
    is ``"error"`` when ``is_error`` is True so the model sees the
    error semantics rather than treating the message as a
    successful result.
    """
    entry: dict[str, Any]
    if isinstance(output, dict | list):
        entry = {"json": output}
    else:
        entry = {"text": str(output)}
    return {
        "toolResult": {
            "toolUseId": tool_use_id,
            "content": [entry],
            "status": "error" if is_error else "success",
        }
    }


def _validate_tool_payload(
    payload: dict[str, Any] | None,
    *,
    schema: type[BaseModel],
    label: str,
) -> dict[str, Any]:
    """Validate the captured ``submit_result`` arguments against ``schema``."""
    if payload is None:
        raise RuntimeStructuredOutputError(
            f"{label}: model did not call the {SUBMIT_RESULT_TOOL} tool; "
            f"no structured payload available."
        )
    try:
        instance = schema.model_validate(payload)
    except ValidationError as exc:
        raise RuntimeStructuredOutputError(
            f"{label}: {SUBMIT_RESULT_TOOL} payload did not validate against "
            f"{schema.__name__}: {exc}",
            body=payload,
        ) from exc
    return instance.model_dump()


def _build_cost_record(
    provider_id: str,
    model_id: str,
    usage: dict[str, Any],
    *,
    finish: str | None,
) -> CostRecord:
    """Build a :class:`CostRecord` from Converse ``usage``.

    Iteration B leaves ``cost_usd=None`` — the pricing table arrives
    in Iteration E alongside ``BUDGET_USD_CAP``. Token counters are
    populated regardless so structured logs are useful immediately.
    """
    return CostRecord(
        provider_id=provider_id,
        model_id=model_id,
        cost_usd=None,
        input_tokens=int(usage.get("inputTokens") or 0),
        output_tokens=int(usage.get("outputTokens") or 0),
        cache_read_tokens=int(usage.get("cacheReadInputTokens") or 0),
        cache_write_tokens=int(usage.get("cacheWriteInputTokens") or 0),
        finish=finish,
    )


def _infer_capabilities(entry: dict[str, Any]) -> frozenset[str]:
    """Derive airframe capability flags from a Bedrock model summary."""
    caps: set[str] = {CAPABILITY_STRUCTURED_OUTPUT, CAPABILITY_TOOLS}
    if entry.get("responseStreamingSupported"):
        caps.add(CAPABILITY_STREAMING)
    input_modalities = entry.get("inputModalities") or []
    if isinstance(input_modalities, list) and "IMAGE" in input_modalities:
        caps.add(CAPABILITY_VISION)
    return frozenset(caps)


def _classify_bedrock_error(exc: Exception) -> Exception:
    """Map a boto3 / aioboto3 / aiohttp exception onto airframe's error hierarchy.

    Covers both the catalog endpoint (``list_foundation_models``) and
    the model-invocation endpoint (``converse`` / ``converse_stream``).
    The execute-path codes (``ValidationException`` for unknown models,
    ``ThrottlingException`` for rate-limits, etc.) are honoured here.
    """
    # Late-import so we never force botocore at module-import time.
    try:
        from botocore.exceptions import (
            ClientError,
            EndpointConnectionError,
            NoCredentialsError,
            NoRegionError,
            PartialCredentialsError,
            ReadTimeoutError,
        )
    except ImportError:
        return exc

    if isinstance(exc, NoCredentialsError | PartialCredentialsError):
        return RuntimeAuthError(f"bedrock: no usable AWS credentials: {exc}")
    if isinstance(exc, NoRegionError):
        return RuntimeAuthError(
            "bedrock: no AWS region resolved. Set AWS_REGION (or pass "
            f"region_name=) — Bedrock is region-pinned. Underlying: {exc}"
        )
    if isinstance(exc, ClientError):
        code = ""
        status: int | None = None
        message = ""
        try:
            err = exc.response.get("Error", {})  # type: ignore[attr-defined]
            code = err.get("Code", "") or ""
            message = err.get("Message", "") or ""
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
        if code == "ResourceNotFoundException":
            return RuntimeModelNotFoundError(f"bedrock: resource not found: {exc}", status=status)
        if code == "ValidationException":
            lower = message.lower()
            if "context" in lower or "token" in lower and "limit" in lower:
                return RuntimeContextOverflowError(
                    f"bedrock: context window exceeded: {exc}", status=status
                )
            if "model" in lower:
                return RuntimeModelNotFoundError(
                    f"bedrock: model not available: {exc}", status=status
                )
            return RuntimeProtocolError(f"bedrock: validation error: {exc}", status=status)
        if code == "ModelStreamErrorException":
            return RuntimeTransientError(f"bedrock: stream error: {exc}", status=status)
        if isinstance(status, int) and 500 <= status < 600:
            return RuntimeTransientError(f"bedrock: transient {status}: {exc}", status=status)
        return RuntimeProtocolError(f"bedrock: {code or 'ClientError'}: {exc}", status=status)
    if isinstance(exc, EndpointConnectionError | ReadTimeoutError):
        return RuntimeTransientError(f"bedrock: network: {exc}")
    # aiohttp.ClientError is the typical transport failure under aioboto3.
    # Late-import: aiohttp may not be installed in every consumer's env.
    try:
        import aiohttp
    except ImportError:
        return exc
    if isinstance(exc, aiohttp.ClientError):
        return RuntimeTransientError(f"bedrock: network: {exc}")
    return exc


__all__ = [
    "DEFAULT_BEDROCK_MODEL",
    "SUBMIT_RESULT_TOOL",
    "BedrockRuntime",
    "BedrockSession",
]
