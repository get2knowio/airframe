"""``OpenCodeServerRuntime`` — :class:`AgentRuntime` over the OpenCode HTTP agent server.

Wraps the official ``opencode-ai`` Stainless-generated Python SDK
(client for ``sst/opencode``'s bespoke REST + SSE server). Distinct
from :class:`~airframe.adapters.opencode_zen.OpenCodeZenRuntime` and
:class:`~airframe.adapters.opencode_go.OpenCodeGoRuntime`: those two
are OpenAI-compat gateways billed per-token / by subscription; this
adapter targets the **agent server** that ``opencode serve`` runs
locally (or that a user has stood up on a remote host).

The agent server is **model-agnostic** — it fronts whatever upstream
providers ``opencode auth login`` has configured (Anthropic, OpenAI
incl. ChatGPT-OAuth subscriptions, OpenRouter, Ollama, vLLM,
llama.cpp, Together, Groq, …). Wrapping it gives airframe a single
agent loop that delivers MCP, permission gating, lifecycle hooks,
and SSE streaming across open-weight and subscription model houses.

**Auth.** HTTP Basic over the OpenCode server. Two slots: explicit
constructor args, then ``OPENCODE_SERVER_USERNAME`` /
``OPENCODE_SERVER_PASSWORD`` env vars. Loopback (``127.0.0.1``,
``localhost``, ``::1``) is allowed unauthenticated since OpenCode's
default ``serve`` posture is unauthenticated-on-localhost; remote
URLs without credentials raise :class:`RuntimeAuthError` at
``__init__()`` to keep an accidental ``opencode serve --hostname
0.0.0.0`` from becoming a remote-bash endpoint.

**Iteration A** lands discovery, capability predicates,
``validate_binding``, lazy SDK import, auth chain, and ``list_models``;
every :class:`Feature` flag starts False.

**Iteration B** flips ``STREAMING`` / ``CANCEL`` / ``SESSION_RESUME``
True by wiring :class:`OpenCodeServerSession` against the live SDK.

**Iteration C** flips ``VISION_INPUT`` / ``FILE_INPUT`` /
``REASONING_EFFORT`` / ``REASONING_BUDGET_TOKENS`` True:

* Polymorphic prompts — :class:`~airframe.inputs.ImageInput` and
  :class:`~airframe.inputs.FileInput` parts translate to OpenCode's
  ``FilePartInputParam`` (``{"type": "file", "mime": "...", "url":
  "..."}``). URL variants pass through; ``bytes_`` and ``path``
  variants encode as ``data:`` URLs so the server can fetch them
  out-of-band without filesystem access to the adapter's host.
* ``thinking=`` pass-through — the agent server does not normalise
  reasoning across upstreams, so the adapter dispatches per
  upstream ``provider_id``: Anthropic gets the
  ``{"thinking": {"type": "enabled", "budget_tokens": N}}``
  envelope, everyone else gets the OpenAI-Chat-Completions
  ``{"reasoning_effort": "..."}`` shape (the most widely-honoured
  envelope across open-weight backends). Sent via the SDK's
  ``extra_body`` pass-through — the SDK has no first-class slot.

``STRUCTURED_OUTPUT_JSON_SCHEMA`` stays False — the 0.1.0a36 SDK
doesn't expose a ``client.mcp`` resource, so the forced-tool shim
the original plan called for has to wait for Iteration D.

**Iteration D** wires :attr:`OpenCodeServerOptions.available_tools`
and :attr:`OpenCodeServerOptions.excluded_tools` through to the
SDK's ``session.chat(tools={"bash": True, "read": False, ...})``
allow/denylist for OpenCode's *built-in* tools.

Per the plan's "Path 2 (fallback)", the rest of D's scope —
``tools=[FunctionTool(...)]``, ``mcp_servers=[McpServerRef(...)]``,
``on_permission=PermissionCallback`` — remains declined. The
opencode-ai 0.1.0a36 SDK does not surface either of:

* an **MCP-runtime-registration** endpoint (only the static
  ``opencode.json`` config-file types ``McpLocalConfig`` /
  ``McpRemoteConfig`` exist); and
* a **permission-reply** endpoint (the SDK emits
  ``permission.updated`` events but has no
  ``client.permission.reply()``).

OpenCode the *server* supports both surfaces; the limitation is
purely the Stainless-generated Python SDK's coverage. Wrapping
raw HTTP routes the SDK doesn't surface would bypass airframe's
"wrap vendor SDKs, don't rewrite them" principle. A later
iteration flips ``TOOLS_FUNCTION`` / ``TOOLS_MCP_*`` /
``PERMISSION_CALLBACK`` once the SDK catches up. Until then,
consumers wanting MCP can pre-register servers via
``opencode.json`` and either restrict which built-ins the model
may invoke via :attr:`OpenCodeServerOptions.excluded_tools`.

**Iteration E** flips ``LIFECYCLE_HOOKS`` / ``BUDGET_USD_CAP`` /
``BUDGET_TURN_CAP`` True. The session emits six of airframe's eight
hook kinds:

* ``session_start`` — when the server-side session id is first
  established (server create or resume).
* ``session_end`` — at :meth:`OpenCodeServerSession.close` boundary.
* ``user_prompt_submit`` — at the top of each
  :meth:`execute` / :meth:`stream`.
* ``pre_tool_use`` / ``post_tool_use`` / ``tool_failure`` —
  synthesised from the ``ToolPart`` lifecycle on the SSE bus.

Not emitted: ``pre_compact`` (the SDK doesn't surface a compaction
event distinct from ``session.idle``) and ``rate_limit`` (OpenCode
raises upstream 429s as ``session.error`` events that the adapter
classifies as :class:`~airframe.errors.RuntimeTransientError`).

Budget caps use the shared
:func:`~airframe.sessions._enforce_budget_pre_turn` helper.
``max_budget_usd`` enforcement depends on the underlying upstream
reporting cost — when ``AssistantMessage.cost`` is absent (some
self-hosted Ollama / llama.cpp deployments) ``cost_usd`` stays
``None`` and the cap can't be enforced; we log at debug and let the
turn through.

Subsequent iterations:

* F — :class:`OpenCodeServerOptions` final wiring; conformance +
  integration suite; docs.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar
from urllib.parse import urlparse

from airframe.cache import CacheConfig
from airframe.cost import CostRecord
from airframe.errors import (
    RuntimeAuthError,
    RuntimeCancelledError,
    RuntimeProtocolError,
    RuntimeServerStartError,
    RuntimeTransientError,
    UnsupportedFeatureError,
)
from airframe.events import (
    ReasoningDelta,
    TextDelta,
    ToolCallResult,
    ToolCallStart,
    TurnComplete,
)
from airframe.features import Feature
from airframe.metadata import RequestMetadata
from airframe.models import ModelInfo
from airframe.options import OpenCodeServerOptions
from airframe.protocol import (
    AgentRuntime,
    AgentSession,
    ProviderModel,
    RuntimeResult,
)
from airframe.sessions import (
    _check_budget_supported,
    _check_hooks_supported,
    _check_provider_options,
    _enforce_budget_pre_turn,
    _fire_hook_event,
    _resolve_native_tools,
    _split_prompt_parts,
)
from airframe.slash_commands import SlashCommand, SlashCommandsConfig

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from pydantic import BaseModel

    from airframe.events import RuntimeEvent
    from airframe.hooks import HookEvent
    from airframe.inputs import Prompt
    from airframe.native_tools import NativeCapability, NativeTool
    from airframe.options import ProviderOptions
    from airframe.permission import PermissionCallback
    from airframe.thinking import ThinkingMode
    from airframe.tools import FunctionTool, McpServerRef

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Default OpenCode server URL. Matches the documented default for
#: ``opencode serve`` (``127.0.0.1:4096``). Overridable via constructor
#: or :envvar:`OPENCODE_SERVER_URL`.
DEFAULT_BASE_URL = "http://127.0.0.1:4096"

#: Default Basic-auth username when only a password env var is set —
#: matches the OpenCode server's documented default.
DEFAULT_USERNAME = "opencode"

#: Hostnames that count as loopback for the no-auth guardrail.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _is_loopback(base_url: str) -> bool:
    """``True`` when ``base_url`` resolves to a loopback hostname.

    Strict string match against the host portion of the URL: we do
    not perform DNS resolution (that would couple ``__init__()`` to
    network state and silently bless a hostname that resolves to a
    loopback today but might not tomorrow). The conservative posture
    is the correct one — the only hostnames a user-typed config can
    intentionally make loopback are the three literals.
    """
    try:
        host = (urlparse(base_url).hostname or "").lower()
    except ValueError:
        return False
    return host in _LOOPBACK_HOSTS


class OpenCodeServerRuntime(AgentRuntime):
    """``AgentRuntime`` over the OpenCode HTTP agent server.

    Args:
        model: Default model identifier used when ``execute()`` is
            called without a :class:`ProviderModel` override. The
            server's available models depend on which upstream
            providers it has configured; ``None`` defers the choice to
            the server's own routing. Honours
            :envvar:`OPENCODE_SERVER_DEFAULT_MODEL` if set.
        base_url: OpenCode server URL. Resolution order: explicit arg
            → :envvar:`OPENCODE_SERVER_URL` → :data:`DEFAULT_BASE_URL`.
        username: HTTP Basic username. Resolution order: explicit arg
            → :envvar:`OPENCODE_SERVER_USERNAME` → :data:`DEFAULT_USERNAME`
            when a password is present.
        password: HTTP Basic password. Resolution order: explicit arg
            → :envvar:`OPENCODE_SERVER_PASSWORD` → unset (loopback only).
        timeout: Per-request timeout forwarded to the SDK. Default
            600s mirrors the other adapters.

    Raises:
        RuntimeAuthError: When the resolved ``base_url`` is non-loopback
            and no Basic-auth credentials resolved. The guardrail
            blocks an accidental ``opencode serve --hostname 0.0.0.0``
            from being treated as an unauthenticated endpoint.
    """

    label = "opencode_server"

    #: Canonical provider ID this adapter serves. Distinct from
    #: ``"opencode-zen"`` and ``"opencode-go"`` — the two OpenAI-compat
    #: gateway adapters that share the OpenCode brand.
    PROVIDER_ID: ClassVar[str] = "opencode"

    #: Vendor SDK that must be importable for this adapter to work.
    #: The Stainless-generated client is published as ``opencode-ai``
    #: (PyPI distribution name) which imports as ``opencode_ai``.
    REQUIRES_PACKAGE: ClassVar[str] = "opencode_ai"

    #: pip extra that brings the vendor SDK in.
    EXTRA_NAME: ClassVar[str] = "opencode"

    #: Iteration E adds LIFECYCLE_HOOKS / BUDGET_USD_CAP /
    #: BUDGET_TURN_CAP on top of C's set. TOOLS_FUNCTION / MCP_* /
    #: PERMISSION_CALLBACK stay False — see Iteration D docstring for
    #: the SDK-gap rationale.
    SUPPORTED_FEATURES: ClassVar[frozenset[Feature]] = frozenset(
        {
            Feature.STREAMING,
            Feature.CANCEL,
            Feature.SESSION_RESUME,
            Feature.VISION_INPUT,
            Feature.FILE_INPUT,
            Feature.REASONING_EFFORT,
            Feature.REASONING_BUDGET_TOKENS,
            Feature.LIFECYCLE_HOOKS,
            Feature.BUDGET_USD_CAP,
            Feature.BUDGET_TURN_CAP,
            Feature.SLASH_COMMANDS,
        }
    )

    #: Six of airframe's eight hook kinds. ``pre_compact`` and
    #: ``rate_limit`` don't have distinct SDK events to map from.
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

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        username: str | None = None,
        password: str | None = None,
        timeout: float = 600.0,
    ) -> None:
        self._default_model = model or os.environ.get("OPENCODE_SERVER_DEFAULT_MODEL")
        self._base_url = self._resolve_base_url(base_url)
        self._username, self._password = self._resolve_auth(username, password)
        self._timeout = timeout
        # Lazy AsyncOpencode client; built on first list_models() /
        # execute() / stream() call so plain construction never touches
        # the network.
        self._client: Any = None
        self._closed = False

    # --- Auth + URL resolution ---------------------------------------------

    @staticmethod
    def _resolve_base_url(base_url: str | None) -> str:
        """Resolve the OpenCode server URL.

        Order: explicit arg → :envvar:`OPENCODE_SERVER_URL` →
        :data:`DEFAULT_BASE_URL`.
        """
        return base_url or os.environ.get("OPENCODE_SERVER_URL") or DEFAULT_BASE_URL

    def _resolve_auth(
        self, username: str | None, password: str | None
    ) -> tuple[str | None, str | None]:
        """Resolve HTTP Basic credentials.

        Order: explicit args → env vars → unset (loopback only).
        When the resolved ``base_url`` is non-loopback and no
        password resolved, raises :class:`RuntimeAuthError` — the
        guardrail that prevents an accidental remote-bash endpoint.
        """
        resolved_password = password or os.environ.get("OPENCODE_SERVER_PASSWORD") or None
        # Username defaults to the OpenCode server's documented default
        # ("opencode") only when *some* password resolved — otherwise
        # leave both None so the guardrail / loopback check fires
        # against the right state.
        if resolved_password is None:
            resolved_username = username or os.environ.get("OPENCODE_SERVER_USERNAME") or None
        else:
            resolved_username = (
                username or os.environ.get("OPENCODE_SERVER_USERNAME") or DEFAULT_USERNAME
            )

        if resolved_password is None and not _is_loopback(self._base_url):
            raise RuntimeAuthError(
                f"OpenCodeServerRuntime: server at {self._base_url!r} is not "
                "loopback and no Basic-auth credentials resolved. Set "
                "OPENCODE_SERVER_PASSWORD (and OPENCODE_SERVER_USERNAME if "
                f"non-default {DEFAULT_USERNAME!r}), or pass username= / "
                "password= explicitly. Loopback (127.0.0.1, localhost, ::1) "
                "is the only base_url accepted without credentials."
            )
        return resolved_username, resolved_password

    def _ensure_client(self) -> Any:
        """Build the :class:`AsyncOpencode` client lazily.

        Raises :class:`ImportError` when the ``opencode-ai`` extra
        isn't installed; the message names the extra so users know
        what to ``pip install``.
        """
        if self._client is not None:
            return self._client
        try:
            from opencode_ai import AsyncOpencode
        except ImportError as exc:
            raise ImportError(
                "OpenCodeServerRuntime requires the 'opencode-ai' package. "
                "Install with: pip install airframe-agents[opencode]"
            ) from exc

        kwargs: dict[str, Any] = {
            "base_url": self._base_url,
            "timeout": self._timeout,
        }
        # AsyncOpencode 0.1.0a36 doesn't expose dedicated username= /
        # password= kwargs — we inject Basic credentials via
        # default_headers so every request carries the Authorization
        # header. Only forward when both halves resolved; loopback
        # without credentials is the documented unauthenticated path.
        if self._username is not None and self._password is not None:
            token = base64.b64encode(f"{self._username}:{self._password}".encode()).decode("ascii")
            kwargs["default_headers"] = {"Authorization": f"Basic {token}"}
        self._client = AsyncOpencode(**kwargs)
        return self._client

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
        # Documented sugar for ``runtime.session(...).execute(...) + close()``.
        # Iteration A's session raises ``UnsupportedFeatureError`` on
        # ``execute()`` since no behaviour is wired yet — Iteration B
        # flips the matching Feature flags and fills the session methods.
        del persona  # accepted on the protocol but not consumed here
        sess = self.session(system=system, model=model, metadata=metadata, cache=cache)
        try:
            return await sess.execute(prompt, schema=schema, thinking=thinking, timeout=timeout)
        finally:
            await sess.close()

    async def reset(self) -> None:
        # Sessionless runtime — the per-conversation session_id lives
        # on OpenCodeServerSession (Iteration B+). Nothing scope-bound
        # to drop here; the shared HTTP client keeps serving siblings.
        return None

    async def close(self) -> None:
        # Idempotent + never raises (runs from finally / __aexit__).
        # Tears down the lazily-built AsyncOpencode HTTP client if any.
        client = self._client
        self._client = None
        self._closed = True
        if client is None:
            return
        try:
            await client.close()
        except Exception as exc:  # noqa: BLE001 — teardown never raises
            logger.debug("opencode_server.client_teardown_failed error=%s", exc)

    def validate_binding(self, binding: ProviderModel) -> bool:
        # The agent server fronts whatever upstream providers it has
        # been configured for. We can't validate a model_id without
        # hitting the server, and a prefix allowlist would lock out
        # every new model the user wires upstream. Accept any non-
        # empty model_id when the provider_id matches.
        return binding.provider_id == self.PROVIDER_ID and bool(binding.model_id)

    def supports(self, feature: Feature, model: ProviderModel | None = None) -> bool:
        return feature in self.SUPPORTED_FEATURES

    def supported_native_tools(
        self, model: ProviderModel | None = None
    ) -> frozenset[NativeCapability]:
        # OpenCode's server runs built-in websearch/webfetch tools, exposed
        # through OpenCodeServerOptions(available_tools=...). Surfacing them as
        # portable native_tools is a follow-up; declined for now.
        return frozenset()

    def unwrap(self, cls: type[T]) -> T:
        if isinstance(self, cls):
            return self  # type: ignore[return-value]
        # If a live client exists and is type-compatible with cls,
        # hand it back. ``isinstance`` is the right check (rather
        # than ``cls is AsyncOpencode``) — it also covers subclasses
        # and survives test monkey-patching that replaces the
        # AsyncOpencode symbol mid-test.
        if self._client is not None and isinstance(self._client, cls):
            return self._client  # type: ignore[return-value]
        # Late-import to keep `import airframe` from pulling opencode-ai.
        try:
            from opencode_ai import AsyncOpencode
        except ImportError:  # pragma: no cover — unwrap with SDK uninstalled
            raise TypeError(
                f"OpenCodeServerRuntime cannot unwrap to {cls!r} without the "
                "'opencode-ai' extra installed."
            ) from None
        wants_async_client = isinstance(cls, type) and (
            cls is AsyncOpencode or issubclass(cls, AsyncOpencode)
        )
        if wants_async_client:
            raise TypeError(
                "OpenCodeServerRuntime: no AsyncOpencode client exists yet — "
                "call list_models() / execute() / stream() first to build it."
            )
        raise TypeError(
            f"OpenCodeServerRuntime cannot unwrap to {cls!r}; only "
            "OpenCodeServerRuntime and AsyncOpencode (after first call) are "
            "supported on the runtime. Vendor session objects live on "
            "AgentSession — use session.unwrap(NativeType)."
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
        """Open a session.

        Iteration A returns a :class:`~airframe.sessions._ThinAgentSession`
        placeholder. Iteration B replaces this with the bespoke
        :class:`OpenCodeServerSession` that owns a server-issued
        ``session_id``, drives the SSE bus, and honours ``resume=``.
        """
        from airframe.errors import UnsupportedFeatureError

        if tools:
            raise UnsupportedFeatureError(
                "OpenCodeServerRuntime: caller-defined `tools=[FunctionTool(...)]` "
                "cannot be wired against the opencode-ai 0.1.0a36 SDK — the SDK "
                "doesn't surface MCP-runtime-registration endpoints. Use OpenCode's "
                "built-in tools (allow/denylist via "
                "`OpenCodeServerOptions(available_tools=..., excluded_tools=...)`), "
                "or pre-register an MCP server in `opencode.json` so the model can "
                "invoke it server-side. Will flip True once SDK exposes MCP.",
                feature=Feature.TOOLS_FUNCTION,
            )
        _resolve_native_tools(
            native_tools,
            adapter_label=self.label,
            provider_id=self.PROVIDER_ID,
            feature_supported=self.supports(Feature.TOOLS_NATIVE),
            supported_capabilities=self.supported_native_tools(model),
        )
        if mcp_servers:
            transport = mcp_servers[0].transport
            feature_map = {
                "stdio": Feature.TOOLS_MCP_STDIO,
                "http": Feature.TOOLS_MCP_HTTP,
                "sse": Feature.TOOLS_MCP_SSE,
            }
            raise UnsupportedFeatureError(
                "OpenCodeServerRuntime: runtime MCP registration via `mcp_servers=` "
                "is not available — the opencode-ai 0.1.0a36 SDK has no "
                "`client.mcp` resource (only the static `opencode.json` "
                "McpLocalConfig / McpRemoteConfig types). Pre-register MCP "
                "servers in `opencode.json` and they'll be available to the "
                "model automatically. Will flip True once SDK exposes MCP.",
                feature=feature_map.get(transport, Feature.TOOLS_MCP_STDIO),
            )
        if on_permission is not None:
            raise UnsupportedFeatureError(
                "OpenCodeServerRuntime: `on_permission=` callback is not "
                "available — the opencode-ai 0.1.0a36 SDK emits "
                "`permission.updated` events but has no permission-reply "
                "endpoint. Iteration E surfaces permission events on "
                "`on_event=` for observation; gating awaits SDK support.",
                feature=Feature.PERMISSION_CALLBACK,
            )
        _check_hooks_supported(
            on_event,
            adapter_label=self.label,
            supports=self.supports,
        )
        _check_provider_options(
            provider_options,
            expected_type=OpenCodeServerOptions,
            adapter_label=self.label,
        )
        # Phase 6 — REQUEST_METADATA soft contract: OpenCode's server
        # HTTP API has no per-request metadata channel today. Silently
        # drop for v1; consumers branch on supports() if they care.
        del metadata
        # Phase 6 — PROMPT_CACHE_CONTROL soft contract: OpenCode's
        # server HTTP API has no explicit cache-key channel. Silently
        # drop for v1.
        del cache
        opencode_options = (
            provider_options if isinstance(provider_options, OpenCodeServerOptions) else None
        )
        return OpenCodeServerSession(
            self,
            resume=resume,
            system=system,
            model=model,
            slash_commands=slash_commands,
            provider_options=opencode_options,
            on_event=on_event,
        )

    async def count_tokens(
        self,
        prompt: Prompt,
        *,
        system: str | None = None,
        model: ProviderModel | None = None,
    ) -> int:
        # Phase 6 — COUNT_TOKENS not yet wired for OpenCode's server.
        # The HTTP agent fronts arbitrary upstream providers; the
        # right counter depends on which upstream the consumer wired,
        # and the server's HTTP API doesn't expose a counter endpoint
        # today. Raise the documented decline.
        del prompt, system, model
        raise UnsupportedFeatureError(
            f"{self.label}: count_tokens() is not supported — the OpenCode "
            f"agent server fronts arbitrary upstream providers and exposes "
            f"no counter endpoint of its own. Check "
            f"runtime.supports(Feature.COUNT_TOKENS) first.",
            feature=Feature.COUNT_TOKENS,
        )

    async def list_models(self) -> list[ModelInfo]:
        """Return models the OpenCode server has configured upstream.

        Hits ``client.app.providers()`` and flattens the response's
        ``{provider_id: {model_id: model_meta}}`` tree into
        :class:`ModelInfo` rows tagged with this adapter's
        ``PROVIDER_ID`` (the airframe-facing identifier — distinct
        from the OpenCode upstream provider IDs the server uses
        internally, which are surfaced on
        :attr:`ModelInfo.raw["provider"]` for callers that need
        upstream routing).

        Raises:
            RuntimeServerStartError: When the OpenCode server is
                unreachable (`opencode serve` not running). Message
                points at the command to run.
            RuntimeAuthError: When the server returns 401 (bad
                Basic-auth credentials or missing password against a
                password-protected server).
            RuntimeTransientError: 5xx / throttling.
            RuntimeProtocolError: Unparseable response shape.
        """
        client = self._ensure_client()
        try:
            payload = await client.app.providers()
        except Exception as exc:  # noqa: BLE001 — classify at boundary
            raise _classify_opencode_error(exc, self._base_url) from exc

        return _models_from_provider_payload(payload, self.PROVIDER_ID)


# --- AgentSession ----------------------------------------------------------


class OpenCodeServerSession:
    """Per-conversation session against the OpenCode server.

    Owns a server-issued :attr:`id` (populated on first turn from
    ``client.session.create()`` or carried in from ``resume=``).
    Streaming drives ``client.session.chat()`` as a background task
    while translating ``message.part.updated`` events on the global
    ``client.event.list()`` bus into airframe's
    :class:`~airframe.events.RuntimeEvent` union. Cancellation calls
    ``client.session.abort()`` and surfaces
    :class:`RuntimeCancelledError` to the awaiter.

    Iteration B caveats:

    * ``schema=`` is not yet wired — the 0.1.0a36 SDK has no MCP
      resource (the forced-tool shim path), so passing a schema
      raises :class:`UnsupportedFeatureError`. Iteration D enables it.
    * Streaming computes text deltas client-side by tracking the
      previous ``TextPart.text`` per part id and yielding only the
      new suffix — OpenCode emits "part updated" snapshots rather
      than native delta events.
    * Routing requires both an upstream ``provider_id`` (anthropic /
      openai / openrouter / …) and a ``model_id``. Resolution order:
      ``OpenCodeServerOptions(provider_id=)`` → lookup via
      ``client.app.providers()`` (single match wins; ambiguous raises
      asking for explicit routing).
    """

    def __init__(
        self,
        runtime: OpenCodeServerRuntime,
        *,
        resume: str | None = None,
        system: str | None = None,
        model: ProviderModel | None = None,
        provider_options: OpenCodeServerOptions | None = None,
        on_event: Callable[[HookEvent], None] | None = None,
        slash_commands: SlashCommandsConfig | None = None,
    ) -> None:
        self._runtime = runtime
        self._system = system
        self._model = model
        self._provider_options = provider_options
        self._on_event = on_event
        # ``id`` is populated on first turn (or carried from resume=).
        # ``_owned`` tracks whether we created the server-side session
        # (and so are responsible for deleting it on close).
        self.id: str | None = resume
        self._owned: bool = resume is None
        self._closed = False
        self._chat_task: asyncio.Task[Any] | None = None
        # Iteration E: running totals for budget enforcement.
        self._turn_count: int = 0
        self._cumulative_cost_usd: float = 0.0
        # Cached provider/model resolution to avoid hitting
        # client.app.providers() on every turn within the session.
        self._provider_for_model: dict[str, str] = {}
        # Phase 6 — SLASH_COMMANDS. Filesystem-only discovery.
        self._slash_commands: SlashCommandsConfig | None = slash_commands

    # --- public AgentSession surface -------------------------------------

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
        self._reject_unsupported_kwargs(schema=schema)
        _check_budget_supported(
            max_turns=max_turns,
            max_budget_usd=max_budget_usd,
            adapter_label="opencode_server",
            supports=self._runtime.supports,
        )
        _enforce_budget_pre_turn(
            max_turns=max_turns,
            max_budget_usd=max_budget_usd,
            cumulative_cost_usd=self._cumulative_cost_usd,
            turn_count=self._turn_count,
            adapter_label="opencode_server",
        )
        text, images, files = _split_prompt_parts(
            prompt,
            adapter_label="opencode_server",
            supports_vision=self._runtime.supports(Feature.VISION_INPUT),
            supports_file=self._runtime.supports(Feature.FILE_INPUT),
        )
        client = self._runtime._ensure_client()
        await self._ensure_session_id(client)
        self._fire_user_prompt_submit(text)
        provider_id, model_id = await self._resolve_routing(client)
        chat_kwargs = self._build_chat_kwargs(
            provider_id=provider_id,
            model_id=model_id,
            text=text,
            images=images,
            files=files,
            thinking=thinking,
            timeout=timeout,
        )
        try:
            message = await client.session.chat(self.id, **chat_kwargs)
        except asyncio.CancelledError:
            await self._abort_quietly(client)
            raise
        except Exception as exc:  # noqa: BLE001 — classify at boundary
            raise _classify_opencode_error(exc, self._runtime._base_url) from exc
        result = _assistant_message_to_result(
            message,
            airframe_provider_id=self._runtime.PROVIDER_ID,
            upstream_provider_id=provider_id,
        )
        self._record_turn_cost(result.cost.cost_usd)
        return result

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
        self._reject_unsupported_kwargs(schema=schema)
        _check_budget_supported(
            max_turns=max_turns,
            max_budget_usd=max_budget_usd,
            adapter_label="opencode_server",
            supports=self._runtime.supports,
        )
        _enforce_budget_pre_turn(
            max_turns=max_turns,
            max_budget_usd=max_budget_usd,
            cumulative_cost_usd=self._cumulative_cost_usd,
            turn_count=self._turn_count,
            adapter_label="opencode_server",
        )
        text, images, files = _split_prompt_parts(
            prompt,
            adapter_label="opencode_server",
            supports_vision=self._runtime.supports(Feature.VISION_INPUT),
            supports_file=self._runtime.supports(Feature.FILE_INPUT),
        )
        client = self._runtime._ensure_client()
        await self._ensure_session_id(client)
        self._fire_user_prompt_submit(text)
        provider_id, model_id = await self._resolve_routing(client)
        chat_kwargs = self._build_chat_kwargs(
            provider_id=provider_id,
            model_id=model_id,
            text=text,
            images=images,
            files=files,
            thinking=thinking,
            timeout=timeout,
        )

        # Subscribe to the global event bus BEFORE dispatching chat so
        # we don't race past early part updates.
        event_stream = await client.event.list()
        # Launch the chat as a background task so we can interleave
        # delta translation with awaiting its terminal AssistantMessage.
        self._chat_task = asyncio.create_task(client.session.chat(self.id, **chat_kwargs))

        try:
            async for evt in _drive_stream(
                event_stream=event_stream,
                chat_task=self._chat_task,
                session_id=self.id or "",
                airframe_provider_id=self._runtime.PROVIDER_ID,
                upstream_provider_id=provider_id,
                base_url=self._runtime._base_url,
                on_event=self._on_event,
            ):
                if isinstance(evt, TurnComplete):
                    self._record_turn_cost(evt.result.cost.cost_usd)
                yield evt
        except asyncio.CancelledError:
            await self._abort_quietly(client)
            raise
        finally:
            self._chat_task = None
            close = getattr(event_stream, "close", None)
            if close is not None:
                try:
                    if asyncio.iscoroutinefunction(close):
                        await close()
                    else:
                        close()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("opencode_server.event_stream_close_failed error=%s", exc)

    async def list_slash_commands(self) -> list[SlashCommand]:
        from airframe.slash_commands import discover

        return discover(self._slash_commands)

    async def cancel(self) -> None:
        """Abort the in-flight turn, if any.

        Idempotent and cheap. When a turn is in flight, calls
        ``client.session.abort(self.id)`` and cancels the background
        chat task — the awaiting :meth:`execute` / :meth:`stream`
        raises :class:`RuntimeCancelledError`. No-op when nothing is
        running or the session was never opened.
        """
        if self.id is None or self._closed:
            return
        client = self._runtime._client
        if client is None:
            return
        await self._abort_quietly(client)

    async def close(self) -> None:
        """Release the server-side session and stop translating events.

        Idempotent and must not raise (runs from ``finally`` /
        ``__aexit__``). Deletes the server-side session only when this
        :class:`OpenCodeServerSession` *created* it — sessions opened
        via ``resume=<id>`` existed before us and outlive us. The
        runtime's HTTP client is not touched (it stays alive for
        sibling sessions).
        """
        if self._closed:
            return
        self._closed = True
        # Fire session_end before tearing down vendor state — the
        # observer may want to inspect the session id while it still
        # exists on the server.
        _fire_hook_event(
            self._on_event,
            "session_end",
            session_id=self.id,
            payload={
                "turn_count": self._turn_count,
                "cost_usd": self._cumulative_cost_usd,
            },
        )
        task = self._chat_task
        self._chat_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except BaseException:  # noqa: BLE001
                pass
        client = self._runtime._client
        if client is None or self.id is None or not self._owned:
            return
        try:
            await client.session.delete(self.id)
        except Exception as exc:  # noqa: BLE001 — teardown never raises
            logger.debug("opencode_server.session_delete_failed error=%s", exc)

    def unwrap(self, cls: type[T]) -> T:
        if isinstance(self, cls):
            return self  # type: ignore[return-value]
        # The 0.1.0a36 SDK doesn't expose a long-lived per-session
        # vendor object on the client — the session id is the handle,
        # and the underlying HTTP client (which IS a vendor object)
        # lives on the runtime. Point unwrap callers at
        # ``runtime.unwrap(AsyncOpencode)`` instead.
        raise TypeError(
            f"OpenCodeServerSession cannot unwrap to {cls!r}. The OpenCode "
            "SDK doesn't surface a per-session vendor object — the session "
            "is identified by its id on the server. Reach the runtime's "
            "AsyncOpencode client via `runtime.unwrap(AsyncOpencode)`."
        )

    # --- internals -------------------------------------------------------

    def _reject_unsupported_kwargs(self, *, schema: Any) -> None:
        """Reject kwargs gated behind features the adapter hasn't shipped yet."""
        if schema is not None:
            raise UnsupportedFeatureError(
                "OpenCodeServerSession: schema= structured output is not "
                "wired — the opencode-ai 0.1.0a36 SDK has no client.mcp "
                "resource, so the forced-tool shim can't ship. Will flip "
                "True once SDK exposes MCP.",
                feature=Feature.STRUCTURED_OUTPUT_JSON_SCHEMA,
            )

    async def _ensure_session_id(self, client: Any) -> None:
        """Create a server-side session if we don't have an id yet.

        Fires ``session_start`` on the first call only — even if the
        session was opened with ``resume=`` (in which case ``self.id``
        is already populated but the observer still wants to know we
        connected). We track this with a sentinel
        (``_session_start_fired``).
        """
        if not getattr(self, "_session_start_fired", False):
            # First call — either create or adopt a resumed id, then
            # fire session_start.
            if self.id is None:
                try:
                    sess = await client.session.create()
                except Exception as exc:  # noqa: BLE001
                    raise _classify_opencode_error(exc, self._runtime._base_url) from exc
                new_id = getattr(sess, "id", None)
                if not isinstance(new_id, str) or not new_id:
                    raise RuntimeProtocolError(
                        "OpenCode server returned a session without an id: " + repr(sess)
                    )
                self.id = new_id
                self._owned = True
            self._session_start_fired = True
            _fire_hook_event(
                self._on_event,
                "session_start",
                session_id=self.id,
                payload={
                    "resumed": not self._owned,
                    "model": self._model.model_id if self._model is not None else None,
                },
            )
            return
        # Subsequent calls — already have an id and already fired.

    async def _resolve_routing(self, client: Any) -> tuple[str, str]:
        """Resolve ``(upstream_provider_id, model_id)`` for this turn.

        Priority:

        1. Explicit ``OpenCodeServerOptions(provider_id=)`` —
           upstream comes from the caller; model from
           :attr:`ProviderModel.model_id` / runtime default.
        2. Model from binding; provider looked up via
           ``client.app.providers()``. Single match wins; ambiguous
           raises with a hint at the explicit-routing path.
        """
        model_id = (
            self._model.model_id if self._model is not None else None
        ) or self._runtime._default_model
        if not isinstance(model_id, str) or not model_id:
            raise UnsupportedFeatureError(
                "OpenCodeServerRuntime: no model resolved for this turn. "
                "Pass `model=ProviderModel('opencode', '<model-id>')` to "
                "session()/execute(), or set OPENCODE_SERVER_DEFAULT_MODEL.",
                feature=Feature.STREAMING,
            )
        opts = self._provider_options
        if opts is not None and opts.provider_id:
            return opts.provider_id, model_id
        cached = self._provider_for_model.get(model_id)
        if cached is not None:
            return cached, model_id
        try:
            payload = await client.app.providers()
        except Exception as exc:  # noqa: BLE001
            raise _classify_opencode_error(exc, self._runtime._base_url) from exc
        upstream = _find_upstream_for_model(payload, model_id)
        if upstream is None:
            raise UnsupportedFeatureError(
                f"OpenCodeServerRuntime: cannot route model {model_id!r} — "
                "no configured upstream provider exposes it. Set "
                "`provider_options=OpenCodeServerOptions(provider_id=...)` "
                "or `opencode auth login <provider>` for the upstream that "
                "hosts this model.",
                feature=Feature.STREAMING,
            )
        self._provider_for_model[model_id] = upstream
        return upstream, model_id

    def _build_chat_kwargs(
        self,
        *,
        provider_id: str,
        model_id: str,
        text: str,
        images: list[Any],
        files: list[Any],
        thinking: Any,
        timeout: float,
    ) -> dict[str, Any]:
        """Assemble the kwargs passed to ``client.session.chat()``.

        Text + attachments map to OpenCode's ``parts`` list:
        ``{"type": "text", "text": "..."}`` followed by
        ``{"type": "file", "mime": "...", "url": "..."}`` for every
        :class:`ImageInput` / :class:`FileInput` (URL / bytes / path
        all collapse to a single ``file`` part — image vs document is
        signalled by ``mime``).

        ``thinking=`` is sent via ``extra_body`` (the SDK has no
        first-class slot); the envelope chosen depends on the
        upstream provider.
        """
        parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for image in images:
            parts.append(_image_to_file_part(image))
        for file in files:
            parts.append(_file_to_file_part(file))

        kwargs: dict[str, Any] = {
            "model_id": model_id,
            "provider_id": provider_id,
            "parts": parts,
            "timeout": timeout,
        }
        if self._system is not None:
            kwargs["system"] = self._system
        tool_filter = _tools_allow_denylist(self._provider_options)
        if tool_filter:
            kwargs["tools"] = tool_filter
        reasoning_envelope = _reasoning_extra_body(thinking, upstream_provider_id=provider_id)
        if reasoning_envelope:
            kwargs["extra_body"] = reasoning_envelope
        return kwargs

    def _fire_user_prompt_submit(self, text: str) -> None:
        """Emit the ``user_prompt_submit`` hook with a length-bounded preview."""
        if self._on_event is None:
            return
        preview = text if len(text) <= 200 else text[:200] + "…"
        _fire_hook_event(
            self._on_event,
            "user_prompt_submit",
            session_id=self.id,
            payload={"prompt": preview, "length": len(text)},
        )

    def _record_turn_cost(self, cost_usd: float | None) -> None:
        """Accumulate one turn's cost against the budget tracker.

        ``None`` is the documented "upstream didn't report cost"
        signal — we increment the turn counter but leave the dollar
        accumulator alone. With ``max_budget_usd`` set against an
        un-reported upstream, the cap is effectively unenforced for
        that turn; we log so it's visible in debug output.
        """
        self._turn_count += 1
        if cost_usd is not None:
            self._cumulative_cost_usd += cost_usd
        else:
            logger.debug(
                "opencode_server.cost_unreported session=%s turn=%d "
                "(BUDGET_USD_CAP is best-effort against this upstream)",
                self.id,
                self._turn_count,
            )

    async def _abort_quietly(self, client: Any) -> None:
        """Best-effort ``client.session.abort()`` for cancel paths.

        Used from both :meth:`cancel` (user-initiated) and the
        ``asyncio.CancelledError`` paths in execute/stream. Logs at
        debug; never raises.
        """
        if self.id is None:
            return
        task = self._chat_task
        if task is not None and not task.done():
            task.cancel()
        try:
            await client.session.abort(self.id)
        except Exception as exc:  # noqa: BLE001 — abort is best-effort
            logger.debug("opencode_server.session_abort_failed error=%s", exc)


# --- Prompt-part translation (Iteration C) ----------------------------------


def _image_to_file_part(image: Any) -> dict[str, Any]:
    """Translate :class:`ImageInput` into an OpenCode ``FilePartInputParam``.

    URL → pass-through (the server fetches). ``bytes_`` → encode as
    ``data:`` URL. ``path`` → read + encode. ``media_type`` is
    inferred from extension when not supplied; defaults to
    ``image/png`` for raw bytes without metadata.
    """
    media_type = image.media_type
    if image.url is not None:
        return {
            "type": "file",
            "mime": media_type or _infer_image_mime(image.url) or "image/jpeg",
            "url": image.url,
        }
    if image.bytes_ is not None:
        mime = media_type or "image/png"
        return {
            "type": "file",
            "mime": mime,
            "url": _data_url_from_bytes(image.bytes_, mime),
        }
    if image.path is not None:
        mime = media_type or _infer_image_mime(image.path) or "image/png"
        return {
            "type": "file",
            "mime": mime,
            "url": _data_url_from_path(image.path, mime),
            "filename": os.path.basename(image.path),
        }
    raise UnsupportedFeatureError(
        "OpenCodeServerSession: ImageInput requires path=, bytes_=, or url=.",
        feature=Feature.VISION_INPUT,
    )


def _file_to_file_part(file: Any) -> dict[str, Any]:
    """Translate :class:`FileInput` into an OpenCode ``FilePartInputParam``."""
    media_type = file.media_type or _infer_file_mime(file.path) or "application/octet-stream"
    return {
        "type": "file",
        "mime": media_type,
        "url": _data_url_from_path(file.path, media_type),
        "filename": os.path.basename(file.path),
    }


def _infer_image_mime(path_or_url: str) -> str | None:
    """Guess an image MIME from a path or URL extension."""
    import mimetypes

    guess, _ = mimetypes.guess_type(path_or_url)
    if guess and guess.startswith("image/"):
        return guess
    return None


def _infer_file_mime(path: str) -> str | None:
    import mimetypes

    guess, _ = mimetypes.guess_type(path)
    return guess


def _data_url_from_bytes(data: bytes, mime: str) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _data_url_from_path(path: str, mime: str) -> str:
    with open(path, "rb") as fh:
        data = fh.read()
    return _data_url_from_bytes(data, mime)


# --- Tool allow/denylist (Iteration D) ---------------------------------------


def _tools_allow_denylist(opts: OpenCodeServerOptions | None) -> dict[str, bool]:
    """Translate :class:`OpenCodeServerOptions` allow/denylist into chat's ``tools=`` map.

    ``session.chat(tools=)`` in opencode-ai 0.1.0a36 is a
    ``Dict[str, bool]`` that allow/denylists OpenCode's *built-in*
    tools by name (``"bash"``, ``"read"``, ``"write"``, ``"edit"``,
    ``"grep"``, ``"webfetch"``, …). Semantics:

    * ``available_tools=("bash", "read")`` produces
      ``{"bash": True, "read": True}`` — implicit "everything else
      off". ``None`` (the default) sends no filter — server defaults
      apply.
    * ``excluded_tools=("write", "edit")`` produces
      ``{"write": False, "edit": False}`` — implicit "everything else
      on". Empty tuple sends no filter.
    * Both set: the allow keys win on overlap; denylist entries that
      aren't in the allowlist are dropped (denying a tool not in the
      allowlist is a no-op).

    Returns ``{}`` for the "no filter at all" case so callers can
    skip threading ``tools=`` through entirely.
    """
    if opts is None:
        return {}
    allow = opts.available_tools
    deny = opts.excluded_tools
    if allow is None and not deny:
        return {}
    filter_map: dict[str, bool] = {}
    if allow is not None:
        for name in allow:
            filter_map[name] = True
    for name in deny:
        # Denying a tool already on the allowlist is contradictory;
        # the explicit deny wins (matches OpenCode's own semantics
        # where any False bit kills the slot).
        filter_map[name] = False
    return filter_map


# --- Reasoning pass-through (Iteration C) ------------------------------------

# Default token budgets per effort level when mapping to Anthropic's
# budget_tokens shape. Anthropic's API exposes a single integer; the
# OpenAI-shape ``reasoning_effort`` enum collapses onto these defaults
# only when the upstream demands a numeric budget. Conservative
# values — callers wanting precise control should pass
# ``thinking={"budget_tokens": N}`` directly.
_EFFORT_TO_BUDGET_TOKENS = {
    "minimal": 1_024,
    "low": 4_096,
    "medium": 8_192,
    "high": 16_384,
}

# Upstream provider_ids that take Anthropic's ``thinking`` envelope.
# Everyone else (openai / openrouter / ollama / together / groq /
# vllm / llama.cpp) speaks the OpenAI Chat Completions
# ``reasoning_effort`` shape.
_ANTHROPIC_UPSTREAMS = frozenset({"anthropic", "claude"})


def _reasoning_extra_body(thinking: Any, *, upstream_provider_id: str) -> dict[str, Any]:
    """Translate ``thinking=`` into the upstream's reasoning envelope.

    The agent server forwards ``extra_body`` verbatim to the chosen
    upstream. Per-upstream shapes:

    * **Anthropic** — ``{"thinking": {"type": "enabled",
      "budget_tokens": N}}``. Effort literals map to a default
      ``budget_tokens`` table; dict-shaped ``{"budget_tokens": N}``
      is forwarded as-is.
    * **Everyone else** — ``{"reasoning_effort": "low|medium|high"}``.
      ``"minimal"`` is OpenAI-family only; we forward it verbatim
      and let the upstream decide (a non-OpenAI model that doesn't
      recognise it ignores the field per the documented pass-through
      contract).

    ``None`` and ``"disabled"`` return an empty dict — no envelope
    sent, the model decides on its own.
    """
    if thinking is None or thinking == "disabled":
        return {}
    is_anthropic = upstream_provider_id in _ANTHROPIC_UPSTREAMS
    if isinstance(thinking, dict):
        # Dict-shaped is the Anthropic-style {"budget_tokens": N}
        # envelope. Forward to Anthropic upstreams as-is; raise on
        # everyone else (OpenAI / OpenRouter / Ollama don't honour it
        # and silently dropping would be a worse failure mode).
        budget_tokens = thinking.get("budget_tokens")
        if not isinstance(budget_tokens, int) or budget_tokens <= 0:
            raise UnsupportedFeatureError(
                "OpenCodeServerSession: thinking={'budget_tokens': N} requires "
                "a positive integer; got " + repr(thinking),
                feature=Feature.REASONING_BUDGET_TOKENS,
            )
        if not is_anthropic:
            raise UnsupportedFeatureError(
                "OpenCodeServerSession: thinking={'budget_tokens': N} is "
                "Anthropic-shape and requires an Anthropic upstream "
                f"(got upstream={upstream_provider_id!r}). For other "
                "backends pass an effort literal "
                "(thinking='low'|'medium'|'high').",
                feature=Feature.REASONING_BUDGET_TOKENS,
            )
        return {"thinking": {"type": "enabled", "budget_tokens": budget_tokens}}

    if isinstance(thinking, str):
        if is_anthropic:
            budget = _EFFORT_TO_BUDGET_TOKENS.get(thinking)
            if budget is None:
                raise UnsupportedFeatureError(
                    f"OpenCodeServerSession: unknown thinking effort "
                    f"{thinking!r}; expected one of {sorted(_EFFORT_TO_BUDGET_TOKENS)}.",
                    feature=Feature.REASONING_EFFORT,
                )
            return {"thinking": {"type": "enabled", "budget_tokens": budget}}
        return {"reasoning_effort": thinking}
    raise UnsupportedFeatureError(
        f"OpenCodeServerSession: thinking= must be None, an effort literal, "
        f"'disabled', or {{'budget_tokens': N}}; got {type(thinking).__name__}.",
        feature=Feature.REASONING_EFFORT,
    )


# --- Stream driver + helpers ------------------------------------------------


async def _drive_stream(
    *,
    event_stream: Any,
    chat_task: asyncio.Task[Any],
    session_id: str,
    airframe_provider_id: str,
    upstream_provider_id: str,
    base_url: str,
    on_event: Callable[[HookEvent], None] | None = None,
) -> AsyncIterator[RuntimeEvent]:
    """Translate ``client.event.list()`` into airframe events.

    OpenCode's bus is global: every event carries a ``session_id``
    field on its ``properties`` (where applicable). We filter to
    ``session_id`` only, compute text/reasoning deltas client-side
    against per-part snapshots, surface tool-call lifecycle, and
    terminate on ``session.idle`` (or ``session.error``). At
    termination we await the background ``chat_task`` for its
    :class:`AssistantMessage`, build a :class:`RuntimeResult`, and
    yield a final :class:`TurnComplete`.

    When ``on_event`` is provided, also synthesises
    :class:`~airframe.hooks.HookEvent` instances for tool lifecycle
    (``pre_tool_use`` / ``post_tool_use`` / ``tool_failure``) at the
    same moments :class:`ToolCallStart` / :class:`ToolCallResult`
    fire.
    """
    text_per_part: dict[str, str] = {}
    reasoning_per_part: dict[str, str] = {}
    tool_started: set[str] = set()
    pending_error: Exception | None = None

    try:
        async for raw_event in event_stream:
            evt_type = _event_type(raw_event)
            if evt_type == "session.error" and _event_session_id(raw_event) == session_id:
                pending_error = _classify_session_error(raw_event, base_url)
                break
            if evt_type == "session.idle" and _event_session_id(raw_event) == session_id:
                break
            if evt_type == "message.part.updated":
                part = _event_part(raw_event)
                if part is None or _part_session_id(part) != session_id:
                    continue
                for airframe_event in _translate_part_update(
                    part,
                    text_per_part=text_per_part,
                    reasoning_per_part=reasoning_per_part,
                    tool_started=tool_started,
                    on_event=on_event,
                    hook_session_id=session_id,
                ):
                    yield airframe_event
    except asyncio.CancelledError:
        # Propagate up — the session class handles abort + cleanup.
        raise

    if pending_error is not None:
        # The chat task is still running; cancel before raising so the
        # background coroutine doesn't leak.
        if not chat_task.done():
            chat_task.cancel()
        raise pending_error

    # Await the chat task for its terminal AssistantMessage. If the
    # task itself raised, classify and surface.
    try:
        final_message = await chat_task
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _classify_opencode_error(exc, base_url) from exc

    result = _assistant_message_to_result(
        final_message,
        airframe_provider_id=airframe_provider_id,
        upstream_provider_id=upstream_provider_id,
    )
    yield TurnComplete(result=result)


def _translate_part_update(
    part: Any,
    *,
    text_per_part: dict[str, str],
    reasoning_per_part: dict[str, str],
    tool_started: set[str],
    on_event: Callable[[HookEvent], None] | None = None,
    hook_session_id: str | None = None,
) -> list[RuntimeEvent]:
    """Translate one ``message.part.updated`` payload into airframe events.

    The SDK emits "part updated" snapshots carrying the full current
    state of the part. We compute deltas against the previous value
    we've seen for that part_id.

    Also fires the matching :class:`~airframe.hooks.HookEvent` kinds
    (``pre_tool_use`` / ``post_tool_use`` / ``tool_failure``)
    alongside :class:`ToolCallStart` / :class:`ToolCallResult` —
    same boundaries, exception-safe via
    :func:`~airframe.sessions._fire_hook_event`.
    """
    out: list[RuntimeEvent] = []
    part_type = _part_type(part)
    part_id = _part_id(part)
    if not part_id:
        return out
    if part_type == "text":
        text = _part_text(part) or ""
        if not text:
            return out
        previous = text_per_part.get(part_id, "")
        if text == previous:
            return out
        if text.startswith(previous):
            delta = text[len(previous) :]
        else:
            # Server replaced the text mid-stream; emit the full text
            # as the next delta rather than a complicated diff.
            delta = text
        text_per_part[part_id] = text
        if delta:
            out.append(TextDelta(text=delta))
    elif part_type == "tool":
        state_name, tool_name, args, result, error = _tool_part_state(part)
        tool_label = tool_name or "<unknown>"
        if part_id not in tool_started:
            tool_started.add(part_id)
            args_preview = _stringify_args(args)
            out.append(
                ToolCallStart(
                    tool_name=tool_label,
                    tool_call_id=part_id,
                    arguments_preview=args_preview,
                )
            )
            _fire_hook_event(
                on_event,
                "pre_tool_use",
                session_id=hook_session_id,
                payload={
                    "tool_name": tool_label,
                    "tool_call_id": part_id,
                    "arguments": args_preview,
                },
            )
        if state_name == "completed":
            out.append(
                ToolCallResult(
                    tool_call_id=part_id,
                    output=result,
                    is_error=False,
                )
            )
            _fire_hook_event(
                on_event,
                "post_tool_use",
                session_id=hook_session_id,
                payload={
                    "tool_name": tool_label,
                    "tool_call_id": part_id,
                    "output": result,
                },
            )
        elif state_name == "error":
            out.append(
                ToolCallResult(
                    tool_call_id=part_id,
                    output=error or "tool execution failed",
                    is_error=True,
                )
            )
            _fire_hook_event(
                on_event,
                "tool_failure",
                session_id=hook_session_id,
                payload={
                    "tool_name": tool_label,
                    "tool_call_id": part_id,
                    "error": error or "tool execution failed",
                },
            )
    elif part_type == "reasoning":
        text = _part_text(part) or ""
        if not text:
            return out
        previous = reasoning_per_part.get(part_id, "")
        if text == previous:
            return out
        delta = text[len(previous) :] if text.startswith(previous) else text
        reasoning_per_part[part_id] = text
        if delta:
            out.append(ReasoningDelta(text=delta))
    # Other part types (step_start, step_finish, snapshot, patch,
    # file) don't map onto airframe's event union today; ignore.
    return out


def _assistant_message_to_result(
    message: Any,
    *,
    airframe_provider_id: str,
    upstream_provider_id: str,
) -> RuntimeResult:
    """Build a :class:`RuntimeResult` from the terminal AssistantMessage."""
    text = _assistant_message_text(message)
    finish = _assistant_message_finish(message)
    tokens = _attr(message, "tokens")
    input_tokens = int(_attr(tokens, "input") or 0) if tokens is not None else 0
    output_tokens = int(_attr(tokens, "output") or 0) if tokens is not None else 0
    reasoning_tokens = int(_attr(tokens, "reasoning") or 0) if tokens is not None else 0
    cache = _attr(tokens, "cache") if tokens is not None else None
    cache_read = int(_attr(cache, "read") or 0) if cache is not None else 0
    cache_write = int(_attr(cache, "write") or 0) if cache is not None else 0
    cost_usd = _attr(message, "cost")
    model_id = _attr(message, "api_model_id") or ""
    cost = CostRecord(
        provider_id=airframe_provider_id,
        model_id=str(model_id),
        cost_usd=float(cost_usd) if isinstance(cost_usd, (int, float)) else None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        finish=finish,
        reasoning_tokens=reasoning_tokens,
    )
    return RuntimeResult(
        text=text,
        structured=None,
        cost=cost,
        finish=finish,
        raw={"upstream_provider": upstream_provider_id},
    )


def _find_upstream_for_model(payload: Any, model_id: str) -> str | None:
    """Return the upstream provider id that hosts ``model_id``.

    Walks ``client.app.providers()`` output. Single match → that
    provider. Multiple → ``None`` (caller raises asking for explicit
    routing). No match → ``None``.
    """
    matches: list[str] = []
    providers: list[Any] = (
        (payload.get("providers") if isinstance(payload, dict) else None)
        or getattr(payload, "providers", None)
        or []
    )
    for entry in providers:
        provider_id = _attr(entry, "id") or _attr(entry, "provider_id")
        if not isinstance(provider_id, str) or not provider_id:
            continue
        models = _attr(entry, "models") or {}
        if isinstance(models, dict) and model_id in models:
            matches.append(provider_id)
        elif isinstance(models, list):
            for m in models:
                if _attr(m, "id") == model_id:
                    matches.append(provider_id)
                    break
    if len(matches) == 1:
        return matches[0]
    return None


# --- low-level field-access helpers (defensive against Pydantic + dict) -----


def _attr(obj: Any, name: str) -> Any:
    """Get ``name`` from ``obj`` whether it's a Pydantic model or dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _event_type(event: Any) -> str | None:
    val = _attr(event, "type")
    return val if isinstance(val, str) else None


def _event_session_id(event: Any) -> str | None:
    props = _attr(event, "properties")
    sid = _attr(props, "session_id")
    return sid if isinstance(sid, str) else None


def _event_part(event: Any) -> Any:
    return _attr(_attr(event, "properties"), "part")


def _part_session_id(part: Any) -> str | None:
    sid = _attr(part, "session_id")
    return sid if isinstance(sid, str) else None


def _part_id(part: Any) -> str | None:
    pid = _attr(part, "id")
    return pid if isinstance(pid, str) else None


def _part_type(part: Any) -> str | None:
    pt = _attr(part, "type")
    return pt if isinstance(pt, str) else None


def _part_text(part: Any) -> str | None:
    txt = _attr(part, "text")
    return txt if isinstance(txt, str) else None


def _stringify_args(args: Any) -> str:
    """Best-effort JSON for the tool-call args preview.

    The event union allows ``args`` to be a dict, str, or None. We
    return a JSON string when possible (matching the OpenAI-compat
    pattern where ``arguments_preview`` is partial-JSON during
    streaming) and the empty string otherwise.
    """
    if args is None:
        return ""
    if isinstance(args, str):
        return args
    try:
        import json as _json

        return _json.dumps(args, default=str)
    except (TypeError, ValueError):
        return repr(args)


def _tool_part_state(part: Any) -> tuple[str | None, str | None, Any, Any, str | None]:
    """Pull ``(state_name, tool_name, args, result, error)`` from a ToolPart."""
    tool_name = _attr(part, "tool")
    state = _attr(part, "state")
    state_name = _attr(state, "status") or _attr(state, "type")
    args = _attr(state, "input") or _attr(state, "args")
    result = _attr(state, "output") or _attr(state, "result")
    error = _attr(state, "error")
    return (
        state_name if isinstance(state_name, str) else None,
        tool_name if isinstance(tool_name, str) else None,
        args,
        result,
        error if isinstance(error, str) else None,
    )


def _assistant_message_text(message: Any) -> str:
    """The plan returns the final assembled text from the AssistantMessage.

    The 0.1.0a36 ``AssistantMessage`` doesn't carry the body parts on
    itself — those live on the message's ``parts`` (fetched via
    ``client.session.messages.list``). For Iteration B, we don't
    refetch — the chat() call returns ``AssistantMessage`` *metadata*
    only, and the text we already captured in the stream loop is the
    canonical source. Callers using execute() (non-streaming) get
    only the text the SDK chose to surface on the message itself
    (``summary`` or empty); upstream-equivalent text appears in the
    text-deltas emitted on ``stream()``.
    """
    summary = _attr(message, "summary")
    if isinstance(summary, str):
        return summary
    return ""


def _assistant_message_finish(message: Any) -> str | None:
    """Best-effort finish-reason from ``AssistantMessage``.

    OpenCode reports a ``mode`` (e.g. ``"chat"``) but no canonical
    finish reason. When the message carries an error we surface
    ``"error"``; otherwise ``"stop"`` for normal completion.
    """
    error = _attr(message, "error")
    if error is not None:
        return "error"
    return "stop"


def _classify_session_error(event: Any, base_url: str) -> Exception:
    """Translate a ``session.error`` event into an airframe error."""
    props = _attr(event, "properties")
    err = _attr(props, "error")
    name = _attr(err, "name") or _attr(err, "type")
    msg = _attr(err, "message") or _attr(err, "data") or repr(err)
    if name in ("ProviderAuthError", "provider_auth_error", "auth"):
        return RuntimeAuthError(f"OpenCode upstream provider rejected credentials: {msg}")
    if name in ("MessageAbortedError", "message_aborted_error", "aborted"):
        return RuntimeCancelledError(f"OpenCode session aborted: {msg}")
    if name in ("MessageOutputLengthError", "output_length"):
        return RuntimeProtocolError(f"OpenCode message output length exceeded: {msg}")
    return RuntimeProtocolError(
        f"OpenCode session.error at {base_url!r}: {name or 'unknown'}: {msg}"
    )


# --- Error classification + payload translation ----------------------------


def _classify_opencode_error(exc: Exception, base_url: str) -> Exception:
    """Map an ``opencode-ai`` / network exception to airframe's taxonomy.

    The Stainless-generated SDK raises ``opencode_ai.APIError``
    subclasses on HTTP status codes (401 / 403 / 404 / 408 / 409 /
    422 / 429 / >=500) and ``opencode_ai.APIConnectionError`` on
    network failure. Reached only from :meth:`list_models` after
    :meth:`_ensure_client` has succeeded — so the SDK is guaranteed
    importable here.
    """
    from opencode_ai import APIConnectionError, APIError, APIStatusError

    if isinstance(exc, APIConnectionError):
        return RuntimeServerStartError(
            f"OpenCode server at {base_url!r} is unreachable. Run "
            "`opencode serve` (or set OPENCODE_SERVER_URL to a server "
            "you've already started). Original error: " + str(exc)
        )
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        if status == 401 or status == 403:
            return RuntimeAuthError(
                f"OpenCode server at {base_url!r} rejected credentials "
                f"(HTTP {status}). Check OPENCODE_SERVER_USERNAME / "
                "OPENCODE_SERVER_PASSWORD."
            )
        if status == 429 or (isinstance(status, int) and status >= 500):
            return RuntimeTransientError(f"OpenCode server returned HTTP {status}: {exc}")
        return RuntimeProtocolError(f"OpenCode server returned HTTP {status}: {exc}")
    if isinstance(exc, APIError):
        return RuntimeProtocolError(f"OpenCode SDK error: {exc}")
    return exc


def _models_from_provider_payload(payload: Any, airframe_provider_id: str) -> list[ModelInfo]:
    """Flatten ``client.app.providers()`` output into :class:`ModelInfo`s.

    The 0.1.0a36 response shape (``AppProvidersResponse``) is::

        {
          "default": {provider_id: model_id, ...},
          "providers": [
            {
              "id": str, "name": str, "env": [str, ...],
              "models": {model_id: Model(...), ...},
              ...
            },
            ...
          ],
        }

    We flatten the ``models`` dict on each provider into a row tagged
    with the airframe-facing ``PROVIDER_ID``; the upstream provider
    ID lands on ``raw["provider"]`` so consumers that need to route a
    turn through a specific backend can see what the server exposes.
    Unknown / malformed entries are skipped — a single bad row
    shouldn't blank the whole catalog.
    """
    providers: list[Any] = []
    if isinstance(payload, dict):
        providers = payload.get("providers") or []  # type: ignore[assignment]
    else:
        providers = getattr(payload, "providers", []) or []

    out: list[ModelInfo] = []
    for entry in providers:
        upstream_provider_id, models_iter = _provider_entry_to_models(entry)
        if upstream_provider_id is None:
            continue
        for model_id, model_entry in models_iter:
            info = _provider_model_to_modelinfo(
                model_id,
                model_entry,
                upstream_provider_id=upstream_provider_id,
                airframe_provider_id=airframe_provider_id,
            )
            if info is not None:
                out.append(info)
    return out


def _provider_entry_to_models(entry: Any) -> tuple[str | None, list[tuple[str, Any]]]:
    """Pull ``(upstream_provider_id, [(model_id, model_entry), ...])`` from one row.

    ``models`` is a ``Dict[str, Model]`` in 0.1.0a36 — we yield
    ``(key, value)`` pairs so the model_id is available even when the
    Model entry's own ``id`` field is missing.
    """
    if isinstance(entry, dict):
        provider_id = entry.get("id") or entry.get("provider_id")
        models = entry.get("models") or {}
    else:
        provider_id = getattr(entry, "id", None) or getattr(entry, "provider_id", None)
        models = getattr(entry, "models", None) or {}
    if not isinstance(provider_id, str) or not provider_id:
        return None, []
    if isinstance(models, dict):
        return provider_id, list(models.items())
    if isinstance(models, list):
        # Defensive: some hypothetical future SDK shape might emit a list.
        return provider_id, [(getattr(m, "id", "") or "", m) for m in models]
    return provider_id, []


def _provider_model_to_modelinfo(
    model_id: str,
    entry: Any,
    *,
    upstream_provider_id: str,
    airframe_provider_id: str,
) -> ModelInfo | None:
    """Translate one upstream-provider model entry into :class:`ModelInfo`."""
    if isinstance(entry, dict):
        entry_id = entry.get("id")
        display_name = entry.get("name") or entry_id or model_id
        limit = entry.get("limit") or {}
        context_window = limit.get("context") if isinstance(limit, dict) else None
        cost = entry.get("cost") or {}
        in_cost = cost.get("input") if isinstance(cost, dict) else None
        out_cost = cost.get("output") if isinstance(cost, dict) else None
        capabilities: list[str] = []
        if entry.get("reasoning"):
            capabilities.append("reasoning")
        if entry.get("tool_call"):
            capabilities.append("tools")
        if entry.get("attachment"):
            capabilities.append("vision")
    else:
        entry_id = getattr(entry, "id", None)
        display_name = getattr(entry, "name", None) or entry_id or model_id
        limit = getattr(entry, "limit", None)
        context_window = getattr(limit, "context", None) if limit is not None else None
        cost = getattr(entry, "cost", None)
        in_cost = getattr(cost, "input", None) if cost is not None else None
        out_cost = getattr(cost, "output", None) if cost is not None else None
        capabilities = []
        if getattr(entry, "reasoning", False):
            capabilities.append("reasoning")
        if getattr(entry, "tool_call", False):
            capabilities.append("tools")
        if getattr(entry, "attachment", False):
            capabilities.append("vision")
    resolved_id = entry_id if isinstance(entry_id, str) and entry_id else model_id
    if not isinstance(resolved_id, str) or not resolved_id:
        return None
    # OpenCode reports cost per million tokens; airframe's ModelInfo
    # expects per 1k. Divide by 1000.
    return ModelInfo(
        id=resolved_id,
        display_name=display_name if isinstance(display_name, str) else resolved_id,
        provider_id=airframe_provider_id,
        context_window=int(context_window) if isinstance(context_window, (int, float)) else None,
        pricing_input_per_1k_usd=(in_cost / 1000.0) if isinstance(in_cost, (int, float)) else None,
        pricing_output_per_1k_usd=(out_cost / 1000.0)
        if isinstance(out_cost, (int, float))
        else None,
        capabilities=frozenset(capabilities),
        raw={"provider": upstream_provider_id, "model_key": model_id},
    )


# Re-exported so consumers can write `from airframe.adapters.opencode_server
# import OpenCodeServerOptions` even though the dataclass lives in
# airframe.options — same convention every other adapter follows.
__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_USERNAME",
    "OpenCodeServerOptions",
    "OpenCodeServerRuntime",
    "OpenCodeServerSession",
]
