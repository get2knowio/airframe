"""``CodexRuntime`` — :class:`AgentRuntime` over the OpenAI Codex SDK.

Wraps :class:`openai_codex_sdk.Codex` to route OpenAI / GPT-family
work through the user's ChatGPT Plus subscription (or an
``OPENAI_API_KEY``) via the official ``openai-codex-sdk`` package.
The SDK spawns the ``codex`` CLI subprocess per turn; Maverick
doesn't allocate ports, juggle passwords, or maintain any client
code.

**Why this exists alongside CopilotRuntime.** Different auth path:
ChatGPT Plus subscription instead of GitHub Copilot. Useful as a
secondary binding when Copilot is rate-limited or the user only has
a ChatGPT Plus seat.

**Auth.** Three options, checked in order:

1. Explicit ``api_key=`` constructor argument — exported as
   ``CODEX_API_KEY`` for the subprocess.
2. ``OPENAI_API_KEY`` / ``CODEX_API_KEY`` env vars (the SDK
   inherits ``os.environ`` for the subprocess by default, so these
   "just work" if set).
3. ``~/.local/share/opencode/auth.json::openai.key`` — the API key
   minted by ``opencode auth login openai`` when the user already
   has opencode auth configured.
4. Implicit fallback: the ``codex`` CLI reads
   ``~/.codex/auth.json`` directly when present (created by
   ``codex login``). No work for us — the subprocess just uses it.

**Structured output.** First-class: the Codex CLI accepts an
``--output-schema`` flag that constrains the final response to a
JSON Schema. We pass ``schema.model_json_schema()`` via
:attr:`TurnOptions.outputSchema` and parse :attr:`Turn.final_response`
as JSON — no tool-forcing pattern needed.

**Lifecycle.** ``Codex()`` is lightweight (no subprocess yet).
``start_thread()`` returns a lightweight ``Thread`` object. Each
``thread.run()`` actually spawns the ``codex exec`` subprocess,
drains its JSONL event stream, and returns a typed ``Turn``. So
there's no persistent server to manage. ``reset()`` drops the
current thread (the next ``execute()`` starts a fresh one);
``close()`` is equivalent.

**Claude is not routed here.** Codex is OpenAI-only by design.
:meth:`validate_binding` rejects any ``model_id`` starting with
``claude-``.

**Cost.** ``Turn.usage`` exposes ``input_tokens``,
``output_tokens``, and ``cached_input_tokens``. The Codex CLI does
not return per-call ``cost_usd``; we look up a per-model rate from
a stub pricing map (real pricing migrates to
``runtime/pricing.py`` in a later phase). Models we haven't priced
report ``cost_usd=None``; tokens are always populated.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from pydantic import BaseModel

from airframe.cost import CostRecord
from airframe.errors import (
    AgentRuntimeError,
    RuntimeAuthError,
    RuntimeCancelledError,
    RuntimeModelNotFoundError,
    RuntimeServerStartError,
    RuntimeStructuredOutputError,
    RuntimeTransientError,
    UnsupportedFeatureError,
)
from airframe.events import ReasoningDelta, RuntimeEvent, TextDelta, TurnComplete
from airframe.features import Feature
from airframe.inputs import Prompt
from airframe.models import ModelInfo
from airframe.options import CodexOptions
from airframe.protocol import (
    AgentRuntime,
    AgentSession,
    ProviderModel,
    RuntimeResult,
    UnsupportedBindingError,
)
from airframe.sessions import (
    _MCP_TRANSPORT_TO_FEATURE,
    _check_budget_supported,
    _check_hooks_supported,
    _check_permission_supported,
    _check_provider_options,
    _enforce_budget_pre_turn,
    _fire_hook_event,
    _split_prompt_parts,
)
from airframe.thinking import ThinkingMode
from airframe.tools import FunctionTool, McpServerRef

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from airframe.hooks import HookEvent
    from airframe.options import ProviderOptions
    from airframe.permission import PermissionCallback

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Default Codex model when no binding is specified. ``gpt-5-codex``
#: is the v0 default — the standard codex model. Selected per-call
#: via ``ProviderModel.model_id``.
DEFAULT_CODEX_MODEL = "gpt-5-codex"

#: Path to the opencode auth file when present.
DEFAULT_AUTH_PATH = Path.home() / ".local" / "share" / "opencode" / "auth.json"

#: Canonical provider ID this adapter serves. Distinguished from a
#: hypothetical ``openai`` provider (direct OpenAI API) — the codex
#: route goes through the codex CLI subprocess.
PROVIDER_ID = "codex"

#: Per-model metadata enrichment. Pricing covers the codex tier; the
#: live OpenAI ``/v1/models`` endpoint surfaces *every* model the
#: account can access (gpt-4, gpt-5, embeddings, etc.) — we only enrich
#: the codex-shaped subset here, and ``list_models()`` filters to that.
_CODEX_METADATA: dict[str, tuple[str, int, float, float]] = {
    # id → (display_name, context_window, input_per_1k, output_per_1k)
    "gpt-5-codex": ("GPT-5 Codex", 256_000, 0.0015, 0.0060),
    "gpt-5-codex-mini": ("GPT-5 Codex Mini", 128_000, 0.00025, 0.0010),
    "o5-codex": ("o5 Codex", 200_000, 0.0030, 0.0120),
}

#: Legacy pricing alias kept for ``_compute_cost_usd``.
_PRICING: dict[str, tuple[float, float]] = {k: (v[2], v[3]) for k, v in _CODEX_METADATA.items()}


def _resolve_api_key(api_key: str | None) -> str | None:
    """Resolve the OpenAI API key from explicit arg → env → opencode auth.json.

    Returns ``None`` when no API key is found in any of the explicit
    sources. That's a valid state: the codex CLI itself reads
    ``~/.codex/auth.json`` (populated by ``codex login``) when no env
    var is set. We only raise :class:`RuntimeAuthError` when the
    subprocess actually fails for auth reasons.
    """
    if api_key:
        return api_key
    env = os.environ.get("OPENAI_API_KEY") or os.environ.get("CODEX_API_KEY")
    if env:
        return env
    auth_path = Path(os.environ.get("OPENCODE_AUTH_PATH") or DEFAULT_AUTH_PATH)
    if auth_path.exists():
        try:
            data = json.loads(auth_path.read_text())
            key = (data.get("openai") or {}).get("key")
            if isinstance(key, str) and key:
                return key
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug(
                "codex_runtime.auth_file_unreadable path=%s error=%s",
                auth_path,
                exc,
            )
    return None


def _translate_thinking_for_codex(thinking: ThinkingMode) -> str | None:
    """Map :data:`ThinkingMode` onto Codex's ``ModelReasoningEffort``.

    Codex's ``ModelReasoningEffort`` literal is
    ``"minimal" | "low" | "medium" | "high"`` — identical to airframe's
    :data:`ReasoningEffort`. No coercion needed; the literal passes
    through verbatim.

    Returns:
        The literal effort string, or ``None`` for ``None`` /
        ``"disabled"`` (caller skips ``modelReasoningEffort`` and
        accepts Codex's own default).

    Raises:
        UnsupportedFeatureError: ``thinking`` is a ``dict`` (the
        ``{"budget_tokens": int}`` shape is Claude-only — Codex has no
        token-budget channel for reasoning).
    """
    if thinking is None or thinking == "disabled":
        return None
    if isinstance(thinking, str):
        return thinking
    if isinstance(thinking, dict):
        raise UnsupportedFeatureError(
            "codex: thinking=<dict> (budget_tokens shape) is Claude-only; "
            "pass a literal effort string ('minimal'|'low'|'medium'|'high') instead",
            feature=Feature.REASONING_BUDGET_TOKENS,
        )
    raise UnsupportedFeatureError(
        f"codex: unsupported thinking= value {thinking!r}",
        feature=Feature.REASONING_EFFORT,
    )


#: Sentinel :attr:`PermissionRequest.tool_name` used when Codex
#: resolves its session-wide :class:`ApprovalMode` from the user's
#: :class:`~airframe.permission.PermissionCallback`. The wildcard
#: signals "all tools, session-wide" — the granularity Codex
#: actually supports.
CODEX_SESSION_PERMISSION_TOOL = "*"


_PERMISSION_DECISION_TO_APPROVAL_MODE: dict[str, str] = {
    "allow": "never",
    "deny": "untrusted",
    "defer": "on-request",
}


async def _resolve_codex_approval_policy(
    callback: PermissionCallback,
) -> str:
    """Derive Codex's :data:`ApprovalMode` from one
    :class:`~airframe.permission.PermissionCallback` invocation.

    Codex's permission surface is **session-wide**: the
    :attr:`ThreadOptions.approval_policy` enum is baked at
    :meth:`Codex.start_thread` time and never re-evaluated per tool
    call. To bridge airframe's per-call
    :class:`~airframe.permission.PermissionCallback` contract onto
    that shape, we call the user's callback **once** at first
    :meth:`execute` with a sentinel
    :class:`~airframe.permission.PermissionRequest`
    (``tool_name="*"``, empty ``tool_args``, an explanatory
    ``reason``) and translate the returned
    :data:`~airframe.permission.PermissionDecision`:

    * ``"allow"`` → ``"never"`` — the CLI never prompts for
      approval; everything is auto-approved.
    * ``"deny"`` → ``"untrusted"`` — strictest policy; the CLI
      denies most operations.
    * ``"defer"`` → ``"on-request"`` — Codex's default; the CLI
      prompts the user per call.

    The result is cached on the session for the rest of its
    lifetime. Consumers needing per-call interception should use
    Claude or Copilot, whose SDKs expose a per-call permission
    channel.
    """
    from airframe.permission import PermissionRequest

    sentinel = PermissionRequest(
        tool_name=CODEX_SESSION_PERMISSION_TOOL,
        tool_args={},
        reason=(
            "codex session policy resolution: airframe is invoking "
            "your PermissionCallback once at session start to derive a "
            "session-wide Codex ApprovalMode. The Codex Python SDK "
            "exposes only session-wide approval, not per-call. The "
            "decision maps to ApprovalMode as: 'allow'→'never', "
            "'deny'→'untrusted', 'defer'→'on-request' (Codex's "
            "default per-call prompting)."
        ),
    )
    decision = await callback.handle(sentinel)
    mapped = _PERMISSION_DECISION_TO_APPROVAL_MODE.get(decision)
    if mapped is None:  # pragma: no cover — Literal narrows this
        raise UnsupportedFeatureError(
            f"codex: PermissionCallback returned unrecognised decision "
            f"{decision!r}; expected one of 'allow', 'deny', 'defer'.",
            feature=Feature.PERMISSION_CALLBACK,
        )
    return mapped


def _build_codex_input(
    full_text: str,
    images: list[Any],
    files: list[Any],
) -> Any:
    """Build the ``input`` argument for ``Thread.run`` / ``Thread.run_streamed``.

    Plain string when no attachments; a list of ``TextInput`` +
    ``LocalImageInput`` parts when one or more images attach. File
    inputs get appended to the text as ``Attached file: <path>`` hints
    — the Codex CLI's working-directory sandbox lets the agent open the
    file with its built-in shell tools.

    **Path-only.** Codex's ``LocalImageInput`` accepts only a
    filesystem path; the CLI subprocess reads the file directly.
    :class:`ImageInput(bytes_=)` and :class:`ImageInput(url=)` raise
    :class:`UnsupportedFeatureError` here — the consumer should write
    the bytes to disk (e.g. via :class:`tempfile.NamedTemporaryFile`)
    and pass ``path=`` instead.
    """
    for img in images:
        if img.path is None:
            kind = "bytes_" if img.bytes_ is not None else "url"
            raise UnsupportedFeatureError(
                f"codex: ImageInput({kind}=...) has no Codex SDK channel — "
                f"the codex CLI's LocalImageInput accepts only a filesystem "
                f"path. Write the bytes to disk (tempfile.NamedTemporaryFile) "
                f"and pass path= instead.",
                feature=Feature.VISION_INPUT,
            )
    if files:
        hints = "\n\n".join(
            f"Attached file: {f.path}" + (f" (media_type: {f.media_type})" if f.media_type else "")
            for f in files
        )
        full_text = f"{full_text}\n\n{hints}" if full_text else hints
    if not images:
        return full_text
    from openai_codex_sdk.types import LocalImageInput, TextInput

    parts: list[Any] = []
    if full_text:
        parts.append(TextInput(type="text", text=full_text))
    for img in images:
        parts.append(LocalImageInput(type="local_image", path=img.path))
    return parts


def _compute_cost_usd(model_id: str, *, input_tokens: int, output_tokens: int) -> float | None:
    """Look up per-1K-token pricing and compute USD cost."""
    rates = _PRICING.get(model_id)
    if rates is None:
        return None
    in_rate, out_rate = rates
    return round((input_tokens / 1000.0) * in_rate + (output_tokens / 1000.0) * out_rate, 6)


def _codex_options_fingerprint(po: CodexOptions | None) -> str:
    """Deterministic fingerprint of the :class:`CodexOptions` value.

    All four populated fields bake into :class:`ThreadOptions` at
    :meth:`Codex.start_thread` / :meth:`resume_thread` time, so a
    change between turns must force a thread rebuild. Value-based —
    frozen dataclass instances with the same fields fingerprint
    identically.
    """
    if po is None:
        return "__no_provider_options__"
    return (
        f"wd={po.working_directory!r}|"
        f"addl={po.additional_directories!r}|"
        f"net={po.network_access_enabled}|"
        f"ws={po.web_search_enabled}"
    )


class CodexRuntime(AgentRuntime):
    """One Codex SDK client per runtime instance.

    Args:
        model: Default Codex model identifier used when ``execute()``
            is called without a ``ProviderModel`` override. Honours
            ``CODEX_MODEL_OVERRIDE`` env var if set for testing.
        api_key: Optional explicit OpenAI API key. When ``None``
            (default), auth resolves via ``OPENAI_API_KEY`` /
            ``CODEX_API_KEY`` env vars → opencode auth.json → falls
            back to ``~/.codex/auth.json`` via the CLI subprocess.
        codex_path: Optional override for the ``codex`` CLI path.
        sandbox_mode: Sandbox mode passed to the codex CLI. Defaults
            to ``read-only`` — typed-output workflows shouldn't be
            writing files. Override to ``workspace-write`` for
            agents that need filesystem access.
        skip_git_repo_check: Skip the CLI's "are you in a git repo?"
            guard. ``True`` by default since Maverick agents
            operate against arbitrary working directories.
    """

    label = "codex"

    #: Canonical provider ID for this adapter.
    PROVIDER_ID: ClassVar[str] = PROVIDER_ID

    #: Vendor SDK that must be importable for this adapter to work.
    REQUIRES_PACKAGE: ClassVar[str] = "openai_codex_sdk"

    #: pip extra that brings the vendor SDK in.
    EXTRA_NAME: ClassVar[str] = "codex"

    #: Features this runtime exposes today.
    #:
    #: * ``STRUCTURED_OUTPUT_JSON_SCHEMA`` — wired via the native
    #:   ``--output-schema`` flow (Phase 0).
    #: * ``STREAMING`` — wired via :class:`CodexAgentSession` using
    #:   :meth:`Thread.run_streamed` and per-item delta diffing on
    #:   ``ItemUpdatedEvent`` / ``ItemCompletedEvent`` (Phase 1,
    #:   Iteration F).
    #: * ``SESSION_RESUME`` — wired via :meth:`Codex.resume_thread`;
    #:   :meth:`AgentRuntime.session` accepts ``resume=<thread_id>``
    #:   (Phase 1, Iteration F). The thread ID surfaces on
    #:   :attr:`AgentSession.id` after the first turn populates the
    #:   underlying :attr:`Thread.id`.
    #: * ``CANCEL`` — wired via :class:`AbortController` /
    #:   :attr:`TurnOptions.signal` (Phase 1, Iteration F).
    #: * ``REASONING_EFFORT`` — wired via
    #:   :attr:`ThreadOptions.model_reasoning_effort`
    #:   (``"minimal" | "low" | "medium" | "high"``). Codex matches our
    #:   :data:`ReasoningEffort` literal exactly, so no coercion
    #:   needed. Baked at Thread-creation time, so a ``thinking=``
    #:   change between turns rebuilds the Thread (Phase 2, Iteration B).
    #:
    #: * ``VISION_INPUT`` — wired via :class:`LocalImageInput` parts on
    #:   :meth:`Thread.run` / :meth:`Thread.run_streamed`. Path-only in
    #:   v0 (Codex's ``LocalImageInput`` accepts only a filesystem
    #:   path); bytes/URL raise :class:`UnsupportedFeatureError`
    #:   (Phase 2, Iteration C).
    #: * ``FILE_INPUT`` — wired by appending an ``Attached file:
    #:   <path>`` hint to the prompt text. The Codex CLI's
    #:   working-directory sandbox lets the agent read the file via
    #:   its built-in shell tools (Phase 2, Iteration C).
    #:
    #: ``REASONING_BUDGET_TOKENS`` stays False — Codex uses the
    #: literal enum, not a token budget. Pass a literal effort string
    #: instead.
    #:
    #: * ``PERMISSION_CALLBACK`` — wired by deriving Codex's
    #:   session-wide :attr:`ThreadOptions.approval_policy` from a
    #:   **single** up-front call to the user's
    #:   :class:`~airframe.permission.PermissionCallback`. The
    #:   callback fires once with a sentinel
    #:   :class:`~airframe.permission.PermissionRequest`
    #:   (``tool_name="*"``) at first ``execute()``; the returned
    #:   :data:`~airframe.permission.PermissionDecision` maps to
    #:   :data:`ApprovalMode`: ``"allow"`` → ``"never"`` (auto-approve
    #:   everything), ``"deny"`` → ``"untrusted"`` (strictest mode),
    #:   ``"defer"`` → ``"on-request"`` (Codex's default per-call
    #:   prompting). **Per-call interception isn't possible** through
    #:   the Codex Python SDK — the policy is session-wide. Consumers
    #:   needing per-call decisions should use Claude or Copilot
    #:   (Phase 5, Iteration B).
    #: * ``LIFECYCLE_HOOKS`` — wired by synthesising
    #:   :class:`~airframe.hooks.HookEvent` from the
    #:   :class:`ItemStartedEvent` / :class:`ItemCompletedEvent`
    #:   stream :meth:`Thread.run_streamed` exposes. Emittable kinds:
    #:   ``session_start``, ``session_end``, ``user_prompt_submit``,
    #:   ``pre_tool_use``, ``post_tool_use``, ``tool_failure``.
    #:   ``pre_compact`` / ``rate_limit`` are **not emittable** —
    #:   the Codex SDK has no equivalent events today (Phase 5,
    #:   Iteration C).
    SUPPORTED_FEATURES: ClassVar[frozenset[Feature]] = frozenset(
        {
            Feature.STRUCTURED_OUTPUT_JSON_SCHEMA,
            Feature.STREAMING,
            Feature.SESSION_RESUME,
            Feature.CANCEL,
            Feature.REASONING_EFFORT,
            Feature.VISION_INPUT,
            Feature.FILE_INPUT,
            Feature.PERMISSION_CALLBACK,
            Feature.LIFECYCLE_HOOKS,
            # Phase 5 Iteration D: client-side accumulation on every
            # turn. Both caps enforced at turn boundary in v0
            # (mid-turn interrupt is additive later).
            Feature.BUDGET_USD_CAP,
            Feature.BUDGET_TURN_CAP,
        }
    )

    #: The :class:`~airframe.hooks.HookEventKind` literals this
    #: adapter can emit through ``on_event=``. The Codex Python
    #: SDK has no compaction / rate-limit events; those kinds
    #: stay unemitted.
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
        api_key: str | None = None,
        codex_path: str | None = None,
        sandbox_mode: str = "read-only",
        skip_git_repo_check: bool = True,
    ) -> None:
        self._default_model = (
            model or os.environ.get("CODEX_MODEL_OVERRIDE") or DEFAULT_CODEX_MODEL
        )
        self._api_key_override = api_key
        self._codex_path = codex_path or os.environ.get("CODEX_PATH")
        self._sandbox_mode = sandbox_mode
        self._skip_git_repo_check = skip_git_repo_check

        self._client: Any | None = None  # openai_codex_sdk.Codex
        # Phase 1 Iteration G: per-conversation Thread state moved off
        # the runtime onto CodexAgentSession.

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
        # ephemeral — the underlying Codex Thread is built and dropped
        # per call. The Codex client (runtime-owned, cheap — no
        # subprocess until thread.run() spawns one) is reused across
        # calls.
        del persona  # accepted in the protocol but not consumed by Codex
        sess = self.session(system=system, model=model)
        try:
            return await sess.execute(prompt, schema=schema, thinking=thinking, timeout=timeout)
        finally:
            await sess.close()

    async def reset(self) -> None:
        # Phase 1 Iteration G: the runtime no longer caches a Thread.
        # Sessions own that. No scope-bound state to drop on the
        # runtime; kept as a no-op for protocol completeness.
        return None

    async def close(self) -> None:
        # Phase 1 Iteration G: the Codex client is cheap (no subprocess
        # until thread.run() spawns one), and the runtime is otherwise
        # sessionless. Drop the cached client reference for
        # post-close()-then-reuse-the-runtime scenarios.
        self._client = None

    def validate_binding(self, binding: ProviderModel) -> bool:
        if binding.provider_id != self.PROVIDER_ID:
            return False
        # Codex is OpenAI-only. Anthropic bindings route through
        # ClaudeCodeRuntime / AnthropicRuntime.
        return not binding.model_id.startswith("claude-")

    def supports(self, feature: Feature, model: ProviderModel | None = None) -> bool:
        return feature in self.SUPPORTED_FEATURES

    def unwrap(self, cls: type[T]) -> T:
        from openai_codex_sdk import Codex, Thread

        if isinstance(self, cls):
            return self  # type: ignore[return-value]
        if cls is Codex:
            if self._client is None:
                raise TypeError(
                    "CodexRuntime.unwrap(Codex): no client exists yet — call execute() first."
                )
            return self._client  # type: ignore[return-value]
        if cls is Thread:
            # Phase 1 Iteration G moved the per-conversation Thread off
            # the runtime onto CodexAgentSession.
            raise TypeError(
                "CodexRuntime no longer owns a Thread — sessions do. "
                "Open a session with `sess = runtime.session(...)`, run a turn, "
                "then call `sess.unwrap(Thread)`."
            )
        raise TypeError(
            f"CodexRuntime cannot unwrap to {cls!r}; supported types are "
            f"CodexRuntime and openai_codex_sdk.Codex. Vendor session objects "
            f"live on AgentSession — use session.unwrap(NativeType)."
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
        """Open a bespoke :class:`CodexAgentSession`.

        Phase 1 Iteration F replaces the
        :class:`~airframe.sessions._ThinAgentSession` placeholder with
        a session that owns its own :class:`Thread` lifecycle: real
        streaming via :meth:`Thread.run_streamed`, native session
        resume via :meth:`Codex.resume_thread`, and cancellation via
        :class:`AbortController` plumbed into
        :attr:`TurnOptions.signal`.

        Args:
            resume: Vendor-assigned Codex thread ID to resume — the
                value surfaced on a prior :attr:`Thread.id` /
                :class:`ThreadStartedEvent`. ``None`` opens a fresh
                thread.
            system: Optional system prompt. Codex has no native
                system-message setting; the adapter concatenates it
                onto every :meth:`execute` / :meth:`stream` prompt
                exactly as the runtime's :meth:`execute` does today.
            model: Default :class:`ProviderModel` for every turn in
                the session.
            tools: Accepted for protocol parity but the Codex Python
                SDK has no tool-registration channel; any non-None
                value raises
                :class:`~airframe.errors.UnsupportedFeatureError`.
                Wire tools through the ``codex`` CLI's config file
                instead. Unlike the other adapters whose
                ``TOOLS_FUNCTION=False`` was a "not yet" gate during
                Iterations B/C, Codex's decline is permanent — its
                Python SDK simply does not surface a tool-registration
                API and consumers branching on
                ``runtime.supports(Feature.TOOLS_FUNCTION)`` should
                treat Codex as a tools-incapable runtime.
            mcp_servers: Accepted for protocol parity but the Codex
                Python SDK has no programmatic MCP-registration
                channel — non-empty list raises
                :class:`~airframe.errors.UnsupportedFeatureError`
                pointing consumers at the ``[[mcp_servers]]`` block
                in ``~/.codex/config.toml`` instead. The decline is
                **permanent** (Phase 4 Iteration D); the
                :attr:`~airframe.errors.UnsupportedFeatureError.feature`
                attribute carries the first ref's transport so
                consumer code branching on
                :data:`~airframe.features.Feature.TOOLS_MCP_STDIO` /
                :data:`~airframe.features.Feature.TOOLS_MCP_HTTP` /
                :data:`~airframe.features.Feature.TOOLS_MCP_SSE`
                still works.
            on_permission: Phase 5 scaffolding accepted by the
                signature; non-None raises
                :class:`~airframe.errors.UnsupportedFeatureError`
                until Phase 5 Iteration B wires Codex's session-wide
                :attr:`Thread.approval_policy` (the user's callback
                fires once to derive the enum value, since Codex has
                no per-call permission channel).
            on_event: Phase 5 scaffolding accepted by the signature;
                non-None raises until Phase 5 Iteration C synthesises
                :class:`~airframe.hooks.HookEvent` from
                :class:`ItemStartedEvent` / :class:`ItemCompletedEvent`
                on the thread stream.
            provider_options: Optional :class:`CodexOptions` namespace
                carrying Codex-only knobs. Four populated fields:
                ``working_directory``, ``additional_directories``,
                ``network_access_enabled``, ``web_search_enabled`` —
                all baked into :class:`ThreadOptions` at
                :meth:`Codex.start_thread` / :meth:`resume_thread`
                time, so a change between turns rebuilds the thread.
                Passing :class:`ClaudeOptions` / :class:`CopilotOptions`
                / :class:`OpenAICompatOptions` here raises
                :class:`UnsupportedFeatureError`.
        """
        if tools:
            raise UnsupportedFeatureError(
                f"{self.label}: function tools cannot be wired through the "
                f"Codex Python SDK — its surface has no tool-registration "
                f"channel. Configure tools through the codex CLI's config "
                f"file (`~/.codex/config.toml`) instead. "
                f"Check runtime.supports(Feature.TOOLS_FUNCTION) before "
                f"passing tools=.",
                feature=Feature.TOOLS_FUNCTION,
            )
        if mcp_servers:
            # Phase 4 Iteration D — Codex declines MCP registration the
            # same way it declines function tools: its Python SDK
            # surface has no MCP-registration channel. Point consumers
            # at the CLI config-file workaround instead of the generic
            # shared-helper message.
            first = mcp_servers[0]
            feature = _MCP_TRANSPORT_TO_FEATURE.get(first.transport, Feature.TOOLS_MCP_STDIO)
            raise UnsupportedFeatureError(
                f"{self.label}: MCP servers cannot be wired through the "
                f"Codex Python SDK — its surface has no programmatic "
                f"MCP-registration channel. Configure MCP servers through "
                f"the codex CLI's config file "
                f"(``~/.codex/config.toml``'s ``[[mcp_servers]]`` block) "
                f"instead. Check runtime.supports(Feature.TOOLS_MCP_STDIO) "
                f"before passing mcp_servers=.",
                feature=feature,
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
            expected_type=CodexOptions,
            adapter_label=self.label,
        )
        codex_options = provider_options if isinstance(provider_options, CodexOptions) else None
        model_id = self._resolve_model(model) if model is not None else self._default_model
        return CodexAgentSession(
            self,
            resume=resume,
            system=system,
            model_id=model_id,
            on_permission=on_permission,
            on_event=on_event,
            provider_options=codex_options,
        )

    async def list_models(self) -> list[ModelInfo]:
        """Return the codex-tier models from OpenAI's models endpoint.

        Hits OpenAI's ``/v1/models`` via :class:`AsyncOpenAI` using
        whichever API key the codex auth chain resolved. Filters to
        IDs we recognise as codex variants (the bare endpoint returns
        every model the account has access to, including ones the
        codex CLI doesn't actually run); the rest are dropped.
        """
        from openai import AsyncOpenAI

        api_key = _resolve_api_key(self._api_key_override)
        if api_key is None:
            raise RuntimeAuthError(
                "CodexRuntime.list_models() needs an OpenAI API key. "
                "Set OPENAI_API_KEY or pass api_key= explicitly."
            )
        client = AsyncOpenAI(api_key=api_key)
        try:
            try:
                page = await client.models.list()
            except Exception as exc:
                raise self._classify_openai_exception(exc) from exc

            out: list[ModelInfo] = []
            for entry in page.data:
                if entry.id not in _CODEX_METADATA:
                    continue
                display, ctx, in_per_1k, out_per_1k = _CODEX_METADATA[entry.id]
                out.append(
                    ModelInfo(
                        id=entry.id,
                        display_name=display,
                        provider_id=self.PROVIDER_ID,
                        context_window=ctx,
                        pricing_input_per_1k_usd=in_per_1k,
                        pricing_output_per_1k_usd=out_per_1k,
                        capabilities=frozenset(),
                        raw=entry,
                    )
                )
            return out
        finally:
            await client.close()

    # --- Internals ---------------------------------------------------------

    def _resolve_model(self, model: ProviderModel | None) -> str:
        if model is None:
            return self._default_model
        if not self.validate_binding(model):
            raise UnsupportedBindingError(
                f"CodexRuntime cannot serve {model.label!r}; "
                f"provider must be {self.PROVIDER_ID!r} "
                f"and the model_id must not start with 'claude-'"
            )
        return model.model_id

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        from openai_codex_sdk import Codex

        options: dict[str, Any] = {}
        if self._codex_path is not None:
            options["codexPathOverride"] = self._codex_path
        api_key = _resolve_api_key(self._api_key_override)
        if api_key is not None:
            options["apiKey"] = api_key
        try:
            self._client = Codex(options)
        except Exception as exc:
            raise self._classify_exception(exc) from exc
        return self._client

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
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        cache_read = int(getattr(usage, "cached_input_tokens", 0) or 0)
        return CostRecord(
            provider_id=self.PROVIDER_ID,
            model_id=model_id,
            cost_usd=_compute_cost_usd(
                model_id, input_tokens=input_tokens, output_tokens=output_tokens
            ),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=0,  # Codex SDK doesn't expose cache-write counts.
            finish="stop",
        )

    def _classify_openai_exception(self, exc: BaseException) -> Exception:
        """Map ``openai`` SDK exceptions raised by ``list_models()``.

        ``execute()`` uses the codex SDK and never sees these; ``list_models()``
        uses ``AsyncOpenAI`` directly against ``/v1/models``, so the openai
        SDK's exception taxonomy needs its own mapping.
        """
        from openai import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            AuthenticationError,
            NotFoundError,
            PermissionDeniedError,
            RateLimitError,
        )

        if isinstance(exc, AuthenticationError | PermissionDeniedError):
            return RuntimeAuthError(f"codex: auth: {exc}")
        if isinstance(exc, NotFoundError):
            return AgentRuntimeError(f"codex: models endpoint not found: {exc}")
        if isinstance(exc, RateLimitError | APITimeoutError | APIConnectionError):
            return RuntimeTransientError(f"codex: transient: {exc}")
        if isinstance(exc, APIStatusError):
            status = getattr(exc, "status_code", None)
            if status is not None and 500 <= status < 600:
                return RuntimeTransientError(f"codex: 5xx: {exc}")
            return AgentRuntimeError(f"codex: api error: {exc}")
        return AgentRuntimeError(f"codex: unexpected {type(exc).__name__}: {exc}")

    def _classify_exception(self, exc: BaseException) -> Exception:
        """Map Codex SDK exceptions onto Maverick's runtime hierarchy."""
        if isinstance(exc, UnsupportedBindingError):
            return exc

        from openai_codex_sdk.errors import (
            CodexAuthError,
            CodexExecError,
            CodexInstallError,
            CodexSdkError,
            EventParseError,
            ThreadRunError,
        )

        if isinstance(exc, CodexAuthError):
            return RuntimeAuthError(f"codex: auth: {exc}")
        if isinstance(exc, CodexInstallError):
            return RuntimeServerStartError(f"codex: install failure: {exc}")
        if isinstance(exc, FileNotFoundError):
            return RuntimeServerStartError(f"codex: CLI not found: {exc}")
        if isinstance(exc, EventParseError):
            return RuntimeStructuredOutputError(f"codex: malformed event stream: {exc}", body=None)
        if isinstance(exc, ThreadRunError):
            msg = str(exc).lower()
            if "auth" in msg or "401" in msg or "unauthorized" in msg or "credentials" in msg:
                return RuntimeAuthError(f"codex: auth: {exc}")
            if "rate" in msg or "429" in msg or "503" in msg or "timeout" in msg:
                return RuntimeTransientError(f"codex: transient: {exc}")
            if "model" in msg and (
                "not supported" in msg
                or "not available" in msg
                or "not found" in msg
                or "does not exist" in msg
            ):
                return RuntimeModelNotFoundError(f"codex: model unavailable on this binding: {exc}")
            if "schema" in msg or "json" in msg:
                return RuntimeStructuredOutputError(
                    f"codex: structured output failed: {exc}", body=None
                )
            return AgentRuntimeError(f"codex: thread run failed: {exc}")
        if isinstance(exc, CodexExecError):
            return RuntimeTransientError(f"codex: exec failure: {exc}")
        if isinstance(exc, CodexSdkError):
            return AgentRuntimeError(f"codex: sdk: {exc}")
        return AgentRuntimeError(f"codex: unexpected {type(exc).__name__}: {exc}")


class CodexAgentSession:
    """Bespoke :class:`~airframe.protocol.AgentSession` for the Codex SDK.

    Phase 1 Iteration F — fourth and final per-vendor session. Owns
    one :class:`Thread` for its lifetime; ``system`` / ``model`` /
    ``resume`` are session-fixed and baked into
    :meth:`Codex.start_thread` (or :meth:`resume_thread`) at first
    use. Schema can vary per turn — the Codex SDK puts
    ``outputSchema`` on :class:`TurnOptions`, not :class:`ThreadOptions`,
    so the same thread serves both plain-text and structured turns
    without rebuild.

    **Streaming.** :meth:`stream` uses :meth:`Thread.run_streamed`
    and translates Codex thread events into airframe events:

    * ``ItemUpdatedEvent`` / ``ItemCompletedEvent`` with an
      :class:`AgentMessageItem` → :class:`TextDelta` carrying the
      *tail* (the bytes not previously emitted for that item id).
    * Same shape for :class:`ReasoningItem` → :class:`ReasoningDelta`.
    * ``TurnCompletedEvent`` → captures :class:`Usage` for the
      trailing :class:`TurnComplete`.
    * ``TurnFailedEvent`` → raises through the runtime's classifier.

    Per-item tail tracking keeps :class:`TextDelta` instances
    appendable — concatenating every yielded delta reconstructs the
    full message text, matching the contract OpenAI-compat's
    chunk-stream sets.

    **Cancellation.** Each turn allocates a fresh
    :class:`AbortController`; :attr:`TurnOptions.signal` carries its
    signal into the underlying ``codex exec`` subprocess.
    :meth:`cancel` calls :meth:`AbortController.abort`; the awaiting
    turn raises :class:`AbortError`, which the session surfaces as
    :class:`~airframe.errors.RuntimeCancelledError`.

    **Resume.** ``session(resume=<thread_id>)`` forwards the ID into
    :meth:`Codex.resume_thread`; the underlying :attr:`Thread.id` is
    pre-populated so :attr:`AgentSession.id` is correct before the
    first turn (in contrast to fresh threads, where ``id`` is ``None``
    until ``thread.started`` fires).
    """

    def __init__(
        self,
        runtime: CodexRuntime,
        *,
        resume: str | None,
        system: str | None,
        model_id: str,
        on_permission: PermissionCallback | None = None,
        on_event: Callable[[HookEvent], None] | None = None,
        provider_options: CodexOptions | None = None,
    ) -> None:
        self._runtime = runtime
        self._resume = resume
        self._system = system
        self._model_id = model_id
        # Lazily constructed on first execute() / stream(); persists for
        # the session's lifetime since the Codex Thread is cheap and
        # multi-turn-safe.
        self._thread: Any | None = None
        # Effort baked into the current Thread, so we know when to
        # rebuild on a thinking= change between turns.
        self._thread_effort: str | None = None
        self._closed = False
        self._in_flight = False
        # Per-turn AbortController so cancel() can signal the in-flight
        # exec; replaced each turn.
        self._abort_controller: Any | None = None
        # Seeded from resume= so consumer code that branches on
        # session.id before the first turn sees the right value.
        self.id: str | None = resume
        # Phase 5 Iteration B: permission callback. Codex's
        # approval_policy is *session-wide*; we resolve it lazily on
        # first execute() by calling the user's callback once with a
        # sentinel request, then cache for the session's lifetime.
        # ``_approval_policy_resolved`` distinguishes "haven't asked
        # yet" from "asked and got None" (which doesn't happen today
        # but keeps the slot honest).
        self._on_permission: PermissionCallback | None = on_permission
        self._approval_policy: str | None = None
        self._approval_policy_resolved = False
        # Phase 5 Iteration C: lifecycle-hook observer. Codex has no
        # native subscription channel; the adapter synthesises
        # events from execute()/stream() boundaries and the
        # ItemStartedEvent/ItemCompletedEvent stream. session_start
        # fires once on first execute(); session_end fires on
        # close() (gated on having fired session_start first).
        self._on_event: Callable[[HookEvent], None] | None = on_event
        self._session_start_fired = False
        self._session_end_fired = False
        # Phase 5 Iteration D: per-session budget trackers. Both
        # caps enforced at turn boundary in v0; mid-turn interrupt
        # is additive later via the existing cancel() plumbing.
        self._cumulative_cost_usd: float = 0.0
        self._turn_count: int = 0
        # ProviderOptions — Codex-only knobs threaded into
        # ThreadOptions at start_thread() / resume_thread() time.
        # A namespace change between turns rebuilds the thread (the
        # cache key in _ensure_thread carries the fingerprint).
        self._provider_options: CodexOptions | None = provider_options

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
        text, images, files = _split_prompt_parts(
            prompt,
            adapter_label=self._runtime.label,
            supports_vision=True,
            supports_file=True,
        )
        await self._resolve_approval_policy()
        thread = self._ensure_thread(thinking=thinking)
        full_text = text if not self._system else f"{self._system}\n\n{text}"
        run_input = _build_codex_input(full_text, images, files)
        self._fire_session_start_if_needed()
        _fire_hook_event(
            self._on_event,
            "user_prompt_submit",
            session_id=self.id,
            payload={"prompt": text, "length": len(text)},
        )

        from openai_codex_sdk import AbortController
        from openai_codex_sdk.abort import AbortError

        controller = AbortController()
        self._abort_controller = controller

        turn_options: dict[str, Any] = {"signal": controller.signal}
        if schema is not None:
            turn_options["outputSchema"] = schema.model_json_schema()

        self._in_flight = True
        try:
            turn = await asyncio.wait_for(
                thread.run(run_input, turn_options),
                timeout=timeout,
            )
        except asyncio.CancelledError as exc:
            raise RuntimeCancelledError(f"{self._runtime.label}: cancelled") from exc
        except TimeoutError as exc:
            raise RuntimeTransientError(
                f"{self._runtime.label}: execute timed out after {timeout}s"
            ) from exc
        except AbortError as exc:
            raise RuntimeCancelledError(f"{self._runtime.label}: aborted") from exc
        except Exception as exc:
            raise self._runtime._classify_exception(exc) from exc
        finally:
            self._in_flight = False
            self._abort_controller = None

        self._update_id_from_thread()
        # Phase 5 Iteration C: synthesise per-tool hooks from
        # turn.items after the turn completes. execute() doesn't see
        # the per-event stream, so we replay item state at end-of-turn.
        if self._on_event is not None:
            self._fire_item_hooks_post_execute(turn)
        result = self._build_result(turn, schema=schema)
        self._turn_count += 1
        self._cumulative_cost_usd += result.cost.cost_usd or 0.0
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
        text, images, files = _split_prompt_parts(
            prompt,
            adapter_label=self._runtime.label,
            supports_vision=True,
            supports_file=True,
        )
        await self._resolve_approval_policy()
        thread = self._ensure_thread(thinking=thinking)
        full_text = text if not self._system else f"{self._system}\n\n{text}"
        run_input = _build_codex_input(full_text, images, files)
        self._fire_session_start_if_needed()
        _fire_hook_event(
            self._on_event,
            "user_prompt_submit",
            session_id=self.id,
            payload={"prompt": text, "length": len(text)},
        )

        from openai_codex_sdk import AbortController
        from openai_codex_sdk.abort import AbortError
        from openai_codex_sdk.errors import ThreadRunError
        from openai_codex_sdk.types import (
            AgentMessageItem,
            CommandExecutionItem,
            ItemCompletedEvent,
            ItemStartedEvent,
            ItemUpdatedEvent,
            McpToolCallItem,
            ReasoningItem,
            TurnCompletedEvent,
            TurnFailedEvent,
        )

        controller = AbortController()
        self._abort_controller = controller

        turn_options: dict[str, Any] = {"signal": controller.signal}
        if schema is not None:
            turn_options["outputSchema"] = schema.model_json_schema()

        try:
            streamed = await thread.run_streamed(run_input, turn_options)
        except AbortError as exc:
            self._abort_controller = None
            raise RuntimeCancelledError(f"{self._runtime.label}: aborted") from exc
        except Exception as exc:
            self._abort_controller = None
            raise self._runtime._classify_exception(exc) from exc

        # Per-item tail tracking so TextDeltas are appendable rather
        # than snapshot-style. Item IDs are stable across
        # ItemUpdatedEvent / ItemCompletedEvent for the same item.
        last_lengths: dict[str, int] = {}
        final_text = ""
        usage: Any = None
        failure: str | None = None
        self._in_flight = True
        # Tool items we've already emitted pre_tool_use for. The SDK
        # may send multiple ItemUpdatedEvents per item (status
        # in_progress→in_progress); we only want one pre_tool_use per
        # item id.
        tool_pre_fired: set[str] = set()
        try:
            async for event in streamed.events:
                if isinstance(event, ItemStartedEvent | ItemUpdatedEvent | ItemCompletedEvent):
                    item = event.item
                    if isinstance(item, AgentMessageItem):
                        tail = item.text[last_lengths.get(item.id, 0) :]
                        if tail:
                            last_lengths[item.id] = len(item.text)
                            yield TextDelta(text=tail)
                        if isinstance(event, ItemCompletedEvent):
                            final_text = item.text
                    elif isinstance(item, ReasoningItem):
                        tail = item.text[last_lengths.get(item.id, 0) :]
                        if tail:
                            last_lengths[item.id] = len(item.text)
                            yield ReasoningDelta(text=tail)
                    elif isinstance(item, CommandExecutionItem | McpToolCallItem):
                        # Phase 5 Iteration C — synthesise tool hooks
                        # from the item event stream. pre_tool_use on
                        # first sighting; post_tool_use / tool_failure
                        # when status flips to completed/failed.
                        if self._on_event is not None and item.id not in tool_pre_fired:
                            tool_pre_fired.add(item.id)
                            self._fire_codex_tool_hook(item, kind="pre_tool_use")
                        if isinstance(event, ItemCompletedEvent) and self._on_event is not None:
                            kind = (
                                "post_tool_use"
                                if getattr(item, "status", "") != "failed"
                                else "tool_failure"
                            )
                            self._fire_codex_tool_hook(item, kind=kind)
                elif isinstance(event, TurnCompletedEvent):
                    usage = event.usage
                elif isinstance(event, TurnFailedEvent):
                    failure = event.error.message
                    break
        except AbortError as exc:
            self._in_flight = False
            self._abort_controller = None
            raise RuntimeCancelledError(f"{self._runtime.label}: aborted") from exc
        except Exception as exc:
            self._in_flight = False
            self._abort_controller = None
            raise self._runtime._classify_exception(exc) from exc
        finally:
            # Reset state for the next turn; the abort controller is
            # turn-scoped (cancel() during the next stream() uses a
            # fresh one).
            self._in_flight = False
            self._abort_controller = None

        if failure is not None:
            raise self._runtime._classify_exception(ThreadRunError(failure))

        self._update_id_from_thread()

        structured: Any = None
        if schema is not None:
            if not final_text:
                raise RuntimeStructuredOutputError(
                    f"{self._runtime.label}: stream completed with empty final agent message",
                    body={},
                )
            try:
                structured = json.loads(final_text)
            except json.JSONDecodeError as exc:
                raise RuntimeStructuredOutputError(
                    f"{self._runtime.label}: final agent message was not valid JSON: {exc}",
                    body=final_text[:500],
                ) from exc

        cost = self._runtime._cost_from_usage(usage, model_id=self._model_id)
        result = RuntimeResult(
            text=final_text,
            structured=structured,
            cost=cost,
            finish="stop",
            raw=None,
        )
        self._turn_count += 1
        self._cumulative_cost_usd += result.cost.cost_usd or 0.0
        yield TurnComplete(result=result)

    async def cancel(self) -> None:
        # No-op when no turn is in flight — per the AgentSession contract.
        if not self._in_flight:
            return
        controller = self._abort_controller
        if controller is None:
            return
        try:
            controller.abort()
        except Exception as exc:  # noqa: BLE001 — cancellation never raises
            logger.debug("%s.session_abort_failed error=%s", self._runtime.label, exc)

    async def close(self) -> None:
        # Phase 5 Iteration C: synthesise session_end at close if
        # session_start ever fired and we haven't already emitted
        # session_end on this close. Repeat close() calls are
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
                session_id=self.id,
                payload={"model": self._model_id},
            )
        # Best-effort cancel of any in-flight turn so the underlying
        # subprocess winds down rather than leaks.
        await self.cancel()
        # Codex Thread holds no persistent state — dropping the
        # reference suffices. The runtime owns the Codex client.
        self._thread = None
        self._thread_effort = None
        self._abort_controller = None

    def unwrap(self, cls: type[T]) -> T:
        from openai_codex_sdk import Thread

        if isinstance(self, cls):
            return self  # type: ignore[return-value]
        if cls is Thread:
            if self._thread is None:
                raise TypeError(
                    "CodexAgentSession.unwrap(Thread): no thread exists yet — "
                    "call execute() or stream() first."
                )
            return self._thread  # type: ignore[return-value]
        raise TypeError(
            f"CodexAgentSession cannot unwrap to {cls!r}; supported types are "
            f"CodexAgentSession and openai_codex_sdk.Thread. "
            f"(Codex client lives on the runtime — call runtime.unwrap(Codex).)"
        )

    # --- Internals ---------------------------------------------------------

    async def _resolve_approval_policy(self) -> str | None:
        """Resolve Codex's session-wide :data:`ApprovalMode` from the
        user's permission callback.

        Codex bakes :attr:`ThreadOptions.approval_policy` at
        ``start_thread()`` time and never re-evaluates it. We ask the
        user's callback exactly once per session (on first
        :meth:`execute` / :meth:`stream`) with a sentinel
        :class:`~airframe.permission.PermissionRequest` whose
        ``reason`` explains the situation, then cache the resulting
        ``ApprovalMode`` for the rest of the session.

        Returns ``None`` when ``on_permission`` wasn't supplied — the
        Thread is built without ``approval_policy`` and Codex uses
        its own default.
        """
        if self._approval_policy_resolved:
            return self._approval_policy
        if self._on_permission is None:
            self._approval_policy = None
            self._approval_policy_resolved = True
            return None
        self._approval_policy = await _resolve_codex_approval_policy(self._on_permission)
        self._approval_policy_resolved = True
        return self._approval_policy

    def _ensure_thread(self, *, thinking: ThinkingMode = None) -> Any:
        effort = _translate_thinking_for_codex(thinking)
        # Cache key combines thinking-effort + provider_options
        # fingerprint — both bake into ThreadOptions at
        # start_thread() / resume_thread() time.
        po_fingerprint = _codex_options_fingerprint(self._provider_options)
        cache_key = f"effort={effort}|po={po_fingerprint}"
        if self._thread is not None and self._thread_effort == cache_key:
            return self._thread
        # Cache-key change between turns — rebuild the Thread. We drop
        # the old Thread reference; the Codex SDK Thread itself holds
        # no subprocess until run() spawns one.
        if self._thread is not None and self._thread_effort != cache_key:
            self._thread = None
        client = self._runtime._ensure_client()
        thread_options: dict[str, Any] = {
            "model": self._model_id,
            "sandboxMode": self._runtime._sandbox_mode,
            "skipGitRepoCheck": self._runtime._skip_git_repo_check,
        }
        if effort is not None:
            thread_options["modelReasoningEffort"] = effort
        # Phase 5 Iteration B: session-wide approval policy derived
        # lazily in _resolve_approval_policy() and cached on self.
        if self._approval_policy is not None:
            thread_options["approval_policy"] = self._approval_policy
        # ProviderOptions — Codex-only knobs not covered by the
        # portable surface. Field names map to ThreadOptions camelCase.
        po = self._provider_options
        if po is not None:
            if po.working_directory is not None:
                thread_options["workingDirectory"] = po.working_directory
            if po.additional_directories:
                thread_options["additionalDirectories"] = list(po.additional_directories)
            if po.network_access_enabled:
                thread_options["networkAccessEnabled"] = True
            if po.web_search_enabled:
                thread_options["webSearchEnabled"] = True
        try:
            if self._resume is not None:
                thread = client.resume_thread(self._resume, thread_options)
            else:
                thread = client.start_thread(thread_options)
        except Exception as exc:
            raise self._runtime._classify_exception(exc) from exc
        self._thread = thread
        self._thread_effort = cache_key
        return thread

    def _fire_session_start_if_needed(self) -> None:
        """Emit ``session_start`` once per session at first execute().

        Codex's SDK has no native ``session_start`` event; the
        adapter synthesises it from the first ``execute()`` /
        ``stream()`` call. Subsequent turns don't re-fire (a single
        session is one ``start`` / ``end`` pair).
        """
        if self._on_event is None or self._session_start_fired:
            return
        self._session_start_fired = True
        _fire_hook_event(
            self._on_event,
            "session_start",
            session_id=self.id,
            payload={
                "model": self._model_id,
                "resumed": self._resume is not None,
            },
        )

    def _fire_codex_tool_hook(self, item: Any, *, kind: str) -> None:
        """Translate a :class:`CommandExecutionItem` /
        :class:`McpToolCallItem` into a tool-use
        :class:`~airframe.hooks.HookEvent`.

        Field set is unified across the two item shapes:

        * ``tool_name`` — ``item.command[:64]`` for command items
          (the first 64 chars of the shell command — long enough to
          identify, short enough to fit in a log line);
          ``f"{server}/{tool}"`` for MCP items so consumers can
          distinguish servers in mixed sessions.
        * ``tool_call_id`` — :attr:`item.id`.
        * ``output`` / ``error`` — captured from
          :attr:`aggregated_output` / :attr:`exit_code` (command) or
          :attr:`result` / :attr:`error` (MCP) on the completion path.
        """
        payload: dict[str, Any] = {"tool_call_id": getattr(item, "id", "")}
        tool_type = getattr(item, "type", None)
        if tool_type == "command_execution":
            command = getattr(item, "command", "") or ""
            payload["tool_name"] = command[:64]
            if kind != "pre_tool_use":
                payload["exit_code"] = getattr(item, "exit_code", None)
                payload["output"] = getattr(item, "aggregated_output", "") or ""
        elif tool_type == "mcp_tool_call":
            server = getattr(item, "server", "") or ""
            tool = getattr(item, "tool", "") or ""
            payload["tool_name"] = f"{server}/{tool}" if server else tool
            if kind == "pre_tool_use":
                payload["arguments"] = getattr(item, "arguments", None)
            else:
                if kind == "tool_failure":
                    err = getattr(item, "error", None)
                    if err is not None:
                        payload["error"] = getattr(err, "message", str(err))
                else:
                    result = getattr(item, "result", None)
                    if result is not None:
                        payload["output"] = result
        else:  # pragma: no cover — defensive; only the two item types reach here
            payload["tool_name"] = str(tool_type)
        _fire_hook_event(
            self._on_event,
            kind,
            session_id=self.id,
            payload=payload,
        )

    def _fire_item_hooks_post_execute(self, turn: Any) -> None:
        """Emit pre/post tool hooks from ``turn.items`` after a
        non-streaming :meth:`execute` completes.

        execute() doesn't iterate the event stream — the
        ``thread.run()`` call returns a single :class:`Turn` after
        the entire turn finishes. To honour the
        :class:`LIFECYCLE_HOOKS` contract on the execute path, we
        replay the per-item state at end-of-turn: each command /
        MCP tool item fires pre_tool_use + post_tool_use (or
        tool_failure) back-to-back. Consumers observing pure
        hooks rather than streams see the same kind set on either
        path, in the same per-tool order.
        """
        from openai_codex_sdk.types import CommandExecutionItem, McpToolCallItem

        items = getattr(turn, "items", []) or []
        for item in items:
            if not isinstance(item, CommandExecutionItem | McpToolCallItem):
                continue
            self._fire_codex_tool_hook(item, kind="pre_tool_use")
            status = getattr(item, "status", "")
            kind = "tool_failure" if status == "failed" else "post_tool_use"
            self._fire_codex_tool_hook(item, kind=kind)

    def _update_id_from_thread(self) -> None:
        if self._thread is None:
            return
        live_id = getattr(self._thread, "id", None)
        if live_id:
            self.id = live_id

    def _build_result(self, turn: Any, *, schema: type[BaseModel] | None) -> RuntimeResult:
        final = turn.final_response or ""
        cost = self._runtime._cost_from_usage(turn.usage, model_id=self._model_id)
        if schema is None:
            return RuntimeResult(
                text=final,
                structured=None,
                cost=cost,
                finish="stop",
                raw=turn,
            )
        if not final:
            raise RuntimeStructuredOutputError(
                f"{self._runtime.label}: turn completed with empty final_response",
                body={"items_count": len(turn.items)},
            )
        try:
            structured = json.loads(final)
        except json.JSONDecodeError as exc:
            raise RuntimeStructuredOutputError(
                f"{self._runtime.label}: final_response was not valid JSON: {exc}",
                body=final[:500],
            ) from exc
        return RuntimeResult(
            text=final,
            structured=structured,
            cost=cost,
            finish="stop",
            raw=turn,
        )


__all__ = [
    "DEFAULT_CODEX_MODEL",
    "CodexAgentSession",
    "CodexRuntime",
]
