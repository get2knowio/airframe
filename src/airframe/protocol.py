"""``AgentRuntime`` protocol — vendor-agnostic agent transport.

Think of airframe as JDBC for LLM agent SDKs: one interface, many
drivers. Agents talk to LLMs through a thin :class:`AgentRuntime`
protocol; implementations live under :mod:`airframe.adapters` and
wrap each vendor's preferred Python SDK (Claude Agent SDK, GitHub
Copilot Python SDK, OpenAI Codex SDK, OpenCode Zen HTTP, etc.). A
consumer receives an :class:`AgentRuntime` at construction and never
sees a vendor-specific type at the call site.

Design principles:

1. **Runtime owns its lifecycle.** Subprocesses, HTTP pools, auth
   tokens, session state — all hidden behind the protocol. The
   consumer interface is :meth:`execute`, :meth:`reset`,
   :meth:`close`.
2. **No opaque handles in the consumer interface.** Avoid
   ``session_id``-juggling. The runtime hides any session state
   inside its own instance.
3. **Scope is explicit, sessions are implicit.** A runtime MAY hold
   context warmth across consecutive :meth:`execute` calls so the
   provider's prompt cache hits accrue within a scope (typically one
   "task"). :meth:`reset` drops that scope.
4. **Errors are vendor-agnostic.** Adapters classify failures into
   the :mod:`airframe.errors` hierarchy so consumer code can
   ``except`` on a neutral type. What to *do* with each error —
   retry, surface, escalate — is consumer policy; airframe doesn't
   prescribe it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from airframe.cost import CostRecord

if TYPE_CHECKING:
    from airframe.events import RuntimeEvent
    from airframe.features import Feature
    from airframe.inputs import Prompt
    from airframe.models import ModelInfo
    from airframe.thinking import ThinkingMode

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ProviderModel:
    """One ``(provider_id, model_id)`` binding.

    The two-string pair is airframe's primitive for identifying which
    vendor + model a call should target. Adapters decide whether they
    can serve a binding by matching on ``provider_id`` (and sometimes
    filtering on ``model_id``) via :meth:`AgentRuntime.validate_binding`.

    Attributes:
        provider_id: Vendor identifier — ``"anthropic"``, ``"openai"``,
            ``"copilot"``, ``"opencode"``, etc.
        model_id: The model identifier the vendor recognises
            (e.g. ``"claude-haiku-4-5"``, ``"gpt-5-mini"``).
    """

    provider_id: str
    model_id: str

    @property
    def label(self) -> str:
        return f"{self.provider_id}/{self.model_id}"

    def to_dict(self) -> dict[str, str]:
        return {"providerID": self.provider_id, "modelID": self.model_id}


@dataclass(frozen=True, slots=True)
class RuntimeResult:
    """Canonical execute() result.

    Attributes:
        text: Concatenated assistant text. For structured-output calls
            this is typically empty or a short acknowledgement — the
            meaningful payload lives in :attr:`structured`.
        structured: Schema-shaped object — already envelope-unwrapped
            for HTTP adapters, already pulled from the forced-tool-call
            args for SDK-based adapters. ``None`` when ``schema`` was
            ``None`` on the execute() call.
        cost: Cost telemetry for this call. Adapters with a vendor-
            computed cost populate ``cost_usd`` directly; others
            compute from token counts × pricing table.
        finish: Provider-reported stop reason
            (``"stop"`` / ``"length"`` / ``"tool_calls"`` /
            ``"end_turn"`` / ``None``).
        raw: The transport-specific result object — kept for
            diagnostics. Not part of the protocol contract; consumers
            should treat it as opaque.
    """

    text: str
    structured: Any
    cost: CostRecord
    finish: str | None
    raw: Any = field(default=None, repr=False)


@runtime_checkable
class AgentRuntime(Protocol):
    """Vendor-agnostic agent runtime.

    Built-in implementations:

    * :class:`airframe.adapters.claude_code.ClaudeCodeRuntime` —
      Claude family via ``claude-agent-sdk``; subscription auth.
    * :class:`airframe.adapters.copilot.CopilotRuntime` —
      Copilot via ``github-copilot-sdk``; GitHub Copilot subscription.
    * :class:`airframe.adapters.codex.CodexRuntime` — OpenAI
      codex via ``openai-codex-sdk``; ChatGPT Plus / OpenAI API.
    * :class:`airframe.adapters.opencode_zen.OpenCodeZenRuntime`
      — opencode-go Zen gateway via HTTP; opencode subscription.

    All implementations satisfy the same protocol; consumers are
    runtime-agnostic by construction.
    """

    label: str
    """Human-readable runtime tag used in structured-log rows
    (``runtime=claude_code``, ``runtime=copilot``, etc.)."""

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
        """Send one prompt, return a canonical typed result.

        Phase 1 Iteration G refactored this into documented sugar for
        ``runtime.session(system=..., model=...).execute(prompt,
        schema=..., thinking=..., timeout=...)`` + ``close()``.
        Single-turn, ephemeral. Consumers wanting context warmth across
        calls open a session explicitly.

        Args:
            prompt: The user message. Either a plain ``str`` (the
                v0-through-Phase-1 shape) or a list of
                :class:`~airframe.inputs.PromptPart` for interleaved
                text + images + files (Phase 2). Adapters that don't
                declare :data:`~airframe.features.Feature.VISION_INPUT`
                / :data:`~airframe.features.Feature.FILE_INPUT` raise
                :class:`~airframe.errors.UnsupportedFeatureError` on
                list-shaped prompts.
            schema: When non-None, the runtime coerces the model's
                response into a schema-conforming dict on
                :attr:`RuntimeResult.structured`. Implementations use
                the vendor's native structured-output mechanism (a
                forced tool call for Claude / Codex / Copilot, native
                JSON-schema mode for the Codex CLI,
                ``response_format=json_schema`` for OpenCode Zen).
                ``None`` means plain text — text answer on
                :attr:`RuntimeResult.text`, ``structured=None``.
            system: Optional system-prompt override. Baked into the
                bespoke session's vendor config at session-construction
                time.
            persona: Optional runtime-specific agent persona label.
                Some adapters honour it (e.g. selecting a bundled
                agent profile); others ignore it.
            model: When non-None, pin this binding for this call.
                Implementations that can't serve the binding raise
                :class:`UnsupportedBindingError`. Callers can check
                :meth:`validate_binding` first to avoid this.
            thinking: Reasoning-effort control (Phase 2). See
                :data:`~airframe.thinking.ThinkingMode`. Adapters that
                don't declare
                :data:`~airframe.features.Feature.REASONING_EFFORT`
                ignore literal-effort values; adapters that don't
                declare
                :data:`~airframe.features.Feature.REASONING_BUDGET_TOKENS`
                raise on dict-shaped values. ``None`` (default) sends
                no reasoning configuration; the model decides.
            timeout: Hard wall-clock budget for the call.

        Returns:
            :class:`RuntimeResult` with text + (optional) structured
            payload + cost + finish reason.

        Raises:
            RuntimeAuthError, RuntimeModelNotFoundError,
            RuntimeStructuredOutputError, RuntimeContextOverflowError,
            RuntimeTransientError, RuntimeProtocolError,
            RuntimeServerStartError, UnsupportedFeatureError:
            classified failures from :mod:`airframe.errors`. The
            caller decides what to do with each.
        """
        ...

    async def reset(self) -> None:
        """Drop accumulated context for a fresh scope.

        Called at scope boundaries (typically between tasks / beads).
        Implementations release scope-bound state — HTTP sessions are
        deleted; subprocess sessions are disconnected; stateless HTTP
        adapters can no-op. Cheap to call; never raises.

        Runtime-wide resources (subprocess pool, HTTP client, auth
        tokens) are kept across :meth:`reset`. Use :meth:`close`
        for full teardown.
        """
        ...

    async def close(self) -> None:
        """Release runtime-wide resources.

        Idempotent. Implementations must not raise — teardown errors
        should be logged at debug level and swallowed.
        """
        ...

    def validate_binding(self, binding: ProviderModel) -> bool:
        """Return ``True`` if this runtime can satisfy the binding.

        Adapters match on ``binding.provider_id`` against their
        :attr:`PROVIDER_ID` class attribute and may further filter on
        ``binding.model_id`` (e.g. ``CopilotRuntime`` rejects
        ``model_id`` starting with ``claude-`` because Claude served
        through Copilot Chat Completions doesn't honour tool calls
        and so can't satisfy the structured-output contract).

        Cheap and non-async — suitable for predicate checks before
        attempting :meth:`execute`. Calling :meth:`execute` with an
        unsupported binding raises :class:`UnsupportedBindingError`.
        """
        ...

    async def list_models(self) -> list[ModelInfo]:
        """Return the live list of models the consumer can pick from.

        Hits the vendor's models endpoint with the user's resolved
        credentials, enriches each entry with whatever metadata
        the adapter knows (context window, pricing, capability flags),
        and returns a list suitable for driving a UI menu.

        The call requires auth and network. A failure (missing
        credentials, vendor down) raises one of:

        Raises:
            RuntimeAuthError: when no usable credentials were found.
            RuntimeTransientError: vendor 5xx / rate-limit / timeout.
            RuntimeProtocolError: vendor returned an unexpected shape.

        Consumers should surface these to the user *before* letting
        them commit to a model selection.
        """
        ...

    def unwrap(self, cls: type[T]) -> T:
        """Return the underlying vendor object cast to ``cls``.

        The documented escape hatch around airframe's portable surface,
        modelled on JDBC 4.0's :class:`java.sql.Wrapper` interface.
        When the portable protocol doesn't expose a vendor-specific
        capability, the consumer reaches the native client (or session,
        or thread) directly through ``unwrap(NativeType)``.

        Per-adapter mappings (Phase 0):

        * :class:`ClaudeCodeRuntime` accepts
          ``unwrap(ClaudeSDKClient)`` — returns the live SDK client
          when one exists (after the first ``execute()``); raises
          :class:`TypeError` if requested before a client is built.
        * :class:`CopilotRuntime` accepts ``unwrap(CopilotClient)`` and
          ``unwrap(CopilotSession)``.
        * :class:`CodexRuntime` accepts ``unwrap(Codex)`` and
          ``unwrap(Thread)``.
        * :class:`OpenAICompatibleRuntime` accepts ``unwrap(AsyncOpenAI)``.

        Every adapter additionally accepts ``unwrap(type(self))`` and
        returns ``self`` — the trivial case that keeps the contract
        consistent across runtimes.

        Args:
            cls: The native type the caller wants to reach. Use a real
                type (e.g. ``from claude_agent_sdk import ClaudeSDKClient;
                runtime.unwrap(ClaudeSDKClient)``), not a string name.

        Returns:
            The underlying vendor object.

        Raises:
            TypeError: when this runtime can't satisfy ``cls`` (the
                requested type isn't one of the runtime's native
                objects, or the lazily-constructed object hasn't
                been built yet).
        """
        ...

    def supports(self, feature: Feature, model: ProviderModel | None = None) -> bool:
        """Return ``True`` if this runtime exposes ``feature``.

        Capability negotiation predicate, modelled on JDBC's
        :class:`DatabaseMetaData` ``supportsXxx()`` family and
        SQLAlchemy's ``Dialect.supports_*`` flags. The contract:

        1. **Cheap and pure.** No network, no SDK version sniffing,
           no subprocess probe. A static lookup table on the adapter.
        2. **Agrees with execute().** If this returns ``True``,
           calling the API associated with ``feature`` must not raise
           :class:`UnsupportedBindingError` purely on capability
           grounds. The TCK in :mod:`airframe.testing.contracts`
           verifies this for every adapter.
        3. **False is the safe default.** Adapters declare what they
           *do* support; everything else is False. Consumers branching
           on ``supports()`` get correct behaviour even when running
           against a future runtime that adds new
           :class:`Feature` enum members.

        Args:
            feature: The :class:`Feature` enum member to query.
            model: Optional :class:`ProviderModel` for per-model
                differentiation. ``None`` asks the runtime-wide
                capability (true for the default model). Most features
                are runtime-wide today and adapters ignore this
                argument; per-model gating arrives later as needed.

        Returns:
            ``True`` when calling the API associated with ``feature``
            on this runtime succeeds; ``False`` otherwise.
        """
        ...

    def session(
        self,
        *,
        resume: str | None = None,
        system: str | None = None,
        model: ProviderModel | None = None,
        provider_options: Any | None = None,
    ) -> AgentSession:
        """Open a multi-turn session against this runtime.

        Phase 1 of the [implementation
        plan](../../docs/implementation-plan.md) introduces sessions
        as the "hinge" abstraction — every later kwarg
        (``thinking=``, ``tools=``, ``mcp_servers=``,
        ``on_permission=``) attaches to the session, not the runtime.

        ADR-004 picks single-active-session per runtime: opening a
        second session before the first is ``close()``'d invalidates
        the first. Adapters whose vendor session is process-bound
        (Claude Code, Copilot) cannot multiplex; OpenAI-compatible
        HTTP could but doesn't, for consistency.

        Args:
            resume: Vendor-assigned session ID to resume a prior
                conversation. ``None`` opens a fresh session.
                Adapters declaring
                :data:`~airframe.features.Feature.SESSION_RESUME`
                honour the kwarg; adapters that don't raise
                :class:`NotImplementedError` on any non-None value.
            system: System-prompt override for the session's lifetime.
                Equivalent to passing ``system=`` on every
                :meth:`AgentSession.execute` call; baked in at
                session-construction time so adapters that materialise
                vendor state lazily (Claude Code's
                :class:`ClaudeSDKClient`) can pre-allocate.
            model: Default :class:`ProviderModel` for every turn in
                this session. Per-turn overrides are not exposed in
                Phase 1 — switch sessions to switch models.
            provider_options: Vendor-specific extension namespace
                (see :mod:`airframe.options`). Accepted in Phase 1
                but unused — each :class:`ProviderOptions` dataclass
                is empty scaffolding; later phases (2+) fill them.

        Returns:
            A fresh :class:`AgentSession`.
        """
        ...


@runtime_checkable
class AgentSession(Protocol):
    """A multi-turn conversation handle scoped to one runtime.

    Phase 1 of the implementation plan introduces sessions as the
    "hinge" abstraction: every later kwarg
    (``thinking=``, ``tools=``, ``mcp_servers=``, ``on_permission=``)
    attaches to :class:`AgentSession`, not :class:`AgentRuntime`. A
    runtime's :meth:`AgentRuntime.session` factory builds one.

    The shape mirrors the per-vendor session abstractions airframe
    wraps — Claude's :class:`ClaudeSDKClient` lifecycle, Copilot's
    :class:`CopilotSession`, Codex's :class:`Thread`, and the
    client-side ``messages=[]`` buffer used for OpenAI-compatible
    HTTP — collapsed onto one neutral interface.

    **Concurrency model (ADR-004).** A runtime owns at most one
    *active* session at a time. ``runtime.session()`` returning a
    second handle before the first is ``close()``'d is permitted but
    the second handle invalidates the first — adapters whose vendor
    session is process-bound (Claude Code, Copilot) cannot multiplex.
    Going concurrent later is additive (a new ``session()`` returning
    a fresh handle); going from concurrent to single is breaking, so
    Phase 1 picks single.

    Attributes:
        id: Vendor-assigned session identifier when one exists, or
            ``None`` for adapters with no server-side session
            (OpenAI-compatible HTTP). Treat as a hint, not a key —
            consumer code branching on ``session.id is None`` will
            need a fallback path for the HTTP-only adapters.
    """

    id: str | None
    """Vendor-assigned session ID, or ``None`` for stateless adapters."""

    async def execute(
        self,
        prompt: Prompt,
        *,
        schema: type[BaseModel] | None = None,
        thinking: ThinkingMode = None,
        timeout: float = 600.0,
    ) -> RuntimeResult:
        """Run one turn, return the canonical :class:`RuntimeResult`.

        Semantically identical to :meth:`AgentRuntime.execute` —
        same kwargs, same return type, same error classification —
        except the session retains context for the next turn.
        Repeated calls accrue prompt-cache warmth where the vendor
        supports it.

        Args:
            prompt: User message for this turn. Either a plain ``str``
                or a list of :class:`~airframe.inputs.PromptPart` for
                interleaved text + images + files. Adapters that don't
                declare :data:`~airframe.features.Feature.VISION_INPUT`
                / :data:`~airframe.features.Feature.FILE_INPUT` raise
                :class:`~airframe.errors.UnsupportedFeatureError` on
                list-shaped prompts.
            schema: When non-None, coerce the response into the
                schema. ``None`` means plain text — text on
                :attr:`RuntimeResult.text`, ``structured=None``.
                Same contract as :meth:`AgentRuntime.execute`.
            thinking: Reasoning-effort control. See
                :data:`~airframe.thinking.ThinkingMode`. Phase 2
                addition; adapters declaring
                :data:`~airframe.features.Feature.REASONING_EFFORT`
                forward to the vendor's native field.
            timeout: Hard wall-clock budget for the turn.
        """
        ...

    def stream(
        self,
        prompt: Prompt,
        *,
        schema: type[BaseModel] | None = None,
        thinking: ThinkingMode = None,
        timeout: float = 600.0,
    ) -> AsyncIterator[RuntimeEvent]:
        """Run one turn, yielding :class:`RuntimeEvent` deltas live.

        The stream always ends with exactly one
        :class:`~airframe.events.TurnComplete` carrying the same
        :class:`RuntimeResult` :meth:`execute` would have returned.
        Cancelled streams (via :meth:`cancel` or task cancellation)
        may end without a :class:`TurnComplete`.

        Adapters declaring :data:`~airframe.features.Feature.STREAMING`
        emit fine-grained deltas as the vendor produces them; adapters
        that don't may emit a single
        :class:`~airframe.events.TextDelta` carrying the full response
        immediately before :class:`TurnComplete`.

        Args:
            prompt: User message for this turn. Same shape as
                :meth:`execute` — bare ``str`` or list of
                :class:`~airframe.inputs.PromptPart`.
            schema: Same as :meth:`execute`. Structured output still
                lands on the trailing :class:`TurnComplete.result`.
            thinking: Same as :meth:`execute`. Adapters surface
                reasoning deltas via
                :class:`~airframe.events.ReasoningDelta` when the
                model emits them.
            timeout: Hard wall-clock budget for the turn.

        Yields:
            :class:`~airframe.events.RuntimeEvent` instances; see
            :mod:`airframe.events` for the variant set.
        """
        ...

    async def cancel(self) -> None:
        """Abort the in-flight turn, if any.

        Cooperative cancellation: the adapter signals its vendor (an
        ``AbortController.abort()`` on Codex, ``client.interrupt()``
        on Claude, ``session.abort()`` on Copilot, ``Task.cancel()``
        on the OpenAI-compatible HTTP request). The cancelled
        :meth:`execute` / :meth:`stream` raises
        :class:`~airframe.errors.RuntimeCancelledError`; a stream may
        end without :class:`~airframe.events.TurnComplete`.

        Cheap and idempotent. The exact behaviour depends on adapter
        capability:

        * Adapters declaring
          :data:`~airframe.features.Feature.CANCEL` abort the in-flight
          turn (no-op when no turn is running).
        * Adapters that don't declare ``CANCEL`` raise
          :class:`~airframe.errors.UnsupportedFeatureError` when a turn
          is in flight; a no-op when nothing is running. Callers
          checking ``runtime.supports(Feature.CANCEL)`` before invoking
          ``cancel()`` never see this error.
        """
        ...

    async def close(self) -> None:
        """Release the session's vendor-side resources.

        Disconnects the vendor session (Claude subprocess link,
        Copilot session handle, Codex thread, client-side message
        buffer) but leaves the parent :class:`AgentRuntime`'s
        runtime-wide resources (subprocess pool, HTTP client, auth
        tokens) intact. Idempotent and must not raise — same
        discipline as :meth:`AgentRuntime.close`.
        """
        ...

    def unwrap(self, cls: type[T]) -> T:
        """Return the underlying vendor session object cast to ``cls``.

        JDBC-:class:`Wrapper`-style escape hatch for session-level
        vendor types — the per-conversation handle each adapter wraps.
        Phase 1 Iteration G moved per-conversation state out of the
        runtime and onto the session, so the vendor objects that used
        to be reachable via :meth:`AgentRuntime.unwrap` now live here.

        Per-adapter mappings:

        * :class:`ClaudeCodeSession` accepts
          ``unwrap(ClaudeSDKClient)`` — returns the live SDK client
          once the session has connected (after the first
          :meth:`execute` or :meth:`stream`); raises
          :class:`TypeError` if requested before then.
        * :class:`CopilotAgentSession` accepts
          ``unwrap(CopilotSession)`` — returns the underlying vendor
          session once :meth:`_ensure_session` has run.
        * :class:`CodexAgentSession` accepts ``unwrap(Thread)`` —
          returns the underlying :class:`Thread` once it has been
          constructed.
        * :class:`OpenAICompatibleSession` accepts no native types
          today — the OpenAI HTTP client lives on the runtime
          (reach it via ``runtime.unwrap(AsyncOpenAI)``); the
          session itself holds only the ``messages=[]`` buffer.

        Every adapter additionally accepts ``unwrap(type(self))`` and
        returns ``self`` — same convention as
        :meth:`AgentRuntime.unwrap`.

        Args:
            cls: The native type the caller wants to reach.

        Returns:
            The underlying vendor object.

        Raises:
            TypeError: when this session can't satisfy ``cls`` (type
                isn't one of the session's native objects, or the
                lazily-constructed object hasn't been built yet).
        """
        ...


class UnsupportedBindingError(Exception):
    """Raised when a runtime is asked to serve a binding it can't serve.

    Distinct from :class:`airframe.errors.AgentRuntimeError` — this
    is a programming error (the caller passed a binding this adapter
    doesn't support; :meth:`AgentRuntime.validate_binding` would
    have returned ``False``), not a runtime failure.
    """


__all__ = [
    "AgentRuntime",
    "AgentSession",
    "ProviderModel",
    "RuntimeResult",
    "UnsupportedBindingError",
]
