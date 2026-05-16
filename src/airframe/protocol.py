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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from airframe.cost import CostRecord

if TYPE_CHECKING:
    from airframe.features import Feature
    from airframe.models import ModelInfo

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
        prompt: str,
        *,
        schema: type[BaseModel] | None = None,
        system: str | None = None,
        persona: str | None = None,
        model: ProviderModel | None = None,
        timeout: float = 600.0,
    ) -> RuntimeResult:
        """Send one prompt, return a canonical typed result.

        Args:
            prompt: The user message.
            schema: When non-None, the runtime coerces the model's
                response into a schema-conforming dict on
                :attr:`RuntimeResult.structured`. Implementations use
                the vendor's native structured-output mechanism (a
                forced tool call for Claude / Codex / Copilot, native
                JSON-schema mode for the Codex CLI,
                ``response_format=json_schema`` for OpenCode Zen).
                ``None`` means plain text — text answer on
                :attr:`RuntimeResult.text`, ``structured=None``.
            system: Optional system-prompt override. Adapters that
                bake the system prompt at scope-construction time
                (Claude Code SDK, etc.) may rebuild internal state
                when this changes between calls.
            persona: Optional runtime-specific agent persona label.
                Some adapters honour it (e.g. selecting a bundled
                agent profile); others ignore it.
            model: When non-None, pin this binding for this call.
                Implementations that can't serve the binding raise
                :class:`UnsupportedBindingError`. Callers can check
                :meth:`validate_binding` first to avoid this.
            timeout: Hard wall-clock budget for the call.

        Returns:
            :class:`RuntimeResult` with text + (optional) structured
            payload + cost + finish reason.

        Raises:
            RuntimeAuthError, RuntimeModelNotFoundError,
            RuntimeStructuredOutputError, RuntimeContextOverflowError,
            RuntimeTransientError, RuntimeProtocolError,
            RuntimeServerStartError: classified failures from
            :mod:`airframe.errors`. The caller decides what to do
            with each.
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


class UnsupportedBindingError(Exception):
    """Raised when a runtime is asked to serve a binding it can't serve.

    Distinct from :class:`airframe.errors.AgentRuntimeError` — this
    is a programming error (the caller passed a binding this adapter
    doesn't support; :meth:`AgentRuntime.validate_binding` would
    have returned ``False``), not a runtime failure.
    """


__all__ = [
    "AgentRuntime",
    "ProviderModel",
    "RuntimeResult",
    "UnsupportedBindingError",
]
