"""``AgentRuntime`` protocol — vendor-agnostic agent transport.

Agents talk to LLMs through a thin :class:`AgentRuntime` protocol.
Implementations live under :mod:`airframe.adapters` and wrap each
vendor's preferred Python SDK (Claude Agent SDK, GitHub Copilot Python
SDK, OpenAI Codex SDK, OpenCode Zen HTTP, etc.). A consumer receives
an :class:`AgentRuntime` at construction and never sees a
vendor-specific type at the call site.

Design principles:

1. **Runtime owns its lifecycle.** Subprocesses, HTTP pools, auth
   tokens, session state, retry logic — all hidden behind the
   protocol. The consumer interface is :meth:`execute`,
   :meth:`reset`, :meth:`aclose`.
2. **No opaque handles in the consumer interface.** Avoid
   ``session_id``-juggling. The runtime hides any session state
   inside its own instance.
3. **Scope is explicit, sessions are implicit.** A runtime MAY hold
   context warmth across consecutive :meth:`execute` calls so the
   provider's prompt cache hits accrue within a scope (typically one
   "task" or "bead"). :meth:`reset` drops that scope.
4. **Errors are vendor-agnostic.** Adapters classify failures into
   the :mod:`airframe.errors` hierarchy (:class:`RuntimeAuthError`,
   :class:`RuntimeTransientError`,
   :class:`RuntimeStructuredOutputError`, etc.) so cascade /
   retry logic doesn't need to know which adapter raised what.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from airframe.cost import CostRecord
from airframe.tiers import ProviderModel


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
                :class:`UnsupportedBindingError`. A cascade should
                call :meth:`validate_binding` first to avoid this.
            timeout: Hard wall-clock budget for the call.

        Returns:
            :class:`RuntimeResult` with text + (optional) structured
            payload + cost + finish reason.

        Raises:
            RuntimeAuthError, RuntimeModelNotFoundError,
            RuntimeStructuredOutputError, RuntimeContextOverflowError,
            RuntimeTransientError, RuntimeProtocolError,
            RuntimeServerStartError: classified failures from
            :mod:`airframe.errors`. The cascade decides what to do
            based on the exception type.
        """
        ...

    async def reset(self) -> None:
        """Drop accumulated context for a fresh scope.

        Called at scope boundaries (typically between tasks / beads).
        Implementations release scope-bound state — HTTP sessions are
        deleted; subprocess sessions are disconnected; stateless HTTP
        adapters can no-op. Cheap to call; never raises.

        Runtime-wide resources (subprocess pool, HTTP client, auth
        tokens) are kept across :meth:`reset`. Use :meth:`aclose`
        for full teardown.
        """
        ...

    async def aclose(self) -> None:
        """Release runtime-wide resources.

        Idempotent. Implementations must not raise — teardown errors
        should be logged at debug level and swallowed.
        """
        ...

    def validate_binding(self, binding: ProviderModel) -> bool:
        """Return ``True`` if this runtime can satisfy the binding.

        Adapters typically match on ``binding.provider_id`` against a
        :attr:`SUPPORTED_PROVIDER_IDS` frozenset and may further filter
        on ``binding.model_id`` (e.g. ``CopilotRuntime`` rejects
        ``model_id`` starting with ``claude-`` per the Phase 0 spike
        finding that Claude-on-Copilot doesn't honour tool calls).

        Used by cascade machinery to short-circuit — bindings a
        runtime can't serve are skipped without attempting them.
        """
        ...


class UnsupportedBindingError(Exception):
    """Raised when a runtime is asked to serve a binding it can't serve.

    Distinct from :class:`airframe.errors.AgentRuntimeError` — this
    is a programming error (the caller should have checked
    :meth:`AgentRuntime.validate_binding` first), not a runtime failure.
    """


__all__ = ["AgentRuntime", "RuntimeResult", "UnsupportedBindingError"]
