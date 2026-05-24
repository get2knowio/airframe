"""Vendor-agnostic exception hierarchy for agent runtimes.

Every adapter classifies its vendor's failures into this hierarchy
so consumer code can ``except`` on a vendor-neutral type. What the
consumer *does* with each error — retry, fall over to another
binding, surface to the user, escalate to a larger model — is a
policy decision airframe deliberately doesn't make.

The base class is :class:`AgentRuntimeError` (not ``RuntimeError``) so
it doesn't shadow Python's builtin :class:`RuntimeError` at every
``except`` site. The subclasses keep the shorter ``Runtime*Error``
prefix because they're already specific (no builtin collision).

The hierarchy carves the failure modes that have meaningfully
different shapes:

* :class:`RuntimeAuthError` — credentials bad, expired, or missing.
* :class:`RuntimeModelNotFoundError` — server says the model isn't
  available on this binding.
* :class:`RuntimeTransientError` — 5xx, rate-limit, brief network
  hiccup. The underlying call was *attempted* and the server / network
  returned a recoverable failure.
* :class:`RuntimeStructuredOutputError` — the model returned but
  didn't produce a payload matching the requested schema. A
  capability or instruction-following gap, not a transport problem.
* :class:`RuntimeContextOverflowError` — prompt exceeded the model's
  context window even after the adapter's compaction.
* :class:`RuntimeProtocolError` — the adapter saw something it can't
  interpret (empty body, malformed JSON-RPC, etc.). Indicates a bug
  or version drift, not a server-side failure.
* :class:`RuntimeServerStartError` — the adapter couldn't bring its
  backend up at all (subprocess didn't launch, HTTP server
  unreachable).
* :class:`RuntimeCancelledError` — the call was aborted
  cooperatively or by an explicit cancel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from airframe.rate_limit import RateLimitInfo


class AgentRuntimeError(Exception):
    """Base class for agent-runtime adapter errors.

    Attributes:
        status: Optional HTTP / RPC status code from the server.
        body: Optional decoded body (JSON or first 500 chars of text).
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        body: Any = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class RuntimeServerStartError(AgentRuntimeError):
    """Failed to launch the runtime backend (subprocess, server, etc.)."""


class RuntimeAuthError(AgentRuntimeError):
    """Provider authentication failed (bad / missing / expired credentials).

    The credential itself is the problem. Retrying the same call with
    the same credential will fail the same way.
    """


class RuntimeModelNotFoundError(AgentRuntimeError):
    """The requested model is not available on this binding.

    Distinct from :class:`RuntimeAuthError` because the credential is
    fine; the binding just doesn't serve that model.
    """


class RuntimeStructuredOutputError(AgentRuntimeError):
    """The model returned without producing the requested typed payload.

    The transport succeeded but the response didn't match the schema
    (or the model refused to call the structured-output tool).
    Indicates a capability or instruction-following gap on this
    model, not a server-side failure.

    Attributes:
        retries: Adapter-reported retry count (often 0 — many adapters
            don't expose this).
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        body: Any = None,
        retries: int = 0,
    ) -> None:
        super().__init__(message, status=status, body=body)
        self.retries = retries


class RuntimeContextOverflowError(AgentRuntimeError):
    """Prompt exceeded the model's context window even after compaction.

    A different model on the same provider — or a shorter prompt — is
    the only path forward. Retrying the same model with the same
    input will hit the same wall.
    """


class RuntimeTransientError(AgentRuntimeError):
    """Transient server/provider error: 5xx, rate limits, brief outages.

    The call was attempted; the server (or network) returned a
    recoverable failure. The underlying condition is expected to
    clear on its own.

    Attributes:
        rate_limit: Typed
            :class:`~airframe.rate_limit.RateLimitInfo` snapshot when
            the throttle response carried quota data. ``None`` for
            non-throttle transient errors (5xx, network blip) and for
            adapters that don't declare
            :data:`~airframe.features.Feature.RATE_LIMIT_TELEMETRY`.
            Consumers writing budget-aware retry policy can branch on
            ``rate_limit.windows[0].retry_after_seconds`` etc.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        body: Any = None,
        rate_limit: RateLimitInfo | None = None,
    ) -> None:
        super().__init__(message, status=status, body=body)
        self.rate_limit = rate_limit


class RuntimeCancelledError(AgentRuntimeError):
    """The call was aborted (cooperatively or by an explicit cancel)."""


class RuntimeProtocolError(AgentRuntimeError):
    """The runtime returned a response that didn't match the expected shape.

    Indicates an adapter / vendor-SDK bug or version drift, not a
    server-side failure. Worth surfacing as a defect rather than
    treating as transient.
    """


class RuntimeBudgetExceededError(AgentRuntimeError):
    """Raised when a session exceeds a caller-supplied budget cap.

    Phase 5 Iteration D companion to the ``max_turns=`` /
    ``max_budget_usd=`` kwargs on :meth:`~airframe.protocol.AgentSession.execute`
    and :meth:`~airframe.protocol.AgentSession.stream`. Raised at a
    turn boundary (mid-turn interrupt is additive later via the
    existing :meth:`~airframe.protocol.AgentSession.cancel` plumbing)
    when the running total would exceed the cap.

    Attributes:
        cap: The caller-supplied threshold (USD or turn count).
        current: The session's running total at the moment the cap
            tripped — equal to or greater than ``cap``.
        kind: ``"usd"`` when the USD-budget cap fired,
            ``"turns"`` when the turn-count cap fired.

    Consumer code can branch on ``kind`` to decide whether to
    retry-with-larger-cap, surface to the user, or fail closed.
    """

    def __init__(
        self,
        message: str,
        *,
        cap: float,
        current: float,
        kind: Literal["usd", "turns"],
        status: int | None = None,
        body: Any = None,
    ) -> None:
        super().__init__(message, status=status, body=body)
        self.cap = cap
        self.current = current
        self.kind = kind


class UnsupportedFeatureError(AgentRuntimeError):
    """Raised when a caller asks an adapter to honour a capability it lacks.

    Phase 1+ companion to :class:`~airframe.protocol.UnsupportedBindingError`.
    A binding mismatch is "this adapter doesn't serve this
    ``(provider, model)``"; an unsupported feature is "this adapter
    serves the binding, but the specific capability you asked for
    (streaming, cancellation, session-resume, MCP, …) is not wired."

    Adapters surface the capability truth via
    :meth:`~airframe.protocol.AgentRuntime.supports` — callers branching
    on ``supports(Feature.X)`` before invoking the feature's API
    never see this error. It's the safety net for callers that skip
    the predicate.

    Modelled on the implementation plan's cross-cutting principle:
    *"No silent fallbacks. A capability declined ⇒ a clear
    UnsupportedFeatureError, never 'best effort succeed.'"*

    Attributes:
        feature: The :class:`~airframe.features.Feature` enum value
            (or its string) the caller asked for. ``None`` when the
            asking site didn't pin a specific enum member.
    """

    def __init__(
        self,
        message: str,
        *,
        feature: str | None = None,
        status: int | None = None,
        body: Any = None,
    ) -> None:
        super().__init__(message, status=status, body=body)
        self.feature = feature


__all__ = [
    "AgentRuntimeError",
    "RuntimeAuthError",
    "RuntimeBudgetExceededError",
    "RuntimeCancelledError",
    "RuntimeContextOverflowError",
    "RuntimeModelNotFoundError",
    "RuntimeProtocolError",
    "RuntimeServerStartError",
    "RuntimeStructuredOutputError",
    "RuntimeTransientError",
    "UnsupportedFeatureError",
]
