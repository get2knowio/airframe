""":class:`PermissionCallback` — gating tool execution at the airframe layer.

Phase 5 of the [implementation plan](../../dev-docs/implementation-plan.md)
introduces ``on_permission=`` on :meth:`AgentRuntime.session`. Each
:class:`PermissionRequest` represents the vendor SDK asking
"shall I let the model run this tool?"; the user-supplied
:class:`PermissionCallback` returns a typed
:data:`PermissionDecision` which the adapter translates to the
vendor's native permission channel:

* **Claude Agent SDK** — :attr:`ClaudeAgentOptions.can_use_tool`.
* **GitHub Copilot SDK** — :attr:`CopilotSession.on_permission_request`
  (mandatory at session creation).
* **OpenAI Codex SDK** — :attr:`Thread.approval_policy` (session-wide
  enum; the callback fires once at session creation to derive the
  enum value, since Codex has no per-call permission channel).
* **OpenAI-compatible HTTP** — declines. Chat Completions has no
  permission wire shape; ``on_permission=`` raises
  :class:`~airframe.errors.UnsupportedFeatureError`. A future
  ``OpenAIResponsesRuntime`` could wire it via the Responses API.

**Shape lock.** ⚠️ The :class:`PermissionRequest` field set, the
:data:`PermissionDecision` literal values, and the
:class:`PermissionCallback` Protocol signature are shape-locked in
v0.8. Consumer code defines callbacks against this contract; adding
new decision values later is additive; renaming existing ones is a
major-version break.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

PermissionDecision = Literal["allow", "deny", "defer"]
"""The three responses a :class:`PermissionCallback` may return.

* ``"allow"`` — let the model invoke the tool. Maps to the vendor's
  "approve" value (Claude ``allow``, Copilot ``approve_once``,
  Codex auto-approve via ``never`` policy).
* ``"deny"`` — block the call; the model sees a tool-execution
  failure and can recover. Maps to vendor "reject" (Claude ``deny``,
  Copilot ``reject``, Codex ``untrusted`` policy).
* ``"defer"`` — fall through to the vendor's default policy. Useful
  when the airframe-level callback only knows how to decide for a
  subset of tools; everything else routes to the SDK's built-in
  prompting / policy. Adapters debug-log the deferral so consumers
  can audit it.
"""


@dataclass(frozen=True, slots=True)
class PermissionRequest:
    """One tool-use permission decision the model is asking for.

    Built by the adapter from the vendor SDK's permission-request
    payload, handed to :meth:`PermissionCallback.handle`. The shape
    is intentionally narrow — three fields cover every vendor's
    permission UI today; richer per-vendor metadata (e.g. Claude's
    tool-input preview, Copilot's invocation context) is reachable
    via :meth:`AgentSession.unwrap` if a consumer needs it.

    Attributes:
        tool_name: The user-facing tool name the model wants to call
            (the ``mcp__<server>__`` prefix is stripped — same
            shape as :attr:`~airframe.events.ToolCallStart.tool_name`).
        tool_args: The model's proposed arguments. Treat as untrusted
            input — the model decided these; the callback is the
            gate before the handler runs.
        reason: Optional human-readable reason the SDK surfaced for
            why this is a permission boundary (e.g. ``"writes to
            filesystem"``). ``None`` when the SDK doesn't expose one.

    Example::

        async def auto_approve_reads(req: PermissionRequest) -> PermissionDecision:
            if req.tool_name in ("read_file", "list_dir"):
                return "allow"
            return "defer"  # let the SDK's default prompt take over
    """

    tool_name: str
    tool_args: dict[str, Any]
    reason: str | None = None


@runtime_checkable
class PermissionCallback(Protocol):
    """The contract user code implements to gate tool execution.

    A single ``async def handle(request) -> PermissionDecision``
    method, kept narrow so adapters can wrap arbitrary user code
    (lambdas, methods, dataclasses with ``__call__``, async classes)
    without ceremony. :class:`Protocol` + :func:`runtime_checkable`
    lets ``isinstance(obj, PermissionCallback)`` work for both
    structural matches and explicit registrations.

    Example::

        class AllowReadsCallback:
            async def handle(self, request: PermissionRequest) -> PermissionDecision:
                return "allow" if request.tool_name.startswith("read_") else "deny"

        runtime.session(on_permission=AllowReadsCallback())
    """

    async def handle(self, request: PermissionRequest) -> PermissionDecision: ...


__all__ = ["PermissionCallback", "PermissionDecision", "PermissionRequest"]
