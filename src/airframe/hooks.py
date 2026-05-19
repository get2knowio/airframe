""":class:`HookEvent` — typed observation of session lifecycle.

Phase 5 of the [implementation plan](../../dev-docs/implementation-plan.md)
introduces ``on_event=`` on :meth:`AgentRuntime.session`. The
adapter translates its vendor SDK's native event stream into
:class:`HookEvent` instances and fans them out to the user's
callable. Consumer code subscribes once per session and observes
without participating — same shape as a logging hook, not a
permission gate.

Per-adapter emission strategy (lands in Iteration C):

* **Claude Agent SDK** — :attr:`ClaudeAgentOptions.hooks` +
  ``include_hook_events=True`` surface the SDK's native hook
  events. The adapter forwards every kind.
* **GitHub Copilot SDK** — :meth:`CopilotSession.on` subscription
  filtering the typed ``*Data`` events into :class:`HookEvent`.
* **Moonshot Kimi SDK** — synthesised from the ``WireMessage``
  stream (``ToolCall`` → ``pre_tool_use``, ``ToolResult`` →
  ``post_tool_use`` / ``tool_failure``, ``CompactionBegin`` →
  ``pre_compact``, plus execute/close boundary events). Emits 7 of
  the 8 kinds — only ``rate_limit`` stays unemitted (Moonshot
  raises 429s as exceptions, not wire events).
* **OpenAI-compatible HTTP** — synthesised from the client-side
  tool-loop. Cannot emit ``pre_compact`` / ``rate_limit`` honestly;
  the emittable set is documented per adapter.

**Shape lock.** ⚠️ The eight :attr:`HookEvent.kind` literals are
shape-locked in v0.8. Consumer code branches on
``event.kind == "pre_tool_use"`` etc; once that ships, renaming
any kind value is a major-version break. Adding new kinds later is
additive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

HookEventKind = Literal[
    "session_start",
    "session_end",
    "user_prompt_submit",
    "pre_tool_use",
    "post_tool_use",
    "tool_failure",
    "pre_compact",
    "rate_limit",
]
"""The eight lifecycle moments airframe normalises across adapters.

* ``"session_start"`` — the underlying vendor session is connected
  / created.
* ``"session_end"`` — the session is being closed (whether by
  explicit ``close()``, error, or natural completion).
* ``"user_prompt_submit"`` — a new user turn has been submitted to
  the session.
* ``"pre_tool_use"`` — the model is about to invoke a tool (after
  any :class:`~airframe.permission.PermissionCallback` decision).
* ``"post_tool_use"`` — a tool returned successfully.
* ``"tool_failure"`` — a tool returned an error (handler raised,
  schema validation failed, vendor reported failure).
* ``"pre_compact"`` — the vendor is about to compact / summarise
  history. Only Claude emits this today; other adapters never fire
  it.
* ``"rate_limit"`` — the vendor signalled a rate-limit / throttle
  event. Only Claude emits this today.

The per-adapter *emittable kinds set* is documented in each
adapter's class docstring so consumers can branch defensively.
"""


@dataclass(frozen=True, slots=True)
class HookEvent:
    """One lifecycle observation surfaced by an :class:`AgentSession`.

    Attributes:
        kind: One of the eight :data:`HookEventKind` literals
            naming what just happened.
        session_id: Vendor-assigned session identifier when one
            exists. ``None`` on adapters with no server-side session
            (OpenAI-compatible HTTP) or before the session has
            connected.
        payload: Adapter-specific bag of fields the kind carries.
            Always a ``dict[str, Any]`` for forward compatibility;
            per-kind typed payloads are deferred (additive later).
            Common fields by kind:

            * ``"pre_tool_use"`` / ``"post_tool_use"`` /
              ``"tool_failure"`` — ``tool_name``, ``tool_call_id``,
              ``arguments`` (preview string), optionally ``output``
              / ``error`` / ``duration_ms``.
            * ``"user_prompt_submit"`` — ``prompt`` (truncated for
              long prompts), ``length``.
            * ``"session_start"`` / ``"session_end"`` — ``model``,
              optionally ``cost_usd`` / ``turn_count`` at end.
            * ``"rate_limit"`` — ``retry_after_seconds``, vendor
              error code.
            * ``"pre_compact"`` — ``messages_before``, optionally
              ``token_count``.

            Consumers writing portable observers should treat any
            specific field as optional and degrade gracefully.

    Example::

        def log_event(event: HookEvent) -> None:
            print(f"[{event.kind}] session={event.session_id} {event.payload}")

        runtime.session(on_event=log_event)
    """

    kind: HookEventKind
    session_id: str | None
    payload: dict[str, Any]


__all__ = ["HookEvent", "HookEventKind"]
