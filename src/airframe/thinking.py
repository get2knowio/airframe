""":class:`ThinkingMode` — portable reasoning-effort control.

Phase 2 of the [implementation plan](../../dev-docs/implementation-plan.md)
introduces the ``thinking=`` kwarg on :meth:`AgentSession.execute` /
:meth:`AgentSession.stream`. This module defines the value type.

Vendors expose extended-thinking / reasoning-effort differently:

* **OpenAI / OpenAI-compat** — ``reasoning_effort`` enum
  (``"low" | "medium" | "high"``), and on some models ``"minimal"``.
* **Anthropic / Claude Code** — ``thinking={"type": "enabled",
  "budget_tokens": N}`` on the Messages API; the Claude Agent SDK
  exposes the same shape on its ``ClaudeAgentOptions``.
* **GitHub Copilot** — ``reasoning_effort`` enum mirroring OpenAI's,
  passed on session creation.
* **Moonshot Kimi** — boolean ``thinking`` kwarg on
  :meth:`Session.create`; the model decides depth itself once
  enabled, so every airframe effort literal collapses to ``True``.

Airframe collapses these onto one union type that covers both
shapes — a literal effort level for the portable case, and the
inline dict for vendors that take a token budget. Adapters that
don't honour a given form raise
:class:`~airframe.errors.UnsupportedFeatureError` so callers
checking :meth:`AgentRuntime.supports` first never hit a surprise.

**Shape lock (ADR-006).** ⚠️ Per the implementation plan, the union
shape is the irreversible decision for Phase 2 — the choice to use
an inline dict for ``budget_tokens`` rather than a dedicated
dataclass matches Anthropic's wire format and is what consumers
already write. A dataclass would be slightly more type-safe but
more verbose; portability wins.
"""

from __future__ import annotations

from typing import Any, Literal

#: Portable literal effort level. Every vendor that exposes
#: reasoning-effort accepts ``"low" | "medium" | "high"`` (or a
#: superset that includes them). ``"minimal"`` is OpenAI-family only;
#: adapters that don't honour it coerce to ``"low"`` with a debug-level
#: log per the implementation plan.
ReasoningEffort = Literal["minimal", "low", "medium", "high"]

#: Value type for ``thinking=`` kwarg. Variants:
#:
#: * ``None`` — adapter / vendor default. No reasoning configuration
#:   sent; the model decides whether to reason on its own.
#: * :data:`ReasoningEffort` literal — portable effort level. Adapters
#:   declaring :data:`~airframe.features.Feature.REASONING_EFFORT`
#:   forward to their vendor's enum.
#: * ``dict`` (Claude-style) — currently the documented shape is
#:   ``{"budget_tokens": int}``. Adapters declaring
#:   :data:`~airframe.features.Feature.REASONING_BUDGET_TOKENS` forward
#:   to ``ClaudeAgentOptions.thinking``; others raise
#:   :class:`~airframe.errors.UnsupportedFeatureError`.
#: * ``"disabled"`` — explicitly turn reasoning off even when the
#:   model supports it. Useful when the consumer wants a fast,
#:   non-reasoning response from a model that would otherwise reason
#:   by default.
#:
#: The union is open-by-convention for future Phase 2+ extensions
#: (e.g., per-model thinking budgets surfaced as a richer dataclass
#: would land alongside a new variant rather than mutating an
#: existing one).
ThinkingMode = ReasoningEffort | Literal["disabled"] | dict[str, Any] | None


__all__ = ["ReasoningEffort", "ThinkingMode"]
