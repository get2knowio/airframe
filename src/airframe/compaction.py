"""Compaction-control surface (scaffolding).

Long-running agent sessions eventually hit their context window and
need to compact — replace older turns with a summary so the agent
can keep working. The :mod:`airframe.hooks` surface already lets
consumers *observe* compaction (the ``pre_compact`` event); this
module is the *configuration* sibling that lets the consumer
control *when* and *how* it happens.

Vendor support today:

* **Claude Agent SDK** — ``PreCompact`` hook + ``session_store`` +
  ``fold_session_summary`` give native control over both trigger
  threshold and the summariser prompt.
* **GitHub Copilot SDK** — ``session.compaction_start`` /
  ``compaction_complete`` events signal compaction is happening,
  but there's no native trigger-threshold knob exposed.
* **OpenAI Responses API** — ``context_management`` request
  parameter takes a structured policy.
* **OpenAI Chat Completions** — no server-side compaction; the
  client owns the messages buffer entirely.

**Phase 6 scope — scaffolding only.** This module ships
:class:`CompactionConfig` + :data:`Feature.COMPACTION_CONTROL` +
the ``session(compaction=...)`` kwarg so consumer code can plan
against the namespace, but no adapter currently flips the feature
flag to ``True``. The real implementation needs per-vendor
translation (Claude's ``fold_session_summary`` API + summariser
prompt overrides, OpenAI Responses' ``context_management`` shape)
plus a session-level :meth:`compact` method on
:class:`~airframe.protocol.AgentSession` — that protocol-shape
change is deferred until a consumer with long-running sessions
needs it.

The kwarg shape locked here is forward-compatible with that future
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class CompactionConfig:
    """Portable compaction policy.

    Attributes:
        trigger: When to compact. ``"auto"`` lets the vendor's
            heuristic decide; ``"manual"`` disables auto-compaction
            and requires an explicit ``session.compact()`` call
            (deferred). ``None`` keeps the vendor default.
        threshold_ratio: Auto-trigger threshold as a fraction of the
            model's context window (0.0–1.0). ``0.8`` means "compact
            when 80% full." ``None`` keeps the vendor default. Only
            honoured when ``trigger="auto"``.
        summary_prompt: Caller-supplied summariser prompt that
            replaces the vendor's default. ``None`` uses the
            vendor's. Vendors that don't expose this raise
            :class:`~airframe.errors.UnsupportedFeatureError` when a
            non-None value is set — the prompt directly shapes what
            information survives compaction, so silent ignore would
            mask a real behavioural difference.
    """

    trigger: Literal["auto", "manual"] | None = "auto"
    threshold_ratio: float | None = None
    summary_prompt: str | None = None


__all__ = ["CompactionConfig"]
