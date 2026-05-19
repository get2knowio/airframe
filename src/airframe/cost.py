"""Cost telemetry record — vendor-agnostic.

Every :class:`AgentRuntime` adapter produces a :class:`CostRecord` on
each :meth:`execute` call. The record is suitable for emitting to
structured logs, cost-tracking JSONL stores, or whatever telemetry
sink the consumer uses.

Adapters with vendor-computed cost (Claude Agent SDK exposes
``total_cost_usd`` directly) populate ``cost_usd`` from the vendor's
report. Adapters without (raw OpenAI-family, Copilot, Kimi) compute
``cost_usd`` from token counts × a per-model pricing rate — and emit
``cost_usd=None`` when no rate is configured for the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CostRecord:
    """One row of cost telemetry — produced by each runtime execute().

    Field names map cleanly onto the typical "agent.cost" structured-log
    payload via :meth:`to_dict`.

    Attributes:
        provider_id: The vendor that served the call
            (``"anthropic"``, ``"openai"``, ``"github-copilot"``, etc.).
        model_id: The model identifier the vendor reports.
        cost_usd: USD cost for this call. ``None`` when the adapter
            cannot compute it (no pricing rate for the model and the
            vendor didn't report cost directly).
        input_tokens: Prompt tokens consumed.
        output_tokens: Completion tokens generated.
        cache_read_tokens: Prompt tokens served from the provider's cache.
        cache_write_tokens: Prompt tokens written to the provider's cache.
        reasoning_tokens: Hidden reasoning / extended-thinking tokens
            the model consumed *in addition to* ``output_tokens``. Each
            SDK reports these under a different name (Claude SDK's
            ``thinking_tokens``, OpenAI's
            ``completion_tokens_details.reasoning_tokens``);
            canonicalised here. ``0`` when
            the model didn't reason, the adapter doesn't expose the
            counter, or the SDK hasn't wired it yet.
        finish: Provider-reported stop reason (``"stop"``, ``"length"``,
            ``"tool_calls"``, ``"end_turn"``, ...). ``None`` if not
            reported.
    """

    provider_id: str | None
    model_id: str | None
    cost_usd: float | None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    finish: str | None
    reasoning_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Render as a structured-log payload."""
        return {
            "providerID": self.provider_id,
            "modelID": self.model_id,
            "cost_usd": self.cost_usd,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "finish": self.finish,
        }


__all__ = ["CostRecord"]
