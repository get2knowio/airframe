"""Unit tests for :class:`CostRecord`.

Targets the canonical cost-telemetry shape — not adapter-specific
population paths (those have their own tests).
"""

from __future__ import annotations

from airframe.cost import CostRecord


def _record(**overrides: object) -> CostRecord:
    base: dict[str, object] = {
        "provider_id": "claude",
        "model_id": "claude-haiku-4-5",
        "cost_usd": 0.001,
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "finish": "stop",
    }
    base.update(overrides)
    return CostRecord(**base)  # type: ignore[arg-type]


def test_reasoning_tokens_defaults_to_zero() -> None:
    """Phase 0 additive field: defaults to 0 when adapter doesn't populate.

    This is the contract every existing adapter relies on — they
    construct CostRecord without the new kwarg and get 0 for free.
    """
    record = _record()
    assert record.reasoning_tokens == 0


def test_reasoning_tokens_round_trips_through_to_dict() -> None:
    """Structured-log payload includes the new field.

    The structured-log shape is a public surface for cost telemetry;
    sinks consuming the dict should see ``reasoning_tokens`` from day
    one rather than discovering it later.
    """
    record = _record(reasoning_tokens=42)
    payload = record.to_dict()
    assert payload["reasoning_tokens"] == 42


def test_reasoning_tokens_in_to_dict_when_unpopulated() -> None:
    """The key is always present — sinks can rely on it.

    Avoids the pattern where a downstream aggregator has to defend
    against ``KeyError`` on adapters that didn't set it.
    """
    record = _record()
    payload = record.to_dict()
    assert "reasoning_tokens" in payload
    assert payload["reasoning_tokens"] == 0
