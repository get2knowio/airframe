"""Unit tests for the reasoning-output surface on :class:`RuntimeResult`.

The cross-vendor reasoning trace (``RuntimeResult.reasoning``) and
:data:`Feature.REASONING_OUTPUT` flag land in Phase 6. Adapters that
declare support populate the field when the model emitted a reasoning
trace; non-supporters (and supporters whose turn produced no
reasoning) leave it ``None``.

Tests here cover:

* The dataclass surface (default ``None``, backward-compat constructor).
* The OpenAI-compat helpers that defensively read
  ``message.reasoning_content`` / ``message.reasoning`` on both the
  non-streaming response message and per-chunk streaming deltas
  (DeepSeek-R1 derivatives use these field names).
"""

from __future__ import annotations

from types import SimpleNamespace

from airframe.adapters.openai_compatible import (
    _extract_delta_reasoning,
    _extract_message_reasoning,
)
from airframe.cost import CostRecord
from airframe.protocol import RuntimeResult


def _cost() -> CostRecord:
    return CostRecord(
        provider_id="dummy",
        model_id="dummy",
        cost_usd=None,
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        finish=None,
    )


def test_runtime_result_reasoning_defaults_to_none() -> None:
    rr = RuntimeResult(text="hi", structured=None, cost=_cost(), finish="stop")
    assert rr.reasoning is None


def test_runtime_result_reasoning_round_trips() -> None:
    rr = RuntimeResult(
        text="hi",
        structured=None,
        cost=_cost(),
        finish="stop",
        reasoning="step 1: think. step 2: answer.",
    )
    assert rr.reasoning == "step 1: think. step 2: answer."


def test_extract_message_reasoning_prefers_reasoning_content() -> None:
    msg = SimpleNamespace(reasoning_content="ds-r1 thought", reasoning="other")
    assert _extract_message_reasoning(msg) == "ds-r1 thought"


def test_extract_message_reasoning_falls_back_to_reasoning() -> None:
    msg = SimpleNamespace(reasoning="lean shape")
    assert _extract_message_reasoning(msg) == "lean shape"


def test_extract_message_reasoning_returns_none_when_absent() -> None:
    msg = SimpleNamespace(content="just text")
    assert _extract_message_reasoning(msg) is None


def test_extract_message_reasoning_treats_empty_as_absent() -> None:
    msg = SimpleNamespace(reasoning_content="", reasoning=None)
    assert _extract_message_reasoning(msg) is None


def test_extract_delta_reasoning_handles_both_field_names() -> None:
    a = SimpleNamespace(reasoning_content="chunk-a")
    b = SimpleNamespace(reasoning="chunk-b")
    assert _extract_delta_reasoning(a) == "chunk-a"
    assert _extract_delta_reasoning(b) == "chunk-b"


def test_extract_delta_reasoning_none_when_no_reasoning_field() -> None:
    delta = SimpleNamespace(content="visible text")
    assert _extract_delta_reasoning(delta) is None
