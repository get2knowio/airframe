"""Unit tests for :class:`RuntimeEvent` and its variants.

Phase 1 of the implementation plan introduces the streaming event
taxonomy on :meth:`AgentSession.stream`. The variant set and
field-by-field shapes here are the ADR-003 shape lock — once consumer
code does ``match event:`` over these types, renaming a field or
removing a variant forces consumer rewrites. These tests snapshot
both so accidental drift is caught at PR time.
"""

from __future__ import annotations

import dataclasses

import pytest

from airframe import (
    CostRecord,
    ReasoningDelta,
    RuntimeEvent,
    RuntimeResult,
    TextDelta,
    ToolCallResult,
    ToolCallStart,
    TurnComplete,
)

# ---------------------------------------------------------------------------
# Variant set — locked at v0.4.0
# ---------------------------------------------------------------------------


def test_runtime_event_union_variants() -> None:
    """``RuntimeEvent`` is exactly these five variants.

    Adding a new variant later is safe (consumers branch with a
    wildcard / ``isinstance`` default arm). Removing one or renaming
    one is a major-version break.
    """
    # ``RuntimeEvent`` is a PEP 604 ``A | B | C`` union; its ``__args__``
    # exposes the member types in declaration order.
    assert RuntimeEvent.__args__ == (  # type: ignore[attr-defined]
        TextDelta,
        ReasoningDelta,
        ToolCallStart,
        ToolCallResult,
        TurnComplete,
    )


# ---------------------------------------------------------------------------
# Field shapes — locked at v0.4.0
# ---------------------------------------------------------------------------


def test_text_delta_shape() -> None:
    """``TextDelta(text: str)``."""
    fields = {f.name: f.type for f in dataclasses.fields(TextDelta)}
    assert fields == {"text": "str"}


def test_reasoning_delta_shape() -> None:
    """``ReasoningDelta(text: str)`` — distinct from TextDelta on purpose."""
    fields = {f.name: f.type for f in dataclasses.fields(ReasoningDelta)}
    assert fields == {"text": "str"}


def test_tool_call_start_shape() -> None:
    """``ToolCallStart(tool_name, tool_call_id, arguments_preview)``."""
    fields = {f.name: f.type for f in dataclasses.fields(ToolCallStart)}
    assert fields == {
        "tool_name": "str",
        "tool_call_id": "str",
        "arguments_preview": "str",
    }


def test_tool_call_result_shape() -> None:
    """``ToolCallResult(tool_call_id, output, is_error)``."""
    fields = {f.name: f.type for f in dataclasses.fields(ToolCallResult)}
    assert fields == {
        "tool_call_id": "str",
        "output": "Any",
        "is_error": "bool",
    }


def test_turn_complete_shape() -> None:
    """``TurnComplete(result: RuntimeResult)`` — carries the same shape ``execute()`` returns."""
    fields = {f.name: f.type for f in dataclasses.fields(TurnComplete)}
    assert fields == {"result": "RuntimeResult"}


# ---------------------------------------------------------------------------
# Discipline — every variant is frozen + slots
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [TextDelta, ReasoningDelta, ToolCallStart, ToolCallResult, TurnComplete],
)
def test_event_variants_are_frozen(cls: type) -> None:
    """All event variants are immutable.

    Same discipline as :class:`~airframe.protocol.RuntimeResult` and
    :class:`~airframe.cost.CostRecord` — events flow through async
    iterators that consumers may buffer; mutability would invite
    hard-to-trace aliasing bugs.
    """
    assert cls.__dataclass_params__.frozen is True
    # ``slots=True`` blocks attribute assignment and saves memory in
    # the hot streaming loop.
    assert "__slots__" in cls.__dict__


# ---------------------------------------------------------------------------
# Construction — sanity checks that the canonical shapes round-trip
# ---------------------------------------------------------------------------


def test_text_delta_construction() -> None:
    delta = TextDelta(text="hello")
    assert delta.text == "hello"


def test_tool_call_start_construction() -> None:
    start = ToolCallStart(
        tool_name="calculator",
        tool_call_id="call_1",
        arguments_preview='{"expr":',
    )
    assert start.tool_name == "calculator"
    assert start.tool_call_id == "call_1"
    assert start.arguments_preview == '{"expr":'


def test_tool_call_result_carries_arbitrary_output() -> None:
    """``output`` is ``Any`` — handlers return JSON-serialisable values."""
    result = ToolCallResult(tool_call_id="call_1", output={"value": 42}, is_error=False)
    assert result.output == {"value": 42}
    assert result.is_error is False


def test_turn_complete_wraps_runtime_result() -> None:
    """``TurnComplete.result`` is the same shape ``execute()`` returns."""
    runtime_result = RuntimeResult(
        text="done",
        structured=None,
        cost=CostRecord(
            provider_id="claude",
            model_id="claude-haiku-4-5",
            cost_usd=0.001,
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=0,
            cache_write_tokens=0,
            finish="end_turn",
        ),
        finish="end_turn",
    )
    event = TurnComplete(result=runtime_result)
    assert event.result is runtime_result
    assert event.result.text == "done"
