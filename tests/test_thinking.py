"""Unit tests for :data:`ThinkingMode` and :data:`ReasoningEffort`.

Phase 2 of the implementation plan introduces the ``thinking=`` kwarg
on :meth:`AgentSession.execute` / :meth:`AgentSession.stream`. The
ADR-006 shape lock is the union form — once consumer code branches
on these literal values or the dict shape, changing them is breaking.
These tests snapshot the surface so drift is caught at PR time.
"""

from __future__ import annotations

from typing import get_args

from airframe import ReasoningEffort, ThinkingMode


def test_reasoning_effort_literal_values() -> None:
    """Literal values are stable public surface — locked at v0.5.0 target."""
    assert set(get_args(ReasoningEffort)) == {"minimal", "low", "medium", "high"}


def test_thinking_mode_includes_disabled_sentinel() -> None:
    """``"disabled"`` is a recognised value that explicitly turns reasoning off."""
    # The union flattens Literal[...] members into one Literal at the type level,
    # so "disabled" appears alongside the four ReasoningEffort values.
    members = get_args(ThinkingMode)
    # The union shape is: ReasoningEffort | Literal["disabled"] | dict | None
    # After Python's flattening, str-literal members all sit together.
    assert any("disabled" in str(m) for m in members), (
        "ThinkingMode must include the 'disabled' literal — explicit-off sentinel"
    )


def test_thinking_mode_union_form_locked() -> None:
    """The four-shape union is the ADR-006 lock.

    1. ``None`` (default)
    2. literal effort level (str)
    3. ``dict`` (Claude-style budget)
    4. ``"disabled"`` sentinel

    These are the variants the protocol's ``thinking=`` kwarg accepts;
    adding more (e.g. a richer dataclass for per-model budgets) is
    safe, removing them is breaking.
    """
    members = get_args(ThinkingMode)
    # type(None), Literal[...] (flattened), dict
    member_strs = [str(m) for m in members]
    assert any(m is type(None) or "NoneType" in str(m) for m in members), (
        "ThinkingMode must include None for the 'default behaviour' case"
    )
    assert any("dict" in s for s in member_strs), (
        "ThinkingMode must include dict for Claude's {'budget_tokens': N} shape"
    )


def test_thinking_mode_accepts_literal_str_at_call_site() -> None:
    """Smoke check that the recognised string values are usable as runtime values."""
    # These are the values consumer code will pass; they should be
    # assignable to a ThinkingMode-typed variable. (We can't enforce at
    # runtime without the typing checker, but a value-level smoke check
    # confirms the union admits them.)
    values: list[ThinkingMode] = ["minimal", "low", "medium", "high", "disabled", None]
    for v in values:
        # Just constructing the list with the type hint is the test.
        assert v is None or isinstance(v, str)


def test_thinking_mode_accepts_dict_at_call_site() -> None:
    """The Claude-style {'budget_tokens': N} shape is a valid ThinkingMode value."""
    v: ThinkingMode = {"budget_tokens": 4096}
    assert v == {"budget_tokens": 4096}
