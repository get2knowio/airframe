"""Unit tests for :class:`HookEvent` shape + the eight ``kind`` literals.

Phase 5 Iteration A — protocol scaffolding. The eight ``kind`` wire
values are shape-locked here so a future rename is caught at PR
time (same discipline ``test_feature_string_values_are_stable``
applies to :class:`~airframe.features.Feature`). Per-adapter
emission tests land in Iteration C.
"""

from __future__ import annotations

import typing
from dataclasses import FrozenInstanceError, fields, is_dataclass

import pytest

from airframe import HookEvent, HookEventKind


def test_hook_event_is_frozen_dataclass() -> None:
    """The shape lock: HookEvent is a frozen dataclass."""
    assert is_dataclass(HookEvent)
    event = HookEvent(kind="session_start", session_id=None, payload={})
    with pytest.raises(FrozenInstanceError):
        event.kind = "session_end"  # type: ignore[misc]


def test_hook_event_uses_slots() -> None:
    """``slots=True`` avoids per-instance __dict__ — pinned for shape."""
    event = HookEvent(kind="session_start", session_id=None, payload={})
    assert not hasattr(event, "__dict__")


def test_hook_event_field_names_and_order() -> None:
    """Field set is part of the public surface and shape-locked."""
    expected = ["kind", "session_id", "payload"]
    actual = [f.name for f in fields(HookEvent)]
    assert actual == expected


def test_hook_event_kind_strings_are_stable() -> None:
    """The eight kind literals are public surface — locked at v0.8.0.

    Consumer code branches on ``event.kind == "pre_tool_use"`` etc.;
    renaming any of these is a major-version break. Snapshotted here
    so any later rename trips this test at PR time.
    """
    args = typing.get_args(HookEventKind)
    assert set(args) == {
        "session_start",
        "session_end",
        "user_prompt_submit",
        "pre_tool_use",
        "post_tool_use",
        "tool_failure",
        "pre_compact",
        "rate_limit",
    }
    # Order matters too — it's what Literal[...] iterates in source
    # order, and it's the order the docstring documents.
    assert args == (
        "session_start",
        "session_end",
        "user_prompt_submit",
        "pre_tool_use",
        "post_tool_use",
        "tool_failure",
        "pre_compact",
        "rate_limit",
    )


def test_hook_event_session_id_optional() -> None:
    """``session_id`` is ``None`` on adapters with no server-side session
    or before the underlying session has been built."""
    event = HookEvent(kind="session_start", session_id=None, payload={"model": "gpt-5-mini"})
    assert event.session_id is None


def test_hook_event_payload_is_dict() -> None:
    """``payload`` is a plain ``dict[str, Any]`` for forward compatibility."""
    event = HookEvent(
        kind="pre_tool_use",
        session_id="sess-1",
        payload={"tool_name": "add", "tool_call_id": "c1", "arguments": '{"a":1}'},
    )
    assert isinstance(event.payload, dict)
    assert event.payload["tool_name"] == "add"


def test_hook_event_exports_at_top_level() -> None:
    """``from airframe import HookEvent`` works — pins the public path."""
    import airframe

    assert airframe.HookEvent is HookEvent
    assert airframe.HookEventKind is HookEventKind
    assert "HookEvent" in airframe.__all__
    assert "HookEventKind" in airframe.__all__
