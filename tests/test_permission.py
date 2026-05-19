"""Unit tests for the :class:`PermissionCallback` / :class:`PermissionRequest`
/ :data:`PermissionDecision` shapes.

Phase 5 Iteration A — protocol scaffolding. Tests here pin the
:class:`PermissionRequest` field set, the frozen+slots invariants,
the :class:`PermissionCallback` Protocol shape, and the literal
values on :data:`PermissionDecision`. Per-adapter capability-gate
tests live in :mod:`tests.test_features`; behavioural permission
round-trip tests land alongside the wiring in Iteration B (Claude /
Copilot / Kimi accepting paths; OpenAI-compat permanent decline).
"""

from __future__ import annotations

import inspect
import typing
from dataclasses import FrozenInstanceError, fields, is_dataclass

import pytest

from airframe import PermissionCallback, PermissionDecision, PermissionRequest


def test_permission_request_is_frozen_dataclass() -> None:
    """The shape lock: PermissionRequest is a frozen dataclass."""
    assert is_dataclass(PermissionRequest)
    req = PermissionRequest(tool_name="read_file", tool_args={"path": "/etc/hosts"})
    with pytest.raises(FrozenInstanceError):
        req.tool_name = "renamed"  # type: ignore[misc]


def test_permission_request_uses_slots() -> None:
    """``slots=True`` avoids per-instance __dict__ — pinned for shape."""
    req = PermissionRequest(tool_name="read_file", tool_args={})
    assert not hasattr(req, "__dict__")


def test_permission_request_field_names_and_order() -> None:
    """Field set is part of the public surface and shape-locked."""
    expected = ["tool_name", "tool_args", "reason"]
    actual = [f.name for f in fields(PermissionRequest)]
    assert actual == expected


def test_permission_request_reason_defaults_to_none() -> None:
    """``reason`` is optional; defaults to None for vendors that don't expose one."""
    req = PermissionRequest(tool_name="x", tool_args={})
    assert req.reason is None


def test_permission_decision_literal_values() -> None:
    """The three literal values are public surface — locked at v0.8.0.

    Once consumer code branches on ``decision == "allow"`` etc.,
    renaming any of these is a major-version break. Snapshotting
    the literal values here catches drift at PR time.
    """
    args = typing.get_args(PermissionDecision)
    assert set(args) == {"allow", "deny", "defer"}


def test_permission_callback_protocol_method_set() -> None:
    """:class:`PermissionCallback` exposes exactly one method: ``handle``.

    Consumer code implements just this signature; adding methods
    later is fine, renaming or removing is breaking.
    """
    members = {
        name
        for name, value in inspect.getmembers(PermissionCallback)
        if not name.startswith("_")
        and (inspect.isfunction(value) or inspect.iscoroutinefunction(value))
    }
    assert members == {"handle"}


def test_permission_callback_handle_is_async() -> None:
    """``handle()`` is async so adapters can await user code that
    needs network / disk."""
    assert inspect.iscoroutinefunction(PermissionCallback.handle)


def test_permission_callback_handle_signature() -> None:
    """``handle(self, request: PermissionRequest) -> PermissionDecision``."""
    sig = inspect.signature(PermissionCallback.handle)
    assert list(sig.parameters) == ["self", "request"]


def test_permission_callback_is_runtime_checkable() -> None:
    """``isinstance`` works on structural matches for ergonomic
    wrapping of plain callables / classes."""

    class _MyCallback:
        async def handle(self, request: PermissionRequest) -> PermissionDecision:
            return "allow"

    assert isinstance(_MyCallback(), PermissionCallback)


def test_permission_exports_at_top_level() -> None:
    """``from airframe import …`` works — pins the public path."""
    import airframe

    assert airframe.PermissionRequest is PermissionRequest
    assert airframe.PermissionCallback is PermissionCallback
    assert airframe.PermissionDecision is PermissionDecision
    for name in ("PermissionRequest", "PermissionCallback", "PermissionDecision"):
        assert name in airframe.__all__
