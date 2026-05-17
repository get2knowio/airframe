"""Unit tests for :class:`FunctionTool` and the ``tools=`` shape lock.

Phase 3 Iteration A — protocol scaffolding. Tests here pin the
:class:`FunctionTool` field set, the frozen+slots invariants, and the
``handler`` signature contract (parsed Pydantic in, awaitable Any out).
Per-adapter capability-gate tests live in ``tests/test_features.py``
and the four ``tests/test_*_session.py`` files; behavioural tool
round-trip tests land alongside the wiring in Iterations B (OpenAI-
compat), C (Claude + Copilot), and D (Codex declination + probe).
"""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, fields, is_dataclass

import pytest
from pydantic import BaseModel

from airframe import FunctionTool, McpServerRef


class _AddParams(BaseModel):
    a: float
    b: float


async def _add(params: _AddParams) -> float:
    return params.a + params.b


def test_function_tool_is_frozen_dataclass() -> None:
    """The shape lock: FunctionTool is a frozen dataclass."""
    assert is_dataclass(FunctionTool)
    tool = FunctionTool(
        name="add",
        description="Add two numbers.",
        params=_AddParams,
        handler=_add,
    )
    with pytest.raises(FrozenInstanceError):
        tool.name = "renamed"  # type: ignore[misc]


def test_function_tool_uses_slots() -> None:
    """``slots=True`` avoids per-instance __dict__ — pinned for performance / shape."""
    tool = FunctionTool(
        name="add",
        description="Add two numbers.",
        params=_AddParams,
        handler=_add,
    )
    assert not hasattr(tool, "__dict__")


def test_function_tool_field_names_and_order() -> None:
    """Field set is part of the public surface and shape-locked."""
    expected = ["name", "description", "params", "handler"]
    actual = [f.name for f in fields(FunctionTool)]
    assert actual == expected


def test_handler_signature_is_async_taking_basemodel() -> None:
    """The canonical handler shape: async, one BaseModel arg, awaitable return."""
    import typing

    sig = inspect.signature(_add)
    params = list(sig.parameters.values())
    assert len(params) == 1
    # `from __future__ import annotations` turns annotations into strings;
    # resolve via get_type_hints to compare against the real class.
    hints = typing.get_type_hints(_add)
    assert hints[params[0].name] is _AddParams
    assert inspect.iscoroutinefunction(_add)


async def test_handler_runs_against_parsed_params() -> None:
    """End-to-end smoke: instantiate, parse args, await the handler."""
    tool = FunctionTool(
        name="add",
        description="Add two numbers.",
        params=_AddParams,
        handler=_add,
    )
    parsed = tool.params(a=17, b=23)
    result = await tool.handler(parsed)
    assert result == 40.0


def test_function_tool_exported_at_top_level() -> None:
    """``from airframe import FunctionTool`` works — pins the public path."""
    import airframe

    assert airframe.FunctionTool is FunctionTool
    assert "FunctionTool" in airframe.__all__


# --- McpServerRef shape lock (Phase 4 Iteration A) -----------------------


def test_mcp_server_ref_is_frozen_dataclass() -> None:
    """The shape lock: McpServerRef is a frozen dataclass."""
    assert is_dataclass(McpServerRef)
    ref = McpServerRef(
        name="everything",
        transport="stdio",
        command=["uvx", "mcp-server-everything"],
    )
    with pytest.raises(FrozenInstanceError):
        ref.name = "renamed"  # type: ignore[misc]


def test_mcp_server_ref_uses_slots() -> None:
    """``slots=True`` avoids per-instance __dict__ — pinned for shape."""
    ref = McpServerRef(
        name="everything",
        transport="stdio",
        command=["uvx", "mcp-server-everything"],
    )
    assert not hasattr(ref, "__dict__")


def test_mcp_server_ref_field_names_and_order() -> None:
    """Field set is part of the public surface and shape-locked.

    Phase 4 Iteration A locks ``name``, ``transport``, ``command``,
    ``url``, ``headers``, ``auth_token``. The plan defers a
    ``provider_options=`` slot for transport-specific knobs (stdio.env,
    http.timeout) until a consumer asks — adding a field later is
    additive; reordering is breaking.
    """
    expected = ["name", "transport", "command", "url", "headers", "auth_token"]
    actual = [f.name for f in fields(McpServerRef)]
    assert actual == expected


def test_mcp_server_ref_stdio_requires_command() -> None:
    """``transport='stdio'`` without ``command`` raises at construction."""
    with pytest.raises(ValueError, match="requires command="):
        McpServerRef(name="oops", transport="stdio")


def test_mcp_server_ref_stdio_rejects_url() -> None:
    """``transport='stdio'`` with ``url`` is a mistake — raise early."""
    with pytest.raises(ValueError, match="does not accept url="):
        McpServerRef(
            name="oops",
            transport="stdio",
            command=["uvx", "foo"],
            url="https://example.com",
        )


def test_mcp_server_ref_http_requires_url() -> None:
    """``transport='http'`` without ``url`` raises."""
    with pytest.raises(ValueError, match="requires url="):
        McpServerRef(name="oops", transport="http")


def test_mcp_server_ref_http_rejects_command() -> None:
    """``transport='http'`` with ``command`` is a mistake — raise early."""
    with pytest.raises(ValueError, match="does not accept command="):
        McpServerRef(
            name="oops",
            transport="http",
            command=["uvx", "foo"],
            url="https://example.com",
        )


def test_mcp_server_ref_sse_requires_url() -> None:
    """``transport='sse'`` without ``url`` raises."""
    with pytest.raises(ValueError, match="requires url="):
        McpServerRef(name="oops", transport="sse")


def test_mcp_server_ref_sse_rejects_command() -> None:
    """``transport='sse'`` with ``command`` is a mistake — raise early."""
    with pytest.raises(ValueError, match="does not accept command="):
        McpServerRef(
            name="oops",
            transport="sse",
            command=["uvx", "foo"],
            url="https://example.com",
        )


def test_mcp_server_ref_carries_headers_and_auth_token() -> None:
    """Bearer auth + extra headers pass through verbatim on network transports."""
    ref = McpServerRef(
        name="remote",
        transport="http",
        url="https://mcp.example.com",
        headers={"X-Trace": "abc"},
        auth_token="secret-bearer-token",
    )
    assert ref.headers == {"X-Trace": "abc"}
    assert ref.auth_token == "secret-bearer-token"


def test_mcp_server_ref_exported_at_top_level() -> None:
    """``from airframe import McpServerRef`` works — pins the public path."""
    import airframe

    assert airframe.McpServerRef is McpServerRef
    assert "McpServerRef" in airframe.__all__
