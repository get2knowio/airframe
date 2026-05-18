"""Behavioural integration suite for :class:`KimiRuntime`.

Run explicitly::

    pytest -m integration tests/test_kimi_integration.py

Auth resolves through ``KIMI_API_KEY``. Tests :func:`pytest.skip`
themselves on :class:`RuntimeAuthError` when no credential surfaces.

**Co-installation note.** ``kimi-agent-sdk`` cannot be co-installed
with ``claude-agent-sdk`` (mcp version conflict — see
``pyproject.toml``'s ``[tool.uv.conflicts]`` and
``CLAUDE.md``). Run this suite from a fresh venv that has the
``[kimi]`` extra installed and the ``[claude]`` extra *not*
installed::

    python3.12 -m venv .venv-kimi
    .venv-kimi/bin/pip install -U pip
    .venv-kimi/bin/pip install -e '.[kimi,testing]'
    .venv-kimi/bin/pytest -m integration tests/test_kimi_integration.py
"""

from __future__ import annotations

import pytest

pytest.importorskip("kimi_agent_sdk")

from airframe.adapters.kimi import KimiRuntime  # noqa: E402
from airframe.testing.integration import (  # noqa: E402, F401
    test_integration_budget_usd_cap_trips,
    test_integration_hook_observer_receives_events,
    test_integration_list_models,
    test_integration_plain_text_execute,
    test_integration_stream_yields_text_then_turn_complete,
    test_integration_thinking_round_trip,
)

# Intentionally NOT imported:
#
# * test_integration_schema_round_trip — KimiRuntime declines
#   ``execute(schema=…)`` with NotImplementedError until a later
#   iteration wires the MCP-based forced-tool path. Including the
#   test in the suite would fail rather than skip, since the test's
#   schema kwarg isn't a capability gate.
# * test_integration_function_tool_round_trip — Kimi declines
#   ``tools=`` permanently (no SDK channel for Python callables);
#   wrap as MCP instead. The integration helper's capability gate
#   would self-skip, but we drop it from the suite to keep the
#   failing-test count honest.
# * test_integration_permission_callback_fires — Kimi's
#   PermissionCallback fires per ApprovalRequest, but driving an
#   ApprovalRequest in the integration suite would require a
#   tool-needing prompt; deferred until a dedicated live probe.


@pytest.fixture
async def adapter_runtime() -> KimiRuntime:
    rt = KimiRuntime()
    try:
        yield rt
    finally:
        await rt.close()
