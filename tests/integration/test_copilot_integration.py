"""Behavioural integration suite for :class:`CopilotRuntime`.

Imports every test from :mod:`airframe.testing.integration` and
runs it against a real :class:`CopilotRuntime`. Gated by the
``integration`` pytest marker.

Run explicitly::

    pytest -m integration tests/test_copilot_integration.py

Auth resolves through Copilot's own chain: ``GITHUB_TOKEN`` /
``GH_TOKEN`` / ``gh auth`` storage. Tests :func:`pytest.skip`
themselves on :class:`RuntimeAuthError`.
"""

from __future__ import annotations

import pytest

pytest.importorskip("copilot")

from airframe.adapters.copilot import CopilotRuntime  # noqa: E402
from airframe.testing.integration import (  # noqa: E402, F401
    test_integration_budget_usd_cap_trips,
    test_integration_function_tool_round_trip,
    test_integration_hook_observer_receives_events,
    test_integration_list_models,
    test_integration_permission_callback_fires,
    test_integration_plain_text_execute,
    test_integration_schema_round_trip,
    test_integration_stream_yields_text_then_turn_complete,
    test_integration_thinking_round_trip,
)


@pytest.fixture
async def adapter_runtime() -> CopilotRuntime:
    rt = CopilotRuntime()
    try:
        yield rt
    finally:
        await rt.close()
