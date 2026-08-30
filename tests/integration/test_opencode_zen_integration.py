"""Behavioural integration suite for :class:`OpenCodeZenRuntime`.

Run explicitly::

    pytest -m integration tests/test_opencode_zen_integration.py

Auth resolves through ``OPENCODE_API_KEY`` / opencode auth.json.
Tests :func:`pytest.skip` themselves on :class:`RuntimeAuthError`.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("openai")

from airframe.adapters.opencode_zen import OpenCodeZenRuntime  # noqa: E402
from airframe.testing.integration import (  # noqa: E402, F401
    test_integration_budget_usd_cap_trips,
    test_integration_function_tool_round_trip,
    test_integration_hook_observer_receives_events,
    test_integration_list_models,
    test_integration_plain_text_execute,
    test_integration_schema_round_trip,
    test_integration_stream_yields_text_then_turn_complete,
    test_integration_thinking_round_trip,
)

# Permission callback is permanently declined on chat-completions —
# skip the test (it requires PERMISSION_CALLBACK + TOOLS_FUNCTION).
# Drop the import rather than ship a skipping shell.


@pytest.fixture
async def adapter_runtime() -> OpenCodeZenRuntime:
    api_key = os.environ.get("OPENCODE_API_KEY")
    if not api_key:
        pytest.skip("OPENCODE_API_KEY not set")
    rt = OpenCodeZenRuntime(api_key=api_key)
    try:
        yield rt
    finally:
        await rt.close()
