"""Behavioural integration suite for :class:`ClaudeCodeRuntime`.

Imports every test from :mod:`airframe.testing.integration` and runs
it against a real :class:`ClaudeCodeRuntime`. Gated by the
``integration`` pytest marker; the default ``make test`` excludes it.

Run explicitly::

    pytest -m integration tests/test_claude_code_integration.py

Auth resolution follows :class:`ClaudeCodeRuntime` itself: any of
``CLAUDE_CODE_OAUTH_TOKEN``, ``~/.claude/.credentials.json``, or
``ANTHROPIC_API_KEY``. Tests :func:`pytest.skip` themselves on
:class:`RuntimeAuthError`.
"""

from __future__ import annotations

import pytest

# Skip the whole module if the vendor SDK isn't installed (matches the
# pattern other in-tree integration suites will follow once we have
# multiple compat-vendor adapters in tree).
pytest.importorskip("claude_agent_sdk")

from airframe.adapters.claude_code import ClaudeCodeRuntime  # noqa: E402
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
async def adapter_runtime() -> ClaudeCodeRuntime:
    """A live ClaudeCodeRuntime instance.

    No credentials passed explicitly — the adapter's own auth chain
    resolves them. The integration tests :func:`pytest.skip` if no
    credential is available.
    """
    rt = ClaudeCodeRuntime()
    try:
        yield rt
    finally:
        await rt.close()
