"""Behavioural integration suite for :class:`OpenCodeServerRuntime`.

Run explicitly::

    pytest -m integration tests/test_opencode_server_integration.py

These tests require a locally-running ``opencode serve`` (default
``http://127.0.0.1:4096``) with at least one upstream provider
configured via ``opencode auth login``. Override the server URL
with ``OPENCODE_SERVER_URL``; for non-loopback URLs also set
``OPENCODE_SERVER_PASSWORD`` (and ``OPENCODE_SERVER_USERNAME`` if
non-default).

Tests skip themselves on :class:`RuntimeServerStartError` /
:class:`RuntimeAuthError` so the suite degrades gracefully when no
server is reachable.
"""

from __future__ import annotations

import pytest

pytest.importorskip("opencode_ai")

from airframe.adapters.opencode_server import OpenCodeServerRuntime  # noqa: E402
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
# * test_integration_schema_round_trip — OpenCode 0.1.0a36 SDK has
#   no client.mcp resource, so the forced-tool shim that delivers
#   ``execute(schema=...)`` on other adapters can't ship. The
#   adapter raises ``UnsupportedFeatureError`` with a documented
#   pointer; this iteration of the integration suite skips schema.
# * test_integration_function_tool_round_trip — same root cause: no
#   runtime MCP registration. Future iteration flips this once the
#   SDK exposes ``client.mcp.*``.
# * test_integration_permission_callback_fires — opencode-ai
#   0.1.0a36 emits ``permission.updated`` events but exposes no
#   reply endpoint. ``PERMISSION_CALLBACK`` stays declined.


@pytest.fixture
async def adapter_runtime() -> OpenCodeServerRuntime:
    rt = OpenCodeServerRuntime()
    try:
        yield rt
    finally:
        await rt.close()
