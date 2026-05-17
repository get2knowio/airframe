"""Behavioural integration suite for :class:`CodexRuntime`.

Run explicitly::

    pytest -m integration tests/test_codex_integration.py

Auth resolves through ``OPENAI_API_KEY`` / ``CODEX_API_KEY`` / the
opencode auth.json / ``~/.codex/auth.json``. Tests
:func:`pytest.skip` themselves on :class:`RuntimeAuthError`.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openai_codex_sdk")

from airframe.adapters.codex import CodexRuntime  # noqa: E402
from airframe.testing.integration import (  # noqa: E402, F401
    test_integration_budget_usd_cap_trips,
    test_integration_hook_observer_receives_events,
    test_integration_list_models,
    test_integration_plain_text_execute,
    test_integration_schema_round_trip,
    test_integration_stream_yields_text_then_turn_complete,
    test_integration_thinking_round_trip,
)

# Codex doesn't declare TOOLS_FUNCTION (no SDK tool-registration channel)
# nor PERMISSION_CALLBACK per-call — the tool-using integration tests
# self-skip via their capability gates, so importing them is harmless
# but adds noise. Drop them from this suite to keep failing-test counts
# honest.


@pytest.fixture
async def adapter_runtime() -> CodexRuntime:
    rt = CodexRuntime()
    try:
        yield rt
    finally:
        await rt.close()
