"""Behavioural integration suite for :class:`ZaiAnthropicRuntime`.

Run explicitly::

    ZAI_API_KEY=... pytest -m integration tests/test_zai_integration.py --timeout=300

Requires a real ``ZAI_API_KEY`` **and** the ``claude`` CLI on ``PATH`` —
the Agent SDK spawns it. Tests :func:`pytest.skip` themselves when either
is missing.

The imported set is deliberately narrower than
:mod:`tests.test_claude_code_integration`'s, because this binding declares
a narrower ``SUPPORTED_FEATURES``:

* ``test_integration_thinking_round_trip`` — ``REASONING_EFFORT`` is
  declined pending verification, so the contract would be asserting the
  decline, which the unit conformance suite already covers.
* ``test_integration_list_models`` — this adapter's ``list_models()`` is a
  static catalog with no network call (``zai.py``), so a green result
  would prove nothing about the endpoint.

Both become worth importing if ``examples/probe_zai.py`` promotes the
corresponding flags. That probe remains the tool for *discovering* what
Z.AI supports; this suite is for *regressing* what we have already
confirmed.
"""

from __future__ import annotations

import os
import shutil

import pytest

pytest.importorskip("claude_agent_sdk")

from airframe.adapters.zai import ZaiAnthropicRuntime  # noqa: E402
from airframe.testing.integration import (  # noqa: E402, F401
    test_integration_budget_usd_cap_trips,
    test_integration_function_tool_round_trip,
    test_integration_hook_observer_receives_events,
    test_integration_plain_text_execute,
    test_integration_schema_round_trip,
    test_integration_stream_yields_text_then_turn_complete,
)


@pytest.fixture
async def adapter_runtime() -> ZaiAnthropicRuntime:
    api_key = os.environ.get("ZAI_API_KEY")
    if not api_key:
        pytest.skip("ZAI_API_KEY not set")
    if shutil.which("claude") is None:
        pytest.skip("the `claude` CLI is not on PATH — the Agent SDK spawns it")
    rt = ZaiAnthropicRuntime(api_key=api_key)
    try:
        yield rt
    finally:
        await rt.close()
