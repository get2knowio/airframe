"""Behavioural integration suite for :class:`BedrockRuntime`.

Run explicitly::

    pytest -m integration tests/test_bedrock_integration.py

Auth resolves through the boto3 four-step chain (explicit args / env
vars / ``AWS_PROFILE`` / IAM instance role). ``AWS_REGION`` must be set
— Bedrock is region-pinned. Tests :func:`pytest.skip` themselves on
:class:`RuntimeAuthError` or when no credentials / region are
detectable in the environment.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("aioboto3")

from airframe.adapters.bedrock import BedrockRuntime  # noqa: E402
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


@pytest.fixture
async def adapter_runtime() -> BedrockRuntime:
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        pytest.skip("AWS_REGION not set; Bedrock is region-pinned")
    rt = BedrockRuntime(region_name=region)
    try:
        yield rt
    finally:
        await rt.close()
