"""Unit tests for the pre-flight ``count_tokens()`` surface.

Phase 6 of the feature roadmap. The OpenAI-compatible adapter family
counts via :mod:`tiktoken`; Claude counts via the anthropic SDK's
:meth:`messages.count_tokens` endpoint (covered by integration tests
since it requires creds); the other four adapters raise
:class:`UnsupportedFeatureError` per the strict capability gate.
"""

from __future__ import annotations

import pytest

from airframe.adapters.bedrock import BedrockRuntime
from airframe.adapters.copilot import CopilotRuntime
from airframe.adapters.opencode_server import OpenCodeServerRuntime
from airframe.adapters.opencode_zen import OpenCodeZenRuntime
from airframe.errors import UnsupportedFeatureError
from airframe.features import Feature
from airframe.inputs import ImageInput
from airframe.protocol import ProviderModel

# ---------------------------------------------------------------------------
# OpenAI-compatible adapter — tiktoken-backed
# ---------------------------------------------------------------------------


async def test_openai_compat_count_tokens_returns_positive_int() -> None:
    """``count_tokens()`` returns a positive count for plain text."""
    runtime = OpenCodeZenRuntime(api_key="dummy")
    try:
        count = await runtime.count_tokens("hello world")
        assert isinstance(count, int)
        assert count > 0
    finally:
        await runtime.close()


async def test_openai_compat_count_tokens_includes_system_prompt() -> None:
    """A system prompt adds tokens to the count."""
    runtime = OpenCodeZenRuntime(api_key="dummy")
    try:
        bare = await runtime.count_tokens("hello")
        with_system = await runtime.count_tokens("hello", system="be helpful")
        assert with_system > bare
    finally:
        await runtime.close()


async def test_openai_compat_count_tokens_falls_back_to_o200k_for_unknown_model() -> None:
    """Compat vendors expose model IDs tiktoken doesn't know; fall back cleanly."""
    runtime = OpenCodeZenRuntime(api_key="dummy")
    try:
        # qwen3 / not-a-real-model aren't in tiktoken's registry; expect
        # fall-back to o200k_base rather than raising.
        count = await runtime.count_tokens(
            "hello",
            model=ProviderModel("opencode-zen", "qwen3-totally-fake"),
        )
        assert count > 0
    finally:
        await runtime.close()


async def test_openai_compat_count_tokens_rejects_image_input() -> None:
    """v1 of count_tokens declines image attachments — deferred."""
    runtime = OpenCodeZenRuntime(api_key="dummy")
    try:
        with pytest.raises(UnsupportedFeatureError):
            await runtime.count_tokens(["hello", ImageInput(url="https://example.com/x.png")])
    finally:
        await runtime.close()


def test_openai_compat_declares_count_tokens_feature() -> None:
    runtime = OpenCodeZenRuntime(api_key="dummy")
    assert runtime.supports(Feature.COUNT_TOKENS) is True


# ---------------------------------------------------------------------------
# Non-supporting adapters — strict UnsupportedFeatureError
# ---------------------------------------------------------------------------


async def test_copilot_count_tokens_raises_unsupported() -> None:
    runtime = CopilotRuntime()
    assert runtime.supports(Feature.COUNT_TOKENS) is False
    with pytest.raises(UnsupportedFeatureError) as excinfo:
        await runtime.count_tokens("hello")
    assert excinfo.value.feature == Feature.COUNT_TOKENS


async def test_bedrock_count_tokens_raises_unsupported() -> None:
    runtime = BedrockRuntime()
    assert runtime.supports(Feature.COUNT_TOKENS) is False
    with pytest.raises(UnsupportedFeatureError):
        await runtime.count_tokens("hello")


async def test_opencode_server_count_tokens_raises_unsupported() -> None:
    runtime = OpenCodeServerRuntime()
    assert runtime.supports(Feature.COUNT_TOKENS) is False
    with pytest.raises(UnsupportedFeatureError):
        await runtime.count_tokens("hello")


def test_claude_runtime_declares_count_tokens_feature() -> None:
    from airframe.adapters.claude_code import ClaudeCodeRuntime

    runtime = ClaudeCodeRuntime()
    assert runtime.supports(Feature.COUNT_TOKENS) is True
