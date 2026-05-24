"""Unit tests for :class:`CacheConfig` and per-adapter forwarding.

Phase 6 of the feature roadmap. Cross-vendor prompt-cache controls
(``session(cache=CacheConfig(...))``) — the OpenAI-compatible adapter
translates onto ``prompt_cache_key`` / ``prompt_cache_retention``;
the other adapters accept the kwarg and drop it silently per the soft
contract.
"""

from __future__ import annotations

import pytest

from airframe.adapters.bedrock import BedrockRuntime
from airframe.adapters.opencode_zen import OpenCodeZenRuntime
from airframe.cache import CacheConfig
from airframe.features import Feature


def test_cache_config_defaults() -> None:
    c = CacheConfig()
    assert c.key is None
    assert c.retention is None


def test_cache_config_construction() -> None:
    c = CacheConfig(key="sess-42", retention="long")
    assert c.key == "sess-42"
    assert c.retention == "long"


def test_cache_config_is_frozen() -> None:
    c = CacheConfig(key="x")
    with pytest.raises(Exception):  # FrozenInstanceError
        c.key = "y"  # type: ignore[misc]


def test_apply_cache_config_maps_key_to_prompt_cache_key() -> None:
    runtime = OpenCodeZenRuntime(api_key="dummy")
    sess = runtime.session(cache=CacheConfig(key="agent:42"))
    kwargs: dict = {}
    sess._apply_cache_config(kwargs)  # type: ignore[attr-defined]
    assert kwargs["prompt_cache_key"] == "agent:42"
    assert "prompt_cache_retention" not in kwargs


def test_apply_cache_config_maps_short_retention_to_in_memory() -> None:
    runtime = OpenCodeZenRuntime(api_key="dummy")
    sess = runtime.session(cache=CacheConfig(key="k", retention="short"))
    kwargs: dict = {}
    sess._apply_cache_config(kwargs)  # type: ignore[attr-defined]
    assert kwargs["prompt_cache_retention"] == "in_memory"


def test_apply_cache_config_maps_long_retention_to_24h() -> None:
    runtime = OpenCodeZenRuntime(api_key="dummy")
    sess = runtime.session(cache=CacheConfig(key="k", retention="long"))
    kwargs: dict = {}
    sess._apply_cache_config(kwargs)  # type: ignore[attr-defined]
    assert kwargs["prompt_cache_retention"] == "24h"


def test_apply_cache_config_with_no_cache_is_noop() -> None:
    runtime = OpenCodeZenRuntime(api_key="dummy")
    sess = runtime.session()  # no cache
    kwargs: dict = {"model": "gpt-5-nano"}
    sess._apply_cache_config(kwargs)  # type: ignore[attr-defined]
    assert kwargs == {"model": "gpt-5-nano"}


def test_apply_cache_config_overrides_provider_options_key() -> None:
    """The portable cache= takes precedence over OpenAICompatOptions.prompt_cache_key."""
    from airframe.options import OpenAICompatOptions

    runtime = OpenCodeZenRuntime(api_key="dummy")
    sess = runtime.session(
        cache=CacheConfig(key="portable-wins"),
        provider_options=OpenAICompatOptions(prompt_cache_key="vendor-loses"),
    )
    kwargs: dict = {}
    sess._apply_provider_options(kwargs)  # type: ignore[attr-defined]
    sess._apply_cache_config(kwargs)  # type: ignore[attr-defined]
    assert kwargs["prompt_cache_key"] == "portable-wins"


def test_openai_compatible_declares_prompt_cache_feature() -> None:
    runtime = OpenCodeZenRuntime(api_key="dummy")
    assert runtime.supports(Feature.PROMPT_CACHE_CONTROL) is True


def test_cache_kwarg_silently_dropped_on_non_supporting_adapter() -> None:
    """Soft contract: passing cache= to a non-supporter doesn't raise."""
    runtime = BedrockRuntime()
    assert runtime.supports(Feature.PROMPT_CACHE_CONTROL) is False
    # Should NOT raise.
    runtime.session(cache=CacheConfig(key="silent-drop", retention="short"))


def test_claude_runtime_does_not_declare_prompt_cache_feature() -> None:
    """Claude Agent SDK manages caching via session warmth, not an explicit key."""
    from airframe.adapters.claude_code import ClaudeCodeRuntime

    runtime = ClaudeCodeRuntime()
    assert runtime.supports(Feature.PROMPT_CACHE_CONTROL) is False
