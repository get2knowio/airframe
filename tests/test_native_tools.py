"""Unit tests for the native (vendor-hosted) tools surface.

Covers the portable :class:`NativeTool` / :class:`NativeCapability` types, the
shared :func:`_resolve_native_tools` gate + :func:`_native_tools_fingerprint`
helper, and the Claude adapter's capability→tool-name translation. SDK-free:
nothing here spawns a vendor subprocess.
"""

from __future__ import annotations

import pytest

from airframe import Feature, NativeCapability, NativeTool
from airframe.errors import UnsupportedFeatureError
from airframe.sessions import _native_tools_fingerprint, _resolve_native_tools

# --- NativeTool construction / validation ----------------------------------


def test_semantic_factories() -> None:
    ws = NativeTool.web_search()
    assert ws.capability is NativeCapability.WEB_SEARCH
    assert not ws.is_raw
    assert ws.provider_id is None and ws.name is None
    assert NativeTool.web_fetch().capability is NativeCapability.WEB_FETCH


def test_raw_factory() -> None:
    raw = NativeTool.raw("claude", "WebSearch")
    assert raw.is_raw
    assert raw.capability is None
    assert raw.provider_id == "claude"
    assert raw.name == "WebSearch"


def test_options_carried() -> None:
    assert NativeTool.web_search(max_uses=3).options == {"max_uses": 3}
    assert NativeTool.web_search().options is None  # empty kwargs → None


def test_xor_rejects_both_modes() -> None:
    with pytest.raises(ValueError, match="EITHER a semantic"):
        NativeTool(capability=NativeCapability.WEB_SEARCH, name="WebSearch")


def test_xor_rejects_neither_mode() -> None:
    with pytest.raises(ValueError, match="needs a semantic"):
        NativeTool()


def test_raw_requires_both_provider_and_name() -> None:
    with pytest.raises(ValueError, match="BOTH provider_id="):
        NativeTool(provider_id="claude")
    with pytest.raises(ValueError, match="BOTH provider_id="):
        NativeTool(name="WebSearch")


# --- _resolve_native_tools gate --------------------------------------------

_SERVED = frozenset({NativeCapability.WEB_SEARCH, NativeCapability.WEB_FETCH})


def _resolve(tools, *, served=_SERVED, supported=True, provider_id="claude"):
    return _resolve_native_tools(
        tools,
        adapter_label="test",
        provider_id=provider_id,
        feature_supported=supported,
        supported_capabilities=served,
    )


def test_resolve_empty_is_noop() -> None:
    assert _resolve(None) == []
    assert _resolve([]) == []


def test_resolve_passes_supported_semantic() -> None:
    tools = [NativeTool.web_search(), NativeTool.web_fetch()]
    assert _resolve(tools) == tools


def test_resolve_raises_on_unsupported_semantic() -> None:
    with pytest.raises(UnsupportedFeatureError) as exc:
        _resolve([NativeTool(capability=NativeCapability.CODE_EXECUTION)])
    assert exc.value.feature == Feature.TOOLS_NATIVE


def test_resolve_raises_when_feature_declined() -> None:
    with pytest.raises(UnsupportedFeatureError) as exc:
        _resolve([NativeTool.web_search()], supported=False, served=frozenset())
    assert exc.value.feature == Feature.TOOLS_NATIVE


def test_resolve_keeps_matching_raw_tool() -> None:
    raw = NativeTool.raw("claude", "WebSearch")
    assert _resolve([raw]) == [raw]


def test_resolve_drops_foreign_raw_tool() -> None:
    # Foreign raw tool on a declining adapter: relevant subset is empty, so no
    # raise even though the feature is unsupported.
    foreign = NativeTool.raw("openai", "web_search")
    assert _resolve([foreign], supported=False, served=frozenset()) == []


def test_resolve_mixed_list_partitions_by_provider() -> None:
    mine = NativeTool.raw("claude", "WebSearch")
    foreign = NativeTool.raw("openai", "web_search")
    semantic = NativeTool.web_fetch()
    assert _resolve([mine, foreign, semantic]) == [mine, semantic]


# --- _native_tools_fingerprint ---------------------------------------------


def test_fingerprint_stable_and_option_sensitive() -> None:
    assert _native_tools_fingerprint(None) == _native_tools_fingerprint([])
    a = _native_tools_fingerprint([NativeTool.web_search()])
    b = _native_tools_fingerprint([NativeTool.web_search(max_uses=3)])
    assert a != b
    # Deterministic regardless of option dict ordering.
    x = _native_tools_fingerprint([NativeTool.web_search(a=1, b=2)])
    y = _native_tools_fingerprint([NativeTool.web_search(b=2, a=1)])
    assert x == y


# --- Claude capability → tool-name translation -----------------------------


def test_claude_translation_maps_names() -> None:
    from airframe.adapters.claude_code import _translate_native_tools_for_claude

    names = _translate_native_tools_for_claude(
        [NativeTool.web_search(), NativeTool.web_fetch(), NativeTool.raw("claude", "WebSearch")]
    )
    assert names == ["WebSearch", "WebFetch", "WebSearch"]


def test_claude_runtime_serves_web_search_and_fetch() -> None:
    from airframe import ClaudeCodeRuntime

    rt = ClaudeCodeRuntime()
    assert rt.supports(Feature.TOOLS_NATIVE)
    assert rt.supported_native_tools() == _SERVED
