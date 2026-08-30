"""Unit tests for :class:`RequestMetadata` and per-adapter forwarding.

The cross-vendor metadata surface (``session(metadata=...)``,
``Feature.REQUEST_METADATA``) lands in Phase 6. Tests here cover:

* :class:`RequestMetadata` dataclass shape and defaults.
* OpenAI-compatible adapter's ``_apply_request_metadata`` helper that
  maps ``user_id`` / ``tags`` / ``request_id`` onto the three native
  OpenAI request channels.
* Silent-ignore contract — passing ``metadata=`` to a non-supporting
  adapter doesn't raise; it just drops the tag.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from airframe.adapters.opencode_zen import OpenCodeZenRuntime
from airframe.features import Feature
from airframe.metadata import RequestMetadata


def test_request_metadata_defaults() -> None:
    md = RequestMetadata()
    assert md.user_id is None
    assert md.request_id is None
    assert md.tags is None


def test_request_metadata_construction() -> None:
    md = RequestMetadata(
        user_id="user-123",
        request_id="req-abc",
        tags={"tenant": "acme", "env": "prod"},
    )
    assert md.user_id == "user-123"
    assert md.request_id == "req-abc"
    assert md.tags == {"tenant": "acme", "env": "prod"}


def test_request_metadata_is_frozen() -> None:
    md = RequestMetadata(user_id="user-1")
    with pytest.raises(Exception):  # FrozenInstanceError / AttributeError
        md.user_id = "user-2"  # type: ignore[misc]


def test_apply_request_metadata_maps_user_id_to_user_kwarg() -> None:
    """``user_id`` → ``user=`` kwarg on chat.completions.create."""
    from airframe.adapters.openai_compatible import OpenAICompatibleSession

    runtime = OpenCodeZenRuntime(api_key="dummy")
    sess = runtime.session(metadata=RequestMetadata(user_id="user-x"))
    assert isinstance(sess, OpenAICompatibleSession)
    kwargs: dict = {}
    sess._apply_request_metadata(kwargs)
    assert kwargs["user"] == "user-x"
    assert "metadata" not in kwargs
    assert "extra_headers" not in kwargs


def test_apply_request_metadata_maps_tags_to_metadata_kwarg() -> None:
    runtime = OpenCodeZenRuntime(api_key="dummy")
    sess = runtime.session(metadata=RequestMetadata(tags={"tenant": "acme"}))
    kwargs: dict = {}
    sess._apply_request_metadata(kwargs)  # type: ignore[attr-defined]
    assert kwargs["metadata"] == {"tenant": "acme"}


def test_apply_request_metadata_maps_request_id_to_extra_header() -> None:
    runtime = OpenCodeZenRuntime(api_key="dummy")
    sess = runtime.session(metadata=RequestMetadata(request_id="req-9"))
    kwargs: dict = {}
    sess._apply_request_metadata(kwargs)  # type: ignore[attr-defined]
    assert kwargs["extra_headers"] == {"X-Request-ID": "req-9"}


def test_apply_request_metadata_merges_with_existing_kwargs() -> None:
    """Pre-existing ``metadata=`` / ``extra_headers=`` are preserved + extended."""
    runtime = OpenCodeZenRuntime(api_key="dummy")
    sess = runtime.session(
        metadata=RequestMetadata(
            tags={"tenant": "acme"},
            request_id="req-9",
        )
    )
    kwargs: dict = {
        "metadata": {"env": "prod"},
        "extra_headers": {"X-Existing": "keep-me"},
    }
    sess._apply_request_metadata(kwargs)  # type: ignore[attr-defined]
    assert kwargs["metadata"] == {"env": "prod", "tenant": "acme"}
    assert kwargs["extra_headers"] == {"X-Existing": "keep-me", "X-Request-ID": "req-9"}


def test_apply_request_metadata_with_no_metadata_is_noop() -> None:
    runtime = OpenCodeZenRuntime(api_key="dummy")
    sess = runtime.session()  # no metadata
    kwargs: dict = {"model": "gpt-5-nano"}
    sess._apply_request_metadata(kwargs)  # type: ignore[attr-defined]
    assert kwargs == {"model": "gpt-5-nano"}


def test_openai_compatible_declares_request_metadata_feature() -> None:
    runtime = OpenCodeZenRuntime(api_key="dummy")
    assert runtime.supports(Feature.REQUEST_METADATA) is True


def test_metadata_kwarg_silently_dropped_on_non_supporting_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``session(metadata=...)`` on adapters that don't declare the feature
    accepts the kwarg without raising — the soft contract."""
    from airframe.adapters.bedrock import BedrockRuntime

    runtime = BedrockRuntime()
    try:
        # Should NOT raise UnsupportedFeatureError — soft contract.
        runtime.session(metadata=RequestMetadata(user_id="silent-drop"))
        # We can't verify the drop directly (no wire-level mock), but the
        # session built fine and supports() returns False as expected.
        assert runtime.supports(Feature.REQUEST_METADATA) is False
    finally:
        monkeypatch.setattr("warnings.warn", MagicMock())  # quiet any teardown noise


def test_claude_runtime_declares_request_metadata_feature() -> None:
    from airframe.adapters.claude_code import ClaudeCodeRuntime

    runtime = ClaudeCodeRuntime()
    assert runtime.supports(Feature.REQUEST_METADATA) is True
