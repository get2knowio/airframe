"""Unit tests for :mod:`airframe.discovery`.

Validates the top-level discovery helpers that drive provider menus
and `provider_id → adapter class` dispatch.

* :func:`airframe.list_providers` filters by which vendor SDKs are
  installed when ``installed_only=True`` (default), returns every
  built-in provider when ``installed_only=False``.
* :func:`airframe.runtime_for` returns the canonical adapter class
  for each known provider ID and raises informative errors for
  unknown / uninstalled providers.

The tests stub :func:`importlib.util.find_spec` so install-state
gating can be exercised without actually uninstalling SDKs in CI.
"""

from __future__ import annotations

from typing import Any

import pytest

from airframe import discovery
from airframe.adapters.claude_code import ClaudeCodeRuntime
from airframe.adapters.codex import CodexRuntime
from airframe.adapters.copilot import CopilotRuntime
from airframe.adapters.opencode_zen import OpenCodeZenRuntime
from airframe.discovery import list_providers, runtime_for


def _stub_find_spec(monkeypatch: pytest.MonkeyPatch, available: set[str]) -> None:
    """Stub ``importlib.util.find_spec`` so only `available` packages resolve."""

    def fake_find_spec(name: str, *args: Any, **kwargs: Any) -> Any:
        return object() if name in available else None

    monkeypatch.setattr(discovery.importlib.util, "find_spec", fake_find_spec)


# ---------------------------------------------------------------------------
# list_providers — install-state gating
# ---------------------------------------------------------------------------


def test_list_providers_returns_all_when_installed_only_false() -> None:
    """``installed_only=False`` surfaces every adapter's PROVIDER_ID."""
    providers = list_providers(installed_only=False)
    assert set(providers) == {"claude", "github-copilot", "codex", "opencode"}


def test_list_providers_sorted_alphabetically() -> None:
    providers = list_providers(installed_only=False)
    assert providers == sorted(providers)


def test_list_providers_filters_when_only_claude_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`pip install airframe-agents[claude]` users see only ``claude``."""
    _stub_find_spec(monkeypatch, available={"claude_agent_sdk"})
    assert list_providers() == ["claude"]


def test_list_providers_filters_when_only_copilot_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_find_spec(monkeypatch, available={"copilot"})
    assert list_providers() == ["github-copilot"]


def test_list_providers_filters_when_only_codex_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_find_spec(monkeypatch, available={"openai_codex_sdk"})
    assert list_providers() == ["codex"]


def test_list_providers_filters_when_only_openai_compat_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``[openai-compat]`` extra brings ``openai`` — gates ``opencode``."""
    _stub_find_spec(monkeypatch, available={"openai"})
    assert list_providers() == ["opencode"]


def test_list_providers_when_nothing_installed_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No SDKs installed → empty list (the honest signal)."""
    _stub_find_spec(monkeypatch, available=set())
    assert list_providers() == []


def test_list_providers_with_two_extras_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two extras installed → both providers surface, sorted."""
    _stub_find_spec(monkeypatch, available={"claude_agent_sdk", "copilot"})
    assert list_providers() == ["claude", "github-copilot"]


# ---------------------------------------------------------------------------
# runtime_for — canonical dispatch
# ---------------------------------------------------------------------------


def test_runtime_for_claude_returns_claude_code_runtime() -> None:
    assert runtime_for("claude") is ClaudeCodeRuntime


def test_runtime_for_github_copilot_returns_copilot_runtime() -> None:
    assert runtime_for("github-copilot") is CopilotRuntime


def test_runtime_for_codex_returns_codex_runtime() -> None:
    assert runtime_for("codex") is CodexRuntime


def test_runtime_for_opencode_returns_opencode_zen_runtime() -> None:
    assert runtime_for("opencode") is OpenCodeZenRuntime


def test_runtime_for_unknown_provider_raises_value_error() -> None:
    with pytest.raises(ValueError) as excinfo:
        runtime_for("not-a-provider")
    msg = str(excinfo.value)
    assert "not-a-provider" in msg
    # Helpful: lists what *is* known so the user sees what's possible.
    assert "claude" in msg or "Known providers" in msg


def test_runtime_for_dropped_alias_raises_value_error() -> None:
    """Previously-allowed alias like ``anthropic`` is no longer routed."""
    with pytest.raises(ValueError):
        runtime_for("anthropic")
    with pytest.raises(ValueError):
        runtime_for("copilot")  # legacy alias, dropped in v0.2.0


def test_runtime_for_uninstalled_provider_raises_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``copilot`` SDK isn't installed, ``runtime_for("github-copilot")`` raises."""
    _stub_find_spec(monkeypatch, available=set())  # nothing installed
    with pytest.raises(ImportError) as excinfo:
        runtime_for("github-copilot")
    msg = str(excinfo.value)
    # Points the user at the right pip extra.
    assert "airframe-agents[copilot]" in msg
    assert "CopilotRuntime" in msg


def test_runtime_for_uninstalled_claude_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_find_spec(monkeypatch, available=set())
    with pytest.raises(ImportError) as excinfo:
        runtime_for("claude")
    assert "airframe-agents[claude]" in str(excinfo.value)


def test_runtime_for_uninstalled_codex_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_find_spec(monkeypatch, available=set())
    with pytest.raises(ImportError) as excinfo:
        runtime_for("codex")
    assert "airframe-agents[codex]" in str(excinfo.value)


def test_runtime_for_uninstalled_opencode_names_the_openai_compat_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_find_spec(monkeypatch, available=set())
    with pytest.raises(ImportError) as excinfo:
        runtime_for("opencode")
    assert "airframe-agents[openai-compat]" in str(excinfo.value)


def test_runtime_for_returns_class_not_instance() -> None:
    """``runtime_for`` returns the class so callers control construction."""
    cls = runtime_for("claude")
    assert isinstance(cls, type)
    assert issubclass(cls, ClaudeCodeRuntime)
