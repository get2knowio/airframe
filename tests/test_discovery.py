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
    assert set(providers) == {
        "bedrock",
        "claude",
        "github-copilot",
        "codex",
        "opencode-zen",
        "opencode-go",
        "openrouter",
    }


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
    """The ``[openai-compat]`` extra brings ``openai`` — gates every
    OpenAI-compatible adapter (``opencode-zen``, ``opencode-go``,
    ``openrouter``)."""
    _stub_find_spec(monkeypatch, available={"openai"})
    assert list_providers() == ["opencode-go", "opencode-zen", "openrouter"]


def test_list_providers_filters_when_only_bedrock_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``[bedrock]`` extra brings ``aioboto3`` — gates only ``bedrock``."""
    _stub_find_spec(monkeypatch, available={"aioboto3"})
    assert list_providers() == ["bedrock"]


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


def test_runtime_for_opencode_zen_returns_opencode_zen_runtime() -> None:
    assert runtime_for("opencode-zen") is OpenCodeZenRuntime


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
        runtime_for("copilot")  # legacy alias
    with pytest.raises(ValueError):
        runtime_for("opencode")  # bare alias replaced by opencode-zen / opencode-go


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


def test_runtime_for_uninstalled_opencode_zen_names_the_openai_compat_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_find_spec(monkeypatch, available=set())
    with pytest.raises(ImportError) as excinfo:
        runtime_for("opencode-zen")
    assert "airframe-agents[openai-compat]" in str(excinfo.value)


def test_runtime_for_uninstalled_bedrock_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_find_spec(monkeypatch, available=set())
    with pytest.raises(ImportError) as excinfo:
        runtime_for("bedrock")
    assert "airframe-agents[bedrock]" in str(excinfo.value)
    assert "BedrockRuntime" in str(excinfo.value)


def test_runtime_for_returns_class_not_instance() -> None:
    """``runtime_for`` returns the class so callers control construction."""
    cls = runtime_for("claude")
    assert isinstance(cls, type)
    assert issubclass(cls, ClaudeCodeRuntime)


# ---------------------------------------------------------------------------
# Third-party adapter discovery via the ``airframe.adapters`` entry point
# ---------------------------------------------------------------------------


class _FakeEntryPoint:
    """Minimal stand-in for :class:`importlib.metadata.EntryPoint`.

    The real EntryPoint type is awkward to construct directly; the
    interface ``discovery._entry_point_runtime_classes`` consumes is
    just ``name``, ``value``, and ``load()``. A small shim is simpler
    than instantiating the real thing.
    """

    def __init__(self, name: str, value: str, target: Any) -> None:
        self.name = name
        self.value = value
        self._target = target

    def load(self) -> Any:
        if isinstance(self._target, Exception):
            raise self._target
        return self._target


class _GoodThirdPartyRuntime:
    """A well-formed third-party adapter — has a PROVIDER_ID."""

    PROVIDER_ID = "fake-third-party"
    REQUIRES_PACKAGE = "fake_third_party_sdk"
    EXTRA_NAME = "fake-third-party"


class _CollidingThirdPartyRuntime:
    """A third-party adapter that tries to shadow a built-in."""

    PROVIDER_ID = "claude"
    REQUIRES_PACKAGE = "fake_third_party_sdk"


class _NoProviderIdRuntime:
    """Malformed — missing PROVIDER_ID."""


def _stub_entry_points(monkeypatch: pytest.MonkeyPatch, entries: list[_FakeEntryPoint]) -> None:
    """Stub ``importlib.metadata.entry_points`` for the discovery module."""

    def fake_entry_points(*, group: str) -> list[_FakeEntryPoint]:
        assert group == discovery.ENTRY_POINT_GROUP
        return entries

    monkeypatch.setattr(discovery, "entry_points", fake_entry_points)


def test_entry_point_adapter_appears_in_list_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A third-party adapter shows up in ``list_providers(installed_only=False)``.

    The unblocker for third-party packages — once they declare their
    runtime under ``airframe.adapters``, ``list_providers`` surfaces
    it alongside the built-ins.
    """
    _stub_entry_points(
        monkeypatch,
        [_FakeEntryPoint("fake", "fake_pkg:Runtime", _GoodThirdPartyRuntime)],
    )
    providers = list_providers(installed_only=False)
    assert "fake-third-party" in providers
    # Built-ins still surface unchanged.
    assert {
        "bedrock",
        "claude",
        "github-copilot",
        "codex",
        "opencode-zen",
        "opencode-go",
        "openrouter",
    } <= set(providers)


def test_entry_point_adapter_routed_by_runtime_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``runtime_for("fake-third-party")`` resolves the third-party class."""
    _stub_entry_points(
        monkeypatch,
        [_FakeEntryPoint("fake", "fake_pkg:Runtime", _GoodThirdPartyRuntime)],
    )
    # find_spec sees the third-party package as installed so we don't
    # hit the ImportError branch.
    _stub_find_spec(monkeypatch, available={"fake_third_party_sdk"})
    assert runtime_for("fake-third-party") is _GoodThirdPartyRuntime


def test_entry_point_adapter_filtered_when_sdk_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same pip-extras filter applies to third-party adapters.

    ``installed_only=True`` (the default) hides any provider whose
    ``REQUIRES_PACKAGE`` isn't importable — entry-point adapters are
    no exception.
    """
    _stub_entry_points(
        monkeypatch,
        [_FakeEntryPoint("fake", "fake_pkg:Runtime", _GoodThirdPartyRuntime)],
    )
    _stub_find_spec(monkeypatch, available=set())  # nothing installed
    assert list_providers(installed_only=True) == []
    # …but installed_only=False still surfaces it.
    assert "fake-third-party" in list_providers(installed_only=False)


def test_entry_point_load_error_is_logged_and_skipped(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A broken plugin doesn't break discovery.

    Third-party packages can ship malformed entry points (import-time
    errors, wrong target type). Discovery must keep working — the rest
    of the consumer's menu shouldn't disappear because one plugin is
    broken.
    """
    broken = _FakeEntryPoint("broken", "broken_pkg:does_not_exist", ImportError("boom"))
    good = _FakeEntryPoint("good", "fake_pkg:Runtime", _GoodThirdPartyRuntime)
    _stub_entry_points(monkeypatch, [broken, good])

    with caplog.at_level("WARNING"):
        providers = list_providers(installed_only=False)

    assert "fake-third-party" in providers
    assert any("failed to load" in record.message for record in caplog.records)


def test_entry_point_without_provider_id_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A class missing ``PROVIDER_ID`` is logged and ignored."""
    _stub_entry_points(
        monkeypatch,
        [_FakeEntryPoint("bad", "bad_pkg:NoId", _NoProviderIdRuntime)],
    )
    with caplog.at_level("WARNING"):
        providers = list_providers(installed_only=False)
    assert "_NoProviderIdRuntime" not in providers
    assert any("no PROVIDER_ID" in record.message for record in caplog.records)


def test_entry_point_shadowing_builtin_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Third-party PROVIDER_ID=='claude' loses to the built-in.

    Conservative-by-default: an installed plugin can't silently
    replace airframe's built-in routing. Authors who genuinely want to
    swap must pick a new provider ID.
    """
    _stub_entry_points(
        monkeypatch,
        [_FakeEntryPoint("clash", "clash:Runtime", _CollidingThirdPartyRuntime)],
    )
    with caplog.at_level("WARNING"):
        providers = list_providers(installed_only=False)
    # Still resolved to the built-in.
    assert runtime_for("claude") is ClaudeCodeRuntime
    # ``claude`` appears once, from the built-in — not duplicated.
    assert providers.count("claude") == 1
    assert any("shadows a built-in" in record.message for record in caplog.records)


def test_entry_point_non_class_target_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If an entry-point resolves to a function or instance, skip it."""

    def not_a_class() -> None: ...

    _stub_entry_points(
        monkeypatch,
        [_FakeEntryPoint("fn", "fn_pkg:not_a_class", not_a_class)],
    )
    with caplog.at_level("WARNING"):
        list_providers(installed_only=False)
    assert any("not a class" in record.message for record in caplog.records)


def test_entry_point_group_constant_is_stable() -> None:
    """``ENTRY_POINT_GROUP`` is public surface — pin it.

    Renaming this would invalidate every third-party
    ``pyproject.toml`` declaration. Snapshotted so accidental renames
    fail review.
    """
    assert discovery.ENTRY_POINT_GROUP == "airframe.adapters"
