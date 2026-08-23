"""Unit tests for :class:`ZaiAnthropicRuntime`.

The adapter inherits ``ClaudeCodeRuntime``'s harness wholesale, so these
tests cover only what makes it a *different binding*: where it points,
which credential it carries, which credentials it refuses to carry, and
the narrower capability surface it declares.
"""

from __future__ import annotations

from typing import Any

import pytest

from airframe.adapters.claude_code import ClaudeCodeRuntime
from airframe.adapters.zai import (
    DEFAULT_ZAI_BASE_URL,
    DEFAULT_ZAI_MODEL,
    ZaiAnthropicRuntime,
)
from airframe.errors import RuntimeAuthError, UnsupportedFeatureError
from airframe.features import Feature
from airframe.protocol import ProviderModel


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise every env var this adapter reads or shadows."""
    for var in (
        "ZAI_API_KEY",
        "ZAI_BASE_URL",
        "ZAI_MODEL_OVERRIDE",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_MODEL_OVERRIDE",
    ):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Identity — this is a distinct binding, not a configured ClaudeCodeRuntime
# ---------------------------------------------------------------------------


def test_provider_id_is_zai_anthropic() -> None:
    """The ID names the wire format, leaving ``zai-openai`` free.

    Z.AI also exposes an OpenAI-compatible surface; claiming a bare
    ``"zai"`` here would strand that future sibling — the same reason
    the Kimi Agent SDK adapter never claimed ``"moonshot"``.
    """
    assert ZaiAnthropicRuntime.PROVIDER_ID == "zai-anthropic"


def test_shares_the_claude_agent_sdk_harness() -> None:
    """Same required package as ``claude`` — one harness, two bindings."""
    assert ZaiAnthropicRuntime.REQUIRES_PACKAGE == ClaudeCodeRuntime.REQUIRES_PACKAGE
    assert issubclass(ZaiAnthropicRuntime, ClaudeCodeRuntime)


def test_validate_binding_rejects_the_claude_provider_id() -> None:
    """A ``claude/...`` binding is not servable here, and vice versa."""
    zai = ZaiAnthropicRuntime(api_key="k")
    assert zai.validate_binding(ProviderModel("zai-anthropic", "glm-4.6"))
    assert not zai.validate_binding(ProviderModel("claude", "claude-sonnet-4-6"))
    assert not ClaudeCodeRuntime().validate_binding(ProviderModel("zai-anthropic", "glm-4.6"))


def test_default_model_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit arg → ``ZAI_MODEL_OVERRIDE`` → :data:`DEFAULT_ZAI_MODEL`."""
    assert ZaiAnthropicRuntime(api_key="k")._default_model == DEFAULT_ZAI_MODEL
    monkeypatch.setenv("ZAI_MODEL_OVERRIDE", "glm-4.5-air")
    assert ZaiAnthropicRuntime(api_key="k")._default_model == "glm-4.5-air"
    assert ZaiAnthropicRuntime(api_key="k", model="glm-explicit")._default_model == "glm-explicit"


def test_claude_model_override_env_does_not_leak_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """``CLAUDE_MODEL_OVERRIDE`` belongs to the other binding.

    The parent constructor consults it, so the subclass must resolve its
    own default *before* delegating or a Claude model id would become
    this runtime's default.
    """
    monkeypatch.setenv("CLAUDE_MODEL_OVERRIDE", "claude-opus-4-7")
    assert ZaiAnthropicRuntime(api_key="k")._default_model == DEFAULT_ZAI_MODEL


# ---------------------------------------------------------------------------
# Auth resolution
# ---------------------------------------------------------------------------


def test_resolve_api_key_prefers_explicit_arg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "from-env")
    assert ZaiAnthropicRuntime(api_key="explicit")._resolve_api_key() == "explicit"


def test_resolve_api_key_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "from-env")
    assert ZaiAnthropicRuntime()._resolve_api_key() == "from-env"


def test_resolve_api_key_raises_without_credential() -> None:
    """No Z.AI credential → a clear error naming the variable to set."""
    with pytest.raises(RuntimeAuthError) as excinfo:
        ZaiAnthropicRuntime()._resolve_api_key()
    assert "ZAI_API_KEY" in str(excinfo.value)


def test_resolve_api_key_never_falls_back_to_anthropic_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anthropic credentials must not stand in for a missing Z.AI key.

    They authenticate a different account at a different vendor; using
    one as a fallback is how a subscription token ends up POSTed to a
    third party.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-mine")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat-not-mine")
    with pytest.raises(RuntimeAuthError):
        ZaiAnthropicRuntime()._resolve_api_key()


# ---------------------------------------------------------------------------
# Subprocess environment — where the endpoint and the credential meet
# ---------------------------------------------------------------------------


def test_subprocess_env_points_the_cli_at_zai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "zai-secret")
    env = ZaiAnthropicRuntime()._subprocess_env()
    assert env["ANTHROPIC_BASE_URL"] == DEFAULT_ZAI_BASE_URL
    assert env["ANTHROPIC_AUTH_TOKEN"] == "zai-secret"


def test_subprocess_env_honours_base_url_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "k")
    monkeypatch.setenv("ZAI_BASE_URL", "https://proxy.internal/anthropic")
    env = ZaiAnthropicRuntime()._subprocess_env()
    assert env["ANTHROPIC_BASE_URL"] == "https://proxy.internal/anthropic"
    # Explicit constructor arg beats the env var.
    env2 = ZaiAnthropicRuntime(base_url="https://explicit/anthropic")._subprocess_env()
    assert env2["ANTHROPIC_BASE_URL"] == "https://explicit/anthropic"


@pytest.mark.parametrize(
    "leaky_var",
    ["ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_AUTH_TOKEN"],
)
def test_subprocess_env_shadows_inherited_anthropic_credentials(
    monkeypatch: pytest.MonkeyPatch,
    leaky_var: str,
) -> None:
    """Regression: an ambient Anthropic credential must not reach Z.AI.

    The Agent SDK merges ``ClaudeAgentOptions.env`` *over* ``os.environ``
    and cannot unset a key, so each Anthropic credential is shadowed with
    an empty string. Without this, a developer with
    ``CLAUDE_CODE_OAUTH_TOKEN`` exported in a shell profile would have
    the ``claude`` CLI carry their Anthropic subscription token to a
    third-party endpoint.
    """
    monkeypatch.setenv("ZAI_API_KEY", "zai-secret")
    monkeypatch.setenv(leaky_var, "sk-ant-SUBSCRIPTION-SECRET")

    env = ZaiAnthropicRuntime()._subprocess_env()

    if leaky_var == "ANTHROPIC_AUTH_TOKEN":
        # This slot is the one the Z.AI credential occupies.
        assert env[leaky_var] == "zai-secret"
    else:
        assert env[leaky_var] == "", f"{leaky_var} must be shadowed, got {env[leaky_var]!r}"
    assert "SUBSCRIPTION-SECRET" not in repr(env)


def test_subprocess_env_requires_a_credential() -> None:
    """Building the env is where a missing Z.AI key surfaces."""
    with pytest.raises(RuntimeAuthError):
        ZaiAnthropicRuntime()._subprocess_env()


def test_parent_subprocess_env_is_unaffected() -> None:
    """``ClaudeCodeRuntime`` keeps its original behaviour.

    The hook was extracted for this subclass; the parent must still
    return an empty dict when no explicit key was passed, and the
    ``ANTHROPIC_API_KEY`` override otherwise.
    """
    assert ClaudeCodeRuntime()._subprocess_env() == {}
    assert ClaudeCodeRuntime(api_key="sk-explicit")._subprocess_env() == {
        "ANTHROPIC_API_KEY": "sk-explicit"
    }


# ---------------------------------------------------------------------------
# Capability surface
# ---------------------------------------------------------------------------


def test_capability_surface_is_a_strict_subset_of_claude() -> None:
    """Narrower than the parent, never wider.

    The harness is identical, so anything this binding claims must be
    something the parent already claims.
    """
    assert ZaiAnthropicRuntime.SUPPORTED_FEATURES < ClaudeCodeRuntime.SUPPORTED_FEATURES


def test_structured_output_is_declared() -> None:
    """The conformance floor holds — every adapter implements ``schema=``."""
    assert ZaiAnthropicRuntime(api_key="k").supports(Feature.STRUCTURED_OUTPUT_JSON_SCHEMA)


@pytest.mark.parametrize(
    "feature",
    [
        Feature.COUNT_TOKENS,
        Feature.VISION_INPUT,
        Feature.FILE_INPUT,
        Feature.REASONING_EFFORT,
        Feature.RATE_LIMIT_TELEMETRY,
    ],
)
def test_endpoint_dependent_features_are_declined_pending_probe(feature: Feature) -> None:
    """Unverified endpoint capabilities report ``False``, not ``True``.

    Overstating produces a runtime failure the consumer was told could
    not happen; understating just routes them elsewhere.
    """
    assert not ZaiAnthropicRuntime(api_key="k").supports(feature)


@pytest.mark.parametrize(
    "feature",
    [
        Feature.STREAMING,
        Feature.SESSION_RESUME,
        Feature.CANCEL,
        Feature.TOOLS_FUNCTION,
        Feature.LIFECYCLE_HOOKS,
        Feature.PERMISSION_CALLBACK,
    ],
)
def test_cli_side_features_are_retained(feature: Feature) -> None:
    """Anything the local CLI implements is protocol-independent."""
    assert ZaiAnthropicRuntime(api_key="k").supports(feature)


# ---------------------------------------------------------------------------
# Overridden endpoints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_models_returns_the_static_catalog() -> None:
    """Z.AI serves chat completion, not Anthropic's ``/v1/models``."""
    models = await ZaiAnthropicRuntime(api_key="k").list_models()
    assert {m.id for m in models} == {"glm-4.6", "glm-4.5-air"}
    assert all(m.provider_id == "zai-anthropic" for m in models)
    glm = next(m for m in models if m.id == "glm-4.6")
    assert glm.context_window == 200_000
    # Subscription-billed: rates are absent rather than invented.
    assert glm.pricing_input_per_1k_usd is None


@pytest.mark.asyncio
async def test_list_models_makes_no_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Static catalog — the ``anthropic`` SDK must never be constructed."""
    import anthropic

    def _explode(**kwargs: Any) -> Any:  # pragma: no cover - must not be reached
        raise AssertionError("list_models() must not construct an HTTP client")

    monkeypatch.setattr(anthropic, "AsyncAnthropic", _explode)
    assert await ZaiAnthropicRuntime(api_key="k").list_models()


@pytest.mark.asyncio
async def test_count_tokens_declines_cleanly() -> None:
    """Inheriting the parent would POST to a route Z.AI does not serve."""
    with pytest.raises(UnsupportedFeatureError) as excinfo:
        await ZaiAnthropicRuntime(api_key="k").count_tokens("hello")
    assert excinfo.value.feature is Feature.COUNT_TOKENS


@pytest.mark.asyncio
async def test_count_tokens_declines_before_touching_the_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import anthropic

    def _explode(**kwargs: Any) -> Any:  # pragma: no cover - must not be reached
        raise AssertionError("count_tokens() must not construct an HTTP client")

    monkeypatch.setattr(anthropic, "AsyncAnthropic", _explode)
    with pytest.raises(UnsupportedFeatureError):
        await ZaiAnthropicRuntime(api_key="k").count_tokens("hello")
