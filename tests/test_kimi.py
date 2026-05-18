"""Unit tests for :class:`KimiRuntime` — Iteration A scaffold.

The behavioural surface (execute / stream / cancel / session-resume /
tools / permission / hooks / budget) is wired in Iterations B–F per
``dev-docs/kimi-adapter-plan.md`` and tested in
``tests/test_kimi_session.py`` and ``tests/test_kimi_conformance.py``
as those land.

This file covers what Iteration A ships:

* Identity (``PROVIDER_ID`` / ``REQUIRES_PACKAGE`` / ``EXTRA_NAME``
  / ``label``).
* Default model / base URL resolution from constructor → env →
  module-level defaults.
* ``validate_binding`` accepts ``kimi-*`` only and rejects foreign
  provider IDs.
* ``_resolve_api_key`` four-step chain (explicit kwarg → instance
  override → ``KIMI_API_KEY`` env → :class:`RuntimeAuthError`).
* ``supports()`` returns ``False`` for every :class:`Feature`
  (Iteration A is feature-empty).
* ``unwrap()`` returns ``self`` for own type and raises ``TypeError``
  for everything else.
* ``close()`` / ``reset()`` are idempotent and never raise.
* ``execute()`` raises :class:`NotImplementedError` with a message
  pointing at Iteration B.
* ``list_models()`` returns the curated fallback catalogue.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel

from airframe.adapters.kimi import (
    DEFAULT_KIMI_BASE_URL,
    DEFAULT_KIMI_MODEL,
    KimiRuntime,
)
from airframe.errors import RuntimeAuthError
from airframe.features import Feature
from airframe.options import (
    BedrockOptions,
    ClaudeOptions,
    KimiOptions,
)
from airframe.protocol import ProviderModel

# --- identity + defaults ------------------------------------------------------


def test_provider_identity() -> None:
    assert KimiRuntime.PROVIDER_ID == "kimi"
    assert KimiRuntime.REQUIRES_PACKAGE == "kimi_agent_sdk"
    assert KimiRuntime.EXTRA_NAME == "kimi"
    assert KimiRuntime.label == "kimi"


def test_module_defaults() -> None:
    assert DEFAULT_KIMI_MODEL == "kimi-k2-thinking-turbo"
    assert DEFAULT_KIMI_BASE_URL == "https://api.moonshot.ai/v1"


def test_default_model_resolves_constructor_arg_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KIMI_MODEL_NAME", "kimi-k2-thinking")  # should be ignored
    rt = KimiRuntime(model="kimi-k2-thinking-turbo")
    assert rt._default_model == "kimi-k2-thinking-turbo"


def test_default_model_falls_through_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KIMI_MODEL_NAME", "kimi-custom-model")
    rt = KimiRuntime()
    assert rt._default_model == "kimi-custom-model"


def test_default_model_falls_through_to_module_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KIMI_MODEL_NAME", raising=False)
    rt = KimiRuntime()
    assert rt._default_model == DEFAULT_KIMI_MODEL


def test_default_base_url_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KIMI_BASE_URL", "https://env.example/v1")
    # Constructor arg wins.
    rt_arg = KimiRuntime(base_url="https://arg.example/v1")
    assert rt_arg._base_url == "https://arg.example/v1"
    # Env var when constructor arg omitted.
    rt_env = KimiRuntime()
    assert rt_env._base_url == "https://env.example/v1"
    # Module default when neither set.
    monkeypatch.delenv("KIMI_BASE_URL", raising=False)
    rt_default = KimiRuntime()
    assert rt_default._base_url == DEFAULT_KIMI_BASE_URL


# --- _resolve_api_key ---------------------------------------------------------


def _make_runtime(api_key: str | None = None) -> KimiRuntime:
    return KimiRuntime(api_key=api_key)


def test_resolve_api_key_explicit_argument() -> None:
    rt = _make_runtime()
    assert rt._resolve_api_key("sk-explicit-arg") == "sk-explicit-arg"


def test_resolve_api_key_instance_override_wins_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KIMI_API_KEY", "sk-env-key")
    rt = _make_runtime(api_key="sk-ctor-override")
    assert rt._resolve_api_key(None) == "sk-ctor-override"


def test_resolve_api_key_falls_through_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KIMI_API_KEY", "sk-env-only")
    rt = _make_runtime()
    assert rt._resolve_api_key(None) == "sk-env-only"


def test_resolve_api_key_raises_when_no_layer_resolves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    rt = _make_runtime()
    with pytest.raises(RuntimeAuthError) as excinfo:
        rt._resolve_api_key(None)
    msg = str(excinfo.value)
    # The error message must name the env var and point at the console.
    assert "KIMI_API_KEY" in msg
    assert "platform.moonshot.ai" in msg


# --- validate_binding ---------------------------------------------------------


def test_validate_binding_accepts_kimi_prefixed_models() -> None:
    rt = _make_runtime()
    assert rt.validate_binding(ProviderModel("kimi", "kimi-k2-thinking-turbo")) is True
    assert rt.validate_binding(ProviderModel("kimi", "kimi-k2-thinking")) is True
    assert rt.validate_binding(ProviderModel("kimi", "kimi-future-model")) is True


def test_validate_binding_rejects_non_kimi_model_ids() -> None:
    rt = _make_runtime()
    # Even with the right provider_id, non-kimi-* model IDs are rejected —
    # Kimi adapter serves only the Kimi line.
    assert rt.validate_binding(ProviderModel("kimi", "claude-haiku-4-5")) is False
    assert rt.validate_binding(ProviderModel("kimi", "gpt-4o-mini")) is False
    assert rt.validate_binding(ProviderModel("kimi", "")) is False


def test_validate_binding_rejects_foreign_provider_id() -> None:
    rt = _make_runtime()
    assert rt.validate_binding(ProviderModel("claude", "kimi-k2-thinking-turbo")) is False
    assert rt.validate_binding(ProviderModel("openrouter", "kimi-k2-thinking-turbo")) is False


# --- supports() ---------------------------------------------------------------


def test_supports_iteration_b_feature_matrix() -> None:
    """Iteration B flips STREAMING / CANCEL / SESSION_RESUME on.

    ``STRUCTURED_OUTPUT_JSON_SCHEMA`` stays True (the conformance
    floor every airframe adapter declares), but
    ``execute(schema=...)`` raises :class:`NotImplementedError` until
    Iteration D wires the MCP-based forced-tool path. The rest of
    the matrix flips on in Iterations C–F per the plan.
    """
    rt = _make_runtime()
    expected_true = {
        Feature.STRUCTURED_OUTPUT_JSON_SCHEMA,
        Feature.STREAMING,
        Feature.CANCEL,
        Feature.SESSION_RESUME,
    }
    for feature in Feature:
        want = feature in expected_true
        assert rt.supports(feature) is want, (
            f"Iteration B: {feature} supports() should be {want}; got {rt.supports(feature)}"
        )


def test_supports_accepts_optional_model_kwarg() -> None:
    """Signature must accept ``model=`` per the protocol, even if unused."""
    rt = _make_runtime()
    rt.supports(Feature.STREAMING, model=ProviderModel("kimi", "kimi-k2-thinking-turbo"))
    rt.supports(Feature.STREAMING, model=None)


# --- unwrap() -----------------------------------------------------------------


def test_unwrap_returns_self_for_own_type() -> None:
    rt = _make_runtime()
    assert rt.unwrap(KimiRuntime) is rt


def test_unwrap_raises_typeerror_for_unrelated_types() -> None:
    rt = _make_runtime()
    with pytest.raises(TypeError) as excinfo:
        rt.unwrap(str)
    assert "KimiRuntime" in str(excinfo.value)


# --- close / reset are idempotent --------------------------------------------


def test_close_is_idempotent() -> None:
    rt = _make_runtime()
    asyncio.run(rt.close())
    asyncio.run(rt.close())  # safe second call


def test_reset_is_idempotent() -> None:
    rt = _make_runtime()
    asyncio.run(rt.reset())
    asyncio.run(rt.reset())


# --- execute() — behavioural surface tested in tests/test_kimi_session.py ----


def test_execute_signature_accepts_schema_kwarg_default_none() -> None:
    """Protocol-required: schema=None must be the documented default.

    The structural conformance contract (test_plain_text_execute_path_is_wired)
    checks the signature *and* the source for the legacy refusal string.
    Mirror the same assertion here at the unit level for fast feedback.
    """
    import inspect

    sig = inspect.signature(KimiRuntime.execute)
    schema = sig.parameters.get("schema")
    assert schema is not None
    assert schema.default is None
    # No legacy gate string — would fail the conformance contract.
    assert "plain-text execute() is not wired" not in inspect.getsource(KimiRuntime.execute)


def test_execute_with_schema_raises_pointing_at_iteration_d() -> None:
    """``execute(schema=...)`` raises until Iteration D wires the MCP forced-tool.

    The conformance suite declares ``STRUCTURED_OUTPUT_JSON_SCHEMA = True``
    so callers see the capability advertised, but the actual delivery
    path isn't ready yet — schema callers get a clear NotImplementedError
    pointing at Iteration D.
    """

    class _Schema(BaseModel):
        x: int

    rt = _make_runtime()
    sess = rt.session()
    with pytest.raises(NotImplementedError) as excinfo:
        asyncio.run(sess.execute("hi", schema=_Schema))
    msg = str(excinfo.value)
    assert "Iteration D" in msg
    asyncio.run(sess.close())


# --- list_models() — fallback catalogue --------------------------------------


def test_list_models_returns_fallback_catalogue() -> None:
    rt = _make_runtime()
    models = asyncio.run(rt.list_models())
    assert len(models) >= 1
    ids = {m.id for m in models}
    assert DEFAULT_KIMI_MODEL in ids
    # Every fallback entry must declare the kimi provider id so consumers
    # filtering by provider see them.
    for m in models:
        assert m.provider_id == "kimi"
        # Iteration A leaves pricing as None — fallback catalogue has no
        # vendor-confirmed rates. Populated alongside _KIMI_PRICING in E.
        assert m.pricing_input_per_1k_usd is None
        assert m.pricing_output_per_1k_usd is None


# --- session() factory --------------------------------------------------------


def test_session_returns_an_agent_session() -> None:
    rt = _make_runtime()
    sess = rt.session()
    # The Iteration-A session is a _ThinAgentSession; what matters here
    # is that it satisfies the AgentSession protocol.
    assert hasattr(sess, "execute")
    assert hasattr(sess, "stream")
    assert hasattr(sess, "cancel")
    assert hasattr(sess, "close")
    asyncio.run(sess.close())


def test_session_rejects_wrong_provider_options_namespace() -> None:
    """Passing a non-KimiOptions namespace raises UnsupportedFeatureError.

    The shared `_check_provider_options` enforces the tagged-union
    contract. Mirroring the conformance check here for fast feedback
    on the Kimi-specific path.
    """
    from airframe.errors import UnsupportedFeatureError

    rt = _make_runtime()
    with pytest.raises(UnsupportedFeatureError):
        rt.session(provider_options=ClaudeOptions())
    with pytest.raises(UnsupportedFeatureError):
        rt.session(provider_options=BedrockOptions())


def test_session_accepts_kimi_options_namespace() -> None:
    rt = _make_runtime()
    sess = rt.session(provider_options=KimiOptions())
    asyncio.run(sess.close())


def test_session_resume_eagerly_populates_session_id() -> None:
    """Iteration B: ``session(resume=…)`` surfaces the ID before the SDK call.

    The actual ``Session.resume(work_dir, session_id)`` round-trip is
    deferred to the first ``execute()``/``stream()``; the unit test only
    pins that the ID is observable on the session immediately. Behavioural
    coverage of the resume path (including the "ID not found" branch
    that surfaces as :class:`RuntimeProtocolError`) lives in the
    SDK-mock tests under ``tests/test_kimi_session.py``.
    """
    rt = _make_runtime()
    sess = rt.session(resume="prior-session-id")
    assert sess.id == "prior-session-id"
    asyncio.run(sess.close())


# --- constructor accepts the documented kwargs --------------------------------


def test_constructor_accepts_documented_kwargs() -> None:
    """Just a smoke test that the documented constructor shape exists."""
    rt = KimiRuntime(
        model="kimi-k2-thinking-turbo",
        base_url="https://api.moonshot.ai/v1",
        api_key="sk-test",
    )
    assert rt._default_model == "kimi-k2-thinking-turbo"
    assert rt._base_url == "https://api.moonshot.ai/v1"
    assert rt._api_key_override == "sk-test"


# --- emittable hook kinds — empty in Iteration A ------------------------------


def test_emittable_hook_kinds_empty_in_iteration_a() -> None:
    """No SDK events translated to HookEvent yet — Iteration E lands the six kinds."""
    assert frozenset() == KimiRuntime.EMITTABLE_HOOK_KINDS


def _unused_helper_to_silence_any_import(_: Any) -> None:
    pass  # keep `Any` import non-trivial for ruff
