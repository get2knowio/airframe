"""Unit tests for :class:`BedrockRuntime` — Iteration A scaffolding scope.

Iteration A lands discovery + capability predicates + the AWS auth
chain + ``list_models()`` against a mocked aioboto3 client. Behaviour-
bearing methods (``execute``, ``session``, ``stream``, ``cancel``)
intentionally raise :class:`NotImplementedError` until Iteration B
wires them; these tests pin that contract too.

Mocks ``aioboto3.Session`` at the boundary so no real AWS calls fire.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from airframe.adapters.bedrock import (
    DEFAULT_BEDROCK_MODEL,
    BedrockRuntime,
    _classify_bedrock_error,
)
from airframe.errors import (
    RuntimeAuthError,
    RuntimeProtocolError,
    RuntimeTransientError,
)
from airframe.features import Feature
from airframe.models import (
    CAPABILITY_STREAMING,
    CAPABILITY_STRUCTURED_OUTPUT,
    CAPABILITY_TOOLS,
    CAPABILITY_VISION,
)
from airframe.protocol import ProviderModel

# ---------------------------------------------------------------------------
# Identity + ClassVars
# ---------------------------------------------------------------------------


def test_provider_identity() -> None:
    """Canonical IDs match the plan's naming reservations."""
    assert BedrockRuntime.PROVIDER_ID == "bedrock"
    assert BedrockRuntime.REQUIRES_PACKAGE == "aioboto3"
    assert BedrockRuntime.EXTRA_NAME == "bedrock"
    assert BedrockRuntime.label == "bedrock"


def test_default_model_is_anthropic_haiku() -> None:
    """Default model is a current-generation Claude on Bedrock."""
    assert DEFAULT_BEDROCK_MODEL.startswith("anthropic.claude-")
    rt = BedrockRuntime()
    assert rt._default_model == DEFAULT_BEDROCK_MODEL


def test_env_var_overrides_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BEDROCK_DEFAULT_MODEL", "amazon.nova-pro-v1:0")
    rt = BedrockRuntime()
    assert rt._default_model == "amazon.nova-pro-v1:0"


def test_constructor_model_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BEDROCK_DEFAULT_MODEL", "amazon.nova-pro-v1:0")
    rt = BedrockRuntime(model="meta.llama3-1-70b-instruct-v1:0")
    assert rt._default_model == "meta.llama3-1-70b-instruct-v1:0"


# ---------------------------------------------------------------------------
# SUPPORTED_FEATURES — Iteration E adds hooks + budget caps
# ---------------------------------------------------------------------------


def test_supported_features_iteration_e_set() -> None:
    """Iteration E adds LIFECYCLE_HOOKS + BUDGET_USD_CAP + BUDGET_TURN_CAP.

    MCP transports + SESSION_RESUME + STRUCTURED_OUTPUT_STRICT stay
    False permanently — Bedrock Converse has no equivalent on the
    wire.
    """
    iteration_b_through_e = {
        Feature.STRUCTURED_OUTPUT_JSON_SCHEMA,
        Feature.STREAMING,
        Feature.CANCEL,
        Feature.REASONING_EFFORT,
        Feature.REASONING_BUDGET_TOKENS,
        Feature.VISION_INPUT,
        Feature.FILE_INPUT,
        Feature.TOOLS_FUNCTION,
        Feature.PERMISSION_CALLBACK,
        Feature.LIFECYCLE_HOOKS,
        Feature.BUDGET_USD_CAP,
        Feature.BUDGET_TURN_CAP,
    }
    for feature in iteration_b_through_e:
        assert feature in BedrockRuntime.SUPPORTED_FEATURES
    assert len(BedrockRuntime.SUPPORTED_FEATURES) == 12


def test_supports_iteration_e_flags_true() -> None:
    rt = BedrockRuntime()
    assert rt.supports(Feature.LIFECYCLE_HOOKS) is True
    assert rt.supports(Feature.BUDGET_USD_CAP) is True
    assert rt.supports(Feature.BUDGET_TURN_CAP) is True


def test_supports_returns_false_for_permanent_declines() -> None:
    """Three flags are *permanent* declines — pin them so future
    iterations don't silently flip them True."""
    rt = BedrockRuntime()
    permanent = {
        Feature.SESSION_RESUME,
        Feature.TOOLS_MCP_STDIO,
        Feature.TOOLS_MCP_HTTP,
        Feature.TOOLS_MCP_SSE,
        Feature.STRUCTURED_OUTPUT_STRICT,
    }
    for feature in permanent:
        assert rt.supports(feature) is False, f"unexpected True for {feature.name}"


def test_emittable_hook_kinds_six_kinds() -> None:
    """Six of the eight canonical kinds — see EMITTABLE_HOOK_KINDS docstring."""
    expected = {
        "session_start",
        "session_end",
        "user_prompt_submit",
        "pre_tool_use",
        "post_tool_use",
        "tool_failure",
    }
    assert set(BedrockRuntime.EMITTABLE_HOOK_KINDS) == expected
    # pre_compact / rate_limit are intentionally absent.
    assert "pre_compact" not in BedrockRuntime.EMITTABLE_HOOK_KINDS
    assert "rate_limit" not in BedrockRuntime.EMITTABLE_HOOK_KINDS


def test_supports_mcp_transports_permanent_decline() -> None:
    """Bedrock Converse has no MCP slot — these three stay False forever.

    Pinned with their own test because future iterations must NOT
    flip them True silently. Consumers wanting MCP-via-Bedrock should
    use ``unwrap(BedrockRuntimeClient)`` and hand-craft a shim.
    """
    rt = BedrockRuntime()
    assert rt.supports(Feature.TOOLS_MCP_STDIO) is False
    assert rt.supports(Feature.TOOLS_MCP_HTTP) is False
    assert rt.supports(Feature.TOOLS_MCP_SSE) is False


def test_supports_accepts_model_kwarg() -> None:
    """Per-model differentiation isn't wired yet but the kwarg is honoured."""
    rt = BedrockRuntime()
    binding = ProviderModel("bedrock", "anthropic.claude-3-5-haiku-20241022-v1:0")
    assert rt.supports(Feature.STREAMING, model=binding) is True
    assert rt.supports(Feature.SESSION_RESUME, model=binding) is False


# ---------------------------------------------------------------------------
# validate_binding
# ---------------------------------------------------------------------------


def test_validate_binding_accepts_bedrock_with_non_empty_model_id() -> None:
    rt = BedrockRuntime()
    assert rt.validate_binding(
        ProviderModel("bedrock", "anthropic.claude-3-5-haiku-20241022-v1:0")
    )


def test_validate_binding_accepts_inference_profile_ids() -> None:
    """Inference profiles (cross-region routing) carry a region prefix."""
    rt = BedrockRuntime()
    assert rt.validate_binding(
        ProviderModel("bedrock", "us.anthropic.claude-3-5-sonnet-20241022-v2:0")
    )
    assert rt.validate_binding(ProviderModel("bedrock", "eu.meta.llama3-1-70b-instruct-v1:0"))


def test_validate_binding_accepts_provisioned_throughput_arns() -> None:
    """PT ARNs are valid model identifiers in Converse."""
    rt = BedrockRuntime()
    arn = "arn:aws:bedrock:us-east-1:123456789012:provisioned-model/abc123def456"
    assert rt.validate_binding(ProviderModel("bedrock", arn))


def test_validate_binding_rejects_empty_model_id() -> None:
    rt = BedrockRuntime()
    assert not rt.validate_binding(ProviderModel("bedrock", ""))


def test_validate_binding_rejects_other_providers() -> None:
    rt = BedrockRuntime()
    assert not rt.validate_binding(ProviderModel("anthropic", "claude-3-5-sonnet"))
    assert not rt.validate_binding(ProviderModel("openai", "gpt-4o"))
    assert not rt.validate_binding(ProviderModel("openrouter", "anthropic/claude-3.5-sonnet"))
    assert not rt.validate_binding(ProviderModel("claude", "claude-haiku-4-5"))


def test_validate_binding_returns_bool_not_truthy() -> None:
    rt = BedrockRuntime()
    own = rt.validate_binding(ProviderModel("bedrock", "amazon.nova-lite-v1:0"))
    foreign = rt.validate_binding(ProviderModel("nope", "x"))
    assert isinstance(own, bool)
    assert isinstance(foreign, bool)


# ---------------------------------------------------------------------------
# unwrap()
# ---------------------------------------------------------------------------


def test_unwrap_self_returns_runtime() -> None:
    rt = BedrockRuntime()
    assert rt.unwrap(BedrockRuntime) is rt


def test_unwrap_unrelated_type_raises_typeerror() -> None:
    class _Unrelated:
        pass

    rt = BedrockRuntime()
    with pytest.raises(TypeError):
        rt.unwrap(_Unrelated)


# ---------------------------------------------------------------------------
# close() + reset() lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_is_idempotent() -> None:
    rt = BedrockRuntime()
    await rt.close()
    await rt.close()
    await rt.close()


@pytest.mark.asyncio
async def test_close_on_fresh_runtime_does_not_raise() -> None:
    rt = BedrockRuntime()
    await rt.close()


@pytest.mark.asyncio
async def test_reset_is_noop() -> None:
    rt = BedrockRuntime()
    await rt.reset()
    await rt.reset()
    # reset() returns None and never raises.


# ---------------------------------------------------------------------------
# session() declines for later-iteration kwargs
# ---------------------------------------------------------------------------


def test_session_with_no_kwargs_returns_bedrock_session() -> None:
    from airframe.adapters.bedrock import BedrockSession

    rt = BedrockRuntime()
    sess = rt.session()
    assert isinstance(sess, BedrockSession)
    assert sess.id is None


def test_session_resume_raises_unsupported_feature() -> None:
    from airframe.errors import UnsupportedFeatureError

    rt = BedrockRuntime()
    with pytest.raises(UnsupportedFeatureError):
        rt.session(resume="some-session-id")


def test_session_with_tools_returns_bedrock_session() -> None:
    """Iteration D — tools= is wired; the session takes the list and
    translates each entry to a Converse toolSpec on each call."""
    from pydantic import BaseModel

    from airframe.adapters.bedrock import BedrockSession
    from airframe.tools import FunctionTool

    class _P(BaseModel):
        x: int

    async def _h(p: BaseModel) -> int:
        return 1

    rt = BedrockRuntime()
    tool = FunctionTool(name="t", description="d", params=_P, handler=_h)
    sess = rt.session(tools=[tool])
    assert isinstance(sess, BedrockSession)


def test_session_with_mcp_servers_raises_permanent_decline() -> None:
    """MCP transports stay False permanently — opt-in to a permanent
    decline, with a message pointing users at the unwrap escape hatch."""
    from airframe.errors import UnsupportedFeatureError
    from airframe.tools import McpServerRef

    rt = BedrockRuntime()
    ref = McpServerRef(name="probe", transport="stdio", command=["echo"])
    with pytest.raises(UnsupportedFeatureError) as exc:
        rt.session(mcp_servers=[ref])
    assert "unwrap" in str(exc.value).lower() or "permanent" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# _resolve_aws_credentials — the 4-step auth chain
# ---------------------------------------------------------------------------


def test_resolve_credentials_explicit_keys_win(monkeypatch: pytest.MonkeyPatch) -> None:
    """Step 1 — explicit constructor args take precedence over env."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-secret")
    monkeypatch.setenv("AWS_PROFILE", "env-profile")
    rt = BedrockRuntime(
        aws_access_key_id="explicit-key",
        aws_secret_access_key="explicit-secret",
        aws_session_token="explicit-token",
    )
    kwargs = rt._resolve_aws_credentials()
    assert kwargs == {
        "aws_access_key_id": "explicit-key",
        "aws_secret_access_key": "explicit-secret",
        "aws_session_token": "explicit-token",
    }
    # Profile name is NOT set — explicit keys override the env profile.


def test_resolve_credentials_explicit_profile_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Step 1.5 — explicit profile_name without keys."""
    monkeypatch.setenv("AWS_PROFILE", "env-profile")
    rt = BedrockRuntime(profile_name="explicit-profile")
    kwargs = rt._resolve_aws_credentials()
    assert kwargs == {"profile_name": "explicit-profile"}


def test_resolve_credentials_env_vars_let_boto_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Step 2 — env vars are read by boto3 itself, not us.

    We return empty kwargs so :class:`aioboto3.Session` walks its
    native credential resolution. Reading env vars ourselves would
    duplicate boto3's logic and break the IAM-instance-role fallback.
    """
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-secret")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    rt = BedrockRuntime()
    assert rt._resolve_aws_credentials() == {}


def test_resolve_credentials_default_chain_returns_empty_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Step 4 — nothing set anywhere; boto3 walks the IAM-role chain.

    Empty kwargs are the contract: that's what tells aioboto3 to use
    its default credential provider (instance profile / ECS task role /
    Lambda role / IRSA).
    """
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    rt = BedrockRuntime()
    assert rt._resolve_aws_credentials() == {}


# ---------------------------------------------------------------------------
# _resolve_region
# ---------------------------------------------------------------------------


def test_resolve_region_explicit_arg_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_REGION", "env-region")
    rt = BedrockRuntime(region_name="us-west-2")
    assert rt._resolve_region() == "us-west-2"


def test_resolve_region_aws_region_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    rt = BedrockRuntime()
    assert rt._resolve_region() == "us-east-1"


def test_resolve_region_aws_default_region_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``AWS_DEFAULT_REGION`` is the legacy env var name; honour it too."""
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    rt = BedrockRuntime()
    assert rt._resolve_region() == "eu-west-1"


def test_resolve_region_aws_region_beats_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    rt = BedrockRuntime()
    assert rt._resolve_region() == "us-east-1"


def test_resolve_region_returns_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    rt = BedrockRuntime()
    assert rt._resolve_region() is None


# ---------------------------------------------------------------------------
# list_models — mocked aioboto3
# ---------------------------------------------------------------------------


def _make_bedrock_client(
    summaries: list[dict[str, Any]] | None = None,
    *,
    raise_on_call: Exception | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Build a stand-in for ``aioboto3.Session().client("bedrock")``.

    Returns ``(session_factory, client)`` — the factory replaces
    ``aioboto3.Session`` so the test can assert how the runtime
    constructs the session, and the client carries the stub
    ``list_foundation_models`` coroutine.
    """
    client = MagicMock()
    if raise_on_call is not None:
        client.list_foundation_models = AsyncMock(side_effect=raise_on_call)
    else:
        client.list_foundation_models = AsyncMock(return_value={"modelSummaries": summaries or []})
    # `async with session.client(...) as c:` requires the returned object
    # to implement the async context-manager protocol.
    client_cm = MagicMock()
    client_cm.__aenter__ = AsyncMock(return_value=client)
    client_cm.__aexit__ = AsyncMock(return_value=None)

    session = MagicMock()
    session.client = MagicMock(return_value=client_cm)

    factory = MagicMock(return_value=session)
    return factory, client


@pytest.fixture
def aioboto3_session(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Default ``aioboto3.Session`` mock returning a small catalog."""
    import aioboto3

    factory, client = _make_bedrock_client(
        summaries=[
            {
                "modelId": "anthropic.claude-3-5-haiku-20241022-v1:0",
                "modelName": "Claude 3.5 Haiku",
                "providerName": "Anthropic",
                "inputModalities": ["TEXT", "IMAGE"],
                "outputModalities": ["TEXT"],
                "responseStreamingSupported": True,
            },
            {
                "modelId": "amazon.nova-pro-v1:0",
                "modelName": "Nova Pro",
                "providerName": "Amazon",
                "inputModalities": ["TEXT", "IMAGE"],
                "outputModalities": ["TEXT"],
                "responseStreamingSupported": True,
            },
            {
                "modelId": "some.unknown-model-v1:0",
                "modelName": "Unknown Newcomer",
                "providerName": "SomeVendor",
                "inputModalities": ["TEXT"],
                "outputModalities": ["TEXT"],
                "responseStreamingSupported": False,
            },
        ],
    )
    monkeypatch.setattr(aioboto3, "Session", factory)
    return factory


@pytest.mark.asyncio
async def test_list_models_returns_enriched_entries(
    monkeypatch: pytest.MonkeyPatch,
    aioboto3_session: MagicMock,
) -> None:
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    rt = BedrockRuntime()
    models = await rt.list_models()
    by_id = {m.id: m for m in models}

    haiku = by_id["anthropic.claude-3-5-haiku-20241022-v1:0"]
    assert haiku.display_name == "Claude 3.5 Haiku (via Bedrock)"
    assert haiku.context_window == 200_000
    assert haiku.provider_id == "bedrock"
    # Vision + streaming + tools + structured-output capabilities inferred.
    assert CAPABILITY_VISION in haiku.capabilities
    assert CAPABILITY_STREAMING in haiku.capabilities
    assert CAPABILITY_TOOLS in haiku.capabilities
    assert CAPABILITY_STRUCTURED_OUTPUT in haiku.capabilities

    # Unknown model still surfaces, with fallback display name from
    # ``modelName`` and no context-window enrichment.
    unknown = by_id["some.unknown-model-v1:0"]
    assert unknown.display_name == "Unknown Newcomer"
    assert unknown.context_window is None
    # No vision modality in the input list → no VISION capability.
    assert CAPABILITY_VISION not in unknown.capabilities


@pytest.mark.asyncio
async def test_list_models_filters_by_text_output_modality(
    monkeypatch: pytest.MonkeyPatch,
    aioboto3_session: MagicMock,
) -> None:
    """The ``byOutputModality="TEXT"`` filter is sent to the API.

    Embedding-only models shouldn't appear in a chat-style menu;
    server-side filtering is cheaper than client-side dropping.
    """
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    rt = BedrockRuntime()
    await rt.list_models()
    # aioboto3_session() returned a session whose client(...) returned
    # a context-manager whose __aenter__ returned `client`.
    session = aioboto3_session.return_value
    client_cm = session.client.return_value
    client = client_cm.__aenter__.return_value
    client.list_foundation_models.assert_awaited_once_with(byOutputModality="TEXT")


@pytest.mark.asyncio
async def test_list_models_routes_to_resolved_region(
    monkeypatch: pytest.MonkeyPatch,
    aioboto3_session: MagicMock,
) -> None:
    """The region resolved by ``_resolve_region`` is what aioboto3 sees."""
    monkeypatch.delenv("AWS_REGION", raising=False)
    rt = BedrockRuntime(region_name="ap-southeast-2")
    await rt.list_models()
    session = aioboto3_session.return_value
    session.client.assert_called_once_with("bedrock", region_name="ap-southeast-2")


@pytest.mark.asyncio
async def test_list_models_raises_auth_error_when_no_region(
    monkeypatch: pytest.MonkeyPatch,
    aioboto3_session: MagicMock,
) -> None:
    """Missing region surfaces as :class:`RuntimeAuthError` — not a fallthrough.

    Silent fallback to ``us-east-1`` would route to a different
    catalog than the user expects. Fail loud at the boundary.
    """
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    rt = BedrockRuntime()
    with pytest.raises(RuntimeAuthError) as exc:
        await rt.list_models()
    assert "AWS_REGION" in str(exc.value)


@pytest.mark.asyncio
async def test_list_models_skips_summaries_without_model_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defensive: malformed summary entries don't crash the listing."""
    import aioboto3

    factory, _client = _make_bedrock_client(
        summaries=[
            {"modelName": "no-id-field"},
            {"modelId": "", "modelName": "empty-id"},
            {"modelId": "ok.model-v1:0", "modelName": "OK"},
        ]
    )
    monkeypatch.setattr(aioboto3, "Session", factory)
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    rt = BedrockRuntime()
    models = await rt.list_models()
    assert [m.id for m in models] == ["ok.model-v1:0"]


# ---------------------------------------------------------------------------
# _classify_bedrock_error — boundary translation
# ---------------------------------------------------------------------------


def test_classify_no_credentials_error_maps_to_auth() -> None:
    from botocore.exceptions import NoCredentialsError

    classified = _classify_bedrock_error(NoCredentialsError())
    assert isinstance(classified, RuntimeAuthError)


def test_classify_no_region_error_maps_to_auth() -> None:
    from botocore.exceptions import NoRegionError

    classified = _classify_bedrock_error(NoRegionError())
    assert isinstance(classified, RuntimeAuthError)
    assert "AWS_REGION" in str(classified)


def test_classify_client_error_access_denied_maps_to_auth() -> None:
    from botocore.exceptions import ClientError

    exc = ClientError(
        error_response={
            "Error": {"Code": "AccessDeniedException", "Message": "denied"},
            "ResponseMetadata": {"HTTPStatusCode": 403},
        },
        operation_name="ListFoundationModels",
    )
    classified = _classify_bedrock_error(exc)
    assert isinstance(classified, RuntimeAuthError)
    assert classified.status == 403


def test_classify_client_error_throttling_maps_to_transient() -> None:
    from botocore.exceptions import ClientError

    exc = ClientError(
        error_response={
            "Error": {"Code": "ThrottlingException", "Message": "slow down"},
            "ResponseMetadata": {"HTTPStatusCode": 429},
        },
        operation_name="ListFoundationModels",
    )
    classified = _classify_bedrock_error(exc)
    assert isinstance(classified, RuntimeTransientError)


def test_classify_client_error_5xx_maps_to_transient() -> None:
    from botocore.exceptions import ClientError

    exc = ClientError(
        error_response={
            "Error": {"Code": "InternalServerException", "Message": "oops"},
            "ResponseMetadata": {"HTTPStatusCode": 502},
        },
        operation_name="ListFoundationModels",
    )
    classified = _classify_bedrock_error(exc)
    assert isinstance(classified, RuntimeTransientError)


def test_classify_unknown_client_error_maps_to_protocol() -> None:
    from botocore.exceptions import ClientError

    exc = ClientError(
        error_response={
            "Error": {"Code": "WeirdNewThing", "Message": "?"},
            "ResponseMetadata": {"HTTPStatusCode": 418},
        },
        operation_name="ListFoundationModels",
    )
    classified = _classify_bedrock_error(exc)
    assert isinstance(classified, RuntimeProtocolError)


def test_classify_endpoint_connection_error_maps_to_transient() -> None:
    from botocore.exceptions import EndpointConnectionError

    exc = EndpointConnectionError(endpoint_url="https://bedrock.us-east-1.amazonaws.com/")
    classified = _classify_bedrock_error(exc)
    assert isinstance(classified, RuntimeTransientError)


def test_classify_passes_through_unrelated_exception() -> None:
    sentinel = ValueError("totally unrelated")
    assert _classify_bedrock_error(sentinel) is sentinel


# ---------------------------------------------------------------------------
# list_models with classified errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_models_propagates_classified_auth_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aioboto3
    from botocore.exceptions import ClientError

    err = ClientError(
        error_response={
            "Error": {"Code": "UnrecognizedClientException", "Message": "bad key"},
            "ResponseMetadata": {"HTTPStatusCode": 401},
        },
        operation_name="ListFoundationModels",
    )
    factory, _client = _make_bedrock_client(raise_on_call=err)
    monkeypatch.setattr(aioboto3, "Session", factory)
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    rt = BedrockRuntime()
    with pytest.raises(RuntimeAuthError):
        await rt.list_models()


# ---------------------------------------------------------------------------
# Discovery integration
# ---------------------------------------------------------------------------


def test_runtime_for_resolves_bedrock() -> None:
    from airframe import list_providers, runtime_for

    assert "bedrock" in list_providers(installed_only=False)
    assert runtime_for("bedrock") is BedrockRuntime


def test_top_level_export_is_present() -> None:
    """``from airframe import BedrockRuntime`` works without touching aioboto3."""
    import airframe

    assert airframe.BedrockRuntime is BedrockRuntime
    assert "BedrockRuntime" in airframe.__all__
