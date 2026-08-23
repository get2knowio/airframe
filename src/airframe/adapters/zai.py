"""``ZaiAnthropicRuntime`` — Z.AI's GLM models over their Anthropic-compatible endpoint.

Z.AI fronts the GLM family behind an endpoint that speaks Anthropic's
Messages wire format, which is what lets the ``claude`` CLI talk to it
at all. This adapter reuses :class:`~airframe.adapters.claude_code.ClaudeCodeRuntime`'s
entire harness — subprocess lifecycle, session semantics, streaming,
tools, hooks — and changes only the three things that make it a
different *binding*: the base URL, the auth variable, and the honest
feature surface.

**Why its own ``PROVIDER_ID``.** Per the binding rule in ``CLAUDE.md``,
an ID names a ``(wire protocol, endpoint, auth mechanism, feature
surface)`` tuple, not a vendor and not a harness. Sharing ``"claude"``
would make ``supports()`` — a ClassVar manifest — answer for two
endpoints with different real capabilities. The ID is
``"zai-anthropic"`` rather than a bare ``"zai"`` because Z.AI also
exposes an OpenAI-compatible surface; that one belongs to a future
``"zai-openai"`` sibling built on
:class:`~airframe.adapters.openai_compatible.OpenAICompatibleRuntime`.
Same reasoning that kept the Kimi Agent SDK adapter from claiming
``"moonshot"``.

**Auth.** ``ZAI_API_KEY`` (or an explicit ``api_key=``) becomes
``ANTHROPIC_AUTH_TOKEN`` in the subprocess environment. Anthropic
subscription credentials are actively shadowed — see
:meth:`ZaiAnthropicRuntime._subprocess_env`.

**Feature surface is deliberately understated.** Everything the
``claude`` CLI implements locally (streaming, session resume, tools,
hooks, permission callbacks, budget caps) is protocol-independent and
kept. Everything that depends on what the *endpoint* implements —
extended thinking, vision, token counting, rate-limit telemetry — is
reported ``False`` until verified against a live key. A ``supports()``
that overstates is worse than one that understates: a consumer checking
the predicate simply routes elsewhere, while one that overstates
produces a runtime failure the consumer was told couldn't happen. ``examples/probe_zai.py`` checks
each unverified feature; promote them in :data:`_UNVERIFIED_FEATURES`
as the probe confirms them.
"""

from __future__ import annotations

import logging
import os
from typing import ClassVar

from airframe.adapters.claude_code import (
    DEFAULT_MAX_TURNS,
    ClaudeCodeRuntime,
    _ModelMeta,
)
from airframe.errors import RuntimeAuthError, UnsupportedFeatureError
from airframe.features import Feature
from airframe.inputs import Prompt
from airframe.models import (
    CAPABILITY_STREAMING,
    CAPABILITY_TOOLS,
    ModelInfo,
)
from airframe.protocol import ProviderModel

logger = logging.getLogger(__name__)

#: Z.AI's Anthropic-compatible base URL. Override via ``ZAI_BASE_URL`` env.
DEFAULT_ZAI_BASE_URL = "https://api.z.ai/api/anthropic"

#: Default model when no binding is specified.
DEFAULT_ZAI_MODEL = "glm-4.6"

#: Anthropic credentials that must not follow a Z.AI base URL. The Agent
#: SDK merges :attr:`ClaudeAgentOptions.env` *over* ``os.environ`` and
#: offers no way to unset a key, so these are shadowed with empty strings.
_ANTHROPIC_CREDENTIAL_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
)

#: Features whose support depends on what Z.AI's endpoint implements
#: rather than on the local CLI. Reported ``False`` pending a live probe
#: (``examples/probe_zai.py``); move one out of this set once confirmed.
#:
#: ``STRUCTURED_OUTPUT_JSON_SCHEMA`` is deliberately *absent* despite
#: being equally unverified. The conformance suite declares it the floor
#: every airframe adapter must meet
#: (``test_supports_structured_output_json_schema_is_true``), so
#: declaring it ``False`` would not produce a cautious adapter — it would
#: produce a non-conforming one. If the probe shows Z.AI does not honour
#: the CLI's ``--json-schema`` flag, that is a blocker on shipping this
#: adapter at all, not a bit to flip off.
_UNVERIFIED_FEATURES = frozenset(
    {
        Feature.REASONING_EFFORT,
        Feature.REASONING_BUDGET_TOKENS,
        Feature.REASONING_OUTPUT,
        Feature.VISION_INPUT,
        Feature.FILE_INPUT,
        Feature.COUNT_TOKENS,
        Feature.RATE_LIMIT_TELEMETRY,
        Feature.REQUEST_METADATA,
    }
)

#: Z.AI's coding-plan catalog. Billed as a flat-fee subscription, so
#: per-token rates are ``None`` rather than invented — same choice
#: ``opencode-go`` makes for its subscription gateway.
_METADATA: dict[str, _ModelMeta] = {
    "glm-4.6": _ModelMeta(
        "GLM-4.6",
        context_window=200_000,
        capabilities=frozenset({CAPABILITY_TOOLS, CAPABILITY_STREAMING}),
    ),
    "glm-4.5-air": _ModelMeta(
        "GLM-4.5-Air",
        context_window=128_000,
        capabilities=frozenset({CAPABILITY_TOOLS, CAPABILITY_STREAMING}),
    ),
}


class ZaiAnthropicRuntime(ClaudeCodeRuntime):
    """Z.AI GLM models driven through the Claude Agent SDK harness.

    Args:
        model: Default GLM model id. Falls back to ``ZAI_MODEL_OVERRIDE``
            then :data:`DEFAULT_ZAI_MODEL`.
        max_turns: Hard cap on agent turns within one ``execute()``.
        api_key: Z.AI API key. When ``None``, resolves from
            ``ZAI_API_KEY``.
        base_url: Endpoint override. Falls back to ``ZAI_BASE_URL`` then
            :data:`DEFAULT_ZAI_BASE_URL`.
    """

    label = "zai_anthropic"

    PROVIDER_ID: ClassVar[str] = "zai-anthropic"
    REQUIRES_PACKAGE: ClassVar[str] = "claude_agent_sdk"
    EXTRA_NAME: ClassVar[str] = "claude"

    SUPPORTED_FEATURES: ClassVar[frozenset[Feature]] = (
        ClaudeCodeRuntime.SUPPORTED_FEATURES - _UNVERIFIED_FEATURES
    )

    def __init__(
        self,
        *,
        model: str | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(
            model=model or os.environ.get("ZAI_MODEL_OVERRIDE") or DEFAULT_ZAI_MODEL,
            max_turns=max_turns,
            api_key=api_key,
        )
        self._base_url = base_url or os.environ.get("ZAI_BASE_URL") or DEFAULT_ZAI_BASE_URL

    def _resolve_api_key(self) -> str:
        """Return the Z.AI key, or raise if none is configured.

        Resolution: explicit ``api_key=`` constructor arg → ``ZAI_API_KEY``
        env var. Deliberately does *not* fall back to any Anthropic
        credential — those authenticate a different account against a
        different vendor.

        Raises:
            RuntimeAuthError: No Z.AI credential found.
        """
        key = self._api_key_override or os.environ.get("ZAI_API_KEY")
        if not key:
            raise RuntimeAuthError(
                f"{self.label}: no Z.AI credential found. Set ZAI_API_KEY or pass "
                f"api_key= to {type(self).__name__}. Anthropic credentials are not "
                f"used as a fallback — they authenticate a different vendor."
            )
        return key

    def _subprocess_env(self) -> dict[str, str]:
        """Aim the ``claude`` CLI at Z.AI and shadow Anthropic credentials.

        The Agent SDK merges this over ``os.environ``, so an inherited
        ``CLAUDE_CODE_OAUTH_TOKEN`` (a shell profile, a CI secret) would
        otherwise ride along to Z.AI. Merging cannot remove a key, so each
        Anthropic credential is shadowed with an empty string before the
        Z.AI token is set — the same invariant
        :func:`~airframe.adapters.claude_code._is_anthropic_endpoint`
        enforces on airframe's own HTTP calls, applied at the subprocess
        boundary.

        Returns:
            Env overrides for the ``claude`` subprocess.
        """
        env = dict.fromkeys(_ANTHROPIC_CREDENTIAL_VARS, "")
        env["ANTHROPIC_BASE_URL"] = self._base_url
        env["ANTHROPIC_AUTH_TOKEN"] = self._resolve_api_key()
        return env

    async def list_models(self) -> list[ModelInfo]:
        """Return the static Z.AI catalog.

        Z.AI's Anthropic-compatible surface exposes chat completion, not
        Anthropic's ``/v1/models`` discovery endpoint, so there is nothing
        live to query — unlike :meth:`ClaudeCodeRuntime.list_models`, which
        wraps the real thing. The catalog comes from :data:`_METADATA`.

        Returns:
            One :class:`ModelInfo` per known GLM model.
        """
        return [
            ModelInfo(
                id=model_id,
                provider_id=self.PROVIDER_ID,
                display_name=meta.display_name,
                context_window=meta.context_window,
                pricing_input_per_1k_usd=meta.input_per_1k,
                pricing_output_per_1k_usd=meta.output_per_1k,
                capabilities=meta.capabilities,
            )
            for model_id, meta in _METADATA.items()
        ]

    async def count_tokens(
        self,
        prompt: Prompt,
        *,
        system: str | None = None,
        model: ProviderModel | None = None,
    ) -> int:
        """Not available — Z.AI exposes no token-counting endpoint.

        Raises:
            UnsupportedFeatureError: Always. Inheriting
                :meth:`ClaudeCodeRuntime.count_tokens` would POST to
                Anthropic's ``/v1/messages/count_tokens`` path on Z.AI's
                base URL, which is not a route Z.AI serves.
        """
        raise UnsupportedFeatureError(
            f"{self.label}: count_tokens() is not available — Z.AI's "
            f"Anthropic-compatible endpoint does not expose a token-counting "
            f"route. Estimate client-side or route counting through a provider "
            f"that supports it.",
            feature=Feature.COUNT_TOKENS,
        )


__all__ = [
    "DEFAULT_ZAI_BASE_URL",
    "DEFAULT_ZAI_MODEL",
    "ZaiAnthropicRuntime",
]
