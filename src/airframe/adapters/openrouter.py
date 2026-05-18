"""``OpenRouterRuntime`` — OpenAI-compatible adapter for the OpenRouter gateway.

Thin subclass of :class:`OpenAICompatibleRuntime` configured for
``https://openrouter.ai/api/v1`` — OpenRouter's OpenAI Chat
Completions-compatible router, which fronts 200+ models from
Anthropic, OpenAI, Google, Meta, Mistral, DeepSeek, and friends
behind one HTTP endpoint and one billing relationship.

**Per-model feature heterogeneity.** Because OpenRouter is a router
rather than a single vendor, what works in any given call depends
on the underlying model the request gets routed to. Function tools,
strict-mode JSON schema, vision inputs — all wire-compatible but
not uniformly supported across the catalog. Treat
:meth:`AgentRuntime.supports` as the *adapter's* declared surface;
silent degradation on a specific model is the failure mode rather
than a hard refusal. Document per-model compliance for any model
your application hard-depends on.

**Model identifiers carry vendor prefixes** — OpenRouter routes by
``<vendor>/<model>`` strings (e.g. ``"anthropic/claude-3.5-sonnet"``,
``"openai/gpt-4o-mini"``, ``"google/gemini-pro-1.5"``,
``"meta-llama/llama-3.1-70b-instruct"``). The adapter passes the
string through unchanged.

**Auth.** The API key resolves in this order:

1. Explicit ``api_key=`` constructor argument.
2. ``OPENROUTER_API_KEY`` env var.

OpenRouter has no on-disk auth-file convention the way opencode
does — there's no equivalent to ``~/.local/share/opencode/auth.json``.
If neither source is set, the first call raises
:class:`RuntimeAuthError`.
"""

from __future__ import annotations

import logging
import os
from typing import ClassVar

from airframe.adapters.openai_compatible import ModelMeta, OpenAICompatibleRuntime
from airframe.errors import RuntimeAuthError

logger = logging.getLogger(__name__)


#: Default OpenRouter base URL. Override via ``OPENROUTER_BASE_URL`` env.
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#: Default model. ``openai/gpt-4o-mini`` is cheap, broadly available,
#: and one of the most reliable structured-output backends in the
#: OpenRouter catalog. Override via :class:`ProviderModel` or
#: ``OPENROUTER_DEFAULT_MODEL``.
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"


# Curated subset with stable, broadly-used routes. OpenRouter's
# full catalog is dynamic and per-model pricing shifts as upstream
# vendors adjust; unknown IDs return ``cost_usd=None`` from
# ``list_models()`` rather than guessed values. To enrich a new
# model id, add a ``ModelMeta`` entry with current rates from
# https://openrouter.ai/models.
#
# Rates below are per-1k tokens (OpenRouter's site lists per-million
# rates — divide by 1000). Context windows reflect what the upstream
# vendor exposes through the router.
_METADATA: dict[str, ModelMeta] = {
    "openai/gpt-4o-mini": ModelMeta("GPT-4o Mini (via OpenRouter)", 128_000, 0.00015, 0.0006),
    "openai/gpt-4o": ModelMeta("GPT-4o (via OpenRouter)", 128_000, 0.0025, 0.01),
    "anthropic/claude-3.5-haiku": ModelMeta(
        "Claude 3.5 Haiku (via OpenRouter)", 200_000, 0.0008, 0.004
    ),
    "anthropic/claude-3.5-sonnet": ModelMeta(
        "Claude 3.5 Sonnet (via OpenRouter)", 200_000, 0.003, 0.015
    ),
    "google/gemini-pro-1.5": ModelMeta(
        "Gemini 1.5 Pro (via OpenRouter)", 2_000_000, 0.00125, 0.005
    ),
    "google/gemini-flash-1.5": ModelMeta(
        "Gemini 1.5 Flash (via OpenRouter)", 1_000_000, 0.000075, 0.0003
    ),
    "meta-llama/llama-3.1-70b-instruct": ModelMeta(
        "Llama 3.1 70B Instruct (via OpenRouter)", 131_072, 0.00059, 0.00079
    ),
    "meta-llama/llama-3.1-8b-instruct": ModelMeta(
        "Llama 3.1 8B Instruct (via OpenRouter)", 131_072, 0.00002, 0.00005
    ),
    "deepseek/deepseek-chat": ModelMeta(
        "DeepSeek Chat (via OpenRouter)", 64_000, 0.00014, 0.00028
    ),
    "mistralai/mistral-large": ModelMeta("Mistral Large (via OpenRouter)", 128_000, 0.002, 0.006),
}


class OpenRouterRuntime(OpenAICompatibleRuntime):
    """OpenAI-compatible adapter for the OpenRouter model gateway."""

    label = "openrouter"

    PROVIDER_ID: ClassVar[str] = "openrouter"
    DEFAULT_BASE_URL: ClassVar[str] = DEFAULT_OPENROUTER_BASE_URL
    DEFAULT_MODEL: ClassVar[str] = DEFAULT_OPENROUTER_MODEL
    METADATA: ClassVar[dict[str, ModelMeta]] = _METADATA

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 600.0,
    ) -> None:
        super().__init__(
            model=model or os.environ.get("OPENROUTER_DEFAULT_MODEL"),
            base_url=base_url or os.environ.get("OPENROUTER_BASE_URL"),
            api_key=api_key,
            timeout=timeout,
        )

    def _resolve_api_key(self, api_key: str | None) -> str:
        """Resolve the OpenRouter API key: explicit → env."""
        if api_key:
            return api_key
        env = os.environ.get("OPENROUTER_API_KEY")
        if env:
            return env
        raise RuntimeAuthError(
            "OpenRouterRuntime: no API key found. Set OPENROUTER_API_KEY, "
            "pass api_key= explicitly, or mint one at https://openrouter.ai/keys."
        )


__all__ = [
    "DEFAULT_OPENROUTER_BASE_URL",
    "DEFAULT_OPENROUTER_MODEL",
    "OpenRouterRuntime",
]
