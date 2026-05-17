"""``OpenCodeGoRuntime`` — OpenAI-compatible adapter for the opencode-go subscription.

Thin subclass of :class:`OpenAICompatibleRuntime` configured for
``https://opencode.ai/zen/go/v1`` — the subscription endpoint that
fronts the user's flat-fee opencode-go plan. Distinct from
:class:`OpenCodeZenRuntime` (``https://opencode.ai/zen/v1``), which
is per-token billed; the two endpoints serve different model
catalogs and bill differently even though they share an auth scheme.

The 14 models listed below come bundled with the subscription. At the
caller's margin, every token is $0 (you've already paid the monthly
fee), so :class:`~airframe.cost.CostRecord.cost_usd` reports ``0.0``
on every turn — token counts are still populated for budget tracking.
Per-token Zen models live on the sibling adapter.

**Auth.** The API key resolves in this order:

1. Explicit ``api_key=`` constructor argument.
2. ``OPENCODE_API_KEY`` env var (same env var as Zen — opencode-go
   and Zen share auth).
3. ``~/.local/share/opencode/auth.json::opencode-go.key`` — the key
   minted by ``opencode auth login opencode-go``.

If none is available, the first call raises :class:`RuntimeAuthError`.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import ClassVar

from airframe.adapters.openai_compatible import ModelMeta, OpenAICompatibleRuntime
from airframe.errors import RuntimeAuthError

logger = logging.getLogger(__name__)


#: Subscription gateway base URL. Override via ``OPENCODE_GO_BASE_URL`` env.
DEFAULT_GO_BASE_URL = "https://opencode.ai/zen/go/v1"

#: Default model when no binding is specified. ``kimi-k2.6`` is the
#: most reliable structured-output choice in the subscription catalog;
#: several other bundled models either reject ``response_format`` or
#: silently return non-JSON, so we lead with one that round-trips
#: cleanly. Override via :class:`ProviderModel` or
#: ``OPENCODE_GO_DEFAULT_MODEL``.
DEFAULT_GO_MODEL = "kimi-k2.6"

#: Path to the opencode auth file when present.
DEFAULT_AUTH_PATH = Path.home() / ".local" / "share" / "opencode" / "auth.json"


# Subscription models have flat-fee billing — per-call cost is $0 at
# the caller's margin. We still record tokens / context windows so
# budget caps and capability checks work.
_METADATA: dict[str, ModelMeta] = {
    "deepseek-v4-flash": ModelMeta("DeepSeek V4 Flash", 1_000_000, 0.0, 0.0),
    "deepseek-v4-pro": ModelMeta("DeepSeek V4 Pro", 1_000_000, 0.0, 0.0),
    "glm-5": ModelMeta("GLM-5", 202_752, 0.0, 0.0),
    "glm-5.1": ModelMeta("GLM-5.1", 202_752, 0.0, 0.0),
    "kimi-k2.5": ModelMeta("Kimi K2.5", 262_144, 0.0, 0.0),
    "kimi-k2.6": ModelMeta("Kimi K2.6", 262_144, 0.0, 0.0),
    "mimo-v2-omni": ModelMeta("MiMo V2 Omni", 262_144, 0.0, 0.0),
    "mimo-v2-pro": ModelMeta("MiMo V2 Pro", 1_048_576, 0.0, 0.0),
    "mimo-v2.5": ModelMeta("MiMo V2.5", 1_000_000, 0.0, 0.0),
    "mimo-v2.5-pro": ModelMeta("MiMo V2.5 Pro", 1_048_576, 0.0, 0.0),
    "minimax-m2.5": ModelMeta("MiniMax M2.5", 204_800, 0.0, 0.0),
    "minimax-m2.7": ModelMeta("MiniMax M2.7", 204_800, 0.0, 0.0),
    "qwen3.5-plus": ModelMeta("Qwen3.5 Plus", 262_144, 0.0, 0.0),
    "qwen3.6-plus": ModelMeta("Qwen3.6 Plus", 262_144, 0.0, 0.0),
}


class OpenCodeGoRuntime(OpenAICompatibleRuntime):
    """OpenAI-compatible adapter for the opencode-go subscription gateway."""

    label = "opencode_go"

    PROVIDER_ID: ClassVar[str] = "opencode-go"
    DEFAULT_BASE_URL: ClassVar[str] = DEFAULT_GO_BASE_URL
    DEFAULT_MODEL: ClassVar[str] = DEFAULT_GO_MODEL
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
            model=model or os.environ.get("OPENCODE_GO_DEFAULT_MODEL"),
            base_url=base_url or os.environ.get("OPENCODE_GO_BASE_URL"),
            api_key=api_key,
            timeout=timeout,
        )

    def _resolve_api_key(self, api_key: str | None) -> str:
        """Resolve the opencode-go API key: explicit → env → auth.json."""
        if api_key:
            return api_key
        env = os.environ.get("OPENCODE_API_KEY")
        if env:
            return env
        auth_path = Path(os.environ.get("OPENCODE_AUTH_PATH") or DEFAULT_AUTH_PATH)
        if auth_path.exists():
            try:
                data = json.loads(auth_path.read_text())
                key = (data.get("opencode-go") or {}).get("key")
                if isinstance(key, str) and key:
                    return key
            except (OSError, json.JSONDecodeError) as exc:
                logger.debug(
                    "opencode_go_runtime.auth_file_unreadable path=%s error=%s",
                    auth_path,
                    exc,
                )
        raise RuntimeAuthError(
            "OpenCodeGoRuntime: no API key found. Set OPENCODE_API_KEY, "
            "pass api_key= explicitly, or run `opencode auth login opencode-go`."
        )


__all__ = [
    "DEFAULT_GO_BASE_URL",
    "DEFAULT_GO_MODEL",
    "OpenCodeGoRuntime",
]
