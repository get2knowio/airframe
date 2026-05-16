"""``OpenCodeZenRuntime`` — OpenAI-compatible adapter for the opencode-go Zen gateway.

Thin subclass of :class:`OpenAICompatibleRuntime` configured for
``https://opencode.ai/zen/v1`` — the OpenAI-compatible HTTP endpoint
that fronts the user's opencode-go subscription.

**Auth.** The API key resolves in this order:

1. Explicit ``api_key=`` constructor argument.
2. ``OPENCODE_API_KEY`` env var.
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


#: Default Zen gateway base URL. Override via ``OPENCODE_ZEN_BASE_URL`` env.
DEFAULT_ZEN_BASE_URL = "https://opencode.ai/zen/v1"

#: Default model when no binding is specified.
DEFAULT_ZEN_MODEL = "gpt-5-nano"

#: Path to the opencode auth file when present.
DEFAULT_AUTH_PATH = Path.home() / ".local" / "share" / "opencode" / "auth.json"


_METADATA: dict[str, ModelMeta] = {
    # Zen "free" tier
    "minimax-m2.5-free": ModelMeta("MiniMax M2.5 (free)", 200_000, 0.0, 0.0),
    "deepseek-v4-flash-free": ModelMeta("DeepSeek V4 Flash (free)", 64_000, 0.0, 0.0),
    "qwen3.6-plus-free": ModelMeta("Qwen 3.6 Plus (free)", 128_000, 0.0, 0.0),
    "nemotron-3-super-free": ModelMeta("Nemotron 3 Super (free)", 128_000, 0.0, 0.0),
    # Zen paid tier — placeholder rates pending real pricing table.
    "gpt-5-nano": ModelMeta("GPT-5 Nano", 128_000, 0.0001, 0.0002),
    "gpt-5-mini": ModelMeta("GPT-5 Mini", 256_000, 0.0003, 0.0006),
    "big-pickle": ModelMeta("Big Pickle", 200_000, 0.0005, 0.0015),
    "glm-5": ModelMeta("GLM-5", 128_000, 0.0002, 0.0004),
    "qwen3.6-plus": ModelMeta("Qwen 3.6 Plus", 128_000, 0.0003, 0.0009),
}


class OpenCodeZenRuntime(OpenAICompatibleRuntime):
    """OpenAI-compatible adapter targeted at opencode-go's Zen gateway."""

    label = "opencode_zen"

    PROVIDER_ID: ClassVar[str] = "opencode"
    DEFAULT_BASE_URL: ClassVar[str] = DEFAULT_ZEN_BASE_URL
    DEFAULT_MODEL: ClassVar[str] = DEFAULT_ZEN_MODEL
    METADATA: ClassVar[dict[str, ModelMeta]] = _METADATA

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 600.0,
    ) -> None:
        # Honour the legacy OPENCODE_ZEN_BASE_URL / OPENCODE_ZEN_DEFAULT_MODEL
        # env vars for backward compat with existing scripts.
        super().__init__(
            model=model or os.environ.get("OPENCODE_ZEN_DEFAULT_MODEL"),
            base_url=base_url or os.environ.get("OPENCODE_ZEN_BASE_URL"),
            api_key=api_key,
            timeout=timeout,
        )

    def _resolve_api_key(self, api_key: str | None) -> str:
        """Resolve the Zen API key from explicit arg → env → opencode auth.json."""
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
                    "opencode_zen_runtime.auth_file_unreadable path=%s error=%s",
                    auth_path,
                    exc,
                )
        raise RuntimeAuthError(
            "OpenCodeZenRuntime: no API key found. Set OPENCODE_API_KEY, "
            "pass api_key= explicitly, or run `opencode auth login opencode-go`."
        )


__all__ = [
    "DEFAULT_ZEN_BASE_URL",
    "DEFAULT_ZEN_MODEL",
    "OpenCodeZenRuntime",
]
