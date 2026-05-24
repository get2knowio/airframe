"""Cross-vendor prompt-cache controls.

Most vendors expose some flavour of "explicit cache key" — a stable
identifier the consumer hands to the API so subsequent calls with the
same key reuse the cached prompt prefix and skip re-tokenisation /
re-embedding. OpenAI exposes ``prompt_cache_key`` +
``prompt_cache_retention``; Anthropic exposes ``cache_control`` markers
on individual content blocks; some compat vendors honour the OpenAI
shape. The vendor's heuristic typically lands a 30-50% hit rate on
long-running agentic workflows; an explicit key from the consumer
("this same system prompt + codebase context across every turn of
this session") routinely hits 90%+.

:class:`CacheConfig` is the portable surface — pass it once at
``runtime.session(cache=...)`` and let the adapter translate to its
vendor's native channel.

**Contract — soft, like** :class:`~airframe.metadata.RequestMetadata`.
Passing ``cache=`` to an adapter that doesn't declare
:data:`~airframe.features.Feature.PROMPT_CACHE_CONTROL` silently drops
the kwarg rather than raising. The call still succeeds correctly —
just without the speed-up / cost reduction the cache would have
provided. Consumers who care branch on
:meth:`AgentRuntime.supports` first; consumers who just want
opportunistic caching pass ``cache=`` and forget.

Per-adapter mapping (today):

* **OpenAI-compatible** — ``key`` → ``prompt_cache_key=``;
  ``retention="short"`` → ``"in_memory"`` (~5min);
  ``retention="long"`` → ``"24h"``.
* **Claude Agent SDK** — silently dropped; the agent SDK manages
  caching via session warmth and doesn't expose a key channel.
* **Bedrock / Copilot / Kimi / OpenCode** — silently dropped.

The ``retention`` literal is deliberately coarse (``"short"`` /
``"long"``) because vendor windows differ. Consumers wanting precise
control should reach the vendor's native field through
:class:`~airframe.options.OpenAICompatOptions.prompt_cache_retention`
via ``provider_options=``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class CacheConfig:
    """Portable prompt-cache configuration.

    Attributes:
        key: Stable identifier the vendor uses to look up a cached
            prompt prefix. Reuse the same key across calls in the same
            "scope" (session, task, user) to maximise hits. ``None``
            falls back to the vendor's heuristic.
        retention: Coarse cache-window hint. ``"short"`` ≈ in-memory
            / 5-minute window; ``"long"`` ≈ persistent / hours.
            Adapters map to the nearest vendor equivalent (OpenAI:
            ``"in_memory"`` / ``"24h"``). ``None`` keeps the vendor
            default.
    """

    key: str | None = None
    retention: Literal["short", "long"] | None = None


__all__ = ["CacheConfig"]
