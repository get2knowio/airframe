"""Model discovery types — :class:`ModelInfo` and capability flags.

Each :class:`AgentRuntime` adapter implements ``list_models()`` to
return the set of models the consumer can pick from in a menu / UI.
The vendor APIs themselves typically return only model IDs (and
sometimes display names); the per-adapter implementations enrich
those with a curated metadata table for context window, pricing,
and capability flags.

The expected use case is **driving a UI** — e.g. a "provider →
model" pulldown in a config form. ``list_models()`` is an async call
that hits the live vendor endpoint with the user's resolved
credentials; a failed call (auth missing, network down) is a signal
the consumer should surface to the user *before* letting them pick
a model that would later fail to run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Capability flags. Adapters set whichever apply to each model. Free-form
# strings rather than an enum so adapters can declare vendor-specific
# capabilities (e.g. ``"reasoning_effort"`` for Copilot, ``"web_search"``
# for some OpenAI models) without changing airframe's surface.
CAPABILITY_VISION = "vision"
CAPABILITY_TOOLS = "tools"
CAPABILITY_STRUCTURED_OUTPUT = "structured_output"
CAPABILITY_STREAMING = "streaming"
CAPABILITY_REASONING_EFFORT = "reasoning_effort"


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """One model exposed by a provider.

    Drives UI menus and consumer-side model selection. The vendor API
    returns ``id`` (and sometimes ``display_name``); the rest is
    enriched from a per-adapter curated metadata table. Fields the
    adapter can't determine come back ``None`` / empty —
    UIs render those gracefully (no context window shown, no pricing,
    no capability badges).

    Attributes:
        id: The model identifier — what goes into
            :attr:`ProviderModel.model_id` when selecting this model.
        display_name: Human-friendly label for menus
            (e.g. ``"Claude Haiku 4.5"`` not ``"claude-haiku-4-5"``).
            Falls back to ``id`` when the vendor doesn't provide one.
        provider_id: The canonical provider ID this model belongs to
            (matches the runtime's :attr:`PROVIDER_ID`).
        context_window: Maximum token context window the model supports,
            or ``None`` if the adapter doesn't know.
        pricing_input_per_1k_usd: USD per 1K input tokens, or ``None``.
        pricing_output_per_1k_usd: USD per 1K output tokens, or ``None``.
        capabilities: Set of capability flags (see ``CAPABILITY_*``
            module constants). Empty when the adapter has no opinion.
        raw: Vendor-specific raw response — kept for diagnostics /
            advanced consumers. Treat as opaque.
    """

    id: str
    display_name: str
    provider_id: str
    context_window: int | None = None
    pricing_input_per_1k_usd: float | None = None
    pricing_output_per_1k_usd: float | None = None
    capabilities: frozenset[str] = field(default_factory=frozenset)
    raw: object = field(default=None, repr=False)


__all__ = [
    "CAPABILITY_REASONING_EFFORT",
    "CAPABILITY_STREAMING",
    "CAPABILITY_STRUCTURED_OUTPUT",
    "CAPABILITY_TOOLS",
    "CAPABILITY_VISION",
    "ModelInfo",
]
