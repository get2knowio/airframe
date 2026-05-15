"""Provider/model binding primitives — vendor-agnostic.

A binding is a ``(provider_id, model_id)`` pair that identifies *which*
vendor an agent should route through and *which* model it should ask
for. Bindings are deliberately simple strings on both sides — adapters
match on ``provider_id`` to decide whether they can serve the binding
(via :meth:`AgentRuntime.validate_binding`).

A :class:`Tier` is an ordered list of bindings — a cascade. The cascade
machinery isn't part of airframe's core (different consumers have
different retry/fallback policies); airframe ships the data types and
leaves the cascade walk to the consumer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderModel:
    """One ``(provider_id, model_id)`` binding.

    Attributes:
        provider_id: Vendor identifier — ``"anthropic"``, ``"openai"``,
            ``"copilot"``, ``"opencode"``, etc. Each adapter declares
            which provider IDs it accepts via
            :attr:`AgentRuntime.SUPPORTED_PROVIDER_IDS`.
        model_id: The model identifier the vendor recognises
            (e.g. ``"claude-haiku-4-5"``, ``"gpt-5-mini"``).
    """

    provider_id: str
    model_id: str

    @property
    def label(self) -> str:
        return f"{self.provider_id}/{self.model_id}"

    def to_dict(self) -> dict[str, str]:
        return {"providerID": self.provider_id, "modelID": self.model_id}


@dataclass(frozen=True, slots=True)
class Tier:
    """An ordered cascade of bindings for one agent role.

    Tiers are role-named (``"review"``, ``"implement"``, etc.) so the
    agent declaration stays stable while consumers tune the model
    cascade for each role.

    Attributes:
        name: Role name (``"review"`` / ``"implement"`` / ...).
        bindings: Ordered tuple of :class:`ProviderModel`. The first
            entry is the preferred binding; later entries are
            fallbacks. Must be non-empty.
    """

    name: str
    bindings: tuple[ProviderModel, ...]

    def __post_init__(self) -> None:
        if not self.bindings:
            raise ValueError(f"Tier {self.name!r} must have at least one binding")


__all__ = ["ProviderModel", "Tier"]
