"""Unit tests for :mod:`airframe.options`.

Pins the *shape* of the public surface — frozen+slots, tagged union
(no shared base), value-equality / hashability. The dataclass
*bodies* (which knobs each adapter exposes) are tested per-adapter
in the corresponding ``tests/test_*_session.py`` file; this module
only owns the cross-cutting invariants.

Phase 0 shipped the namespaces empty; v0.5.0-readiness populates
:class:`ClaudeOptions` (three fields), :class:`CopilotOptions`
(four), :class:`OpenAICompatOptions` (six), :class:`BedrockOptions`
and threads each adapter's fields through into
the matching vendor SDK at session-build time.
"""

from __future__ import annotations

from dataclasses import is_dataclass

from airframe import (
    ClaudeOptions,
    CopilotOptions,
    OpenAICompatOptions,
)


def test_all_options_are_frozen_dataclasses() -> None:
    """Same discipline as CostRecord / RuntimeResult / ProviderModel.

    Frozen + slots gives us cheap, hashable value objects that won't
    be silently mutated after construction. Tested up front because
    flipping ``frozen=False`` later would be a breaking change to
    consumers relying on hashability.
    """
    for cls in (ClaudeOptions, CopilotOptions, OpenAICompatOptions):
        assert is_dataclass(cls), f"{cls.__name__} must be a dataclass"
        # ``__dataclass_params__`` is the canonical introspection surface
        # for the ``frozen`` / ``slots`` flags passed to ``@dataclass``.
        params = cls.__dataclass_params__  # type: ignore[attr-defined]
        assert params.frozen, f"{cls.__name__} must be frozen=True"
        # slots=True suppresses ``__dict__``; checked separately so the
        # failure mode is specific if either flag is dropped.
        assert not hasattr(cls(), "__dict__"), f"{cls.__name__} must be slots=True"


def test_options_have_no_common_base() -> None:
    """Vercel-style tagged union — no shared base class.

    A shared base class would tempt consumers to pass a neutral
    ``ProviderOptions()`` value, which defeats the point of static
    type-checking on per-adapter options. Each class inherits only
    from :class:`object`.
    """
    for cls in (ClaudeOptions, CopilotOptions, OpenAICompatOptions):
        # __mro__ should be (cls, object) — anything in between would
        # signal an accidental shared base.
        assert cls.__mro__ == (cls, object), (
            f"{cls.__name__} should not share a base class with other options; "
            f"got MRO {cls.__mro__}"
        )


def test_options_field_inventory() -> None:
    """Pin the per-namespace populated field set.

    Field additions are additive (consumers default-construct or
    cherry-pick), so this test exists mainly to catch accidental
    *removals* / *renames* that would break consumers branching on
    ``ClaudeOptions.fork_session`` etc.
    """
    assert set(ClaudeOptions.__dataclass_fields__) == {
        "append_system_prompt",
        "fork_session",
        "strict_mcp_config",
    }
    assert set(CopilotOptions.__dataclass_fields__) == {
        "available_tools",
        "excluded_tools",
        "skill_directories",
        "working_directory",
    }
    assert set(OpenAICompatOptions.__dataclass_fields__) == {
        "prompt_cache_key",
        "prompt_cache_retention",
        "service_tier",
        "safety_identifier",
        "verbosity",
        "store",
    }


def test_options_constructible_with_no_args() -> None:
    """Default construction works — important for Phase 1 wiring.

    Phase 1 will add ``runtime.session(provider_options=...)``; a
    default-constructible empty options object is the natural "no
    vendor-specific knobs" signal.
    """
    ClaudeOptions()
    CopilotOptions()
    OpenAICompatOptions()


def test_options_are_hashable() -> None:
    """Frozen dataclasses are hashable; useful for cache keys / sets.

    Adapters in later phases may cache sessions keyed by
    ``(model, system, schema, provider_options)`` — that requires the
    options object to hash. Locked in now to prevent a future
    "let's add an unhashable list field" PR from breaking the
    contract silently.
    """
    assert hash(ClaudeOptions()) == hash(ClaudeOptions())
    assert hash(CopilotOptions()) == hash(CopilotOptions())
    assert hash(OpenAICompatOptions()) == hash(OpenAICompatOptions())
