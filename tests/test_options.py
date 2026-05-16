"""Unit tests for :mod:`airframe.options`.

Phase 0 ships the scaffold with empty dataclasses. These tests pin
the *shape* of the public surface — they exist mostly to catch
accidental field renames or base-class introductions that would break
the tagged-union pattern.
"""

from __future__ import annotations

from dataclasses import is_dataclass

from airframe import (
    ClaudeOptions,
    CodexOptions,
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
    for cls in (ClaudeOptions, CopilotOptions, CodexOptions, OpenAICompatOptions):
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
    for cls in (ClaudeOptions, CopilotOptions, CodexOptions, OpenAICompatOptions):
        # __mro__ should be (cls, object) — anything in between would
        # signal an accidental shared base.
        assert cls.__mro__ == (cls, object), (
            f"{cls.__name__} should not share a base class with other options; "
            f"got MRO {cls.__mro__}"
        )


def test_options_are_empty_in_phase_0() -> None:
    """Bodies are deliberately empty pending Phase 2+ feature work.

    Populating fields before a feature consumes them would lock names
    without a real user to validate the shape against. This test
    documents that decision — if a later phase adds a field, it
    should also update this test to match.
    """
    for cls in (ClaudeOptions, CopilotOptions, CodexOptions, OpenAICompatOptions):
        # __dataclass_fields__ is the canonical introspection surface.
        assert cls.__dataclass_fields__ == {}, (
            f"{cls.__name__} should have no fields yet "
            f"(Phase 0 scaffold); found {list(cls.__dataclass_fields__)}"
        )


def test_options_constructible_with_no_args() -> None:
    """Default construction works — important for Phase 1 wiring.

    Phase 1 will add ``runtime.session(provider_options=...)``; a
    default-constructible empty options object is the natural "no
    vendor-specific knobs" signal.
    """
    ClaudeOptions()
    CopilotOptions()
    CodexOptions()
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
    assert hash(CodexOptions()) == hash(CodexOptions())
    assert hash(OpenAICompatOptions()) == hash(OpenAICompatOptions())
