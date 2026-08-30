"""Unit tests for :class:`ImageInput`, :class:`FileInput`, and :data:`Prompt`.

Phase 2 of the implementation plan makes :meth:`AgentSession.execute`
and :meth:`AgentSession.stream` accept a polymorphic ``prompt``
argument. These tests pin the dataclass shapes (the public surface
adapter code interprets), the post-init validation, and the union
form so accidental drift is caught at PR time.
"""

from __future__ import annotations

import dataclasses
from typing import get_args

import pytest

from airframe import FileInput, ImageInput, Prompt, PromptPart

# ---------------------------------------------------------------------------
# ImageInput shape + validation
# ---------------------------------------------------------------------------


def test_image_input_fields() -> None:
    """``ImageInput(path=, bytes_=, url=, media_type=)`` — locked field set."""
    fields = {f.name: f.type for f in dataclasses.fields(ImageInput)}
    assert fields == {
        "path": "str | None",
        "bytes_": "bytes | None",
        "url": "str | None",
        "media_type": "str | None",
    }


def test_image_input_frozen_and_slots() -> None:
    """ImageInput is frozen + slots — immutable, memory-efficient."""
    assert ImageInput.__dataclass_params__.frozen is True
    assert "__slots__" in ImageInput.__dict__


def test_image_input_accepts_path() -> None:
    img = ImageInput(path="/tmp/test.png")
    assert img.path == "/tmp/test.png"
    assert img.bytes_ is None
    assert img.url is None


def test_image_input_accepts_bytes() -> None:
    img = ImageInput(bytes_=b"\x89PNG\r\n", media_type="image/png")
    assert img.bytes_ == b"\x89PNG\r\n"
    assert img.media_type == "image/png"


def test_image_input_accepts_url() -> None:
    img = ImageInput(url="https://example.com/foo.jpg")
    assert img.url == "https://example.com/foo.jpg"


def test_image_input_requires_at_least_one_source() -> None:
    """The 'at-least-one-of-{path, bytes_, url}' invariant fires at construction."""
    with pytest.raises(ValueError, match="exactly one"):
        ImageInput()


# ---------------------------------------------------------------------------
# FileInput shape
# ---------------------------------------------------------------------------


def test_file_input_fields() -> None:
    """``FileInput(path, media_type=)`` — path is required, media_type optional."""
    fields = {f.name: f.type for f in dataclasses.fields(FileInput)}
    assert fields == {
        "path": "str",
        "media_type": "str | None",
    }


def test_file_input_frozen_and_slots() -> None:
    assert FileInput.__dataclass_params__.frozen is True
    assert "__slots__" in FileInput.__dict__


def test_file_input_construction() -> None:
    f = FileInput(path="/tmp/doc.pdf", media_type="application/pdf")
    assert f.path == "/tmp/doc.pdf"
    assert f.media_type == "application/pdf"


def test_file_input_media_type_optional() -> None:
    f = FileInput(path="/tmp/notes.md")
    assert f.media_type is None


# ---------------------------------------------------------------------------
# PromptPart + Prompt union shape
# ---------------------------------------------------------------------------


def test_prompt_part_union_members() -> None:
    """``PromptPart = str | ImageInput | FileInput`` — locked variant set."""
    assert set(get_args(PromptPart)) == {str, ImageInput, FileInput}


def test_prompt_union_includes_str_and_list() -> None:
    """``Prompt = str | list[PromptPart]`` — both shapes addressable."""
    args = get_args(Prompt)
    # str is one arm; list[PromptPart] is the other.
    assert str in args
    assert any("list" in str(a) for a in args)


def test_prompt_str_is_valid_at_call_site() -> None:
    """A bare str must remain a valid Prompt value (back-compat with v0–Phase 1)."""
    p: Prompt = "hello"
    assert p == "hello"


def test_prompt_list_is_valid_at_call_site() -> None:
    """A list of interleaved str + ImageInput + FileInput is a valid Prompt."""
    p: Prompt = [
        "Look at this image:",
        ImageInput(path="/tmp/x.png"),
        "And this document:",
        FileInput(path="/tmp/notes.pdf"),
    ]
    assert len(p) == 4
    assert isinstance(p[1], ImageInput)
    assert isinstance(p[3], FileInput)
