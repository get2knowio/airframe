""":class:`ImageInput` / :class:`FileInput` — polymorphic prompt parts.

Phase 2 of the [implementation plan](../../docs/implementation-plan.md)
makes :meth:`AgentSession.execute` and :meth:`AgentSession.stream`
accept ``prompt: Prompt`` where :data:`Prompt` is either a plain
``str`` (the v0-through-Phase-1 shape, still works) or a list of
:data:`PromptPart` — interleaved text, images, and files.

Each vendor handles attachments differently:

* **Anthropic / Claude Code** — images and files reach the model via
  the SDK's Read tool (auto-allowed for prompt-attached paths) or
  inline content blocks.
* **GitHub Copilot** — :class:`FileAttachment` shape on the session's
  ``send`` call.
* **OpenAI Codex** — :class:`LocalImageInput` on the Input shape.
* **OpenAI-compatible HTTP** — content parts on the user message
  (``[{"type": "text", ...}, {"type": "image_url", ...}, ...]``).

Airframe collapses the prompt-part shape onto one neutral pair of
dataclasses; each adapter translates to its vendor's wire format.
Adapters that don't declare
:data:`~airframe.features.Feature.VISION_INPUT` /
:data:`~airframe.features.Feature.FILE_INPUT` raise
:class:`~airframe.errors.UnsupportedFeatureError` when handed a
list-shaped prompt — the "no silent fallbacks" principle.

**Shape lock.** ⚠️ The dataclass fields and the ``str |
list[PromptPart]`` union form are public surface. Adding a new
:data:`PromptPart` variant later (audio, video) is safe (consumers
branch with ``isinstance``); removing or renaming fields is a
major-version break.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ImageInput:
    """An image attachment to send alongside text.

    Exactly one of :attr:`path`, :attr:`bytes_`, or :attr:`url` must
    be set — adapters route differently depending on which is
    populated (path → file read, bytes_ → inline base64 encode, url →
    pass-through reference). :attr:`media_type` is optional; adapters
    sniff from the path extension when omitted.

    Attributes:
        path: Local filesystem path to the image. Adapters resolve
            relative paths against the runtime's working directory
            (usually the process cwd).
        bytes_: Raw image bytes. Trailing underscore avoids shadowing
            the ``bytes`` builtin.
        url: HTTPS URL the vendor can fetch directly. Not every vendor
            supports remote fetches — adapters that require the bytes
            locally will download first.
        media_type: MIME type (e.g. ``"image/png"``, ``"image/jpeg"``,
            ``"image/webp"``). When ``None`` adapters infer from the
            path extension or HTTP ``Content-Type``.
    """

    path: str | None = None
    bytes_: bytes | None = None
    url: str | None = None
    media_type: str | None = None

    def __post_init__(self) -> None:
        if self.path is None and self.bytes_ is None and self.url is None:
            raise ValueError("ImageInput needs exactly one of path=, bytes_=, or url= set")


@dataclass(frozen=True, slots=True)
class FileInput:
    """A document attachment (PDF, plaintext, markdown, ...).

    Distinct from :class:`ImageInput` because vendors route documents
    through different attachment slots (Anthropic's file uploads,
    OpenAI's input_file content parts, Copilot's
    :class:`FileAttachment`). The split keeps adapter dispatch
    explicit rather than sniffing from path extension.

    Attributes:
        path: Local filesystem path to the file. Required — bytes /
            URL variants land in a later phase if a consumer asks.
        media_type: MIME type (e.g. ``"application/pdf"``,
            ``"text/markdown"``). When ``None`` adapters infer from
            the path extension.
    """

    path: str
    media_type: str | None = None


#: One element of a list-shaped prompt. ``str`` parts are interleaved
#: text; :class:`ImageInput` / :class:`FileInput` are typed attachments.
PromptPart = str | ImageInput | FileInput

#: Value type for the ``prompt`` argument on
#: :meth:`AgentSession.execute` / :meth:`AgentSession.stream`. A bare
#: ``str`` keeps Phase 0–1 call sites working unchanged; a
#: ``list[PromptPart]`` enables vision + file input on adapters that
#: declare the corresponding :class:`~airframe.features.Feature`.
Prompt = str | list[PromptPart]


__all__ = ["FileInput", "ImageInput", "Prompt", "PromptPart"]
