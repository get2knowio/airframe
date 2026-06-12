""":class:`NativeTool` — vendor-hosted built-in tools, referenced not implemented.

A third tool shape alongside :class:`~airframe.tools.FunctionTool` (consumer
supplies a Python handler) and :class:`~airframe.tools.McpServerRef` (external
MCP server). A *native tool* is one the wrapped vendor SDK both **describes to
the model and executes itself** — Claude's ``WebSearch`` / ``WebFetch``, OpenAI's
``web_search`` / ``code_interpreter``, Kimi's ``$web_search``, Copilot's
``fetch_webpage``. The consumer neither implements nor hosts them; it only asks
the runtime to *enable* them, by reference.

The two shapes a :class:`NativeTool` can take:

* **Semantic** — a portable :class:`NativeCapability` (``WEB_SEARCH``,
  ``WEB_FETCH``, …). Each adapter maps the capability to its vendor's native
  tool name, or declines. This is the cross-vendor surface: the same
  ``NativeTool.web_search()`` enables Claude's ``WebSearch`` on
  :class:`ClaudeCodeRuntime` and (in future) OpenAI's ``web_search`` on a
  Responses-API runtime.
* **Raw** — an escape hatch carrying a ``provider_id`` + vendor tool ``name``
  for capabilities not yet in the taxonomy (``NativeTool.raw("claude",
  "WebSearch")``). A raw tool is honoured **only** by the adapter whose
  ``PROVIDER_ID`` matches; other adapters ignore it, so a single mixed list can
  carry per-provider raw tools and each runtime picks out its own.

**Scope.** This abstracts *hosted / server-side* built-in tools — the vendor's
infrastructure runs them, so they're portable to reference. It deliberately does
**not** cover *local-execution* built-ins (``Bash`` / ``Read`` / ``Write``)
which run on the SDK host and are about the agent's environment, not a portable
capability.

Capability negotiation is via :meth:`AgentRuntime.supported_native_tools` (which
:class:`NativeCapability` values an adapter can serve) gated structurally by
:data:`~airframe.features.Feature.TOOLS_NATIVE`. Per airframe's no-silent-
fallback principle, an explicitly requested semantic capability an adapter can't
serve raises :class:`~airframe.errors.UnsupportedFeatureError` rather than
silently dropping — graceful degradation is the consumer's job, by checking
``supported_native_tools()`` before passing ``native_tools=``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class NativeCapability(StrEnum):
    """Portable taxonomy of vendor-hosted built-in tool capabilities.

    :class:`enum.StrEnum` (matching :class:`~airframe.features.Feature`) so
    members compare equal to their wire value and serialise cleanly in logs /
    config. The whole forward-looking set is declared here; adapters turn
    support on by listing members in their ``SUPPORTED_NATIVE_TOOLS`` — they do
    not add enum members.

    Members map to vendor native tools as follows (✗ = no native equivalent):

    * ``WEB_SEARCH`` — Claude ``WebSearch`` · OpenAI ``web_search`` ·
      Kimi ``$web_search`` · OpenCode ``websearch``.
    * ``WEB_FETCH`` — Claude ``WebFetch`` · Copilot ``fetch_webpage`` ·
      OpenCode ``webfetch``.
    * ``CODE_EXECUTION`` — OpenAI ``code_interpreter``.
    * ``FILE_SEARCH`` — OpenAI ``file_search``.
    * ``IMAGE_GENERATION`` — OpenAI ``image_generation``.
    * ``COMPUTER_USE`` — OpenAI ``computer_use``.
    """

    WEB_SEARCH = "web_search"
    WEB_FETCH = "web_fetch"
    CODE_EXECUTION = "code_execution"
    FILE_SEARCH = "file_search"
    IMAGE_GENERATION = "image_generation"
    COMPUTER_USE = "computer_use"


@dataclass(frozen=True, slots=True)
class NativeTool:
    """One vendor-hosted built-in tool, enabled by reference.

    Exactly one of the two addressing modes must be set (enforced in
    :meth:`__post_init__`):

    * **Semantic** — ``capability`` set, ``provider_id`` / ``name`` ``None``.
      Portable; every adapter that lists the capability in
      ``SUPPORTED_NATIVE_TOOLS`` enables its native equivalent.
    * **Raw** — ``provider_id`` *and* ``name`` set, ``capability`` ``None``.
      Honoured only by the adapter whose ``PROVIDER_ID`` equals ``provider_id``;
      other adapters ignore it.

    Attributes:
        capability: The portable :class:`NativeCapability` to enable, or
            ``None`` for a raw tool.
        provider_id: For a raw tool, the adapter this tool targets
            (``"claude"``, ``"github-copilot"``, …). ``None`` for a semantic
            tool.
        name: For a raw tool, the exact vendor tool name to enable
            (``"WebSearch"``, ``"web_search"``, …). ``None`` for a semantic
            tool.
        options: Optional vendor-specific tuning forwarded to the native tool
            where the adapter supports it (e.g. ``{"max_uses": 5,
            "allowed_domains": [...]}`` for web search). Adapters that don't
            consume a given key ignore it; the value still participates in the
            session cache fingerprint so changing it forces a reconnect.

    Raises:
        ValueError: neither or both addressing modes set, or a raw tool missing
            ``provider_id`` / ``name``.

    Example::

        from airframe import NativeCapability, NativeTool

        if NativeCapability.WEB_SEARCH in runtime.supported_native_tools():
            sess = runtime.session(native_tools=[NativeTool.web_search()])
            result = await sess.execute("What did the band release in 1971?")
    """

    capability: NativeCapability | None = None
    provider_id: str | None = None
    name: str | None = None
    options: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        is_raw = self.provider_id is not None or self.name is not None
        if self.capability is not None and is_raw:
            raise ValueError(
                "NativeTool accepts EITHER a semantic capability= OR a raw "
                "(provider_id=, name=) pair, not both; got "
                f"capability={self.capability!r}, provider_id={self.provider_id!r}, "
                f"name={self.name!r}."
            )
        if self.capability is None and not is_raw:
            raise ValueError(
                "NativeTool needs a semantic capability= or a raw (provider_id=, "
                "name=) pair; got neither. Use NativeTool.web_search() / "
                "NativeTool.web_fetch() / NativeTool.raw(provider_id=..., name=...)."
            )
        if self.capability is None and (not self.provider_id or not self.name):
            raise ValueError(
                "A raw NativeTool needs BOTH provider_id= and name= set; got "
                f"provider_id={self.provider_id!r}, name={self.name!r}. Raw tools "
                "are addressed to one adapter so they never leak to another vendor."
            )

    @property
    def is_raw(self) -> bool:
        """``True`` for a raw (provider-addressed) tool, ``False`` for semantic."""
        return self.capability is None

    @classmethod
    def web_search(cls, **options: Any) -> NativeTool:
        """A portable :data:`NativeCapability.WEB_SEARCH` tool."""
        return cls(capability=NativeCapability.WEB_SEARCH, options=options or None)

    @classmethod
    def web_fetch(cls, **options: Any) -> NativeTool:
        """A portable :data:`NativeCapability.WEB_FETCH` tool."""
        return cls(capability=NativeCapability.WEB_FETCH, options=options or None)

    @classmethod
    def raw(cls, provider_id: str, name: str, **options: Any) -> NativeTool:
        """A raw vendor tool addressed to ``provider_id``.

        Escape hatch for native tools without a :class:`NativeCapability` entry
        yet, or for passing exact vendor names. Only the adapter whose
        ``PROVIDER_ID`` matches ``provider_id`` enables it; others ignore it.
        """
        return cls(provider_id=provider_id, name=name, options=options or None)


__all__ = ["NativeCapability", "NativeTool"]
