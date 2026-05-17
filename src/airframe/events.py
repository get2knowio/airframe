""":class:`RuntimeEvent` — the streaming event taxonomy.

Phase 1 of the [implementation plan](../../docs/implementation-plan.md)
introduces :class:`~airframe.protocol.AgentSession` and its
``stream()`` method. This module defines the discriminated union
``stream()`` yields, modelled on the deltas every major agent SDK
already exposes:

* Anthropic's ``message_delta`` / ``content_block_delta`` /
  ``input_json_delta`` events.
* OpenAI Chat Completions' ``ChatCompletionChunk`` (``delta.content``,
  ``delta.tool_calls``).
* GitHub Copilot's ``ASSISTANT_MESSAGE_DELTA`` /
  ``ASSISTANT_REASONING_DELTA`` / ``TOOL_USE_*`` session events.
* OpenAI Codex's ``ItemStartedEvent`` / ``ItemCompletedEvent``.

The union is intentionally small. Five variants cover what every
adapter can map onto today; vendor-specific events surface via
``unwrap()`` on the native client when consumers need them.

**Shape lock.** ⚠️ Per ADR-003 in the implementation plan, the variant
set and field-by-field shapes here are an irreversible decision —
once consumer code does ``match event:`` over these types, renaming
fields or removing variants forces consumer rewrites. *Adding* a new
variant is safe (consumers branch with a wildcard or ``isinstance``);
removing or renaming is a major-version break.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from airframe.protocol import RuntimeResult


@dataclass(frozen=True, slots=True)
class TextDelta:
    """A chunk of assistant-visible text.

    Concatenating the ``text`` of every :class:`TextDelta` in order,
    for one turn, equals :attr:`RuntimeResult.text` on the eventual
    :class:`TurnComplete`. Adapters that don't natively stream text
    (rare today) may emit a single ``TextDelta`` carrying the full
    response immediately before :class:`TurnComplete`.
    """

    text: str


@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    """A chunk of *hidden* reasoning / extended-thinking text.

    Distinct from :class:`TextDelta`: reasoning text is the model's
    private chain-of-thought and is not part of the assistant-visible
    response. Anthropic surfaces it via ``thinking`` content blocks;
    OpenAI/Codex via ``reasoning`` output items on Responses /
    ``ItemStartedEvent`` of kind ``reasoning`` on Codex. Adapters that
    can't surface reasoning text (Copilot, today) simply never emit
    this variant — capability negotiation lives on
    :data:`~airframe.features.Feature.REASONING_EFFORT`.
    """

    text: str


@dataclass(frozen=True, slots=True)
class ToolCallStart:
    """Fired when the model asks to invoke a tool.

    Attributes:
        tool_name: Tool the model is calling. For
            :class:`~airframe.tools.FunctionTool` (Phase 3) this is
            :attr:`FunctionTool.name`; for MCP tools (Phase 4) the
            vendor-reported tool name.
        tool_call_id: Vendor-assigned identifier for *this* invocation.
            Pairs the start with its matching :class:`ToolCallResult`.
            Adapters synthesise an ID when the vendor doesn't supply
            one — never empty.
        arguments_preview: Best-effort partial-JSON view of the tool
            arguments as they stream in. May be incomplete /
            unparseable mid-stream. Consumers wanting the parsed
            arguments should wait for :class:`ToolCallResult` (which
            carries them on the matching tool's handler invocation) or
            read the structured payload from the next turn.
    """

    tool_name: str
    tool_call_id: str
    arguments_preview: str


@dataclass(frozen=True, slots=True)
class ToolCallResult:
    """Fired when a tool invocation completes.

    Attributes:
        tool_call_id: Matches the originating :class:`ToolCallStart`.
        output: The tool's return value as the handler produced it.
            Adapters serialise this back into the conversation per
            their vendor's tool-result wire format.
        is_error: ``True`` when the handler raised or returned an
            error sentinel — the model sees the error and can recover
            on a subsequent turn. ``False`` for successful results.
    """

    tool_call_id: str
    output: Any
    is_error: bool


@dataclass(frozen=True, slots=True)
class TurnComplete:
    """Final event in every successful stream.

    Carries the same :class:`RuntimeResult` that
    :meth:`~airframe.protocol.AgentSession.execute` would have returned
    for the same prompt — text + (optional) structured payload + cost
    + finish reason. Consumers that only need the final result can
    drain the stream and read this last event; consumers that want
    progressive UI also consume the deltas in between.

    Implementations must yield exactly one :class:`TurnComplete` per
    turn, and it must be the last event in the stream. Cancelled
    streams may end without a :class:`TurnComplete` — see
    :meth:`~airframe.protocol.AgentSession.cancel`.
    """

    result: RuntimeResult


#: Discriminated union of every event :meth:`AgentSession.stream`
#: yields. Use ``match event:`` (PEP 634) or ``isinstance`` to
#: dispatch. The union is open-by-convention: later phases may add
#: variants (e.g. permission-decision events in Phase 5) and consumer
#: code should branch with a wildcard / default arm so unknown variants
#: don't crash.
RuntimeEvent = TextDelta | ReasoningDelta | ToolCallStart | ToolCallResult | TurnComplete


__all__ = [
    "ReasoningDelta",
    "RuntimeEvent",
    "TextDelta",
    "ToolCallResult",
    "ToolCallStart",
    "TurnComplete",
]
