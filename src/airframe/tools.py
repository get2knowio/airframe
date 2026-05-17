""":class:`FunctionTool` — typed Python callable the model can invoke.

Phase 3 of the [implementation plan](../../docs/implementation-plan.md)
introduces ``tools=[FunctionTool(...)]`` on
:meth:`AgentRuntime.session`. Each :class:`FunctionTool` registers one
Python coroutine the model can call mid-turn; airframe handles the
per-vendor wire translation so consumer code is identical across
adapters.

Per-vendor routing (lands in Iterations B + C):

* **OpenAI-compatible HTTP** — translated to
  ``tools=[{"type":"function","function":{...}}]`` on
  :meth:`chat.completions.create`. The model returns ``tool_calls``;
  airframe invokes each handler, appends a ``role="tool"`` message
  with the result, and re-calls. Client-side tool-loop with an
  iteration cap.
* **Claude Agent SDK** — translated into an in-process MCP server
  via :func:`claude_agent_sdk.create_sdk_mcp_server` + the
  :func:`claude_agent_sdk.tool` decorator; attached via
  ``ClaudeAgentOptions.mcp_servers``. The SDK dispatches; airframe
  surfaces ``ToolUseBlock`` / ``ToolResultBlock`` as
  :class:`~airframe.events.ToolCallStart` /
  :class:`~airframe.events.ToolCallResult`.
* **GitHub Copilot SDK** — translated to
  :func:`copilot.define_tool` and passed via
  :meth:`CopilotClient.create_session(tools=...)`. Coexists with the
  forced-``submit_result`` pattern when ``schema=`` is also set.
* **OpenAI Codex SDK** — declines. ``CodexRuntime`` doesn't expose a
  Python tool-registration API; tools must be wired through the
  ``codex`` CLI's config file. ``session(tools=...)`` raises
  :class:`~airframe.errors.UnsupportedFeatureError` with
  ``feature=Feature.TOOLS_FUNCTION``.

**Shape lock.** ⚠️ The :class:`FunctionTool` field set and the
``handler: (BaseModel) -> Awaitable[Any]`` signature are
shape-locked in v0.6. Consumer code defines handlers against this
signature; broadening it later (e.g. adding a context argument)
must be done with a kwarg-only optional parameter so existing
handlers keep working.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class FunctionTool:
    """One Python tool the model can invoke during a session.

    Attributes:
        name: Tool identifier the model uses to call this tool. Must
            be a valid identifier — vendor APIs generally restrict to
            ``[a-zA-Z0-9_-]`` and reject duplicates within a session.
        description: One-sentence description shown to the model.
            Plain text; no markdown.
        params: Pydantic :class:`BaseModel` subclass whose schema is
            exposed to the vendor and whose instance is parsed from
            the model's call arguments before the handler runs.
            Empty-argument tools use a model with no fields.
        handler: Async callable invoked with the parsed
            :class:`BaseModel`. Must return a JSON-serialisable value
            — adapters serialise the return into the conversation
            wire format. Raise to signal a tool error; the resulting
            :class:`~airframe.events.ToolCallResult` carries
            ``is_error=True`` and the model can recover on a
            subsequent turn.

    Example::

        from pydantic import BaseModel

        class AddParams(BaseModel):
            a: float
            b: float

        async def add(params: AddParams) -> float:
            return params.a + params.b

        calculator = FunctionTool(
            name="add",
            description="Add two numbers.",
            params=AddParams,
            handler=add,
        )

        async with runtime.session(tools=[calculator]) as sess:
            result = await sess.execute("What is 17 plus 23?")
    """

    name: str
    description: str
    params: type[BaseModel]
    handler: Callable[[BaseModel], Awaitable[Any]]


__all__ = ["FunctionTool"]
