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
from typing import Any, Literal

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


McpTransport = Literal["stdio", "http", "sse"]
"""Wire transports an :class:`McpServerRef` may declare.

Mirrors the three transports the in-tree adapters can route through
their vendor SDKs:

* ``"stdio"`` — launch a local subprocess and speak JSON-RPC over its
  stdin/stdout. Needs ``command``. Supported by Claude + Copilot.
* ``"http"`` — connect to a remote MCP server over HTTP. Needs
  ``url``. Supported by Claude + Copilot.
* ``"sse"`` — connect to a remote MCP server over Server-Sent
  Events. Needs ``url``. Supported by Claude only (Copilot's SDK has
  no SSE channel; refs of this transport raise on translation).
"""


@dataclass(frozen=True, slots=True)
class McpServerRef:
    """One MCP server the model may invoke tools on during a session.

    Phase 4 of the [implementation
    plan](../../docs/implementation-plan.md) introduces
    ``mcp_servers=[McpServerRef(...)]`` on
    :meth:`AgentRuntime.session`. Each ref names a Model Context
    Protocol server the adapter wires through its vendor SDK's native
    MCP slot — Claude's :attr:`ClaudeAgentOptions.mcp_servers`,
    Copilot's :meth:`CopilotClient.create_session(mcp_servers=...)`.

    Iteration A of Phase 4 ships only the scaffolding: every adapter's
    ``session()`` accepts the kwarg and a non-empty list raises
    :class:`~airframe.errors.UnsupportedFeatureError` until its
    transport-specific capability flag flips True in Iteration B
    (Claude — all three transports) and Iteration C (Copilot — stdio
    + http; SSE keeps a permanent decline). Codex and OpenAI-compat
    decline all transports permanently in Iteration D.

    Per-vendor routing (lands in Iterations B + C):

    * **Claude Agent SDK** — translated to the matching typed config
      (:class:`McpStdioServerConfig` / :class:`McpHttpServerConfig` /
      :class:`McpSSEServerConfig`) keyed by ``name`` and passed via
      :attr:`ClaudeAgentOptions.mcp_servers`. The
      :attr:`auth_token` becomes an ``Authorization: Bearer …``
      header (merged with caller-supplied ``headers=``).
    * **GitHub Copilot SDK** — translated to
      :class:`MCPStdioServerConfig` / :class:`MCPHTTPServerConfig`
      and passed via :meth:`CopilotClient.create_session`. SSE refs
      raise :class:`~airframe.errors.UnsupportedFeatureError` at
      translation time.
    * **OpenAI Codex SDK** — declines all transports. ``CodexRuntime``
      has no programmatic MCP-registration channel; wire MCP servers
      through ``~/.codex/config.toml``'s ``[[mcp_servers]]`` block
      instead.
    * **OpenAI-compatible HTTP** — declines all transports. MCP-as-tool
      is a Responses-API shape; the Chat Completions family this base
      wraps cannot serve it. A future ``OpenAIResponsesRuntime``
      could translate to the Responses-API ``{"type":"mcp",...}``
      tool shape.

    Attributes:
        name: Server identifier. Used as the dict key on Claude's
            :attr:`ClaudeAgentOptions.mcp_servers` and as the tool-name
            prefix the adapter strips on stream events
            (``mcp__<name>__<tool>`` → ``<tool>``). Must be unique
            within a session and a valid identifier — vendor APIs
            generally restrict to ``[a-zA-Z0-9_-]``.
        transport: One of ``"stdio"``, ``"http"``, ``"sse"``.
            Determines which other fields are required (see
            :meth:`__post_init__` validation).
        command: For ``transport="stdio"``, the argv list to launch
            (e.g. ``["uvx", "mcp-server-everything"]``). Required for
            stdio; must be ``None`` for the network transports.
        url: For ``transport="http"`` or ``"sse"``, the endpoint to
            connect to. Required for the network transports; must be
            ``None`` for stdio.
        headers: Optional dict of extra HTTP headers (network
            transports only). Caller-supplied entries pass through
            verbatim; the cache fingerprint participates from the
            *sorted keys* only — never the values — so sensitive
            header values don't leak into cache identity.
        auth_token: Optional bearer token. Materialised as
            ``Authorization: Bearer <token>`` when the adapter builds
            the vendor config; merged with any caller-supplied
            ``headers={"Authorization": …}`` (caller wins on
            collision). Never participates in the cache fingerprint.

    Raises:
        ValueError: ``__post_init__`` rejects transport/field mismatches
            — stdio without ``command``, network transports without
            ``url``, or fields set for the wrong transport (e.g.
            ``url`` on a stdio ref, ``command`` on an http ref).

    Example::

        from airframe import McpServerRef

        everything = McpServerRef(
            name="everything",
            transport="stdio",
            command=["uvx", "mcp-server-everything"],
        )

        async with runtime.session(mcp_servers=[everything]) as sess:
            result = await sess.execute("List the tools you can call.")
    """

    name: str
    transport: McpTransport
    command: list[str] | None = None
    url: str | None = None
    headers: dict[str, str] | None = None
    auth_token: str | None = None

    def __post_init__(self) -> None:
        if self.transport == "stdio":
            if not self.command:
                raise ValueError(
                    f"McpServerRef(name={self.name!r}, transport='stdio') requires "
                    f"command=[...] (the argv to launch); got command={self.command!r}."
                )
            if self.url is not None:
                raise ValueError(
                    f"McpServerRef(name={self.name!r}, transport='stdio') does not "
                    f"accept url=; got url={self.url!r}. Drop the url or switch to "
                    f"transport='http'/'sse'."
                )
        elif self.transport in ("http", "sse"):
            if not self.url:
                raise ValueError(
                    f"McpServerRef(name={self.name!r}, transport={self.transport!r}) "
                    f"requires url=; got url={self.url!r}."
                )
            if self.command is not None:
                raise ValueError(
                    f"McpServerRef(name={self.name!r}, transport={self.transport!r}) "
                    f"does not accept command=; got command={self.command!r}. Drop the "
                    f"command or switch to transport='stdio'."
                )
        else:  # pragma: no cover — Literal narrows this at type-check time
            raise ValueError(
                f"McpServerRef(name={self.name!r}) has unknown transport "
                f"{self.transport!r}; expected one of 'stdio', 'http', 'sse'."
            )


__all__ = ["FunctionTool", "McpServerRef", "McpTransport"]
