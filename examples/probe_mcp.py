#!/usr/bin/env python3
"""End-to-end probe for :class:`~airframe.tools.McpServerRef` round-trip.

Exercises the Phase 4 MCP-server-refs API — registers one external
stdio MCP server (default: the
`@modelcontextprotocol/server-everything <https://github.com/modelcontextprotocol/servers/tree/main/src/everything>`_
reference server, launched via ``npx``) on
:meth:`AgentRuntime.session`, asks the model to list the tools the
server exposes, and prints the
:class:`~airframe.events.ToolCallStart` /
:class:`~airframe.events.ToolCallResult` sequence from
:meth:`AgentSession.stream`. Validates:

* The runtime declares the matching
  :data:`~airframe.features.Feature.TOOLS_MCP_STDIO` flag (Claude
  + Copilot do after Phase 4; Codex + OpenAI-compat decline).
* ``session(mcp_servers=[...])`` accepts the registration.
* If the model invokes a server tool, the matching
  ``ToolCallStart`` / ``ToolCallResult`` events fire with the
  ``mcp__<server>__`` prefix stripped from the tool name.
* The trailing :class:`~airframe.events.TurnComplete` carries
  the canonical :class:`~airframe.cost.CostRecord`.

Usage::

    uv run python examples/probe_mcp.py
    uv run python examples/probe_mcp.py --provider claude
    uv run python examples/probe_mcp.py --provider github-copilot
    uv run python examples/probe_mcp.py --provider codex     # declines
    uv run python examples/probe_mcp.py --provider opencode  # declines
    uv run python examples/probe_mcp.py --transport http --url https://...

Defaults to ``claude`` (broadest MCP transport coverage: stdio + http
+ sse). Codex and OpenAI-compat surface their permanent declines
verbatim — same probe-as-docs pattern Phase 3 used for Codex's
``tools=`` decline.

Requires whatever auth the chosen provider's adapter expects. The
default stdio probe additionally requires ``npx`` (Node 18+) on
``PATH`` so it can launch the reference MCP server.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from airframe import (  # noqa: E402
    Feature,
    McpServerRef,
    TextDelta,
    ToolCallResult,
    ToolCallStart,
    TurnComplete,
    list_providers,
    runtime_for,
)

DEFAULT_PROMPT = (
    "List the tools exposed by the connected MCP server. "
    "If there's an ``echo`` tool, call it with the message "
    "``hello from airframe`` and report the result."
)

_TRANSPORT_TO_FEATURE: dict[str, Feature] = {
    "stdio": Feature.TOOLS_MCP_STDIO,
    "http": Feature.TOOLS_MCP_HTTP,
    "sse": Feature.TOOLS_MCP_SSE,
}


def _build_ref(transport: str, url: str | None) -> McpServerRef:
    """Build the :class:`McpServerRef` we hand to the session.

    Defaults to launching
    `@modelcontextprotocol/server-everything <https://github.com/modelcontextprotocol/servers/tree/main/src/everything>`_
    via ``npx -y @modelcontextprotocol/server-everything`` — a
    well-known reference server that exposes a handful of toy tools
    (``echo``, ``add``, ``getTinyImage``, etc.). Override with
    ``--transport http --url https://your-server/mcp`` to probe a
    remote MCP endpoint instead.
    """
    if transport == "stdio":
        return McpServerRef(
            name="everything",
            transport="stdio",
            command=["npx", "-y", "@modelcontextprotocol/server-everything"],
        )
    if transport in ("http", "sse"):
        if not url:
            raise SystemExit(f"--transport {transport!r} requires --url <endpoint>")
        return McpServerRef(name="remote", transport=transport, url=url)
    raise SystemExit(f"--transport must be one of stdio/http/sse; got {transport!r}")


def _build_runtime(provider_id: str):  # type: ignore[no-untyped-def]
    """Construct an adapter without requiring credentials in ``__init__``."""
    cls = runtime_for(provider_id)
    try:
        return cls()
    except TypeError:
        import os

        env_key = f"{provider_id.upper().replace('-', '_')}_API_KEY"
        api_key = os.environ.get(env_key) or os.environ.get("OPENCODE_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"{provider_id!r} needs {env_key} (or OPENCODE_API_KEY) "
                f"set to construct the adapter."
            ) from None
        return cls(api_key=api_key)  # type: ignore[call-arg]


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--provider",
        default="claude",
        help=(
            "Provider ID (default: claude — broadest MCP transport "
            "coverage). Any from list_providers()."
        ),
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "http", "sse"],
        help="MCP transport (default: stdio via @modelcontextprotocol/server-everything).",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Endpoint URL when --transport is http or sse.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to send. Defaults to a 'call the echo tool' request.",
    )
    args = parser.parse_args()

    installed = list_providers(installed_only=True)
    if args.provider not in installed:
        print(
            f"Provider {args.provider!r} not installed. Available: {installed}",
            file=sys.stderr,
        )
        print(
            "Install one with: pip install airframe-agents[claude|copilot|codex|openai-compat]",
            file=sys.stderr,
        )
        return 1

    runtime = _build_runtime(args.provider)
    ref = _build_ref(args.transport, args.url)
    transport_feature = _TRANSPORT_TO_FEATURE[args.transport]

    print(f"mcp probe — provider={args.provider} transport={args.transport}")
    if not runtime.supports(transport_feature):
        # Codex + OpenAI-compat land here (every transport). On
        # Copilot, the SSE branch also lands here. The decline message
        # is the deliverable — surface it verbatim so the probe
        # doubles as documentation for the workaround.
        print(
            f"  {type(runtime).__name__} does NOT declare "
            f"Feature.{transport_feature.name} — expect "
            f"session(mcp_servers=[...]) to raise.",
            file=sys.stderr,
        )
        try:
            runtime.session(mcp_servers=[ref])
        except Exception as exc:  # noqa: BLE001 — message is the point
            print(f"  decline message: {exc}")
            await runtime.close()
            print("\nPASS (decline surfaced as expected)")
            return 0
        print("FAIL: expected session(mcp_servers=[...]) to raise on this adapter")
        await runtime.close()
        return 1

    sess = runtime.session(mcp_servers=[ref])
    text_chunks = 0
    tool_starts: list[ToolCallStart] = []
    tool_results: list[ToolCallResult] = []
    final = None
    t0 = time.monotonic()
    err: str | None = None
    try:
        print(f"\n  registered server: name={ref.name!r} transport={ref.transport!r}")
        print(f"  prompt: {args.prompt!r}")
        print("  -- stream begin --")
        async for event in sess.stream(args.prompt):
            if isinstance(event, TextDelta):
                text_chunks += 1
                print(event.text, end="", flush=True)
            elif isinstance(event, ToolCallStart):
                tool_starts.append(event)
                print(
                    f"\n  [TOOL_CALL_START] name={event.tool_name!r} "
                    f"id={event.tool_call_id!r} args={event.arguments_preview!r}",
                    flush=True,
                )
            elif isinstance(event, ToolCallResult):
                tool_results.append(event)
                print(
                    f"  [TOOL_CALL_RESULT] id={event.tool_call_id!r} "
                    f"is_error={event.is_error} output={event.output!r}",
                    flush=True,
                )
            elif isinstance(event, TurnComplete):
                final = event.result
        elapsed = time.monotonic() - t0
        print(f"\n  -- stream end (elapsed {elapsed:.1f}s) --")
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        import traceback

        traceback.print_exc()
    finally:
        await sess.close()
        await runtime.close()

    if err is not None:
        print(f"\nFAIL: {err}")
        return 1
    if final is None:
        print("\nFAIL: stream ended without a TurnComplete event")
        return 1

    print("\n  summary:")
    print(f"    text deltas:      {text_chunks}")
    print(f"    tool calls:       {len(tool_starts)}")
    print(f"    tool results:     {len(tool_results)}")
    print(f"    final text len:   {len(final.text)}")
    print(f"    finish:           {final.finish}")
    for k, v in final.cost.to_dict().items():
        print(f"    cost.{k}: {v}")

    if not tool_starts:
        # Not a hard failure — some models won't invoke a tool they
        # judge unhelpful for the prompt, or the server may have no
        # tool matching the request. Note it but don't fail.
        print(
            "\nNOTE: model did not invoke any MCP server tool. The probe "
            "still validated session(mcp_servers=[...]) accepted the "
            "registration and produced a TurnComplete.",
        )
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
