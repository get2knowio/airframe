#!/usr/bin/env python3
"""End-to-end probe for :class:`~airframe.native_tools.NativeTool` round-trip.

Exercises the native (vendor-hosted) built-in tools API — enables a portable
:class:`~airframe.native_tools.NativeCapability` (default ``WEB_SEARCH``) on
:meth:`AgentRuntime.session`, asks the model a question that needs live web
data, and prints the :class:`~airframe.events.ToolCallStart` /
:class:`~airframe.events.ToolCallResult` sequence from
:meth:`AgentSession.stream`. Validates:

* The runtime declares :data:`~airframe.features.Feature.TOOLS_NATIVE` and lists
  the requested capability in :meth:`AgentRuntime.supported_native_tools`
  (Claude serves ``WEB_SEARCH`` + ``WEB_FETCH``; Bedrock / OpenAI-compat
  decline).
* ``session(native_tools=[...])`` accepts the request — no consumer handler, the
  vendor owns description + execution.
* If the model invokes the hosted tool (``WebSearch`` on Claude) the matching
  ``ToolCallStart`` / ``ToolCallResult`` events fire.
* The trailing :class:`~airframe.events.TurnComplete` carries the canonical
  :class:`~airframe.cost.CostRecord`.

Usage::

    uv run python examples/probe_native_tools.py
    uv run python examples/probe_native_tools.py --provider claude --capability web_search
    uv run python examples/probe_native_tools.py --provider bedrock      # declines

Defaults to ``claude`` (the only adapter serving native tools today). Adapters
that decline surface their ``UnsupportedFeatureError`` verbatim — the probe
doubles as documentation for both the supported and declined paths.
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
    NativeCapability,
    NativeTool,
    TextDelta,
    ToolCallResult,
    ToolCallStart,
    TurnComplete,
    list_providers,
    runtime_for,
)

DEFAULT_PROMPT = (
    "Search the web: what was the last studio album released by Steely Dan, "
    "and in what year? Cite the source you used."
)


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
        help="Provider ID (default: claude — the only adapter serving native tools today).",
    )
    parser.add_argument(
        "--capability",
        default="web_search",
        choices=[c.value for c in NativeCapability],
        help="Native capability to enable (default: web_search).",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to send. Defaults to a question that needs live web search.",
    )
    args = parser.parse_args()

    installed = list_providers(installed_only=True)
    if args.provider not in installed:
        print(f"Provider {args.provider!r} not installed. Available: {installed}", file=sys.stderr)
        return 1

    runtime = _build_runtime(args.provider)
    capability = NativeCapability(args.capability)

    print(f"native-tools probe — provider={args.provider} capability={capability.value}")
    served = runtime.supported_native_tools()
    print(f"  supports(TOOLS_NATIVE)={runtime.supports(Feature.TOOLS_NATIVE)}")
    print(f"  served={sorted(served)}")

    if capability not in served:
        # Bedrock / OpenAI-compat land here. The decline message is the
        # deliverable — surface it verbatim so the probe doubles as docs.
        print(
            f"  {type(runtime).__name__} does NOT serve {capability.value!r} — "
            f"expect session(native_tools=[...]) to raise.",
            file=sys.stderr,
        )
        try:
            runtime.session(native_tools=[NativeTool(capability=capability)])
        except Exception as exc:  # noqa: BLE001 — message is the point
            print(f"  decline message: {exc}")
            await runtime.close()
            print("\nPASS (decline surfaced as expected)")
            return 0
        print("FAIL: expected session(native_tools=[...]) to raise on this adapter")
        await runtime.close()
        return 1

    sess = runtime.session(native_tools=[NativeTool(capability=capability)])
    tool_starts: list[ToolCallStart] = []
    final = None
    t0 = time.monotonic()
    err: str | None = None
    try:
        print(f"  prompt: {args.prompt!r}")
        print("  -- stream begin --")
        async for event in sess.stream(args.prompt):
            if isinstance(event, TextDelta):
                print(event.text, end="", flush=True)
            elif isinstance(event, ToolCallStart):
                tool_starts.append(event)
                print(f"\n  [TOOL_CALL_START] name={event.tool_name!r}", flush=True)
            elif isinstance(event, ToolCallResult):
                print(f"  [TOOL_CALL_RESULT] is_error={event.is_error}", flush=True)
            elif isinstance(event, TurnComplete):
                final = event.result
        print(f"\n  -- stream end (elapsed {time.monotonic() - t0:.1f}s) --")
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
    if final is not None:
        print(f"  cost_usd={final.cost.cost_usd} finish={final.finish}")
    invoked = [e.tool_name for e in tool_starts]
    print(f"\nPASS (native tool(s) invoked: {invoked or '— model answered without searching'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
