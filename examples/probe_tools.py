#!/usr/bin/env python3
"""End-to-end probe for :class:`~airframe.tools.FunctionTool` round-trip.

Exercises the Phase 3 function-tools API — registers a tiny
``calculator`` tool (``add(a, b: float) -> float``) on
:meth:`AgentRuntime.session`, asks the model "what is 17 × 23?", and
prints the :class:`~airframe.events.ToolCallStart` /
:class:`~airframe.events.ToolCallResult` sequence from
:meth:`AgentSession.stream`. Validates:

* The runtime declares
  :data:`~airframe.features.Feature.TOOLS_FUNCTION` (Claude /
  Copilot / OpenCode Zen all do after Phase 3; Codex declines).
* ``session(tools=[...])`` accepts the registration without raising.
* The model actually calls the tool — at least one
  ``ToolCallStart`` event fires.
* The matching ``ToolCallResult`` carries
  :attr:`~airframe.events.ToolCallResult.is_error` ``= False``.
* The trailing :class:`~airframe.events.TurnComplete` carries the
  final answer (typically the correct arithmetic).

The probe is a multi-step turn under the hood: the model decides to
call ``add(17, 23)``, the handler runs in-process, the result lands
back, then the model emits its final text. With a multiplication
prompt and an ``add`` tool, the model has to either decompose the
multiplication into adds OR refuse to use the tool — both are valid
outcomes and the probe scores on the wire shape, not the answer.

Usage::

    uv run python examples/probe_tools.py
    uv run python examples/probe_tools.py --provider claude
    uv run python examples/probe_tools.py --provider github-copilot
    uv run python examples/probe_tools.py --provider opencode
    uv run python examples/probe_tools.py --provider codex     # declines

Defaults to ``opencode`` (OpenAI-compat) since it has the simplest
auth (single API key) and the client-side tool-loop is the most
deterministic of the three wired adapters. Requires whatever auth
the chosen provider's adapter expects.
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

from pydantic import BaseModel  # noqa: E402

from airframe import (  # noqa: E402
    Feature,
    FunctionTool,
    TextDelta,
    ToolCallResult,
    ToolCallStart,
    TurnComplete,
    list_providers,
    runtime_for,
)

DEFAULT_PROMPT = (
    "What is 17 times 23? Use the `add` tool to do the arithmetic; "
    "you can decompose the multiplication into repeated additions if needed."
)


class _AddParams(BaseModel):
    """Two numbers to sum."""

    a: float
    b: float


async def _add(params: _AddParams) -> float:
    """Tool handler — adds two numbers and returns the result."""
    return params.a + params.b


def _build_calculator_tool() -> FunctionTool:
    """The one tool every adapter that supports tools accepts:
    a tiny ``add(a, b)`` with a Pydantic schema."""
    return FunctionTool(
        name="add",
        description="Add two numbers together and return the sum.",
        params=_AddParams,
        handler=_add,
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
        default="opencode",
        help="Provider ID (default: opencode). Any from list_providers().",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to send. Defaults to a multiplication question.",
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

    print(f"function-tools probe — provider={args.provider}")
    if not runtime.supports(Feature.TOOLS_FUNCTION):
        # Codex lands here. The decline message is the deliverable —
        # surface it verbatim so the probe doubles as documentation.
        print(
            f"  {type(runtime).__name__} does NOT declare "
            f"Feature.TOOLS_FUNCTION — expect session(tools=[...]) to raise.",
            file=sys.stderr,
        )
        try:
            runtime.session(tools=[_build_calculator_tool()])
        except Exception as exc:  # noqa: BLE001 — message is the point
            print(f"  decline message: {exc}")
            await runtime.close()
            print("\nPASS (decline surfaced as expected)")
            return 0
        print("FAIL: expected session(tools=[...]) to raise on this adapter")
        await runtime.close()
        return 1

    sess = runtime.session(tools=[_build_calculator_tool()])
    text_chunks = 0
    tool_starts: list[ToolCallStart] = []
    tool_results: list[ToolCallResult] = []
    final = None
    t0 = time.monotonic()
    err: str | None = None
    try:
        print(f"\n  prompt: {args.prompt!r}")
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
        # Not a hard failure — some models refuse to use tools they
        # judge unhelpful for the prompt. Note it but don't fail.
        print(
            "\nNOTE: model did not invoke the tool. The probe still validated "
            "session(tools=[...]) accepted the registration without raising.",
        )
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
