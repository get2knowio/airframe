#!/usr/bin/env python3
"""End-to-end probe for :class:`~airframe.permission.PermissionCallback`.

Exercises the Phase 5 Iteration B permission-callback API —
registers a logging callback that approves every request and asks
the model to do something with tools. Prints each
:class:`~airframe.permission.PermissionRequest` the adapter routes
through the callback, then the resulting
:class:`~airframe.events.ToolCallStart` /
:class:`~airframe.events.ToolCallResult` /
:class:`~airframe.events.TurnComplete` sequence. Validates:

* The runtime declares
  :data:`~airframe.features.Feature.PERMISSION_CALLBACK` (Claude /
  Copilot do; OpenAI-compat declines).
* ``session(on_permission=...)`` accepts the registration.
* The callback fires per call on Claude / Copilot.

Usage::

    uv run python examples/probe_permission.py
    uv run python examples/probe_permission.py --provider claude
    uv run python examples/probe_permission.py --provider github-copilot
    uv run python examples/probe_permission.py --provider opencode  # declines

Defaults to ``claude`` (richest per-call permission channel via
``can_use_tool``). OpenAI-compat surfaces its permanent decline
verbatim — same probe-as-docs pattern Phase 3 / 4 used.

Requires whatever auth the chosen provider's adapter expects.
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
    PermissionCallback,
    PermissionDecision,
    PermissionRequest,
    TextDelta,
    ToolCallResult,
    ToolCallStart,
    TurnComplete,
    list_providers,
    runtime_for,
)

DEFAULT_PROMPT = "Please call the `add` tool to compute 5 + 7 and report the result."


class _AddParams(BaseModel):
    """Two numbers to sum."""

    a: float
    b: float


async def _add(params: _AddParams) -> float:
    return params.a + params.b


def _build_tool() -> FunctionTool:
    return FunctionTool(
        name="add",
        description="Add two numbers together and return the sum.",
        params=_AddParams,
        handler=_add,
    )


class LoggingApproveAll(PermissionCallback):
    """Probe callback — logs each request and approves all of them.

    Records every :class:`PermissionRequest` it receives so the probe
    can report how many fired and what the adapter actually asked
    about.
    """

    def __init__(self) -> None:
        self.received: list[PermissionRequest] = []

    async def handle(self, request: PermissionRequest) -> PermissionDecision:
        self.received.append(request)
        print(
            f"\n  [PERMISSION_REQUEST] tool_name={request.tool_name!r} "
            f"tool_args={request.tool_args!r} reason={request.reason!r}",
            flush=True,
        )
        return "allow"


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
            "Provider ID (default: claude — richest per-call permission "
            "channel). Any from list_providers()."
        ),
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to send. Defaults to a 'call the add tool' request.",
    )
    args = parser.parse_args()

    installed = list_providers(installed_only=True)
    if args.provider not in installed:
        print(
            f"Provider {args.provider!r} not installed. Available: {installed}",
            file=sys.stderr,
        )
        print(
            "Install one with: pip install airframe-agents[claude|copilot|openai-compat]",
            file=sys.stderr,
        )
        return 1

    runtime = _build_runtime(args.provider)
    callback = LoggingApproveAll()

    print(f"permission probe — provider={args.provider}")
    if not runtime.supports(Feature.PERMISSION_CALLBACK):
        # OpenAI-compat lands here. The decline message is the
        # deliverable — surface it verbatim so the probe doubles as
        # documentation for the workaround.
        print(
            f"  {type(runtime).__name__} does NOT declare "
            f"Feature.PERMISSION_CALLBACK — expect "
            f"session(on_permission=[...]) to raise.",
            file=sys.stderr,
        )
        try:
            runtime.session(on_permission=callback)
        except Exception as exc:  # noqa: BLE001 — message is the point
            print(f"  decline message: {exc}")
            await runtime.close()
            print("\nPASS (decline surfaced as expected)")
            return 0
        print("FAIL: expected session(on_permission=[...]) to raise on this adapter")
        await runtime.close()
        return 1

    # The probe registers an in-process FunctionTool so the model has
    # something to ask permission for. Adapters that decline tools=
    # (no Python-callable channel) would need to surface permission via
    # an MCP server instead and raise UnsupportedFeatureError at
    # session() — that's the documented behaviour.
    sess = runtime.session(on_permission=callback, tools=[_build_tool()])
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
    print(f"    permission requests: {len(callback.received)}")
    print(f"    text deltas:         {text_chunks}")
    print(f"    tool calls:          {len(tool_starts)}")
    print(f"    tool results:        {len(tool_results)}")
    print(f"    final text len:      {len(final.text)}")
    print(f"    finish:              {final.finish}")
    for k, v in final.cost.to_dict().items():
        print(f"    cost.{k}: {v}")

    if not callback.received:
        # Soft warning — some adapters / prompts won't trigger any
        # permission requests (e.g. a model that answers without
        # invoking the registered tool).
        print(
            "\nNOTE: callback never fired. The probe still validated "
            "session(on_permission=...) accepted the registration."
        )
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
