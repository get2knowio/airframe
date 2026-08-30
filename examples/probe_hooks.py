#!/usr/bin/env python3
"""End-to-end probe for :class:`~airframe.hooks.HookEvent` emission.

Exercises the Phase 5 Iteration C lifecycle-hook API — registers a
logging ``on_event=`` observer, asks the model to call a function tool,
and prints every :class:`HookEvent` the adapter emits during the
session lifecycle (session_start → user_prompt_submit → pre_tool_use →
post_tool_use / tool_failure → session_end). Validates:

* The runtime declares :data:`~airframe.features.Feature.LIFECYCLE_HOOKS`
  (all four built-in adapters do after Iteration C).
* ``session(on_event=...)`` accepts the registration.
* The observer receives events in causal order, and the kinds emitted
  are a subset of the adapter's :attr:`EMITTABLE_HOOK_KINDS` ClassVar.

Usage::

    uv run python examples/probe_hooks.py
    uv run python examples/probe_hooks.py --provider claude
    uv run python examples/probe_hooks.py --provider github-copilot
    uv run python examples/probe_hooks.py --provider opencode

Defaults to ``claude`` (richest emittable-kind set — includes
``pre_compact`` and ``rate_limit`` neither Copilot nor
OpenAI-compat fire). The other adapters skip a handful of kinds (see
each adapter's :attr:`EMITTABLE_HOOK_KINDS`); the probe reports the
declared set per run.

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
    TextDelta,
    ToolCallResult,
    ToolCallStart,
    TurnComplete,
    list_providers,
    runtime_for,
)
from airframe.hooks import HookEvent  # noqa: E402

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


class LoggingObserver:
    """Probe observer — logs each :class:`HookEvent` and counts kinds.

    Captures every event in arrival order so the probe can report on
    causal ordering and the per-kind emission count.
    """

    def __init__(self) -> None:
        self.received: list[HookEvent] = []

    def __call__(self, event: HookEvent) -> None:
        self.received.append(event)
        payload_preview = {
            k: (v if not isinstance(v, str) or len(v) <= 80 else v[:77] + "...")
            for k, v in event.payload.items()
        }
        print(
            f"  [HOOK_EVENT] kind={event.kind!r} session_id={event.session_id!r} "
            f"payload={payload_preview!r}",
            flush=True,
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
        help=(
            "Provider ID (default: claude — richest emittable-kind set). Any from list_providers()."
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
    observer = LoggingObserver()

    print(f"hooks probe — provider={args.provider}")
    if not runtime.supports(Feature.LIFECYCLE_HOOKS):
        # No built-in adapter lands here after Iteration C, but the
        # probe stays honest about the capability check — third-party
        # adapters may legitimately decline.
        print(
            f"  {type(runtime).__name__} does NOT declare "
            f"Feature.LIFECYCLE_HOOKS — expect "
            f"session(on_event=[...]) to raise.",
            file=sys.stderr,
        )
        try:
            runtime.session(on_event=observer)
        except Exception as exc:  # noqa: BLE001 — message is the point
            print(f"  decline message: {exc}")
            await runtime.close()
            print("\nPASS (decline surfaced as expected)")
            return 0
        print("FAIL: expected session(on_event=[...]) to raise on this adapter")
        await runtime.close()
        return 1

    emittable = getattr(type(runtime), "EMITTABLE_HOOK_KINDS", frozenset())
    print(f"  declared EMITTABLE_HOOK_KINDS: {sorted(emittable)}")

    sess = runtime.session(on_event=observer, tools=[_build_tool()])
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
        # close() should fire session_end if the adapter hasn't already.
        await sess.close()
        await runtime.close()

    if err is not None:
        print(f"\nFAIL: {err}")
        return 1
    if final is None:
        print("\nFAIL: stream ended without a TurnComplete event")
        return 1

    # Per-kind histogram for quick visual scan.
    kinds_seen: dict[str, int] = {}
    for event in observer.received:
        kinds_seen[event.kind] = kinds_seen.get(event.kind, 0) + 1

    print("\n  summary:")
    print(f"    hook events:    {len(observer.received)}")
    for kind in sorted(kinds_seen):
        print(f"      {kind}: {kinds_seen[kind]}")
    print(f"    text deltas:    {text_chunks}")
    print(f"    tool calls:     {len(tool_starts)}")
    print(f"    tool results:   {len(tool_results)}")
    print(f"    final text len: {len(final.text)}")
    print(f"    finish:         {final.finish}")
    for k, v in final.cost.to_dict().items():
        print(f"    cost.{k}: {v}")

    # Causal-ordering sanity check: session_start (if emitted) must
    # come before anything else. session_end (if emitted) must come
    # last.
    if "session_start" in kinds_seen:
        first = observer.received[0]
        if first.kind != "session_start":
            print(
                f"\nWARN: first event was {first.kind!r}, expected 'session_start' "
                f"(causal-ordering check)."
            )
    if "session_end" in kinds_seen:
        last = observer.received[-1]
        if last.kind != "session_end":
            print(
                f"\nWARN: last event was {last.kind!r}, expected 'session_end' "
                f"(causal-ordering check)."
            )

    # Every emitted kind should be in the adapter's declared set.
    undeclared = sorted(set(kinds_seen) - emittable)
    if undeclared:
        print(
            f"\nFAIL: emitted kinds {undeclared!r} aren't in "
            f"EMITTABLE_HOOK_KINDS {sorted(emittable)!r} — fix the adapter."
        )
        return 1

    if not observer.received:
        print(
            "\nNOTE: observer never fired. The probe still validated "
            "session(on_event=...) accepted the registration."
        )
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
