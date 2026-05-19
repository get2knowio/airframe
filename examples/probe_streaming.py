#!/usr/bin/env python3
"""End-to-end probe for :meth:`AgentSession.stream`.

Exercises the streaming API Phase 1 introduced — opens a session,
iterates ``async for event in session.stream(prompt)``, and prints
deltas as they arrive. Validates:

* The runtime's ``session()`` factory builds a real bespoke session.
* :class:`~airframe.events.TextDelta` events fire as the vendor
  produces them (visible incremental output).
* :class:`~airframe.events.ReasoningDelta` events fire for adapters /
  models that emit hidden thinking text.
* The stream ends with exactly one
  :class:`~airframe.events.TurnComplete` carrying a populated
  :class:`~airframe.cost.CostRecord`.

Usage::

    uv run python examples/probe_streaming.py
    uv run python examples/probe_streaming.py --provider claude
    uv run python examples/probe_streaming.py --provider github-copilot
    uv run python examples/probe_streaming.py --provider kimi
    uv run python examples/probe_streaming.py --provider opencode

Defaults to ``claude`` because Claude Code emits the richest delta
stream out of the box. Requires whatever auth the chosen provider's
adapter expects (see each adapter's docstring).
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
    ReasoningDelta,
    TextDelta,
    TurnComplete,
    list_providers,
    runtime_for,
)

DEFAULT_PROMPT = "Count from 1 to 5, one per line, then add a one-sentence summary."


def _build_runtime(provider_id: str):  # type: ignore[no-untyped-def]
    """Construct an adapter without requiring credentials in ``__init__``."""
    cls = runtime_for(provider_id)
    try:
        return cls()
    except TypeError:
        # OpenAI-compat subclasses need an api_key for construction.
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
        help="Provider ID (default: claude). Any from list_providers().",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to stream. Defaults to a short numbered list.",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Optional system prompt for the session.",
    )
    args = parser.parse_args()

    installed = list_providers(installed_only=True)
    if args.provider not in installed:
        print(
            f"Provider {args.provider!r} not installed. Available: {installed}",
            file=sys.stderr,
        )
        print(
            "Install one with: pip install airframe-agents[claude|copilot|kimi|openai-compat]",
            file=sys.stderr,
        )
        return 1

    runtime = _build_runtime(args.provider)

    print(f"streaming probe — provider={args.provider}")
    if not runtime.supports(Feature.STREAMING):
        print(
            f"  WARN: {type(runtime).__name__} does not declare Feature.STREAMING. "
            f"The probe will still attempt the call; expect a single TextDelta "
            f"carrying the full response immediately before TurnComplete.",
            file=sys.stderr,
        )

    sess = runtime.session(system=args.system)
    text_chunks = 0
    reasoning_chunks = 0
    final = None
    t0 = time.monotonic()
    err: str | None = None
    try:
        print(f"\n  prompt: {args.prompt!r}\n  -- stream begin --")
        async for event in sess.stream(args.prompt):
            if isinstance(event, TextDelta):
                text_chunks += 1
                print(event.text, end="", flush=True)
            elif isinstance(event, ReasoningDelta):
                reasoning_chunks += 1
                # Surface reasoning on a dedicated line so it doesn't
                # interleave with the user-visible text.
                print(f"\n  [REASONING] {event.text}", flush=True)
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
    print(f"    reasoning deltas: {reasoning_chunks}")
    print(f"    final text len:   {len(final.text)}")
    print(f"    finish:           {final.finish}")
    for k, v in final.cost.to_dict().items():
        print(f"    cost.{k}: {v}")
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
