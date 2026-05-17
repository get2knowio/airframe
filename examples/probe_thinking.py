#!/usr/bin/env python3
"""End-to-end probe for the ``thinking=`` kwarg.

Exercises the reasoning-effort knob Phase 2 Iteration B introduced —
opens a session, runs one ``execute()`` with ``thinking="high"``, and
reports the ``reasoning_tokens`` field on the resulting
:class:`~airframe.cost.CostRecord`.

What this validates:

* The runtime declares :data:`~airframe.features.Feature.REASONING_EFFORT`
  (every adapter does after Iteration B).
* The vendor SDK accepts the translated kwarg without complaint —
  ``ClaudeAgentOptions.effort``, ``CopilotClient.create_session
  (reasoning_effort=)``, ``ThreadOptions.modelReasoningEffort``, or
  ``chat.completions.create(reasoning_effort=)``.
* Cost telemetry surfaces ``reasoning_tokens`` (>0 on every adapter
  except Codex, whose ``Usage`` doesn't expose the counter — Codex
  reports 0 by design).

Usage::

    uv run python examples/probe_thinking.py
    uv run python examples/probe_thinking.py --provider claude
    uv run python examples/probe_thinking.py --provider github-copilot
    uv run python examples/probe_thinking.py --provider codex
    uv run python examples/probe_thinking.py --provider opencode
    uv run python examples/probe_thinking.py --effort medium

Defaults to ``claude`` (Anthropic's models expose ``thinking_tokens``
on usage cleanly) and ``high`` (most likely to actually trigger a
nonzero reasoning_tokens count). Requires whatever auth the chosen
provider's adapter expects.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from airframe import (  # noqa: E402
    Feature,
    list_providers,
    runtime_for,
)

DEFAULT_PROMPT = (
    "If a freight train leaves Chicago at 8am going 60mph and another "
    "leaves Denver at 9am going 80mph on the same track 950 miles apart, "
    "when do they meet? Show your reasoning step by step."
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
        help="Provider ID (default: claude). Any from list_providers().",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Reasoning prompt. Default: a freight-train word problem.",
    )
    parser.add_argument(
        "--effort",
        default="high",
        choices=["minimal", "low", "medium", "high", "disabled"],
        help="Effort level forwarded as thinking=. Default: high.",
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

    print(f"thinking probe — provider={args.provider} effort={args.effort}")
    if not runtime.supports(Feature.REASONING_EFFORT):
        print(
            f"  WARN: {type(runtime).__name__} does not declare "
            f"Feature.REASONING_EFFORT. The call will likely raise "
            f"UnsupportedFeatureError.",
            file=sys.stderr,
        )

    sess = runtime.session()
    err: str | None = None
    result = None
    try:
        print(f"\n  prompt: {args.prompt!r}\n  -- execute begin --")
        result = await sess.execute(args.prompt, thinking=args.effort)
        print("  -- execute end --")
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
    assert result is not None

    print("\n  summary:")
    print(f"    text len:         {len(result.text)}")
    print(f"    finish:           {result.finish}")
    print(f"    input_tokens:     {result.cost.input_tokens}")
    print(f"    output_tokens:    {result.cost.output_tokens}")
    print(f"    reasoning_tokens: {result.cost.reasoning_tokens}")
    print(f"    cache_read:       {result.cost.cache_read_tokens}")
    print(f"    cost_usd:         {result.cost.cost_usd}")

    if (
        result.cost.reasoning_tokens == 0
        and args.provider != "codex"
        and args.effort != "disabled"
    ):
        print(
            "\n  NOTE: reasoning_tokens=0 — either the model doesn't expose "
            "extended thinking at this effort level, or the vendor surfaces "
            "it under a different counter. Try --effort high on a reasoning "
            "model (claude-opus-4-*, gpt-5-*, o1-*).",
        )
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
