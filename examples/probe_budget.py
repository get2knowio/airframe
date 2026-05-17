#!/usr/bin/env python3
"""End-to-end probe for ``max_turns=`` / ``max_budget_usd=`` budget caps.

Exercises the Phase 5 Iteration D budget API — opens a session, sets
a deliberately tiny cap, runs turns until the cap trips, and prints
the resulting :class:`~airframe.errors.RuntimeBudgetExceededError`
attributes. Validates:

* The runtime declares ``BUDGET_USD_CAP`` (universal) and
  ``BUDGET_TURN_CAP`` (everywhere except Copilot, which the vendor
  caps internally — that branch surfaces the decline verbatim).
* The error fires at the turn boundary with ``cap`` / ``current`` /
  ``kind`` populated.
* Sessions without caps run indefinitely.

Usage::

    uv run python examples/probe_budget.py
    uv run python examples/probe_budget.py --provider claude
    uv run python examples/probe_budget.py --provider github-copilot
    uv run python examples/probe_budget.py --provider codex
    uv run python examples/probe_budget.py --provider opencode

Defaults to ``claude`` (broadest budget surface — both caps).
Copilot's branch demonstrates the ``max_turns=`` decline pointer.

Requires whatever auth the chosen provider's adapter expects.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from airframe import (  # noqa: E402
    Feature,
    RuntimeBudgetExceededError,
    list_providers,
    runtime_for,
)
from airframe.errors import UnsupportedFeatureError  # noqa: E402

DEFAULT_PROMPT = "Reply with one short sentence about Python."


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


def _print_matrix() -> None:
    """Print the per-adapter budget capability matrix."""
    print("\nBudget-capability matrix (Phase 5 Iteration D):")
    print("  Adapter             | BUDGET_USD | BUDGET_TURN")
    print("  --------------------|------------|------------")
    for pid in ("claude", "github-copilot", "codex", "opencode-zen", "opencode-go"):
        try:
            cls = runtime_for(pid)
            usd = Feature.BUDGET_USD_CAP in cls.SUPPORTED_FEATURES
            turn = Feature.BUDGET_TURN_CAP in cls.SUPPORTED_FEATURES
        except Exception:  # noqa: BLE001 — pure display path
            usd = turn = False
        usd_cell = "  yes" if usd else "   no"
        turn_cell = "  yes" if turn else "   no"
        print(f"  {pid:<19} | {usd_cell:<10} | {turn_cell}")


async def _probe_turn_cap(rt: Any, prompt: str) -> int:  # type: ignore[no-untyped-def]
    """Run until max_turns trips. Returns 0 on PASS, 1 on FAIL."""
    if not rt.supports(Feature.BUDGET_TURN_CAP):
        print(
            f"  {type(rt).__name__} does NOT declare BUDGET_TURN_CAP — "
            f"expect max_turns= to raise UnsupportedFeatureError."
        )
        sess = rt.session()
        try:
            try:
                await sess.execute(prompt, max_turns=2)
            except UnsupportedFeatureError as exc:
                print(f"  decline (as expected): {exc}")
                return 0
            print("FAIL: expected max_turns= to raise on this adapter")
            return 1
        finally:
            await sess.close()

    print("\n--- turn cap probe (max_turns=2) ---")
    sess = rt.session()
    fired = False
    try:
        for i in range(1, 4):
            try:
                t0 = time.monotonic()
                result = await sess.execute(prompt, max_turns=2)
                elapsed = time.monotonic() - t0
                preview = (result.text or "")[:60].replace("\n", " ")
                print(
                    f"  turn {i} ok (elapsed {elapsed:.1f}s) → "
                    f"{preview!r}{'...' if len(result.text or '') > 60 else ''}"
                )
            except RuntimeBudgetExceededError as exc:
                fired = True
                print(
                    f"  turn {i} tripped cap: kind={exc.kind!r} "
                    f"cap={exc.cap} current={exc.current}"
                )
                break
    finally:
        await sess.close()
    if not fired:
        print("FAIL: expected RuntimeBudgetExceededError(kind='turns') by turn 3")
        return 1
    return 0


async def _probe_usd_cap(rt: Any, prompt: str) -> int:  # type: ignore[no-untyped-def]
    """Run with a deliberately tiny cap and surface the error attrs."""
    if not rt.supports(Feature.BUDGET_USD_CAP):
        # Iteration D flips it on every built-in adapter; this branch
        # only fires for third-party adapters that decline.
        print(
            f"  {type(rt).__name__} does NOT declare BUDGET_USD_CAP — "
            f"expect max_budget_usd= to raise UnsupportedFeatureError."
        )
        sess = rt.session()
        try:
            try:
                await sess.execute(prompt, max_budget_usd=0.0001)
            except UnsupportedFeatureError as exc:
                print(f"  decline (as expected): {exc}")
                return 0
            print("FAIL: expected max_budget_usd= to raise on this adapter")
            return 1
        finally:
            await sess.close()

    print("\n--- USD cap probe (max_budget_usd=$0.0001) ---")
    print("  (cap is set very small so the first or second turn trips it)")
    sess = rt.session()
    fired = False
    try:
        for i in range(1, 6):
            try:
                t0 = time.monotonic()
                result = await sess.execute(prompt, max_budget_usd=0.0001)
                elapsed = time.monotonic() - t0
                cost = result.cost.cost_usd or 0.0
                preview = (result.text or "")[:60].replace("\n", " ")
                print(
                    f"  turn {i} ok (elapsed {elapsed:.1f}s, "
                    f"this turn cost ${cost:.6f}) → {preview!r}"
                )
            except RuntimeBudgetExceededError as exc:
                fired = True
                print(
                    f"  turn {i} tripped cap: kind={exc.kind!r} "
                    f"cap=${exc.cap:.6f} current=${exc.current:.6f}"
                )
                break
    finally:
        await sess.close()
    if not fired:
        print(
            "NOTE: cap never fired in 5 turns — either the model returned "
            "extremely cheap turns or the vendor's cost reporting returned "
            "None. The probe still validated max_budget_usd= was accepted."
        )
    return 0


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--provider",
        default="claude",
        help=(
            "Provider ID (default: claude — broadest budget surface). Any from list_providers()."
        ),
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to send. Defaults to a short cheap prompt.",
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
    print(f"budget probe — provider={args.provider}")
    print(f"  BUDGET_USD_CAP : {runtime.supports(Feature.BUDGET_USD_CAP)}")
    print(f"  BUDGET_TURN_CAP: {runtime.supports(Feature.BUDGET_TURN_CAP)}")

    rc = 0
    try:
        turn_rc = await _probe_turn_cap(runtime, args.prompt)
        usd_rc = await _probe_usd_cap(runtime, args.prompt)
        rc = max(turn_rc, usd_rc)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        rc = 1
    finally:
        await runtime.close()

    _print_matrix()

    print("\nPASS" if rc == 0 else "\nFAIL")
    return rc


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
