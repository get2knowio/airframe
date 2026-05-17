#!/usr/bin/env python3
"""Multi-provider parity probe.

Runs the *same* typed-output prompt against every installed adapter
via the discovery API — no per-vendor imports, no per-vendor
conditionals. Drives home the point that consumer code stays
identical regardless of which provider it ends up calling.

    uv run python examples/probe_parity.py
    uv run python examples/probe_parity.py --providers claude,codex
    AIRFRAME_PROBE_MODEL_CODEX=gpt-5.5 uv run python examples/probe_parity.py

Outcomes per provider:

* ``PASS`` — structured payload returned with the right shape.
* ``SKIP`` — :class:`RuntimeAuthError`; provider installed but no
  credentials on this machine. Expected for adapters you don't use.
* ``FAIL`` — anything else; vendor / network / classification bug.

Exit code is 0 unless at least one provider raised a non-auth
failure. Skips are non-fatal — having creds for only a subset of
adapters is the normal case.

Per-provider model override (handy when a default isn't accessible
on your auth path, e.g. ``gpt-5-codex`` on ChatGPT-account auth):
set ``AIRFRAME_PROBE_MODEL_<PROVIDER>`` where ``<PROVIDER>`` is the
upper-cased provider ID with hyphens swapped for underscores
(``AIRFRAME_PROBE_MODEL_GITHUB_COPILOT`` etc.).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from pydantic import BaseModel  # noqa: E402

from airframe import ProviderModel, list_providers, runtime_for  # noqa: E402
from airframe.errors import RuntimeAuthError  # noqa: E402


class Answer(BaseModel):
    answer: int
    rationale: str


PROMPT = "What is 17 + 25? Reply with answer and a short rationale."


def _model_override(provider_id: str) -> str | None:
    env_key = "AIRFRAME_PROBE_MODEL_" + provider_id.upper().replace("-", "_")
    return os.environ.get(env_key)


async def probe_one(provider_id: str) -> tuple[str, str]:
    """Run the probe for one provider. Returns (status, summary line)."""
    try:
        cls = runtime_for(provider_id)
    except ImportError as exc:
        return "skip", f"adapter not installed: {exc}"
    except ValueError as exc:
        return "fail", f"unknown provider: {exc}"

    runtime = cls()
    model_id = _model_override(provider_id)
    kwargs: dict = {"schema": Answer}
    if model_id is not None:
        kwargs["model"] = ProviderModel(provider_id, model_id)

    t0 = time.monotonic()
    try:
        result = await runtime.execute(PROMPT, **kwargs)
        elapsed = time.monotonic() - t0
        answer = (result.structured or {}).get("answer")
        cost_str = (
            f"${result.cost.cost_usd:.4f}" if result.cost.cost_usd is not None else "cost=n/a"
        )
        summary = (
            f"answer={answer} via {result.cost.provider_id}/{result.cost.model_id} "
            f"({elapsed:.1f}s, {cost_str})"
        )
        if answer != 42:
            return "fail", f"wrong answer: {summary}"
        return "pass", summary
    except RuntimeAuthError as exc:
        return "skip", f"no creds: {exc}"
    except Exception as exc:  # noqa: BLE001
        return "fail", f"{type(exc).__name__}: {exc}"
    finally:
        await runtime.close()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--providers",
        default="all",
        help='Comma-separated provider IDs to probe (default: "all" = every installed).',
    )
    args = parser.parse_args()

    installed = list_providers()
    if args.providers == "all":
        targets = installed
    else:
        requested = [p.strip() for p in args.providers.split(",") if p.strip()]
        targets = [p for p in requested if p in installed]
        missing = [p for p in requested if p not in installed]
        if missing:
            print(f"WARN: skipping providers without installed adapter: {missing}")

    if not targets:
        print("No installed providers to probe. Try `pip install airframe-agents[all]`.")
        return 1

    print(
        f"Probing {len(targets)} provider(s) via runtime_for(pid)() — "
        f"same prompt, same schema, every adapter:\n  {targets}\n"
    )

    results: dict[str, tuple[str, str]] = {}
    for pid in targets:
        status, summary = await probe_one(pid)
        marker = {"pass": "PASS", "skip": "SKIP", "fail": "FAIL"}[status]
        print(f"  [{pid:<16}] {marker}  {summary}")
        results[pid] = (status, summary)

    # --- Summary ---------------------------------------------------------------
    passes = sum(1 for s, _ in results.values() if s == "pass")
    skips = sum(1 for s, _ in results.values() if s == "skip")
    fails = sum(1 for s, _ in results.values() if s == "fail")

    print(
        f"\n{passes}/{len(results)} providers returned identical structured "
        f"output via the same code path."
    )
    if skips:
        print(f"  ({skips} skipped — no credentials available)")
    if fails:
        print(f"  ({fails} failed — see lines above)")

    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
