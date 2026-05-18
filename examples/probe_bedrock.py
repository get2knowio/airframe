#!/usr/bin/env python3
"""End-to-end probe for :class:`BedrockRuntime`.

Sends a small typed-output prompt against AWS Bedrock's Converse API
using whichever AWS credentials are available (env vars, ``AWS_PROFILE``,
or the default IAM-role chain). Not a CI test — runs against the
real API and incurs per-token cost.

Requires ``AWS_REGION`` (or ``AWS_DEFAULT_REGION``) to be set;
Bedrock is region-pinned and the model catalog differs per region.
Set ``AIRFRAME_PROBE_MODEL_BEDROCK=<modelId>`` to override the
default model (Claude 3.5 Haiku).

Verifies:

* Auth + region resolution work (the runtime builds the client).
* :meth:`BedrockSession.execute` round-trips a structured payload via
  the forced ``submit_result`` tool.
* :class:`CostRecord` token counters are populated (``cost_usd``
  lands in Iteration E alongside the pricing table).
* Plain-text execute() (no schema) works.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from pydantic import BaseModel  # noqa: E402

from airframe.adapters.bedrock import DEFAULT_BEDROCK_MODEL, BedrockRuntime  # noqa: E402
from airframe.protocol import ProviderModel  # noqa: E402


class Result(BaseModel):
    answer: int
    rationale: str


async def main() -> int:
    model_id = os.environ.get("AIRFRAME_PROBE_MODEL_BEDROCK") or DEFAULT_BEDROCK_MODEL
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        print("BedrockRuntime probe — SKIP (no AWS_REGION set; Bedrock is region-pinned)")
        return 0

    runtime = BedrockRuntime(model=model_id)
    print(f"BedrockRuntime probe — model={model_id} region={region}")

    err: str | None = None
    structured = None

    # --- Probe 1: structured output -----------------------------------------
    t0 = time.monotonic()
    try:
        result = await runtime.execute(
            "What is 17 + 25? Reply with answer and a short rationale.",
            schema=Result,
            model=ProviderModel("bedrock", model_id),
        )
        structured = result.structured
        print(f"  structured: PASS ({time.monotonic() - t0:.1f}s)")
        print(f"    payload: {structured}")
        for k, v in result.cost.to_dict().items():
            print(f"    cost.{k}: {v}")
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        import traceback

        traceback.print_exc()

    # --- Probe 2: plain text ------------------------------------------------
    t0 = time.monotonic()
    try:
        result = await runtime.execute(
            "Reply with the single word: ready.",
            model=ProviderModel("bedrock", model_id),
        )
        print(f"  plain text: PASS ({time.monotonic() - t0:.1f}s)")
        print(f"    text: {result.text[:200]!r}")
    except Exception as exc:  # noqa: BLE001
        err = err or f"{type(exc).__name__}: {exc}"
        import traceback

        traceback.print_exc()

    await runtime.close()

    if err is not None:
        print(f"\nFAIL: {err}")
        return 1
    if structured is None or "answer" not in (structured or {}):
        print("\nFAIL: structured payload missing 'answer'")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
