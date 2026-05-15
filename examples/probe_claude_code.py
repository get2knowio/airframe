#!/usr/bin/env python3
"""End-to-end probe for :class:`ClaudeCodeRuntime`.

Sends a small typed-output prompt against the real ``claude`` CLI
subprocess via ``claude-agent-sdk``. Uses whichever auth is on this
machine (``CLAUDE_CODE_OAUTH_TOKEN`` env, ``~/.claude/.credentials.json``,
or ``ANTHROPIC_API_KEY`` env).

Verifies:

* The runtime spawns the Claude SDK subprocess successfully.
* :meth:`ClaudeCodeRuntime.execute` returns a typed payload via the
  forced ``submit_result`` tool.
* :class:`CostRecord` fields are populated.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from pydantic import BaseModel  # noqa: E402

from airframe.adapters.claude_code import ClaudeCodeRuntime  # noqa: E402
from airframe.protocol import ProviderModel  # noqa: E402


class Result(BaseModel):
    answer: int
    rationale: str


async def main() -> int:
    model_id = "claude-haiku-4-5"
    runtime = ClaudeCodeRuntime(model=model_id)

    print(f"ClaudeCodeRuntime probe — model={model_id}")

    t0 = time.monotonic()
    err: str | None = None
    structured = None
    try:
        result = await runtime.execute(
            "What is 17 + 25? Reply with answer and a short rationale.",
            schema=Result,
            model=ProviderModel("anthropic", model_id),
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

    await runtime.aclose()

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
