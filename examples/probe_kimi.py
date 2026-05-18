"""Live probe — :class:`KimiRuntime`.

Single-turn execute against Moonshot's Kimi API via the kimi-agent-sdk.
Run with:

    KIMI_API_KEY=sk-... uv run python examples/probe_kimi.py

The probe exercises the Iteration-B surface: ``runtime.session()`` →
``execute()`` → cost telemetry. ``schema=`` is deliberately omitted —
structured output lands in Iteration D via the MCP-based forced-tool
path; running this probe with ``schema=`` would surface a clear
:class:`NotImplementedError`.

**Important install note.** The Kimi adapter conflicts with the Claude
adapter at the ``mcp`` dep boundary (``kimi-cli`` 1.12 pins
``fastmcp 2.12.5`` which requires ``mcp<1.17``; ``claude-agent-sdk``
requires ``mcp>=1.23``). Install the Kimi extra in a fresh venv that
*doesn't* have the Claude extra installed:

    python3.12 -m venv .venv-kimi
    .venv-kimi/bin/pip install -U pip
    .venv-kimi/bin/pip install 'airframe-agents[kimi]'
    .venv-kimi/bin/python examples/probe_kimi.py

Until Moonshot publishes a ``kimi-agent-sdk`` that widens the
``kimi-cli`` range, that's the only way to run this live.
"""

from __future__ import annotations

import asyncio
import os
import sys

from airframe import KimiRuntime, ProviderModel


async def main() -> None:
    if not os.environ.get("KIMI_API_KEY"):
        print(
            "KIMI_API_KEY not set. Mint a key at "
            "https://platform.moonshot.ai/console/api-keys and try again.",
            file=sys.stderr,
        )
        sys.exit(2)

    runtime = KimiRuntime()
    try:
        session = runtime.session(
            system="You are a precise, concise assistant.",
            model=ProviderModel("kimi", "kimi-k2-thinking-turbo"),
        )
        try:
            result = await session.execute("In one sentence: what is airframe-agents?")
            print(f"\n--- text ---\n{result.text}\n")
            print("--- cost ---")
            print(f"provider:        {result.cost.provider_id}")
            print(f"model:           {result.cost.model_id}")
            print(f"input tokens:    {result.cost.input_tokens}")
            print(f"output tokens:   {result.cost.output_tokens}")
            print(f"cost_usd:        {result.cost.cost_usd}  # populated in Iteration E")
            print(f"finish:          {result.cost.finish}")
            print(f"session id:      {session.id}")
        finally:
            await session.close()
    finally:
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
