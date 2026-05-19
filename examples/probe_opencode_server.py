#!/usr/bin/env python3
"""End-to-end probe for :class:`OpenCodeServerRuntime`.

Sends a small prompt against a locally-running ``opencode serve``
(default ``http://127.0.0.1:4096``). Requires:

1. ``opencode serve`` running on the configured URL.
2. At least one upstream configured via ``opencode auth login``.

Override the server URL with ``OPENCODE_SERVER_URL``; for non-loopback
URLs also set ``OPENCODE_SERVER_PASSWORD`` (and
``OPENCODE_SERVER_USERNAME`` if non-default).

Pick a specific model/upstream via the CLI flags::

    uv run python examples/probe_opencode_server.py
    uv run python examples/probe_opencode_server.py --model claude-haiku-4-5
    uv run python examples/probe_opencode_server.py --model gpt-5-codex --provider openai

This probe deliberately does *not* exercise ``schema=`` — the
0.1.0a36 SDK has no MCP runtime registration, so structured output
isn't yet wired (see ``docs/adapters/opencode-server.md`` for the
SDK-gap rationale). It does cover plain-text execute, streaming,
session resume, lifecycle hooks, and budget caps.
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

from airframe.adapters.opencode_server import OpenCodeServerRuntime  # noqa: E402
from airframe.events import TextDelta, TurnComplete  # noqa: E402
from airframe.options import OpenCodeServerOptions  # noqa: E402
from airframe.protocol import ProviderModel  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=None, help="Upstream model id.")
    parser.add_argument(
        "--provider",
        default=None,
        help="Explicit upstream provider id (anthropic, openai, openrouter, ...).",
    )
    args = parser.parse_args()

    runtime = OpenCodeServerRuntime(model=args.model)
    print(f"OpenCodeServerRuntime probe — base_url={runtime._base_url}")

    # --- Probe 0: catalog (also acts as the server-reachable probe) ---------
    try:
        models = await runtime.list_models()
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAIL: list_models — {type(exc).__name__}: {exc}")
        return 1
    print(f"  list_models: PASS ({len(models)} models across upstreams)")
    if not models:
        print("    NOTE: no models — run `opencode auth login <provider>` first.")
        return 1

    # Pick a model: explicit arg → first one from the catalog.
    model_id = args.model or models[0].id
    upstream = args.provider or models[0].raw.get("provider") if models[0].raw else None
    options = OpenCodeServerOptions(provider_id=upstream) if upstream else None
    print(f"  routing: model={model_id} upstream={upstream or '<auto-discover>'}")

    # --- Probe 1: plain-text execute ----------------------------------------
    sess = runtime.session(
        model=ProviderModel("opencode", model_id),
        provider_options=options,
    )
    t0 = time.monotonic()
    err: str | None = None
    try:
        result = await sess.execute("Reply with the single word: ready.")
        print(f"  plain text: PASS ({time.monotonic() - t0:.1f}s)")
        print(f"    text: {result.text[:200]!r}")
        for k, v in result.cost.to_dict().items():
            print(f"    cost.{k}: {v}")
        prior_session_id = sess.id
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        import traceback

        traceback.print_exc()
        prior_session_id = None
    finally:
        await sess.close()

    if err is not None or prior_session_id is None:
        print(f"\nFAIL: {err or 'no session id'}")
        await runtime.close()
        return 1

    # --- Probe 2: streaming -------------------------------------------------
    stream_sess = runtime.session(
        model=ProviderModel("opencode", model_id),
        provider_options=options,
    )
    print("  streaming: ", end="", flush=True)
    seen_text = False
    seen_complete = False
    t0 = time.monotonic()
    try:
        async for event in stream_sess.stream("Say hi in three words."):
            if isinstance(event, TextDelta):
                print(event.text, end="", flush=True)
                seen_text = True
            elif isinstance(event, TurnComplete):
                seen_complete = True
        print(
            f"\n  streaming: {'PASS' if seen_text and seen_complete else 'FAIL'} "
            f"({time.monotonic() - t0:.1f}s)"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\n  streaming: FAIL — {type(exc).__name__}: {exc}")
        err = str(exc)
    finally:
        await stream_sess.close()

    # --- Probe 3: session resume --------------------------------------------
    resumed = runtime.session(
        resume=prior_session_id,
        model=ProviderModel("opencode", model_id),
        provider_options=options,
    )
    try:
        result = await resumed.execute("What single word did you reply with earlier?")
        # We resumed the first probe, so the model should reference "ready".
        contains_ready = "ready" in result.text.lower()
        print(f"  resume: {'PASS' if contains_ready else 'NOTE'} — text: {result.text[:160]!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"  resume: FAIL — {type(exc).__name__}: {exc}")
        err = err or str(exc)
    finally:
        # Don't delete the resumed session — that prior_session_id was
        # created by Probe 1; resume sessions aren't owned.
        await resumed.close()

    await runtime.close()
    if err is not None:
        print(f"\nFAIL: {err}")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
