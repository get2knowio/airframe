#!/usr/bin/env python3
"""Live probe for ``zai-anthropic`` — verifies the unverified feature flags.

``ZaiAnthropicRuntime`` ships with several capabilities declared ``False``
because they depend on what Z.AI's Anthropic-compatible endpoint actually
implements, not on the local ``claude`` CLI. This probe checks each one
against a real key so the flags can be promoted on evidence rather than
optimism.

Requires:

* ``ZAI_API_KEY`` — a real Z.AI key.
* The ``claude`` CLI on ``PATH`` (the Agent SDK spawns it).
* ``pip install airframe-agents[claude]``.

Usage::

    ZAI_API_KEY=... uv run python examples/probe_zai.py
    ZAI_API_KEY=... uv run python examples/probe_zai.py --model glm-4.5-air

Each check prints ``PASS`` / ``FAIL`` / ``SKIP`` and a one-line reason.
A ``PASS`` on a currently-declined feature is the signal to remove it
from ``_UNVERIFIED_FEATURES`` in ``src/airframe/adapters/zai.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from pydantic import BaseModel  # noqa: E402

from airframe.adapters.zai import ZaiAnthropicRuntime  # noqa: E402
from airframe.features import Feature  # noqa: E402


class _Answer(BaseModel):
    """Trivial schema for the structured-output check."""

    capital: str
    country: str


def _report(name: str, ok: bool | None, detail: str) -> None:
    status = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
    print(f"  [{status}] {name}: {detail}")


async def _check_plain_text(rt: ZaiAnthropicRuntime) -> None:
    """Baseline — if this fails nothing else is meaningful."""
    try:
        result = await rt.execute("Reply with exactly the word: pong")
        ok = bool(result.text and result.text.strip())
        _report("plain text execute", ok, repr((result.text or "")[:60]))
        cost = result.cost
        _report(
            "cost telemetry",
            cost.cost_usd is not None or bool(cost.input_tokens or cost.output_tokens),
            f"cost_usd={cost.cost_usd} in={cost.input_tokens} out={cost.output_tokens}",
        )
    except Exception as exc:  # noqa: BLE001 - a probe reports, never crashes
        _report("plain text execute", False, f"{type(exc).__name__}: {exc}")


async def _check_structured_output(rt: ZaiAnthropicRuntime) -> None:
    """The conformance floor. A FAIL here blocks shipping the adapter."""
    try:
        result = await rt.execute(
            "What is the capital of France?",
            schema=_Answer,
        )
        ok = isinstance(result.structured, _Answer)
        _report(
            "structured output (--json-schema)",
            ok,
            f"structured={result.structured!r}",
        )
        if not ok:
            print(
                "         ^ this is the conformance floor "
                "(test_supports_structured_output_json_schema_is_true).\n"
                "           A FAIL means zai-anthropic cannot ship as a conforming\n"
                "           adapter without a client-side schema fallback."
            )
    except Exception as exc:  # noqa: BLE001
        _report("structured output (--json-schema)", False, f"{type(exc).__name__}: {exc}")


async def _check_streaming(rt: ZaiAnthropicRuntime) -> None:
    try:
        sess = rt.session()
        try:
            deltas = 0
            async for _event in sess.stream("Count from one to five."):
                deltas += 1
            _report("streaming", deltas > 0, f"{deltas} events")
        finally:
            await sess.close()
    except Exception as exc:  # noqa: BLE001
        _report("streaming", False, f"{type(exc).__name__}: {exc}")


async def _check_thinking(rt: ZaiAnthropicRuntime) -> None:
    """Currently declined — GLM reasons, but does the endpoint expose it?"""
    from airframe.errors import UnsupportedFeatureError

    try:
        sess = rt.session()
        try:
            result = await sess.execute("Think briefly, then say 'ok'.", thinking="medium")
            _report(
                "thinking / reasoning effort",
                True,
                f"accepted; reasoning={(result.reasoning or '')[:40]!r}",
            )
        finally:
            await sess.close()
    except UnsupportedFeatureError:
        _report(
            "thinking / reasoning effort",
            None,
            "declined by airframe's capability gate (expected while unverified) — "
            "temporarily add REASONING_EFFORT to SUPPORTED_FEATURES to test for real",
        )
    except Exception as exc:  # noqa: BLE001
        _report("thinking / reasoning effort", False, f"{type(exc).__name__}: {exc}")


async def _check_session_resume(rt: ZaiAnthropicRuntime) -> None:
    try:
        sess = rt.session()
        try:
            await sess.execute("Remember the number 42. Reply 'ok'.")
            second = await sess.execute("What number did I ask you to remember?")
            ok = "42" in (second.text or "")
            _report("session continuity", ok, repr((second.text or "")[:60]))
        finally:
            await sess.close()
    except Exception as exc:  # noqa: BLE001
        _report("session continuity", False, f"{type(exc).__name__}: {exc}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", help="GLM model id (default: the adapter's default).")
    parser.add_argument("--base-url", help="Override the Z.AI base URL.")
    args = parser.parse_args()

    if not os.environ.get("ZAI_API_KEY"):
        print("ZAI_API_KEY is not set — nothing to probe.", file=sys.stderr)
        return 1

    rt = ZaiAnthropicRuntime(model=args.model, base_url=args.base_url)
    print(f"Probing {rt.PROVIDER_ID} at {rt._base_url} with model {rt._default_model}\n")

    declined = sorted(
        f.name for f in Feature if not rt.supports(f) and f.name not in {"PROMPT_CACHE_CONTROL"}
    )
    print(f"Declared unsupported ({len(declined)}): {', '.join(declined)}\n")

    print("Live checks:")
    await _check_plain_text(rt)
    await _check_structured_output(rt)
    await _check_streaming(rt)
    await _check_session_resume(rt)
    await _check_thinking(rt)

    print(
        "\nPromote any PASS above out of _UNVERIFIED_FEATURES in "
        "src/airframe/adapters/zai.py,\nthen re-run `make ci` — the conformance "
        "suite asserts the opposite branch\nfor every flag you flip."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
