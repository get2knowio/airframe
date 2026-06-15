"""Live probe: enumerate models across every installed adapter.

This script demonstrates the v0.2.0 discovery API end-to-end against
real vendor endpoints. It walks :func:`airframe.list_providers`,
constructs each adapter via :func:`airframe.runtime_for`, and prints
the live ``list_models()`` response — the same call a UI would make
to populate a "provider → model" pulldown.

Skipped automatically by pytest because the filename starts with
``probe_`` (the suite collects ``test_*.py`` only). Run manually::

    uv run python tests/probe_list_models.py
    uv run python tests/probe_list_models.py --provider claude
    uv run python tests/probe_list_models.py --installed-only=false

Auth requirements (each adapter raises ``RuntimeAuthError`` if its
credentials are missing):

* ``claude`` — ``ANTHROPIC_API_KEY`` env var (OAuth tokens don't work
  for ``/v1/models``).
* ``github-copilot`` — ``gh auth login`` or ``GITHUB_TOKEN``.
* ``opencode`` — ``OPENCODE_API_KEY`` or ``opencode auth login opencode-go``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from airframe import list_providers, runtime_for
from airframe.errors import (
    AgentRuntimeError,
    RuntimeAuthError,
    RuntimeTransientError,
)


async def _probe_one(provider_id: str) -> int:
    """Probe one provider; return 0 on success, 1 on adapter failure."""
    print(f"\n=== {provider_id} ===")
    try:
        rt_cls = runtime_for(provider_id)
    except (ValueError, ImportError) as exc:
        print(f"  dispatch failed: {exc}")
        return 1

    runtime = rt_cls()
    try:
        try:
            models = await runtime.list_models()
        except RuntimeAuthError as exc:
            print(f"  auth: {exc}")
            return 1
        except RuntimeTransientError as exc:
            print(f"  transient: {exc}")
            return 1
        except AgentRuntimeError as exc:
            print(f"  runtime error: {exc}")
            return 1

        if not models:
            print("  (no models returned)")
            return 0

        # Width-aware columns for a tidy menu.
        id_w = max(len(m.id) for m in models)
        name_w = max(len(m.display_name) for m in models)
        for m in models:
            ctx = f"{m.context_window:>8,}" if m.context_window else " " * 8
            price_in = (
                f"${m.pricing_input_per_1k_usd:.4f}"
                if m.pricing_input_per_1k_usd is not None
                else "—"
            )
            price_out = (
                f"${m.pricing_output_per_1k_usd:.4f}"
                if m.pricing_output_per_1k_usd is not None
                else "—"
            )
            caps = ",".join(sorted(m.capabilities)) if m.capabilities else "—"
            print(
                f"  {m.id:<{id_w}}  {m.display_name:<{name_w}}  "
                f"ctx={ctx}  in={price_in:<8}  out={price_out:<8}  caps={caps}"
            )
        print(f"  ({len(models)} model{'s' if len(models) != 1 else ''})")
        return 0
    finally:
        await runtime.close()


async def _run(provider: str | None, installed_only: bool) -> int:
    if provider is not None:
        return await _probe_one(provider)

    providers = list_providers(installed_only=installed_only)
    if not providers:
        scope = "installed" if installed_only else "known"
        print(f"No {scope} providers — run `pip install airframe-agents[all]`.")
        return 1

    print(f"Providers: {providers}")
    failures = 0
    for pid in providers:
        failures += await _probe_one(pid)
    return min(failures, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        help="Probe just one provider ID (skip discovery).",
    )
    parser.add_argument(
        "--installed-only",
        type=lambda v: v.lower() not in ("false", "0", "no"),
        default=True,
        help="Filter to providers whose SDK is installed (default: true).",
    )
    args = parser.parse_args()
    return asyncio.run(_run(args.provider, args.installed_only))


if __name__ == "__main__":
    sys.exit(main())
