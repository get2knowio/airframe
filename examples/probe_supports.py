#!/usr/bin/env python3
"""Capability matrix probe — what every adapter declares it supports.

Prints a Feature × adapter matrix so consumers and contributors can
see at a glance which adapter exposes which capabilities. No network
calls; no auth required; pure ``supports()`` lookups.

Usage::

    uv run python examples/probe_supports.py
    uv run python examples/probe_supports.py --installed-only
    uv run python examples/probe_supports.py --provider claude

The matrix should match what's in ``docs/capabilities.md`` (and
the README summary table). After Phase 5 every adapter declares
the full feature set with vendor-specific declines documented
per-adapter in ``docs/adapters/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from airframe import Feature, list_providers, runtime_for  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--installed-only",
        action="store_true",
        help="Only show adapters whose vendor SDK is installed. Default: show all.",
    )
    parser.add_argument(
        "--provider",
        help="Limit output to a single provider id (e.g. ``claude``).",
    )
    args = parser.parse_args()

    provider_ids = list_providers(installed_only=args.installed_only)
    if args.provider:
        if args.provider not in provider_ids:
            print(
                f"Provider {args.provider!r} not found. Known: {provider_ids}",
                file=sys.stderr,
            )
            return 1
        provider_ids = [args.provider]

    if not provider_ids:
        print(
            "No providers available. Install at least one extra: "
            "pip install airframe-agents[claude|copilot|codex|openai-compat|all]",
            file=sys.stderr,
        )
        return 1

    # Instantiate each runtime. Construction defers SDK / auth to first
    # execute() so it's safe without credentials.
    runtimes: list[tuple[str, object]] = []
    for provider_id in provider_ids:
        cls = runtime_for(provider_id)
        # OpenAICompatibleRuntime subclasses require api_key= to construct,
        # so route around that for the probe.
        try:
            runtime = cls()
        except TypeError:
            runtime = cls(api_key="dummy-for-supports-probe")  # type: ignore[call-arg]
        runtimes.append((provider_id, runtime))

    # Pretty-print as Feature × adapter matrix.
    name_w = max(len(f.name) for f in Feature) + 2
    col_w = max(len(p) for p, _ in runtimes) + 2

    header = "Feature".ljust(name_w) + "".join(p.center(col_w) for p, _ in runtimes)
    print(header)
    print("-" * len(header))
    for feature in Feature:
        row = feature.name.ljust(name_w)
        for _provider, runtime in runtimes:
            mark = "✓" if runtime.supports(feature) else "·"  # type: ignore[attr-defined]
            row += mark.center(col_w)
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
