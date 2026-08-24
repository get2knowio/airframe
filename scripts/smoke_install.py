#!/usr/bin/env python3
"""Smoke-test an *installed* ``airframe-agents`` distribution.

Run against a clean environment that has the built wheel installed — not
against the source tree. The release workflow's ``install-matrix`` job runs
this once per (Python version x pip extra) combination against the exact
artifact it is about to publish.

The point is to exercise things the in-repo test suite structurally cannot:

* ``uv sync --all-extras --group dev`` installs the *source tree* with every
  vendor SDK present. A wheel that ships the wrong files, declares a runtime
  dependency that only the dev group was satisfying, or carries a broken
  extra passes that gate and fails on a user's machine.
* The lazy-SDK-import invariant is only meaningfully testable where the
  vendor SDKs are **absent**. In the ``no extras`` matrix row they are.

Deliberately imports nothing beyond the standard library and ``airframe``
itself — importing a vendor SDK here would defeat the check it performs.

Usage::

    .venv/bin/python scripts/smoke_install.py

Exits non-zero on the first failed check, printing what broke.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import subprocess
import sys

#: Vendor SDKs that ``import airframe`` must never pull in. Adapter modules
#: import these inside the method that needs them, gated on the optional
#: extra being installed; a module-level import would make every consumer
#: pay for every vendor.
VENDOR_SDKS = (
    "claude_agent_sdk",
    "anthropic",
    "openai",
    "aioboto3",
    "botocore",
    "opencode_ai",
    "copilot",
    "tiktoken",
)

_failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record one check; print immediately so CI logs stay readable."""
    status = "ok  " if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        _failures.append(f"{name}: {detail}" if detail else name)


def main() -> int:
    print("Smoke-testing the installed airframe-agents distribution\n")
    print(f"  python: {sys.version.split()[0]}")
    print(f"  exe:    {sys.executable}\n")

    # --- The distribution is actually installed under its PyPI name -------
    try:
        dist_version = importlib.metadata.version("airframe-agents")
    except importlib.metadata.PackageNotFoundError:
        print("  [FAIL] airframe-agents is not installed in this environment")
        return 1
    print(f"Distribution: airframe-agents {dist_version}\n")

    # --- Which extras made it in? Reported, not asserted: the matrix row --
    # --- decides, and the row's own name already records the intent. ------
    installed_sdks = [m for m in VENDOR_SDKS if importlib.util.find_spec(m) is not None]
    bare_install = not installed_sdks
    print(f"Vendor SDKs present: {installed_sdks or '(none — bare install)'}\n")

    print("Checks:")

    # --- The import itself ------------------------------------------------
    try:
        import airframe
    except Exception as exc:  # noqa: BLE001 - the whole point is to report it
        check("import airframe", False, f"{type(exc).__name__}: {exc}")
        return _summary()

    check("import airframe", True)

    # --- Lazy SDK imports -------------------------------------------------
    # Only decisive on the bare row, but harmless and still meaningful
    # elsewhere: a module-level vendor import would show up here too.
    leaked = sorted(m for m in VENDOR_SDKS if m in sys.modules)
    check(
        "import airframe pulls in no vendor SDK",
        not leaked,
        f"leaked {leaked}" if leaked else "",
    )

    # --- Version is single-sourced from metadata --------------------------
    check(
        "airframe.__version__ matches distribution metadata",
        getattr(airframe, "__version__", None) == dist_version,
        f"__version__={getattr(airframe, '__version__', None)!r} vs {dist_version!r}",
    )

    # --- Public surface ---------------------------------------------------
    missing = [name for name in getattr(airframe, "__all__", []) if not hasattr(airframe, name)]
    check(
        "every name in __all__ is importable",
        not missing,
        f"missing {missing}" if missing else f"{len(getattr(airframe, '__all__', []))} names",
    )

    # --- Discovery --------------------------------------------------------
    try:
        declared = set(airframe.list_providers(installed_only=False))
        available = set(airframe.list_providers())
    except Exception as exc:  # noqa: BLE001
        check("list_providers()", False, f"{type(exc).__name__}: {exc}")
        return _summary()

    # Deliberately not asserting an exact roster — that belongs to
    # tests/test_discovery.py, which runs on every PR and does not need a
    # release to catch a regression. What only an installed artifact can
    # prove is that discovery imports at all from the shipped files.
    check(
        "list_providers(installed_only=False) returns a non-empty roster",
        bool(declared),
        f"{len(declared)} providers: {sorted(declared)}",
    )
    check(
        "list_providers() is a subset of declared",
        available <= declared,
        f"available={sorted(available)}",
    )

    # The real packaging check, and the roster-independent one: every
    # adapter module the wheel claims to serve must actually be importable
    # from the wheel. A module dropped by a bad `packages =` or a stray
    # sdist exclude shows up here as an unexpected exception rather than
    # the documented ImportError-naming-an-extra.
    for provider_id in sorted(declared):
        try:
            airframe.runtime_for(provider_id)
            outcome = "class returned"
            ok = True
        except ImportError as exc:
            # Documented: SDK absent for this extra. Fine, and the message
            # is contractually required to name the extra to install.
            ok = "airframe-agents[" in str(exc)
            outcome = "ImportError names extra" if ok else str(exc)[:80]
        except Exception as exc:  # noqa: BLE001
            outcome = f"{type(exc).__name__}: {exc}"[:100]
            ok = False
        check(f"runtime_for({provider_id!r}) resolves from the wheel", ok, outcome)

    # A bare install has no vendor SDKs, so nothing is runnable. That is the
    # honest signal the discovery layer documents — and a package that
    # reports providers it cannot construct is worse than one reporting none.
    if bare_install:
        check(
            "bare install reports no runnable providers",
            available == set(),
            f"reported {sorted(available)} with no SDKs installed",
        )
    else:
        check(
            "installed extras surface at least one provider",
            bool(available),
            f"SDKs {installed_sdks} present but list_providers() is empty",
        )

    # --- Console script ---------------------------------------------------
    # The entry point is declared in [project.scripts]; a wheel can ship the
    # code and still get the script wiring wrong.
    proc = subprocess.run(
        [sys.executable, "-m", "airframe.cli", "providers"],
        capture_output=True,
        text=True,
    )
    check(
        "airframe.cli runs as a module",
        proc.returncode == 0,
        (proc.stderr or proc.stdout).strip()[:160],
    )

    return _summary()


def _summary() -> int:
    print()
    if _failures:
        print(f"{len(_failures)} check(s) failed:")
        for failure in _failures:
            print(f"  - {failure}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
