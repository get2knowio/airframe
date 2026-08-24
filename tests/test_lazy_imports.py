"""``import airframe`` must not drag in any vendor SDK.

CLAUDE.md states the invariant: adapter modules import their vendor SDK
*inside the method that needs it*, gated on the optional extra being
installed. A module-level import would make every consumer pay the import
cost of every vendor — and would break the bare
``pip install airframe-agents`` case outright, since the SDK would be
absent.

These tests run in a subprocess with a clean interpreter, because
``sys.modules`` in the pytest process is already polluted: the rest of the
suite imports the vendor SDKs deliberately, and the dev environment
installs all of them. Checking ``sys.modules`` in-process would either
pass vacuously or fail for reasons unrelated to airframe.

This is the PR-time counterpart to ``scripts/smoke_install.py``, which
performs the same check against a *built wheel* in an environment where
the SDKs are genuinely absent. Both are worth having: this one catches the
regression on the commit that introduces it; that one catches the case
where the packaging, rather than the code, is what went wrong.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

#: Vendor SDKs no airframe import path may pull in eagerly. Keep in sync
#: with the same list in ``scripts/smoke_install.py``.
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


def _imported_sdks_after(statement: str) -> list[str]:
    """Return the vendor SDKs present in ``sys.modules`` after ``statement``.

    Runs in a fresh interpreter so the result reflects only what
    ``statement`` caused to be imported.

    Args:
        statement: Python source executed before the check.

    Returns:
        Sorted vendor SDK module names found in ``sys.modules``.

    Raises:
        AssertionError: The subprocess itself failed, which means the
            statement could not even be executed.
    """
    probe = (
        f"{statement}\n"
        "import sys, json\n"
        f"print(json.dumps(sorted(m for m in {VENDOR_SDKS!r} if m in sys.modules)))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"probe failed to execute {statement!r}:\n{proc.stderr or proc.stdout}"
    )
    import json

    return list(json.loads(proc.stdout.strip().splitlines()[-1]))


def test_import_airframe_pulls_in_no_vendor_sdk() -> None:
    """The headline invariant — plain ``import airframe`` is cheap."""
    leaked = _imported_sdks_after("import airframe")
    assert not leaked, (
        f"`import airframe` eagerly imported {leaked}. Adapter modules must "
        f"import their vendor SDK inside the method that needs it, so a "
        f"consumer who installed no extras can still `import airframe`."
    )


def test_discovery_pulls_in_no_vendor_sdk() -> None:
    """``list_providers()`` inspects adapters without importing their SDKs.

    Discovery decides availability with ``importlib.util.find_spec`` on
    each adapter's ``REQUIRES_PACKAGE``. ``find_spec`` locates a module
    without executing it; switching to a real import would make the
    menu-building path as expensive as using every adapter at once.
    """
    leaked = _imported_sdks_after(
        "import airframe\nairframe.list_providers()\nairframe.list_providers(installed_only=False)"
    )
    assert not leaked, f"discovery eagerly imported {leaked}"


@pytest.mark.parametrize(
    "module",
    [
        "airframe.adapters.claude_code",
        "airframe.adapters.copilot",
        "airframe.adapters.bedrock",
        "airframe.adapters.opencode_server",
        "airframe.adapters.opencode_zen",
        "airframe.adapters.opencode_go",
        "airframe.adapters.openrouter",
        "airframe.adapters.openai_compatible",
    ],
)
def test_adapter_module_import_pulls_in_no_vendor_sdk(module: str) -> None:
    """Importing an adapter *module* is still free.

    Stronger than the top-level check and the one that actually regresses:
    a contributor adding ``import openai`` at the top of an adapter breaks
    this while `import airframe` might still look fine if that adapter is
    not re-exported.
    """
    leaked = _imported_sdks_after(f"import {module}")
    assert not leaked, (
        f"`import {module}` eagerly imported {leaked}. Move the vendor "
        f"import inside the method that needs it."
    )


def test_vendor_sdk_list_covers_every_declared_requirement() -> None:
    """Guard the guard: every adapter's ``REQUIRES_PACKAGE`` is watched.

    Without this, adding an adapter with a new vendor SDK would silently
    create an unwatched eager-import path — the test above would keep
    passing while the invariant quietly stopped being enforced.
    """
    from airframe.discovery import _builtin_runtime_classes

    required = {
        pkg
        for cls in _builtin_runtime_classes()
        if (pkg := getattr(cls, "REQUIRES_PACKAGE", None))
    }
    unwatched = sorted(required - set(VENDOR_SDKS))
    assert not unwatched, (
        f"{unwatched} is declared as a REQUIRES_PACKAGE but absent from "
        f"VENDOR_SDKS — add it here and in scripts/smoke_install.py so the "
        f"lazy-import invariant is actually enforced for it."
    )
