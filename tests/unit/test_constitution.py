"""Enforcement for `.specify/memory/constitution.md`.

A constitution principle that isn't a test is a suggestion. Every
principle in that document either names a test in this module, names a
shared contract in :mod:`airframe.testing.contracts`, or states
explicitly that review is the enforcement. This module is the first
category, plus a drift test tying the document to this file.

Nothing here is decorative: each test fails on a real violation of the
principle it cites, not on a proxy for one.
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONSTITUTION = REPO_ROOT / ".specify" / "memory" / "constitution.md"
ADAPTERS_DIR = REPO_ROOT / "src" / "airframe" / "adapters"


def _adapter_classes() -> list[type[Any]]:
    """Every concrete built-in adapter class.

    Returns:
        The adapter classes ``airframe.discovery`` exposes as
        built-ins, in its own stable order.
    """
    from airframe.discovery import _builtin_runtime_classes

    return list(_builtin_runtime_classes())


# ---------------------------------------------------------------------------
# Principle I — Protocol Narrowness
# ---------------------------------------------------------------------------

#: The protocol surface the constitution fixes. Widening this list is a
#: breaking change for every third-party adapter, so it is spelled out
#: here rather than derived — the test exists to make the widening
#: deliberate, and a derived list would rubber-stamp it.
DECLARED_PROTOCOL_METHODS = frozenset(
    {
        "execute",
        "reset",
        "close",
        "validate_binding",
        "count_tokens",
        "list_models",
        "unwrap",
        "supports",
        "supported_native_tools",
        "session",
    }
)


def test_protocol_surface_is_the_declared_five() -> None:
    """``AgentRuntime`` exposes only the constitutionally declared members.

    The five methods plus ``supports`` / ``unwrap`` are the contract;
    ``count_tokens``, ``supported_native_tools`` and ``session`` are
    the capability-gated additions that shipped with Phase 0. Anything
    else means the protocol grew without an amendment.
    """
    from airframe.protocol import AgentRuntime

    actual = {
        name
        for name in vars(AgentRuntime)
        if not name.startswith("_") and callable(vars(AgentRuntime)[name])
    }
    assert actual == DECLARED_PROTOCOL_METHODS, (
        "AgentRuntime's surface changed. Widening the protocol is a breaking "
        "change for every third-party adapter — amend "
        f"{CONSTITUTION.relative_to(REPO_ROOT)} (Principle I) and bump its "
        f"version before updating this list. Added: "
        f"{sorted(actual - DECLARED_PROTOCOL_METHODS)}; "
        f"removed: {sorted(DECLARED_PROTOCOL_METHODS - actual)}"
    )


# ---------------------------------------------------------------------------
# Principle II — Wrap Vendor SDKs, Don't Rewrite Them
# ---------------------------------------------------------------------------

#: Adapters permitted to issue raw HTTP. Empty by design: every vendor
#: airframe wraps ships an SDK (``anthropic``, ``openai``, ``copilot``,
#: ``aioboto3``, ``opencode-ai``), and that SDK already owns headers,
#: auth refresh, retries and rate-limit telemetry. Adding an entry here
#: means claiming a vendor has no SDK to wrap — state the reason inline.
RAW_HTTP_ALLOWED: frozenset[str] = frozenset()

#: Dotted prefixes that issue a request. Deliberately narrower than
#: "imports an HTTP library": ``urllib.parse.urlparse`` parses a URL and
#: ``aiohttp.ClientError`` is an exception type used for error
#: classification under Principle IV — neither one talks to a vendor.
RAW_HTTP_CALLS = (
    "httpx.",
    "requests.",
    "aiohttp.ClientSession",
    "aiohttp.request",
    "urllib.request.urlopen",
    "http.client.",
)


def _dotted_name(node: ast.AST) -> str:
    """Reconstruct a dotted attribute chain from an AST node.

    Args:
        node: The ``func`` of a call, or any attribute expression.

    Returns:
        The dotted source form (``"httpx.AsyncClient"``), or ``""`` if
        the chain is not a plain name/attribute path.
    """
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


@pytest.mark.parametrize(
    "adapter_path",
    sorted(p for p in ADAPTERS_DIR.glob("*.py") if p.name != "__init__.py"),
    ids=lambda p: p.name,
)
def test_no_hand_rolled_vendor_http(adapter_path: Path) -> None:
    """Adapters reach vendors through the vendor's SDK, not a raw client.

    Checks calls rather than imports, so importing ``aiohttp`` to
    ``isinstance``-check the transport exception ``aioboto3`` raises —
    error classification, Principle IV — is not mistaken for issuing a
    request.

    Args:
        adapter_path: One adapter module, supplied by parametrisation.
    """
    if adapter_path.stem in RAW_HTTP_ALLOWED:  # pragma: no cover - empty today
        pytest.skip(f"{adapter_path.stem} is documented as HTTP by design")

    tree = ast.parse(adapter_path.read_text())
    offenders = sorted(
        {
            name
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (name := _dotted_name(node.func))
            and name.startswith(RAW_HTTP_CALLS)
        }
    )
    assert not offenders, (
        f"{adapter_path.name} issues raw HTTP via {offenders}. Principle II: "
        "wrap the vendor's official SDK — it already owns headers, auth "
        "refresh, retries and rate-limit telemetry. If this vendor genuinely "
        "has no SDK, add the module to RAW_HTTP_ALLOWED with the reason."
    )


# ---------------------------------------------------------------------------
# Principle V — Lazy SDK Imports
# ---------------------------------------------------------------------------

#: Vendor SDKs that must not be importable side effects of
#: ``import airframe``. Each is installed in the dev environment, so a
#: regression here shows up rather than being masked by absence.
VENDOR_SDK_MODULES = (
    "claude_agent_sdk",
    "anthropic",
    "copilot",
    "openai",
    "aioboto3",
    "boto3",
    "botocore",
    "opencode_ai",
    "tiktoken",
)


def test_import_airframe_pulls_no_vendor_sdk() -> None:
    """``import airframe`` must not drag a vendor SDK into memory.

    Run in a clean subprocess: the in-process ``sys.modules`` is
    already polluted by the rest of the suite, so checking it here
    would pass for the wrong reason.
    """
    probe = (
        "import sys, json; import airframe; "
        f"print(json.dumps([m for m in {VENDOR_SDK_MODULES!r} if m in sys.modules]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    leaked = __import__("json").loads(result.stdout.strip().splitlines()[-1])
    assert leaked == [], (
        f"`import airframe` pulled in {leaked}. Principle V: adapter modules "
        "import their vendor SDK inside the method that needs it, so a "
        "consumer who installs one extra pays for one SDK."
    )


# ---------------------------------------------------------------------------
# Principle VI — Provider IDs Are Strict
# ---------------------------------------------------------------------------

#: Reserved for future direct-API adapters. A subscription, gateway or
#: agent-server adapter must not take one of these.
RESERVED_PROVIDER_IDS = frozenset({"anthropic", "openai", "bedrock-agents", "moonshot", "codex"})


@pytest.mark.parametrize("runtime_cls", _adapter_classes(), ids=lambda c: c.__name__)
def test_every_adapter_declares_provider_classvars(runtime_cls: type[Any]) -> None:
    """Each adapter declares the three discovery ClassVars, non-empty.

    Args:
        runtime_cls: One built-in adapter class.
    """
    for attr in ("PROVIDER_ID", "REQUIRES_PACKAGE", "EXTRA_NAME"):
        value = getattr(runtime_cls, attr, None)
        assert isinstance(value, str) and value, (
            f"{runtime_cls.__name__}.{attr} is missing or empty. Principle VI: "
            "these three drive extras-aware discovery and the "
            "`airframe.adapters` entry-point group."
        )


def test_provider_ids_are_unique() -> None:
    """No two built-in adapters claim the same ``PROVIDER_ID``.

    Distinct wire formats get distinct IDs even under one brand — a
    collision would make `runtime_for()` silently ambiguous.
    """
    ids = [cls.PROVIDER_ID for cls in _adapter_classes()]
    duplicates = sorted({pid for pid in ids if ids.count(pid) > 1})
    assert not duplicates, f"duplicate PROVIDER_IDs: {duplicates}"


def test_reserved_provider_ids_are_not_taken() -> None:
    """Reserved IDs stay free for the adapters they name.

    ``"anthropic"`` and ``"openai"`` belong to future direct-API
    adapters; ``"bedrock-agents"``, ``"moonshot"`` and ``"codex"`` name
    siblings that must not be folded into an existing adapter.
    """
    taken = sorted(
        cls.PROVIDER_ID for cls in _adapter_classes() if cls.PROVIDER_ID in RESERVED_PROVIDER_IDS
    )
    assert not taken, (
        f"{taken} are reserved provider IDs. Principle VI: a subscription, "
        "gateway or agent-server adapter must not take an ID reserved for a "
        "direct-API or differently-shaped sibling."
    )


def test_extras_named_by_adapters_exist_in_pyproject() -> None:
    """Every ``EXTRA_NAME`` resolves to a real optional-dependency extra.

    An adapter naming an extra that does not exist makes
    ``list_providers()`` advise an install command that cannot work.
    """
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)
    declared = set(pyproject["project"]["optional-dependencies"])
    named = {cls.EXTRA_NAME for cls in _adapter_classes()}
    missing = sorted(named - declared)
    assert not missing, f"adapters name extras that pyproject.toml does not declare: {missing}"


# ---------------------------------------------------------------------------
# Principle VII — Conformance Is Shared, Not Copied
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("runtime_cls", _adapter_classes(), ids=lambda c: c.__name__)
def test_every_adapter_has_a_conformance_suite(runtime_cls: type[Any]) -> None:
    """Each built-in adapter runs the shared contracts.

    Args:
        runtime_cls: One built-in adapter class.
    """
    suites = list((REPO_ROOT / "tests" / "unit").glob("test_*_conformance.py"))
    wired = {
        suite.stem
        for suite in suites
        if re.search(rf"\b{runtime_cls.__name__}\b", suite.read_text())
    }
    assert wired, (
        f"{runtime_cls.__name__} has no tests/unit/test_*_conformance.py. "
        "Principle VII: contract-worthy behaviour lives in "
        "airframe.testing.contracts and every adapter runs it — including "
        "thin OpenAICompatibleRuntime subclasses."
    )


def test_contracts_module_exports_only_test_functions() -> None:
    """``contracts.py`` holds shared tests, not adapter-specific helpers.

    Anything public in that module is imported by name into every
    adapter's conformance suite, so a non-test public symbol would be
    silently collected as one.
    """
    import airframe.testing.contracts as contracts

    public = [
        name
        for name, obj in vars(contracts).items()
        if not name.startswith("_")
        and callable(obj)
        and getattr(obj, "__module__", "") == contracts.__name__
    ]
    non_tests = sorted(n for n in public if not n.startswith("test_"))
    assert not non_tests, (
        f"public non-test callables in contracts.py: {non_tests}. Keep helpers "
        "private (leading underscore) so conformance suites can import * safely."
    )


# ---------------------------------------------------------------------------
# Principle VIII — One Command Surface
# ---------------------------------------------------------------------------

WORKFLOWS = REPO_ROOT / ".github" / "workflows"


#: Workflows that carry a gating job. Both must reach the gate through
#: the verb: `ci.yml` on every push and PR, `release.yml` as the job the
#: publish depends on. A third workflow that starts gating belongs here.
GATING_WORKFLOWS = ("ci.yml", "release.yml")

#: Tools that constitute a hand-written gate step. Their presence
#: anywhere in a gating workflow — a `run:` line, a block scalar, a step
#: name — means the workflow is describing the gate rather than invoking
#: it.
GATE_TOOLS = ("ruff", "mypy", "pytest")


@pytest.mark.parametrize("workflow_name", GATING_WORKFLOWS)
def test_gating_workflows_invoke_only_the_check_verb(workflow_name: str) -> None:
    """Each gating workflow runs ``mise run check`` and nothing inline.

    The failure this prevents is the one the portfolio standard calls
    out by name: a CI step list that agrees with the local gate today
    and drifts from it tomorrow. Checked per workflow rather than for
    ``ci.yml`` alone, because ``release.yml`` gates the publish and can
    drift the same way.

    Args:
        workflow_name: One gating workflow, supplied by parametrisation.
    """
    workflow = (WORKFLOWS / workflow_name).read_text()
    assert "mise run check" in workflow, (
        f"{workflow_name} must invoke `mise run check` — it gates something."
    )

    # Comments are prose. Everything else is the workflow describing
    # what it does, so a gate tool named there is a gate step.
    code = "\n".join(line for line in workflow.splitlines() if not line.lstrip().startswith("#"))
    inline = sorted({tool for tool in GATE_TOOLS if re.search(rf"\b{tool}\b", code)})
    assert not inline, (
        f"{workflow_name} names {inline} outside a comment. Principle VIII: "
        "a gating workflow invokes the verb, never a parallel list of steps "
        "that can drift from it."
    )


def _gate_task_commands(tasks: dict[str, Any]) -> dict[str, list[str]]:
    """Every command reachable from the ``check`` verb.

    Args:
        tasks: The ``[tasks]`` table parsed from ``mise.toml``.

    Returns:
        Task name to its list of shell commands, for ``check`` and
        everything ``check`` delegates to, transitively.
    """
    resolved: dict[str, list[str]] = {}
    pending = ["check"]
    while pending:
        name = pending.pop()
        if name in resolved or name not in tasks:
            continue
        run = tasks[name].get("run", "")
        commands = list(run) if isinstance(run, list) else [run]
        resolved[name] = commands
        for command in commands:
            pending.extend(re.findall(r"mise run ([\w-]+)", command))
    return resolved


def test_no_task_swallows_failure() -> None:
    """No task in the gate graph hides a non-zero exit code.

    ``|| true`` or a ``grep`` fallback turns ``check`` into a check in
    name only — the exact defect the portfolio standard found in a
    sibling repo. Scoped to the tasks ``check`` actually reaches:
    housekeeping verbs like ``clean`` legitimately tolerate a failing
    ``find``.
    """
    with (REPO_ROOT / "mise.toml").open("rb") as fh:
        mise = tomllib.load(fh)

    gate = _gate_task_commands(mise.get("tasks", {}))
    assert "check" in gate, "mise.toml declares no `check` task"

    offenders = [
        f"{name}: {command}"
        for name, commands in gate.items()
        for command in commands
        if "|| true" in command or "|| echo" in command or "|| :" in command
    ]
    assert not offenders, (
        f"gate tasks swallow failures: {offenders}. A `check` that cannot "
        "fail is a check in name only."
    )


def test_check_composition_propagates_failure() -> None:
    """A failure inside a nested task escapes to the runner's exit code.

    ``check`` is a list of ``mise run <verb>`` calls, so "the gate
    returns non-zero" is a claim about how mise composes tasks. The two
    hidden ``_selftest-*`` tasks in ``mise.toml`` exist so this is
    measured rather than assumed.
    """
    mise = shutil.which("mise")
    if mise is None:  # pragma: no cover - depends on the local toolchain
        pytest.skip("mise is not installed in this environment")

    for task in ("_selftest-fail", "_selftest-nested"):
        result = subprocess.run(
            [mise, "run", task],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=120,
        )
        assert result.returncode != 0, (
            f"`mise run {task}` reported success for a command that exited 1 — "
            "the gate cannot be trusted to fail."
        )


def test_canonical_verbs_all_exist() -> None:
    """The portfolio verb contract is fully implemented.

    Every repo in the portfolio answers to the same verbs, so a CI
    workflow, a devcontainer hook and an agent instruction file can all
    assume they are there.
    """
    with (REPO_ROOT / "mise.toml").open("rb") as fh:
        mise = tomllib.load(fh)
    canonical = {"setup", "build", "test", "lint", "fmt", "check", "release"}
    missing = sorted(canonical - set(mise.get("tasks", {})))
    assert not missing, f"mise.toml is missing canonical verbs: {missing}"


# ---------------------------------------------------------------------------
# The document itself
# ---------------------------------------------------------------------------


def test_constitution_version_header_and_footer_agree() -> None:
    """The Sync Impact Report and the footer name the same version.

    The amendment procedure requires updating both. This is the test
    that notices when only one was updated.
    """
    text = CONSTITUTION.read_text()
    header = re.search(r"Version change:\s*.*?→\s*([0-9]+\.[0-9]+\.[0-9]+)", text)
    footer = re.search(r"\*\*Version\*\*:\s*([0-9]+\.[0-9]+\.[0-9]+)", text)
    assert header and footer, "constitution must carry a Sync Impact Report and a footer"
    assert header.group(1) == footer.group(1), (
        f"Sync Impact Report says {header.group(1)} but the footer says "
        f"{footer.group(1)} — the amendment procedure requires both."
    )


def test_every_principle_names_its_enforcement() -> None:
    """No principle is left without a stated enforcement.

    "Review" is an acceptable answer only when it is stated as the
    answer. A principle with no ``*Enforced by:*`` line at all is a
    defect in the document.
    """
    text = CONSTITUTION.read_text()
    body = text.split("## Core Principles", 1)[1].split("## Quality Gates", 1)[0]
    sections = re.split(r"^### ", body, flags=re.MULTILINE)[1:]
    unenforced = [
        section.splitlines()[0].strip() for section in sections if "*Enforced by:*" not in section
    ]
    assert not unenforced, (
        f"principles with no stated enforcement: {unenforced}. Name the "
        "command that fails, or say outright that review is the enforcement."
    )


def test_principles_citing_this_module_actually_exist_here() -> None:
    """Every test the constitution names is a real test in this module.

    A citation that no longer resolves is how a governance doc rots
    quietly — the principle still reads as enforced when it is not.
    """
    text = CONSTITUTION.read_text()
    cited = set(re.findall(r"test_constitution\.py::(\w+)", text))
    cited |= set(re.findall(r"^`?::(\w+)`?", text, flags=re.MULTILINE))
    defined = set(re.findall(r"^def (test_\w+)", Path(__file__).read_text(), re.M))
    missing = sorted(cited - defined)
    assert not missing, f"the constitution cites tests that do not exist here: {missing}"
