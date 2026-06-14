#!/usr/bin/env python3
"""Env-aware provider smoke test — exercise airframe against whatever
credentials you actually have on this machine.

This is the "does it really work end to end" check that complements the
focused ``probe_*.py`` scripts. It:

  1. Loads a ``.env`` of provider API keys (without clobbering real env).
  2. For every provider an installed adapter declares, decides whether
     it's RUNNABLE (SDK installed + credentials present), and echoes the
     ones it's SKIPPING with the reason (adapter not installed / which
     env var to set).
  3. For each runnable provider, exercises the real code paths:
       - declared capabilities (``supports()``)
       - a structured-output round-trip (``execute(schema=...)``)
       - if the provider serves native web tools, a live
         ``session(native_tools=[...]).stream(...)`` round-trip — the
         exact path the Copilot + OpenCode native-tools work added.

Usage::

    uv run python examples/smoke_providers.py            # load ./.env (or repo .env), run all
    uv run python examples/smoke_providers.py --env ~/.airframe.env
    uv run python examples/smoke_providers.py --provider claude,github-copilot
    uv run python examples/smoke_providers.py --dry-run  # readiness only, no network
    uv run python examples/smoke_providers.py --no-native # skip the web-tools exercise

Per-provider model override (some providers have no default model):
set ``AIRFRAME_PROBE_MODEL_<PROVIDER>`` with the provider ID upper-cased
and hyphens→underscores, e.g. ``AIRFRAME_PROBE_MODEL_GITHUB_COPILOT``.

Exit code: 0 unless a runnable provider produced a FAIL (auth-skips and
not-installed-skips are non-fatal — having creds for a subset is normal).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from pydantic import BaseModel  # noqa: E402

from airframe import (  # noqa: E402
    Feature,
    NativeCapability,
    NativeTool,
    ProviderModel,
    list_providers,
    runtime_for,
)
from airframe.errors import RuntimeAuthError  # noqa: E402
from airframe.events import TextDelta, ToolCallStart, TurnComplete  # noqa: E402

# --- Provider registry ------------------------------------------------------
# Credential model: each provider lists alternative env-var GROUPS. A group
# is satisfied when every var in it is set; the provider is authed when ANY
# group is satisfied. (Most are single-var "any-of"; bedrock shows the
# multi-option shape.)
CREDENTIALS: dict[str, list[list[str]]] = {
    "claude": [["CLAUDE_CODE_OAUTH_TOKEN"], ["ANTHROPIC_API_KEY"]],
    "github-copilot": [["GITHUB_TOKEN"], ["GH_TOKEN"]],
    "openrouter": [["OPENROUTER_API_KEY"]],
    "opencode-zen": [["OPENCODE_API_KEY"]],
    "opencode-go": [["OPENCODE_API_KEY"]],
    "opencode": [["OPENCODE_SERVER_URL"]],  # also needs a running server
    "bedrock": [["AWS_ACCESS_KEY_ID"], ["AWS_PROFILE"]],
    "kimi": [["KIMI_API_KEY"]],
}

# pip extra that installs each provider's SDK (for the skip hint).
EXTRA_FOR: dict[str, str] = {
    "claude": "claude",
    "github-copilot": "copilot",
    "kimi": "kimi",
    "bedrock": "bedrock",
    "openrouter": "openai-compat",
    "opencode-zen": "openai-compat",
    "opencode-go": "openai-compat",
    "opencode": "opencode",
}

# Curated capabilities to surface per provider (declared, not exercised).
DISPLAY_FEATURES = [
    Feature.STREAMING,
    Feature.STRUCTURED_OUTPUT_JSON_SCHEMA,
    Feature.TOOLS_FUNCTION,
    Feature.TOOLS_NATIVE,
    Feature.VISION_INPUT,
    Feature.SESSION_RESUME,
]


class Answer(BaseModel):
    answer: int
    rationale: str


STRUCTURED_PROMPT = "What is 17 + 25? Reply with the integer answer and a one-line rationale."
NATIVE_PROMPT = (
    "Use your web tool to look up the latest stable Python version, "
    "then answer in one sentence."
)


# --- .env loading -----------------------------------------------------------
def load_dotenv(path: Path, *, override: bool) -> int:
    """Minimal dependency-free .env loader. Real env wins unless override."""
    if not path.exists():
        return 0
    loaded = 0
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if override or key not in os.environ:
            os.environ[key] = val
            loaded += 1
    return loaded


# --- Readiness --------------------------------------------------------------
def credential_status(pid: str) -> tuple[bool, str]:
    """(authed, human detail) for a provider's credential requirement."""
    groups = CREDENTIALS.get(pid)
    if groups is None:
        return False, "no known credential mapping"
    for group in groups:
        if all(os.environ.get(v) for v in group):
            return True, "via " + "+".join(group)
    wanted = " | ".join("+".join(g) for g in groups)
    return False, f"set one of: {wanted}"


def model_for(pid: str) -> ProviderModel | None:
    env_key = "AIRFRAME_PROBE_MODEL_" + pid.upper().replace("-", "_")
    mid = os.environ.get(env_key)
    return ProviderModel(pid, mid) if mid else None


def build_runtime(pid: str):
    cls = runtime_for(pid)
    try:
        return cls()
    except TypeError:
        # OpenAI-compat runtimes require api_key= at construction. Pull the
        # first present credential var and pass it through.
        for group in CREDENTIALS.get(pid, []):
            for var in group:
                if os.environ.get(var):
                    return cls(api_key=os.environ[var])  # type: ignore[call-arg]
        raise


# --- Exercises --------------------------------------------------------------
async def exercise_structured(runtime, pid: str) -> tuple[str, str, str]:
    t0 = time.monotonic()
    res = await runtime.execute(
        STRUCTURED_PROMPT, schema=Answer, model=model_for(pid), timeout=120
    )
    dt = time.monotonic() - t0
    ans = (res.structured or {}).get("answer")
    cost = f"${res.cost.cost_usd:.4f}" if res.cost.cost_usd is not None else "cost=n/a"
    status = "PASS" if ans == 42 else "FAIL"
    return (
        "structured-output",
        status,
        f"answer={ans} {res.cost.provider_id}/{res.cost.model_id} {dt:.1f}s {cost}",
    )


async def exercise_native(runtime, pid: str) -> tuple[str, str, str]:
    served = runtime.supported_native_tools(model_for(pid))
    if not served:
        return ("native-web-tools", "N/A", "serves no native tools")
    cap = (
        NativeCapability.WEB_SEARCH
        if NativeCapability.WEB_SEARCH in served
        else sorted(served)[0]
    )
    sess = runtime.session(native_tools=[NativeTool(capability=cap)], model=model_for(pid))
    tool_calls, chars = 0, 0
    t0 = time.monotonic()
    try:
        async for ev in sess.stream(NATIVE_PROMPT):
            if isinstance(ev, ToolCallStart):
                tool_calls += 1
            elif isinstance(ev, TextDelta):
                chars += len(ev.text)
            elif isinstance(ev, TurnComplete):
                pass
    finally:
        await sess.close()
    dt = time.monotonic() - t0
    status = "PASS" if tool_calls > 0 else "WARN"
    note = "" if tool_calls > 0 else " (answered without calling the web tool)"
    return (f"native:{cap.value}", status, f"tool_calls={tool_calls} chars={chars} {dt:.1f}s{note}")


async def run_provider(pid: str, *, do_native: bool) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    try:
        runtime = build_runtime(pid)
    except Exception as exc:  # noqa: BLE001
        return [("construct", "FAIL", f"{type(exc).__name__}: {exc}")]
    try:
        declared = " ".join(
            f.name.lower() for f in DISPLAY_FEATURES if runtime.supports(f)
        )
        rows.append(("declares", "INFO", declared or "(none)"))

        try:
            rows.append(await exercise_structured(runtime, pid))
        except RuntimeAuthError as exc:
            rows.append(("structured-output", "SKIP", f"auth rejected: {exc}"))
        except Exception as exc:  # noqa: BLE001
            rows.append(("structured-output", "FAIL", f"{type(exc).__name__}: {exc}"))

        if do_native and runtime.supports(Feature.TOOLS_NATIVE):
            try:
                rows.append(await exercise_native(runtime, pid))
            except RuntimeAuthError as exc:
                rows.append(("native-web-tools", "SKIP", f"auth rejected: {exc}"))
            except Exception as exc:  # noqa: BLE001
                rows.append(("native-web-tools", "FAIL", f"{type(exc).__name__}: {exc}"))
    finally:
        await runtime.close()
    return rows


# --- Main -------------------------------------------------------------------
def resolve_env_path(arg: str | None) -> Path | None:
    if arg:
        return Path(arg).expanduser()
    for candidate in (Path.cwd() / ".env", REPO_ROOT / ".env"):
        if candidate.exists():
            return candidate
    return None


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", help="Path to .env (default: ./.env then repo .env)")
    parser.add_argument("--provider", help="Comma-separated provider IDs (default: all known)")
    parser.add_argument(
        "--no-native", action="store_true", help="Skip the native web-tools exercise"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Readiness + declared caps only; no network"
    )
    parser.add_argument(
        "--override-env",
        action="store_true",
        help="Let .env values override already-set process env (default: real env wins)",
    )
    args = parser.parse_args()

    env_path = resolve_env_path(args.env)
    if env_path:
        n = load_dotenv(env_path, override=args.override_env)
        print(f"Loaded {n} var(s) from {env_path}")
    else:
        print("No .env found — using process environment only.")

    installed = set(list_providers(installed_only=True))
    known = list_providers(installed_only=False)
    if args.provider:
        requested = [p.strip() for p in args.provider.split(",") if p.strip()]
        unknown = [p for p in requested if p not in known]
        if unknown:
            print(f"WARN: unknown provider id(s) ignored: {unknown}")
        known = [p for p in known if p in requested]

    print(f"\nKnown providers: {known}")
    print(f"Adapters installed (SDK present): {sorted(installed)}\n")

    runnable: list[str] = []
    print("== Readiness ==")
    for pid in known:
        authed, detail = credential_status(pid)
        if pid not in installed:
            extra = EXTRA_FOR.get(pid, pid)
            print(f"  SKIP  {pid:<16} adapter not installed (pip install airframe-agents[{extra}])")
        elif not authed:
            print(f"  SKIP  {pid:<16} no credentials — {detail}")
        else:
            print(f"  READY {pid:<16} {detail}")
            runnable.append(pid)

    if args.dry_run:
        print("\n(dry-run: skipping network exercises)")
        return 0

    if not runnable:
        print("\nNothing runnable. Add keys to your .env. See examples? `.env.example`.")
        return 0

    print("\n== Exercising runnable providers ==")
    any_fail = False
    for pid in runnable:
        print(f"\n[{pid}]")
        for name, status, detail in await run_provider(pid, do_native=not args.no_native):
            if status == "FAIL":
                any_fail = True
            print(f"  {status:<5} {name:<22} {detail}")

    print(
        f"\nSummary: {len(runnable)} provider(s) exercised, "
        f"{len(known) - len(runnable)} skipped. "
        + ("Some checks FAILED — see above." if any_fail else "No failures.")
    )
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
