#!/usr/bin/env python3
"""End-to-end probe for :attr:`AgentSession` resume.

Demonstrates the two-turn pattern Phase 1 Iteration C+ wired across
the three SDK-based adapters:

1. Open a fresh session, send a turn that says something memorable.
2. Capture :attr:`AgentSession.id`.
3. Close the session.
4. Open a *new* session with ``resume=<id>``, send a follow-up that
   references the prior turn.
5. Verify the model's reply demonstrates continuity (i.e., the
   resumed session loaded the conversation history).

Usage::

    uv run python examples/probe_session_resume.py
    uv run python examples/probe_session_resume.py --provider claude
    uv run python examples/probe_session_resume.py --provider github-copilot
    uv run python examples/probe_session_resume.py --provider codex

Defaults to ``claude``. OpenAI-compat is rejected because the
chat-completions wire format has no server-side session — see the
``OpenAICompatibleRuntime`` docstring for why.

The follow-up prompt deliberately asks the model to recall the
"secret word" from turn 1. A successful resume produces a reply
containing that word; a fresh-session reply will admit it doesn't
know.
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

from airframe import Feature, list_providers, runtime_for  # noqa: E402

SECRET_WORD = "spectrolite"
TURN_1_PROMPT = (
    f"Remember the secret word: {SECRET_WORD}. Acknowledge briefly that you've noted it."
)
TURN_2_PROMPT = "What was the secret word I asked you to remember? Reply with just the word."


def _build_runtime(provider_id: str):  # type: ignore[no-untyped-def]
    cls = runtime_for(provider_id)
    try:
        return cls()
    except TypeError:
        env_key = f"{provider_id.upper().replace('-', '_')}_API_KEY"
        api_key = os.environ.get(env_key) or os.environ.get("OPENCODE_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"{provider_id!r} needs {env_key} (or OPENCODE_API_KEY) "
                f"set to construct the adapter."
            ) from None
        return cls(api_key=api_key)  # type: ignore[call-arg]


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--provider",
        default="claude",
        help="Provider ID (default: claude). Must declare Feature.SESSION_RESUME.",
    )
    args = parser.parse_args()

    installed = list_providers(installed_only=True)
    if args.provider not in installed:
        print(
            f"Provider {args.provider!r} not installed. Available: {installed}",
            file=sys.stderr,
        )
        return 1

    runtime = _build_runtime(args.provider)
    if not runtime.supports(Feature.SESSION_RESUME):
        print(
            f"FAIL: {type(runtime).__name__} does not declare Feature.SESSION_RESUME. "
            f"Resume is wired on Claude Code, Copilot, and Codex; OpenAI-compatible "
            f"adapters can't resume because chat-completions has no server-side session.",
            file=sys.stderr,
        )
        await runtime.close()
        return 1

    print(f"session-resume probe — provider={args.provider}")

    # ----- Turn 1: open a fresh session, send the secret. ------------------
    sess1 = runtime.session()
    captured_id: str | None = None
    err: str | None = None
    try:
        t0 = time.monotonic()
        print(f"\n  turn 1 ({sess1.id=})")
        print(f"    prompt: {TURN_1_PROMPT!r}")
        result1 = await sess1.execute(TURN_1_PROMPT)
        print(f"    reply:  {result1.text[:200]!r}")
        print(f"    elapsed: {time.monotonic() - t0:.1f}s")
        captured_id = sess1.id
        if captured_id is None:
            print(
                "FAIL: session.id was not populated after the first turn — "
                "the adapter should surface the vendor session_id here.",
                file=sys.stderr,
            )
            return 1
        print(f"    captured session id: {captured_id}")
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        import traceback

        traceback.print_exc()
    finally:
        await sess1.close()

    if err is not None or captured_id is None:
        await runtime.close()
        print(f"\nFAIL: {err or 'no session id'}")
        return 1

    # ----- Turn 2: open a new session with resume=<id>. --------------------
    sess2 = runtime.session(resume=captured_id)
    text2 = ""
    try:
        t0 = time.monotonic()
        print(f"\n  turn 2 (resumed; {sess2.id=})")
        print(f"    prompt: {TURN_2_PROMPT!r}")
        result2 = await sess2.execute(TURN_2_PROMPT)
        text2 = result2.text
        print(f"    reply:  {text2[:200]!r}")
        print(f"    elapsed: {time.monotonic() - t0:.1f}s")
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        import traceback

        traceback.print_exc()
    finally:
        await sess2.close()
        await runtime.close()

    if err is not None:
        print(f"\nFAIL: {err}")
        return 1

    # The model should reproduce the secret word in turn 2. Case-insensitive
    # contains is a tolerant check (model may quote, add punctuation, etc.).
    if SECRET_WORD.lower() not in text2.lower():
        print(
            f"\nFAIL: resumed session reply did not contain the secret word "
            f"{SECRET_WORD!r}. The session may not have loaded prior history. "
            f"Reply was: {text2!r}"
        )
        return 1

    print(f"\nPASS — resumed session recalled the secret word {SECRET_WORD!r}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
