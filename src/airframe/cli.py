"""``airframe`` command-line entry point — one prompt, any provider.

The CLI is a thin shell over the discovery layer and the
:class:`~airframe.protocol.AgentRuntime` protocol. It exists so a
prompt can be run against any installed adapter from a shell, a
Makefile, or — the motivating use case — a single reusable GitHub
Action step that picks its provider per workflow job::

    airframe run --provider opencode-zen --prompt "Triage issue #42"
    airframe run --provider claude --prompt-file review.md --format json

Auth is deliberately *not* a CLI concern: every adapter constructor
falls back to its vendor's environment variables
(``GITHUB_TOKEN`` for Copilot, ``ANTHROPIC_API_KEY`` for Claude, the
AWS chain for Bedrock, ``OPENCODE_*`` for the gateways). In CI the
job maps secrets onto those env vars and the CLI stays
provider-neutral.

Subcommands:

* ``run`` — execute one prompt against ``--provider`` and print the
  result (plain text by default, or a JSON envelope with cost / finish
  / structured payload via ``--format json``).
* ``providers`` — list provider IDs servable on this machine, mirroring
  :func:`airframe.list_providers`.

Exit codes are stable so callers (and Action steps) can branch:

============  ====================================================
code          meaning
============  ====================================================
``0``         success
``2``         usage error (bad flags; argparse-emitted)
``3``         unknown provider, or its SDK extra isn't installed
``4``         authentication failed (bad / missing credentials)
``5``         other classified runtime failure
``1``         unexpected / unclassified error
============  ====================================================
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from airframe import __version__

if TYPE_CHECKING:
    from collections.abc import Sequence

    from airframe.protocol import RuntimeResult

#: Stable process exit codes. Kept as a small table so the GitHub
#: Action (and any other caller) can branch on failure class without
#: scraping stderr.
EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_PROVIDER = 3
EXIT_AUTH = 4
EXIT_RUNTIME = 5


@dataclass(frozen=True, slots=True)
class _RunArgs:
    """Resolved, validated inputs for one ``run`` invocation."""

    provider: str
    model: str | None
    prompt: str
    system: str | None
    timeout: float
    fmt: str
    output_file: str | None


def _read_prompt(args: argparse.Namespace) -> str:
    """Resolve the prompt text from flags or stdin.

    Precedence: ``--prompt`` wins, then ``--prompt-file``, then piped
    stdin. An empty/whitespace-only result is treated as missing so a
    stray empty file or closed pipe fails loudly rather than sending a
    blank turn.

    Args:
        args: Parsed ``run`` namespace carrying ``prompt`` /
            ``prompt_file``.

    Returns:
        The prompt string, stripped of trailing newline noise.

    Raises:
        ValueError: when no prompt source yielded usable text.
    """
    if args.prompt is not None:
        text = args.prompt
    elif args.prompt_file is not None:
        with open(args.prompt_file, encoding="utf-8") as fh:
            text = fh.read()
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        raise ValueError("no prompt given — pass --prompt, --prompt-file, or pipe text on stdin")
    text = text.strip()
    if not text:
        raise ValueError("prompt is empty")
    return text


def _render(result: RuntimeResult, run: _RunArgs) -> str:
    """Format an execute() result for stdout / file output.

    Args:
        result: The canonical execute() result.
        run: The resolved run inputs (provider / model / format).

    Returns:
        Plain assistant text for ``--format text``; a pretty-printed
        JSON envelope (provider, model, text, structured, finish,
        cost) for ``--format json``.
    """
    if run.fmt == "json":
        envelope = {
            "provider": run.provider,
            "model": run.model,
            "text": result.text,
            "structured": result.structured,
            "finish": result.finish,
            "cost_usd": getattr(result.cost, "cost_usd", None),
        }
        return json.dumps(envelope, indent=2, default=str)
    return result.text


async def _execute(run: _RunArgs) -> RuntimeResult:
    """Build the runtime for ``run.provider`` and execute one prompt.

    The runtime's default model is set from ``--model`` at construction
    time; auth is left to each adapter's env-var fallback chain. The
    runtime is always ``close()``d, even on failure.

    Args:
        run: Resolved, validated run inputs.

    Returns:
        The canonical :class:`~airframe.protocol.RuntimeResult`.
    """
    from airframe import runtime_for

    runtime_cls = runtime_for(run.provider)
    # The AgentRuntime protocol deliberately doesn't constrain
    # constructors, but every built-in adapter takes a keyword-only
    # `model=` and resolves auth from its own env-var chain.
    runtime = runtime_cls(model=run.model)  # type: ignore[call-arg]
    try:
        return await runtime.execute(
            run.prompt,
            system=run.system,
            timeout=run.timeout,
        )
    finally:
        await runtime.close()


def _cmd_run(args: argparse.Namespace) -> int:
    """Handle ``airframe run``.

    Returns:
        A process exit code from the stable table in the module
        docstring.
    """
    from airframe.errors import AgentRuntimeError, RuntimeAuthError

    try:
        run = _RunArgs(
            provider=args.provider,
            model=args.model,
            prompt=_read_prompt(args),
            system=args.system,
            timeout=args.timeout,
            fmt=args.format,
            output_file=args.output_file,
        )
    except (ValueError, OSError) as exc:
        print(f"airframe: {exc}", file=sys.stderr)
        return EXIT_UNEXPECTED

    # Resolve the adapter class up front so an unknown provider / missing
    # extra is reported as a provider error (exit 3), distinct from a
    # runtime failure during execution (exit 5).
    from airframe import runtime_for

    try:
        runtime_for(run.provider)
    except (ValueError, ImportError) as exc:
        print(f"airframe: {exc}", file=sys.stderr)
        return EXIT_PROVIDER

    try:
        result = asyncio.run(_execute(run))
    except RuntimeAuthError as exc:
        print(f"airframe: authentication failed: {exc}", file=sys.stderr)
        return EXIT_AUTH
    except AgentRuntimeError as exc:
        print(f"airframe: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_RUNTIME

    rendered = _render(result, run)
    print(rendered)
    if run.output_file is not None:
        with open(run.output_file, "w", encoding="utf-8") as fh:
            fh.write(rendered)
            fh.write("\n")
    return EXIT_OK


def _cmd_providers(args: argparse.Namespace) -> int:
    """Handle ``airframe providers``.

    Returns:
        Always ``0`` — discovery never raises.
    """
    from airframe import list_providers

    providers = list_providers(installed_only=not args.all)
    if args.format == "json":
        print(json.dumps(providers, indent=2))
    else:
        for provider in providers:
            print(provider)
    return EXIT_OK


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser and its subcommands."""
    parser = argparse.ArgumentParser(
        prog="airframe",
        description="Run one prompt against any installed agent SDK adapter.",
    )
    parser.add_argument("--version", action="version", version=f"airframe {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser(
        "run",
        help="execute one prompt against a provider",
        description="Execute a single prompt against --provider and print the result.",
    )
    run.add_argument(
        "--provider",
        required=True,
        help="canonical provider id (e.g. claude, github-copilot, opencode-zen, bedrock)",
    )
    run.add_argument(
        "--model",
        default=None,
        help="model id to pin (defaults to the adapter's default model)",
    )
    prompt_src = run.add_argument_group("prompt source (one of)")
    prompt_src.add_argument("--prompt", default=None, help="inline prompt text")
    prompt_src.add_argument(
        "--prompt-file",
        default=None,
        help="path to a file holding the prompt (use - for stdin)",
    )
    run.add_argument("--system", default=None, help="optional system-prompt override")
    run.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="hard wall-clock budget in seconds (default: 600)",
    )
    run.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="text prints the assistant reply; json prints a full envelope",
    )
    run.add_argument(
        "--output-file",
        default=None,
        help="also write the rendered output to this path",
    )
    run.set_defaults(func=_cmd_run)

    providers = sub.add_parser(
        "providers",
        help="list provider ids servable on this machine",
        description="List provider ids, filtered by which adapter SDKs are installed.",
    )
    providers.add_argument(
        "--all",
        action="store_true",
        help="include providers whose SDK extra isn't installed",
    )
    providers.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="text prints one id per line; json prints an array",
    )
    providers.set_defaults(func=_cmd_providers)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector excluding ``prog``. Defaults to
            ``sys.argv[1:]`` when ``None``.

    Returns:
        A process exit code. Wired as the ``airframe`` console script,
        so a returned int becomes the process status.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
