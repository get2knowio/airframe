#!/usr/bin/env python3
"""End-to-end probe for :class:`~airframe.inputs.ImageInput` routing.

Exercises the polymorphic-prompt API Phase 2 Iterations C + D
introduced — opens a session and runs one ``execute()`` with
``prompt=[<text>, ImageInput(...)]``. Validates:

* The runtime declares :data:`~airframe.features.Feature.VISION_INPUT`
  (every adapter does after Iteration C).
* The adapter's content-builder accepts the chosen ``ImageInput``
  variant — ``path=``, ``bytes_=``, or ``url=`` — and routes it to
  the right per-vendor channel.
* The call completes and the model returns *some* response. The
  default test image is a tiny 1×1 PNG so the goal is to validate
  the wire plumbing, not the model's caption quality. Pass
  ``--image-path`` to point at a real image if you want to read the
  model's reaction.

Per-adapter variant support (set by Iteration D):

* OpenAI-compat — path, bytes, url (all three).
* Copilot — path, bytes (URL raises).
* Claude Code — path only (bytes / URL raise with a
  "write to disk and pass path=" message).
* Bedrock — path, bytes (URL raises — Converse needs the bytes
  locally; per-vendor: Anthropic / Nova / Llama 3.2 honour it).

Usage::

    uv run python examples/probe_vision.py
    uv run python examples/probe_vision.py --provider claude
    uv run python examples/probe_vision.py --provider github-copilot
    uv run python examples/probe_vision.py --provider opencode
    uv run python examples/probe_vision.py --provider bedrock
    uv run python examples/probe_vision.py --variant bytes
    uv run python examples/probe_vision.py --variant url \\
        --image-url https://example.com/cat.png
    uv run python examples/probe_vision.py --image-path /tmp/photo.jpg

Defaults to ``opencode-zen`` (OpenAI-compat, the only adapter that
natively handles every ``ImageInput`` variant) and the ``path``
variant against an auto-generated 1×1 test PNG.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from airframe import (  # noqa: E402
    Feature,
    ImageInput,
    list_providers,
    runtime_for,
)

DEFAULT_PROMPT = "Describe this image in one sentence."

# A minimal valid 1×1 transparent PNG. Bundled inline so the probe
# has no external dependencies; the goal is wire-protocol validation,
# not visual quality.
TINY_PNG_1x1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _build_runtime(provider_id: str):  # type: ignore[no-untyped-def]
    """Construct an adapter without requiring credentials in ``__init__``."""
    cls = runtime_for(provider_id)
    try:
        return cls()
    except TypeError:
        import os

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
        default="opencode-zen",
        help="Provider ID (default: opencode-zen). Any from list_providers().",
    )
    parser.add_argument(
        "--variant",
        default="path",
        choices=["path", "bytes", "url"],
        help="ImageInput variant to send. Default: path.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Text to send alongside the image. Default: a one-sentence ask.",
    )
    parser.add_argument(
        "--image-path",
        default=None,
        help="Path to an image file (variant=path). Default: a generated 1×1 PNG.",
    )
    parser.add_argument(
        "--image-url",
        default=None,
        help="HTTPS URL of an image (variant=url).",
    )
    args = parser.parse_args()

    installed = list_providers(installed_only=True)
    if args.provider not in installed:
        print(
            f"Provider {args.provider!r} not installed. Available: {installed}",
            file=sys.stderr,
        )
        print(
            "Install one with: pip install airframe-agents[claude|copilot|openai-compat|bedrock]",
            file=sys.stderr,
        )
        return 1

    runtime = _build_runtime(args.provider)

    print(f"vision probe — provider={args.provider} variant={args.variant}")
    if not runtime.supports(Feature.VISION_INPUT):
        print(
            f"  FAIL: {type(runtime).__name__} does not declare "
            f"Feature.VISION_INPUT — bailing before the call.",
            file=sys.stderr,
        )
        await runtime.close()
        return 1

    # Build the ImageInput per the chosen variant.
    tmp_path: Path | None = None
    if args.variant == "path":
        if args.image_path:
            img = ImageInput(path=args.image_path)
        else:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(TINY_PNG_1x1)
                tmp_path = Path(tmp.name)
            img = ImageInput(path=str(tmp_path), media_type="image/png")
            print(f"  generated test image at {tmp_path}")
    elif args.variant == "bytes":
        img = ImageInput(bytes_=TINY_PNG_1x1, media_type="image/png")
        print(f"  using inline bytes ({len(TINY_PNG_1x1)} bytes)")
    else:  # url
        if not args.image_url:
            print(
                "  --variant url requires --image-url <https://...>",
                file=sys.stderr,
            )
            await runtime.close()
            return 1
        img = ImageInput(url=args.image_url, media_type="image/png")
        print(f"  using URL {args.image_url}")

    sess = runtime.session()
    err: str | None = None
    result = None
    try:
        print(f"\n  prompt: {args.prompt!r}\n  -- execute begin --")
        result = await sess.execute([args.prompt, img])
        print("  -- execute end --")
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        import traceback

        traceback.print_exc()
    finally:
        await sess.close()
        await runtime.close()
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    if err is not None:
        print(f"\nFAIL: {err}")
        return 1
    assert result is not None

    print("\n  summary:")
    print(f"    text:          {result.text[:200]!r}{'...' if len(result.text) > 200 else ''}")
    print(f"    finish:        {result.finish}")
    print(f"    input_tokens:  {result.cost.input_tokens}")
    print(f"    output_tokens: {result.cost.output_tokens}")
    print(f"    cost_usd:      {result.cost.cost_usd}")
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
