"""Top-level provider discovery — for menus and config dispatch.

Two entry points:

* :func:`list_providers` — what providers are usable on this machine
  given which optional extras the consumer installed.
* :func:`runtime_for` — given a ``provider_id``, return the adapter
  class that serves it.

The discovery layer respects airframe's optional dependency extras:
if the consumer ran ``pip install airframe-agents[copilot]`` only,
:func:`list_providers` returns ``["github-copilot"]`` and the other
providers are silently filtered. That keeps UI menus honest about
what the local install can actually run.

Pass ``installed_only=False`` to see every provider an airframe
adapter declares, regardless of which SDKs are present — useful for
documentation, "what's possible if I install X?" UIs, and tests.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from airframe.protocol import AgentRuntime


def _builtin_runtime_classes() -> list[type[AgentRuntime]]:
    """All built-in adapter classes, in stable order for menus.

    Imported lazily so plain ``import airframe`` doesn't force the
    adapter modules (which themselves lazy-import vendor SDKs) into
    memory.
    """
    from airframe.adapters.claude_code import ClaudeCodeRuntime
    from airframe.adapters.codex import CodexRuntime
    from airframe.adapters.copilot import CopilotRuntime
    from airframe.adapters.opencode_zen import OpenCodeZenRuntime

    return [ClaudeCodeRuntime, CopilotRuntime, CodexRuntime, OpenCodeZenRuntime]


def _is_available(runtime_cls: type[AgentRuntime]) -> bool:
    """``True`` when this adapter's vendor SDK is importable.

    Adapters declare their required package via ``REQUIRES_PACKAGE``.
    We use :func:`importlib.util.find_spec` (no import side effects)
    so calling :func:`list_providers` doesn't pull in SDKs the
    consumer hasn't asked for.
    """
    pkg = getattr(runtime_cls, "REQUIRES_PACKAGE", None)
    if pkg is None:
        return True
    return importlib.util.find_spec(pkg) is not None


def list_providers(*, installed_only: bool = True) -> list[str]:
    """Return canonical provider IDs servable on this machine.

    Args:
        installed_only: When ``True`` (default), filter out providers
            whose adapter SDK isn't installed. When ``False``, return
            every provider an airframe adapter declares — useful for
            documentation / discovery.

    Returns:
        Sorted list of canonical provider IDs. Empty when
        ``installed_only=True`` and no adapter SDKs are installed
        (which is the honest signal that the consumer needs at least
        one extra).
    """
    classes = _builtin_runtime_classes()
    if installed_only:
        classes = [cls for cls in classes if _is_available(cls)]
    return sorted({cls.PROVIDER_ID for cls in classes})  # type: ignore[attr-defined]


def runtime_for(provider_id: str) -> type[AgentRuntime]:
    """Return the adapter class that serves ``provider_id``.

    Looks up the canonical provider ID across built-in adapters and
    returns the matching class (uninstantiated). Consumers instantiate
    it with whatever auth / model / config they need.

    Raises:
        ValueError: when no adapter serves ``provider_id``.
        ImportError: when an adapter serves the provider but its
            vendor SDK isn't installed. The error message names the
            extra the consumer needs.
    """
    for cls in _builtin_runtime_classes():
        if provider_id != cls.PROVIDER_ID:  # type: ignore[attr-defined]
            continue
        if not _is_available(cls):
            extra = getattr(cls, "EXTRA_NAME", cls.__name__)
            pkg = getattr(cls, "REQUIRES_PACKAGE", "<unknown>")
            raise ImportError(
                f"Provider {provider_id!r} is served by "
                f"{cls.__name__}, which requires the {pkg!r} package. "
                f"Install with: pip install airframe-agents[{extra}]"
            )
        return cls
    available = list_providers(installed_only=False)
    raise ValueError(
        f"No airframe adapter serves provider {provider_id!r}. Known providers: {available}"
    )


__all__ = ["list_providers", "runtime_for"]
