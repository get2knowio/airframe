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

**Third-party adapter discovery.** In addition to the four built-in
adapters, :func:`list_providers` consults the ``airframe.adapters``
:pep:`621` entry-point group. A third-party package such as
``airframe-adapters-together`` declares its runtime in
``pyproject.toml``::

    [project.entry-points."airframe.adapters"]
    together = "airframe_adapters_together:TogetherRuntime"

…and ``airframe.list_providers()`` picks it up automatically. The
same pip-extras filtering applies — the entry-point runtime declares
its own ``REQUIRES_PACKAGE`` and is hidden from menus when that
package isn't importable. Modelled on SLF4J's :class:`ServiceLoader`
binding discovery and Python's standard
:mod:`importlib.metadata` entry points
(https://packaging.python.org/en/latest/specifications/entry-points/).

Built-in providers shadow third-party entries with the same
``PROVIDER_ID``. That's the conservative default: consumers upgrading
airframe shouldn't have their behaviour silently swapped by an
installed plugin. Third-party authors wanting to *replace* a built-in
must pick a different ``PROVIDER_ID``.
"""

from __future__ import annotations

import importlib.util
import logging
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from airframe.protocol import AgentRuntime

logger = logging.getLogger(__name__)

#: Entry-point group third-party adapters register under. Convention
#: matches the package name (``airframe.adapters``) so the discovery
#: namespace mirrors the module namespace. Public surface — renaming
#: this would invalidate every third-party package's declaration.
ENTRY_POINT_GROUP = "airframe.adapters"


def _builtin_runtime_classes() -> list[type[AgentRuntime]]:
    """All built-in adapter classes, in stable order for menus.

    Imported lazily so plain ``import airframe`` doesn't force the
    adapter modules (which themselves lazy-import vendor SDKs) into
    memory.
    """
    from airframe.adapters.bedrock import BedrockRuntime
    from airframe.adapters.claude_code import ClaudeCodeRuntime
    from airframe.adapters.copilot import CopilotRuntime
    from airframe.adapters.opencode_go import OpenCodeGoRuntime
    from airframe.adapters.opencode_server import OpenCodeServerRuntime
    from airframe.adapters.opencode_zen import OpenCodeZenRuntime
    from airframe.adapters.openrouter import OpenRouterRuntime
    from airframe.adapters.zai import ZaiAnthropicRuntime

    # KimiRuntime is intentionally NOT registered for now. `kimi-agent-sdk`
    # pins `kimi-cli<1.13 → fastmcp 2.12.5 → mcp<1.17`, which can't co-install
    # with `claude-agent-sdk` (`mcp>=1.23`) — see #29. Until upstream ships a
    # release widening those pins, hiding it from discovery keeps it out of
    # menus / `list_providers()` so consumers don't hit the conflict. The
    # adapter module and its `KimiRuntime` export stay importable; re-enable by
    # restoring the `from airframe.adapters.kimi import KimiRuntime` import and
    # the list entry below. Tracking: #29 (mcp), #36 (live gaps), PR #35 (fix).
    return [
        ClaudeCodeRuntime,
        CopilotRuntime,
        OpenCodeServerRuntime,
        OpenCodeZenRuntime,
        OpenCodeGoRuntime,
        OpenRouterRuntime,
        BedrockRuntime,
        ZaiAnthropicRuntime,
        # KimiRuntime,  # disabled pending kimi-agent-sdk mcp alignment (#29)
    ]


def _entry_point_runtime_classes() -> list[type[AgentRuntime]]:
    """Adapter classes registered under the ``airframe.adapters`` group.

    A malformed entry point (import error, target isn't a class, target
    is missing ``PROVIDER_ID``, target collides with a built-in) is
    logged at WARNING and skipped — the rest of discovery keeps
    working. A broken third-party plugin shouldn't crash the
    consumer's menu rendering.
    """
    builtin_ids = {
        cls.PROVIDER_ID  # type: ignore[attr-defined]
        for cls in _builtin_runtime_classes()
    }
    discovered: list[type[AgentRuntime]] = []
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            target = ep.load()
        except Exception as exc:  # noqa: BLE001 — discovery never raises
            logger.warning(
                "airframe.discovery: entry point %r (%s) failed to load: %s",
                ep.name,
                ep.value,
                exc,
            )
            continue
        if not isinstance(target, type):
            logger.warning(
                "airframe.discovery: entry point %r (%s) resolved to %r, not a class; skipping",
                ep.name,
                ep.value,
                target,
            )
            continue
        provider_id = getattr(target, "PROVIDER_ID", None)
        if not isinstance(provider_id, str) or not provider_id:
            logger.warning(
                "airframe.discovery: entry point %r (%s) has no PROVIDER_ID ClassVar; skipping",
                ep.name,
                target.__name__,
            )
            continue
        if provider_id in builtin_ids:
            logger.warning(
                "airframe.discovery: third-party adapter %s declares "
                "PROVIDER_ID=%r which shadows a built-in; skipping",
                target.__name__,
                provider_id,
            )
            continue
        discovered.append(target)
    return discovered


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


def _all_runtime_classes() -> list[type[AgentRuntime]]:
    """Built-in adapters + every adapter registered via entry points.

    Stable order: built-ins first (so a consumer's mental model
    "ClaudeCodeRuntime is always first" survives the addition of any
    third-party adapter), then entry-point adapters in iteration
    order from :func:`importlib.metadata.entry_points`.
    """
    return [*_builtin_runtime_classes(), *_entry_point_runtime_classes()]


def list_providers(*, installed_only: bool = True) -> list[str]:
    """Return canonical provider IDs servable on this machine.

    Args:
        installed_only: When ``True`` (default), filter out providers
            whose adapter SDK isn't installed. When ``False``, return
            every provider an airframe adapter declares — useful for
            documentation / discovery.

    Returns:
        Sorted list of canonical provider IDs. Includes both built-in
        and third-party adapters registered under the
        ``airframe.adapters`` entry-point group. Empty when
        ``installed_only=True`` and no adapter SDKs are installed
        (which is the honest signal that the consumer needs at least
        one extra).
    """
    classes = _all_runtime_classes()
    if installed_only:
        classes = [cls for cls in classes if _is_available(cls)]
    return sorted({cls.PROVIDER_ID for cls in classes})  # type: ignore[attr-defined]


def runtime_for(provider_id: str) -> type[AgentRuntime]:
    """Return the adapter class that serves ``provider_id``.

    Looks up the canonical provider ID across built-in adapters and
    third-party adapters registered via the ``airframe.adapters``
    entry-point group, returning the matching class (uninstantiated).
    Consumers instantiate it with whatever auth / model / config they
    need.

    Raises:
        ValueError: when no adapter serves ``provider_id``.
        ImportError: when an adapter serves the provider but its
            vendor SDK isn't installed. The error message names the
            extra the consumer needs.
    """
    for cls in _all_runtime_classes():
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


__all__ = ["ENTRY_POINT_GROUP", "list_providers", "runtime_for"]
