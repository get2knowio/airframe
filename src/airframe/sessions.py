"""Shared :class:`AgentSession` helpers.

Phase 1 iteration B lands the :meth:`AgentRuntime.session` factory and
gives every built-in adapter a minimal session implementation. The
real per-vendor session lifecycles — Claude Code's ``ClaudeSDKClient``
ownership migration, Copilot's ``CopilotSession`` resume plumbing,
Codex's per-turn subprocess, OpenAI-compatible client-side
``messages=[]`` multi-turn — land in subsequent iterations as each
adapter's Feature bits (``STREAMING``, ``SESSION_RESUME``, ``CANCEL``)
flip on.

Iteration B's :class:`_ThinAgentSession` is the placeholder for that
work: a thin lifecycle wrapper that forwards :meth:`execute` to the
underlying runtime, synthesises :meth:`stream` as a single
:class:`~airframe.events.TurnComplete`, and no-ops :meth:`cancel` /
:meth:`close`. Adapter sessions today are all instances of this class;
later iterations replace per-adapter sessions with bespoke
implementations as needed.

The class is intentionally private (leading underscore) — third-party
adapter authors should target :class:`~airframe.protocol.AgentSession`
directly, not subclass this helper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from airframe.errors import UnsupportedFeatureError
from airframe.events import TurnComplete
from airframe.features import Feature
from airframe.inputs import FileInput, ImageInput

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from pydantic import BaseModel

    from airframe.events import RuntimeEvent
    from airframe.inputs import Prompt
    from airframe.protocol import AgentRuntime, ProviderModel, RuntimeResult
    from airframe.thinking import ThinkingMode
    from airframe.tools import FunctionTool, McpServerRef


def _coerce_prompt_or_raise(prompt: Prompt, *, adapter_label: str) -> str:
    """Return ``prompt`` as a plain ``str`` or raise for list-shaped prompts.

    Kept for the few call sites that don't yet declare any input
    capability. Adapters wiring :class:`~airframe.inputs.ImageInput` /
    :class:`~airframe.inputs.FileInput` should use
    :func:`_split_prompt_parts` instead so the helper handles the
    feature-gating and the bytes/URL deferral consistently.
    """
    if isinstance(prompt, str):
        return prompt
    raise UnsupportedFeatureError(
        f"{adapter_label}: list-shaped prompts (vision / file inputs) are not "
        f"wired yet on this adapter. Check runtime.supports(Feature.VISION_INPUT) "
        f"/ runtime.supports(Feature.FILE_INPUT) before passing list[PromptPart].",
        feature="vision_input",
    )


def _split_prompt_parts(
    prompt: Prompt,
    *,
    adapter_label: str,
    supports_vision: bool,
    supports_file: bool,
) -> tuple[str, list[ImageInput], list[FileInput]]:
    """Split a polymorphic ``prompt`` into ``(text, images, files)``.

    Phase 2 Iteration C — the shared helper every vision-capable
    adapter uses to translate :class:`~airframe.inputs.Prompt` into the
    three buckets they actually route over their vendor SDK. String
    parts are joined with ``"\\n\\n"`` separators so adapters can pass
    plain text along the existing prompt slot;
    :class:`~airframe.inputs.ImageInput` / :class:`~airframe.inputs.FileInput`
    parts come back as typed lists for the caller's vendor-specific
    attachment plumbing.

    Each :class:`ImageInput` variant (``path=``, ``bytes_=``, ``url=``)
    passes through to the adapter — per-vendor support differs, so
    each adapter's content builder decides which variants it can
    natively serve and raises :class:`UnsupportedFeatureError` on the
    rest. This helper only gates the *feature category*
    (vision vs file).

    Args:
        prompt: A bare ``str`` (always returned as ``(prompt, [], [])``)
            or a ``list[PromptPart]``.
        adapter_label: Adapter name baked into the error message so
            consumers know which runtime declined.
        supports_vision: Whether this adapter declares
            :data:`~airframe.features.Feature.VISION_INPUT`. ``False``
            and an :class:`ImageInput` part raises.
        supports_file: Whether this adapter declares
            :data:`~airframe.features.Feature.FILE_INPUT`. ``False``
            and a :class:`FileInput` part raises.

    Raises:
        UnsupportedFeatureError: A part type the adapter doesn't
            advertise as a Feature.
        TypeError: An element of the list is neither ``str`` nor a
            known :data:`~airframe.inputs.PromptPart` variant.
    """
    if isinstance(prompt, str):
        return prompt, [], []

    text_parts: list[str] = []
    images: list[ImageInput] = []
    files: list[FileInput] = []
    for part in prompt:
        if isinstance(part, str):
            text_parts.append(part)
        elif isinstance(part, ImageInput):
            if not supports_vision:
                raise UnsupportedFeatureError(
                    f"{adapter_label}: ImageInput parts are not wired on this "
                    f"adapter. Check runtime.supports(Feature.VISION_INPUT) "
                    f"first.",
                    feature=Feature.VISION_INPUT,
                )
            images.append(part)
        elif isinstance(part, FileInput):
            if not supports_file:
                raise UnsupportedFeatureError(
                    f"{adapter_label}: FileInput parts are not wired on this "
                    f"adapter. Check runtime.supports(Feature.FILE_INPUT) "
                    f"first.",
                    feature=Feature.FILE_INPUT,
                )
            files.append(part)
        else:
            raise TypeError(
                f"{adapter_label}: unrecognised PromptPart {type(part).__name__}; "
                f"expected str | ImageInput | FileInput"
            )
    return "\n\n".join(text_parts), images, files


def _check_tools_supported(
    tools: list[FunctionTool] | None,
    *,
    adapter_label: str,
    feature_supported: bool,
) -> None:
    """Gate ``session(tools=...)`` against the adapter's capability flag.

    Phase 3 Iteration A scaffolds the ``tools=`` kwarg on every
    :meth:`AgentRuntime.session` signature but defers the per-adapter
    wiring to Iterations B (OpenAI-compat), C (Claude + Copilot), and
    D (Codex declination). Until each adapter flips
    :data:`~airframe.features.Feature.TOOLS_FUNCTION` True, a non-None
    ``tools=`` raises here so consumer code gets a clear capability
    decline rather than a silently-ignored kwarg.

    A ``None`` (or empty) list is always permitted — it's the no-op
    default and doesn't exercise any unwired surface.

    Args:
        tools: The list passed to ``session(tools=...)``.
        adapter_label: Adapter name for the error message.
        feature_supported: ``runtime.supports(Feature.TOOLS_FUNCTION)``
            — passed in by the caller so this helper doesn't need a
            runtime reference.

    Raises:
        UnsupportedFeatureError: ``tools`` is non-None / non-empty
            *and* the adapter hasn't flipped ``TOOLS_FUNCTION``.
    """
    if not tools:
        return
    if feature_supported:
        return
    raise UnsupportedFeatureError(
        f"{adapter_label}: tools= is not wired on this adapter yet. "
        f"Check runtime.supports(Feature.TOOLS_FUNCTION) before "
        f"passing tools=[FunctionTool(...)].",
        feature=Feature.TOOLS_FUNCTION,
    )


_MCP_TRANSPORT_TO_FEATURE: dict[str, Feature] = {
    "stdio": Feature.TOOLS_MCP_STDIO,
    "http": Feature.TOOLS_MCP_HTTP,
    "sse": Feature.TOOLS_MCP_SSE,
}


def _compose_mcp_headers(ref: McpServerRef) -> dict[str, str]:
    """Compose the ``headers`` dict for a network-transport
    :class:`~airframe.tools.McpServerRef`.

    Shared by every adapter that wires MCP. :attr:`McpServerRef.auth_token`
    becomes ``Authorization: Bearer <token>``; caller-supplied
    :attr:`McpServerRef.headers` layers on top so an explicit
    ``Authorization`` value wins on collision (the shorthand is just a
    shorthand). Returns an empty dict when neither field is set so the
    caller can drop the ``headers`` key entirely and keep the vendor
    wire shape minimal.
    """
    merged: dict[str, str] = {}
    if ref.auth_token is not None:
        merged["Authorization"] = f"Bearer {ref.auth_token}"
    if ref.headers:
        merged.update(ref.headers)
    return merged


def _mcp_servers_fingerprint(refs: list[McpServerRef] | None) -> str:
    """Build a deterministic, secret-free fingerprint of an MCP refs list.

    Shared by every adapter that bakes ``mcp_servers`` into a vendor
    session at connect time (Claude's
    :attr:`ClaudeAgentOptions.mcp_servers`, Copilot's
    :meth:`CopilotClient.create_session` ``mcp_servers=`` kwarg). The
    fingerprint participates from each ref's ``name``, ``transport``,
    ``command``, ``url``, and the *sorted keys* of ``headers`` —
    never the header values, never ``auth_token``. That way sensitive
    material doesn't enter the in-process cache identity, and
    rotating a bearer token doesn't accidentally invalidate the cache
    (caller can ``close()`` the session if they want a hard reset).
    """
    if not refs:
        return "__no_mcp_servers__"
    parts: list[str] = []
    for ref in refs:
        header_keys = ",".join(sorted((ref.headers or {}).keys()))
        cmd = ",".join(ref.command) if ref.command else ""
        parts.append(f"{ref.name}|{ref.transport}|{cmd}|{ref.url or ''}|{header_keys}")
    return "||".join(parts)


def _check_mcp_servers_supported(
    refs: list[McpServerRef] | None,
    *,
    adapter_label: str,
    supports: Callable[[Feature], bool],
) -> None:
    """Gate ``session(mcp_servers=...)`` against per-transport capability flags.

    Phase 4 Iteration A scaffolds the ``mcp_servers=`` kwarg on every
    :meth:`AgentRuntime.session` signature but defers the per-adapter
    wiring to Iterations B (Claude — all three transports), C (Copilot
    — stdio + http; SSE declines), and D (Codex + OpenAI-compat
    permanent declines). Until each adapter flips the matching
    :data:`~airframe.features.Feature.TOOLS_MCP_STDIO` /
    :data:`~airframe.features.Feature.TOOLS_MCP_HTTP` /
    :data:`~airframe.features.Feature.TOOLS_MCP_SSE` True, a non-empty
    list raises here with the *specific* feature of the first
    unsupported ref — so a mixed list (stdio + http) on a stdio-only
    adapter raises with ``feature=TOOLS_MCP_HTTP`` when the http ref
    comes second.

    A ``None`` (or empty) list is always permitted — it's the no-op
    default and doesn't exercise any unwired surface.

    Args:
        refs: The list passed to ``session(mcp_servers=...)``.
        adapter_label: Adapter name for the error message.
        supports: Callable answering ``runtime.supports(feature)`` —
            passed in so this helper doesn't need a runtime reference
            (same shape as :func:`_check_tools_supported`).

    Raises:
        UnsupportedFeatureError: ``refs`` contains at least one entry
            whose transport's :class:`Feature` flag returns ``False``.
    """
    if not refs:
        return
    for ref in refs:
        feature = _MCP_TRANSPORT_TO_FEATURE.get(ref.transport)
        if feature is None:  # pragma: no cover — Literal narrows this
            raise UnsupportedFeatureError(
                f"{adapter_label}: McpServerRef(name={ref.name!r}) declares "
                f"unknown transport {ref.transport!r}; expected one of "
                f"'stdio', 'http', 'sse'.",
                feature=None,
            )
        if supports(feature):
            continue
        raise UnsupportedFeatureError(
            f"{adapter_label}: mcp_servers= entry {ref.name!r} uses "
            f"transport {ref.transport!r}, which is not wired on this "
            f"adapter yet. Check runtime.supports(Feature.{feature.name}) "
            f"before passing mcp_servers=[McpServerRef(...)].",
            feature=feature,
        )


class _ThinAgentSession:
    """Iteration B :class:`AgentSession` — thin wrapper over ``runtime.execute()``.

    Does the minimum to satisfy the protocol:

    * :meth:`execute` forwards to ``runtime.execute()`` with the
      session's bound ``system`` / ``model`` carried through.
    * :meth:`stream` calls :meth:`execute` and yields a single
      :class:`~airframe.events.TurnComplete` carrying the result.
      Iteration C+ replaces this with real per-vendor streaming as
      :data:`~airframe.features.Feature.STREAMING` flips on for each
      adapter.
    * :meth:`cancel` is a no-op when no turn is in flight (the only
      state this class tracks). Calling it during an in-flight turn
      raises :class:`~airframe.errors.UnsupportedFeatureError` — every
      adapter declares ``Feature.CANCEL=False`` today.
    * :meth:`close` flips an idempotent ``_closed`` flag; subsequent
      :meth:`execute` / :meth:`stream` raise :class:`RuntimeError`.

    The underlying runtime owns process / HTTP resources — closing a
    session never tears those down. That's :meth:`AgentRuntime.close`'s
    job.
    """

    def __init__(
        self,
        runtime: AgentRuntime,
        *,
        system: str | None = None,
        model: ProviderModel | None = None,
    ) -> None:
        self._runtime = runtime
        self._system = system
        self._model = model
        self._closed = False
        self._in_flight = False
        self.id: str | None = None

    async def execute(
        self,
        prompt: Prompt,
        *,
        schema: type[BaseModel] | None = None,
        thinking: ThinkingMode = None,
        timeout: float = 600.0,
    ) -> RuntimeResult:
        if self._closed:
            raise RuntimeError("session is closed")
        prompt_str = _coerce_prompt_or_raise(prompt, adapter_label="_ThinAgentSession")
        self._in_flight = True
        try:
            return await self._runtime.execute(
                prompt_str,
                schema=schema,
                system=self._system,
                model=self._model,
                thinking=thinking,
                timeout=timeout,
            )
        finally:
            self._in_flight = False

    async def stream(
        self,
        prompt: Prompt,
        *,
        schema: type[BaseModel] | None = None,
        thinking: ThinkingMode = None,
        timeout: float = 600.0,
    ) -> AsyncIterator[RuntimeEvent]:
        # Phase 1 Iteration B placeholder: synthesise the stream as a
        # single TurnComplete. Real per-vendor delta streaming lands in
        # each adapter's bespoke session.
        result = await self.execute(prompt, schema=schema, thinking=thinking, timeout=timeout)
        yield TurnComplete(result=result)

    async def cancel(self) -> None:
        if not self._in_flight:
            return
        raise UnsupportedFeatureError(
            "cancel() is not wired on this adapter yet; "
            "check runtime.supports(Feature.CANCEL) before calling",
            feature="cancel",
        )

    async def close(self) -> None:
        self._closed = True

    def unwrap(self, cls: type[Any]) -> Any:
        # The thin wrapper has no vendor-specific state to expose.
        # Identity-cast is the only supported case.
        if isinstance(self, cls):
            return self
        raise TypeError(
            f"{type(self).__name__} has no vendor-specific session object to "
            f"unwrap to {cls!r}; reach the runtime's vendor objects via "
            f"runtime.unwrap() instead."
        )


def _open_thin_session(
    runtime: AgentRuntime,
    *,
    resume: str | None,
    system: str | None,
    model: ProviderModel | None,
    provider_options: Any | None,
) -> _ThinAgentSession:
    """Build a :class:`_ThinAgentSession`, enforcing Iteration B's gates.

    Centralises the ``resume=`` / ``provider_options=`` handling so
    every adapter's :meth:`session` factory is one-liner that calls
    this. Subsequent iterations may replace per-adapter factories with
    bespoke session classes; this helper exists for the period where
    every adapter shares the thin implementation.

    Raises:
        NotImplementedError: when ``resume`` is non-None. Session
            resume lands as Phase 1's Iteration C work; until then,
            adapters declaring :data:`~airframe.features.Feature.SESSION_RESUME`
            don't exist.
    """
    if resume is not None:
        raise NotImplementedError(
            "session(resume=...) is not wired yet; "
            "Iteration C of Phase 1 lands per-adapter resume. "
            "Check runtime.supports(Feature.SESSION_RESUME) first."
        )
    # provider_options accepted but unused — Phase 2+ fills each
    # ProviderOptions dataclass as the corresponding feature lands.
    del provider_options
    return _ThinAgentSession(runtime, system=system, model=model)


__all__ = [
    "_ThinAgentSession",
    "_check_mcp_servers_supported",
    "_check_tools_supported",
    "_coerce_prompt_or_raise",
    "_compose_mcp_headers",
    "_mcp_servers_fingerprint",
    "_open_thin_session",
    "_split_prompt_parts",
]
