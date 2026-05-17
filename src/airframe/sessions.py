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
    from collections.abc import AsyncIterator

    from pydantic import BaseModel

    from airframe.events import RuntimeEvent
    from airframe.inputs import Prompt
    from airframe.protocol import AgentRuntime, ProviderModel, RuntimeResult
    from airframe.thinking import ThinkingMode


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
    "_coerce_prompt_or_raise",
    "_open_thin_session",
    "_split_prompt_parts",
]
