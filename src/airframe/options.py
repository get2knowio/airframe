"""``ProviderOptions`` — typed vendor-specific extension points.

The answer to the "60-field ``ClaudeAgentOptions`` sprawl" problem
documented in :doc:`feature-roadmap` §6.5. Borrowed directly from
Vercel AI SDK's ``providerOptions`` pattern and the Spring Cloud
Stream binder namespace convention: portable kwargs on the protocol
core, typed-but-optional per-vendor namespaces for everything else.

The four namespaces:

* :class:`ClaudeOptions` — knobs honoured by
  :class:`airframe.adapters.claude_code.ClaudeCodeRuntime`.
* :class:`CopilotOptions` — knobs honoured by
  :class:`airframe.adapters.copilot.CopilotRuntime`.
* :class:`CodexOptions` — knobs honoured by
  :class:`airframe.adapters.codex.CodexRuntime`.
* :class:`OpenAICompatOptions` — knobs honoured by every subclass of
  :class:`airframe.adapters.openai_compatible.OpenAICompatibleRuntime`
  (today: :class:`~airframe.adapters.opencode_zen.OpenCodeZenRuntime`).

Each dataclass is :func:`frozen <dataclasses.dataclass>` /
``slots=True`` — same discipline as :class:`~airframe.cost.CostRecord`,
:class:`~airframe.protocol.RuntimeResult`, and
:class:`~airframe.protocol.ProviderModel`. Tagged union (no common
base class) so static type checkers catch
``CopilotRuntime(provider_options=ClaudeOptions(...))`` at lint time
— the failure mode JMS' untyped ``setStringProperty`` notoriously
allowed.

**Field-population policy.** Fields land when a portable surface
can't honestly express the knob (e.g. Claude's ``fork_session``,
``append_system_prompt``, ``strict_mcp_config`` — none of which
generalise across the other three adapters). Once any one field
lands per dataclass the namespace's *shape* is validated; further
fields are additive. Mismatched types raise
:class:`~airframe.errors.UnsupportedFeatureError` at the adapter
boundary (e.g. passing :class:`ClaudeOptions` to
:class:`CopilotRuntime.session`).

**Where the options attach.**
:meth:`AgentRuntime.session(provider_options=...)` and each adapter's
:meth:`session` accept a :class:`ProviderOptions` value (or
``None``). Per-execute / per-stream override is **not** supported
today; per-vendor knobs that vary per turn instead get a portable
kwarg on :meth:`AgentSession.execute` / :meth:`AgentSession.stream`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClaudeOptions:
    """Vendor-specific options for :class:`ClaudeCodeRuntime`.

    Attributes:
        append_system_prompt: Text appended to the (resolved) system
            prompt at session-build time. Distinct from ``system=``
            on :meth:`session`, which *replaces* whatever Claude's
            CLI default would be. Use this when you want to keep
            Claude's default system prompt and just bolt on
            project-specific context. Maps to
            :attr:`ClaudeAgentOptions.append_system_prompt` (Claude
            Agent SDK 0.2+).
        fork_session: When ``resume=<id>`` is supplied to
            :meth:`session`, ``fork_session=True`` resumes a fresh
            *copy* of that session — the parent's state is preserved
            and the new turns land on the fork. Equivalent to
            ``git checkout -b`` on a session. Maps to
            :attr:`ClaudeAgentOptions.fork_session`. Ignored when
            ``resume=`` is ``None``.
        strict_mcp_config: When ``True``, Claude's CLI rejects any
            MCP server reference whose advertised tool set doesn't
            match the runtime's compiled MCP config — fail closed
            instead of silently dropping unknown tools. Maps to
            :attr:`ClaudeAgentOptions.strict_mcp_config`. Defaults
            to ``False`` (CLI default) so existing
            :class:`McpServerRef` consumers are unaffected.
    """

    append_system_prompt: str | None = None
    fork_session: bool = False
    strict_mcp_config: bool = False


@dataclass(frozen=True, slots=True)
class CopilotOptions:
    """Vendor-specific options for :class:`CopilotRuntime`.

    Attributes:
        available_tools: Allowlist of Copilot built-in tool names the
            model may invoke (e.g. ``("read", "write", "shell")``).
            ``None`` (default) means "no allowlist — let the SDK's
            default policy apply". Empty tuple means "no built-in
            tools at all". Maps to
            :attr:`CopilotClient.create_session.available_tools`.
        excluded_tools: Denylist of Copilot built-in tool names the
            model may NOT invoke. Applied after ``available_tools``.
            Maps to
            :attr:`CopilotClient.create_session.excluded_tools`.
        skill_directories: Extra directories to scan for skill
            definitions (Copilot's skill packs). Maps to
            :attr:`CopilotClient.create_session.skill_directories`.
        working_directory: Override for the working directory the
            Copilot CLI subprocess runs in. ``None`` inherits the
            parent process's cwd. Maps to
            :attr:`CopilotClient.create_session.working_directory`.
    """

    available_tools: tuple[str, ...] | None = None
    excluded_tools: tuple[str, ...] = ()
    skill_directories: tuple[str, ...] = ()
    working_directory: str | None = None


@dataclass(frozen=True, slots=True)
class CodexOptions:
    """Vendor-specific options for :class:`CodexRuntime`.

    Attributes:
        working_directory: Override for the Codex CLI's working
            directory. ``None`` inherits the parent process's cwd.
            Maps to :attr:`ThreadOptions.working_directory`.
        additional_directories: Extra directories Codex is allowed
            to read from / write into (subject to ``sandbox_mode``).
            Maps to :attr:`ThreadOptions.additional_directories`.
        network_access_enabled: When ``True``, Codex's sandboxed
            shell may make outbound network calls. ``False``
            (default) matches Codex CLI behaviour. Maps to
            :attr:`ThreadOptions.network_access_enabled`.
        web_search_enabled: When ``True``, the model may invoke
            Codex's built-in web-search tool. ``False`` (default)
            keeps web search off. Maps to
            :attr:`ThreadOptions.web_search_enabled`.
    """

    working_directory: str | None = None
    additional_directories: tuple[str, ...] = ()
    network_access_enabled: bool = False
    web_search_enabled: bool = False


@dataclass(frozen=True, slots=True)
class OpenAICompatOptions:
    """Vendor-specific options for OpenAI-compatible HTTP runtimes.

    Most of these fields are OpenAI-only — compat vendors (Together,
    Groq, Fireworks, OpenRouter, OpenCodeZen) silently ignore
    unrecognised kwargs in their server-side validation. Passing a
    field a vendor doesn't honour is a no-op, not an error.

    Attributes:
        prompt_cache_key: Explicit cache key for OpenAI's prompt
            caching. Reuse the same key across calls to maximise cache
            hits. Lands as the ``prompt_cache_key=`` kwarg on
            :meth:`chat.completions.create`.
        prompt_cache_retention: ``"in_memory"`` or ``"24h"`` — how
            long OpenAI keeps the cached prompt. Lands as the
            ``prompt_cache_retention=`` kwarg.
        service_tier: ``"auto"`` / ``"default"`` / ``"flex"`` /
            ``"priority"`` — OpenAI service tier hint. Lands as the
            ``service_tier=`` kwarg.
        safety_identifier: Opaque per-user identifier OpenAI uses
            for abuse-detection routing. Lands as the
            ``safety_identifier=`` kwarg.
        verbosity: ``"low"`` / ``"medium"`` / ``"high"`` — OpenAI
            response-length hint. Lands as the ``verbosity=`` kwarg.
        store: When ``True``, OpenAI persists the request/response
            pair for the Responses API to retrieve later. ``None``
            (default) keeps the vendor default.
    """

    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None
    service_tier: str | None = None
    safety_identifier: str | None = None
    verbosity: str | None = None
    store: bool | None = None


#: Tagged union of provider-specific options. Adapters
#: :func:`isinstance`-match to decide whether they accept the value
#: passed in. No common base class on purpose — see module docstring.
ProviderOptions = ClaudeOptions | CopilotOptions | CodexOptions | OpenAICompatOptions


__all__ = [
    "ClaudeOptions",
    "CodexOptions",
    "CopilotOptions",
    "OpenAICompatOptions",
    "ProviderOptions",
]
