"""``ProviderOptions`` — typed vendor-specific extension points.

The answer to the "60-field ``ClaudeAgentOptions`` sprawl" problem
documented in :doc:`feature-roadmap` §6.5. Borrowed directly from
Vercel AI SDK's ``providerOptions`` pattern and the Spring Cloud
Stream binder namespace convention: portable kwargs on the protocol
core, typed-but-optional per-vendor namespaces for everything else.

The five namespaces:

* :class:`ClaudeOptions` — knobs honoured by
  :class:`airframe.adapters.claude_code.ClaudeCodeRuntime`.
* :class:`CopilotOptions` — knobs honoured by
  :class:`airframe.adapters.copilot.CopilotRuntime`.
* :class:`KimiOptions` — knobs honoured by
  :class:`airframe.adapters.kimi.KimiRuntime`.
* :class:`OpenAICompatOptions` — knobs honoured by every subclass of
  :class:`airframe.adapters.openai_compatible.OpenAICompatibleRuntime`
  (today: :class:`~airframe.adapters.opencode_zen.OpenCodeZenRuntime`).
* :class:`BedrockOptions` — knobs honoured by
  :class:`airframe.adapters.bedrock.BedrockRuntime` (per-session
  region override, Bedrock Guardrails policy, performance latency
  hint, and a pass-through to Converse's
  ``additionalModelRequestFields`` for vendor-specific knobs
  airframe doesn't expose first-class).

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
from typing import Any


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


@dataclass(frozen=True, slots=True)
class BedrockOptions:
    """Vendor-specific options for :class:`BedrockRuntime`.

    Attributes:
        region_name: Override the runtime-level AWS region for this
            session. Bedrock catalogs are per-region, so a session
            that targets a model only available in ``us-west-2``
            can override an ``us-east-1`` runtime without rebuilding
            it. When set, the session opens its own
            :class:`aioboto3.Session` client; otherwise sessions
            share the runtime's lazy client.
        inference_profile_arn: When set, replaces the session's
            ``modelId`` on every Converse call with the full
            inference-profile ARN. Lets callers route through
            provisioned-throughput or cross-region inference profiles
            without juggling the prefix in :class:`ProviderModel`.
        guardrail_id: Bedrock Guardrails policy identifier the model
            should run under. Lands as
            ``guardrailConfig.guardrailIdentifier`` on each
            ``converse`` / ``converse_stream`` call.
        guardrail_version: Optional companion version pinned for the
            ``guardrail_id``. Defaults to AWS's "DRAFT" when omitted
            but the field is set; only meaningful when
            ``guardrail_id`` is also set.
        performance_latency: ``"standard"`` or ``"optimized"``.
            Lands as ``performanceConfig.latency``; ``"optimized"``
            opts into Bedrock's latency-optimised inference path
            (limited model availability — silently ignored on
            unsupported models per Bedrock's per-vendor field
            handling).
        additional_model_fields: Pass-through into Converse's
            ``additionalModelRequestFields`` for vendor knobs
            airframe doesn't have first-class support for (Anthropic
            ``top_k``, Meta ``top_p``, Cohere ``search_result_format``,
            etc.). Merged with any ``thinking`` field airframe builds
            from ``thinking=`` — user keys win on collision.
            Document field validity per-vendor: this is the honest
            escape hatch, validation is the caller's problem.

    All fields default to ``None`` / empty — passing
    :class:`BedrockOptions()` is a no-op.
    """

    region_name: str | None = None
    inference_profile_arn: str | None = None
    guardrail_id: str | None = None
    guardrail_version: str | None = None
    performance_latency: str | None = None
    additional_model_fields: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class KimiOptions:
    """Vendor-specific options for :class:`KimiRuntime`.

    Iteration F completes the namespace; every field maps to a
    :meth:`kimi_agent_sdk.Session.create` /
    :meth:`Session.resume` kwarg or to a downstream
    :class:`kimi_cli.config.Config` slot the adapter can't express
    portably.

    Attributes:
        working_directory: Per-session working directory the SDK's
            filesystem-affecting tools operate relative to. Resolves
            on the adapter side via ``KaosPath`` (the SDK's required
            path type) — pass a plain ``str`` or ``pathlib.Path``.
            ``None`` defaults to the current working directory.
        yolo: Auto-approve every tool / shell invocation at the SDK
            boundary. **Mutually exclusive with** ``on_permission=``
            on :meth:`session` — passing both raises
            :class:`~airframe.errors.UnsupportedFeatureError` at
            session-construction time. Default ``False``: when
            ``on_permission`` is ``None`` and ``yolo`` is False the
            adapter still passes ``yolo=True`` to the SDK so it
            doesn't stall waiting for human input
            (a session with no permission policy must auto-approve
            to make any progress). The explicit ``yolo=True``
            option matters when paired with skill / agent-file
            configurations that would otherwise prompt for
            confirmation outside the airframe permission channel.
        additional_mcp_servers: Extra raw MCP-config entries passed
            verbatim to :meth:`Session.create(mcp_configs=...)`
            *after* the entries airframe synthesises from
            :class:`~airframe.tools.McpServerRef`. Each entry should
            match the fastmcp ``MCPConfig`` dict shape
            (``{"mcpServers": {<name>: {<server-config>}}}`` or a
            bare server-config dict). Use this slot for vendor-specific
            knobs (``description``, ``icon``, ``cwd``,
            ``authentication``) that
            :class:`~airframe.tools.McpServerRef` doesn't surface.
        skill_directories: Additional skill directories the Kimi
            agent picks up at session start. Maps to
            :meth:`Session.create(skills_dir=...)` (first entry —
            the SDK accepts a single dir; airframe surfaces a tuple
            so future SDK versions widening the surface require no
            airframe-side change). When empty the SDK falls back to
            its default discovery: the brand-specific kimi/claude/
            codex dirs depending on
            :attr:`Config.merge_all_available_skills`.
        additional_config_fields: Pass-through merged onto the
            in-process :class:`kimi_cli.config.Config` instance the
            adapter constructs (and only when the adapter has reason
            to instantiate one explicitly — Iteration F leaves config
            construction to the SDK's defaults, so this slot is the
            documented escape hatch should a future iteration
            materialise a Config object).

    All fields default to ``None`` / ``False`` / empty — passing
    :class:`KimiOptions()` is a no-op.
    """

    working_directory: str | None = None
    yolo: bool = False
    additional_mcp_servers: tuple[Any, ...] = ()
    skill_directories: tuple[str, ...] = ()
    additional_config_fields: dict[str, Any] | None = None


#: Tagged union of provider-specific options. Adapters
#: :func:`isinstance`-match to decide whether they accept the value
#: passed in. No common base class on purpose — see module docstring.
ProviderOptions = (
    ClaudeOptions | CopilotOptions | OpenAICompatOptions | BedrockOptions | KimiOptions
)


__all__ = [
    "BedrockOptions",
    "ClaudeOptions",
    "CopilotOptions",
    "KimiOptions",
    "OpenAICompatOptions",
    "ProviderOptions",
]
