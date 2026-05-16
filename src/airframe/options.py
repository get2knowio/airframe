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

**Phase 0 ships the scaffold; the bodies are deliberately empty.**
Later phases fill each dataclass as the relevant feature lands:

* Phase 2 may add ``ClaudeOptions.thinking_budget_tokens`` (Claude's
  explicit-budget reasoning form, which doesn't generalise) and
  similar minimal/xhigh outliers.
* Phase 4 may add ``ClaudeOptions.strict_mcp_config`` and per-vendor
  MCP-config knobs that don't fit the portable
  :class:`McpServerRef` shape.
* Phase 5 may add ``ClaudeOptions.fork_session``,
  ``ClaudeOptions.enable_file_checkpointing``,
  ``CodexOptions.skip_git_repo_check``, and so on.

Populating these in Phase 0 without a user would lock field names
without a feature to validate the shape against — see
:doc:`implementation-plan` Phase 0 non-goals.

**Where the options attach** — Phase 0 defines the types but wires
them nowhere; ``AgentRuntime.execute()`` accepts no
``provider_options=`` kwarg today. Phase 1 attaches them to the new
``runtime.session(provider_options=...)`` factory; until then,
constructing one and passing it anywhere will be a static type error.
This is deliberate: defining empty types you can't yet pass avoids
a soft deprecation between v0.3.0 and v0.4.0.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClaudeOptions:
    """Vendor-specific options for :class:`ClaudeCodeRuntime`.

    Empty in Phase 0. Phase 2+ will populate as features land that
    don't fit the portable protocol surface (Claude-only thinking
    budget tokens, file-checkpointing, fork-session, sandbox config,
    plugins, etc.).
    """


@dataclass(frozen=True, slots=True)
class CopilotOptions:
    """Vendor-specific options for :class:`CopilotRuntime`.

    Empty in Phase 0. Phase 2+ will populate (custom-agents config,
    skill directories, elicitation handlers, etc.).
    """


@dataclass(frozen=True, slots=True)
class CodexOptions:
    """Vendor-specific options for :class:`CodexRuntime`.

    Empty in Phase 0. Phase 2+ will populate (``skip_git_repo_check``,
    ``additional_directories``, the future profile-backed
    permission/sandbox configuration as it replaces the legacy
    ``approval_policy`` / ``sandbox_mode`` flags, etc.).
    """


@dataclass(frozen=True, slots=True)
class OpenAICompatOptions:
    """Vendor-specific options for OpenAI-compatible HTTP runtimes.

    Empty in Phase 0. Phase 2+ will populate (``prompt_cache_key``,
    ``prompt_cache_retention``, ``service_tier``, ``store``,
    ``verbosity``, ``safety_identifier``, etc. — OpenAI-only knobs
    that don't generalise to other compat vendors).
    """


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
