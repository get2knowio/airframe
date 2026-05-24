"""Portable slash-commands surface (filesystem-based, scaffolding).

Slash commands and [Agent Skills](agent-skills) are sibling
conventions: both are filesystem-defined assets in a folder + YAML
frontmatter + Markdown body shape, both ride on the same
``.claude/`` / ``.opencode/`` / ``.agents/`` layered discovery paths
that four of airframe's adapters already implement. They differ in
*who triggers them*:

* **Skills** are model-invoked autonomously (the model loads a
  skill into context when it decides one is relevant).
* **Slash commands** are user-invoked explicitly (the consumer's
  UI sends ``/refactor file.py`` and the adapter expands the
  command body into a prompt before forwarding).

That difference means slash commands need an *invocation* surface
(the consumer's UI palette must enumerate available commands and
let the user pick one) on top of the *configuration* surface that
Skills need.

**Phase 6 scope — scaffolding only.** This module ships
:class:`SlashCommandsConfig` + :data:`Feature.SLASH_COMMANDS` so
consumer code can plan against the namespace, but no adapter
currently flips the feature flag to ``True``. The actual filesystem
discovery + invocation surface is deferred until a consumer needs
it. Per the codebase pattern documented in
:mod:`airframe.features`: "Today only one feature returns True;
other enum members exist to lock the names, and return False."

When a real implementation lands, the per-adapter mapping mirrors
Skills (Claude / GitHub Copilot / Kimi / OpenCode-server will wire
the filesystem discovery; Bedrock / OpenAI-compat / OpenRouter
return ``False``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class SlashCommandsConfig:
    """Portable slash-command configuration.

    Attributes:
        enabled: Which commands the session may invoke.
            ``"all"`` enables every discovered command;
            ``list[str]`` enables only those by name;
            ``None`` disables the slash-command feature for this
            session.
        search_paths: Additional directories to scan beyond the
            adapter's defaults. Each path is searched for
            ``commands/*.md`` (or the adapter's equivalent shape).
            ``None`` uses adapter defaults only
            (``.claude/commands/``, ``.opencode/command/``, etc.).
        include_user_global: When ``True`` (default), include the
            user-global command directory (``~/.claude/commands/``
            or equivalent) alongside the project-local paths.
    """

    enabled: Literal["all"] | list[str] | None = "all"
    search_paths: list[Path] | None = None
    include_user_global: bool = True


__all__ = ["SlashCommandsConfig"]
