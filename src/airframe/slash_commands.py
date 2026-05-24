"""Portable slash-commands discovery surface (filesystem-based).

Slash commands and [Agent Skills](agent-skills) are sibling
conventions: both are filesystem-defined assets in a folder + YAML
frontmatter + Markdown body shape, both ride on the same
``.claude/`` / ``.opencode/`` / ``.agents/`` layered discovery paths
that four of airframe's adapters already implement. They differ in
*who triggers them*:

* **Skills** are model-invoked autonomously (the model loads a
  skill into context when it decides one is relevant).
* **Slash commands** are user-invoked explicitly (the consumer's UI
  enumerates them in a palette and the user picks one).

This module ships the *discovery* half — a cross-cutting filesystem
walker that returns :class:`SlashCommand` objects describing every
command the consumer's user has authored. The consumer's UI surfaces
them in a palette; when the user picks one, the consumer expands the
:attr:`SlashCommand.body` template (substituting any args) and calls
the existing :meth:`~airframe.protocol.AgentSession.execute` with the
expanded text. There's no separate invocation channel — slash
commands are just stored prompt templates.

The *configuration* half — :class:`SlashCommandsConfig` — controls
*which* directories to search and *which* commands to expose. Pass
it once at ``runtime.session(slash_commands=...)``; the session
stashes it and uses it on every :meth:`list_slash_commands` call.

Discovery convention (matching Claude Code / OpenCode / Codex):

* Project-local: ``.claude/commands/*.md``, ``.opencode/command/*.md``,
  ``.agents/commands/*.md``, walked upward from the session's
  ``cwd`` to the git worktree root.
* User-global: ``~/.claude/commands/*.md``, ``~/.opencode/command/*.md``
  when :attr:`SlashCommandsConfig.include_user_global` is ``True``
  (the default).
* Extra paths from :attr:`SlashCommandsConfig.search_paths` join
  the search.

Each file is parsed for a minimal YAML frontmatter block
(``---``-delimited key/value pairs at the top) plus a Markdown
body. Frontmatter values are kept as raw strings; consumers wanting
structured metadata can parse them further.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class SlashCommandsConfig:
    """Portable slash-command configuration.

    Attributes:
        enabled: Which commands the session may surface.
            ``"all"`` enables every discovered command;
            ``list[str]`` enables only those by name;
            ``None`` disables the slash-command feature for this
            session.
        search_paths: Additional directories to scan beyond the
            adapter's defaults. Each path is searched for
            ``commands/*.md``.
            ``None`` uses adapter defaults only
            (``.claude/commands/``, ``.opencode/command/``, etc.).
        include_user_global: When ``True`` (default), include the
            user-global command directory (``~/.claude/commands/``
            or equivalent) alongside the project-local paths.
    """

    enabled: Literal["all"] | list[str] | None = "all"
    search_paths: list[Path] | None = None
    include_user_global: bool = True


@dataclass(frozen=True, slots=True)
class SlashCommand:
    """One discovered slash command.

    Attributes:
        name: Command identifier — typically the file stem
            (``refactor.md`` → ``"refactor"``) unless the
            frontmatter explicitly sets a different ``name`` field.
        description: Human-readable description from the frontmatter
            (used in the consumer's palette UI). ``None`` when the
            command file omits a ``description`` field.
        body: The Markdown body — everything after the closing
            ``---`` of the frontmatter block. Consumers substitute
            their own args into this template before passing to
            :meth:`execute`.
        source_path: Filesystem path the command was loaded from.
            Useful for "edit this command" UX and for de-duplication
            (later-found commands with the same name override
            earlier ones).
        frontmatter: Raw frontmatter as a ``dict[str, str]`` —
            unparsed values. Consumers that need typed metadata
            parse this further. Empty dict when the file has no
            frontmatter block.
    """

    name: str
    description: str | None
    body: str
    source_path: Path
    frontmatter: dict[str, str] = field(default_factory=dict)


#: Vendor-conventional locations for project-local command files,
#: relative to ``cwd``. Ordered by priority (later wins on name
#: collision because consumers expect "more specific" paths to
#: override more general ones).
_PROJECT_RELATIVE_DIRS: tuple[str, ...] = (
    ".agents/commands",
    ".opencode/command",
    ".claude/commands",
)

#: Vendor-conventional locations for user-global command files,
#: relative to ``~``. Ordered by priority (later wins).
_USER_GLOBAL_DIRS: tuple[str, ...] = (
    ".agents/commands",
    ".config/opencode/command",
    ".claude/commands",
)


def discover(
    config: SlashCommandsConfig | None = None,
    *,
    cwd: Path | None = None,
) -> list[SlashCommand]:
    """Walk the filesystem and return every visible slash command.

    Discovery proceeds in three passes (later passes override earlier
    ones on name collision — the standard "more specific path wins"
    convention):

    1. User-global directories (``~/.claude/commands/``, ...) when
       :attr:`SlashCommandsConfig.include_user_global` is ``True``.
    2. Project-local directories walked upward from ``cwd`` to the
       git worktree root (or filesystem root if no ``.git`` found).
    3. Additional :attr:`SlashCommandsConfig.search_paths`.

    The returned list is filtered against
    :attr:`SlashCommandsConfig.enabled` (``"all"`` keeps everything,
    a list keeps only those names, ``None`` returns an empty list)
    and sorted by command name.

    Args:
        config: Optional :class:`SlashCommandsConfig`. ``None`` uses
            the default: ``enabled="all"``, no extra search paths,
            ``include_user_global=True``.
        cwd: Starting directory for the project-local walk. ``None``
            uses :func:`os.getcwd`.

    Returns:
        Sorted list of :class:`SlashCommand` objects. Empty when no
        commands are discovered or ``config.enabled is None``.
    """
    if config is None:
        config = SlashCommandsConfig()
    if config.enabled is None:
        return []

    cwd_path = cwd or Path(os.getcwd())
    # Name-keyed dict so later-found commands override earlier ones
    # (more-specific path wins). Iteration order is insertion order,
    # which we sort at the end anyway.
    by_name: dict[str, SlashCommand] = {}

    if config.include_user_global:
        home = Path.home()
        for rel in _USER_GLOBAL_DIRS:
            for cmd in _scan_directory(home / rel):
                by_name[cmd.name] = cmd

    # Project walk: from cwd up to the git worktree root (or fs root).
    for project_root in _walk_up_to_worktree(cwd_path):
        for rel in _PROJECT_RELATIVE_DIRS:
            for cmd in _scan_directory(project_root / rel):
                by_name[cmd.name] = cmd

    if config.search_paths:
        for extra in config.search_paths:
            for cmd in _scan_directory(extra):
                by_name[cmd.name] = cmd

    if isinstance(config.enabled, list):
        allowed = set(config.enabled)
        by_name = {n: c for n, c in by_name.items() if n in allowed}

    return sorted(by_name.values(), key=lambda c: c.name)


def _walk_up_to_worktree(start: Path) -> list[Path]:
    """Yield ``start`` and every parent directory up to the worktree.

    A ``.git`` directory or file marks the worktree root. If none is
    found, walking stops at the filesystem root. Returned in
    bottom-up order (closest ancestor first) so callers can apply
    "more specific overrides more general" naturally — but we
    actually want the opposite for slash commands (closest cwd
    wins), so callers iterate this list and let later entries
    override earlier ones in the dict.
    """
    out: list[Path] = []
    seen: set[Path] = set()
    current = start.resolve()
    while True:
        if current in seen:
            break
        seen.add(current)
        out.append(current)
        if (current / ".git").exists():
            break
        if current.parent == current:
            break
        current = current.parent
    # Reverse so the outermost ancestor comes first; then iteration
    # order in discover() means cwd-local entries override ancestors.
    return list(reversed(out))


def _scan_directory(directory: Path) -> list[SlashCommand]:
    """Parse every ``*.md`` file in ``directory`` into a SlashCommand.

    Missing or unreadable directories return an empty list — discovery
    is best-effort, not strict. A malformed individual file is skipped
    (logged at debug level for visibility) so one broken command
    doesn't poison the whole list.
    """
    if not directory.is_dir():
        return []
    out: list[SlashCommand] = []
    for entry in sorted(directory.glob("*.md")):
        try:
            cmd = _parse_command_file(entry)
        except (OSError, UnicodeDecodeError):
            continue
        out.append(cmd)
    return out


_FRONTMATTER_DELIM = "---"
_FRONTMATTER_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.*)$")


def _parse_command_file(path: Path) -> SlashCommand:
    """Parse one ``.md`` file into a :class:`SlashCommand`.

    The frontmatter format is intentionally minimal — ``key: value``
    pairs between ``---`` markers, values kept as raw strings.
    Quoted values have their quotes stripped (``"foo"`` /
    ``'foo'`` → ``foo``). Lists and nested mappings aren't parsed;
    consumers that need full YAML can post-process the raw value
    strings themselves. Adding ``pyyaml`` as a hard dep just for
    this would be heavy.
    """
    text = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(text)

    name = frontmatter.get("name") or path.stem
    description = frontmatter.get("description")
    return SlashCommand(
        name=name,
        description=description,
        body=body,
        source_path=path,
        frontmatter=frontmatter,
    )


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split YAML-style frontmatter from the body.

    Recognises the canonical ``---\\nkey: value\\n---\\nbody`` shape.
    Files without a frontmatter block return an empty dict + the
    whole text as body. Unterminated frontmatter (one ``---`` and
    no closing) is treated the same way — defensive against
    malformed files.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FRONTMATTER_DELIM:
        return {}, text
    # Find closing ---
    closing_idx: int | None = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == _FRONTMATTER_DELIM:
            closing_idx = i
            break
    if closing_idx is None:
        return {}, text
    fm_lines = lines[1:closing_idx]
    body_lines = lines[closing_idx + 1 :]
    body = "\n".join(body_lines).lstrip("\n")
    fm = _parse_frontmatter_block(fm_lines)
    return fm, body


def _parse_frontmatter_block(lines: list[str]) -> dict[str, str]:
    """Minimal key/value parser for the frontmatter region."""
    out: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _FRONTMATTER_LINE_RE.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        if value.startswith(('"', "'")) and value.endswith(value[0]) and len(value) >= 2:
            value = value[1:-1]
        out[key] = value
    return out


__all__ = ["SlashCommand", "SlashCommandsConfig", "discover"]
