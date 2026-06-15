"""Unit tests for :mod:`airframe.slash_commands` discovery + parsing.

The module is filesystem-only (no vendor SDK calls) so all tests
run against tmp_path-rooted command files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from airframe.slash_commands import (
    SlashCommand,
    SlashCommandsConfig,
    discover,
)


def _write_cmd(directory: Path, name: str, content: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.md"
    path.write_text(content, encoding="utf-8")
    return path


def test_slash_commands_config_defaults() -> None:
    c = SlashCommandsConfig()
    assert c.enabled == "all"
    assert c.search_paths is None
    assert c.include_user_global is True


def test_discover_with_no_commands_returns_empty(tmp_path: Path) -> None:
    config = SlashCommandsConfig(include_user_global=False)
    assert discover(config, cwd=tmp_path) == []


def test_discover_parses_frontmatter_and_body(tmp_path: Path) -> None:
    _write_cmd(
        tmp_path / ".claude/commands",
        "refactor",
        "---\nname: refactor\ndescription: Refactor a file\n---\nRefactor {file}.\n",
    )
    config = SlashCommandsConfig(include_user_global=False)
    cmds = discover(config, cwd=tmp_path)
    assert len(cmds) == 1
    cmd = cmds[0]
    assert isinstance(cmd, SlashCommand)
    assert cmd.name == "refactor"
    assert cmd.description == "Refactor a file"
    assert "Refactor {file}." in cmd.body
    assert cmd.frontmatter == {"name": "refactor", "description": "Refactor a file"}


def test_discover_falls_back_to_filename_when_name_field_missing(tmp_path: Path) -> None:
    _write_cmd(
        tmp_path / ".claude/commands",
        "explain",
        "---\ndescription: Explain code\n---\nExplain {code}.\n",
    )
    cmds = discover(SlashCommandsConfig(include_user_global=False), cwd=tmp_path)
    assert cmds[0].name == "explain"


def test_discover_handles_files_with_no_frontmatter(tmp_path: Path) -> None:
    _write_cmd(tmp_path / ".claude/commands", "bare", "Just a body — no frontmatter.\n")
    cmds = discover(SlashCommandsConfig(include_user_global=False), cwd=tmp_path)
    assert len(cmds) == 1
    assert cmds[0].name == "bare"
    assert cmds[0].description is None
    assert cmds[0].frontmatter == {}
    assert cmds[0].body == "Just a body — no frontmatter.\n"


def test_discover_strips_quoted_frontmatter_values(tmp_path: Path) -> None:
    _write_cmd(
        tmp_path / ".claude/commands",
        "quoted",
        "---\nname: \"quoted-name\"\ndescription: 'Has quotes'\n---\nbody\n",
    )
    cmds = discover(SlashCommandsConfig(include_user_global=False), cwd=tmp_path)
    assert cmds[0].name == "quoted-name"
    assert cmds[0].description == "Has quotes"


def test_discover_finds_opencode_path_too(tmp_path: Path) -> None:
    _write_cmd(
        tmp_path / ".opencode/command",
        "review",
        "---\ndescription: Review\n---\nReview the diff.\n",
    )
    cmds = discover(SlashCommandsConfig(include_user_global=False), cwd=tmp_path)
    assert {c.name for c in cmds} == {"review"}


def test_discover_unions_commands_across_locations(tmp_path: Path) -> None:
    _write_cmd(tmp_path / ".claude/commands", "claude-only", "Claude body")
    _write_cmd(tmp_path / ".opencode/command", "opencode-only", "OpenCode body")
    cmds = discover(SlashCommandsConfig(include_user_global=False), cwd=tmp_path)
    assert {c.name for c in cmds} == {"claude-only", "opencode-only"}


def test_discover_later_path_overrides_earlier_on_name_collision(tmp_path: Path) -> None:
    """Project-local .claude/commands should win over .opencode/command."""
    _write_cmd(tmp_path / ".opencode/command", "shared", "OpenCode version")
    _write_cmd(tmp_path / ".claude/commands", "shared", "Claude version")
    cmds = discover(SlashCommandsConfig(include_user_global=False), cwd=tmp_path)
    assert len(cmds) == 1
    assert "Claude version" in cmds[0].body


def test_discover_filters_to_enabled_list(tmp_path: Path) -> None:
    _write_cmd(tmp_path / ".claude/commands", "keep", "body")
    _write_cmd(tmp_path / ".claude/commands", "drop", "body")
    config = SlashCommandsConfig(enabled=["keep"], include_user_global=False)
    cmds = discover(config, cwd=tmp_path)
    assert [c.name for c in cmds] == ["keep"]


def test_discover_with_enabled_none_returns_empty(tmp_path: Path) -> None:
    _write_cmd(tmp_path / ".claude/commands", "anything", "body")
    config = SlashCommandsConfig(enabled=None, include_user_global=False)
    assert discover(config, cwd=tmp_path) == []


def test_discover_honours_extra_search_paths(tmp_path: Path) -> None:
    extra = tmp_path / "custom-commands"
    _write_cmd(extra, "custom", "From custom path")
    config = SlashCommandsConfig(
        search_paths=[extra],
        include_user_global=False,
    )
    cmds = discover(config, cwd=tmp_path)
    assert any(c.name == "custom" for c in cmds)


def test_discover_skips_malformed_files(tmp_path: Path) -> None:
    """A file with unterminated frontmatter still parses cleanly (as body-only)."""
    _write_cmd(tmp_path / ".claude/commands", "broken", "---\nthis-frontmatter-never-closes")
    cmds = discover(SlashCommandsConfig(include_user_global=False), cwd=tmp_path)
    # Doesn't raise — body is the whole thing including the unclosed ---
    assert len(cmds) == 1
    assert cmds[0].name == "broken"


def test_discover_walks_up_to_worktree(tmp_path: Path) -> None:
    """Commands in an ancestor directory are discovered."""
    nested = tmp_path / "a/b/c"
    nested.mkdir(parents=True)
    _write_cmd(tmp_path / ".claude/commands", "ancestor", "From ancestor")
    cmds = discover(SlashCommandsConfig(include_user_global=False), cwd=nested)
    assert any(c.name == "ancestor" for c in cmds)


async def test_session_list_slash_commands(tmp_path: Path) -> None:
    """OpenCodeZenRuntime (representative OAI-compat adapter) wires the method."""
    from airframe.adapters.opencode_zen import OpenCodeZenRuntime

    _write_cmd(tmp_path / ".claude/commands", "via-session", "Body from session test")
    runtime = OpenCodeZenRuntime(api_key="dummy")
    # search_paths makes the test deterministic — no dependence on cwd.
    config = SlashCommandsConfig(
        search_paths=[tmp_path / ".claude/commands"],
        include_user_global=False,
        enabled="all",
    )
    sess = runtime.session(slash_commands=config)
    try:
        cmds = await sess.list_slash_commands()
        assert any(c.name == "via-session" for c in cmds)
    finally:
        await sess.close()


def _make_oai_compat() -> object:
    import airframe

    return airframe.OpenCodeZenRuntime(api_key="dummy")


def _make_claude() -> object:
    import airframe

    return airframe.ClaudeCodeRuntime()


def _make_bedrock() -> object:
    import airframe

    return airframe.BedrockRuntime()


def _make_opencode_server() -> object:
    import airframe

    return airframe.OpenCodeServerRuntime()


def _make_copilot() -> object:
    import airframe

    return airframe.CopilotRuntime()


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(_make_oai_compat, id="oai-compat"),
        pytest.param(_make_claude, id="claude"),
        pytest.param(_make_bedrock, id="bedrock"),
        pytest.param(_make_opencode_server, id="opencode-server"),
        pytest.param(_make_copilot, id="copilot"),
    ],
)
async def test_every_adapter_declares_slash_commands_feature(factory: object) -> None:
    """Every adapter declares Feature.SLASH_COMMANDS=True — discovery
    is filesystem-only and adapter-agnostic."""
    from airframe.features import Feature

    runtime = factory()  # type: ignore[operator]
    try:
        assert runtime.supports(Feature.SLASH_COMMANDS) is True
    finally:
        await runtime.close()
