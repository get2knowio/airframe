"""Unit tests for :meth:`AgentRuntime.unwrap`.

Modelled on JDBC's :class:`java.sql.Wrapper`. The contract is:

* Calling ``unwrap(type(self))`` returns ``self`` for every adapter
  — the trivial case keeps the contract consistent.
* Calling ``unwrap(<native type>)`` returns the underlying vendor
  object when one has been built, raises :class:`TypeError`
  otherwise.
* Calling ``unwrap(<unrelated type>)`` raises :class:`TypeError`.

The native-client construction paths are mocked at the boundary —
the integration tests that actually exercise the constructed objects
live alongside the existing per-adapter test suites.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from airframe.adapters.claude_code import ClaudeCodeRuntime
from airframe.adapters.codex import CodexRuntime
from airframe.adapters.copilot import CopilotRuntime
from airframe.adapters.opencode_zen import OpenCodeZenRuntime

# --- Trivial case: unwrap(type(self)) returns self ----------------------------


class _UnrelatedType:
    """Stand-in for "a type this runtime doesn't unwrap to."""


def test_claude_code_unwraps_self() -> None:
    runtime = ClaudeCodeRuntime()
    assert runtime.unwrap(ClaudeCodeRuntime) is runtime


def test_copilot_unwraps_self() -> None:
    runtime = CopilotRuntime()
    assert runtime.unwrap(CopilotRuntime) is runtime


def test_codex_unwraps_self() -> None:
    runtime = CodexRuntime()
    assert runtime.unwrap(CodexRuntime) is runtime


def test_opencode_zen_unwraps_self() -> None:
    runtime = OpenCodeZenRuntime(api_key="dummy")
    assert runtime.unwrap(OpenCodeZenRuntime) is runtime


# --- Unsupported types raise TypeError ---------------------------------------


@pytest.mark.parametrize(
    "runtime_factory",
    [
        ClaudeCodeRuntime,
        CopilotRuntime,
        CodexRuntime,
        lambda: OpenCodeZenRuntime(api_key="dummy"),
    ],
    ids=["claude_code", "copilot", "codex", "opencode_zen"],
)
def test_unwrap_unrelated_type_raises_typeerror(runtime_factory: Any) -> None:
    """Every adapter rejects types it doesn't unwrap to with a clear error.

    Maps to JDBC ``Wrapper.unwrap`` raising ``SQLException`` for
    unsupported casts. The message should name the runtime and the
    requested type so callers know which adapter refused.
    """
    runtime = runtime_factory()
    with pytest.raises(TypeError) as exc_info:
        runtime.unwrap(_UnrelatedType)
    assert type(runtime).__name__ in str(exc_info.value)


# --- Native unwrap before client construction raises -------------------------


def test_claude_code_unwrap_native_redirects_to_session() -> None:
    """Phase 1 Iteration G: ClaudeSDKClient moved off the runtime onto session.

    ``runtime.unwrap(ClaudeSDKClient)`` raises with a clear message
    pointing the caller to ``session.unwrap(ClaudeSDKClient)``.
    """
    from claude_agent_sdk import ClaudeSDKClient

    runtime = ClaudeCodeRuntime()
    with pytest.raises(TypeError, match="sessions do"):
        runtime.unwrap(ClaudeSDKClient)


def test_copilot_unwrap_session_redirects_to_session() -> None:
    """Phase 1 Iteration G: CopilotSession moved onto AgentSession.

    ``CopilotClient`` is runtime-owned and still unwraps normally;
    ``CopilotSession`` raises with a redirect.
    """
    from copilot import CopilotClient
    from copilot.session import CopilotSession

    runtime = CopilotRuntime()
    # CopilotClient still raises pre-construction (runtime-owned but lazy).
    with pytest.raises(TypeError, match="no client exists yet"):
        runtime.unwrap(CopilotClient)
    # CopilotSession now points users to the session.
    with pytest.raises(TypeError, match="sessions do"):
        runtime.unwrap(CopilotSession)


def test_codex_unwrap_thread_redirects_to_session() -> None:
    """Phase 1 Iteration G: Thread moved onto AgentSession.

    ``Codex`` is runtime-owned and still unwraps normally;
    ``Thread`` raises with a redirect.
    """
    from openai_codex_sdk import Codex, Thread

    runtime = CodexRuntime()
    with pytest.raises(TypeError, match="no client exists yet"):
        runtime.unwrap(Codex)
    with pytest.raises(TypeError, match="sessions do"):
        runtime.unwrap(Thread)


# --- Native unwrap after client construction returns the live object ----------


def test_copilot_unwrap_returns_live_client() -> None:
    """``CopilotClient`` is still runtime-owned — unwrap returns it after construction."""
    from copilot import CopilotClient

    runtime = CopilotRuntime()
    mock_client = MagicMock(spec=CopilotClient)
    runtime._client = mock_client  # noqa: SLF001 — test scaffolding
    assert runtime.unwrap(CopilotClient) is mock_client


def test_codex_unwrap_returns_live_client() -> None:
    """``Codex`` is still runtime-owned — unwrap returns it after construction."""
    from openai_codex_sdk import Codex

    runtime = CodexRuntime()
    mock_client = MagicMock(spec=Codex)
    runtime._client = mock_client  # noqa: SLF001 — test scaffolding
    assert runtime.unwrap(Codex) is mock_client


# --- Session-level unwrap (Phase 1 Iteration G — new home for native types) ----


def test_claude_code_session_unwrap_native_before_execute_raises() -> None:
    """``session.unwrap(ClaudeSDKClient)`` before the first turn raises."""
    from claude_agent_sdk import ClaudeSDKClient

    runtime = ClaudeCodeRuntime()
    sess = runtime.session()
    with pytest.raises(TypeError, match="no client exists yet"):
        sess.unwrap(ClaudeSDKClient)


def test_copilot_session_unwrap_native_before_execute_raises() -> None:
    """``session.unwrap(CopilotSession)`` before the first turn raises."""
    from copilot.session import CopilotSession

    runtime = CopilotRuntime()
    sess = runtime.session()
    with pytest.raises(TypeError, match="no session exists yet"):
        sess.unwrap(CopilotSession)


def test_codex_session_unwrap_native_before_execute_raises() -> None:
    """``session.unwrap(Thread)`` before the first turn raises."""
    from openai_codex_sdk import Thread

    runtime = CodexRuntime()
    sess = runtime.session()
    with pytest.raises(TypeError, match="no thread exists yet"):
        sess.unwrap(Thread)


def test_opencode_zen_unwrap_builds_client_lazily() -> None:
    """OpenAI-compat unwraps to AsyncOpenAI — built on demand.

    Different from the subprocess adapters: there's no
    "client doesn't exist yet" failure mode because the HTTP client
    is cheap to construct and ``_ensure_client()`` is happy to do it
    on first call. So ``unwrap(AsyncOpenAI)`` always succeeds after
    auth resolves.
    """
    from openai import AsyncOpenAI

    runtime = OpenCodeZenRuntime(api_key="dummy")
    client = runtime.unwrap(AsyncOpenAI)
    assert isinstance(client, AsyncOpenAI)
    # And it's the same instance on subsequent calls.
    assert runtime.unwrap(AsyncOpenAI) is client
