"""Built-in :class:`AgentRuntime` adapters.

Each adapter wraps a single vendor SDK. Adapters are organised by
vendor module so consumers can ``from airframe.adapters.claude_code
import ClaudeCodeRuntime`` if they prefer narrower imports; the
top-level :mod:`airframe` re-exports each adapter for convenience.
"""

from __future__ import annotations

from airframe.adapters.claude_code import ClaudeCodeRuntime
from airframe.adapters.copilot import CopilotRuntime
from airframe.adapters.opencode_zen import OpenCodeZenRuntime

__all__ = [
    "ClaudeCodeRuntime",
    "CopilotRuntime",
    "OpenCodeZenRuntime",
]
