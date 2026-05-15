"""Airframe — vendor-neutral agent runtime.

One protocol, pluggable adapters for Claude Code, GitHub Copilot,
OpenAI Codex, and OpenCode Zen.

Quick start::

    from airframe import ClaudeCodeRuntime, ProviderModel
    from pydantic import BaseModel

    class Brief(BaseModel):
        summary: str
        risks: list[str]

    runtime = ClaudeCodeRuntime()
    result = await runtime.execute(
        "Brief me on the project structure.",
        schema=Brief,
        model=ProviderModel("anthropic", "claude-haiku-4-5"),
    )
    print(result.structured)         # {"summary": "...", "risks": [...]}
    print(result.cost.cost_usd)      # 0.0042
    await runtime.aclose()

Each adapter's SDK is an optional install (``pip install
airframe-agents[claude]``); see the README for the full extras matrix.
"""

from __future__ import annotations

from airframe.cost import CostRecord
from airframe.errors import (
    AgentRuntimeError,
    RuntimeAuthError,
    RuntimeCancelledError,
    RuntimeContextOverflowError,
    RuntimeModelNotFoundError,
    RuntimeProtocolError,
    RuntimeServerStartError,
    RuntimeStructuredOutputError,
    RuntimeTransientError,
)
from airframe.protocol import (
    AgentRuntime,
    ProviderModel,
    RuntimeResult,
    UnsupportedBindingError,
)

__version__ = "0.1.0"

# Adapter imports live at the top level for ergonomic use, but the
# adapter modules themselves lazy-import their underlying SDK so
# `import airframe` doesn't pull in claude-agent-sdk / codex / copilot
# / openai unless the consumer actually instantiates the adapter.
from airframe.adapters.claude_code import ClaudeCodeRuntime
from airframe.adapters.codex import CodexRuntime
from airframe.adapters.copilot import CopilotRuntime
from airframe.adapters.opencode_zen import OpenCodeZenRuntime

__all__ = [
    "AgentRuntime",
    "AgentRuntimeError",
    "ClaudeCodeRuntime",
    "CodexRuntime",
    "CopilotRuntime",
    "CostRecord",
    "OpenCodeZenRuntime",
    "ProviderModel",
    "RuntimeAuthError",
    "RuntimeCancelledError",
    "RuntimeContextOverflowError",
    "RuntimeModelNotFoundError",
    "RuntimeProtocolError",
    "RuntimeResult",
    "RuntimeServerStartError",
    "RuntimeStructuredOutputError",
    "RuntimeTransientError",
    "UnsupportedBindingError",
    "__version__",
]
