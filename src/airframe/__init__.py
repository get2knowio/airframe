"""Airframe — vendor-neutral agent runtime.

One protocol, pluggable adapters for Claude Code, GitHub Copilot,
Moonshot Kimi, AWS Bedrock, the OpenCode HTTP agent server, and the
OpenAI-compatible family (OpenCode Zen / Go, OpenRouter).

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
        model=ProviderModel("claude", "claude-haiku-4-5"),
    )
    print(result.structured)         # {"summary": "...", "risks": [...]}
    print(result.cost.cost_usd)      # 0.0042
    await runtime.close()

Discovery for UI menus::

    from airframe import list_providers, runtime_for

    for provider in list_providers():
        rt_cls = runtime_for(provider)
        runtime = rt_cls()
        models = await runtime.list_models()
        for m in models:
            print(provider, m.id, m.display_name, m.context_window)

Each adapter's SDK is an optional install (``pip install
airframe-agents[claude]``); see the README for the full extras matrix.
"""

from __future__ import annotations

from airframe.cache import CacheConfig
from airframe.cost import CostRecord
from airframe.discovery import list_providers, runtime_for
from airframe.errors import (
    AgentRuntimeError,
    RuntimeAuthError,
    RuntimeBudgetExceededError,
    RuntimeCancelledError,
    RuntimeContextOverflowError,
    RuntimeModelNotFoundError,
    RuntimeProtocolError,
    RuntimeServerStartError,
    RuntimeStructuredOutputError,
    RuntimeTransientError,
    UnsupportedFeatureError,
)
from airframe.events import (
    ReasoningDelta,
    RuntimeEvent,
    TextDelta,
    ToolCallResult,
    ToolCallStart,
    TurnComplete,
)
from airframe.features import Feature
from airframe.hooks import HookEvent, HookEventKind
from airframe.inputs import FileInput, ImageInput, Prompt, PromptPart
from airframe.metadata import RequestMetadata
from airframe.models import (
    CAPABILITY_REASONING_EFFORT,
    CAPABILITY_STREAMING,
    CAPABILITY_STRUCTURED_OUTPUT,
    CAPABILITY_TOOLS,
    CAPABILITY_VISION,
    ModelInfo,
)
from airframe.options import (
    BedrockOptions,
    ClaudeOptions,
    CopilotOptions,
    KimiOptions,
    OpenAICompatOptions,
    OpenCodeServerOptions,
    ProviderOptions,
)
from airframe.permission import PermissionCallback, PermissionDecision, PermissionRequest
from airframe.protocol import (
    AgentRuntime,
    AgentSession,
    ProviderModel,
    RuntimeResult,
    UnsupportedBindingError,
)
from airframe.rate_limit import RateLimitInfo, RateLimitWindow
from airframe.slash_commands import SlashCommand, SlashCommandsConfig
from airframe.thinking import ReasoningEffort, ThinkingMode
from airframe.tools import FunctionTool, McpServerRef

__version__ = "0.8.0"

# Adapter imports live at the top level for ergonomic use, but the
# adapter modules themselves lazy-import their underlying SDK so
# `import airframe` doesn't pull in claude-agent-sdk / copilot /
# openai unless the consumer actually instantiates the adapter.
from airframe.adapters.bedrock import BedrockRuntime
from airframe.adapters.claude_code import ClaudeCodeRuntime
from airframe.adapters.copilot import CopilotRuntime
from airframe.adapters.kimi import KimiRuntime
from airframe.adapters.opencode_go import OpenCodeGoRuntime
from airframe.adapters.opencode_server import OpenCodeServerRuntime
from airframe.adapters.opencode_zen import OpenCodeZenRuntime
from airframe.adapters.openrouter import OpenRouterRuntime

__all__ = [
    "AgentRuntime",
    "AgentRuntimeError",
    "AgentSession",
    "BedrockOptions",
    "BedrockRuntime",
    "CAPABILITY_REASONING_EFFORT",
    "CAPABILITY_STREAMING",
    "CAPABILITY_STRUCTURED_OUTPUT",
    "CAPABILITY_TOOLS",
    "CAPABILITY_VISION",
    "CacheConfig",
    "ClaudeCodeRuntime",
    "ClaudeOptions",
    "CopilotOptions",
    "CopilotRuntime",
    "CostRecord",
    "Feature",
    "FileInput",
    "FunctionTool",
    "HookEvent",
    "HookEventKind",
    "ImageInput",
    "KimiOptions",
    "KimiRuntime",
    "McpServerRef",
    "ModelInfo",
    "OpenAICompatOptions",
    "OpenCodeGoRuntime",
    "OpenCodeServerOptions",
    "OpenCodeServerRuntime",
    "OpenCodeZenRuntime",
    "OpenRouterRuntime",
    "PermissionCallback",
    "PermissionDecision",
    "PermissionRequest",
    "Prompt",
    "PromptPart",
    "ProviderModel",
    "ProviderOptions",
    "RateLimitInfo",
    "RateLimitWindow",
    "ReasoningDelta",
    "ReasoningEffort",
    "RequestMetadata",
    "RuntimeAuthError",
    "RuntimeBudgetExceededError",
    "RuntimeCancelledError",
    "RuntimeContextOverflowError",
    "RuntimeEvent",
    "RuntimeModelNotFoundError",
    "RuntimeProtocolError",
    "RuntimeResult",
    "RuntimeServerStartError",
    "RuntimeStructuredOutputError",
    "RuntimeTransientError",
    "SlashCommand",
    "SlashCommandsConfig",
    "TextDelta",
    "ThinkingMode",
    "ToolCallResult",
    "ToolCallStart",
    "TurnComplete",
    "UnsupportedBindingError",
    "UnsupportedFeatureError",
    "__version__",
    "list_providers",
    "runtime_for",
]
