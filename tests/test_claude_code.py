"""Unit tests for :class:`ClaudeCodeRuntime`.

Mocks :mod:`claude_agent_sdk` at the boundary — no real subprocess, no
real Anthropic calls. Validates:

* Binding validation (Claude bindings pass; non-Claude bindings rejected).
* Structured-output happy path: ``ResultMessage.structured_output``
  lands on :attr:`RuntimeResult.structured`.
* Missing ``structured_output`` → :class:`RuntimeStructuredOutputError`.
* SDK auth failure → :class:`RuntimeAuthError`.
* SDK transient → :class:`RuntimeTransientError`.
* SDK CLI-not-found → :class:`RuntimeServerStartError`.
* Cost record populated from ``ResultMessage.usage`` + ``total_cost_usd``.
* ``ClaudeAgentOptions.output_format`` is set to the schema's JSON Schema.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from airframe.adapters.claude_code import ClaudeCodeRuntime
from airframe.errors import (
    AgentRuntimeError,
    RuntimeAuthError,
    RuntimeProtocolError,
    RuntimeServerStartError,
    RuntimeStructuredOutputError,
    RuntimeTransientError,
)
from airframe.protocol import ProviderModel, UnsupportedBindingError


class _Schema(BaseModel):
    summary: str
    count: int


class _FakeResultMessage:
    """Stand-in for ``claude_agent_sdk.ResultMessage``.

    A real class (not MagicMock) so the runtime's
    ``isinstance(msg, ResultMessage)`` check succeeds when the SDK
    symbol is monkeypatched to this type.
    """

    def __init__(
        self,
        *,
        is_error: bool = False,
        stop_reason: str | None = "end_turn",
        result: str | None = "",
        structured_output: Any = None,
        total_cost_usd: float | None = 0.05,
        usage: dict[str, Any] | None = None,
        subtype: str | None = None,
        errors: list[str] | None = None,
    ) -> None:
        self.is_error = is_error
        self.stop_reason = stop_reason
        self.result = result
        self.structured_output = structured_output
        self.total_cost_usd = total_cost_usd
        self.usage = usage or {
            "input_tokens": 100,
            "output_tokens": 200,
            "cache_read_input_tokens": 50,
            "cache_creation_input_tokens": 10,
        }
        self.subtype = subtype
        self.errors = errors


@pytest.fixture
def mock_sdk(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Mock the ``claude_agent_sdk`` symbols ``ClaudeCodeRuntime`` imports lazily."""
    import claude_agent_sdk as sdk

    mock_client = MagicMock()
    mock_client.connect = AsyncMock()
    mock_client.disconnect = AsyncMock()
    mock_client.query = AsyncMock()
    mock_client_factory = MagicMock(return_value=mock_client)

    monkeypatch.setattr(sdk, "ClaudeSDKClient", mock_client_factory)
    monkeypatch.setattr(sdk, "ClaudeAgentOptions", MagicMock())
    monkeypatch.setattr(sdk, "ResultMessage", _FakeResultMessage)

    return {
        "ClaudeSDKClient": mock_client_factory,
        "client": mock_client,
        "ResultMessage": _FakeResultMessage,
    }


def test_validate_binding_accepts_canonical_provider() -> None:
    """v0.2.0 canonical provider is `claude`; aliases dropped."""
    rt = ClaudeCodeRuntime()
    assert rt.validate_binding(ProviderModel("claude", "claude-haiku-4-5"))


def test_validate_binding_rejects_aliases_and_others() -> None:
    rt = ClaudeCodeRuntime()
    # `anthropic` reserved for a future direct-API AnthropicRuntime.
    assert not rt.validate_binding(ProviderModel("anthropic", "claude-haiku-4-5"))
    assert not rt.validate_binding(ProviderModel("claude-code", "claude-sonnet-4.6"))
    assert not rt.validate_binding(ProviderModel("github-copilot", "gpt-5.3-codex"))
    assert not rt.validate_binding(ProviderModel("openai", "gpt-5.5"))


@pytest.mark.asyncio
async def test_execute_rejects_unsupported_binding() -> None:
    rt = ClaudeCodeRuntime()
    with pytest.raises(UnsupportedBindingError):
        await rt.execute(
            "hello",
            schema=_Schema,
            model=ProviderModel("github-copilot", "gpt-5.3-codex"),
        )


@pytest.mark.asyncio
async def test_execute_plain_text_returns_text(mock_sdk: dict[str, MagicMock]) -> None:
    """``schema=None`` honours the protocol's plain-text contract.

    ``ResultMessage.result`` carries the concatenated final assistant
    text in plain-text mode; ``structured_output`` is absent. The
    adapter must return text on ``RuntimeResult.text`` and leave
    ``structured`` as ``None``.
    """
    rt = ClaudeCodeRuntime()
    final = _FakeResultMessage(
        result="The answer is 42.",
        structured_output=None,
        stop_reason="end_turn",
    )

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    result = await rt.execute(
        "ask something",
        model=ProviderModel("claude", "claude-haiku-4-5"),
    )

    assert result.text == "The answer is 42."
    assert result.structured is None
    assert result.finish == "end_turn"
    assert result.cost.input_tokens == 100
    assert result.cost.output_tokens == 200
    assert result.cost.provider_id == "claude"


@pytest.mark.asyncio
async def test_execute_plain_text_does_not_set_output_format(
    mock_sdk: dict[str, MagicMock],
) -> None:
    """No ``output_format`` on ClaudeAgentOptions when schema is None.

    The whole point of plain-text mode: the SDK isn't asked to
    enforce a JSON schema. The legacy structured-output forcing must
    not leak into the schema=None path.
    """
    import claude_agent_sdk as sdk

    captured_kwargs: dict[str, Any] = {}

    def capturing_options(**kwargs: Any) -> MagicMock:
        captured_kwargs.update(kwargs)
        return MagicMock()

    final = _FakeResultMessage(result="hi", structured_output=None)

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    with patch.object(sdk, "ClaudeAgentOptions", side_effect=capturing_options):
        rt = ClaudeCodeRuntime()
        await rt.execute("hi")

    assert "output_format" not in captured_kwargs


@pytest.mark.asyncio
async def test_execute_plain_text_forwards_system_prompt(
    mock_sdk: dict[str, MagicMock],
) -> None:
    """``system=`` lands in ClaudeAgentOptions.system_prompt verbatim.

    The load-bearing test: every downstream persona depends on the
    caller-supplied system prompt actually reaching the model in
    text mode.
    """
    import claude_agent_sdk as sdk

    captured_kwargs: dict[str, Any] = {}

    def capturing_options(**kwargs: Any) -> MagicMock:
        captured_kwargs.update(kwargs)
        return MagicMock()

    final = _FakeResultMessage(result="ok", structured_output=None)

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    with patch.object(sdk, "ClaudeAgentOptions", side_effect=capturing_options):
        rt = ClaudeCodeRuntime()
        await rt.execute("hi", system="You are the navigator.")

    assert captured_kwargs.get("system_prompt") == "You are the navigator."


@pytest.mark.asyncio
async def test_execute_plain_text_ignores_persona(mock_sdk: dict[str, MagicMock]) -> None:
    """``persona=`` is accepted but unused by ClaudeCodeRuntime.

    Per the protocol docstring: "Some adapters honour it; others
    ignore it." Claude ignores it. The contract is just that
    passing a value doesn't crash.
    """
    rt = ClaudeCodeRuntime()
    final = _FakeResultMessage(result="ok", structured_output=None)

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    result = await rt.execute("hi", persona="navigator")
    assert result.text == "ok"


@pytest.mark.asyncio
async def test_execute_plain_text_classifies_auth_error(
    mock_sdk: dict[str, MagicMock],
) -> None:
    """Auth errors classify the same way in text mode as in schema mode.

    Same code path through ``_classify_exception``; this test pins
    the contract that the classification doesn't depend on the
    presence of a schema.
    """
    from claude_agent_sdk import ClaudeSDKError

    rt = ClaudeCodeRuntime()
    mock_sdk["client"].connect.side_effect = ClaudeSDKError("401 unauthorized")

    with pytest.raises(RuntimeAuthError):
        await rt.execute("hi")


@pytest.mark.asyncio
async def test_execute_returns_structured_output(mock_sdk: dict[str, MagicMock]) -> None:
    """``ResultMessage.structured_output`` lands on ``RuntimeResult.structured``."""
    rt = ClaudeCodeRuntime()

    expected = {"summary": "ok", "count": 42}
    final = _FakeResultMessage(structured_output=expected)

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    result = await rt.execute(
        "say hi",
        schema=_Schema,
        model=ProviderModel("claude", "claude-haiku-4-5"),
    )

    assert result.structured == expected
    assert result.finish == "end_turn"
    assert result.cost.cost_usd == 0.05
    assert result.cost.input_tokens == 100
    assert result.cost.output_tokens == 200
    assert result.cost.cache_read_tokens == 50
    assert result.cost.cache_write_tokens == 10
    assert result.cost.provider_id == "claude"
    assert result.cost.model_id == "claude-haiku-4-5"


@pytest.mark.asyncio
async def test_execute_missing_structured_output_raises(
    mock_sdk: dict[str, MagicMock],
) -> None:
    """If the SDK returns without a structured_output payload, we raise."""
    rt = ClaudeCodeRuntime()

    final = _FakeResultMessage(
        stop_reason="end_turn",
        result="I refused to produce structured output",
        structured_output=None,
    )

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    with pytest.raises(RuntimeStructuredOutputError):
        await rt.execute("hi", schema=_Schema)


@pytest.mark.asyncio
async def test_execute_is_error_result_raises_runtime_error(
    mock_sdk: dict[str, MagicMock],
) -> None:
    rt = ClaudeCodeRuntime()

    final = _FakeResultMessage(
        is_error=True,
        subtype="error_max_turns",
        errors=["Reached max turns"],
    )

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive

    with pytest.raises(AgentRuntimeError):
        await rt.execute("hi", schema=_Schema)


@pytest.mark.asyncio
async def test_execute_empty_stream_raises_protocol_error(
    mock_sdk: dict[str, MagicMock],
) -> None:
    rt = ClaudeCodeRuntime()

    async def fake_receive() -> Any:
        # No ResultMessage in the stream at all.
        if False:
            yield  # type: ignore[unreachable] — make this an async generator

    mock_sdk["client"].receive_response = fake_receive

    with pytest.raises(RuntimeProtocolError):
        await rt.execute("hi", schema=_Schema)


@pytest.mark.asyncio
async def test_execute_cli_not_found_raises_server_start_error(
    mock_sdk: dict[str, MagicMock],
) -> None:
    from claude_agent_sdk import CLINotFoundError

    rt = ClaudeCodeRuntime()
    mock_sdk["client"].connect.side_effect = CLINotFoundError("no claude CLI")

    with pytest.raises(RuntimeServerStartError):
        await rt.execute("hi", schema=_Schema)


@pytest.mark.asyncio
async def test_execute_auth_failure_raises_runtime_auth_error(
    mock_sdk: dict[str, MagicMock],
) -> None:
    from claude_agent_sdk import ClaudeSDKError

    rt = ClaudeCodeRuntime()
    mock_sdk["client"].connect.side_effect = ClaudeSDKError("401 unauthorized")

    with pytest.raises(RuntimeAuthError):
        await rt.execute("hi", schema=_Schema)


@pytest.mark.asyncio
async def test_execute_timeout_raises_transient_error(
    mock_sdk: dict[str, MagicMock],
) -> None:
    rt = ClaudeCodeRuntime()

    async def slow_receive() -> Any:
        import asyncio

        await asyncio.sleep(10)
        if False:
            yield  # type: ignore[unreachable]

    mock_sdk["client"].receive_response = slow_receive

    with pytest.raises(RuntimeTransientError):
        await rt.execute("hi", schema=_Schema, timeout=0.05)


@pytest.mark.asyncio
async def test_reset_disconnects_client(mock_sdk: dict[str, MagicMock]) -> None:
    rt = ClaudeCodeRuntime()
    final = _FakeResultMessage(structured_output={"summary": "x", "count": 1})

    async def fake_receive() -> Any:
        yield final

    mock_sdk["client"].receive_response = fake_receive
    await rt.execute("hi", schema=_Schema)
    assert rt._client is not None  # noqa: SLF001

    await rt.reset()
    assert rt._client is None  # noqa: SLF001
    assert mock_sdk["client"].disconnect.await_count >= 1


@pytest.mark.asyncio
async def test_reset_with_no_client_is_noop() -> None:
    rt = ClaudeCodeRuntime()
    # Should not raise even with no client constructed yet.
    await rt.reset()
    await rt.close()


def test_api_key_override_passes_through_env(mock_sdk: dict[str, MagicMock]) -> None:
    """When api_key= is passed, it lands in ClaudeAgentOptions.env."""
    import claude_agent_sdk as sdk

    captured_kwargs: dict[str, Any] = {}

    def capturing_options(**kwargs: Any) -> MagicMock:
        captured_kwargs.update(kwargs)
        return MagicMock()

    with patch.object(sdk, "ClaudeAgentOptions", side_effect=capturing_options):
        rt = ClaudeCodeRuntime(api_key="sk-ant-test-key")
        import asyncio

        async def go() -> None:
            await rt._ensure_client(  # noqa: SLF001
                schema=_Schema, system=None, model="claude-haiku-4-5"
            )

        asyncio.get_event_loop().run_until_complete(go())

    env = captured_kwargs.get("env", {})
    assert env.get("ANTHROPIC_API_KEY") == "sk-ant-test-key"


def test_output_format_uses_schema_json_schema(mock_sdk: dict[str, MagicMock]) -> None:
    """The runtime passes a native json_schema output_format to the SDK."""
    import claude_agent_sdk as sdk

    captured_kwargs: dict[str, Any] = {}

    def capturing_options(**kwargs: Any) -> MagicMock:
        captured_kwargs.update(kwargs)
        return MagicMock()

    with patch.object(sdk, "ClaudeAgentOptions", side_effect=capturing_options):
        rt = ClaudeCodeRuntime()
        import asyncio

        async def go() -> None:
            await rt._ensure_client(  # noqa: SLF001
                schema=_Schema, system=None, model="claude-haiku-4-5"
            )

        asyncio.get_event_loop().run_until_complete(go())

    output_format = captured_kwargs.get("output_format")
    assert output_format == {
        "type": "json_schema",
        "schema": _Schema.model_json_schema(),
    }
    # No MCP servers / forced tool wiring.
    assert "mcp_servers" not in captured_kwargs
    assert "allowed_tools" not in captured_kwargs


def test_no_system_prompt_prefix_when_system_is_none(
    mock_sdk: dict[str, MagicMock],
) -> None:
    """Without a caller-supplied system prompt, we don't synthesise one.

    The SDK applies its own default; we no longer prepend a tool-forcing
    prefix.
    """
    import claude_agent_sdk as sdk

    captured_kwargs: dict[str, Any] = {}

    def capturing_options(**kwargs: Any) -> MagicMock:
        captured_kwargs.update(kwargs)
        return MagicMock()

    with patch.object(sdk, "ClaudeAgentOptions", side_effect=capturing_options):
        rt = ClaudeCodeRuntime()
        import asyncio

        async def go() -> None:
            await rt._ensure_client(  # noqa: SLF001
                schema=_Schema, system=None, model="claude-haiku-4-5"
            )

        asyncio.get_event_loop().run_until_complete(go())

    assert "system_prompt" not in captured_kwargs


def test_system_prompt_passes_through_unmodified(mock_sdk: dict[str, MagicMock]) -> None:
    """Caller-supplied system prompts land in ClaudeAgentOptions verbatim."""
    import claude_agent_sdk as sdk

    captured_kwargs: dict[str, Any] = {}

    def capturing_options(**kwargs: Any) -> MagicMock:
        captured_kwargs.update(kwargs)
        return MagicMock()

    with patch.object(sdk, "ClaudeAgentOptions", side_effect=capturing_options):
        rt = ClaudeCodeRuntime()
        import asyncio

        async def go() -> None:
            await rt._ensure_client(  # noqa: SLF001
                schema=_Schema,
                system="You are the navigator.",
                model="claude-haiku-4-5",
            )

        asyncio.get_event_loop().run_until_complete(go())

    assert captured_kwargs.get("system_prompt") == "You are the navigator."
