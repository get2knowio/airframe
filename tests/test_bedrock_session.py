"""Unit tests for :class:`BedrockSession` — Iteration B behaviour.

Mocks the aioboto3 bedrock-runtime client at the boundary so no
real Converse calls fire. Covers:

* ``execute()`` single-turn with and without ``schema=``.
* Multi-turn message accumulation across calls.
* User message rollback on failure.
* ``stream()`` chunk translation (text + reasoning deltas + final
  ``TurnComplete`` carrying schema-shaped payload).
* ``cancel()`` during in-flight execute.
* ``close()`` clears the messages buffer; subsequent execute raises.
* ``runtime.execute()`` sugar opens + closes a per-call session.
* Error classification at the converse boundary
  (ValidationException / Throttling / context overflow).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from airframe.adapters.bedrock import (
    SUBMIT_RESULT_TOOL,
    BedrockRuntime,
    BedrockSession,
    _build_submit_result_tool_config,
)
from airframe.errors import (
    RuntimeCancelledError,
    RuntimeContextOverflowError,
    RuntimeModelNotFoundError,
    RuntimeStructuredOutputError,
    RuntimeTransientError,
    UnsupportedFeatureError,
)
from airframe.events import ReasoningDelta, TextDelta, TurnComplete
from airframe.features import Feature


class _Schema(BaseModel):
    summary: str
    count: int


# ---------------------------------------------------------------------------
# aioboto3 bedrock-runtime client mock
# ---------------------------------------------------------------------------


def _converse_response(
    *,
    text: str = "ok",
    tool_input: dict[str, Any] | None = None,
    stop_reason: str = "end_turn",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> dict[str, Any]:
    """Build a stand-in for ``client.converse()`` output."""
    content: list[dict[str, Any]] = []
    if text:
        content.append({"text": text})
    if tool_input is not None:
        content.append(
            {
                "toolUse": {
                    "toolUseId": "tool_call_abc",
                    "name": SUBMIT_RESULT_TOOL,
                    "input": tool_input,
                }
            }
        )
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "usage": {"inputTokens": input_tokens, "outputTokens": output_tokens},
        "stopReason": stop_reason,
    }


async def _async_iter_from(chunks: list[dict[str, Any]]):
    """Helper: yield each chunk from a plain list as an async iterator."""
    for c in chunks:
        yield c


def _converse_stream_response(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a stand-in for ``client.converse_stream()`` output.

    Bedrock returns ``{"stream": <async iter>, "ResponseMetadata": ...}``.
    """
    return {"stream": _async_iter_from(chunks)}


@pytest.fixture
def runtime_with_mock_client(monkeypatch: pytest.MonkeyPatch) -> tuple[BedrockRuntime, MagicMock]:
    """Construct a runtime whose ``_get_runtime_client`` returns a mock."""
    rt = BedrockRuntime(region_name="us-east-1")
    client = MagicMock()
    client.converse = AsyncMock(return_value=_converse_response(text="hello"))
    client.converse_stream = AsyncMock()

    async def _stub_get_client() -> Any:
        return client

    monkeypatch.setattr(rt, "_get_runtime_client", _stub_get_client)
    return rt, client


# ---------------------------------------------------------------------------
# execute() — single-turn happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_plain_text_returns_text(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, client = runtime_with_mock_client
    sess = rt.session()
    result = await sess.execute("hi there")
    assert result.text == "hello"
    assert result.structured is None
    assert result.finish == "end_turn"
    assert result.cost.input_tokens == 100
    assert result.cost.output_tokens == 50
    assert result.cost.provider_id == "bedrock"
    assert result.cost.cost_usd is None  # Pricing table lands in Iteration E

    # Outgoing call: messages contains the user turn; modelId set.
    call_kwargs = client.converse.await_args.kwargs
    assert call_kwargs["modelId"] == rt._default_model
    assert call_kwargs["messages"] == [{"role": "user", "content": [{"text": "hi there"}]}]
    assert "toolConfig" not in call_kwargs


@pytest.mark.asyncio
async def test_execute_structured_uses_forced_tool(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, client = runtime_with_mock_client
    client.converse = AsyncMock(
        return_value=_converse_response(
            text="", tool_input={"summary": "ok", "count": 7}, stop_reason="tool_use"
        )
    )
    sess = rt.session()
    result = await sess.execute("please answer", schema=_Schema)
    assert result.structured == {"summary": "ok", "count": 7}
    assert result.finish == "tool_use"

    call_kwargs = client.converse.await_args.kwargs
    tool_config = call_kwargs["toolConfig"]
    assert tool_config["tools"][0]["toolSpec"]["name"] == SUBMIT_RESULT_TOOL
    assert tool_config["toolChoice"]["tool"]["name"] == SUBMIT_RESULT_TOOL
    # The inputSchema reflects the user's Pydantic schema.
    schema_json = tool_config["tools"][0]["toolSpec"]["inputSchema"]["json"]
    assert schema_json["properties"]["summary"]["type"] == "string"
    assert schema_json["properties"]["count"]["type"] == "integer"


@pytest.mark.asyncio
async def test_execute_schema_missing_tool_use_raises_structured_output(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, client = runtime_with_mock_client
    # Model returns text only, never calls the tool.
    client.converse = AsyncMock(
        return_value=_converse_response(text="I'm just going to say words")
    )
    sess = rt.session()
    with pytest.raises(RuntimeStructuredOutputError) as exc:
        await sess.execute("answer", schema=_Schema)
    assert SUBMIT_RESULT_TOOL in str(exc.value)


@pytest.mark.asyncio
async def test_execute_schema_invalid_payload_raises_structured_output(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, client = runtime_with_mock_client
    # Model called the tool but the payload doesn't validate.
    client.converse = AsyncMock(
        return_value=_converse_response(
            text="",
            tool_input={"summary": "ok"},  # missing 'count'
        )
    )
    sess = rt.session()
    with pytest.raises(RuntimeStructuredOutputError):
        await sess.execute("answer", schema=_Schema)


@pytest.mark.asyncio
async def test_execute_with_system_prompt_lands_on_call(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, client = runtime_with_mock_client
    sess = rt.session(system="You are a helpful assistant.")
    await sess.execute("hi")
    call_kwargs = client.converse.await_args.kwargs
    assert call_kwargs["system"] == [{"text": "You are a helpful assistant."}]


# ---------------------------------------------------------------------------
# Multi-turn message accumulation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_messages_accumulate_across_turns(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, client = runtime_with_mock_client
    client.converse = AsyncMock(
        side_effect=[
            _converse_response(text="first reply"),
            _converse_response(text="second reply"),
        ]
    )
    sess = rt.session()
    await sess.execute("first user")
    await sess.execute("second user")

    # The second call's messages include turn 1's full exchange plus
    # turn 2's user message. The turn-2 assistant message is only
    # appended after the call returns.
    second_call_kwargs = client.converse.await_args_list[1].kwargs
    msgs = second_call_kwargs["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0]["content"][0]["text"] == "first user"
    assert msgs[2]["content"][0]["text"] == "second user"
    # After the second call returns, the buffer carries all four turns.
    assert [m["role"] for m in sess._messages] == ["user", "assistant", "user", "assistant"]


@pytest.mark.asyncio
async def test_failed_call_rolls_back_user_message(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    """A failed converse() call must not leave the user message in history.

    Subsequent successful execute()s would otherwise send a malformed
    history (user-followed-by-user) that Bedrock rejects.
    """
    rt, client = runtime_with_mock_client
    from botocore.exceptions import ClientError

    err = ClientError(
        error_response={
            "Error": {"Code": "ThrottlingException", "Message": "slow"},
            "ResponseMetadata": {"HTTPStatusCode": 429},
        },
        operation_name="Converse",
    )
    client.converse = AsyncMock(side_effect=err)
    sess = rt.session()
    with pytest.raises(RuntimeTransientError):
        await sess.execute("first attempt")
    # Buffer is empty again so the next call sends a clean history.
    assert sess._messages == []


# ---------------------------------------------------------------------------
# stream() chunk translation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_yields_text_deltas_and_turn_complete(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, client = runtime_with_mock_client
    chunks = [
        {"contentBlockStart": {"start": {}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "Hello "}}},
        {"contentBlockDelta": {"delta": {"text": "world"}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 42, "outputTokens": 7}}},
    ]
    client.converse_stream = AsyncMock(return_value=_converse_stream_response(chunks))
    sess = rt.session()
    events = []
    async for event in sess.stream("say hi"):
        events.append(event)

    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    completes = [e for e in events if isinstance(e, TurnComplete)]
    assert [d.text for d in text_deltas] == ["Hello ", "world"]
    assert len(completes) == 1
    final = completes[0].result
    assert final.text == "Hello world"
    assert final.finish == "end_turn"
    assert final.cost.input_tokens == 42
    assert final.cost.output_tokens == 7


@pytest.mark.asyncio
async def test_stream_yields_reasoning_deltas(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, client = runtime_with_mock_client
    chunks = [
        {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "thinking..."}}}},
        {"contentBlockDelta": {"delta": {"text": "answer"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    client.converse_stream = AsyncMock(return_value=_converse_stream_response(chunks))
    sess = rt.session()
    events = []
    async for event in sess.stream("solve this"):
        events.append(event)

    reasoning = [e for e in events if isinstance(e, ReasoningDelta)]
    text = [e for e in events if isinstance(e, TextDelta)]
    assert [r.text for r in reasoning] == ["thinking..."]
    assert [t.text for t in text] == ["answer"]


@pytest.mark.asyncio
async def test_stream_skips_redacted_reasoning_without_crashing(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    """Anthropic-on-Bedrock emits ``reasoningContent`` chunks with only a
    ``redactedContent`` field (no ``text``) for safety-redacted thinking.
    Must skip rather than crash or emit an empty ReasoningDelta.
    """
    rt, client = runtime_with_mock_client
    chunks = [
        {"contentBlockDelta": {"delta": {"reasoningContent": {"redactedContent": "..."}}}},
        {"contentBlockDelta": {"delta": {"text": "answer"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    client.converse_stream = AsyncMock(return_value=_converse_stream_response(chunks))
    sess = rt.session()
    events = []
    async for event in sess.stream("redacted"):
        events.append(event)
    reasoning = [e for e in events if isinstance(e, ReasoningDelta)]
    assert reasoning == []


@pytest.mark.asyncio
async def test_stream_structured_output_assembles_tool_input(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    """Streaming with ``schema=`` assembles ``toolUse.input`` chunks
    into a JSON object and validates against the schema."""
    rt, client = runtime_with_mock_client
    chunks = [
        {
            "contentBlockStart": {
                "start": {"toolUse": {"toolUseId": "tu_1", "name": SUBMIT_RESULT_TOOL}},
                "contentBlockIndex": 0,
            }
        },
        {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"summary": "o'}}}},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": 'k", "count": 9}'}}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "tool_use"}},
    ]
    client.converse_stream = AsyncMock(return_value=_converse_stream_response(chunks))
    sess = rt.session()
    events = []
    async for event in sess.stream("answer this", schema=_Schema):
        events.append(event)

    completes = [e for e in events if isinstance(e, TurnComplete)]
    assert len(completes) == 1
    assert completes[0].result.structured == {"summary": "ok", "count": 9}
    assert completes[0].result.finish == "tool_use"


@pytest.mark.asyncio
async def test_stream_appends_assistant_turn_to_history(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, client = runtime_with_mock_client
    chunks = [
        {"contentBlockDelta": {"delta": {"text": "reply"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    client.converse_stream = AsyncMock(return_value=_converse_stream_response(chunks))
    sess = rt.session()
    async for _ in sess.stream("hello"):
        pass
    assert [m["role"] for m in sess._messages] == ["user", "assistant"]
    assert sess._messages[1]["content"] == [{"text": "reply"}]


# ---------------------------------------------------------------------------
# cancel()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_during_execute_raises_cancelled(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, client = runtime_with_mock_client
    started = asyncio.Event()

    async def _slow_converse(**_: Any) -> dict[str, Any]:
        started.set()
        await asyncio.sleep(5.0)
        return _converse_response()

    client.converse = _slow_converse
    sess = rt.session()
    task = asyncio.create_task(sess.execute("slow"))
    await started.wait()
    await sess.cancel()
    with pytest.raises(RuntimeCancelledError):
        await task
    # Buffer rolled back so the next call sends a clean history.
    assert sess._messages == []


@pytest.mark.asyncio
async def test_cancel_when_idle_is_noop(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, _ = runtime_with_mock_client
    sess = rt.session()
    await sess.cancel()
    await sess.cancel()


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_clears_messages(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, _ = runtime_with_mock_client
    sess = rt.session()
    await sess.execute("hi")
    assert len(sess._messages) == 2
    await sess.close()
    assert sess._messages == []


@pytest.mark.asyncio
async def test_close_is_idempotent(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, _ = runtime_with_mock_client
    sess = rt.session()
    await sess.close()
    await sess.close()
    await sess.close()


@pytest.mark.asyncio
async def test_execute_after_close_raises(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, _ = runtime_with_mock_client
    sess = rt.session()
    await sess.close()
    with pytest.raises(RuntimeError):
        await sess.execute("nope")


# ---------------------------------------------------------------------------
# unwrap()
# ---------------------------------------------------------------------------


def test_session_unwrap_self(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, _ = runtime_with_mock_client
    sess = rt.session()
    assert sess.unwrap(BedrockSession) is sess


def test_session_unwrap_unrelated_raises(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, _ = runtime_with_mock_client
    sess = rt.session()

    class _Unrelated:
        pass

    with pytest.raises(TypeError) as exc:
        sess.unwrap(_Unrelated)
    # Should point users at the runtime-level escape hatch.
    assert "runtime" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Runtime.execute() sugar
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runtime_execute_is_sugar_for_session(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, client = runtime_with_mock_client
    client.converse = AsyncMock(
        return_value=_converse_response(text="", tool_input={"summary": "ok", "count": 1})
    )
    result = await rt.execute("hi", schema=_Schema)
    assert result.structured == {"summary": "ok", "count": 1}


# ---------------------------------------------------------------------------
# Iteration B gates — thinking / budget kwargs decline cleanly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_thinking_raises_unsupported(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, _ = runtime_with_mock_client
    sess = rt.session()
    with pytest.raises(UnsupportedFeatureError) as exc:
        await sess.execute("hi", thinking="medium")
    assert exc.value.feature == Feature.REASONING_EFFORT


@pytest.mark.asyncio
async def test_execute_max_turns_raises_unsupported(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, _ = runtime_with_mock_client
    sess = rt.session()
    with pytest.raises(UnsupportedFeatureError) as exc:
        await sess.execute("hi", max_turns=3)
    assert exc.value.feature == Feature.BUDGET_TURN_CAP


@pytest.mark.asyncio
async def test_execute_polymorphic_prompt_raises_unsupported(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    from airframe.inputs import ImageInput

    rt, _ = runtime_with_mock_client
    sess = rt.session()
    with pytest.raises(UnsupportedFeatureError) as exc:
        await sess.execute(["caption:", ImageInput(path="/tmp/nope.png")])
    assert exc.value.feature == Feature.VISION_INPUT


# ---------------------------------------------------------------------------
# Error classification at the converse boundary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validation_exception_with_model_maps_to_model_not_found(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, client = runtime_with_mock_client
    from botocore.exceptions import ClientError

    err = ClientError(
        error_response={
            "Error": {
                "Code": "ValidationException",
                "Message": "The provided model identifier is invalid.",
            },
            "ResponseMetadata": {"HTTPStatusCode": 400},
        },
        operation_name="Converse",
    )
    client.converse = AsyncMock(side_effect=err)
    sess = rt.session()
    with pytest.raises(RuntimeModelNotFoundError):
        await sess.execute("hi")


@pytest.mark.asyncio
async def test_validation_exception_context_maps_to_overflow(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, client = runtime_with_mock_client
    from botocore.exceptions import ClientError

    err = ClientError(
        error_response={
            "Error": {
                "Code": "ValidationException",
                "Message": "Input exceeds maximum context length.",
            },
            "ResponseMetadata": {"HTTPStatusCode": 400},
        },
        operation_name="Converse",
    )
    client.converse = AsyncMock(side_effect=err)
    sess = rt.session()
    with pytest.raises(RuntimeContextOverflowError):
        await sess.execute("hi")


# ---------------------------------------------------------------------------
# _build_submit_result_tool_config — pure helper
# ---------------------------------------------------------------------------


def test_build_submit_result_tool_config_shape() -> None:
    cfg = _build_submit_result_tool_config(_Schema)
    assert cfg["toolChoice"] == {"tool": {"name": SUBMIT_RESULT_TOOL}}
    spec = cfg["tools"][0]["toolSpec"]
    assert spec["name"] == SUBMIT_RESULT_TOOL
    assert _Schema.__name__ in spec["description"]
    # inputSchema.json is the literal Pydantic JSON Schema.
    assert spec["inputSchema"]["json"] == _Schema.model_json_schema()


def test_submit_result_tool_config_is_json_serialisable() -> None:
    """Bedrock expects pure JSON in ``toolConfig`` — no Pydantic objects."""
    cfg = _build_submit_result_tool_config(_Schema)
    json.dumps(cfg)
