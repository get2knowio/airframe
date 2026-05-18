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
    # Default model is in _BEDROCK_PRICING — cost is computed.
    assert result.cost.cost_usd is not None
    assert result.cost.cost_usd > 0

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
# Iteration E — budget caps + lifecycle hooks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_max_turns_accepts_then_trips_on_next_turn(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    from airframe.errors import RuntimeBudgetExceededError

    rt, client = runtime_with_mock_client
    sess = rt.session()
    # First turn passes (turn_count goes 0 -> 1).
    await sess.execute("first", max_turns=1)
    # Second turn would push turn_count past the cap.
    with pytest.raises(RuntimeBudgetExceededError) as exc:
        await sess.execute("second", max_turns=1)
    assert exc.value.kind == "turns"


@pytest.mark.asyncio
async def test_execute_max_budget_usd_trips_when_exceeded(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    from airframe.errors import RuntimeBudgetExceededError

    rt, client = runtime_with_mock_client
    sess = rt.session()
    # First call costs a tiny amount; second call would already be at the cap.
    await sess.execute("first")
    # Force the running total above any small cap.
    sess._cumulative_cost_usd = 1.0
    with pytest.raises(RuntimeBudgetExceededError) as exc:
        await sess.execute("second", max_budget_usd=0.5)
    assert exc.value.kind == "usd"


# ---------------------------------------------------------------------------
# Iteration C — vision + file input + thinking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_with_image_bytes_translates_to_image_block(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    from airframe.inputs import ImageInput

    rt, client = runtime_with_mock_client
    sess = rt.session()
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    await sess.execute(["What's in this image?", ImageInput(bytes_=png_bytes)])
    call_kwargs = client.converse.await_args.kwargs
    blocks = call_kwargs["messages"][0]["content"]
    # First block is the text portion; image block follows.
    assert blocks[0] == {"text": "What's in this image?"}
    assert blocks[1] == {"image": {"format": "png", "source": {"bytes": png_bytes}}}


@pytest.mark.asyncio
async def test_execute_with_image_path_reads_and_translates(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
    tmp_path: Any,
) -> None:
    from airframe.inputs import ImageInput

    rt, client = runtime_with_mock_client
    img_path = tmp_path / "diagram.jpeg"
    payload = b"\xff\xd8\xff\xe0" + b"\x00" * 32
    img_path.write_bytes(payload)
    sess = rt.session()
    await sess.execute(["caption:", ImageInput(path=str(img_path))])
    blocks = client.converse.await_args.kwargs["messages"][0]["content"]
    image_block = next(b for b in blocks if "image" in b)
    assert image_block["image"]["format"] == "jpeg"
    assert image_block["image"]["source"]["bytes"] == payload


@pytest.mark.asyncio
async def test_execute_with_image_url_raises_unsupported(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    """Converse needs bytes locally — URL fetching is on the caller."""
    from airframe.inputs import ImageInput

    rt, _ = runtime_with_mock_client
    sess = rt.session()
    with pytest.raises(UnsupportedFeatureError) as exc:
        await sess.execute(["see:", ImageInput(url="https://example.com/x.png")])
    assert exc.value.feature == Feature.VISION_INPUT


@pytest.mark.asyncio
async def test_execute_with_unknown_image_format_raises(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    from airframe.inputs import ImageInput

    rt, _ = runtime_with_mock_client
    sess = rt.session()
    # Random non-magic bytes, no path extension.
    with pytest.raises(UnsupportedFeatureError):
        await sess.execute(["see:", ImageInput(bytes_=b"not-an-image-header")])


@pytest.mark.asyncio
async def test_execute_with_file_input_translates_to_document_block(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
    tmp_path: Any,
) -> None:
    from airframe.inputs import FileInput

    rt, client = runtime_with_mock_client
    doc_path = tmp_path / "spec.pdf"
    doc_path.write_bytes(b"%PDF-1.4 minimal payload")
    sess = rt.session()
    await sess.execute(["summarise:", FileInput(path=str(doc_path))])
    blocks = client.converse.await_args.kwargs["messages"][0]["content"]
    doc_block = next(b for b in blocks if "document" in b)
    assert doc_block["document"]["format"] == "pdf"
    assert doc_block["document"]["name"] == "spec"
    assert doc_block["document"]["source"]["bytes"].startswith(b"%PDF-1.4")


@pytest.mark.asyncio
async def test_execute_with_file_input_unknown_format_raises(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
    tmp_path: Any,
) -> None:
    from airframe.inputs import FileInput

    rt, _ = runtime_with_mock_client
    weird = tmp_path / "thing.xyz"
    weird.write_bytes(b"data")
    sess = rt.session()
    with pytest.raises(UnsupportedFeatureError):
        await sess.execute(["look:", FileInput(path=str(weird))])


@pytest.mark.asyncio
async def test_execute_thinking_low_lands_on_additional_request_fields(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, client = runtime_with_mock_client
    sess = rt.session()  # Default model is anthropic.claude-3-5-haiku-...
    await sess.execute("solve this", thinking="low")
    additional = client.converse.await_args.kwargs.get("additionalModelRequestFields")
    assert additional == {"thinking": {"type": "enabled", "budget_tokens": 1024}}


@pytest.mark.asyncio
async def test_execute_thinking_high_uses_32k_budget(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, client = runtime_with_mock_client
    sess = rt.session()
    await sess.execute("solve this", thinking="high")
    additional = client.converse.await_args.kwargs.get("additionalModelRequestFields")
    assert additional == {"thinking": {"type": "enabled", "budget_tokens": 32768}}


@pytest.mark.asyncio
async def test_execute_thinking_explicit_budget_passes_through(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, client = runtime_with_mock_client
    sess = rt.session()
    await sess.execute("solve this", thinking={"budget_tokens": 4096})
    additional = client.converse.await_args.kwargs.get("additionalModelRequestFields")
    assert additional == {"thinking": {"type": "enabled", "budget_tokens": 4096}}


@pytest.mark.asyncio
async def test_execute_thinking_disabled_omits_field(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, client = runtime_with_mock_client
    sess = rt.session()
    await sess.execute("hi", thinking="disabled")
    assert "additionalModelRequestFields" not in client.converse.await_args.kwargs


@pytest.mark.asyncio
async def test_execute_thinking_for_non_anthropic_silently_drops(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    """Bedrock ignores additionalModelRequestFields per vendor;
    saving the field for vendors that won't honour it is honest."""
    from airframe.protocol import ProviderModel

    rt, client = runtime_with_mock_client
    sess = rt.session(model=ProviderModel("bedrock", "amazon.nova-pro-v1:0"))
    await sess.execute("solve this", thinking="medium")
    assert "additionalModelRequestFields" not in client.converse.await_args.kwargs


@pytest.mark.asyncio
async def test_execute_thinking_for_inference_profile_routes_anthropic(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    """Region-prefixed inference profile IDs still recognise as Anthropic."""
    from airframe.protocol import ProviderModel

    rt, client = runtime_with_mock_client
    sess = rt.session(
        model=ProviderModel("bedrock", "us.anthropic.claude-3-5-sonnet-20241022-v2:0")
    )
    await sess.execute("solve this", thinking="medium")
    additional = client.converse.await_args.kwargs.get("additionalModelRequestFields")
    assert additional == {"thinking": {"type": "enabled", "budget_tokens": 8192}}


@pytest.mark.asyncio
async def test_execute_thinking_minimal_coerces_to_low(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, client = runtime_with_mock_client
    sess = rt.session()
    await sess.execute("solve this", thinking="minimal")
    additional = client.converse.await_args.kwargs.get("additionalModelRequestFields")
    assert additional == {"thinking": {"type": "enabled", "budget_tokens": 1024}}


@pytest.mark.asyncio
async def test_execute_thinking_unknown_string_raises(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, _ = runtime_with_mock_client
    sess = rt.session()
    with pytest.raises(UnsupportedFeatureError):
        await sess.execute("hi", thinking="ultra")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_execute_thinking_dict_missing_budget_raises(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, _ = runtime_with_mock_client
    sess = rt.session()
    with pytest.raises(UnsupportedFeatureError):
        await sess.execute("hi", thinking={"some_other_key": 100})


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


# ---------------------------------------------------------------------------
# Iteration D — function tools + permission gating
# ---------------------------------------------------------------------------


class _AddParams(BaseModel):
    a: int
    b: int


def _make_calc_tool(handler=None):
    from airframe.tools import FunctionTool

    async def default_handler(p: _AddParams) -> int:
        return p.a + p.b

    return FunctionTool(
        name="calculator",
        description="Add two integers.",
        params=_AddParams,
        handler=handler or default_handler,
    )


def _tool_use_response(
    tool_use_id: str, name: str, args: dict[str, Any], stop_reason: str = "tool_use"
) -> dict[str, Any]:
    """Build a converse response where the model only calls a tool."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": tool_use_id, "name": name, "input": args}}],
            }
        },
        "usage": {"inputTokens": 50, "outputTokens": 20},
        "stopReason": stop_reason,
    }


@pytest.mark.asyncio
async def test_execute_with_tool_dispatches_handler_and_appends_result(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, client = runtime_with_mock_client
    tool = _make_calc_tool()
    # First call: model invokes calculator(a=17, b=25). Second: final text.
    client.converse = AsyncMock(
        side_effect=[
            _tool_use_response("tc_1", "calculator", {"a": 17, "b": 25}),
            _converse_response(text="The answer is 42"),
        ]
    )
    sess = rt.session(tools=[tool])
    result = await sess.execute("what is 17 + 25?")
    assert result.text == "The answer is 42"
    # Two converse() calls were made.
    assert client.converse.await_count == 2
    # Second call sees the user message + assistant toolUse + toolResult.
    second_msgs = client.converse.await_args_list[1].kwargs["messages"]
    assert [m["role"] for m in second_msgs] == ["user", "assistant", "user"]
    assert second_msgs[1]["content"][0]["toolUse"]["name"] == "calculator"
    tool_result = second_msgs[2]["content"][0]["toolResult"]
    assert tool_result["toolUseId"] == "tc_1"
    assert tool_result["status"] == "success"
    assert tool_result["content"][0] == {"text": "42"}


@pytest.mark.asyncio
async def test_execute_with_tool_propagates_handler_error_to_model(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, client = runtime_with_mock_client

    async def boom(_p: _AddParams) -> int:
        raise ValueError("intentional")

    tool = _make_calc_tool(handler=boom)
    client.converse = AsyncMock(
        side_effect=[
            _tool_use_response("tc_2", "calculator", {"a": 1, "b": 2}),
            _converse_response(text="Sorry, retried."),
        ]
    )
    sess = rt.session(tools=[tool])
    result = await sess.execute("compute")
    assert result.text == "Sorry, retried."
    tool_result = client.converse.await_args_list[1].kwargs["messages"][2]["content"][0][
        "toolResult"
    ]
    assert tool_result["status"] == "error"
    assert "intentional" in tool_result["content"][0]["text"]


@pytest.mark.asyncio
async def test_execute_with_tool_arg_parse_failure_is_error(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, client = runtime_with_mock_client
    tool = _make_calc_tool()
    client.converse = AsyncMock(
        side_effect=[
            # Wrong arg shape — params validation fails.
            _tool_use_response("tc_3", "calculator", {"a": "not-int", "b": 2}),
            _converse_response(text="Try again."),
        ]
    )
    sess = rt.session(tools=[tool])
    await sess.execute("compute")
    tool_result = client.converse.await_args_list[1].kwargs["messages"][2]["content"][0][
        "toolResult"
    ]
    assert tool_result["status"] == "error"
    assert "schema" in tool_result["content"][0]["text"].lower()


@pytest.mark.asyncio
async def test_execute_with_unknown_tool_name_returns_error(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, client = runtime_with_mock_client
    # No tools registered, but model invents one anyway.
    client.converse = AsyncMock(
        side_effect=[
            _tool_use_response("tc_4", "phantom", {}),
            _converse_response(text="OK."),
        ]
    )
    sess = rt.session()
    await sess.execute("hi")
    tool_result = client.converse.await_args_list[1].kwargs["messages"][2]["content"][0][
        "toolResult"
    ]
    assert tool_result["status"] == "error"
    assert "not registered" in tool_result["content"][0]["text"]


@pytest.mark.asyncio
async def test_execute_tool_loop_caps_at_max_iterations(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    from airframe.adapters.bedrock import MAX_TOOL_ITERATIONS
    from airframe.errors import RuntimeProtocolError

    rt, client = runtime_with_mock_client
    tool = _make_calc_tool()
    # Always invoke the tool; never produce final text.
    client.converse = AsyncMock(
        return_value=_tool_use_response("tc_loop", "calculator", {"a": 1, "b": 1})
    )
    sess = rt.session(tools=[tool])
    with pytest.raises(RuntimeProtocolError) as exc:
        await sess.execute("forever")
    assert str(MAX_TOOL_ITERATIONS) in str(exc.value)
    assert client.converse.await_count == MAX_TOOL_ITERATIONS


@pytest.mark.asyncio
async def test_execute_toolconfig_carries_user_tools_with_schema(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    """schema= and tools= coexist: both ride toolConfig.tools."""
    rt, client = runtime_with_mock_client
    tool = _make_calc_tool()
    client.converse = AsyncMock(
        return_value=_converse_response(
            text="", tool_input={"summary": "ok", "count": 3}, stop_reason="tool_use"
        )
    )
    sess = rt.session(tools=[tool])
    await sess.execute("answer", schema=_Schema)
    tool_config = client.converse.await_args.kwargs["toolConfig"]
    names = [t["toolSpec"]["name"] for t in tool_config["tools"]]
    assert SUBMIT_RESULT_TOOL in names
    assert "calculator" in names
    # toolChoice still pins submit_result.
    assert tool_config["toolChoice"]["tool"]["name"] == SUBMIT_RESULT_TOOL


# --- Permission callback ----------------------------------------------------


class _RecordingCallback:
    """Async callback that records every request + returns a fixed decision."""

    def __init__(self, decision: str = "allow") -> None:
        self.requests: list[Any] = []
        self.decision = decision

    async def handle(self, request: Any) -> str:
        self.requests.append(request)
        return self.decision


@pytest.mark.asyncio
async def test_permission_allow_invokes_handler(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, client = runtime_with_mock_client
    tool = _make_calc_tool()
    cb = _RecordingCallback("allow")
    client.converse = AsyncMock(
        side_effect=[
            _tool_use_response("tc_a", "calculator", {"a": 5, "b": 7}),
            _converse_response(text="12"),
        ]
    )
    sess = rt.session(tools=[tool], on_permission=cb)
    result = await sess.execute("add")
    assert result.text == "12"
    assert len(cb.requests) == 1
    assert cb.requests[0].tool_name == "calculator"
    assert cb.requests[0].tool_args == {"a": 5, "b": 7}
    # The handler ran — second-call toolResult should be success with "12".
    tool_result = client.converse.await_args_list[1].kwargs["messages"][2]["content"][0][
        "toolResult"
    ]
    assert tool_result["status"] == "success"


@pytest.mark.asyncio
async def test_permission_deny_skips_handler_returns_error(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, client = runtime_with_mock_client
    invoked = []

    async def tracked(_p: _AddParams) -> int:
        invoked.append(True)
        return -1

    tool = _make_calc_tool(handler=tracked)
    cb = _RecordingCallback("deny")
    client.converse = AsyncMock(
        side_effect=[
            _tool_use_response("tc_d", "calculator", {"a": 1, "b": 2}),
            _converse_response(text="OK, abandoning."),
        ]
    )
    sess = rt.session(tools=[tool], on_permission=cb)
    await sess.execute("compute")
    assert invoked == []
    tool_result = client.converse.await_args_list[1].kwargs["messages"][2]["content"][0][
        "toolResult"
    ]
    assert tool_result["status"] == "error"
    assert "denied" in tool_result["content"][0]["text"].lower()


@pytest.mark.asyncio
async def test_permission_defer_falls_through_to_allow(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, client = runtime_with_mock_client
    tool = _make_calc_tool()
    cb = _RecordingCallback("defer")
    client.converse = AsyncMock(
        side_effect=[
            _tool_use_response("tc_def", "calculator", {"a": 2, "b": 3}),
            _converse_response(text="5"),
        ]
    )
    sess = rt.session(tools=[tool], on_permission=cb)
    result = await sess.execute("compute")
    assert result.text == "5"


@pytest.mark.asyncio
async def test_permission_callback_raise_returns_error_to_model(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, client = runtime_with_mock_client
    tool = _make_calc_tool()

    class _BoomCallback:
        async def handle(self, _request: Any) -> str:
            raise RuntimeError("permission system broke")

    client.converse = AsyncMock(
        side_effect=[
            _tool_use_response("tc_boom", "calculator", {"a": 1, "b": 1}),
            _converse_response(text="Got it"),
        ]
    )
    sess = rt.session(tools=[tool], on_permission=_BoomCallback())
    await sess.execute("compute")
    tool_result = client.converse.await_args_list[1].kwargs["messages"][2]["content"][0][
        "toolResult"
    ]
    assert tool_result["status"] == "error"


# --- Streaming tool events ---------------------------------------------------


def _tool_use_stream_chunks(
    tool_use_id: str, name: str, args_json: str, stop_reason: str = "tool_use"
) -> list[dict[str, Any]]:
    """Build a streaming sequence where the model fires a single tool call."""
    return [
        {
            "contentBlockStart": {
                "start": {"toolUse": {"toolUseId": tool_use_id, "name": name}},
                "contentBlockIndex": 0,
            }
        },
        {
            "contentBlockDelta": {
                "delta": {"toolUse": {"input": args_json}},
                "contentBlockIndex": 0,
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": stop_reason}},
    ]


def _final_text_stream_chunks(text: str) -> list[dict[str, Any]]:
    return [
        {"contentBlockStart": {"start": {}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": text}, "contentBlockIndex": 0}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 30, "outputTokens": 5}}},
    ]


@pytest.mark.asyncio
async def test_lifecycle_hooks_fire_around_execute(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    """session_start fires on first execute; user_prompt_submit on each;
    session_end fires on close()."""
    rt, _ = runtime_with_mock_client
    events: list[Any] = []
    sess = rt.session(on_event=events.append)
    # No events fire on bare session() — hook plumbing waits for first turn.
    assert events == []
    await sess.execute("first")
    kinds_after_first = [e.kind for e in events]
    assert kinds_after_first == ["session_start", "user_prompt_submit"]
    await sess.execute("second")
    assert [e.kind for e in events][-1] == "user_prompt_submit"
    # session_start fired exactly once.
    assert sum(1 for e in events if e.kind == "session_start") == 1
    await sess.close()
    assert events[-1].kind == "session_end"
    # session_end payload carries the running cost + turn count.
    assert events[-1].payload["turn_count"] == 2
    assert events[-1].payload["cumulative_cost_usd"] > 0


@pytest.mark.asyncio
async def test_session_end_skipped_on_never_used_session(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    """Closing a session that never ran a turn doesn't fire session_end."""
    rt, _ = runtime_with_mock_client
    events: list[Any] = []
    sess = rt.session(on_event=events.append)
    await sess.close()
    assert events == []


@pytest.mark.asyncio
async def test_session_end_fires_only_once(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, _ = runtime_with_mock_client
    events: list[Any] = []
    sess = rt.session(on_event=events.append)
    await sess.execute("hi")
    await sess.close()
    await sess.close()
    await sess.close()
    end_count = sum(1 for e in events if e.kind == "session_end")
    assert end_count == 1


@pytest.mark.asyncio
async def test_tool_hooks_fire_around_handler(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    """pre_tool_use fires before the handler; post_tool_use fires after success."""
    rt, client = runtime_with_mock_client
    tool = _make_calc_tool()
    client.converse = AsyncMock(
        side_effect=[
            _tool_use_response("tc_hooks", "calculator", {"a": 1, "b": 2}),
            _converse_response(text="3"),
        ]
    )
    events: list[Any] = []
    sess = rt.session(tools=[tool], on_event=events.append)
    await sess.execute("compute")
    kinds = [e.kind for e in events]
    assert "pre_tool_use" in kinds
    assert "post_tool_use" in kinds
    # pre fires before post.
    assert kinds.index("pre_tool_use") < kinds.index("post_tool_use")
    pre = next(e for e in events if e.kind == "pre_tool_use")
    assert pre.payload["tool_name"] == "calculator"
    assert pre.payload["tool_call_id"] == "tc_hooks"
    post = next(e for e in events if e.kind == "post_tool_use")
    assert post.payload["output"] == 3


@pytest.mark.asyncio
async def test_tool_failure_hook_fires_on_handler_error(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    """tool_failure fires (in place of post_tool_use) when handler raises."""
    rt, client = runtime_with_mock_client

    async def boom(_p: _AddParams) -> int:
        raise RuntimeError("nope")

    tool = _make_calc_tool(handler=boom)
    client.converse = AsyncMock(
        side_effect=[
            _tool_use_response("tc_fail", "calculator", {"a": 1, "b": 2}),
            _converse_response(text="Skipping."),
        ]
    )
    events: list[Any] = []
    sess = rt.session(tools=[tool], on_event=events.append)
    await sess.execute("compute")
    kinds = [e.kind for e in events]
    assert "tool_failure" in kinds
    assert "post_tool_use" not in kinds
    fail = next(e for e in events if e.kind == "tool_failure")
    assert "nope" in fail.payload["error"]


@pytest.mark.asyncio
async def test_cumulative_cost_accumulates_across_turns(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    rt, _ = runtime_with_mock_client
    sess = rt.session()
    await sess.execute("first")
    cost_after_one = sess._cumulative_cost_usd
    await sess.execute("second")
    assert sess._cumulative_cost_usd == pytest.approx(2 * cost_after_one)
    assert sess._turn_count == 2


# --- pricing table -----------------------------------------------------------


def test_compute_cost_usd_known_model() -> None:
    from airframe.adapters.bedrock import _compute_cost_usd

    # Default model: anthropic.claude-3-5-haiku — 0.0008 input, 0.004 output per 1k.
    # 1000 in + 500 out = 1.0 * 0.0008 + 0.5 * 0.004 = 0.0028
    cost = _compute_cost_usd(
        "anthropic.claude-3-5-haiku-20241022-v1:0",
        input_tokens=1000,
        output_tokens=500,
    )
    assert cost == pytest.approx(0.0028, rel=1e-3)


def test_compute_cost_usd_unknown_model_is_none() -> None:
    from airframe.adapters.bedrock import _compute_cost_usd

    # Inference-profile prefix not in the table.
    cost = _compute_cost_usd(
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        input_tokens=1000,
        output_tokens=500,
    )
    assert cost is None
    # PT ARN not in the table either.
    cost = _compute_cost_usd(
        "arn:aws:bedrock:us-east-1:123456789012:provisioned-model/abc",
        input_tokens=1,
        output_tokens=1,
    )
    assert cost is None


def test_compute_cost_usd_curated_models_all_have_rates() -> None:
    """Every entry in _BEDROCK_METADATA must also appear in _BEDROCK_PRICING.

    Pinned so adding a model to the metadata catalog without
    pricing surfaces as a test failure — easy mistake to make.
    """
    from airframe.adapters.bedrock import _BEDROCK_METADATA, _BEDROCK_PRICING

    missing = set(_BEDROCK_METADATA) - set(_BEDROCK_PRICING)
    assert not missing, f"models in metadata but missing from pricing: {sorted(missing)}"


@pytest.mark.asyncio
async def test_stream_emits_tool_call_events_and_loops_to_final(
    runtime_with_mock_client: tuple[BedrockRuntime, MagicMock],
) -> None:
    from airframe.events import ToolCallResult, ToolCallStart

    rt, client = runtime_with_mock_client
    tool = _make_calc_tool()
    first = _converse_stream_response(
        _tool_use_stream_chunks("tc_s", "calculator", '{"a": 9, "b": 1}')
    )
    second = _converse_stream_response(_final_text_stream_chunks("10"))
    client.converse_stream = AsyncMock(side_effect=[first, second])
    sess = rt.session(tools=[tool])
    events = []
    async for event in sess.stream("9 + 1?"):
        events.append(event)

    starts = [e for e in events if isinstance(e, ToolCallStart)]
    results = [e for e in events if isinstance(e, ToolCallResult)]
    completes = [e for e in events if isinstance(e, TurnComplete)]
    assert len(starts) == 1
    assert starts[0].tool_name == "calculator"
    assert starts[0].tool_call_id == "tc_s"
    assert len(results) == 1
    assert results[0].tool_call_id == "tc_s"
    assert results[0].is_error is False
    assert len(completes) == 1
    assert completes[0].result.text == "10"
    assert completes[0].result.finish == "end_turn"
