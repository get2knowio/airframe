# `prior_messages=` (history-seed) protocol feature plan

Companion to [`implementation-plan.md`](./implementation-plan.md). Specs
a new portable protocol surface on `AgentRuntime.session()`:
``prior_messages: list[HistoryMessage] | None``. Adapters seed their
internal message buffer (or server-side session, where supported)
from the caller-supplied transcript before the first turn runs.

Targeted at one specific deployment shape — **stateless backend,
frontend owns the transcript** — which the existing three multi-turn
mechanisms (long-lived `AgentSession`, `resume=<server_session_id>`,
single-call multi-turn loop) don't serve.

## Motivation

Today, airframe supports three multi-turn shapes:

| Mechanism | Assumption |
|---|---|
| Hold the `AgentSession` object in memory across requests | The Python process handling turn N is the one that handled turn N-1. Sticky sessions / in-memory state. |
| `runtime.session(resume=<id>)` | The adapter has server-issued session ids (declares `Feature.SESSION_RESUME`). Claude / Copilot / Kimi / OpenCode Server do; OpenAI-compat / Bedrock don't. |
| Single `session.execute()` drives an internal multi-turn agent loop | The "conversation" is one logical task. Tool-use sub-turns happen inside one user-facing question. |

The fourth shape — **bring-your-own-transcript** — isn't covered.
It's the canonical pattern for:

* React (or any client-rendered) chat UIs where the transcript lives
  in client state.
* Serverless deployments (Lambda, Cloud Run, fly.io machines) where
  each request lands on a fresh process.
* Horizontally-scaled backends without sticky sessions.
* CLI / agent tools that persist transcripts to disk and replay them.

Concrete consumer: Earlybird's `ask` feature — chat panel that POSTs
`{ history: [...prior turns], message: "new question" }` per request.
The backend is a pure function from `(transcript, new message)` to
streamed response. Today this consumer either has to (a) paste prior
turns into the first user message as plain text (works, but loses
role boundaries and defeats vendor prompt caching), or (b) reach
under airframe via `runtime.unwrap(AsyncOpenAI)` and build the
`messages=[...]` list themselves (defeats the abstraction).

## Non-goals

* **Tool-call replay in v1.** The vendor wire-shape divergence
  (OpenAI's `tool_calls` + `role="tool"` vs Anthropic's `tool_use` +
  `tool_result` content blocks vs Kimi's WireMessage shape) is real
  engineering. Ship text-only history first; defer tool-call replay
  to a follow-up iteration once a concrete consumer asks. ~80% of
  the chat-UI value is in text-only.
* **Auto-truncation.** Caller-supplied history may exceed the model's
  context window. Airframe doesn't trim — that's a vendor-aware
  decision belonging to the consumer (token counts, what to drop
  first, summarisation strategy all depend on the model). Adapters
  pass through and let the vendor raise its native context-overflow
  error, which airframe already classifies as
  :class:`RuntimeContextOverflowError`.
* **Conversation-state mutation mid-session.** ``prior_messages`` is
  seed-only — passed once at ``session()`` construction. The session
  doesn't expose a "rewind, edit turn 3, replay" API. Consumers that
  want to edit history rebuild the transcript and open a new session.
* **A "load from URL" or "load from file" convenience.** Just a
  Python list of typed messages. Persistence is the consumer's
  problem.

## The shape

```python
from dataclasses import dataclass
from typing import Any, Literal

@dataclass(frozen=True, slots=True)
class HistoryMessage:
    """One turn of caller-supplied conversation history.

    Seed for ``AgentRuntime.session(prior_messages=...)``. Adapters
    translate to their native message shape at session-construction
    time.

    Attributes:
        role: ``"user"`` or ``"assistant"``. ``"system"`` is not
            accepted here — system prompts go through the existing
            ``system=`` kwarg on ``session()``.
        content: The message text. Adapters that natively support
            multi-part content blocks may translate downstream, but
            the portable seed shape is plain string.
    """
    role: Literal["user", "assistant"]
    content: str

# Future-additive (NOT in v1 — deferred to a follow-up iteration):
#
# @dataclass(frozen=True, slots=True)
# class HistoryToolCall:
#     name: str
#     arguments: dict[str, Any]
#     result: Any
#     is_error: bool = False
#     tool_call_id: str | None = None
#
# HistoryMessage gains: tool_calls: list[HistoryToolCall] | None = None
```

`AgentRuntime.session()` gains a new kwarg:

```python
def session(
    self,
    *,
    resume: str | None = None,
    system: str | None = None,
    model: ProviderModel | None = None,
    tools: list[FunctionTool] | None = None,
    mcp_servers: list[McpServerRef] | None = None,
    on_permission: PermissionCallback | None = None,
    on_event: Callable[[HookEvent], None] | None = None,
    provider_options: ProviderOptions | None = None,
    prior_messages: list[HistoryMessage] | None = None,   # ← new
) -> AgentSession: ...
```

## New `Feature` flag

```python
# In airframe.features.Feature:
HISTORY_SEED = "history_seed"
```

Declared True by every adapter that wires ``prior_messages=``.
Consumers branch on ``runtime.supports(Feature.HISTORY_SEED)`` before
relying on the kwarg. Mismatched-feature path raises
:class:`UnsupportedFeatureError` (matching every other capability
gate in airframe).

Name rationale: `HISTORY_SEED` over `PRIOR_MESSAGES` because the
*semantic* is "seed the session with this state", not "we have
prior messages". The flag describes the *capability* (can you seed
history?) rather than the *kwarg* (what's the kwarg called?).

## Mutual exclusion with `resume=`

``prior_messages=`` and ``resume=`` are different mechanisms:

* ``resume=<server_session_id>`` — adopt an existing server-side
  conversation. The adapter doesn't send any history because the
  server already has it.
* ``prior_messages=[...]`` — seed a *fresh* session with caller-
  supplied turns. The adapter packages them in whatever shape the
  vendor expects.

Passing both at the same ``session()`` call raises
:class:`UnsupportedFeatureError` with a message clarifying the
distinction. (Future extension: an adapter could conceivably accept
both — "resume the server session, then prepend this client-side
history" — but the semantics are ambiguous enough that v1 declines
the combination.)

## Per-adapter mapping

| Adapter | Native seed point | Notes |
|---|---|---|
| **OpenAI-compat** (OpenCode Zen / Go, OpenRouter, future siblings) | `OpenAICompatibleSession._messages: list[ChatCompletionMessageParam]` | Trivial — already client-side buffer. Each `HistoryMessage` becomes one entry with `role` + `content`. |
| **Bedrock** | `BedrockSession._messages: list[dict]` (Converse `messages=` list) | Trivial — same shape. Anthropic-on-Bedrock with caller-supplied long history may want a prompt-cache hint, but that's additive. |
| **Claude Code** | `ClaudeAgentOptions` has no "seed history" slot natively — the SDK starts fresh each connect | Use the SDK's `append_system_prompt` / system-prompt prefix? **No** — that conflates roles. Better: prepend by replaying each `HistoryMessage` as `client.query(role, content)` calls before the user's actual turn fires. Verify this leaves the session in a clean state for the real first turn. |
| **Copilot** | `CopilotSession` accepts initial messages on `create_session`? | Investigate at impl time. If not, same replay-then-execute pattern as Claude. |
| **Kimi** | `kimi_agent_sdk.Session` — investigate `Session.create()` history kwarg | Likely a replay pattern. |
| **OpenCode Server** | Server-side sessions; `prior_messages=` either declines (use `resume=` to adopt a server session instead) or pushes each as `client.session.chat(parts=...)` pre-turn | Likely declines — the value prop of OpenCode is the server-side session, so clients wanting history-seed should use the OpenAI-compat path through OpenCode instead. |

**The declaration matrix** (target state after v1):

| Feature | Bedrock | Claude | Copilot | Kimi | OpenAI-compat | OpenCode |
|---|---|---|---|---|---|---|
| `HISTORY_SEED` | ✓ | ✓ | ✓ (TBD at impl time) | ✓ (TBD) | ✓ | ✗ (use `resume=` instead) |

## Iteration breakdown

Three iterations, ~250 LOC + tests total.

### Iteration A — Protocol scaffolding + text-only OpenAI-compat + Bedrock

~150 LOC. Lands the shape against the two adapters where it's
trivial — they already maintain `_messages` buffers we'd seed
directly.

* `HistoryMessage` dataclass in `src/airframe/inputs.py` (alongside
  `ImageInput` / `FileInput` — same family of "consumer-supplied
  typed input shapes").
* `Feature.HISTORY_SEED` added to `src/airframe/features.py`.
* `AgentRuntime.session()` protocol signature gains
  ``prior_messages: list[HistoryMessage] | None = None``.
* Shared helper in `src/airframe/sessions.py`:

  ```python
  def _check_prior_messages_supported(
      prior_messages: list[HistoryMessage] | None,
      *,
      adapter_label: str,
      supports: Callable[[Feature], bool],
  ) -> None:
      """Raise UnsupportedFeatureError when caller passes prior_messages
      and the adapter doesn't declare HISTORY_SEED."""

  def _validate_prior_messages(
      prior_messages: list[HistoryMessage] | None,
  ) -> None:
      """Validate the seed list: alternating turns, no empty content,
      role in {"user", "assistant"}, etc. Raises ValueError."""
  ```
* `OpenAICompatibleSession.__init__` accepts `prior_messages=`,
  seeds `self._messages` before any turn runs. Add `Feature.HISTORY_SEED`
  to `OpenAICompatibleRuntime.SUPPORTED_FEATURES`. Update
  `OpenAICompatibleRuntime.session()` to thread the kwarg through.
* `BedrockSession.__init__` same pattern. `BedrockRuntime.session()`
  threads the kwarg.
* `_open_thin_session` accepts the kwarg too — passes to the thin
  session for adapters that haven't yet wired their bespoke session
  class (third-party adapter authors will hit this path).
* Mutual exclusion: when ``prior_messages`` and ``resume`` are both
  non-None, raise ``UnsupportedFeatureError``.
* Unit tests in `tests/test_inputs.py` for `HistoryMessage`
  validation, `tests/test_features.py` for the matrix declaration,
  `tests/test_opencode_zen_session.py` /
  `tests/test_bedrock_session.py` for the seed behaviour.

**Stopping point.** Earlybird's use case is fully served — they're
on OpenAI-compat. Bedrock users with the same shape get it
simultaneously. Claude / Copilot / Kimi / OpenCode Server callers
get a clean `UnsupportedFeatureError` (correct, not silently
wrong) until iteration B / C wire them.

### Iteration B — SDK-based adapter coverage (Claude / Copilot / Kimi)

~80 LOC. Lands the same surface on the subprocess / native-session
adapters.

For each, the pattern is:
* If the SDK's session-create accepts a history list, pass it.
* If not, replay each `HistoryMessage` as a no-tool, no-action
  pre-turn against the live session before the user's first real
  turn fires. Verify the underlying SDK doesn't run the agent loop
  during replay (it shouldn't — these are assistant-shaped messages
  the SDK treats as context, not new user turns).
* `_check_provider_options` /
  `_check_prior_messages_supported` gates as appropriate.
* Per-adapter unit tests against the mocked SDK to verify the
  history landed in the right shape.

**Open question for impl time:** does `ClaudeSDKClient` accept
caller-supplied prior turns cleanly, or does the replay introduce
duplicate `user_prompt_submit` hook events / cost-record entries?
Need to verify hooks fire only for the real first turn, not for
the replay. May need to disable `on_event` during replay.

### Iteration C — Wrap-up (docs, probes, OpenCode-Server decline)

~30 LOC + docs.

* `OpenCodeServerRuntime.session()` raises
  `UnsupportedFeatureError(feature=Feature.HISTORY_SEED)` with a
  message pointing callers at `resume=` instead (since OpenCode's
  server-side sessions are the canonical "carry history" path on
  that adapter).
* `examples/probe_history_seed.py` — multi-turn round-trip against
  every adapter that declares the feature, exercising:
  ```python
  history = [
      HistoryMessage(role="user", content="My name is Alice."),
      HistoryMessage(role="assistant", content="Nice to meet you, Alice."),
  ]
  sess = runtime.session(prior_messages=history)
  result = await sess.execute("What's my name?")
  assert "Alice" in result.text  # model has the context
  ```
* `docs/capabilities.md` — new `HISTORY_SEED` row.
* `docs/architecture.md` — short paragraph on the bring-your-own-
  transcript pattern as the fourth conversation-state mechanism.
* `docs/cookbook.md` — recipe entry.
* `CHANGELOG.md` entry.

## Risks and decisions to flag during execution

1. **Hook double-fire during replay.** Iteration B's replay pattern
   on subprocess SDKs (Claude / Copilot / Kimi) may emit
   `user_prompt_submit` hooks for each replayed turn. The
   `on_event` callback should NOT fire during replay — only for the
   real first turn after seeding. Track this with a `_seeding: bool`
   flag and gate `_fire_hook_event` calls accordingly.
2. **Cost-record contamination during replay.** Same issue — replay
   turns may return tokens=0/cost=0 results that pollute the
   session's cumulative cost. Skip cost accumulation during
   `_seeding=True`.
3. **Schema validation during seed.** `prior_messages` from an
   untrusted source (a deserialised cookie, a client-supplied
   JSON body) could carry malformed roles or content. The validator
   needs to raise `ValueError` (not `UnsupportedFeatureError`)
   for bad shape — the consumer's input is the bug, not the
   adapter's capability.
4. **Tool-call replay deferral is load-bearing.** If a consumer
   asks for tool-call replay before iteration B ships, the answer
   is "compose the tool's output into the assistant message text"
   — same workaround Earlybird is doing today. Document this in
   the docs/architecture.md note.
5. **Context-overflow surfacing.** A 50-turn history seeded against
   a 4k-context model raises whatever the vendor raises. Airframe
   already classifies these as `RuntimeContextOverflowError` per
   adapter; verify each adapter's seed path doesn't swallow the
   error.
6. **Empty `prior_messages=[]` vs `None`.** `None` means "no
   history kwarg passed; default behaviour." `[]` (empty list)
   should be equivalent — not an error, not a silent no-op that
   needs a special log message. Validate at the gate.
7. **`role="system"` deliberately rejected.** System prompts go
   through the existing `system=` kwarg. Accepting `role="system"`
   in `HistoryMessage` would conflict with that and introduce
   precedence questions. Raise `ValueError` on `role="system"`.

## Definition of done

* `HistoryMessage` dataclass shipped on `airframe.inputs`.
* `Feature.HISTORY_SEED` declared True on OpenAI-compat + Bedrock
  + Claude + Copilot + Kimi adapters (the five that ship it after
  iteration B); declined cleanly on OpenCode Server.
* `examples/probe_history_seed.py` round-trips against every
  declaring adapter.
* Conformance contract added:
  `test_session_prior_messages_agrees_with_history_seed_capability`
  in `airframe.testing.contracts`.
* Behavioural integration test in `airframe.testing.integration`:
  `test_integration_history_seed_round_trip`.
* `docs/capabilities.md` matrix row added.
* `docs/architecture.md` fourth-conversation-state-mechanism note.
* `CHANGELOG.md` entry.
* Earlybird (the originating consumer) confirms their `ask` feature
  works end-to-end without the option-A workaround.

## When to start

**v0.9.0 candidate.** The opencode-ai SDK gap (which blocks
OpenCode's tool / permission flags) is the only Phase 1 work that
needs to land first — and even that's independent. History-seed
touches none of the same code paths, so the two can ship in
parallel.

Two signals that would accelerate it:

1. A second consumer beyond Earlybird asking for the same shape.
2. The Bedrock community's prompt-caching feature lands a richer
   API that benefits from explicit transcript seeding. Bedrock
   already supports prompt caching today, but a documented
   `prior_messages=` API is the clean integration point.

## Open questions for the implementer

1. **Should the kwarg name be `prior_messages` or `history`?**
   `prior_messages` mirrors the OpenAI Chat Completions parameter
   name and is explicit about ordering ("prior to the next turn").
   `history` is shorter but ambiguous about whether the *current*
   user message is part of it. Lean `prior_messages`.
2. **Should `HistoryMessage.content` accept a `Prompt` (the
   polymorphic `str | list[PromptPart]` from Phase 2) or just
   `str`?** v1 says `str` for shape simplicity. Future iteration
   can widen to `Prompt` once the vision/file inputs are stable.
3. **Should declining the feature use a separate error type?**
   Today `UnsupportedFeatureError(feature=...)` is the universal
   gate. No reason to special-case. Stay consistent.
4. **OpenCode Server: decline outright, or accept and replay via
   `session.chat()` calls?** Iteration C ships as decline-with-
   pointer-at-`resume=`. The "accept and replay" path is feasible
   but conflates two mechanisms; the cleaner story is "the
   server-side sessions are *the* history mechanism on OpenCode."

## Implementation wiring checklist

Beyond the per-adapter `session()` updates, each iteration touches:

### Source wiring

- [ ] `src/airframe/inputs.py` — `HistoryMessage` dataclass (and
      `__all__` entry).
- [ ] `src/airframe/__init__.py` — export `HistoryMessage` at the
      top level.
- [ ] `src/airframe/features.py` — `Feature.HISTORY_SEED` literal.
- [ ] `src/airframe/protocol.py` — `AgentRuntime.session()`
      signature gains the kwarg + docstring entry.
- [ ] `src/airframe/sessions.py` — `_check_prior_messages_supported`
      + `_validate_prior_messages` helpers.
- [ ] Each adapter's `session()` factory threads the kwarg through.
- [ ] Each adapter's bespoke `*Session.__init__` accepts and seeds.

### Test wiring

- [ ] `tests/test_inputs.py` — `HistoryMessage` field-shape +
      validation tests.
- [ ] `tests/test_features.py` — `HISTORY_SEED` per-adapter
      declaration matrix.
- [ ] `tests/test_<adapter>_session.py` — seed-behaviour tests
      per adapter.
- [ ] `src/airframe/testing/contracts.py` — conformance contract
      for the feature gate.
- [ ] `src/airframe/testing/integration.py` — live round-trip.
- [ ] `tests/test_<adapter>_conformance.py` — import the new
      contract.
- [ ] `tests/test_<adapter>_integration.py` — import the new
      integration test.

### Probe + examples wiring

- [ ] `examples/probe_history_seed.py` — new probe.
- [ ] `examples/probe_supports.py` — picks up the new flag
      automatically.

### Documentation

- [ ] `docs/capabilities.md` — matrix row + per-feature semantics
      section.
- [ ] `docs/architecture.md` — fourth-mechanism note in the
      conversation-state section.
- [ ] `docs/cookbook.md` — recipe entry.
- [ ] `docs/reference.md` — `HistoryMessage` mention + matrix row.
- [ ] Each adapter's `docs/adapters/*.md` — short note in the
      supported-features table.
- [ ] `CHANGELOG.md` — Unreleased entry.

## Closest in-tree templates to read first

| File | What to learn from it |
|---|---|
| `src/airframe/sessions.py` — `_check_hooks_supported`, `_check_provider_options` | Shape of the capability-gate helpers. The new `_check_prior_messages_supported` follows the same pattern. |
| `src/airframe/adapters/openai_compatible.py` | The trivial-seed case — `_messages` buffer is already client-side. ~10 LOC change. |
| `src/airframe/adapters/bedrock.py` | Same shape as OpenAI-compat (`_messages` list, Converse-format dicts). |
| `src/airframe/adapters/claude_code.py` — `_ensure_client` cache key | Replay-pattern impl reference. Cache key must include a fingerprint of `prior_messages` so reconnects don't drop seeded state. |
| `src/airframe/testing/contracts.py` — existing capability-gate contracts | Template for the new conformance contract. |

## First commit in a fresh session

A reasonable Iteration A first commit:

```
src/airframe/features.py                          # +HISTORY_SEED enum value
src/airframe/inputs.py                            # +HistoryMessage dataclass
src/airframe/__init__.py                          # +export
src/airframe/protocol.py                          # +session() signature update
src/airframe/sessions.py                          # +helpers
src/airframe/adapters/openai_compatible.py        # +seed _messages + flip flag
src/airframe/adapters/bedrock.py                  # +seed _messages + flip flag
src/airframe/testing/contracts.py                 # +contract
tests/test_inputs.py                              # +HistoryMessage tests
tests/test_features.py                            # +declaration matrix update
tests/test_opencode_zen_session.py                # +seed behaviour
tests/test_opencode_zen_conformance.py            # +import contract
tests/test_bedrock_session.py                     # +seed behaviour
tests/test_bedrock_conformance.py                 # +import contract
```

That should pass `mise run check` cleanly. Iteration B adds the SDK
adapters; Iteration C adds OpenCode-Server decline + docs.
