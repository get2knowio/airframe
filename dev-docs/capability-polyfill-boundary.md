# Capability polyfill boundary — how far airframe fills gaps

**Status: proposal / design note.** Codifies the line airframe should
hold when "polyfilling" a feature onto adapters whose vendor SDK
lacks it — so we broaden the set of usable backends without drifting
into being an agent framework in our own right. Sibling to
[`feature-roadmap.md`](./feature-roadmap.md) §6 (patterns from mature
abstraction frameworks); this doc is specifically about the *shim vs.
gate* decision.

## The tension

Some airframe backends are full agent runtimes (Claude Agent SDK,
Copilot SDK, OpenCode server): they run their own tool loop, host
web search, manage sessions. Others are bare chat surfaces
(OpenAI-compatible Chat Completions behind `opencode-zen`,
`opencode-go`, `openrouter`, most of Bedrock Converse): text in,
text out, at most a single round-trip `tools=[]` slot.

The appeal of polyfilling is real: if airframe can make web search or
tool calling *work* on the bare backends too, a consumer can pick any
backend for a job that needs those capabilities — maximum features
across maximum abstracted agents. The hazard is equally real:
synthesizing enough capability to make a bare model behave like an
agent means airframe starts *containing* an agent. That is the
Hermes/PI space — projects that are agents in their own right — and
airframe explicitly is not trying to replicate them. It is "JDBC for
agent SDKs," not an agent.

Two well-documented failure modes bracket the sweet spot:

- **The LiteLLM trap** (flatten everything to the lowest common
  denominator; leak the rest through an opaque `extra_body`). Already
  discussed in [`feature-roadmap.md`](./feature-roadmap.md) §6.12.
  Under-abstracting.
- **The LangChain trap** (over-abstracting: hidden base prompts you
  didn't write, model-controlled loops you can't get a stack trace
  out of, five layers of indirection to change one detail). The
  2025–26 backlash against agent frameworks is precisely a backlash
  against *originating* behaviour the caller can't see or replace.
  Over-abstracting.

Polyfilling pushes toward the LangChain end. The rule below keeps us
off that slope.

## The principle: **translate and dispatch, never originate**

Airframe may, on a backend that lacks a native version of a feature:

1. **Translate** — reshape the request and parse the response into
   the vendor-neutral surface (this is the whole job; always fine).
2. **Dispatch** — run a *bounded, mechanical* round-trip loop that
   hands control to a capability **the caller or the vendor supplies**
   (a consumer `FunctionTool` handler; a vendor-hosted native tool).

Airframe must **never originate** a capability: never ship its own web
search, retrieval/memory, code executor, planner, agent-to-agent
router, or self-correction prompt scaffolding. The moment airframe
*contains the thing a tool does* — rather than plumbing a call to
something the caller or vendor provides — it has become an agent.

Restated as three concentric rings:

| Ring | What it is | Polyfill? | Examples |
| --- | --- | --- | --- |
| **0 — Transport** | Request/response shaping over the vendor call | Always | `execute`/`stream`, `schema=`, `thinking=`, vision/file parts, `metadata=`, reasoning-trace extraction, rate-limit parsing |
| **1 — Dispatch** | A capped, mechanical loop that fills in the round-trip for backends whose SDK doesn't run one — dispatching only to caller/vendor-supplied capabilities | Yes, but legible + capped + delegate-to-native | Client-side `FunctionTool` loop on OAI-compat (`max_turns`-bounded); forced-`submit_result` structured output on Copilot |
| **2 — Origination** | Airframe supplies judgment or capability of its own | **Never** (gate with `supports()`) | A built-in web-search/RAG implementation, planner, multi-agent router, conversation memory, retry/fallback policy, hidden prompt scaffolds |

Ring 2 is exactly the list airframe's README already assigns to the
consumer ("retry policy, fallback across vendors, conversation memory,
multi-agent orchestration"). The polyfill boundary is just that same
line, viewed from the capability side instead of the plumbing side.

## Airframe is already correctly placed

This is not a hypothetical — the two tool modules already encode the
two stances, and they're consistent with the principle:

- **`native_tools.py` (hosted tools) = Ring 0, gated.**
  `NativeTool.web_search()` is a *reference* to a vendor-hosted tool.
  It maps to Claude's `WebSearch` where the vendor runs it, and raises
  `UnsupportedFeatureError` on a backend that has none — "graceful
  degradation is the consumer's job, by checking
  `supported_native_tools()`." Airframe never runs the search itself.
  This is the conservative, correct call: **web search is gated, never
  synthesized.**

- **`tools.py` + OAI-compat adapter (function tools) = Ring 1,
  dispatched.** For a caller-supplied `FunctionTool`, the OAI-compat
  adapter *already* runs a client-side loop: read `tool_calls`,
  dispatch each **consumer** handler, append `role="tool"`, re-call,
  bounded by `max_turns`. On the agent SDKs the vendor's own loop does
  this instead. Airframe orchestrates the round-trips but originates
  nothing — every side effect lives in the caller's handler. That is
  the sweet spot in working code.

The `submit_result` forced-structured-output shim on Copilot is the
other Ring-1 citizen: mechanical (advertise one tool, read its args),
using Copilot's own tool primitive, originating no hidden reasoning
prompt.

What airframe pointedly does **not** do — and shouldn't start — is the
thing the roadmap's Agent Skills entry already forbids: "Don't fake
skills on non-supporting adapters by inlining SKILL.md content into
the system prompt … vendor shimming is the kind of 'helpful'
abstraction the codebase deliberately avoids." That instinct is the
Ring-2 tripwire; keep it.

## The decision rule (for the next feature)

When a feature is native on some backends and absent on others, ask in
order:

1. **Is the gap pure request/response shaping?** → Ring 0. Add it;
   it's portable and thin. (Vision, reasoning traces, cache keys,
   token counting all landed here.)
2. **Is the gap a mechanical, *terminating*, capped round-trip loop
   that only dispatches to a capability the caller or vendor supplies?**
   → Ring 1. Allowed, if it stays legible and capped, and if it
   **delegates to the native loop wherever the vendor already runs
   one** (wrap SDKs, don't rewrite them). Never let airframe's loop
   shadow a vendor loop that exists.
3. **Would airframe have to supply the capability, judgment, or a
   prompt the caller didn't write?** → Ring 2. Do **not** polyfill.
   Gate it: `supports(Feature.X)` / `supported_native_tools()` returns
   the truth, `UnsupportedFeatureError` on request, and the consumer
   either brings the capability as a `FunctionTool` or picks a stronger
   backend.

Three guard-rails that keep Ring 1 from rotting into Ring 2:

- **Capped, not model-controlled.** Every synthesized loop has an
  explicit `max_turns`/iteration ceiling the caller sees. (The
  "LangChain fails after 200 steps with no stack trace" failure mode
  is a loop with no visible bound.)
- **No hidden prompts.** Airframe may inject *tool definitions* (a
  `submit_result` schema); it may not inject reasoning/planning
  instructions into the system or user prompt. Hidden base prompts are
  the canonical leaky abstraction.
- **Delegate to native.** If the vendor SDK runs the loop / hosts the
  tool, use it via the adapter; the synthesized version exists *only*
  to fill a genuine gap, never to homogenize behaviour that already
  works natively.

## Worked verdicts

| Candidate polyfill | Ring | Verdict |
| --- | --- | --- |
| Forward a **hosted web-search** flag to backends that have one; gate the rest | 0 | ✅ Shipped (`native_tools.py`). Correct. |
| **Originate** web search for a bare chat model (airframe runs the query) | 2 | ❌ Never. Consumer brings `FunctionTool(name="web_search", handler=…)`; Ring-1 loop dispatches it. Airframe makes it *possible and portable* without *containing* a search engine. |
| **Function tool calling** on bare chat models via a capped client-side loop | 1 | ✅ Shipped (OAI-compat). Line held: caller owns handlers, `max_turns` caps it. |
| **Structured output** on a no-native-JSON backend, via forced tool or a bounded two-call "reason then structure" pass | 1 | ✅ Forced-tool shipped (Copilot). A 2-call variant is defensible (bounded, terminating, no hidden reasoning prompt). Token-level constrained decoding (Outlines/XGrammar) is **not available** to a hosted-SDK wrapper — don't attempt it. |
| **Conversation memory / retrieval** injected by airframe | 2 | ❌ README already assigns memory to the consumer. |
| **Retry / fallback across vendors** baked into the runtime | 2 | ❌ Ships as an optional, legible middleware *shape* above the protocol (roadmap §6.9), never inside an adapter. |
| **Multi-agent routing / planning / self-correction** | 2 | ❌ This *is* Hermes/PI. Out of scope, probably forever. |

## If we ever do want the tool-execution loop to be portable "batteries"

There is one legitimate way to offer more turnkey capability without
becoming an agent: ship the Ring-1 loop as an **explicit, opt-in,
replaceable helper above the protocol** — e.g. a small
`airframe.harness.tool_loop(runtime, tools, prompt, max_turns=…)` —
rather than burying it deeper in adapters. Precedent: OpenAI's
`tool_runner`, Anthropic's tool-use loop, and Vercel AI SDK's
`maxSteps` all ship a *bounded, legible* loop as a helper on top of
the transport primitive, not as the transport itself. The test it must
pass: a consumer can read it in one sitting, cap it, and swap it out.
The moment it grows a planner, a memory, or a prompt the caller didn't
write, it has crossed into Ring 2 and should be spun out as a separate
project, not shipped as airframe core.

## Sources

- [Why we no longer use LangChain (HN discussion)](https://news.ycombinator.com/item?id=40739982) · [Octomind — Why we no longer use LangChain](https://octomind.dev/blog/why-we-no-longer-use-langchain-for-building-our-ai-agents) — the over-abstraction / hidden-prompt / undebuggable-loop failure mode.
- [The Orchestration Framework Trap (TianPan)](https://tianpan.co/blog/2026-04-19-orchestration-framework-trap-langchain-production) · [Why LLM frameworks are being replaced by Agent SDKs (MindStudio)](https://www.mindstudio.ai/blog/llm-frameworks-replaced-by-agent-sdks) — the shift toward thin SDKs over native function-calling.
- [LiteLLM docs](https://docs.litellm.ai/docs/) — the flatten-to-OpenAI-shape stance and its `extra_body` leak (contrast case; see roadmap §6.12).
- [Vercel AI SDK — tools & `maxSteps`](https://ai-sdk.dev/docs/foundations/tools) · [middleware](https://ai-sdk.dev/docs/ai-sdk-core/middleware) — a bounded tool loop shipped legibly as a helper; `providerOptions` for the vendor-specific remainder.
- [Instructor](https://techsy.io/en/blog/llm-structured-outputs-guide) · [Outlines / XGrammar constrained decoding](https://medium.com/@emrekaratas-ai/structured-output-generation-in-llms-json-schema-and-grammar-based-decoding-6a5c58b698a6) — structured-output polyfills; the token-level ones require decode control a hosted-SDK wrapper doesn't have.
- [Continue LLM abstraction layer (capability detection)](https://deepwiki.com/continuedev/continue/4.1-extension-architecture) · [Two Sigma — A guide to LLM abstractions](https://www.twosigma.com/articles/a-guide-to-large-language-model-abstractions/) — feature-detection over lowest-common-denominator design.
