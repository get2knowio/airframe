# Agent SDK landscape survey — 2026-06

A point-in-time scan of the agent-SDK landscape against airframe's
current adapter lineup and the existing `dev-docs/*-adapter-plan.md`
files. Goal: identify (1) **new vendor SDKs worth adapting** that
aren't already planned, and (2) **feature updates in SDKs we already
wrap** that should be folded in.

> **Provenance caveat.** Findings come from web research in June 2026.
> Architectural facts (wire format, auth shape, whether a programmatic
> Python surface exists) are stable and load-bearing for adapter
> decisions. Specific *product names and version numbers* moving fast
> right now (Qwen3.7-x, Kimi K2.6/K2.7, Antigravity 2.0, GPT-5.x) are
> reported as found and should be re-verified against vendor docs
> before they land in code or pricing tables. Sources at the bottom.

## What we already cover or plan

Shipped (`src/airframe/adapters/`): `claude`, `github-copilot`,
`opencode`, `opencode-zen`, `opencode-go`, `openrouter`, `bedrock`,
`kimi`. Base: `OpenAICompatibleRuntime`. (Note: `CodexRuntime` was
removed in 0.7.0 per `CLAUDE.md`; `"codex"` reserved.)

Planned in dev-docs: `GeminiRuntime` (google-genai),
`MistralRuntime` (Agents API), `KimiRuntime`, `BedrockRuntime`
(shipped), `OpenCodeServerRuntime`. Reserved IDs: `anthropic`,
`openai`, `bedrock-agents`, `moonshot`, `codex`, `mistral-completions`,
`mistral-vertex`, `google`, `vertex`.

The feature roadmap already specs streaming, sessions/resume,
cancel, reasoning effort + output, vision/file inputs, function
tools, MCP refs, permission callbacks, lifecycle hooks, budget caps,
skills, rate-limit telemetry, request metadata, and token counting.
That surface is comprehensive — this survey does **not** re-propose it.

---

## Part 1 — New SDK candidates (gap analysis)

### 1. Qwen / Alibaba (DashScope) — **strongest gap; not in any plan**

This is the clearest omission. Three distinct surfaces exist:

- **DashScope OpenAI-compatible endpoint** (Alibaba Cloud Model
  Studio). Qwen models are reachable through the stock `openai`
  Python SDK with a swapped `base_url` + `DASHSCOPE_API_KEY`. This is
  a **~30 LOC `OpenAICompatibleRuntime` subclass** — the canonical
  `opencode_zen.py` shape. Tool use, streaming, JSON mode, and vision
  all work over the compat surface.
- **Qwen-Agent** — a higher-level Python agent *framework* (above the
  protocol; not an adapter target, same reasoning we use to decline
  ADK/Strands).
- **Qwen Code CLI** — a subprocess coding agent that is a *fork of
  Gemini CLI*, now with its own SDK. This is the Claude-Code-shaped
  surface. Lower priority than the compat path because the model is
  already reachable over OpenAI-compat.

**Recommendation:** ship `QwenRuntime` as an `OpenAICompatibleRuntime`
subclass against DashScope, provider ID `"qwen"`. Reserve
`"dashscope"` (gateway-naming sibling) and `"qwen-code"` (future
subprocess agent surface) the same way Mistral reserves
`mistral-completions`/`mistral-vertex`. Auth note for the plan:
**Qwen's OAuth free tier was discontinued 2026-04-15 — API key is the
only programmatic path**, so the auth chain is just
`api_key=` → `DASHSCOPE_API_KEY` (no on-disk credential helper).

Effort: lowest of any candidate. Strategic value: high — fills the
single largest model-house gap (no Chinese-frontier path today except
indirectly via OpenRouter), and Qwen3-Coder is competitive with
Sonnet-class models on agentic coding benchmarks.

### 2. DeepSeek — **easy compat subclass; not planned**

DeepSeek's API is OpenAI- *and* Anthropic-compatible. Like Qwen, a
thin `OpenAICompatibleRuntime` subclass (provider ID `"deepseek"`)
gets you there. The reasoning-trace plumbing already in the roadmap
(`reasoning_content` on DeepSeek-R1 derivatives is explicitly cited
in the roadmap's `REASONING_OUTPUT` section) means the base already
half-handles DeepSeek. Today DeepSeek is reachable via OpenRouter —
same "unblocked but feature-poor" situation as Gemini/Mistral
fallbacks. A direct adapter is low-effort but lower-priority than
Qwen (DeepSeek has no distinctive agent surface the compat path
misses).

**Recommendation:** signal-gate it. Add `"deepseek"` as a reserved ID
now; build when a consumer asks. Note the model-name churn: legacy
names retire 2026-07-24 in favour of `deepseek-v4-flash` /
`deepseek-v4-pro` (verify).

### 3. Google Antigravity SDK — **may be the Gemini subscription trigger**

The `google-genai` plan explicitly defers a `GeminiCliRuntime`
"until Google ships an official agent SDK (claude-agent-sdk analogue)
to wrap." Google now appears to ship exactly that: an
`antigravity-sdk-python` library exposing "the same tools, agent
loop, and context management that power Google Antigravity,
programmable in Python," able to host agents on your own infra. This
is the subprocess/agent-shaped Gemini surface the plan was waiting
for, and it would unlock the *subscription-auth* path that
`google-genai` (API-key-only) can't reach.

**Recommendation:** keep `GeminiRuntime` (google-genai) as the first
Google adapter per the existing plan, but add a note to
`google-genai-adapter-plan.md` that the deferred `GeminiCliRuntime`
trigger may have fired — evaluate the Antigravity SDK's stability and
auth model. Verify it's a real, documented, versioned package before
acting (Antigravity branding is moving fast).

### 4. "Above the protocol" — note, don't adapt

These are orchestration frameworks, not vendor model SDKs. Wrapping
them violates the "narrow protocol; orchestration is consumer
responsibility" invariant — same rationale already used to decline
Google ADK in the Gemini plan.

- **OpenAI Agents SDK** — big April 2026 update: subagents, "code
  mode," a harness + sandbox layer (Blaxel/Cloudflare/Daytona/E2B/
  Modal/Runloop/Vercel), and — notably — it is now **provider-
  agnostic across 100+ models via Chat Completions**. Strategically
  this is a *competitor to airframe's positioning*, not a wrap
  target. Worth watching: it validates the abstraction thesis while
  sitting one layer up (orchestration over a neutral execute).
- **AWS Strands Agents 1.0** (multi-agent, A2A) — framework; airframe
  already wraps Bedrock Converse at the correct (lower) level.
- **Google ADK 2.0** (graph runtime) — framework; already declined.
- **Microsoft Agent Framework** (AutoGen + Semantic Kernel lineage) —
  framework; same call.

No action beyond a one-line "considered and declined, here's why" in
the roadmap so the question doesn't get re-litigated.

---

## Part 2 — Feature updates in SDKs we already wrap

### GitHub Copilot SDK — preview → **GA**

The roadmap records "public preview since 2026-04-02,
`github-copilot-sdk` 0.3.0." It reportedly reached **GA on
2026-06-02**, across Node/TypeScript, Python, Go, .NET, Rust, and
Java, with a stable, production-supported API and added **multi-client
workflows** (different clients contributing tools/permissions to one
session). **Action:** bump the pinned `github-copilot-sdk` version,
re-check the `SessionEventType` surface for new events, and update the
"public preview" language in `feature-roadmap.md` §2.2 and any
`docs/adapters/copilot.md`. Verify whether GA changed the import name
or auth posture.

### Kimi (Moonshot) — model generation moved

The `kimi` adapter and plan should track current models: **Kimi K2.6**
(reported April 2026 — 256K context across variants, a ~400M-param
MoonViT vision encoder for native image/video, "Agent Swarm" up to
~300 sub-agents) and **Kimi K2.7 Code** (reported 2026-06-12,
long-horizon agentic coding, ~30% fewer thinking tokens). **Action:**
refresh `list_models()` metadata and any `_KIMI_PRICING`/context-window
constants to the K2.6/K2.7 generation; the vision encoder may let the
Kimi adapter flip `Feature.VISION_INPUT` once the SDK surfaces image
input. Re-verify version names against Moonshot docs.

### Claude Agent SDK — incremental

Roadmap already tracks `claude-agent-sdk` 0.2.82 in depth. Newer
items to confirm: a **dedicated monthly Agent SDK credit pool for
subscription plans from 2026-06-15** (billing/auth-doc note, not a
protocol change), and **A2A** surfacing in the SDK reference alongside
subagents/hooks/skills. No new protocol surface implied beyond what
the roadmap specs; just keep the version + auth notes current.

### Codex / OpenAI — roadmap is stale here

`CLAUDE.md` says 0.7.0 removed `CodexRuntime` (the `openai-codex-sdk`
went unmaintained) and reserved `"codex"` for a future wrapper of
OpenAI's official `openai-codex` SDK once it leaves alpha.
`feature-roadmap.md` still treats `CodexRuntime` as a live, wrapped
adapter throughout §1–§3. **Action:** reconcile the roadmap with the
0.7.0 removal — mark the Codex column as "removed; reserved" so the
matrix stops implying a shipped adapter. Separately, OpenAI's *Codex
SDK* (distinct from the *Agents SDK*) continues under Codex-pinned
versioning; re-evaluate the `"codex"` reservation only if/when it
exits alpha.

---

## Part 3 — Prioritized recommendations

1. **Write `qwen-adapter-plan.md` and build `QwenRuntime`** as an
   `OpenAICompatibleRuntime` subclass against DashScope. Lowest
   effort, highest strategic value, fills the biggest gap. Reserve
   `"dashscope"` and `"qwen-code"`.
2. **Fold in the Copilot GA update** — version bump + event-surface
   re-check + roadmap language. Cheap, keeps a shipped adapter honest.
3. **Refresh Kimi model metadata** to the K2.6/K2.7 generation;
   evaluate flipping `VISION_INPUT`.
4. **Reconcile the roadmap's Codex references** with the 0.7.0 removal.
5. **Annotate the Gemini plan** that the Antigravity SDK may be the
   deferred `GeminiCliRuntime` trigger.
6. **Reserve `"deepseek"`**; signal-gate the adapter.
7. **Add a short "frameworks considered and declined" note** (OpenAI
   Agents SDK, Strands, ADK, MS Agent Framework) to the roadmap.

The recurring theme: the frontier moved toward **provider-agnostic
orchestration frameworks** sitting *above* a neutral execute layer
(OpenAI Agents SDK going multi-provider is the headline). That
validates airframe's niche rather than threatening it — airframe is
the JDBC, those frameworks are the ORMs. Stay at the protocol layer;
keep adding model houses (Qwen next) and keep the wrapped-SDK feature
surface current.

## Sources

- Qwen Code (open-source agent / SDK): https://qwen.ai/qwencode , https://github.com/QwenLM/qwen-code
- Qwen3-Coder: https://qwenlm.github.io/blog/qwen3-coder/ , https://github.com/QwenLM/Qwen3-Coder
- DashScope OpenAI-compatible API: https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope , https://www.alibabacloud.com/help/en/model-studio/first-api-call-to-qwen
- DeepSeek API (OpenAI/Anthropic-compatible): https://api-docs.deepseek.com/
- Google ADK: https://google.github.io/adk-docs/ , https://github.com/google/adk-python , https://pypi.org/project/google-adk/
- Gemini CLI SDK / headless: https://deepwiki.com/google-gemini/gemini-cli/5.9-sdk-and-programmatic-api , https://google-gemini.github.io/gemini-cli/docs/cli/headless.html
- Google Antigravity (SDK / managed agents): https://blog.google/innovation-and-ai/technology/developers-tools/managed-agents-gemini-api/ , https://apidog.com/blog/google-antigravity-2/
- OpenAI Agents SDK (subagents/harness/sandbox, multi-provider): https://openai.com/index/the-next-evolution-of-the-agents-sdk/ , https://techcrunch.com/2026/04/15/openai-updates-its-agents-sdk-to-help-enterprises-build-safer-more-capable-agents/
- AWS Strands Agents 1.0: https://aws.amazon.com/blogs/opensource/introducing-strands-agents-an-open-source-ai-agents-sdk/ , https://www.infoq.com/news/2026/03/aws-strands-agents/
- Claude Agent SDK: https://code.claude.com/docs/en/agent-sdk/overview
- GitHub Copilot SDK GA: https://github.blog/changelog/2026-06-02-copilot-sdk-is-now-generally-available/ , https://github.blog/news-insights/company-news/build-an-agent-into-any-app-with-the-github-copilot-sdk/
- Kimi K2.6 / K2.7: https://kimi-k2.org/blog/24-kimi-k2-6-release , https://codersera.com/blog/kimi-k2-7-complete-guide-2026/
