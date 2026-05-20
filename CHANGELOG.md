# Changelog

All notable changes to airframe are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

---

## [0.8.0] — 2026-05-20

Minor release: adds the `OpenCodeServerRuntime` adapter wrapping
`sst/opencode`'s HTTP agent server. New provider ID: `"opencode"`,
distinct from the existing `"opencode-zen"` / `"opencode-go"` (those
two wrap OpenCode's OpenAI-compat *gateway* endpoints; this one
targets the local agent server you start with `opencode serve`).
Additive; no breaking changes.

### Added

- **`OpenCodeServerRuntime`** — new adapter wrapping `sst/opencode`'s
  HTTP agent server (`opencode serve`) via the official
  `opencode-ai` Stainless-generated Python SDK. Distinct from the
  existing `OpenCodeZenRuntime` / `OpenCodeGoRuntime` (those two wrap
  OpenCode's per-token / subscription gateways at
  `https://opencode.ai/zen/*`; this one targets the local agent
  server). Provider ID: `"opencode"`. Lands fully wired across six
  iterations:
  - **A**. Protocol scaffolding (discovery, capability predicates,
    `validate_binding`, lazy SDK import, auth chain with non-loopback
    guardrail, `list_models` via `client.app.providers()`).
  - **B**. SDK-backed `OpenCodeServerSession` — `execute` /
    `stream` / `cancel` / `Session(resume=...)`; SSE bus translation
    with client-side delta computation against
    `message.part.updated` snapshots; SDK exception classification.
  - **C**. Polymorphic prompts (`ImageInput` / `FileInput` →
    OpenCode's `FilePartInputParam`; bytes/path encode as data URLs)
    and reasoning pass-through via `extra_body` (Anthropic upstream
    gets `{"thinking": {"type": "enabled", "budget_tokens": N}}`,
    everyone else gets `{"reasoning_effort": "..."}`).
  - **D**. `OpenCodeServerOptions.available_tools` /
    `excluded_tools` → `session.chat(tools={...: True/False})`
    allow/denylist for OpenCode's built-in tools. `tools=` (caller
    `FunctionTool`s), `mcp_servers=`, `on_permission=` decline
    permanently for this SDK version — opencode-ai 0.1.0a36 has no
    `client.mcp` / `client.permission` resources. The plan's "Path 2
    fallback".
  - **E**. Lifecycle hooks (6 of 8 kinds — all but `pre_compact` and
    `rate_limit`, neither of which the SDK surfaces) +
    `_enforce_budget_pre_turn` for `max_turns` /
    `max_budget_usd`. Cost accumulates from `AssistantMessage.cost`
    when the upstream reports it; `BUDGET_USD_CAP` is best-effort
    against Ollama / llama.cpp deployments that don't.
  - **F**. `OpenCodeServerOptions` final surface (`provider_id`,
    `available_tools`, `excluded_tools`, `working_directory`,
    `permission_mode`, `additional_mcp_servers`,
    `additional_request_fields`) with the
    upstream-provider-routing dispatch. Conformance + integration
    test wrappers, per-adapter docs page, README + auth.md +
    capabilities.md updates.
- **`[opencode]` pip extra** — `airframe-agents[opencode]` installs
  `opencode-ai>=0.1.0a36,<0.2`. Tight pin against a pre-1.0 SDK; bump
  the upper bound deliberately, not via `pip install -U`.
- **`OpenCodeServerOptions` provider-options namespace** — seven
  fields per the surface above. `provider_id` is the most important
  for callers — it explicitly routes a session through a specific
  upstream (Anthropic / OpenAI / OpenRouter / …). Without it the
  adapter auto-discovers from `client.app.providers()`; ambiguous
  matches raise asking for explicit routing.
- **Subscription-routing path**: `opencode auth login openai`
  against a ChatGPT account, then `OpenCodeServerRuntime` →
  OpenCode → Codex chat via the user's subscription. The path the
  removed `CodexRuntime` was originally meant to deliver, now
  served through a maintained SDK.
- **`RuntimeServerStartError` → `pytest.skip`** in
  `airframe.testing.integration`. Adapters that depend on a local
  server (today: OpenCode) skip the suite cleanly when the server
  isn't reachable, mirroring the `RuntimeAuthError` skip pattern.

### Changed

- **`CLAUDE.md` reserved-IDs paragraph** — added `"opencode"` to
  the active provider IDs list and clarified the three-way split
  with `"opencode-zen"` / `"opencode-go"`.
- **`docs/capabilities.md`** — new OpenCode column. Five `SDK gap`
  entries (`STRUCTURED_OUTPUT_JSON_SCHEMA`, `TOOLS_*`,
  `PERMISSION_CALLBACK`) document features OpenCode the *server*
  supports but the opencode-ai 0.1.0a36 SDK hasn't surfaced yet.
- **README provider table** — new `OpenCodeServerRuntime` row
  alphabetised between OpenCode Go and OpenCode Zen.
- **`docs/auth.md`** — new `## OpenCodeServerRuntime` section
  covering the HTTP Basic chain + loopback guardrail.

### SDK-gap declines (will flip True once `opencode-ai` catches up)

`STRUCTURED_OUTPUT_JSON_SCHEMA`, `TOOLS_FUNCTION`,
`TOOLS_MCP_STDIO` / `_HTTP` / `_SSE`, and `PERMISSION_CALLBACK`
stay False on `OpenCodeServerRuntime` because the 0.1.0a36 SDK has
neither a `client.mcp` resource nor a `client.permission`
reply endpoint. OpenCode the *server* supports both surfaces;
wrapping raw HTTP routes the SDK doesn't surface would violate
CLAUDE.md's "wrap vendor SDKs, don't rewrite them" invariant. A
follow-up iteration will flip the flags once Stainless regenerates
the client with those routes.

---

## [0.7.0] — 2026-05-19

Major release: adds the `KimiRuntime` adapter and **removes
`CodexRuntime` and the `[codex]` pip extra**. The Codex removal is a
breaking change for anyone importing `CodexRuntime` or installing
`airframe-agents[codex]`; per pre-1.0 SemVer convention the minor
version is bumped.

### Removed

- **`CodexRuntime` adapter, in full.** Deleted
  `src/airframe/adapters/codex.py` (~1500 LOC), the `CodexOptions`
  dataclass, the `[codex]` pip extra (`openai-codex-sdk>=0.1.11`
  dependency), `examples/probe_codex.py`, `docs/adapters/codex.md`,
  and the four Codex test modules (`test_codex.py`,
  `test_codex_session.py`, `test_codex_conformance.py`,
  `test_codex_integration.py`). All cross-references in src docstrings,
  per-adapter docs, the README, and `CLAUDE.md` are scrubbed.
- **`"codex"` no longer appears in `list_providers()`,
  `runtime_for()`, the `airframe.adapters` entry-point group, or the
  `airframe.testing.integration._PROVIDER_AUTH` map.**

**Why.** Two things converged. (1) The package
`airframe-agents` was wrapping — `openai-codex-sdk` on PyPI — has
murky provenance: it self-declares `author: OpenAI` but it does not
live in `github.com/openai/codex` (OpenAI's official Codex repo),
nor has it kept pace with the Codex CLI's recent releases. (2) The
actually-official Python SDK at `github.com/openai/codex/sdk/python`
is published as `openai-codex` (currently `0.131.0a4` — still alpha)
and has a different architecture entirely (JSON-RPC app-server v2
over stdio); it is not a drop-in replacement, and notably neither
Python package exposes `client.models.list()` (only the Node
`@openai/codex` SDK does). That made the post-v0.6.3 false-negative
in Maverick's `doctor` command — `list_models()` rejecting ChatGPT
OAuth tokens against `api.openai.com/v1/models` — fundamentally
unfixable from inside `CodexRuntime.list_models()` without
hand-rolling more code on top of a package whose maintainership we
can't verify. Wrapping uncertain-provenance code isn't earning its
weight pre-1.0, so the cleaner move is to remove the adapter and
revisit once the official `openai-codex` package leaves alpha.

**Migration.** Direct users of `CodexRuntime` should either pin
`airframe-agents<0.7` or migrate to a still-supported adapter:
`ClaudeCodeRuntime`, `CopilotRuntime`, `BedrockRuntime`, or an
OpenAI-compatible gateway adapter
(`OpenCodeZenRuntime` / `OpenCodeGoRuntime` /
`OpenRouterRuntime`). Codex subscription holders can route through
opencode adapters, which already accept the Codex provider on the
opencode side.

### Reserved

- **`"codex"` stays reserved** for a possible future adapter wrapping
  the official `openai-codex` Python SDK once it leaves alpha and
  surfaces a stable enough JSON-RPC surface to wrap. The provider ID
  is not re-usable for any other vendor.

### Added

- **`KimiRuntime`** — new adapter wrapping Moonshot AI's
  `kimi-agent-sdk`, the official Python surface over the `kimi-cli`
  subprocess. Architecturally the closest sibling to
  `ClaudeCodeRuntime`: subprocess-class agent SDK with native
  streaming, per-call `ApprovalRequest` dispatch, MCP server
  registration, and session resume by id. Lands fully wired across
  six iterations:
  - **A**. Protocol scaffolding (discovery, capability predicates,
    `validate_binding`, lazy SDK import, fallback model catalogue).
  - **B**. SDK-backed `KimiSession` — `execute` / `stream` /
    `cancel` / `Session.resume`; `WireMessage` → airframe event
    translation; SDK exception classification.
  - **C**. Polymorphic prompts (`ImageInput` URL / bytes / path →
    `ImageURLPart`) and reasoning (`thinking=` → SDK's boolean
    `thinking` kwarg, session rebuild on toggle).
  - **D**. `PermissionCallback` → `ApprovalRequest.resolve`
    dispatch (`allow → approve`, `deny → reject`, `defer → reject`
    with feedback explaining the SDK's synchronous channel).
    `McpServerRef` → fastmcp `MCPConfig` dict translation for
    stdio / http / sse transports. `tools=` declined permanently
    pointing at `mcp_servers=` (no Python-callable channel in the
    SDK).
  - **E**. Lifecycle hooks (7 of 8 kinds — all but `rate_limit`,
    which Moonshot raises as exceptions rather than wire events),
    pre-turn budget enforcement via the shared
    `_enforce_budget_pre_turn` helper, in-tree `_KIMI_PRICING`
    table populating `CostRecord.cost_usd`.
  - **F**. `KimiOptions` final surface (`working_directory`,
    `yolo`, `additional_mcp_servers`, `skill_directories`,
    `additional_config_fields`) with the `yolo` ↔ `on_permission`
    mutual-exclusion gate. Integration test wrapper at
    `tests/test_kimi_integration.py`. Per-adapter docs page,
    `## KimiRuntime` section in `docs/auth.md`, Kimi column in
    `docs/capabilities.md`, README row.
- **`[kimi]` pip extra** — `airframe-agents[kimi]` installs
  `kimi-agent-sdk>=0.0.5,<0.1`. **Co-installation conflict with
  `[claude]`** — `kimi-agent-sdk` 0.0.5 → `kimi-cli` 1.12 →
  `fastmcp` 2.12.5 → `mcp<1.17`, but `claude-agent-sdk` 0.2
  requires `mcp>=1.23`. Until upstream resolves the two cannot
  co-install. Declared in `[tool.uv.conflicts]`; `[all]` excludes
  `[kimi]` for the same reason; users wanting both must split into
  separate venvs.
- **`KimiOptions` provider-options namespace** — five fields per
  the surface above. Mutually-exclusive `yolo` + `on_permission`
  gate at `runtime.session()` time.
- **`_PROVIDER_AUTH["kimi"]`** in `airframe.testing.integration`
  for the integration suite's env-var check.

### Changed

- **README provider table** — new Kimi row alphabetised between
  Copilot and OpenCode Go; the Codex row is gone.
- **README capability matrix** — new Kimi column; Codex column
  dropped.
- **`docs/capabilities.md`** — new Kimi column; Codex column
  dropped.
- **`docs/architecture.md`** — adapter row redrawn for the five
  remaining families (Claude / Copilot / Kimi / Bedrock / OpenAI-
  compatible); per-SDK section gains a "### Kimi Agent SDK" entry
  in place of "### Codex SDK".

### Moonshot reservation

- `"moonshot"` remains reserved as a future OpenAI-compat sibling
  that would front Moonshot's `api.moonshot.ai/v1` chat-completions
  surface — distinct from `KimiRuntime`, which wraps the Kimi Agent
  SDK subprocess.

---

## [0.6.3] — 2026-05-18

Bug fix release: `CodexRuntime` no longer reads opencode's
credentials file, and now correctly reads the Codex CLI's own
`~/.codex/auth.json`.

### Fixed

- **`CodexRuntime._resolve_api_key()` opencode credential leak.** The
  on-disk auth fallback was reading
  `~/.local/share/opencode/auth.json::openai.key` — opencode's
  credentials file, not Codex's. That meant `CodexRuntime` would
  silently surface an opencode-minted key for a user who had never
  run `codex login`, and it ignored the perfectly good key (or
  ChatGPT-OAuth bundle) that Codex itself writes to
  `~/.codex/auth.json`. The fix replaces the opencode read with a
  Codex-only resolver that handles both shapes Codex writes:
  - **Static key** (`{"OPENAI_API_KEY": "sk-..."}` from
    `codex login --api-key=…`) → lifted into `Codex({"apiKey": ...})`.
  - **ChatGPT OAuth bundle** (`{"OPENAI_API_KEY": null, "auth_mode":
    "...", "tokens": {"access_token", "refresh_token", "id_token",
    "account_id"}}` from `codex login` against a ChatGPT
    Plus/Pro subscription) → deliberately *not* lifted. The Codex CLI
    subprocess reads the file itself and refreshes those tokens;
    airframe stays out of that path.
- **`CodexRuntime.list_models()` OAuth-tailored error.** ChatGPT
  OAuth access tokens aren't valid against `api.openai.com/v1/models`.
  When `list_models()` detects the OAuth bundle but no static key
  surfaces, it now raises a tailored `RuntimeAuthError` pointing the
  user at `codex login --api-key=…` or `OPENAI_API_KEY=` rather than
  surfacing a generic 401 from the OpenAI SDK.

### Changed

- **Env-var override renamed**: `OPENCODE_AUTH_PATH` →
  `CODEX_AUTH_PATH` on the Codex adapter. The opencode env override
  never made sense on `CodexRuntime`; keeping it would perpetuate the
  leak. `OPENCODE_AUTH_PATH` remains in force on `OpenCodeZenRuntime`
  and `OpenCodeGoRuntime`, where it correctly points at opencode's
  own credentials file.
- **Cross-adapter audit.** Confirmed no other adapter reads opencode
  credentials. `claude_code.py`, `copilot.py`, `bedrock.py`,
  `kimi.py`, and `openrouter.py` are all clean. `opencode_zen.py` and
  `opencode_go.py` legitimately read opencode's auth.json — those
  *are* the opencode adapters.

### Tests

- New regression guard `test_resolve_api_key_ignores_opencode_auth_json`
  asserts that even pointing `CODEX_AUTH_PATH` at an opencode-shaped
  file (`{"openai": {"key": "..."}}`) returns `None` — the wrong
  schema is silently rejected.
- New `test_resolve_api_key_returns_none_for_oauth_only_auth_json`
  asserts the OAuth bundle is **not** lifted into `apiKey`.
- New `test_codex_list_models_oauth_only_auth_json_raises_tailored_error`
  asserts the tailored ChatGPT-OAuth error from `list_models()`.

---

## [0.6.2] — 2026-05-18

Bug fix release: `ClaudeCodeRuntime.list_models()` now works under
Claude Max subscription OAuth auth, not just under `ANTHROPIC_API_KEY`.

### Fixed

- **`ClaudeCodeRuntime.list_models()` under OAuth.** The previous
  implementation hand-rolled a `GET /v1/models` via `httpx` and only
  sent the `x-api-key` header — Claude Max subscription users
  (whose auth is an OAuth bearer token) saw `RuntimeAuthError`. Root
  cause: `/v1/models` *does* accept Bearer tokens, but only when the
  request also carries `anthropic-beta: oauth-2025-04-20`. Airframe's
  hand-rolled call didn't include the beta header. The fix delegates
  to `anthropic.AsyncAnthropic.models.list()`, which picks the right
  header set automatically based on whether `api_key=` or
  `auth_token=` was passed. Auth-slot resolution follows airframe's
  documented four-step chain (explicit `api_key=` → `CLAUDE_CODE_OAUTH_TOKEN`
  env → `ANTHROPIC_API_KEY` env → `~/.claude/.credentials.json`).

### Added

- **`anthropic` SDK dependency** on the `[claude]` extra
  (`anthropic>=0.40,<1`). Also added to `[all]` and the test
  dependency group.
- **`_read_claude_credentials_oauth_token()` helper** that parses
  `~/.claude/.credentials.json` defensively — every malformed-input
  case returns `None` so the caller falls through to a clean
  `RuntimeAuthError`. Path overridable via the
  `CLAUDE_CREDENTIALS_PATH` env var (testing hook).

### Changed

- **`CLAUDE.md` gains a new key invariant: "Wrap vendor SDKs; don't
  rewrite them."** Codifies the principle that surfaced this bug —
  before hand-rolling HTTP, headers, retry policy, error parsing, or
  auth handling against a vendor endpoint, check whether the official
  SDK already exposes that surface and use it if it does. After the
  audit prompted by this fix, the codebase has no remaining raw
  `httpx` callers in adapters, no hand-rolled subprocesses, no
  retry/backoff loops duplicating SDK behavior.
- **`tests/test_list_models.py` Claude section rewritten.** Mocks
  `anthropic.AsyncAnthropic` instead of `httpx`. Five new tests cover
  the four auth-resolution paths (explicit api_key, OAuth env,
  ANTHROPIC_API_KEY-defer-to-SDK, credentials-file fallback), the
  no-credentials error message, and the SDK's `APIStatusError` /
  `APIConnectionError` classification.

### Notes

- No protocol shape changes; no other adapter touched. Wheel/sdist
  surface identical to v0.6.1 except for the version string,
  CHANGELOG entry, CLAUDE.md update, and the
  `src/airframe/adapters/claude_code.py` / `tests/test_list_models.py`
  contents.

---

## [0.6.1] — 2026-05-18

README-only patch release. The v0.6.0 README had two gaps surfaced by
the PyPI-rendered project page:

- **Bedrock missing from end-user-facing surfaces** despite shipping
  as a first-class adapter. Added to the tagline, the `Why?` per-SDK
  description, the `## Install` extras list (between alphabetised
  sibling lines), and the `## Documentation` per-adapter-notes link
  list.
- **Relative repo links 404 on PyPI.** Every link to `docs/*.md`,
  `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`, and `LICENSE`
  resolved against `pypi.org` instead of the GitHub repo. All
  converted to absolute `https://github.com/get2knowio/airframe/blob/main/...`
  URLs so the PyPI-rendered README links work end-to-end. (GitHub
  resolves the absolute URLs identically; no regression for users
  reading the README on github.com.)

No code changes; no behaviour changes; the published wheel/sdist
contents are identical to v0.6.0 except for the README and version
strings.

---

## [0.6.0] — 2026-05-18

A pre-release that captures the **OpenRouter** and **AWS Bedrock**
adapters landing on top of the v0.5.0 Phase-1/2/3/4/5 surface, plus
the v1.0-readiness docs sprint. Snapped before work on a sixth
adapter (`OpenCodeServerRuntime` — the bespoke OpenCode agent server)
begins, so consumers can pin to a known-good five-adapter baseline.

Headline additions:

- **`BedrockRuntime`** — AWS Bedrock Converse API adapter. Closes the
  enterprise / managed-cloud bucket. Twelve `Feature` flags True;
  fronts Anthropic Claude, Meta Llama, Mistral, Cohere, Amazon Nova,
  AI21 Jamba behind one AWS-billed envelope with IAM-rooted auth and
  region pinning.
- **`OpenRouterRuntime`** — OpenAI-compatible adapter for the
  OpenRouter gateway (300+ models behind one credit pool).
- **`OpenCodeGoRuntime`** — OpenAI-compatible adapter for the
  opencode-go subscription endpoint at `https://opencode.ai/zen/go/v1`.
- **v1.0-readiness docs sprint** — `SECURITY.md`,
  `docs/auth.md` / `docs/capabilities.md` / `docs/reference.md` /
  `docs/cookbook.md`, per-adapter pages under `docs/adapters/`, a
  third-party-adapter guide, PyPI badges + project URLs.
- **v1.0-readiness pass** — `airframe.testing.integration`
  pytest-marker-gated behavioural suite, twelve new structural
  contracts in `airframe.testing.contracts`, populated
  `ClaudeOptions` / `CopilotOptions` / `CodexOptions` /
  `OpenAICompatOptions` namespaces.
- **GitHub Actions release pipeline** — `.github/workflows/release.yml`
  triggers on `v*` tag pushes: matrix-tested across Python
  3.11/3.12/3.13, creates a GitHub Release with auto-generated notes,
  builds via `uv build`, and publishes to PyPI via OIDC Trusted
  Publishing (`pypi` environment, no long-lived tokens). Mirrors the
  pattern used by `get2knowio/remo` and `get2knowio/climax`.

Forward-looking dev-docs (not in the published sdist):

- **`dev-docs/opencode-adapter-plan.md`** — full implementation plan
  for the next adapter: `OpenCodeServerRuntime` wrapping the bespoke
  `sst/opencode` agent HTTP server. Phase 1 candidate; distinct from
  the existing `OpenCodeZenRuntime` / `OpenCodeGoRuntime` gateway
  adapters that share the brand.
- **`dev-docs/feature-roadmap.md`** — new "Adapter expansion
  candidates" subsection indexing the three in-flight adapter plans
  (OpenCode agent server, Bedrock, Gemini).

Iteration-level detail for Bedrock and the v1.0-readiness work
follows. `OpenRouterRuntime` (commit `52b5240`) and `OpenCodeGoRuntime`
(commit `7b6ff98`) land as thin `OpenAICompatibleRuntime` subclasses
in the canonical ~30-LOC shape — see their module docstrings for
auth chains and the `git log v0.5.0..v0.6.0` range for the full
diff. Listed in `README.md`'s providers table and
`docs/capabilities.md`.

### BedrockRuntime — AWS Bedrock Converse API adapter

New built-in adapter wrapping AWS Bedrock's
[Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html)
— the vendor-normalised model-invocation endpoint that fronts
Anthropic Claude, Meta Llama, Mistral, Cohere, Amazon Nova, and
AI21 Jamba behind one AWS-billed envelope with IAM-rooted auth and
region pinning. Closes the fourth and last adapter bucket — the
enterprise / managed-cloud path that the existing
subscription-style and per-token-gateway adapters can't reach (they
depend on creds / endpoints that aren't reachable from regulated /
VPC-only environments).

#### Added

- **`BedrockRuntime`** at `airframe.adapters.bedrock` —
  `PROVIDER_ID="bedrock"`, `REQUIRES_PACKAGE="aioboto3"`,
  `EXTRA_NAME="bedrock"`. Lazy-imports `aioboto3` so plain
  `import airframe` doesn't pull boto3 in.
- **`BedrockSession`** — bespoke session with a client-side
  `messages=[]` buffer, multi-turn rollback-on-failure discipline,
  cooperative cancellation, and a client-side tool loop capped at
  `MAX_TOOL_ITERATIONS=20` matching the OpenAI-compat bound.
- **`BedrockOptions`** — per-session `region_name` override
  (opens a session-private client), `inference_profile_arn`
  (replaces the call-time `modelId`), `guardrail_id` /
  `guardrail_version`, `performance_latency`, and
  `additional_model_fields` (pass-through to Converse's
  `additionalModelRequestFields`).
- **Twelve `Feature` flags True**: `STRUCTURED_OUTPUT_JSON_SCHEMA`,
  `STREAMING`, `CANCEL`, `REASONING_EFFORT`,
  `REASONING_BUDGET_TOKENS` (both Anthropic-on-Bedrock only),
  `VISION_INPUT`, `FILE_INPUT`, `TOOLS_FUNCTION`,
  `PERMISSION_CALLBACK`, `LIFECYCLE_HOOKS`, `BUDGET_USD_CAP`,
  `BUDGET_TURN_CAP`.
- **`EMITTABLE_HOOK_KINDS`** — six of airframe's eight canonical
  kinds. `pre_compact` (Converse has no compaction) and
  `rate_limit` (botocore's transient-retry chain swallows throttle
  events) stay unemittable.
- **In-tree `_BEDROCK_PRICING`** — point-in-time `us-east-1`
  per-1k-token rates for the curated catalog (Claude 3.5
  Haiku/Sonnet/Opus, Nova Micro/Lite/Pro, Llama 3.1
  8B/70B/405B Instruct, Mistral Large 2, Cohere Command R+).
  Unknown model IDs report `cost_usd=None` rather than guess.
- **Structured output via forced `submit_result` tool** in
  Converse's `toolConfig` slot. User `FunctionTool` entries coexist
  with the forced tool — both ride `toolConfig.tools`.
- **Guardrail intervention handling** — `stopReason ==
  "guardrail_intervened"` surfaces as `RuntimeProtocolError` with a
  clear message rather than failing structured-output validation on
  a truncated payload.
- **Redacted reasoning safety** — `reasoningContent` chunks with
  only a `redactedContent` field (Anthropic-on-Bedrock safety
  redaction) are skipped without crashing.
- **`examples/probe_bedrock.py`** — minimal live probe; skips
  cleanly when `AWS_REGION` is unset.
- **`docs/adapters/bedrock.md`** — full adapter page covering
  install, auth, model catalog, supported features, `BedrockOptions`
  reference, vendor quirks, and the inference-profile / PT-ARN
  gotcha.
- **`docs/auth.md`** gains a `BedrockRuntime` section covering the
  boto3 four-step chain and region resolution.
- **`docs/capabilities.md`** matrix gains a Bedrock column.
- **`docs/reference.md`** adapter table and top-level-exports
  snippet include Bedrock.
- **`tests/test_bedrock_conformance.py`** — wires the structural
  contract suite at `airframe.testing.contracts` against
  `BedrockRuntime`.
- **`tests/test_bedrock_integration.py`** — pytest-marker-gated
  behavioural tests against live AWS, mirroring
  `test_opencode_zen_integration.py`.
- **`airframe.testing.integration._PROVIDER_AUTH`** entry for
  `bedrock` so the integration suite self-skips when no AWS
  credentials are detectable.
- **`airframe.testing.contracts._check_provider_options` matching**
  dict now covers `bedrock → BedrockOptions` so the cross-namespace
  rejection contract holds for the new adapter.

#### Permanent declines (pinned with their own tests)

- `TOOLS_MCP_STDIO` / `TOOLS_MCP_HTTP` / `TOOLS_MCP_SSE` —
  Bedrock Converse has no MCP slot. The decline message points
  callers at `runtime.unwrap(BedrockRuntimeClient)` for
  hand-crafted MCP shims.
- `SESSION_RESUME` — Converse is stateless from the client side;
  the messages buffer doesn't survive process restart.
- `STRUCTURED_OUTPUT_STRICT` — Converse has no equivalent of
  OpenAI's strict tool-schema mode.

#### Changed

- `pyproject.toml` gains a `[bedrock]` extra
  (`aioboto3>=13`); also added to the `all = [...]` list and the
  `dependency-groups.test` block so the unit suite can mock at the
  `aioboto3` boundary.
- `airframe.__init__` re-exports `BedrockRuntime` + `BedrockOptions`;
  `airframe.discovery._builtin_runtime_classes` registers the new
  adapter so `list_providers()` includes `"bedrock"` when the
  extra is installed.
- `airframe.options.ProviderOptions` is now a five-way union
  (`ClaudeOptions | CopilotOptions | CodexOptions |
  OpenAICompatOptions | BedrockOptions`).
- `README.md` + `CLAUDE.md` provider lists include `bedrock`;
  README capability matrix gains a Bedrock column.

#### Iteration history

Landed as six independently-mergeable commits per
[`dev-docs/bedrock-adapter-plan.md`](dev-docs/bedrock-adapter-plan.md):

- **A.** Scaffolding — discovery, capability predicates, AWS
  credential + region resolution chains, `list_models()` via
  `bedrock.list_foundation_models(byOutputModality="TEXT")`,
  boundary error classification.
- **B.** `BedrockSession`, `execute()` / `stream()` / `cancel()`,
  forced-`submit_result`-tool structured output;
  `STRUCTURED_OUTPUT_JSON_SCHEMA` / `STREAMING` / `CANCEL` flipped
  True.
- **C.** Polymorphic prompts (image / file content blocks) and
  Anthropic-on-Bedrock thinking budget via
  `additionalModelRequestFields`; `VISION_INPUT` / `FILE_INPUT` /
  `REASONING_EFFORT` / `REASONING_BUDGET_TOKENS` flipped True.
- **D.** Function tools + client-side tool loop with permission
  gating; `TOOLS_FUNCTION` / `PERMISSION_CALLBACK` flipped True.
- **E.** Lifecycle hook emission (six kinds), per-session budget
  caps, in-tree pricing table; `LIFECYCLE_HOOKS` /
  `BUDGET_USD_CAP` / `BUDGET_TURN_CAP` flipped True.
- **F.** `BedrockOptions` wrap-up, conformance + integration test
  wrappers, docs, README / CHANGELOG / capability-matrix updates.

### v1.0-readiness docs sprint

Documentation-focused follow-up to the v1.0-readiness work below.
No code changes; the goal is parity with the PyPI documentation
patterns of comparable multi-vendor libraries (`litellm`,
`pydantic-ai`, `instructor`).

#### Added

- **`SECURITY.md`** at repo root — disclosure workflow (GitHub
  security advisory preferred, email fallback), in-scope /
  out-of-scope, supported-versions table, credential-handling
  guarantees.
- **`docs/auth.md`** — single page summarising all four adapters'
  credential resolution chains side by side, plus CI patterns and
  the "credentials never leave airframe" guarantees.
- **`docs/capabilities.md`** — per-`Feature` semantics across
  adapters; the README matrix as the executive summary, per-feature
  deep dives here.
- **`docs/reference.md`** — hand-curated API reference covering
  every top-level export with cross-links into the source. Modelled
  on `openai-python`'s `api.md`.
- **`docs/cookbook.md`** — table of `examples/probe_*.py` scripts
  grouped by phase, with one-sentence descriptions and the shared
  probe-script shape consumers can copy.
- **`docs/adapters/claude.md`**, **`docs/adapters/copilot.md`**,
  **`docs/adapters/codex.md`**, **`docs/adapters/opencode-zen.md`**
  — per-adapter pages: install extra, auth chain link, supported
  features table, `<Vendor>Options` field reference, model IDs,
  structured-output mechanism, vendor quirks & landmines, native
  escape hatches.
- **`docs/adapters/third-party.md`** — how to write a custom adapter
  against the `airframe.adapters` entry-point group. Covers both
  the `OpenAICompatibleRuntime` subclass shape (~30 lines) and the
  full bespoke `AgentRuntime` shape; conformance + integration
  contract usage; a complete checklist.
- **PyPI badges** on the README (PyPI version, Python versions,
  license, CI status).
- **PyPI project URLs** in `pyproject.toml` — `Documentation`,
  `Architecture`, `Security` populate the PyPI sidebar.

#### Changed

- **README rewritten** — 572 → ~300 lines. New order: badges →
  tagline → Quickstart → Supported providers → Capability matrix →
  Why? → Install → Sessions/streaming → Errors → Escape hatch →
  Live probes → Documentation (link tree) → Development. The
  protocol Python block moved to `docs/reference.md`; the
  streaming + tools + MCP mega-example moved to `docs/cookbook.md`
  / per-adapter pages; the errors deep-dive moved to
  `docs/reference.md#errors`.
- **`docs/architecture.md`** — "Where to look next" section updated
  to point at the new docs (auth, capabilities, reference,
  per-adapter pages, third-party).
- **Dev-internal docs moved out of `docs/`** —
  `docs/implementation-plan.md` → `dev-docs/implementation-plan.md`;
  `docs/feature-roadmap.md` → `dev-docs/feature-roadmap.md`. Both
  are dev-internal; the published `docs/` tree is now end-user-only.
- **sdist exclusions** in `pyproject.toml` — `dev-docs/`,
  `.devcontainer/`, `.github/`, `CLAUDE.md`, `memory/` excluded
  from the published source distribution.
- **In-repo path updates** for the moved dev-docs — `CLAUDE.md`,
  module docstrings under `src/airframe/`, `docs/architecture.md`,
  `examples/probe_supports.py`, and `tests/test_features.py` all
  point at `dev-docs/implementation-plan.md` / `feature-roadmap.md`
  where the references survived the move.

#### Notes

- Tests unchanged; the 683 unit + 33 integration-skip count carries
  through.
- The CHANGELOG remains a single file; per the audit recommendation,
  archiving pre-1.0 entries to a `CHANGELOG.archive.md` waits for
  the 1.0 release itself.
- No `MIGRATION.md` or `CODE_OF_CONDUCT.md` shipped — both can wait
  until they're load-bearing (a breaking change for MIGRATION; a
  third external contributor for CODE_OF_CONDUCT).

---

### v1.0-readiness pass (docs + TCK + ProviderOptions + integration suite)

A documentation-honesty + test-breadth sweep before cutting v1.0.
No protocol shape changes — every public surface touched here was
either undocumented, mis-documented, or under-tested.

#### Added

- **`airframe.testing.integration`** — pytest-marker-gated
  behavioural conformance suite that mirrors `airframe.testing.contracts`
  but exercises live vendor endpoints. Nine tests cover schema
  round-trip, plain-text execute, `list_models()`, streaming,
  thinking, function-tool round-trip, permission callback firing,
  hook observation, and budget-cap enforcement. Each test
  `pytest.skip`s itself when credentials are absent so the suite
  stays usable on partially-configured machines. Run with
  `pytest -m integration`. Four in-tree wrapper modules
  (`tests/test_<adapter>_integration.py`) provide the per-adapter
  fixture; third-party authors follow the same import-into-suite
  pattern as for the structural contracts.
- **Twelve new structural contracts** in `airframe.testing.contracts`
  covering Phase 2–5: thinking signature, polymorphic-prompt
  decline path, `tools=` ↔ `TOOLS_FUNCTION`, `mcp_servers=` ↔
  per-transport `TOOLS_MCP_*`, `on_permission=` ↔
  `PERMISSION_CALLBACK`, `on_event=` ↔ `LIFECYCLE_HOOKS`,
  `EMITTABLE_HOOK_KINDS` is a subset of the eight canonical
  literals, `max_turns=` ↔ `BUDGET_TURN_CAP`,
  `max_budget_usd=` ↔ `BUDGET_USD_CAP`, cross-namespace
  `ProviderOptions` rejection. All four in-tree conformance test
  modules import the new contracts; the suite count went from
  ~16 to ~27 per adapter.
- **`ClaudeOptions` populated with three fields** —
  `append_system_prompt`, `fork_session`, `strict_mcp_config`
  thread through into `ClaudeAgentOptions` at connect time with the
  namespace fingerprint joining the cache key.
- **`CopilotOptions` populated with four fields** —
  `available_tools`, `excluded_tools`, `skill_directories`,
  `working_directory` thread through into `CopilotClient.create_session`
  at session-creation time.
- **`CodexOptions` populated with four fields** —
  `working_directory`, `additional_directories`,
  `network_access_enabled`, `web_search_enabled` thread through into
  `ThreadOptions` at `Codex.start_thread` / `resume_thread` time.
- **`OpenAICompatOptions` populated with six fields** —
  `prompt_cache_key`, `prompt_cache_retention`, `service_tier`,
  `safety_identifier`, `verbosity`, `store` merge into every
  `chat.completions.create()` call via a new
  `_apply_provider_options()` helper.
- **Shared `_check_provider_options` helper** in `airframe.sessions`
  — every adapter calls it at the top of `session()` to enforce
  the tagged-union contract (passing `CopilotOptions` to
  `ClaudeCodeRuntime` raises `UnsupportedFeatureError`).
- **22 new per-adapter unit tests** across the four
  `test_*_session.py` files: each field lands on the corresponding
  SDK kwarg, default omits the kwarg, value change forces cache
  rebuild (Claude / Copilot / Codex — fields bake at session-build
  time), wrong-namespace raises.

#### Changed

- **`AgentRuntime.session(provider_options=)`** retyped from
  `Any | None` to `ProviderOptions | None` on the protocol and on
  every adapter's signature. Static type checkers now catch
  cross-vendor mistakes that previously slipped through.
- **README rewritten for Phase 1–5 surface.** The "The protocol"
  section now shows all eight methods (the original five plus
  `session`, `supports`, `unwrap`) and the new `AgentSession`
  protocol. Added: a new "Sessions, streaming, and the new kwargs"
  section showing tools / mcp / permission / hooks / budget /
  provider_options together; current capability matrix replacing
  the v0.5-era "still False" placeholder text; correct
  `examples/probe_*.py` paths (the old README pointed at
  `tests/probe_*.py` which moved in Phase 1).
- **`docs/architecture.md` capability matrix refreshed** —
  Phase 2/3/4/5 column placeholders replaced with the actual
  post-Phase-5 ✓/✗ matrix; the "Streaming" section's Phase 3 caveat
  updated; the "Where to look next" probe inventory expanded to
  include the Phase 2–5 probes that landed.
- **Stale docstrings rewritten.** `claude_code.py` module header
  no longer describes the pre-Iteration-G mental model (where the
  runtime owned the SDK client); `sessions.py` module preamble no
  longer claims every adapter uses `_ThinAgentSession`. Both files
  now describe their post-Iteration-G role honestly.
- **`tests/test_options.py::test_options_are_empty_in_phase_0`
  replaced with `test_options_field_inventory`** — pins the
  populated field set per namespace so future removals/renames are
  caught at PR time.

#### Notes

- `airframe.testing.integration` was promised in the v0.3.0
  CHANGELOG "for v0.4.0" — three releases later it lands. The
  earlier deferral is now stale and is closed by this change.
- `_ThinAgentSession` and `_open_thin_session` remain in
  `airframe.sessions` as a reference implementation for third-party
  adapter authors; the docstring now accurately states "none of the
  built-in adapters use this any more".

---

**Phase 5 of the [implementation plan](docs/implementation-plan.md)
is complete — permission, hooks, budget.** Iteration A landed the
protocol-surface shape locks (:class:`PermissionRequest`,
:class:`PermissionCallback`, :data:`PermissionDecision`,
:class:`HookEvent` with the eight ``kind`` literals, the four new
kwargs on :meth:`session` / :meth:`execute` / :meth:`stream` gated
by shared capability helpers). Iteration B wires the permission
callback: Claude / Copilot / Codex accept it natively, OpenAI-compat
declines permanently. Iteration C wires lifecycle hooks across all
four adapters — each declares its emittable-kinds subset via the
new :attr:`EMITTABLE_HOOK_KINDS` ClassVar; observers receive
:class:`~airframe.hooks.HookEvent` instances in causal order with
synthetic ``session_end`` at :meth:`close` if the vendor SDK never
fired one. **Iteration D wires the budget caps** — ``max_turns=`` /
``max_budget_usd=`` honoured at turn boundary across all four
adapters, with the new :class:`RuntimeBudgetExceededError` carrying
``cap`` / ``current`` / ``kind`` attributes for retry-with-larger-cap
or fail-closed branching.

### Added (Phase 5, Iteration D — budget caps + wrap-up)

- **New error class:** :class:`~airframe.errors.RuntimeBudgetExceededError`
  (`AgentRuntimeError` subclass) with ``cap: float``,
  ``current: float``, and ``kind: Literal["usd", "turns"]``
  attributes. Raised at the turn boundary when ``max_turns=`` or
  ``max_budget_usd=`` would be exceeded by the about-to-fire turn;
  mid-turn interrupts are additive later via
  :meth:`AgentSession.cancel`. Exported from top-level :mod:`airframe`.
- **Shared helper:** :func:`airframe.sessions._enforce_budget_pre_turn`
  — every wiring adapter calls it at the top of
  :meth:`AgentSession.execute` / :meth:`AgentSession.stream`. Checks
  ``max_turns`` first (raises with ``kind="turns"``), then
  ``max_budget_usd`` (raises with ``kind="usd"``). The session
  tracks ``_turn_count`` and ``_cumulative_cost_usd`` (sum of
  ``RuntimeResult.cost.cost_usd`` across all turns); both reset at
  :meth:`close`.
- **Claude:** ``max_turns=`` overrides the runtime-default
  ``DEFAULT_MAX_TURNS=60`` by riding into
  :attr:`ClaudeAgentOptions.max_turns` at connect time. The cache
  key carries the value so a turn-cap change forces reconnect (same
  pattern as ``schema=`` / ``thinking=``). ``max_budget_usd`` uses
  the shared client-side accumulator.
- **Copilot:** ``max_budget_usd`` uses the shared client-side
  accumulator. ``max_turns=`` is **declined permanently** —
  Copilot's SDK caps internal turns at the CLI level via the
  runtime's ``--max-turns`` config, so a user-facing
  ``max_turns=`` on per-execute would mislead consumers.
  ``Feature.BUDGET_TURN_CAP`` stays False on Copilot.
- **Codex / OpenAI-compat:** both flip
  :data:`Feature.BUDGET_USD_CAP` and :data:`Feature.BUDGET_TURN_CAP`
  True; both caps enforced via the shared helper. OpenAI-compat's
  ``max_turns`` is distinct from the internal
  ``MAX_TOOL_ITERATIONS=20`` runaway guard (the latter is a fail-safe
  for misbehaving tool loops; ``max_turns=`` is a caller-facing
  per-session budget).
- :data:`Feature.BUDGET_USD_CAP` flipped True on all four adapters.
  :data:`Feature.BUDGET_TURN_CAP` flipped True on Claude / Codex /
  OpenAI-compat; stays False on Copilot.
- ``examples/probe_budget.py`` — multi-provider live probe with a
  deliberately tiny cap; verifies the error fires with the right
  ``kind`` / ``cap`` / ``current`` attributes and prints the
  per-adapter capability matrix at the end. Copilot's branch
  demonstrates the ``max_turns=`` decline.
- 25 new unit tests across the four adapter session test files
  covering: gate-supported declaration, turn-cap firing at the
  cumulative count, USD-cap firing at the cumulative cost,
  stream-path uses the same enforce, no-caps default opens cleanly,
  ``RuntimeBudgetExceededError`` attribute correctness, and
  ``max_turns=`` decline on Copilot. ``tests/test_features.py``
  added ``test_phase_5_final_matrix`` pinning the endgame coverage
  table (the Iteration D deliverable) plus
  ``test_budget_usd_cap_universal`` and
  ``test_budget_turn_cap_universal_except_copilot``.

### Changed (Phase 5, Iteration D)

- ``tests/test_features.py``: replaced
  ``test_phase_5_budget_stays_false_iteration_c`` with the
  Iteration-D matrix tests above; replaced
  ``test_execute_max_turns_kwarg_raises_on_every_adapter`` with
  ``test_execute_max_turns_kwarg_raises_only_on_copilot``; replaced
  ``test_execute_max_budget_usd_kwarg_raises_on_every_adapter``
  with ``test_execute_max_budget_usd_kwarg_no_longer_raises``.
  ``test_unwired_features_stay_false`` admits ``BUDGET_USD_CAP`` and
  ``BUDGET_TURN_CAP`` into the ``any_adapter_may_support`` set.
- ``ClaudeCodeSession._ensure_client`` now accepts a ``max_turns=``
  kwarg and bakes it into the cache key fragment so a per-execute
  override forces reconnect.

### Added (Phase 5, Iteration C — lifecycle hooks)

- Shared :func:`airframe.sessions._fire_hook_event` helper — every
  adapter funnels emissions through it. Provides two invariants the
  per-adapter wiring would otherwise duplicate: (1) no-op when
  ``on_event`` is ``None`` so per-event call sites need no guard,
  (2) exception safety — a raising observer is caught (except
  ``KeyboardInterrupt`` / ``SystemExit``) and debug-logged so it
  cannot break the session.
- ``EMITTABLE_HOOK_KINDS: ClassVar[frozenset[str]]`` on every
  built-in adapter, documenting the subset of the eight
  :data:`~airframe.hooks.HookEventKind` literals it actually emits.
  Adapters differ by vendor capability: **Claude** emits all eight
  (the only adapter with native ``pre_compact`` + ``rate_limit``);
  **Copilot** emits seven (no ``rate_limit``); **Codex** and
  **OpenAI-compat** emit six (no ``pre_compact``, no
  ``rate_limit``). Consumers writing portable observers can branch
  defensively on the per-runtime set.
- **Claude:** :func:`_build_claude_hooks_config` registers
  :class:`HookMatcher` callbacks for the native SDK kinds
  (``PreToolUse``, ``PostToolUse``, ``UserPromptSubmit``,
  ``PreCompact``, ``SessionStart``, ``SessionEnd``,
  ``Stop`` / ``SubagentStop``). Each callback returns ``{}`` (pure
  observation — never blocks tool use). :func:`_extract_claude_hook_payload`
  normalises per-kind payloads. ``hooks=`` joins
  :attr:`ClaudeAgentOptions` at :meth:`_ensure_client` — connect-time
  bake means observer identity joins the cache key.
- **Copilot:** :meth:`CopilotAgentSession._on_copilot_hook_event`
  subscribes alongside the existing capture handler via a second
  :meth:`CopilotSession.on` registration. Translates
  :class:`SessionStartData` / :class:`SessionShutdownData` /
  :class:`UserMessageData` / :class:`ToolExecutionStartData` /
  :class:`ToolExecutionCompleteData` /
  :class:`SessionCompactionStartData` into the matching
  :class:`HookEvent`. ``submit_result`` tool calls are suppressed
  consistently with the streaming-event filter (structured-output
  plumbing isn't a user-visible tool).
- **Codex:** synthetic ``session_start`` at first ``execute()`` /
  ``stream()`` (no native event); ``user_prompt_submit`` per turn;
  ``pre_tool_use`` / ``post_tool_use`` / ``tool_failure`` translated
  from :class:`CommandExecutionItem` / :class:`McpToolCallItem`
  events. The execute path replays ``turn.items`` post-completion
  (``execute()`` doesn't iterate the event stream — the
  :func:`_fire_item_hooks_post_execute` helper produces the same
  per-tool ordering as the streaming path). Streaming path emits
  ``pre_tool_use`` exactly once per item id via a
  ``tool_pre_fired`` set so repeated ``ItemUpdatedEvent`` frames
  don't multi-fire.
- **OpenAI-compat:** synthetic ``session_start`` at first call;
  ``user_prompt_submit`` per turn; ``pre_tool_use`` /
  ``post_tool_use`` / ``tool_failure`` fired around each
  :meth:`_invoke_tool` call inside the client-side tool loop on both
  ``execute()`` and ``stream()`` paths. ``session_id`` is always
  ``None`` (Chat Completions has no server-side session concept).
- :data:`Feature.LIFECYCLE_HOOKS` flipped True on all four adapters.
- ``examples/probe_hooks.py`` — multi-provider live probe that
  registers a logging observer, drives a tool round-trip, and
  reports the per-kind histogram plus causal-ordering sanity checks
  (``session_start`` first, ``session_end`` last, every emitted kind
  in the declared :attr:`EMITTABLE_HOOK_KINDS`).
- :meth:`close` on every adapter session synthesises ``session_end``
  if the vendor SDK never fired one, gated on a
  ``_session_end_fired`` flag so repeat closes are idempotent.
  Codex / OpenAI-compat additionally gate on
  ``_session_start_fired`` — closing a session that never ran a
  turn does NOT emit a phantom end.
- 43 new unit tests across the four adapter session test files
  covering: hook wiring (SDK options / subscriptions), per-kind
  translation, ordering invariants, idempotent ``close()``, no-op
  when ``on_event`` is omitted, observer-raises tolerance, and
  per-adapter ``EMITTABLE_HOOK_KINDS`` matrix pins. The cross-adapter
  matrix tests in ``tests/test_features.py`` were updated for the
  Iteration-C surface.

### Changed (Phase 5, Iteration C)

- ``tests/test_features.py``: replaced
  ``test_phase_5_hooks_and_budget_stay_false_iteration_b`` with
  ``test_phase_5_budget_stays_false_iteration_c`` (only budget
  remains scaffolded). New ``test_lifecycle_hooks_universal`` pins
  ``LIFECYCLE_HOOKS=True`` on all four adapters;
  ``test_emittable_hook_kinds_matrix`` pins each adapter's emittable
  set so a regression where (e.g.) Codex stops emitting
  ``post_tool_use`` is caught at PR time.
  ``test_unwired_features_stay_false`` admits ``LIFECYCLE_HOOKS``
  into the ``any_adapter_may_support`` set.
- ``CopilotAgentSession._tear_down_session`` refactored to
  unsubscribe both the capture handler and the new hook-event
  handler via a shared loop, keeping cleanup symmetric.

### Added (Phase 5, Iteration B — permission callback)

- **Claude:** ``_translate_permission_for_claude(cb)`` wraps the
  user's :class:`~airframe.permission.PermissionCallback` as
  :attr:`ClaudeAgentOptions.can_use_tool`. Decision mapping:
  ``"allow"`` → :class:`PermissionResultAllow`, ``"deny"`` →
  :class:`PermissionResultDeny` (with the request's ``reason`` as
  the message), ``"defer"`` → :class:`PermissionResultAllow` with a
  debug log (Claude's binary result type has no third option; the
  existing ``permission_mode="bypassPermissions"`` default already
  matches the "defer" intent). Callback identity joins the
  :meth:`ClaudeCodeSession._ensure_client` cache key — a callback
  swap forces reconnect because ``can_use_tool`` is baked at
  connect time.
- **Copilot:** ``_translate_permission_for_copilot(cb)`` wraps the
  callback as Copilot's ``_PermissionHandlerFn`` (replacing the
  ``PermissionHandler.approve_all`` default when supplied). Decision
  mapping: ``"allow"`` → ``"approve-once"``, ``"deny"`` →
  ``"reject"``, ``"defer"`` → ``"user-not-available"`` (Copilot's
  default policy takes over). Callback identity joins the session
  cache key so a swap forces a session rebuild.
- **Codex:** ``approval_policy`` is *session-wide*, not per-call —
  the airframe per-call contract bridges by invoking the callback
  exactly **once** at first ``execute()`` with a sentinel
  :class:`PermissionRequest` (``tool_name="*"``, explanatory
  ``reason``). The returned decision maps to
  :data:`ApprovalMode`: ``"allow"`` → ``"never"`` (auto-approve
  everything), ``"deny"`` → ``"untrusted"`` (strictest), ``"defer"``
  → ``"on-request"`` (Codex's default per-call prompting). Resolved
  policy is cached for the session's lifetime. Consumers needing
  per-call interception should use Claude or Copilot; the
  limitation is documented in the class docstring and surfaced via
  the sentinel ``reason``.
- **OpenAI-compat:** permanent decline. Inline raise in
  :meth:`OpenAICompatibleRuntime.session` returns
  :class:`UnsupportedFeatureError` (``feature=PERMISSION_CALLBACK``)
  with an actionable pointer at a future
  ``OpenAIResponsesRuntime``. Symmetric with Phase 4 Iteration D's
  permanent ``mcp_servers=`` decline.
- :data:`Feature.PERMISSION_CALLBACK` flipped True on Claude /
  Copilot / Codex.
- ``examples/probe_permission.py`` — multi-provider live probe that
  registers a logging callback approving every request. Defaults to
  ``claude`` (richest per-call permission channel); OpenAI-compat
  surfaces the decline verbatim (probe-as-docs).
- 17 new unit tests across the four adapter session test files
  covering: wrap-into-vendor-channel translation, decision mapping
  per adapter (allow / deny / defer), defer-coercion debug logging
  on Claude, ``user-not-available`` mapping on Copilot, session-wide
  ApprovalMode mapping on Codex, called-once semantics on Codex,
  callback-identity cache invalidation on Claude / Copilot,
  no-callback default behaviour on every adapter, permanent decline
  message on OpenAI-compat, and the cross-adapter matrix in
  ``tests/test_features.py``.

### Changed (Phase 5, Iteration B)

- ``tests/test_features.py``: replaced
  ``test_phase_5_features_stay_false_iteration_a`` and
  ``test_session_on_permission_kwarg_raises_on_every_adapter`` with
  per-adapter expectations now that PERMISSION_CALLBACK has flipped
  on three of four. New tests
  ``test_permission_callback_universal_except_openai_compat``,
  ``test_phase_5_hooks_and_budget_stay_false_iteration_b``, and
  ``test_session_on_permission_kwarg_raises_only_on_openai_compat``
  pin the Iteration-B matrix. ``test_unwired_features_stay_false``
  admits ``PERMISSION_CALLBACK`` into the
  ``any_adapter_may_support`` set.
- ``OpenAICompatibleRuntime`` no longer calls
  :func:`airframe.sessions._check_permission_supported` — the
  permanent decline is now inline with a vendor-specific message
  (same shape as the ``mcp_servers=`` decline added in Phase 4
  Iteration D).

---

**Phase 4 of the [implementation plan](docs/implementation-plan.md) is
complete — MCP server references (Tier 2).** Iteration A landed the
protocol-surface shape locks (the :class:`McpServerRef` dataclass,
the new :data:`Feature.TOOLS_MCP_SSE` enum member, and the
``mcp_servers=`` kwarg on every adapter's :meth:`session` signature
gated by the shared capability helper). Iteration B wired
:class:`ClaudeCodeRuntime` — all three transports (stdio, http, sse).
Iteration C wired :class:`CopilotRuntime` — stdio + http natively;
SSE declined per the plan with an actionable "switch to http"
message. Iteration D codifies the Codex + OpenAI-compat permanent
declines with vendor-specific actionable pointers (``~/.codex/config.toml``
for Codex, a future ``OpenAIResponsesRuntime`` for OpenAI-compat),
ships the multi-provider probe, and pins the final transport matrix.

### Added (Phase 4, Iteration D — declines, probe, wrap-up)

- `CodexRuntime.session(mcp_servers=<non-empty>)` raises
  :class:`~airframe.errors.UnsupportedFeatureError` with a
  Codex-specific message pointing consumers at the
  ``[[mcp_servers]]`` block in ``~/.codex/config.toml``. Symmetric
  with Phase 3 Iteration D's ``tools=`` decline — the decline is
  **permanent**, not a "wait for the next iteration" gate. The
  :attr:`~airframe.errors.UnsupportedFeatureError.feature` attribute
  carries the *first ref's transport* so consumer code branching on
  ``Feature.TOOLS_MCP_STDIO`` / ``TOOLS_MCP_HTTP`` / ``TOOLS_MCP_SSE``
  still works.
- `OpenAICompatibleRuntime.session(mcp_servers=<non-empty>)` raises
  with an OpenAI-compat-specific message: ``"Chat Completions has no
  MCP-as-tool wire shape; that lives on the Responses API. A future
  ``OpenAIResponsesRuntime`` (separate from this compat family) could
  translate to the Responses-API ``{"type": "mcp", ...}`` tool
  shape."`` Pointer to the future direct-API option. Same
  first-ref-transport on ``.feature`` as Codex.
- `examples/probe_mcp.py` — multi-provider live probe. Registers
  one external stdio MCP server (default:
  ``@modelcontextprotocol/server-everything`` launched via
  ``npx``), prompts the model to list the server's tools (and call
  ``echo`` if available), and prints the
  :class:`~airframe.events.ToolCallStart` /
  :class:`~airframe.events.ToolCallResult` /
  :class:`~airframe.events.TextDelta` /
  :class:`~airframe.events.TurnComplete` sequence from
  :meth:`AgentSession.stream`. Defaults to ``claude`` (broadest
  transport coverage); accepts
  ``--provider claude|github-copilot|codex|opencode`` and
  ``--transport stdio|http|sse``. Adapters that decline surface
  their message verbatim — probe-as-docs, same pattern Phase 3
  used.
- 8 new unit tests in `tests/test_codex_session.py` and
  `tests/test_openai_compatible_session.py` covering per-adapter
  decline messages (both content *and* ``feature=`` attribute),
  per-transport feature mapping, ``mcp_servers=None`` /
  ``mcp_servers=[]`` no-op paths, and the
  ``supports(TOOLS_MCP_*) == False`` capability matrix on both
  adapters.
- `tests/test_features.py::test_mcp_transports_final_matrix` pins
  the Phase-4 endgame table: Claude all three transports, Copilot
  stdio + http, Codex + OpenAI-compat none. The
  ``TOOLS_MCP_IN_PROCESS`` flag stays False on every adapter — Phase
  4 doesn't expose ``transport="in_process"`` on
  :class:`McpServerRef`; the Phase 3 in-process MCP path on Claude
  is internal plumbing for ``tools=`` rather than a user-facing
  capability.

### Changed (Phase 4, Iteration D)

- `CodexRuntime` and `OpenAICompatibleRuntime` no longer call
  :func:`airframe.sessions._check_mcp_servers_supported`. Both now
  raise their vendor-specific declines inline (mirroring the way
  Phase 3 Iteration D unhooked Codex from
  :func:`~airframe.sessions._check_tools_supported`). The shared
  helper still services Claude and Copilot (their transport flags
  flipped True in Iterations B and C mean the helper short-circuits
  to a no-op).
- `docs/implementation-plan.md` — Phase 4 marked ✅ complete; each
  iteration heading marked ✅; dependency-graph annotation updated
  from ``[gated on divergence signal]`` to ``[✅ shipped]``.

### Added (Phase 4, Iteration C — Copilot MCP wiring)

- `_translate_mcp_servers_for_copilot(refs)` builds the dict shape
  :meth:`CopilotClient.create_session`'s ``mcp_servers=`` slot
  expects. **Stdio** maps to ``{"type": "local", "command": <head>,
  "args": <tail>}`` — Copilot's wire enum is ``"local"``, not
  ``"stdio"``, and ``args`` is always emitted (the schema requires
  it). **Http** maps to ``{"type": "http", "url": <url>, "headers":
  <merged>}``. **Sse** raises defensively — the SSE decline is
  enforced at the :meth:`CopilotRuntime.session` boundary so the
  translator should never see one.
- `CopilotAgentSession(mcp_servers=...)` — accepted on the session
  factory; translated lazily at :meth:`_ensure_session` and passed
  via :meth:`CopilotClient.create_session`'s ``mcp_servers=`` kwarg.
  Co-exists with ``tools=`` (FunctionTool) and the forced
  ``submit_result`` tool — the three-way combination passes through
  separate ``create_session`` kwargs, no shadowing.
- `_ensure_session` cache key gains an ``mcp=<fingerprint>``
  fragment (Copilot bakes ``mcp_servers`` at create-session time,
  same as ``tools=`` and ``reasoning_effort``). Refs change → session
  destroy + rebuild.
- `Feature.TOOLS_MCP_STDIO` / `TOOLS_MCP_HTTP` flipped True on
  :attr:`CopilotRuntime.SUPPORTED_FEATURES`. ``TOOLS_MCP_SSE`` stays
  **False per the plan**; refs of that transport surface a
  Copilot-specific :class:`~airframe.errors.UnsupportedFeatureError`
  pointing consumers at the ``http`` transport. The decline runs
  *before* the shared capability gate so the consumer gets the
  actionable message rather than the generic one.
- 12 new unit tests in `tests/test_copilot_session.py` covering
  stdio→``local`` translation (with the always-emit-``args``
  invariant), http with bearer-auth + header merging precedence,
  the SSE-rejection-with-http-hint path, mixed-transport sessions,
  ``tools=`` + ``mcp_servers=`` coexistence, the three-way
  ``submit_result`` + custom tools + MCP combination, the
  no-``mcp_servers=`` ⇒ kwarg-omitted path, cache invalidation on
  refs-change, and the per-flag capability matrix.

### Changed (Phase 4, Iteration C)

- **Shared helpers extracted to `airframe.sessions`.**
  `_mcp_servers_fingerprint(refs)` and `_compose_mcp_headers(ref)`
  moved out of `claude_code.py` so Copilot reuses the same
  secret-free fingerprint logic and the same
  ``Authorization: Bearer`` shorthand-with-caller-override merge.
  Previous Claude callers updated; tests updated to import from the
  new home.
- `tests/test_features.py`:
  `test_non_claude_adapters_still_decline_mcp_transports` and
  `test_session_mcp_servers_kwarg_raises_on_non_claude_adapters`
  evolved into per-adapter expectations now that Copilot also
  declares stdio + http True. New tests
  `test_copilot_declares_stdio_and_http_but_not_sse`,
  `test_codex_and_openai_compat_still_decline_all_mcp_transports`,
  `test_copilot_session_sse_decline_carries_http_hint`, and
  `test_session_mcp_servers_kwarg_raises_on_codex_and_openai_compat`
  pin the Iteration-C matrix.

### Note: SDK observation

While wiring this, we noticed Copilot's
:class:`MCPServerConfigHTTPType` enum exposes both
``HTTP`` and ``SSE`` as valid remote transports — i.e. the Copilot
SDK does have an SSE channel today. The plan was written when this
was not the case and explicitly declines SSE on Copilot. This
iteration follows the plan as written; flipping
:data:`Feature.TOOLS_MCP_SSE` True on Copilot would be a small
follow-up if a downstream consumer needs it (the translator change
is one line: extend the ``"http"`` arm to also accept ``"sse"``;
the gate change is removing the pre-helper SSE decline).

### Added (Phase 4, Iteration B — Claude MCP wiring)

- `_translate_mcp_servers_for_claude(refs)` builds a dict keyed by
  :attr:`McpServerRef.name` where each value is the matching SDK
  TypedDict — :class:`McpStdioServerConfig` /
  :class:`McpHttpServerConfig` / :class:`McpSSEServerConfig`.
  Stdio splits :attr:`McpServerRef.command` (an argv list) into the
  SDK's ``command: str`` + ``args: list[str]``; ``args`` is omitted
  for a one-element command. Http / sse pass :attr:`McpServerRef.url`
  through verbatim. :attr:`McpServerRef.auth_token` becomes
  ``Authorization: Bearer <token>`` (merged with
  :attr:`McpServerRef.headers`; caller-supplied ``Authorization``
  wins on collision).
- `ClaudeCodeSession(mcp_servers=...)` — accepted on the session
  factory. Translated lazily at connect time; the translated dict is
  **merged** with the in-process tools server (Phase 3) into
  :attr:`ClaudeAgentOptions.mcp_servers`. Tool-name collisions with
  the reserved ``airframe_tools`` slot raise
  :class:`ValueError` at connect — no silent shadowing.
- `_mcp_servers_fingerprint(refs)` — deterministic, secret-free
  fingerprint added to :meth:`ClaudeCodeSession._ensure_client`'s
  cache key. Participates from ``name``, ``transport``, ``command``,
  ``url``, and the *sorted keys* of ``headers``; never includes
  header values or ``auth_token``, so rotating a bearer token doesn't
  accidentally invalidate the cache (and sensitive material doesn't
  enter cache identity).
- `_strip_mcp_prefix` generalised from the Phase-3 single-server form
  to a multi-server set the session tracks: the in-process
  ``airframe_tools`` (when ``tools=`` is set) plus every
  :attr:`McpServerRef.name`. Tool calls routed through a recognised
  server come back with the bare name on
  :class:`~airframe.events.ToolCallStart`; unrecognised prefixes pass
  through verbatim per Phase 4 risk note #6.
- Per-server wildcard allowed-tools entry: each external server adds
  ``mcp__<server>__*`` to :attr:`ClaudeAgentOptions.allowed_tools`
  so every tool the server exposes is auto-allowed (parallels the
  per-tool entries the in-process tools server adds).
- `Feature.TOOLS_MCP_STDIO` / `TOOLS_MCP_HTTP` / `TOOLS_MCP_SSE`
  flipped True on :attr:`ClaudeCodeRuntime.SUPPORTED_FEATURES`.
  `TOOLS_MCP_IN_PROCESS` stays False — Phase 4 doesn't expose an
  in-process :class:`McpServerRef` transport; the in-process MCP
  server is internal plumbing for ``tools=``.
- 14 new unit tests in `tests/test_claude_code_session.py` covering
  per-transport translation, bearer/header merging precedence,
  mixed-transport sessions, ``tools=`` + ``mcp_servers=`` coexistence,
  name collisions, cache invalidation on refs-change, fingerprint
  determinism + secret exclusion, external-server prefix stripping,
  and unknown-prefix pass-through.

### Changed (Phase 4, Iteration B)

- `tests/test_features.py`:
  `test_mcp_transport_features_stay_false_iteration_a` and
  `test_session_mcp_servers_kwarg_raises_on_every_adapter` evolved
  into per-adapter expectations now that Claude declares the three
  transport flags True. New tests
  `test_claude_declares_all_three_mcp_transports`,
  `test_non_claude_adapters_still_decline_mcp_transports`,
  `test_session_mcp_servers_kwarg_raises_on_non_claude_adapters`, and
  `test_claude_session_mcp_servers_kwarg_opens_cleanly` pin the
  Iteration-B matrix.
- `test_unwired_features_stay_false` admits
  ``TOOLS_MCP_{STDIO,HTTP,SSE}`` into the
  ``any_adapter_may_support`` set (Claude declares them; the other
  three still decline).

Phase 3 — function tools (Tier 1) — is complete. Iteration A landed
the protocol-surface shape locks; Iteration B wired the OpenAI-
compatible family; Iteration C wired Claude + Copilot through their
respective SDK tool-registration channels; Iteration D codifies
Codex's permanent decline and ships the multi-provider probe.

### Added (Phase 4, Iteration A — protocol scaffolding for MCP refs)

- `airframe.tools.McpServerRef` — frozen+slots dataclass with
  ``name``, ``transport`` (``Literal["stdio", "http", "sse"]``),
  ``command``, ``url``, ``headers``, ``auth_token``.
  ``__post_init__`` validates the transport/field combo: stdio
  requires ``command`` and rejects ``url``; http and sse require
  ``url`` and reject ``command``. Re-exported at the top level as
  ``from airframe import McpServerRef``.
- `Feature.TOOLS_MCP_SSE = "tools_mcp_sse"` — new enum member
  alongside the existing ``TOOLS_MCP_STDIO`` / ``TOOLS_MCP_HTTP`` /
  ``TOOLS_MCP_IN_PROCESS``. Snapshotted in
  `test_feature_string_values_are_stable` from Iteration A onwards
  so a future rename is caught at PR time.
- `airframe.sessions._check_mcp_servers_supported(refs, *,
  adapter_label, supports)` — shared capability gate, symmetric with
  `_check_tools_supported`. Iterates the list, looks up the matching
  ``Feature.TOOLS_MCP_{STDIO,HTTP,SSE}`` per ref, and raises
  :class:`~airframe.errors.UnsupportedFeatureError` on the first
  decline with the specific feature attached. ``supports`` is a
  callable so the helper doesn't need a runtime reference.
- ``mcp_servers: list[McpServerRef] | None = None`` kwarg on
  :meth:`AgentRuntime.session` and every adapter's :meth:`session`.
  Each adapter calls ``_check_mcp_servers_supported`` at the top;
  Iteration A's stopping point is "non-empty list raises immediately,
  empty list / ``None`` opens cleanly." Iterations B–D flip the True
  bits and replace the generic decline messages with vendor-specific
  ones where the decline is *permanent* (Codex's CLI-config pointer,
  OpenAI-compat's "Responses-API only" pointer).
- 11 new unit tests in `tests/test_tools.py` pinning the
  :class:`McpServerRef` shape — frozen+slots, field order, the four
  validation paths (stdio missing command, stdio with url, http/sse
  missing url, http/sse with command), bearer/header pass-through,
  and the top-level export.
- 3 new unit tests in `tests/test_features.py` pinning the matrix at
  the start of Phase 4: every transport flag is False on every
  adapter; ``session(mcp_servers=[<ref>])`` raises with the
  expected ``feature=`` attribute on every adapter for each
  transport; ``mcp_servers=None`` / ``mcp_servers=[]`` opens
  cleanly everywhere.

### Added (Phase 3, Iteration D — Codex decline + probe + docs)

- `CodexRuntime.session(tools=<non-empty>)` raises
  :class:`~airframe.errors.UnsupportedFeatureError` with a
  Codex-specific message pointing consumers at the ``codex`` CLI's
  config file (``~/.codex/config.toml``). Replaces the generic
  shared-helper decline used during Iterations A–C — Codex's
  ``TOOLS_FUNCTION=False`` is **permanent**, not a "wait for the next
  iteration" gate, and the error text now says so explicitly.
  ``tools=None`` / ``tools=[]`` are no-op and don't trigger the
  decline (consistent with the other three adapters' "empty list is
  the same as omitted" handling).
- `examples/probe_tools.py` — multi-provider live probe. Registers a
  tiny ``add(a, b: float) -> float`` :class:`FunctionTool`, prompts
  "what is 17 times 23?", and prints the resulting
  :class:`~airframe.events.ToolCallStart` /
  :class:`~airframe.events.ToolCallResult` /
  :class:`~airframe.events.TextDelta` /
  :class:`~airframe.events.TurnComplete` sequence from
  :meth:`AgentSession.stream`. Defaults to ``opencode`` (simplest
  auth; deterministic client-side tool-loop); accepts
  ``--provider claude|github-copilot|opencode|codex``. The Codex
  branch surfaces the decline message verbatim so the probe doubles
  as documentation for the workaround.
- 3 new unit tests in `tests/test_codex_session.py` covering the
  CLI-config decline message, the ``tools=None`` / ``tools=[]``
  no-op paths, and the capability flag staying False.

### Changed (Phase 3, Iteration D)

- `tests/test_features.py`:
  `test_three_adapters_declare_tools_function` renamed to
  `test_tools_function_universal_except_codex` to pin the final
  Phase 3 matrix: Claude + Copilot + OpenAI-compat all declare
  ``TOOLS_FUNCTION=True``; Codex alone stays False. The matching
  `…_kwarg_raises_…` test now also asserts the decline message
  carries both ``codex`` and ``config`` so message rot doesn't
  silently regress the workaround pointer.
- `CodexRuntime` no longer imports
  :func:`airframe.sessions._check_tools_supported`. The
  Codex-specific decline lives inline in
  :meth:`CodexRuntime.session`; the shared helper still services
  Claude / Copilot / OpenAI-compat (their ``TOOLS_FUNCTION`` flips
  to ``True`` mean the helper short-circuits cleanly).

### Added (Phase 3, Iteration C — Claude + Copilot tool dispatch)

- **Claude:** `_translate_tools_for_claude(tools)` builds an in-process
  MCP server via `claude_agent_sdk.create_sdk_mcp_server(...)`. Each
  `FunctionTool` becomes one `@tool`-decorated coroutine that
  validates incoming args against the user's Pydantic schema, awaits
  the handler, and wraps the return in the SDK's
  `{"content": [{"type":"text","text":...}]}` envelope. Handler
  exceptions / argument-validation failures come back as
  `isError=True` so the model can recover.
- `ClaudeCodeSession(tools=...)` — accepted on the session factory.
  Tools are baked into `ClaudeAgentOptions.mcp_servers` at connect
  time, with matching `mcp__airframe_tools__<name>` entries appended
  to `allowed_tools`. The `_ensure_client` cache key gains a
  `tools=<fingerprint>` fragment so a tools-change between calls
  forces a reconnect.
- `ClaudeCodeSession.stream()` translates `ToolUseBlock` on
  `AssistantMessage` content into `ToolCallStart` and the matching
  `ToolResultBlock` on `UserMessage` content into `ToolCallResult`.
  The `mcp__airframe_tools__` prefix is stripped from `tool_name`
  so consumers see the same `FunctionTool.name` they registered.
- Helpers exported from `airframe.adapters.claude_code`:
  `AIRFRAME_MCP_SERVER_NAME = "airframe_tools"`.
- `Feature.TOOLS_FUNCTION` flipped **True** on
  `ClaudeCodeRuntime.SUPPORTED_FEATURES`.
- **Copilot:** `_translate_one_copilot_tool(ft)` wraps each
  `FunctionTool` as a `copilot.define_tool(...)` registration. The
  `(params, invocation_context) -> result` SDK handler signature is
  adapted to the airframe `(BaseModel) -> Awaitable[Any]` shape by
  ignoring the invocation context. `skip_permission=True` matches the
  existing session-wide `PermissionHandler.approve_all` policy.
- `CopilotAgentSession(tools=...)` — accepted on the session factory.
  Tools are passed via `CopilotClient.create_session(tools=...)` at
  session-creation time. The session cache key (already keyed on
  schema + reasoning-effort) gains a `tools=<fingerprint>` fragment
  so a tools-change forces a rebuild.
- **`submit_result` + custom tools coexistence:** when both `schema=`
  and `tools=` are passed to `CopilotRuntime.session(...)`, the
  adapter prepends the forced `submit_result` tool to the user's
  list. The model sees the structured-output gate in slot 0, then
  the user's tools. Streaming events for `submit_result` are
  filtered out (it's structured-output plumbing, not a user-visible
  tool call).
- `CopilotAgentSession.stream()` translates
  `ToolExecutionStartData` → `ToolCallStart` and
  `ToolExecutionCompleteData` → `ToolCallResult`. Failure responses
  (`success=False`) come back with `is_error=True` and the
  `<code>: <message>` text from `ToolExecutionCompleteError`.
- `Feature.TOOLS_FUNCTION` flipped **True** on
  `CopilotRuntime.SUPPORTED_FEATURES`.
- 8 new unit tests in `tests/test_claude_code_session.py` covering:
  mcp_servers + allowed_tools wiring, the @tool wrapper's validation
  + envelope behaviour, handler-exception → `isError=True`,
  invalid-args → validation error, streaming `ToolCallStart` /
  `ToolCallResult` shape, `ToolResultBlock(is_error=True)`
  propagation, tools-change reconnects, no-tools omits the kwargs.
- 8 new unit tests in `tests/test_copilot_session.py` covering:
  `tools=` reaches `create_session`, the `(params, invocation)`
  adapter signature, `submit_result` + custom-tools coexistence,
  no-tools omits the kwarg, streaming `ToolCallStart` /
  `ToolCallResult` shape, `submit_result` filtered from stream,
  `success=False` → `is_error=True`, tools-change rebuilds.

### Changed (Phase 3, Iteration C)

- `tests/test_features.py` updated for the new asymmetry:
  `test_openai_compatible_declares_tools_function` replaced with
  `test_three_adapters_declare_tools_function` pinning that Claude,
  Copilot, and OpenAI-compat all declare TOOLS_FUNCTION while Codex
  alone stays False; the matching `…_kwarg_raises_…` test now only
  asserts the rejection contract on Codex.

### Added (Phase 3, Iteration B — OpenAI-compat tool round-trip)

- `_translate_tools_for_openai(tools) → list[dict]` — translates
  each `FunctionTool` to the OpenAI wire shape
  (`{"type":"function","function":{"name":…,"description":…,
  "parameters":<json_schema>}}`). Schemas come from
  `FunctionTool.params.model_json_schema()`; the result is computed
  once at session construction and reused on every
  `chat.completions.create()` call.
- `OpenAICompatibleSession(tools=…)` — accepts and caches a
  `list[FunctionTool]` for the session's lifetime. The wire payload
  is precomputed; the per-name lookup table drives dispatch.
- Client-side tool-loop in `_do_execute`: when the response carries
  `tool_calls`, dispatch each handler, append the assistant message
  (with the original `tool_calls` payload) and one `role="tool"`
  reply per call, then re-call. Loops until the model produces a
  final text response or `MAX_TOOL_ITERATIONS=20` round-trips
  elapse — runaway loops surface as
  `RuntimeProtocolError` with a "model kept requesting tools without
  producing a final response" message. Handler exceptions, unknown
  tool names, and Pydantic validation failures all flow back to the
  model as `role="tool"` replies (with `is_error=True` on the
  matching streaming event), so the model can recover on its next
  turn.
- Tool-loop in `stream()`: accumulates `delta.tool_calls` fragments
  by `index` across chunks, then for each call emits `ToolCallStart`
  (with the accumulated `arguments` as `arguments_preview`), invokes
  the handler, emits `ToolCallResult`, and appends the `role="tool"`
  message before the next iteration. Exactly one `TurnComplete` at
  the very end of the user turn — intermediate model turns produce
  tool events but no `TurnComplete` (consistent with the docstring
  clarified in Iteration A).
- `_serialize_tool_output(output)` — JSON-encodes a handler return
  value for the `role="tool"` content field. Strings pass through;
  everything else round-trips via `json.dumps(default=str)`;
  unserialisable types fall back to `repr()`.
- `MAX_TOOL_ITERATIONS = 20` exported from
  `airframe.adapters.openai_compatible`.
- `Feature.TOOLS_FUNCTION` flipped **True** on
  `OpenAICompatibleRuntime.SUPPORTED_FEATURES`. The other three
  adapters keep their Iteration A capability — non-None `tools=`
  still raises `UnsupportedFeatureError` until Iterations C and D
  land.
- 13 new unit tests in `tests/test_openai_compatible_session.py`
  covering: single-tool round-trip, parallel tool calls (multiple
  per assistant message), handler-exception → `is_error` recovery,
  unknown-tool-name → `is_error` message, iteration cap raising
  `RuntimeProtocolError`, buffer rollback on mid-loop failure,
  no-tools sessions omit the `tools=` kwarg, streaming
  `ToolCallStart` / `ToolCallResult` event shape, streaming handler
  exception → `is_error=True`, and streaming iteration cap.

### Changed (Phase 3, Iteration B)

- `tests/test_features.py::test_unwired_features_stay_false` now
  permits `TOOLS_FUNCTION` to vary per adapter (the OpenAI-compat
  family flipped it True; the other three are still False).
- `test_no_adapter_declares_tools_function_yet` replaced with
  `test_openai_compatible_declares_tools_function` pinning the new
  asymmetry; `test_session_tools_kwarg_raises_unsupported_feature`
  replaced with `…_on_unwired_adapters` which exempts OpenAI-compat
  from the rejection contract.
- `OpenAICompatibleSession.execute()` and `.stream()` now snapshot
  the message buffer length up front and roll back to that length
  on any failure, replacing the older "pop one user message if it
  was the last entry" logic. The tool-loop may append multiple
  intermediate messages, so the single-pop rollback would have left
  intermediate state behind.

### Added (Phase 3, Iteration A — protocol scaffolding)

- `airframe.tools.FunctionTool` — frozen+slots dataclass with
  ``name``, ``description``, ``params: type[BaseModel]``, and
  ``handler: Callable[[BaseModel], Awaitable[Any]]``. The
  ``handler`` signature is the **shape lock** for Phase 3: parsed
  Pydantic in, JSON-serialisable Any out. Matches Copilot's
  ``define_tool`` and LangChain's ``@tool`` decorator.
- `tools: list[FunctionTool] | None = None` kwarg on
  :meth:`AgentRuntime.session` Protocol and every adapter's
  ``session()`` (Claude / Copilot / Codex / OpenAI-compat).
- `airframe.sessions._check_tools_supported(tools, *, adapter_label,
  feature_supported)` — shared helper every adapter calls at the
  top of ``session()``. Raises
  :class:`~airframe.errors.UnsupportedFeatureError` with
  ``feature=Feature.TOOLS_FUNCTION`` on a non-None list while the
  adapter's capability flag is False — the same "no silent
  fallback" pattern the prompt-parts helper uses.
- Top-level export: `FunctionTool`.
- 6 new tests in `tests/test_tools.py` pinning the dataclass shape
  (frozen+slots, field order, handler signature, public export
  path).
- 2 new feature-matrix tests: ``TOOLS_FUNCTION`` stays False on every
  adapter; ``session(tools=...)`` raises ``UnsupportedFeatureError``
  on every adapter.

### Changed (Phase 3, Iteration A)

- `TurnComplete` docstring clarified: one ``TurnComplete`` per
  *user turn*, not per *model turn*. Tool round-trips (Phase 3)
  produce intermediate ``ToolCallStart`` / ``ToolCallResult`` pairs
  but only one trailing ``TurnComplete`` with the final model
  turn's result.


---

## [0.5.0] — 2026-05-17

The biggest jump since v0.3.0: bundles both **Phase 1**
(`AgentSession` + streaming + cancel + session resume) and **Phase 2**
(`thinking=` + vision / file input) of the
[implementation plan](docs/implementation-plan.md). No 0.4.x was cut
— Phase 1 + Phase 2 land together to mark a clean break before
Phase 3 (function tools).

Headline additions:

- **`AgentSession`** — the new hinge surface.
  `runtime.session(...)` returns a bespoke per-vendor session that
  owns its own lifecycle; `execute()` becomes documented sugar for
  `runtime.session().execute() + close()`.
- **`session.stream()`** — `AsyncIterator[RuntimeEvent]` with
  `TextDelta` / `ReasoningDelta` / `TurnComplete` events across
  every adapter.
- **`session.cancel()`** — vendor-native interrupt on every adapter.
- **`session(resume=<id>)`** — server-side session resume on the
  three SDK adapters (Claude / Copilot / Codex).
- **`thinking=`** — universal `REASONING_EFFORT` kwarg on every
  adapter; Claude additionally honours
  `thinking={"budget_tokens": N}` via `REASONING_BUDGET_TOKENS`.
- **`prompt: list[PromptPart]`** — polymorphic prompt with
  `ImageInput` / `FileInput` parts. Vision wired on every adapter;
  files wired on Claude / Copilot / Codex (OpenAI-compat stays
  False — varies wildly across compat vendors).
- **Reasoning-token telemetry** on `CostRecord.reasoning_tokens`
  for every adapter that exposes the counter.

Iteration-by-iteration detail follows. Phase 2 iterations land
first (most recent), then Phase 1 retrospective.

### Phase 2 — Inputs and reasoning

### Added (Phase 2, Iteration A — protocol scaffolding)

- `airframe.thinking.ThinkingMode` — union value type for the
  ``thinking=`` kwarg. Variants: ``None`` (default), literal
  ``ReasoningEffort`` (``"minimal"|"low"|"medium"|"high"``), the
  Claude-style ``dict`` (currently documented as
  ``{"budget_tokens": int}``), and ``"disabled"`` (explicit-off
  sentinel). The shape is the **ADR-006 lock** — dict literal for
  ``budget_tokens`` rather than a dedicated dataclass matches
  Anthropic's wire format and what consumers already write.
- `airframe.thinking.ReasoningEffort` — exported `Literal` for the
  portable effort levels.
- `airframe.inputs.ImageInput` — frozen+slots dataclass with
  ``path | bytes_ | url`` + optional ``media_type``. Post-init
  enforces at-least-one-source.
- `airframe.inputs.FileInput` — frozen+slots dataclass with
  ``path`` + optional ``media_type``.
- `airframe.inputs.PromptPart` = ``str | ImageInput | FileInput``.
- `airframe.inputs.Prompt` = ``str | list[PromptPart]`` — back-compat
  with the v0–Phase-1 bare-``str`` shape.
- `AgentRuntime.execute()`, `AgentSession.execute()`, and
  `AgentSession.stream()` all accept the new ``thinking=`` kwarg and
  the polymorphic ``prompt: Prompt`` parameter on the Protocol.
- Top-level exports: `ThinkingMode`, `ReasoningEffort`, `ImageInput`,
  `FileInput`, `Prompt`, `PromptPart`.
- 19 new unit tests across `tests/test_thinking.py` and
  `tests/test_inputs.py` pinning the union shape and the dataclass
  field sets.

### Changed (Phase 2, Iteration A)

- All four adapters' `execute()` / `session.execute()` /
  `session.stream()` signatures accept the new kwargs:
  - ``thinking=`` is accepted but ignored — Iteration B wires it
    per-adapter (Claude → ``ClaudeAgentOptions.thinking``; Copilot →
    ``reasoning_effort`` on session; Codex →
    ``ThreadOptions.model_reasoning_effort``; OpenAI-compat →
    ``chat.completions.create(reasoning_effort=...)``).
  - List-shaped prompts raise `UnsupportedFeatureError` via the
    shared `airframe.sessions._coerce_prompt_or_raise` helper —
    Iteration C wires per-adapter image / file routing and flips
    `Feature.VISION_INPUT` / `Feature.FILE_INPUT`.
- `tests/test_agent_session_protocol.py` updated for the new
  parameter order (`schema`, `thinking`, `timeout`).

### Added (Phase 2, Iteration B — `thinking=` wiring)

- `_translate_thinking_for_openai`, `_translate_thinking_for_claude`,
  `_translate_thinking_for_copilot`, `_translate_thinking_for_codex`
  — per-adapter helpers that map `ThinkingMode` onto each vendor's
  native channel:
  - OpenAI-compat → `chat.completions.create(reasoning_effort=...)`.
  - Claude Code → `ClaudeAgentOptions.effort` for the literal
    string, `ClaudeAgentOptions.thinking={"type": "enabled",
    "budget_tokens": N}` for the dict shape, and
    `thinking={"type": "disabled"}` for explicit-off.
  - Copilot → `CopilotClient.create_session(reasoning_effort=...)`
    / `resume_session(reasoning_effort=...)`.
  - Codex → `ThreadOptions.modelReasoningEffort`.
- `Feature.REASONING_EFFORT` flipped True on **all four adapters**.
- `Feature.REASONING_BUDGET_TOKENS` flipped True on `ClaudeCodeRuntime`
  only — the dict shape is Anthropic-only. The other three adapters
  raise `UnsupportedFeatureError` at translation time when handed a
  dict.
- `CostRecord.reasoning_tokens` populated from each vendor's counter:
  Anthropic `thinking_tokens`, OpenAI
  `completion_tokens_details.reasoning_tokens`, Copilot
  `usage.reasoning_tokens`. (Codex's `Usage` doesn't expose a
  reasoning counter — stays 0.)
- 25 new session-level unit tests across the four
  `tests/test_*_session.py` files pinning per-adapter wire shape, the
  no-coercion / coerce-minimal-to-low behaviors, the dict-on-non-Claude
  decline path, and the cache-invalidation-on-thinking-change semantics
  for the three SDK adapters that bake effort at session-creation time.
- 2 new feature-matrix tests pin `REASONING_EFFORT` as universal and
  `REASONING_BUDGET_TOKENS` as Claude-only.

### Changed (Phase 2, Iteration B)

- The Claude / Copilot / Codex session caches now key on
  `(schema, thinking)` (Codex: `(model, thinking)`) — a thinking
  change between turns rebuilds the underlying vendor session /
  thread because each vendor bakes the effort at connect /
  create-session / start-thread time. OpenAI-compat is per-call, so
  its session is stateless on this axis.
- `"minimal"` coerces to `"low"` on Claude and Copilot at debug-log
  level (no `"minimal"` rung in either vendor's enum). Not silent —
  the log records the coercion. OpenAI-compat and Codex accept
  `"minimal"` natively.

### Added (Phase 2, Iteration C — vision + file input)

- `airframe.sessions._split_prompt_parts(prompt, *, adapter_label,
  supports_vision, supports_file)` — shared helper returning
  `(text, images, files)`. Path-only in v0;
  `ImageInput.bytes_`/`url` raise `UnsupportedFeatureError` with a
  "materialise to disk and pass path=" message; unsupported parts
  raise with the corresponding `Feature` attribute.
- `_build_user_content` (OpenAI-compat) — promotes a user message's
  `content` from a bare string to the content-parts list shape
  (`[{"type":"text",...},{"type":"image_url","image_url":{"url":"data:..;base64,..."}}]`)
  when images are present. Path bytes are base64-encoded inline; the
  `media_type` is taken from `ImageInput.media_type` or sniffed via
  `mimetypes.guess_type` (defaults to `image/png`).
- `_build_codex_input` (Codex) — switches the `Thread.run` /
  `Thread.run_streamed` input from a string to a
  `list[TextInput | LocalImageInput]` when images are present.
  `FileInput` parts append `Attached file: <path>` hints to the
  prompt text (Codex's working-directory sandbox lets the agent open
  them with its built-in shell tools).
- `_build_copilot_attachments` (Copilot) — translates both
  `ImageInput` and `FileInput` into Copilot's
  `FileAttachment` TypedDict (`{"type":"file","path":...}`), passed
  via `session.send_and_wait(prompt, attachments=...)`.
- `_build_claude_prompt` (Claude Code) — appends an `Attached files
  (use the Read tool to access):` block to the prompt text and adds
  `"Read"` to `ClaudeAgentOptions.allowed_tools`. The
  `_ensure_client` cache key grows a `has_attachments` fragment so a
  no-attachment ↔ with-attachment switch reconnects with the right
  `allowed_tools`.
- `Feature.VISION_INPUT` flipped True on **all four adapters**.
- `Feature.FILE_INPUT` flipped True on **Claude / Copilot / Codex**;
  stays False on OpenAI-compat (the roadmap calls out that file
  routing varies wildly across compat vendors).
- 22 new session-level unit tests across the four
  `tests/test_*_session.py` files pinning the per-adapter wire shape,
  the path-only constraint (bytes_/url decline), the
  `FileInput`-decline path on OpenAI-compat, and Claude's
  cache-invalidation-on-attachments-change semantics.
- 2 new feature-matrix tests pin `VISION_INPUT` as universal and
  `FILE_INPUT` as Claude/Copilot/Codex-only.

### Changed (Phase 2, Iteration C)

- All four adapters now call `_split_prompt_parts` instead of
  `_coerce_prompt_or_raise`; the old helper is kept exported for
  third-party adapters that haven't wired any input capability yet
  but is no longer used in-tree.
- `ClaudeCodeSession._ensure_client` takes a new
  `has_attachments: bool` kwarg and uses it both in the cache key
  and to set `ClaudeAgentOptions.allowed_tools=["Read"]`.
- `CopilotAgentSession.execute` / `.stream` now pass
  `attachments=...` to `CopilotSession.send_and_wait` (previously
  unused).
- `CodexAgentSession.execute` / `.stream` now build their `Thread.run`
  input via `_build_codex_input` rather than passing the prompt
  string directly.

### Added (Phase 2, Iteration D — wrap-up: bytes/URL + probes)

- `ImageInput(bytes_=)` and `ImageInput(url=)` support on
  **OpenAI-compat** (native). `_build_user_content` now handles all
  three variants: ``path=`` reads + base64-encodes, ``bytes_=``
  base64-encodes inline (defaults media type to ``image/png`` when
  omitted), ``url=`` passes through as-is for the vendor to fetch.
- `ImageInput(bytes_=)` support on **Copilot** via
  :class:`BlobAttachment` (``{"type":"blob","data":<b64>,
  "mimeType":...}``). `_build_copilot_attachments` now routes
  ``path=`` → `FileAttachment` and ``bytes_=`` → `BlobAttachment`.
  ``url=`` still raises — Copilot's SDK has no URL channel; the
  error message recommends fetching the image and passing
  ``bytes_=`` instead.
- Sharper bytes/URL decline message on **Claude** and **Codex**:
  both now point the consumer to ``tempfile.NamedTemporaryFile`` +
  ``path=`` as the workaround. Neither vendor's SDK has a native
  bytes/URL channel for vision (Claude uses the Read tool which
  reads from disk; Codex uses `LocalImageInput` which is path-only).
- `examples/probe_thinking.py` — live-vendor probe for ``thinking=``.
  Multi-provider, default ``--effort high``, reports
  ``reasoning_tokens`` from the resulting `CostRecord`.
- `examples/probe_vision.py` — live-vendor probe for `ImageInput`.
  Defaults to OpenAI-compat (the only adapter that natively handles
  every variant); takes ``--variant path|bytes|url`` and bundles a
  67-byte inline 1×1 PNG so the probe runs without external assets.
- 8 new session-level unit tests across `test_openai_compatible_session.py`
  and `test_copilot_session.py` covering the bytes / URL paths.

### Changed (Phase 2, Iteration D)

- `_split_prompt_parts` no longer rejects `ImageInput(bytes_=)` /
  `ImageInput(url=)`. Per-vendor variant support differs, so the
  shared helper only gates the feature *category* (vision vs file);
  each adapter's content builder decides which variants it can
  natively serve.
- Iteration C's "path-only in v0" tests on OpenAI-compat and Copilot
  were replaced with positive tests for the new variants. The
  remaining "path-only" assertions on Claude and Codex now also
  assert the error message includes the ``tempfile`` /  ``path=``
  workaround hint.

### Phase 1 — AgentSession + streaming + cancel

The "hinge" phase from the
[implementation plan](docs/implementation-plan.md) — introduces
`AgentSession`, streaming, cancellation, and session resume. Shipped
in iterations A–H over the development cycle; bundled here as part
of the v0.5.0 release.

### Added (Phase 1, Iteration H — wrap-up: probes + architecture refresh)

The last Phase 1 deliverables. No more SDK glue, no more API
changes; just docs + examples that exercise everything the prior
iterations shipped.

- `examples/probe_streaming.py` — live-vendor probe for
  `session.stream()`. Iterates `async for event in
  sess.stream(prompt)`, prints `TextDelta` content live,
  `ReasoningDelta` on a dedicated line, and a summary of the
  trailing `TurnComplete`. Defaults to `claude` but accepts
  `--provider {claude,github-copilot,codex,opencode}`.
- `examples/probe_session_resume.py` — two-turn probe for the
  resume path. Opens a session, captures `session.id` after the
  first turn, closes; reopens with `resume=<id>` and verifies the
  second turn shows continuity (the model recalls a "secret word"
  from the prior turn). Rejects OpenAI-compat with a clear error
  since chat-completions has no server-side session.
- `docs/architecture.md` refreshed for the runtime/session split:
  new staircase diagram showing both protocols + adapter layer,
  runtime-vs-session ownership table, streaming `RuntimeEvent`
  taxonomy, end-of-Phase-1 capability matrix, per-adapter
  operational notes updated to reflect the bespoke sessions, and
  the new `unwrap()` split (runtime-level vs session-level types).

### Changed (Phase 1, Iteration G — `execute()` is sugar over `session()`)

**Breaking** — but breaking pre-1.0 in service of the cleaner shape
the plan called for in Phase 1.

- `AgentRuntime.execute(prompt, ...)` is now documented sugar for
  `runtime.session(system=..., model=...).execute(prompt, schema=...,
  timeout=...)` + `close()`. Single-turn, ephemeral. Consumers wanting
  context warmth across calls open a session explicitly and reuse it.
- **Behaviour change:** the runtime no longer caches per-conversation
  state across consecutive `execute()` calls. Each call opens a fresh
  session, runs one turn, and tears it down. For SDK-based adapters
  that spawn a CLI subprocess per session (Claude Code, Copilot,
  Codex), this means a fresh subprocess per `execute()`. Consumers
  that relied on the previous warm-cache behaviour should migrate to
  the session API:

  ```python
  # before — execute() reused state across calls
  for prompt in prompts:
      result = await runtime.execute(prompt, system=PROMPT)
  ```

  ```python
  # after — explicit session reuses the underlying SDK handle
  sess = runtime.session(system=PROMPT)
  try:
      for prompt in prompts:
          result = await sess.execute(prompt)
  finally:
      await sess.close()
  ```

- `AgentSession.unwrap(cls)` added to the protocol. Vendor session
  types moved off the runtime onto the session:
  - `ClaudeCodeSession.unwrap(ClaudeSDKClient)` — was
    `ClaudeCodeRuntime.unwrap(ClaudeSDKClient)`.
  - `CopilotAgentSession.unwrap(CopilotSession)` — was
    `CopilotRuntime.unwrap(CopilotSession)`.
  - `CodexAgentSession.unwrap(Thread)` — was
    `CodexRuntime.unwrap(Thread)`.
  Runtime-level vendor types (`CopilotClient`, `Codex`, `AsyncOpenAI`)
  stay on the runtime — those are runtime-wide, long-lived.
- `runtime.unwrap(SessionType)` now raises with a redirect message
  pointing callers to `session.unwrap(SessionType)`.
- `runtime.reset()` is a no-op on every adapter — there's nothing
  scope-bound to drop on the runtime any more. Kept for protocol
  completeness and back-compat.
- `ClaudeCodeRuntime` is now genuinely sessionless: `close()` is also
  a no-op (no SDK client, no HTTP client, no subprocess to release).
- `CopilotRuntime.close()` still releases the long-lived
  `CopilotClient` (runtime-owned).
- `CodexRuntime.close()` still drops the cached `Codex` reference
  (cheap, but back-compat with pre-G behaviour).

### Removed (Phase 1, Iteration G)

- `ClaudeCodeRuntime._ensure_client`, `_query_and_drain` — the per-
  conversation logic moved into `ClaudeCodeSession`.
- `CopilotRuntime._ensure_session`, `_on_event`, runtime-level
  capture slots (`_captured_payload`, `_captured_usage`,
  `_captured_error`, `_last_assistant_message`, `_session`,
  `_session_key`) — all moved into `CopilotAgentSession`.
- `CodexRuntime._ensure_thread`, `_thread`, `_thread_key` — all
  moved into `CodexAgentSession`.

### Added (Phase 1, Iteration G)

- `AgentSession.unwrap(cls)` on the protocol with full docstring
  covering per-adapter mappings.
- `_ThinAgentSession.unwrap(cls)` — identity-cast only (no vendor
  state); raises with a helpful message pointing callers to
  `runtime.unwrap()` for vendor objects.
- `tests/test_unwrap.py` extended with three new session-level
  pre-execute tests covering the new vendor-type unwrap path
  through the session.

### Added (Phase 1, Iteration F — Codex end-to-end)

Fourth and final per-vendor session. All four in-tree adapters now
have bespoke `AgentSession` implementations; the
`_ThinAgentSession` placeholder is no longer reachable from any
adapter's `session()` factory (kept in `airframe.sessions` for
third-party adapters that want a quick start).

- `airframe.adapters.codex.CodexAgentSession` — bespoke
  `AgentSession` owning one `Thread` for its lifetime.
  `system` / `model` / `resume` are session-fixed and baked into
  `Codex.start_thread` (or `resume_thread`) at first use. Schema can
  vary per turn — the Codex SDK puts `outputSchema` on `TurnOptions`
  (not `ThreadOptions`), so the same thread serves both plain-text
  and structured turns without rebuild.
- Real streaming via `Thread.run_streamed`. Per-item tail tracking
  on `ItemUpdatedEvent` / `ItemCompletedEvent` keeps `TextDelta`
  instances appendable (concatenated deltas reconstruct the full
  message text, matching the contract OpenAI-compat's chunk-stream
  sets). `AgentMessageItem` → `TextDelta`; `ReasoningItem` →
  `ReasoningDelta`; `TurnCompletedEvent` → trailing `TurnComplete`
  with cost from `Usage`; `TurnFailedEvent` raises through the
  runtime's classifier.
- Native session resume — ``session(resume=<thread_id>)`` forwards
  the ID to `Codex.resume_thread`; `AgentSession.id` is seeded from
  `resume=` and updated from `Thread.id` after the first turn (live
  `thread.started` event surfaces the id on fresh threads).
- Native cancellation via per-turn `AbortController` plumbed into
  `TurnOptions.signal`. `cancel()` calls `controller.abort()`; the
  awaiting turn raises `AbortError`, surfaced as
  `RuntimeCancelledError`. Honours no-op-when-idle via `_in_flight`.
- `Feature.STREAMING`, `Feature.SESSION_RESUME`, and `Feature.CANCEL`
  flipped to `True` on `CodexRuntime.SUPPORTED_FEATURES`. With this,
  every in-tree adapter declares streaming + cancel; the three
  SDK-based adapters (Claude Code, Copilot, Codex) all declare
  session resume; OpenAI-compat is the only adapter without
  `SESSION_RESUME` (chat-completions has no server-side session).
- 17 new unit tests in `tests/test_codex_session.py` covering the
  factory, capability surface, resume-id seeding + `resume_thread`
  routing, fresh thread via `start_thread`, thread reuse across
  turns, schema-can-vary-without-rebuild, system-prompt
  concatenation, appendable streaming deltas, reasoning deltas,
  structured-output parsing during streaming, cancellation in
  execute + mid-stream, `TurnFailedEvent` raising, and `close()`
  lifecycle.

### Changed (Phase 1, Iteration F)

- `tests/test_features.py` extended: replaced
  `test_codex_still_has_only_structured_output` with
  `test_codex_declares_streaming_resume_and_cancel`, and added
  `test_three_sdk_adapters_declare_session_resume` to pin the
  end-of-Phase-1 capability matrix (the three SDK adapters declare
  resume; OpenAI-compat doesn't).

### Added (Phase 1, Iteration E — Copilot end-to-end)

Third per-vendor session. Copilot now flips the same three feature
bits Claude does — streaming, native server-side resume, and
cooperative cancellation are all native to the SDK.

- `airframe.adapters.copilot.CopilotAgentSession` — bespoke
  `AgentSession` owning a `CopilotSession` for its lifetime.
  `system` / `model` / `resume` are session-fixed and baked into
  `CopilotClient.create_session` (or `resume_session`) at creation;
  schema can vary per turn — the session is destroyed and rebuilt
  when the schema fingerprint changes since the `submit_result` tool
  is baked in at session-creation time.
- Real streaming via per-session `session.on(handler)` subscription
  pushing deltas through an `asyncio.Queue` that the generator
  drains:
  - `AssistantMessageDeltaData` → `TextDelta`.
  - `AssistantReasoningDeltaData` → `ReasoningDelta`.
  - Uses `loop.call_soon_threadsafe` since the SDK dispatches
    handlers off its own thread.
- Native session resume — ``session(resume=<session_id>)`` forwards
  the ID to `CopilotClient.resume_session`; `AgentSession.id` is
  seeded with the resume ID and updated to the live
  `CopilotSession.session_id` once the underlying session is built.
- Native cancellation via `CopilotSession.abort()`; honours the
  no-op-when-idle contract via an `_in_flight` flag.
- `Feature.STREAMING`, `Feature.SESSION_RESUME`, and `Feature.CANCEL`
  flipped to `True` on `CopilotRuntime.SUPPORTED_FEATURES`. Only
  `CodexRuntime` is left on the Phase 0 single-bit declaration.
- 15 new unit tests in `tests/test_copilot_session.py` covering the
  factory, capability surface, resume-id seeding + `resume_session`
  routing, `create_session` path for fresh sessions, plain-text +
  structured execute, client reuse across same-schema turns, rebuild
  on schema change, streaming TextDelta / ReasoningDelta extraction,
  `abort()` invocation on cancel, idle no-op, and `close()` lifecycle
  (destroys the vendor session but leaves the runtime's client alive).

### Changed (Phase 1, Iteration E)

- `tests/test_features.py` extended: added
  `test_copilot_declares_streaming_resume_and_cancel` pinning the new
  flips; renamed `test_copilot_and_codex_still_have_only_structured_output`
  to `test_codex_still_has_only_structured_output` since only Codex
  remains on the Phase 0 declaration.

### Added (Phase 1, Iteration D — Claude Code end-to-end)

Second per-vendor session. `ClaudeCodeRuntime` now flips three feature
bits at once — Claude is the most fully-featured of the four families
for session-shaped APIs (streaming, server-side resume, cooperative
cancellation are all native to the SDK).

- `airframe.adapters.claude_code.ClaudeCodeSession` — bespoke
  `AgentSession` owning a `ClaudeSDKClient` for its lifetime. ``system``
  / ``model`` / ``resume`` are session-fixed and baked into
  `ClaudeAgentOptions` at connect; ``schema`` can vary per turn — the
  client reconnects when the schema fingerprint changes since
  ``output_format`` is connect-time-bound.
- Real streaming via ``include_partial_messages=True``. The session
  translates Anthropic stream events into airframe events:
  - ``content_block_delta`` with ``text_delta`` → `TextDelta`.
  - ``content_block_delta`` with ``thinking_delta`` → `ReasoningDelta`.
  - Fallback: `TextBlock` content on `AssistantMessage` emits a
    `TextDelta` when StreamEvents didn't deliver text (older CLI
    versions / non-streaming content paths).
- Native session resume — ``session(resume=<session_id>)`` forwards
  the ID into `ClaudeAgentOptions.resume`; the SDK materialises the
  prior conversation from local-disk session store on connect.
  `AgentSession.id` is seeded with the resume ID immediately and is
  updated from each `ResultMessage.session_id` after every turn.
- Native cancellation via `ClaudeSDKClient.interrupt()` plus
  `asyncio.Task.cancel()` on the wrapping execute task. `cancel()`
  honours the no-op-when-idle contract by tracking an `_in_flight`
  flag.
- `Feature.STREAMING`, `Feature.SESSION_RESUME`, and `Feature.CANCEL`
  flipped to `True` on `ClaudeCodeRuntime.SUPPORTED_FEATURES`. Copilot
  + Codex still report `False` on all three until their iterations.
- 14 new unit tests in `tests/test_claude_code_session.py` covering
  the factory, capability surface, resume-id seeding, options forwarding,
  client reuse across same-schema turns, reconnect on schema change,
  streaming TextDelta / ReasoningDelta extraction, AssistantMessage
  fallback, cancellation (with `interrupt()` awaited), idle no-op,
  and `close()` lifecycle.

### Changed (Phase 1, Iteration D)

- `tests/test_features.py` divergence assertions extended: the
  `any_adapter_may_support` set now includes `SESSION_RESUME`;
  added `test_claude_code_declares_streaming_resume_and_cancel` and
  `test_copilot_and_codex_still_have_only_structured_output` to pin
  exactly which adapter declares what at the end of Iteration D.

### Added (Phase 1, Iteration C — OpenAI-compat end-to-end)

First per-vendor session implementation replacing the Iteration B
placeholder. The OpenAI-compatible family (`OpenCodeZenRuntime` today;
future Together / Groq / Fireworks / OpenRouter subclasses
automatically inherit) now serves real multi-turn conversations,
streaming, and cancellation.

- `airframe.adapters.openai_compatible.OpenAICompatibleSession` —
  bespoke `AgentSession` with a client-side `messages=[]` buffer
  accumulating user + assistant turns across calls. Failures
  (including cancellation) roll back the user message so retries
  send clean history.
- Real streaming via `stream=True` on `chat.completions.create()` plus
  `stream_options={"include_usage": True}` so the trailing
  `TurnComplete` carries a populated `CostRecord`. Per-chunk text
  becomes `TextDelta`; the model's `finish_reason` lands on the
  result.
- Cancellation: `session.cancel()` aborts both `execute()` (wrapping
  `asyncio.Task.cancel()`) and `stream()` (closes the underlying
  `AsyncStream` + sets a flag the generator checks between yields).
  In both cases the message buffer rolls back.
- `Feature.STREAMING` and `Feature.CANCEL` flipped to `True` on
  `OpenAICompatibleRuntime.SUPPORTED_FEATURES`. The other three adapter
  families still report `False` until each lands its own iteration.
- `session(resume=...)` raises `UnsupportedFeatureError` on
  chat-completions vendors per the plan (no server-side session).
  Subclasses backed by the Responses API can override.
- 15 new unit tests in `tests/test_openai_compatible_session.py`
  exercising multi-turn buffer, streaming chunk shape, structured-
  output round-trip during streaming, cancellation, and `close()`
  semantics — all with a mocked `openai` SDK.

### Changed (Phase 1, Iteration C)

- `tests/test_features.py::test_phase_0_only_structured_output_is_true`
  renamed to `test_unwired_features_stay_false` — the assertion now
  permits per-adapter divergence on features whose APIs have landed
  (STRUCTURED_OUTPUT_JSON_SCHEMA, STREAMING, CANCEL) while still
  guarding against any adapter declaring a not-yet-wired capability.
  Adds a companion `test_openai_compatible_declares_streaming_and_cancel`
  pinning the new flips for this family and asserting the other three
  still report `False`.
- `OpenAICompatibleRuntime.execute()` (legacy single-call entrypoint)
  now shares the `_build_response_format()` helper with the new
  session — same JSON-schema wire shape, no behavioural change.
- TCK contract `test_session_resume_not_implemented_until_feature_flips`
  now also accepts `UnsupportedFeatureError` (Iteration C's bespoke
  signal for "this capability will *never* land here") in addition to
  Iteration B's `NotImplementedError` ("hasn't been wired yet").

### Added (Phase 1, Iteration B — `session()` factory + thin sessions)

- `AgentRuntime.session(*, resume=None, system=None, model=None,
  provider_options=None) -> AgentSession` factory on the protocol.
  Every built-in adapter wires it. Iteration B picks
  single-active-session per runtime per ADR-004; concurrent sessions
  can be added later additively.
- `airframe.sessions._ThinAgentSession` — Phase 1 Iteration B's shared
  thin wrapper: forwards `execute()` to the underlying runtime,
  synthesises `stream()` as a single `TurnComplete`, no-ops `cancel()`
  when no turn is in flight, idempotent `close()`. Marked private —
  third-party adapters target `AgentSession` directly. Per-adapter
  bespoke sessions (with real per-vendor streaming, native session
  resume, native cancellation) replace this in Iteration C+ as each
  adapter's `Feature` bits flip on.
- `airframe.UnsupportedFeatureError` — capability-decline signal.
  Phase 1+ companion to `UnsupportedBindingError`: a binding mismatch
  is "this adapter doesn't serve this `(provider, model)`"; an
  unsupported feature is "this adapter serves the binding, but the
  capability you asked for is not wired." Honours the implementation
  plan's "no silent fallbacks" principle. Carries an optional
  `feature` attribute naming the declined capability.
- Six new conformance contracts in `airframe.testing.contracts`:
  `test_session_factory_returns_agent_session`,
  `test_session_close_is_idempotent`,
  `test_session_close_on_fresh_session_is_safe`,
  `test_session_cancel_when_idle_is_noop`,
  `test_session_stream_is_async_generator`,
  `test_session_resume_not_implemented_until_feature_flips`. Every
  in-tree conformance file imports them; third-party adapters get the
  same coverage automatically.

### Changed (Phase 1, Iteration B)

- `AgentSession.cancel()` docstring tightened: when a turn is in
  flight, adapters not declaring `Feature.CANCEL` raise
  `UnsupportedFeatureError` (was: "raise `RuntimeProtocolError`",
  which conflated capability gaps with adapter bugs). When no turn is
  in flight, the call remains a no-op regardless of capability.

### Phase 1 status

Phase 1 is done. Per the [implementation
plan](docs/implementation-plan.md):

- Protocol surface (`AgentSession`, `RuntimeEvent`,
  `runtime.session()`, `session.unwrap()`) — ✓
- Per-adapter bespoke sessions (Claude Code, Copilot, Codex,
  OpenAI-compat) with real streaming, resume, and cancellation
  wired per-vendor — ✓
- `Feature.STREAMING` / `Feature.SESSION_RESUME` / `Feature.CANCEL`
  flipped on every adapter that supports them — ✓
- `AgentRuntime.execute()` refactored into sugar over `session()` — ✓
- TCK conformance contracts extended with the session-level surface — ✓
- Live-vendor probes (`probe_streaming.py`, `probe_session_resume.py`)
  + architecture-doc refresh — ✓

Next: Phase 2 (inputs & reasoning) — `thinking=` kwarg on
`session.execute()`, polymorphic `prompt` for vision / file inputs,
populated `CostRecord.reasoning_tokens`.

### Added (Phase 1, Iteration A — protocol scaffolding)

- `airframe.events` module — five frozen dataclasses (`TextDelta`,
  `ReasoningDelta`, `ToolCallStart`, `ToolCallResult`, `TurnComplete`)
  and the `RuntimeEvent = TextDelta | ReasoningDelta | ToolCallStart
  | ToolCallResult | TurnComplete` discriminated union. ADR-003 shape
  lock — the variant set and field shapes are public surface and
  snapshotted in `tests/test_events.py`. Adding a variant later is
  safe (consumers branch with a wildcard); renaming or removing is
  a major-version break.
- `AgentSession` Protocol in `airframe.protocol` — the multi-turn
  conversation handle scoped to one runtime. Declares
  `id: str | None`, `execute()`, `stream()`, `cancel()`, `close()`.
  ADR-004 picks single-active-session per runtime: ``runtime.session()``
  returning a second handle before the first is ``close()``'d
  invalidates the first. Going concurrent later is additive; going
  from concurrent to single is breaking, so Phase 1 picks single.
- New top-level exports: `AgentSession`, `RuntimeEvent`, `TextDelta`,
  `ReasoningDelta`, `ToolCallStart`, `ToolCallResult`,
  `TurnComplete`.

### Deferred (Iteration A — landed or moved out in Iteration B)

The factory, the structural TCK additions, and the `_ThinAgentSession`
landed in Iteration B above. Real per-vendor streaming, cancellation,
session-resume, the `execute()`-as-sugar refactor, and the two new
probe examples all moved to Iteration C+; the architecture-doc
refresh follows when those land.

## [0.3.0] — 2026-05-16

Phase 0 of the [implementation plan](docs/implementation-plan.md):
foundations that lock the shape of every extension point Phases 1–6
will use, plus a v0 contract-gap fix that wires plain-text
`execute(schema=None)` across every built-in adapter.

### Fixed

- Plain-text `execute(schema=None)` is honoured on every built-in
  adapter, closing a v0 contract gap. The
  `AgentRuntime.execute()` docstring has always promised
  "`None` means plain text — text answer on `RuntimeResult.text`,
  `structured=None`," but `ClaudeCodeRuntime`, `CopilotRuntime`, and
  `CodexRuntime` refused with `NotImplementedError`.
  (`OpenAICompatibleRuntime` / `OpenCodeZenRuntime` were already
  correct.) Per-adapter wiring:
  - `ClaudeCodeRuntime` — omits `output_format` from
    `ClaudeAgentOptions` when `schema is None`;
    `ResultMessage.result` carries the final assistant text. Cache
    key uses a `"__plain_text__"` sentinel so plain-text and
    structured sessions don't collide on `(model, system)`.
  - `CopilotRuntime` — doesn't register the `submit_result` tool
    when `schema is None`; doesn't prepend the forced-tool prefix
    to the system message; caller-supplied `system=` passes
    through verbatim.
  - `CodexRuntime` — omits `outputSchema` from `TurnOptions` when
    `schema is None`; `turn.final_response` is the free-form text.
    Empty `final_response` is a legitimate outcome rather than a
    structured-output violation.

  Motivation: a downstream consumer codebase (Maverick) just
  migrated five long-running personas onto airframe, each of
  which grew a single-field Pydantic schema purely to satisfy the
  `schema is None` gate. With the gate gone, those wrappers
  vanish and personas call
  `runtime.execute(prompt, system=PERSONA_SYSTEM_PROMPT)` directly.

### Added

- `airframe.Feature` enum — capability-negotiation predicates
  modelled on JDBC `DatabaseMetaData.supportsXxx()` and SQLAlchemy
  `Dialect.supports_*`. Ships the whole forward-looking set (19
  members covering Phases 1–6); later phases flip `True` bits onto
  each adapter without adding new members. Enum string values are
  public surface and snapshotted in tests.
- `AgentRuntime.supports(feature, model=None) -> bool` — pure,
  cheap lookup against an adapter's `SUPPORTED_FEATURES` class set.
  Today every in-tree adapter declares only
  `Feature.STRUCTURED_OUTPUT_JSON_SCHEMA`; Phase 1+ adds more.
- `AgentRuntime.unwrap(cls) -> cls` — JDBC-`Wrapper`-style escape
  hatch to the native SDK object. Each adapter accepts
  `unwrap(type(self))` plus its native types
  (`ClaudeCodeRuntime.unwrap(ClaudeSDKClient)`,
  `CopilotRuntime.unwrap(CopilotClient | CopilotSession)`,
  `CodexRuntime.unwrap(Codex | Thread)`,
  `OpenAICompatibleRuntime.unwrap(AsyncOpenAI)`). Unsupported types
  raise `TypeError`.
- `airframe.options` module with four empty frozen dataclasses
  (`ClaudeOptions`, `CopilotOptions`, `CodexOptions`,
  `OpenAICompatOptions`) and a `ProviderOptions` tagged-union alias.
  Vercel-style namespaced extension points for the
  `ClaudeAgentOptions`-sprawl problem; Phase 2+ populates each as
  features land. Not wired anywhere in v0.3.0 — Phase 1 attaches them
  to `runtime.session(provider_options=...)`.
- `CostRecord.reasoning_tokens: int` (default `0`) — canonicalises
  hidden reasoning / extended-thinking tokens across vendors. Phase 2
  wires real population from each SDK's native counter.
- Third-party adapter discovery via the `airframe.adapters`
  entry-point group. Third-party packages declare their runtime in
  `pyproject.toml` and `airframe.list_providers()` picks it up
  automatically. Modelled on SLF4J's `ServiceLoader` binding
  discovery; same pip-extras filtering applies. Built-ins shadow
  third-party entries with the same `PROVIDER_ID`. Malformed plugins
  (import errors, missing `PROVIDER_ID`, collisions) log a warning
  and are skipped — discovery never crashes.
- `discovery.ENTRY_POINT_GROUP = "airframe.adapters"` constant —
  public surface, snapshotted in tests.
- `airframe.testing` submodule with shared conformance contracts.
  Adapter authors import test functions from
  `airframe.testing.contracts` and provide an `adapter_runtime`
  pytest fixture; pytest collects the imported tests under the local
  fixture. SQLAlchemy `testing.suite` pattern. Ten structural
  contracts covering `close()` idempotency,
  `unwrap(type(self))` returning self, `supports()` purity,
  `validate_binding` behaviour, and the plain-text `execute()`
  path being wired
  (`test_plain_text_execute_path_is_wired`).
- New `[testing]` pyproject extra brings pytest + pytest-asyncio for
  adapter authors running the conformance suite. The main package
  does not depend on pytest.
- Per-adapter conformance test files (`tests/test_*_conformance.py`)
  for all four in-tree adapters — canonical examples for third-party
  authors.
- `examples/probe_supports.py` — capability matrix probe printing
  the live Feature × adapter truth table.
- `docs/feature-roadmap.md` — feature audit of the four native SDKs
  airframe wraps, prioritised list of cross-vendor features, and
  abstraction patterns borrowed from JDBC / SLF4J / OpenTelemetry /
  SQLAlchemy / JAAS.
- `docs/implementation-plan.md` — phased plan (Phase 0–6) ordered
  by dependency, with version targets, irreversible-shape-lock
  gating points, and adapter migration tables.

### Changed

- `AgentRuntime` protocol now requires two additional methods:
  `supports()` and `unwrap()`. Built-in adapters all implement them;
  external implementations of the protocol (none known) would need
  to add them.
- `CostRecord` constructor accepts an additional optional kwarg
  (`reasoning_tokens=0`); the structured-log dict via `to_dict()`
  always includes the key.

### Deferred

- Behavioural conformance contracts (401 → `RuntimeAuthError`,
  schema= round-trip, `input_tokens > 0` on a successful call,
  plain-text round-trip returns non-empty `text`) require live
  vendor credentials and naturally co-locate with Phase 1
  streaming/multi-turn integration test infrastructure. The
  existing `examples/probe_*.py` scripts already exercise these
  end-to-end against real vendors.

  > Resolved (post-Phase-5, v1.0-readiness pass):
  > `airframe.testing.integration` now ships. See the
  > `[Unreleased]` block at the top of this file.

## [0.2.0] — 2026-05-16

Model discovery API + an OpenAI-compatible base class. The headline
addition is `list_models()`: every adapter can now enumerate the
models a UI consumer can pick from, against the user's resolved
credentials. The headline change is install-state gating: which
providers `list_providers()` returns now reflects which pip extras
the consumer installed, so menus stay honest.

### Added

- `AgentRuntime.list_models() -> list[ModelInfo]` — live model
  discovery against the vendor's models endpoint. Driving UI menus
  is the expected use case; the call requires auth + network, so a
  failure is the consumer's signal to surface the issue before
  letting the user pick a model that would later fail to execute.
- `ModelInfo` — `id`, `display_name`, `provider_id`, optional
  `context_window`, `pricing_input_per_1k_usd`,
  `pricing_output_per_1k_usd`, capability flags, vendor-raw payload.
- Capability constants: `CAPABILITY_VISION`, `CAPABILITY_TOOLS`,
  `CAPABILITY_STRUCTURED_OUTPUT`, `CAPABILITY_STREAMING`,
  `CAPABILITY_REASONING_EFFORT`.
- `airframe.list_providers(installed_only=True)` — sorted list of
  provider IDs. Default filters to providers whose vendor SDK is
  importable (so `pip install airframe-agents[copilot]` users see
  `["github-copilot"]`, not the full menu). Pass
  `installed_only=False` for the full registry.
- `airframe.runtime_for(provider_id)` — returns the adapter class
  for a canonical provider ID. Raises `ValueError` for unknown IDs;
  raises `ImportError` (with a `pip install airframe-agents[...]`
  hint) when the adapter exists but its SDK isn't installed.
- `OpenAICompatibleRuntime` base — shared HTTP / `response_format` /
  `list_models` / error classification for OpenAI-compatible
  vendors. Subclasses (Zen today; Together / Groq / Fireworks /
  OpenRouter possible) ship a `PROVIDER_ID`, default base URL,
  default model, per-model metadata table, and a vendor-specific
  `_resolve_api_key()` hook — typically ~30 lines.
- `examples/probe_list_models.py` (committed as
  `tests/probe_list_models.py`) — end-to-end menu probe against every
  installed adapter; tabular output with context window / pricing /
  capabilities per model.
- `ClassVar` declarations on every adapter: `PROVIDER_ID`,
  `REQUIRES_PACKAGE`, `EXTRA_NAME` — drive discovery / install
  gating / error messages.

### Changed (breaking)

- **Provider IDs are strict.** Aliases dropped; each adapter has one
  canonical `PROVIDER_ID`. Update consumer code that referenced the
  old strings:
  - `ClaudeCodeRuntime`: was `{"anthropic", "claude", "claude-code"}`,
    now `"claude"`. The string `"anthropic"` is reserved for a
    future direct-API `AnthropicRuntime`.
  - `CopilotRuntime`: was `{"github-copilot", "copilot", "github"}`,
    now `"github-copilot"`.
  - `CodexRuntime`: was `{"openai", "codex"}`, now `"codex"`. The
    string `"openai"` is reserved for a future direct-API
    `OpenAIRuntime`.
  - `OpenCodeZenRuntime`: was `{"opencode-zen", "opencode"}`, now
    `"opencode"`.
- **The `[opencode-zen]` extra was renamed to `[openai-compat]`** —
  one extra for the whole OpenAI-compatible family (Zen today; room
  for siblings without exploding the extras matrix). All siblings
  share the `openai>=1.50,<3` SDK, so per-child extras would be a
  fiction. Update install commands:
  `pip install airframe-agents[opencode-zen]` →
  `pip install airframe-agents[openai-compat]`.
- `ClaudeCodeRuntime` now uses the Claude Agent SDK's native
  `output_format={"type": "json_schema", ...}` instead of the
  forced `submit_result` MCP tool. The SDK enforces the schema
  server-side; the validated payload lands on
  `ResultMessage.structured_output`. (Already shipped in 0.1.x via
  `479b576`; called out here because consumers using the lower-level
  capture surface would notice.)
- `CostRecord.provider_id` now reflects the canonical `PROVIDER_ID`
  (e.g. `"claude"`, `"codex"`, `"github-copilot"`) rather than
  vendor-name strings (`"anthropic"`, `"openai"`).

### Migration

```python
# Before (0.1.x)
ProviderModel("anthropic", "claude-haiku-4-5")
ProviderModel("openai", "gpt-5-codex")
ProviderModel("copilot", "gpt-5-mini")
ProviderModel("opencode-zen", "gpt-5-nano")

# After (0.2.0)
ProviderModel("claude", "claude-haiku-4-5")
ProviderModel("codex", "gpt-5-codex")
ProviderModel("github-copilot", "gpt-5-mini")
ProviderModel("opencode", "gpt-5-nano")
```

## [0.1.0] — 2026-05-15

Initial release. Extracted from the Maverick project after stabilising
through five migration phases.

### Added

- `AgentRuntime` protocol with `execute / reset / close /
  validate_binding`.
- `RuntimeResult`, `CostRecord`, `ProviderModel` data types.
- Vendor-agnostic error hierarchy: `AgentRuntimeError` base plus
  `RuntimeAuthError`, `RuntimeModelNotFoundError`,
  `RuntimeTransientError`, `RuntimeStructuredOutputError`,
  `RuntimeContextOverflowError`, `RuntimeProtocolError`,
  `RuntimeServerStartError`, `RuntimeCancelledError`.
- Four adapters:
  - `ClaudeCodeRuntime` via `claude-agent-sdk`. Claude Max
    subscription / OAuth / API-key auth. Forced `submit_result`
    MCP tool for structured output.
  - `CopilotRuntime` via `github-copilot-sdk`. GitHub Copilot
    subscription. Rejects `claude-*` model IDs (Phase 0 finding:
    Claude on Copilot can't honour tool calls).
  - `CodexRuntime` via `openai-codex-sdk`. ChatGPT Plus
    subscription / `OPENAI_API_KEY`. Native JSON-schema mode (no
    tool-forcing required).
  - `OpenCodeZenRuntime` via the `openai` SDK pointed at
    `https://opencode.ai/zen/v1`. Direct HTTP; no subprocess.
- Optional dependency extras: `[claude]`, `[copilot]`, `[codex]`,
  `[opencode-zen]`, `[all]`.
- 115 unit tests; 4 end-to-end probe scripts under `examples/`.
