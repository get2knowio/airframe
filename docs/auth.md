# Authentication

How airframe-agents resolves credentials per adapter, in one place.
Every adapter has a chain — checked in order, falling through to
the next when the previous isn't set — so the same code works
across CI (env vars), local dev (vendor credential files), and
production (explicit constructor argument).

## Quick reference

| Adapter | Chain (first match wins) | Pip extra |
|---|---|---|
| **ClaudeCodeRuntime** | `CLAUDE_CODE_OAUTH_TOKEN` → `~/.claude/.credentials.json` → `ANTHROPIC_API_KEY` | `airframe-agents[claude]` |
| **CopilotRuntime** | `github_token=` constructor arg → `GITHUB_TOKEN` env → `GH_TOKEN` env → `use_logged_in_user=True` (`gh auth login` storage) | `airframe-agents[copilot]` |
| **CodexRuntime** | `api_key=` constructor arg → `OPENAI_API_KEY` env → `CODEX_API_KEY` env → `~/.local/share/opencode/auth.json::openai.key` → implicit `~/.codex/auth.json` (CLI-managed) | `airframe-agents[codex]` |
| **OpenCodeZenRuntime** | `api_key=` constructor arg → `OPENCODE_API_KEY` env → `~/.local/share/opencode/auth.json::opencode.key` | `airframe-agents[openai-compat]` |
| **OpenCodeGoRuntime** | `api_key=` constructor arg → `OPENCODE_API_KEY` env → `~/.local/share/opencode/auth.json::opencode-go.key` | `airframe-agents[openai-compat]` |
| **OpenRouterRuntime** | `api_key=` constructor arg → `OPENROUTER_API_KEY` env | `airframe-agents[openai-compat]` |
| **BedrockRuntime** | explicit `aws_access_key_id`/`secret`/`session_token` → `AWS_ACCESS_KEY_ID`+`AWS_SECRET_ACCESS_KEY` env → `AWS_PROFILE` env → default boto3 chain (IAM instance / ECS task / Lambda / IRSA) | `airframe-agents[bedrock]` |

`list_models()` calls always require a credential — the vendor's
models endpoint won't honour an anonymous request. Tests / scripts
that don't intend to call live vendors should mock at the adapter
boundary instead of supplying fake credentials.

## ClaudeCodeRuntime

```python
from airframe import ClaudeCodeRuntime
runtime = ClaudeCodeRuntime()  # resolves auth lazily on first call
```

Three sources, checked in order at `execute()` time:

1. **`CLAUDE_CODE_OAUTH_TOKEN` env var** — long-lived OAuth token
   minted by `claude setup-token`. Best for CI / non-interactive
   contexts; the value is opaque to airframe and forwarded
   verbatim to the Claude Agent SDK.
2. **`~/.claude/.credentials.json`** — the interactive Claude Code
   OAuth flow's stored token. Created automatically when you run
   `claude login` (or sign in via the Claude desktop app). Best
   for local dev on a developer's machine.
3. **`ANTHROPIC_API_KEY` env var** — pay-per-token API access via
   Anthropic's standard API key. Works for production deployments
   without a Claude Max subscription.

### Notes

- OAuth tokens (sources 1 and 2) drive subscription-billed access.
  API keys (source 3) drive metered API billing — same model
  catalogue, different billing path.
- `list_models()` **only works with `ANTHROPIC_API_KEY`** — Anthropic's
  `/v1/models` endpoint doesn't accept OAuth bearer tokens. If
  your auth chain resolves to an OAuth path, `list_models()` raises
  `RuntimeAuthError` with an actionable message.
- Provide an override at construction:
  `ClaudeCodeRuntime(api_key="sk-ant-...")` (the parameter is
  `api_key=` even though the source is named `ANTHROPIC_API_KEY`).

## CopilotRuntime

```python
from airframe import CopilotRuntime
runtime = CopilotRuntime()  # uses gh-CLI auth by default
runtime_explicit = CopilotRuntime(github_token="ghp_...")
```

Three sources, checked in order at `execute()` time:

1. **`github_token=` constructor arg** — explicit GitHub PAT.
   Highest precedence.
2. **`GITHUB_TOKEN` env var** — typically set in GitHub Actions
   automatically. Falls back to **`GH_TOKEN`**, the official `gh`
   CLI env var alias.
3. **`use_logged_in_user=True`** (the default when no explicit
   token is supplied) — the SDK picks up the OAuth credentials
   stored by `gh auth login`. Best for local dev.

### Notes

- The token needs Copilot scope on the GitHub account. PATs without
  Copilot access raise `RuntimeAuthError` on first call.
- `cli_path=` overrides the Copilot CLI binary path; honours
  `COPILOT_CLI_PATH` env var.
- The Copilot CLI does not serve Claude models — `validate_binding()`
  returns False for any `model_id` starting with `claude-`. Route
  Claude work through `ClaudeCodeRuntime` instead.

## CodexRuntime

```python
from airframe import CodexRuntime
runtime = CodexRuntime()  # env or opencode auth or codex-CLI auth
runtime_explicit = CodexRuntime(api_key="sk-proj-...")
```

Four sources, checked in order:

1. **`api_key=` constructor arg** — explicit OpenAI API key.
   Exported as `CODEX_API_KEY` for the subprocess.
2. **`OPENAI_API_KEY` env var** — the standard OpenAI variable.
   Falls back to **`CODEX_API_KEY`**, the codex-CLI alias. The
   `openai-codex-sdk` subprocess inherits `os.environ`, so these
   work without explicit forwarding.
3. **`~/.local/share/opencode/auth.json`** — the API key minted by
   `opencode auth login openai` when the user already has opencode
   auth configured. Path override:
   **`OPENCODE_AUTH_PATH`** env var.
4. **Implicit fallback to `~/.codex/auth.json`** — populated by
   `codex login`. Airframe doesn't read this file directly; the
   `codex` CLI subprocess does. No work for us.

### Notes

- The implicit `~/.codex/auth.json` path means `CodexRuntime()`
  with no env vars **may still work** if the user has run
  `codex login` — the codex CLI subprocess reads its own credential
  file directly.
- `codex_path=` overrides the Codex CLI binary path; honours
  `CODEX_CLI_PATH` env var.
- `sandbox_mode=` controls the codex subprocess's filesystem
  sandbox (`"read-only"`, `"workspace-write"`, `"danger-full-access"`).
  Defaults to `"read-only"`. Lives at the runtime constructor, not
  in `CodexOptions`, because it's a security-relevant default that
  shouldn't vary per session.

## OpenCodeZenRuntime

```python
from airframe import OpenCodeZenRuntime
runtime = OpenCodeZenRuntime()  # env var or opencode auth.json
runtime_explicit = OpenCodeZenRuntime(api_key="opc_...")
```

Three sources, checked in order:

1. **`api_key=` constructor arg** — explicit Zen gateway key.
2. **`OPENCODE_API_KEY` env var** (shared with `OpenCodeGoRuntime`).
3. **`~/.local/share/opencode/auth.json::opencode.key`** —
   the per-token Zen key minted by `opencode auth login opencode`.
   Same auth file as `OpenCodeGoRuntime` reads, different slot
   inside it (the Go subscription key lives under `opencode-go`).

### Notes

- Unlike the three SDK adapters, `OpenCodeZenRuntime` is HTTP-only
  (no subprocess), so the credential never leaves the Python
  process.
- `base_url=` overrides the Zen endpoint URL; honours
  `OPENCODE_ZEN_BASE_URL` env var.

## OpenCodeGoRuntime

```python
from airframe import OpenCodeGoRuntime
runtime = OpenCodeGoRuntime()  # env var or opencode auth.json
runtime_explicit = OpenCodeGoRuntime(api_key="opc_...")
```

Three sources, checked in order:

1. **`api_key=` constructor arg** — explicit subscription key.
2. **`OPENCODE_API_KEY` env var** (shared with `OpenCodeZenRuntime`).
3. **`~/.local/share/opencode/auth.json::opencode-go.key`** — the
   subscription key minted by `opencode auth login opencode-go`.
   Same auth file as `OpenCodeZenRuntime`, distinct slot.

### Notes

- Picks the **subscription** gateway (`https://opencode.ai/zen/go/v1`),
  which serves the 14 bundled models for $0 per call at the caller's
  margin. For per-token access to GPT/Claude/Gemini, use
  `OpenCodeZenRuntime` instead.
- `base_url=` overrides the gateway URL; honours `OPENCODE_GO_BASE_URL`
  env var.
- HTTP-only — no subprocess, credential stays in-process.

## OpenRouterRuntime

```python
from airframe import OpenRouterRuntime
runtime = OpenRouterRuntime()  # picks up OPENROUTER_API_KEY
runtime_explicit = OpenRouterRuntime(api_key="sk-or-...")
```

Two sources, checked in order:

1. **`api_key=` constructor arg** — explicit OpenRouter key.
2. **`OPENROUTER_API_KEY` env var** — mint at
   [https://openrouter.ai/keys](https://openrouter.ai/keys).

### Notes

- **No on-disk auth-file convention.** OpenRouter doesn't have an
  equivalent of `~/.local/share/opencode/auth.json`. The auth chain
  ends at the env var; no filesystem fallback.
- `base_url=` overrides the gateway URL; honours `OPENROUTER_BASE_URL`
  env var (e.g. for self-hosted proxies).
- HTTP-only — no subprocess; credential stays in-process.
- The same `OpenAICompatibleRuntime` base class powers any future
  compat-vendor adapter (Together / Groq / Fireworks). Each
  subclass overrides `_resolve_api_key()` with its own env-var and
  credential-file chain — the base class doesn't impose one.

## BedrockRuntime

```python
from airframe import BedrockRuntime
# Explicit credentials.
runtime = BedrockRuntime(
    region_name="us-east-1",
    aws_access_key_id="AKIA…",
    aws_secret_access_key="…",
    aws_session_token="…",   # optional — for STS-issued creds
)
# Or pick everything up from env / profile / instance-role.
runtime = BedrockRuntime(region_name="us-east-1")
runtime = BedrockRuntime(profile_name="my-bedrock-profile")
```

### Credential chain (first match wins)

1. **Explicit constructor args.** `aws_access_key_id` +
   `aws_secret_access_key` (+ optional `aws_session_token`) win
   outright. Forwarded to `aioboto3.Session(...)` so boto3 builds
   the signing context directly from your values.
2. **`AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` (+ optional
   `AWS_SESSION_TOKEN`) env vars.** Standard AWS env-var auth.
   `aioboto3.Session()` picks these up natively when no explicit
   credentials are passed.
3. **`AWS_PROFILE` env var** → `~/.aws/credentials` /
   `~/.aws/config` profile resolution. Equivalent to passing
   `profile_name=` to the constructor.
4. **Default credential chain.** IAM instance profile (EC2), ECS
   task role, Lambda execution role, IRSA (EKS). `botocore` walks
   this when no explicit credentials or profile are set —
   airframe doesn't override.

### Region resolution (independent from credentials)

Bedrock is region-pinned and catalogs differ per region, so region
resolution gets first-class treatment separately from the credential
chain:

1. **Explicit `region_name=` constructor arg.**
2. **`AWS_REGION` env var.**
3. **`AWS_DEFAULT_REGION` env var.**
4. *(Not honoured by airframe today)* `~/.aws/config` `region` —
   `aioboto3.Session(...)` would resolve it but airframe raises
   `RuntimeAuthError` at the first network call rather than fall
   through, because silent fallback to a default region would route
   traffic to a different model catalog than the caller expects.

### Notes

- **Region is mandatory.** Missing region surfaces as
  `RuntimeAuthError` with a clear "set AWS_REGION" message. The
  honest signal beats a confusing "model not found" later.
- **Cross-account assume-role.** Not handled in-adapter. Run
  `aws sts assume-role` (or your SDK equivalent) ahead of time,
  export the session credentials, then construct `BedrockRuntime()`
  — it'll pick them up via the env-var path.
- **KMS-encrypted prompts.** boto3 honours whatever the resolved
  session's KMS config provides; airframe doesn't intervene.
- **Provisioned-throughput ARNs.** Pass the PT ARN as a
  `ProviderModel.model_id` (or via
  `BedrockOptions(inference_profile_arn=...)`); the adapter doesn't
  provision or report on PT.
- **`AIRFRAME_PROBE_MODEL_BEDROCK`** overrides the default model
  used by `examples/probe_bedrock.py` for region-sensitive testing.

## Cross-cutting notes

### Credential precedence in CI

In CI, prefer the env-var path on every adapter — credentials are
short-lived, scoped, and don't leave artifacts on disk between
jobs. Set:

```yaml
env:
  CLAUDE_CODE_OAUTH_TOKEN: ${{ secrets.CLAUDE_CODE_OAUTH_TOKEN }}
  GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}  # usually auto-set
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  OPENCODE_API_KEY: ${{ secrets.OPENCODE_API_KEY }}
```

The integration suite (`pytest -m integration`) skips itself when
no credential is available — set whichever vars you have and the
matching tests run.

### Credentials never leave airframe

`airframe-agents`:

- Reads vendor-owned credential files but never writes to them.
- Never logs raw credentials, even at DEBUG.
- Defers token validation to the underlying vendor SDK — if a
  token is malformed or revoked, the vendor's auth failure
  surfaces as `RuntimeAuthError` with the vendor's message.

See [SECURITY.md](../SECURITY.md) for the security policy and
disclosure workflow.

### Adapter construction is auth-free

All four runtimes can be constructed without credentials — auth
resolution defers to the first `execute()` / `list_models()` /
`session()` call that actually needs them. This means:

- Tests can construct adapters without environment setup (the
  in-tree unit suite does this 600+ times with mocked SDKs).
- `runtime.supports(Feature.X)` and `runtime.validate_binding(...)`
  work without credentials — pure capability lookups.
- Failing fast on missing credentials is the *caller's* choice via
  `try: await runtime.list_models() except RuntimeAuthError:` at
  startup.
