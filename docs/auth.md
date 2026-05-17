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
| **OpenCodeZenRuntime** | `api_key=` constructor arg → `OPENCODE_API_KEY` env → `~/.local/share/opencode/auth.json::opencode-zen.key` | `airframe-agents[openai-compat]` |

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
- The Copilot CLI **rejects Claude models** — `validate_binding()`
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
  `codex login` previously. That's by design — we let the CLI
  handle its own credential file.
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
2. **`OPENCODE_API_KEY` env var**.
3. **`~/.local/share/opencode/auth.json::opencode-zen.key`** —
   the key minted by `opencode auth login zen`. Same auth file
   as the Codex chain reads, different key path inside it.

### Notes

- Unlike the three SDK adapters, `OpenCodeZenRuntime` is HTTP-only
  (no subprocess), so the credential never leaves the Python
  process.
- `base_url=` overrides the Zen endpoint URL; honours
  `OPENCODE_ZEN_BASE_URL` env var.
- The same `OpenAICompatibleRuntime` base class powers any future
  compat-vendor adapter (Together / Groq / Fireworks). Each
  subclass overrides `_resolve_api_key()` with its own env-var and
  credential-file chain — the base class doesn't impose one.

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
