# ZaiAnthropicRuntime

Runs [Z.AI](https://z.ai)'s GLM models through the **Claude Agent SDK
harness**. Z.AI exposes an endpoint that speaks Anthropic's Messages wire
format, which is what lets the `claude` CLI drive it at all — so this
adapter reuses [`ClaudeCodeRuntime`](./claude.md)'s entire machinery
(subprocess lifecycle, sessions, streaming, tools, hooks) and changes only
what makes it a different binding.

| | |
|---|---|
| **PROVIDER_ID** | `zai-anthropic` |
| **Pip extra** | `airframe-agents[claude]` (shared with `claude`) |
| **Vendor SDK** | `claude-agent-sdk` + the `claude` CLI binary |
| **Transport** | `claude` subprocess aimed at `https://api.z.ai/api/anthropic` |
| **Authentication** | `ZAI_API_KEY` → `ANTHROPIC_AUTH_TOKEN` in the subprocess env |
| **Billing** | Z.AI coding-plan subscription — `cost_usd` reflects whatever the CLI reports |

## Why it isn't just `claude` with a base URL

You *can* point `ClaudeCodeRuntime` at Z.AI by exporting
`ANTHROPIC_BASE_URL`, and that keeps working as an escape hatch. It is not
a supported binding, for the reason spelled out in the binding rule in
`CLAUDE.md`: `supports()` is a ClassVar manifest, so one `PROVIDER_ID`
can carry exactly one honest capability surface. `claude` declares 22
features against Anthropic's endpoint; several of those are unverified
against Z.AI's. Sharing the ID would make `supports()` lie about one of
the two.

The ID is `zai-anthropic` rather than a bare `zai` because Z.AI also
exposes an OpenAI-compatible surface. That one belongs to a future
`zai-openai` sibling built on `OpenAICompatibleRuntime` — the same
reasoning that kept the Kimi Agent SDK adapter from claiming `moonshot`.

## Install

```bash
pip install airframe-agents[claude]
```

You also need the `claude` CLI on `PATH` — the Agent SDK spawns it. The
CLI does not need to be logged in to Anthropic; this adapter supplies its
own endpoint and credential.

## Quickstart

```python
from airframe import ProviderModel
from airframe.adapters.zai import ZaiAnthropicRuntime
from pydantic import BaseModel

class Brief(BaseModel):
    summary: str

runtime = ZaiAnthropicRuntime()  # picks up ZAI_API_KEY from env
result = await runtime.execute(
    "Brief me on the project structure.",
    schema=Brief,
    model=ProviderModel("zai-anthropic", "glm-4.6"),
)
print(result.structured)
await runtime.close()
```

## Authentication

Resolution order:

1. Explicit `api_key=` constructor argument.
2. `ZAI_API_KEY` env var.

There is deliberately **no fallback to Anthropic credentials**. An
Anthropic API key or subscription OAuth token authenticates a different
account at a different vendor; treating one as a stand-in is how a
subscription token ends up POSTed to a third party.

For the same reason, `_subprocess_env()` actively **shadows** any inherited
`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, and `CLAUDE_CODE_OAUTH_TOKEN`
with empty strings before setting the Z.AI credential. The Agent SDK merges
`ClaudeAgentOptions.env` *over* `os.environ` and offers no way to unset a
key, so shadowing is the only available mechanism. Without it, a developer
with `CLAUDE_CODE_OAUTH_TOKEN` exported in a shell profile would have the
CLI carry their Anthropic subscription token to Z.AI.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `ZAI_API_KEY` | — | Required. The Z.AI credential. |
| `ZAI_BASE_URL` | `https://api.z.ai/api/anthropic` | Endpoint override (proxies, staging). |
| `ZAI_MODEL_OVERRIDE` | `glm-4.6` | Default model when no binding is passed. |

`CLAUDE_MODEL_OVERRIDE` is **not** consulted — it belongs to the other
binding.

## Capability surface

Narrower than `claude`, and deliberately so. Features split into two
groups:

- **CLI-side** — streaming, session resume, cancel, function tools, MCP
  transports, permission callbacks, lifecycle hooks, budget caps, slash
  commands. These are implemented by the local `claude` CLI and are
  independent of which endpoint answers, so they are retained.
- **Endpoint-dependent** — extended thinking, vision, file input, token
  counting, rate-limit telemetry, request metadata. These depend on what
  Z.AI implements and are reported `False` **pending live verification**.

A `supports()` that overstates is worse than one that understates: a
consumer checking the predicate simply routes elsewhere, while one that
overstates produces a runtime failure the consumer was told could not
happen.

`STRUCTURED_OUTPUT_JSON_SCHEMA` is declared `True` despite being equally
unverified, because the conformance suite makes it the floor every
airframe adapter must meet. If it turns out Z.AI does not honour the CLI's
`--json-schema` flag, that is a blocker on the adapter rather than a bit
to flip off.

### Verifying the unverified flags

```bash
ZAI_API_KEY=... uv run python examples/probe_zai.py
```

Each check prints `PASS` / `FAIL` / `SKIP`. Promote any `PASS` out of
`_UNVERIFIED_FEATURES` in `src/airframe/adapters/zai.py`, then re-run
`make ci` — the conformance suite asserts the opposite branch for every
flag you flip, so a wrong promotion fails loudly.

## Known gaps

- **`count_tokens()` raises `UnsupportedFeatureError`.** Z.AI's
  Anthropic-compatible surface does not serve Anthropic's
  `/v1/messages/count_tokens` route; inheriting the parent's
  implementation would POST to a path that isn't there.
- **`list_models()` returns a static catalog.** There is no `/v1/models`
  discovery endpoint to query, so the two known GLM models come from an
  in-module table. Per-token pricing is `None` rather than invented,
  since the coding plan bills as a flat-fee subscription.
- **Cost telemetry is whatever the CLI reports.** `total_cost_usd` may
  come back `None` or `0.0` against a third-party backend; token counts
  are more likely to be populated than the dollar figure.
