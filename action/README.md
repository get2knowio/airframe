# Airframe Agent — GitHub Action

Run an agent prompt against **any** Airframe-supported provider from one
reusable step, and pick the provider per workflow job. Cheap model to
triage an issue, a premium model to review a PR — same step, different
`provider:`.

```yaml
- uses: get2knowio/airframe/action@v1
  id: triage
  with:
    provider: opencode-zen
    api-key: ${{ secrets.OPENCODE_API_KEY }}
    prompt: "Triage issue #${{ github.event.issue.number }}"

- uses: get2knowio/airframe/action@v1
  id: review
  with:
    provider: claude
    api-key: ${{ secrets.ANTHROPIC_API_KEY }}
    prompt-file: review-prompt.md
```

The agent's reply is exposed as `steps.<id>.outputs.result`.

## How it works

The action is a thin composite wrapper: it provisions
[`uv`](https://github.com/astral-sh/setup-uv), installs
`airframe-agents` with just the extra the chosen provider needs, and
runs the [`airframe` CLI](../src/airframe/cli.py). All the
provider/model/auth logic lives in the library, not in YAML.

## Inputs

| input | required | default | notes |
|---|---|---|---|
| `provider` | ✅ | — | `claude`, `github-copilot`, `opencode-zen`, `opencode-go`, `openrouter`, `bedrock`, `kimi` (the `opencode` server adapter isn't usable here — see below) |
| `prompt` | — | — | inline prompt text |
| `prompt-file` | — | — | path to a prompt file (used when `prompt` is empty) |
| `model` | — | adapter default | model id to pin |
| `system` | — | — | system-prompt override |
| `format` | — | `text` | `text` (reply) or `json` (envelope with cost/finish/structured) |
| `timeout` | — | `600` | wall-clock budget, seconds |
| `api-key` | — | — | routed to the right env var by provider (see Auth) |
| `github-token` | — | `${{ github.token }}` | for `github-copilot` |
| `airframe-version` | — | latest | pin `airframe-agents` for reproducibility |
| `python-version` | — | `3.12` | Python uv provisions |

## Outputs

| output | description |
|---|---|
| `result` | assistant text, or the JSON envelope when `format: json` |

## Auth

Pass the secret as `api-key`; the action exports it under the variable
the chosen adapter reads:

| provider | credential |
|---|---|
| `claude` | `api-key` → `ANTHROPIC_API_KEY` |
| `opencode-zen`, `opencode-go` | `api-key` → `OPENCODE_API_KEY` |
| `openrouter` | `api-key` → `OPENROUTER_API_KEY` |
| `kimi` | `api-key` → `KIMI_API_KEY` |
| `github-copilot` | `github-token` (defaults to the workflow token) |
| `bedrock` | AWS credentials — configure in a prior step (e.g. `aws-actions/configure-aws-credentials`); the action inherits the `AWS_*` env |

> **Copilot caveat:** the Copilot SDK needs a token with Copilot
> entitlement. The default `GITHUB_TOKEN` may not have it; supply a PAT
> with Copilot access via `github-token` if the default fails.

## Examples

See [`examples/triage-and-review.yml`](examples/triage-and-review.yml)
for the full issue-triage + PR-review workflow.

## Scope

Today the action runs a prompt and returns text — your workflow decides
what to do with it (post a comment, open a PR, set an output). Giving
the agent direct repo/tool access (read files, post comments itself via
MCP / function tools) is a planned follow-up.

**Not supported:** the `opencode` provider. It wraps the local OpenCode
HTTP agent server (`opencode serve`), which the action doesn't stand up
on the ephemeral runner. Reach OpenCode through the gateway adapters
(`opencode-zen` / `opencode-go`) instead — they're plain HTTP and need
no local server.
