# Security policy

## Reporting a vulnerability

If you believe you've found a security vulnerability in
`airframe-agents`, please report it privately rather than opening a
public issue.

Two options, in preference order:

1. **GitHub private security advisory** — preferred. Open
   [Security → Advisories → Report a vulnerability](https://github.com/get2knowio/airframe/security/advisories/new)
   on the airframe repository. GitHub coordinates the disclosure
   workflow and keeps the report off the public issue tracker.
2. **Email** — `paulofallon@gmail.com`. Include a clear description
   of the issue, the affected version(s), and (if possible) a
   minimal reproduction. We'll acknowledge within 5 business days
   and propose a coordinated-disclosure timeline.

Please do **not**:

- Open a public GitHub issue describing an unpatched vulnerability.
- Post details to PyPI, social media, or a public Slack/Discord
  before the coordinated-disclosure window closes.

## Scope

In scope:

- Bugs in `airframe-agents` (`src/airframe/`) that allow unintended
  access to credentials, the host filesystem, the network, or
  enable supply-chain attacks via the entry-point discovery
  mechanism.
- Issues in the conformance / integration test scaffolding that
  could leak credentials when third-party adapters run the suite.
- Documentation that recommends an insecure pattern.

Out of scope (report to the upstream vendor instead):

- Bugs in `claude-agent-sdk`, `github-copilot-sdk`,
  `openai-codex-sdk`, or `openai` — airframe wraps these but the
  vulnerability lives upstream.
- Bugs in third-party adapters distributed outside this repository
  (the `airframe.adapters` entry-point group). Contact the
  adapter author.
- Vendor-side service vulnerabilities (Claude Max, GitHub Copilot,
  ChatGPT Plus, opencode-go Zen). Report to the vendor's own
  security channel.

## Supported versions

Pre-1.0 releases are best-effort: the latest minor (`0.x`) gets
security fixes; older minors do not. Once 1.0 ships, this table
becomes a real LTS policy.

| Version | Supported |
|---|---|
| 1.x (latest minor) | ✓ |
| Pre-1.0 (latest minor) | ✓ |
| Pre-1.0 (older minors) | ✗ — upgrade to latest 0.x |

## Credential handling

`airframe-agents` resolves credentials from a per-adapter chain
(env vars → vendor credential files → OAuth tokens — see
`docs/auth.md`). The library:

- Never logs raw credentials, even at DEBUG level.
- Never writes credentials to disk; the credential files we *read*
  are vendor-owned (e.g. `~/.claude/.credentials.json`).
- Defers all auth resolution to the underlying vendor SDK whenever
  possible, so vendor security advisories propagate naturally.

If you find a code path that violates any of the above, treat it as
a security issue and report per the workflow at the top of this
file.
