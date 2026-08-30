<!--
Sync Impact Report
==================
Version change:      (none) → 1.0.0
Ratified:            2026-08-30
Last amended:        2026-08-30
Rationale:           Initial ratification. Airframe was the one repo in the
                     get2knowio Python portfolio without a constitution,
                     despite being the library the rest of the portfolio
                     depends on.

Principles added:
  I.   Protocol Narrowness
  II.  Wrap Vendor SDKs, Don't Rewrite Them
  III. Teardown Never Raises
  IV.  Errors Are Vendor-Agnostic
  V.   Lazy SDK Imports
  VI.  Provider IDs Are Strict
  VII. Conformance Is Shared, Not Copied
  VIII.One Command Surface

Principles removed: (none)
Principles modified: (none)

Templates / files requiring updates:
  ✅ CLAUDE.md — "Governance" section added, pointing here
  ✅ tests/unit/test_constitution.py — added; enforces I, II, V, VI, VII
  ✅ mise.toml — `check` is the verb named by Principle VIII
  ✅ .github/workflows/ci.yml — single step, `mise run check`
  ⚠️  Principle III (teardown never raises) is enforced by
      `airframe.testing.contracts`, not by test_constitution.py — see
      the principle text for which contract function does it.
  ⚠️  Principle IV has no structural test. Stated as review-enforced
      rather than given a decorative CI row.

Deferred placeholders: (none)
-->

# Airframe Constitution

Airframe is "JDBC for LLM agent SDKs": a vendor-neutral `AgentRuntime`
protocol plus pluggable adapters. Every principle below exists because
a consumer writes code against the protocol and must not have to care
which adapter is underneath.

Each principle names **the command that fails when it is violated**.
A principle that isn't a test is a suggestion — where the honest
answer is "review", it says so rather than being given a decorative
CI row.

## Core Principles

### I. Protocol Narrowness

`AgentRuntime` has five methods — `execute`, `reset`, `close`,
`validate_binding`, `list_models` — plus the capability predicate
`supports(Feature)` and the JDBC-style escape hatch
`unwrap(NativeType)`. Everything above the protocol (retry, fallback,
memory, orchestration) is consumer responsibility.

Widening the protocol is a breaking change to every third-party
adapter in existence, so the bar is: a capability that cannot be
expressed as a `Feature` predicate plus arguments to `execute`.

*Enforced by:* `mise run check` →
`tests/unit/test_constitution.py::test_protocol_surface_is_the_declared_five`.

### II. Wrap Vendor SDKs, Don't Rewrite Them

Airframe exposes a vendor-neutral protocol over each vendor's
official SDK. It does not reimplement the vendor's wire format.
Before hand-rolling HTTP, headers, retry policy, error parsing, or
auth against a vendor endpoint, check whether the official SDK
already exposes that surface — and use it if it does.

Local on-disk credential helpers (reading
`~/.claude/.credentials.json` and the like) are the one legitimate
exception, because no SDK exposes them. The rest of the auth dance
belongs to the SDK.

*Enforced by:* `mise run check` →
`tests/unit/test_constitution.py::test_no_hand_rolled_vendor_http`,
which fails on a raw `httpx`/`requests` call to a vendor host in an
adapter that has an SDK.

### III. Teardown Never Raises

`close()` is idempotent and never raises. It runs from `finally` and
`__aexit__`; shadowing the original exception would be catastrophic.
`reset()` is idempotent too — it drops scope-bound state (sessions,
threads) but keeps runtime-wide resources (subprocess pool, HTTP
client, auth tokens).

*Enforced by:* `mise run check` →
`airframe.testing.contracts.test_close_is_idempotent` and
`test_reset_is_idempotent`, which every in-tree adapter's
`tests/unit/test_*_conformance.py` imports, and which third-party
adapters inherit via `airframe-agents[testing]`.

### IV. Errors Are Vendor-Agnostic

Adapters classify vendor failures at the boundary into the
`airframe.errors.Runtime*Error` hierarchy. A vendor exception type
never escapes past the adapter, because a consumer catching
`RuntimeAuthError` must not also have to catch
`anthropic.AuthenticationError`.

*Enforced by:* code review, plus the per-adapter error-classification
matrix each `tests/unit/test_<vendor>.py` carries. There is no
structural test that proves the absence of a leak across every code
path, and this principle is deliberately not given one rather than
being given a test that only appears to check it.

### V. Lazy SDK Imports

`import airframe` must not pull in `claude-agent-sdk`, `openai`,
`aioboto3`, or any other vendor SDK. Adapter modules import their
vendor SDK inside the method that needs it, gated on the optional
extra being installed. A consumer who installs one extra pays for one
SDK.

*Enforced by:* `mise run check` →
`tests/unit/test_constitution.py::test_import_airframe_pulls_no_vendor_sdk`,
which imports the package in a clean subprocess and inspects
`sys.modules`.

### VI. Provider IDs Are Strict

Every adapter declares `PROVIDER_ID`, `REQUIRES_PACKAGE` and
`EXTRA_NAME` as ClassVars — they drive `airframe.discovery` filtering
by installed extras and the `airframe.adapters` entry-point group.

Provider IDs have no aliases. `"anthropic"` and `"openai"` are
reserved for future direct-API adapters and must not be taken by a
subscription, gateway, or agent-server adapter. Distinct wire formats
get distinct IDs even under one brand — `"opencode"` (local HTTP
agent server), `"opencode-zen"` (per-token gateway) and
`"opencode-go"` (flat-fee gateway) are three adapters, not one with
options.

*Enforced by:* `mise run check` →
`tests/unit/test_constitution.py::test_every_adapter_declares_provider_classvars`,
`::test_provider_ids_are_unique`, and
`::test_reserved_provider_ids_are_not_taken`.

### VII. Conformance Is Shared, Not Copied

Behaviour every adapter must satisfy lives in
`airframe.testing.contracts` as shared pytest functions, not
duplicated per adapter. In-tree adapters import them and supply an
`adapter_runtime` fixture; third-party adapters do the same via
`pip install airframe-agents[testing]`. When adapter behaviour is
contract-worthy, it goes in `contracts.py` so every adapter inherits
it — including ones that do not live in this repo.

*Enforced by:* `mise run check` →
`tests/unit/test_constitution.py::test_every_adapter_has_a_conformance_suite`.

### VIII. One Command Surface

`mise run check` is the gate. It returns a non-zero exit code, it is
what a contributor runs locally, and it is the only step CI invokes.
Nothing in the task graph swallows a failure — no `|| true`, no
`grep`-and-ignore. CI never spells the steps out inline, because two
lists that agree today are two lists that drift tomorrow.

The verbs are the portfolio contract: `setup`, `build`, `test`,
`lint`, `fmt`, `check`, `release`.

*Enforced by:* `mise run check` →
`tests/unit/test_constitution.py::test_ci_invokes_only_the_check_verb` and
`::test_no_task_swallows_failure`.

## Quality Gates

| Gate | Command | Blocking |
|------|---------|----------|
| Format | `mise run fmt-check` | yes |
| Lint | `uv run ruff check src tests examples` | yes |
| Types | `uv run mypy src/airframe` (all extras installed) | yes |
| Tests | `uv run pytest` across 3.12 / 3.13 / 3.14 | yes |
| Coverage floor | `fail_under` in `pyproject.toml` | yes |
| CodeQL | `.github/workflows/codeql.yml` | yes |
| Dependency review | `.github/workflows/dependency-review.yml` | yes |
| Conventional PR title | `.github/workflows/pr-title.yml` | yes |

The type gate runs only after `uv sync --all-extras`. With
`ignore_missing_imports = true`, an uninstalled extra silently
degrades those modules to `Any` and mypy passes while checking
nothing — a green run over a half-installed environment is worse than
no run, because it looks like evidence.

The coverage floor is a ratchet set at the measured number. It only
ever goes up. A floor nobody can meet is a floor that gets deleted.

Mypy strictness is likewise a ratchet: the destination is
`strict = true`, reached a module at a time through
`[[tool.mypy.overrides]]`. New modules are strict from birth. What is
not acceptable is strictness that never moves.

## Development Workflow

- Python 3.12+. Three interpreter numbers must agree:
  `project.requires-python`, `tool.ruff.target-version`, and
  `tool.mypy.python_version`. CI tests the floor **and** the ceiling.
- Type hints on every public function; Google-style docstrings
  (Args / Returns / Raises).
- `tests/unit/` is the default suite. `tests/integration/` talks to
  real vendor endpoints and is marked automatically by
  `tests/conftest.py`.
- Versions derive from git tags via `hatch-vcs`. Nothing carries a
  hand-bumped literal; the tag is the version.
- Conventional commits, enforced on the PR title because this repo
  squash-merges.
- Modules under `src/airframe/` stay under ~500 LOC by convention.

## Governance

This constitution supersedes ad-hoc practice. Where it and a code
comment disagree, this document wins and the comment is a bug.

**Amendment procedure.** Amendments land as a PR that (a) edits this
file, (b) bumps the version below per the policy, (c) updates the
Sync Impact Report at the top of this file, and (d) adds or updates
the enforcing test named by any changed principle — or states
explicitly, in the principle text, that review is the enforcement.

**Versioning policy.** Semantic versioning over the principles:

- **MAJOR** — a principle is removed, or redefined such that
  previously conforming code no longer conforms.
- **MINOR** — a principle is added, or an existing one materially
  expanded.
- **PATCH** — wording, examples, or a corrected reference, with no
  change to what conforms.

**Compliance review.** `mise run check` is the compliance check.
Every principle above either names a command that fails on violation
or states that review is the enforcement. A principle in neither
category is a defect in this document.

**Version**: 1.0.0 | **Ratified**: 2026-08-30 | **Last Amended**: 2026-08-30
