# Contributing to airframe

Thanks for considering a contribution. This project is small enough
that the rules can fit in one file.

## Development setup

`mise` is the task runner and the toolchain pin. It installs the right
Python and `uv` for you, so it is the only prerequisite:

```bash
curl https://mise.run | sh          # once, if you don't have it
git clone https://github.com/get2knowio/airframe
cd airframe
mise run setup                      # uv sync --all-extras --group dev
mise run check                      # the gate — exactly what CI runs
```

`mise tasks` lists everything. The verbs are the same in every
get2knowio repo:

| Verb | Does |
|------|------|
| `mise run setup` | install every extra plus the dev group |
| `mise run test` | full suite |
| `mise run test-fast` | skip the `integration` marker |
| `mise run lint` | ruff check + mypy |
| `mise run fmt` | apply formatting and safe fixes |
| `mise run check` | format + lint + types + tests with the coverage floor |
| `mise run build` | build the sdist and wheel |

Run one test with `uv run pytest tests/unit/test_claude_code.py::test_name -q`.

Install every extra, not a subset. Mypy is configured with
`ignore_missing_imports`, so an uninstalled extra silently degrades
those modules to `Any` — the type check then passes while checking
nothing, which is worse than not running it.

## Code style

- Python 3.12+; type hints on every public function.
- Docstrings: Google style (Args / Returns / Raises).
- Line length 100. `mise run fmt` before pushing.
- Every module opens with `from __future__ import annotations`. This is
  a lint rule, not a convention — a new file without it fails `check`.
- New modules are type-strict from birth. `pyproject.toml` turns on
  every `--strict` mypy flag the tree already satisfies; the leftovers
  are pinned per-module in `[[tool.mypy.overrides]]` with a count
  attached. That list is a ratchet — shorten it, don't extend it. If
  you genuinely must add a module, say why in the comment above it.
- Stay terse. Most modules in `src/airframe/` are under 500 LOC;
  keep them that way.

## Tests

- `tests/unit/` is the default suite. Everything there runs on every PR
  against Python 3.12, 3.13 and 3.14.
- `tests/integration/` talks to real vendor endpoints. The `integration`
  marker is applied automatically by directory (`tests/conftest.py`), so
  put the file in the right place and don't hand-mark it. These
  self-skip without credentials.
- Coverage is gated at a floor set to the current measured number, so it
  only ever ratchets up. A PR that adds uncovered code fails `check`
  even when every test passes.

## Governance

`.specify/memory/constitution.md` holds the principles this project
does not negotiate — protocol narrowness, lazy SDK imports, strict
provider IDs, teardown that never raises, and the rest. Each one names
the command that fails when it is violated, and
`tests/unit/test_constitution.py` is most of that enforcement.

Practically: you are unlikely to read it and then get surprised, but if
`test_constitution.py` fails you have hit a principle rather than a
nitpick. Read the assertion message — it cites the principle.

Changing a principle means bumping the constitution's version and
updating the Sync Impact Report at the top of that file. A test in the
suite checks the two agree.

## Adding an adapter

A new adapter PR should include:

1. **`src/airframe/adapters/<vendor>.py`** — one class implementing
   `AgentRuntime`. Lazy-import the vendor's SDK inside the method
   that needs it so `import airframe` doesn't pull the SDK in. Declare
   `PROVIDER_ID`, `REQUIRES_PACKAGE` and `EXTRA_NAME` as ClassVars;
   discovery is driven by them. Pick a `PROVIDER_ID` that names the
   wire format, not just the brand — `"anthropic"`, `"openai"`,
   `"moonshot"`, `"bedrock-agents"` and `"codex"` are reserved.
2. **Optional dependency extra** in `pyproject.toml` so consumers
   can `pip install airframe-agents[<vendor>]`.
3. **`tests/unit/test_<vendor>.py`** — unit tests with the SDK mocked at
   the boundary. Cover binding validation, structured-output happy
   path, missing structured output, full error classification matrix,
   timeout, lifecycle (`reset` / `close`), and cost.
4. **`tests/unit/test_<vendor>_conformance.py`** — imports the shared
   contracts from `airframe.testing.contracts` and supplies an
   `adapter_runtime` fixture. Required, and enforced: an adapter with no
   conformance suite fails `test_constitution.py`. Behaviour every
   adapter must satisfy belongs in `contracts.py`, not copied per
   adapter — third-party adapters inherit it from there too.
5. **`examples/probe_<vendor>.py`** — end-to-end probe against the
   real vendor. Required to verify the adapter works against the
   actual SDK; the unit tests prove the contract, the probe proves
   the wiring.
6. **`docs/adapters/<vendor>.md`** — short reference covering auth
   chain, supported model IDs, structured-output mechanism, and any
   vendor quirks.
7. **`CHANGELOG.md`** entry under `[Unreleased]`.

Reach the vendor through its official SDK. Airframe wraps SDKs; it does
not reimplement wire formats, auth refresh, retry policy or rate-limit
parsing that the vendor already ships. Reading a local credentials file
is the one accepted exception.

## Reporting bugs

Open an issue at <https://github.com/get2knowio/airframe/issues>.
Include:

- Python version, OS, adapter, vendor SDK version.
- A minimal reproducer (a `RuntimeXxxError` from a real call counts).
- For vendor-side issues (auth failing, model not found): the raw
  vendor error message helps a lot.

## Pull requests

- Branch off `main`.
- One logical change per PR. Adapter-add PRs and protocol-change PRs
  should be separate.
- Run `mise run check` before opening the PR.
- **The PR title must be a conventional commit.** This repo
  squash-merges, so the title becomes the commit message on `main` and
  a workflow checks it. Format is `type: subject`, where `type` is one
  of `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `build`, `ci`,
  `chore`, `revert`, `release`. The subject starts lowercase and has no
  trailing period — `fix: classify bedrock throttling as transient`.
- Don't touch the version. It is derived from git tags by `hatch-vcs`;
  there is no version literal to bump, and release tagging is a
  maintainer step.
- Sign-off the commit (`git commit -s`) is appreciated but not
  required.

## License

By contributing you agree your contribution is licensed under the
MIT License (see `LICENSE`).
