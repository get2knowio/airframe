# Contributing to airframe

Thanks for considering a contribution. This project is small enough
that the rules can fit in one file.

## Development setup

```bash
git clone https://github.com/get2knowio/airframe
cd airframe
uv sync --all-extras --group dev
make ci
```

## Code style

- Python 3.11+; type hints on every public function.
- `make format-fix` before pushing.
- `make ci` is the pre-push gate (lint + format + typecheck + test).
- Docstrings: Google style (Args / Returns / Raises).
- Stay terse. Most modules in `src/airframe/` are under 500 LOC;
  keep them that way.

## Adding an adapter

A new adapter PR should include:

1. **`src/airframe/adapters/<vendor>.py`** — one class implementing
   `AgentRuntime`. Lazy-import the vendor's SDK inside the method
   that needs it so `import airframe` doesn't pull the SDK in.
2. **Optional dependency extra** in `pyproject.toml` so consumers
   can `pip install airframe-agents[<vendor>]`.
3. **`tests/test_<vendor>.py`** — unit tests with the SDK mocked at
   the boundary. Cover binding validation, structured-output happy
   path, missing structured output, full error classification matrix,
   timeout, lifecycle (`reset` / `aclose`), and cost.
4. **`examples/probe_<vendor>.py`** — end-to-end probe against the
   real vendor. Required to verify the adapter works against the
   actual SDK; the unit tests prove the contract, the probe proves
   the wiring.
5. **`docs/adapters/<vendor>.md`** — short reference covering auth
   chain, supported model IDs, structured-output mechanism, and any
   vendor quirks.
6. **`CHANGELOG.md`** entry under `[Unreleased]`.

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
- Run `make ci` before opening the PR.
- Sign-off the commit (`git commit -s`) is appreciated but not
  required.

## License

By contributing you agree your contribution is licensed under the
MIT License (see `LICENSE`).
