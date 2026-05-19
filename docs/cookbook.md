# Cookbook

`examples/probe_*.py` are runnable live-vendor demos. Each probe
exercises one feature end-to-end against a real CLI / HTTP
endpoint and prints the resulting events / cost / errors so you
can see exactly what production traffic would look like.

They're not part of `make test` (pytest collects `test_*.py` only).
Auth issues surface as classified `Runtime*Error` so a
mis-configured credential reports usefully rather than hanging.

## Per-adapter sanity probes

Single-call `execute()` against one adapter — the cheapest way to
confirm credentials are wired and the vendor responds.

| Probe | What it does |
|---|---|
| `examples/probe_bedrock.py` | `BedrockRuntime.execute(schema=)` round-trip |
| `examples/probe_claude_code.py` | `ClaudeCodeRuntime.execute(schema=)` round-trip |
| `examples/probe_copilot.py` | `CopilotRuntime.execute(schema=)` round-trip |
| `examples/probe_kimi.py` | `KimiRuntime.execute()` round-trip (no `schema=` yet — see adapter docs) |
| `examples/probe_opencode_server.py` | `OpenCodeServerRuntime` against a local `opencode serve` — plain-text execute, streaming, session resume |
| `examples/probe_opencode_zen.py` | `OpenCodeZenRuntime.execute(schema=)` round-trip |

```bash
uv run python examples/probe_claude_code.py
```

## Cross-provider parity

| Probe | What it does |
|---|---|
| `examples/probe_parity.py` | Same `execute(schema=)` round-trip against every installed adapter via `runtime_for(pid)()`. Demonstrates that consumer code is identical across vendors — no per-vendor imports, no per-vendor conditionals. |

```bash
uv run python examples/probe_parity.py
uv run python examples/probe_parity.py --providers claude,copilot
AIRFRAME_PROBE_MODEL_CLAUDE=claude-haiku-4-5 uv run python examples/probe_parity.py
```

Outcomes: `PASS` (structured output returned), `SKIP` (no creds for
that adapter), `FAIL` (any other error). Skips are non-fatal — exit
code is 0 unless a provider had creds but raised something else.

## Capability discovery

| Probe | What it does |
|---|---|
| `examples/probe_supports.py` | Prints the live `Feature × adapter` matrix. Filtered to installed extras by default; `--installed-only=false` shows every built-in. |

```bash
uv run python examples/probe_supports.py
uv run python examples/probe_supports.py --provider claude
```

## Sessions, streaming, resume, cancel

| Probe | What it does |
|---|---|
| `examples/probe_streaming.py` | `session.stream(prompt)` against any installed adapter; prints each `TextDelta` / `ReasoningDelta` / `ToolCallStart` / `TurnComplete` |
| `examples/probe_session_resume.py` | Two-turn `session(resume=)` against the three SDK adapters; verifies the second turn sees first-turn context |

## Reasoning + vision/file inputs

| Probe | What it does |
|---|---|
| `examples/probe_thinking.py` | `execute(thinking="medium")` round-trips on adapters declaring `REASONING_EFFORT`; demonstrates `thinking="disabled"` and the `{"budget_tokens": N}` Claude shape |
| `examples/probe_vision.py` | `execute([text, ImageInput(...)])` polymorphic prompt; covers path-, bytes-, and url-based image inputs |

## Function tools

| Probe | What it does |
|---|---|
| `examples/probe_tools.py` | `session(tools=[FunctionTool])` round-trip; the model calls `add(17, 25)` and the handler's `42` lands in the final response |

## External MCP servers

| Probe | What it does |
|---|---|
| `examples/probe_mcp.py` | `session(mcp_servers=[McpServerRef])` with the public `@modelcontextprotocol/server-everything` stdio server; lists the server's tools and calls `echo` |

## Permission, hooks, budget

| Probe | What it does |
|---|---|
| `examples/probe_permission.py` | `session(on_permission=)` with a logging callback that approves everything; surfaces the `PermissionRequest`s the adapter generates per call (Claude/Copilot/Kimi) |
| `examples/probe_hooks.py` | `session(on_event=)` with a logging observer; prints the per-kind histogram and verifies causal ordering (`session_start` first, `session_end` last) |
| `examples/probe_budget.py` | `session.execute(max_turns=2, max_budget_usd=0.0001)` with a deliberately tiny cap; demonstrates `RuntimeBudgetExceededError` firing on the second turn |

## Running the integration suite as tests

The probes' behavioural assertions also live in
`airframe.testing.integration` as pytest tests. Run with:

```bash
pytest -m integration                                    # all adapters; self-skip on missing creds
pytest -m integration tests/test_claude_code_integration.py  # one adapter
```

See [adapters/third-party.md](./adapters/third-party.md) for the
import-into-suite pattern third-party adapters use.

## Composing custom probes

Each probe is a self-contained `asyncio` script. They follow a
consistent shape:

```python
import asyncio
import argparse
from airframe import list_providers, runtime_for

async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="claude")
    args = parser.parse_args()

    if args.provider not in list_providers():
        print(f"{args.provider!r} not installed")
        return 1

    cls = runtime_for(args.provider)
    runtime = cls()  # or cls(api_key=...) for opencode-zen
    try:
        # Your probe logic here.
        result = await runtime.execute("hello", schema=...)
        print(result.cost.to_dict())
    finally:
        await runtime.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
```

The pattern works equally well for ad-hoc local scripts. See any
existing `examples/probe_*.py` for a fuller template.
