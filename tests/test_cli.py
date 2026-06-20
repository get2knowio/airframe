"""Unit tests for :mod:`airframe.cli`.

The CLI is a thin shell over discovery + the ``AgentRuntime``
protocol, so these tests stub the runtime boundary: a fake adapter
class stands in for ``runtime_for(provider)`` and the real adapters /
vendor SDKs / network are never touched. Coverage focuses on the
shell's own logic — prompt resolution, output rendering, and the
stable exit-code table.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pytest

import airframe
from airframe import cli
from airframe.cost import CostRecord
from airframe.errors import RuntimeAuthError, RuntimeTransientError
from airframe.protocol import RuntimeResult


class _FakeRuntime:
    """Minimal stand-in for an adapter: records init, replays a result."""

    last_init: dict[str, Any] = {}
    closed: bool = False
    raises: Exception | None = None
    result = RuntimeResult(
        text="hello from fake",
        structured=None,
        cost=CostRecord(
            provider_id="opencode-zen",
            model_id="gpt-5-mini",
            cost_usd=0.0012,
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=0,
            cache_write_tokens=0,
            finish="stop",
        ),
        finish="stop",
    )

    def __init__(self, *, model: str | None = None) -> None:
        type(self).last_init = {"model": model}
        type(self).closed = False
        self.label = "fake"

    async def execute(self, prompt: str, **kwargs: Any) -> RuntimeResult:
        type(self).last_execute = {"prompt": prompt, **kwargs}
        if type(self).raises is not None:
            raise type(self).raises
        return type(self).result

    async def close(self) -> None:
        type(self).closed = True


@pytest.fixture
def fake_provider(monkeypatch: pytest.MonkeyPatch) -> type[_FakeRuntime]:
    """Point ``runtime_for`` at a fresh fake adapter class."""

    class Fresh(_FakeRuntime):
        raises = None

    monkeypatch.setattr(airframe, "runtime_for", lambda provider: Fresh)
    return Fresh


# --- prompt resolution -----------------------------------------------------


def test_read_prompt_inline_wins() -> None:
    ns = _ns(prompt="inline", prompt_file=None)
    assert cli._read_prompt(ns) == "inline"


def test_read_prompt_from_file(tmp_path: Any) -> None:
    p = tmp_path / "prompt.txt"
    p.write_text("  from file\n", encoding="utf-8")
    ns = _ns(prompt=None, prompt_file=str(p))
    assert cli._read_prompt(ns) == "from file"


def test_read_prompt_from_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.sys, "stdin", _FakeStdin("piped prompt\n"))
    ns = _ns(prompt=None, prompt_file=None)
    assert cli._read_prompt(ns) == "piped prompt"


def test_read_prompt_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.sys, "stdin", _FakeStdin("", tty=True))
    ns = _ns(prompt=None, prompt_file=None)
    with pytest.raises(ValueError, match="no prompt given"):
        cli._read_prompt(ns)


def test_read_prompt_empty_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    ns = _ns(prompt="   ", prompt_file=None)
    with pytest.raises(ValueError, match="empty"):
        cli._read_prompt(ns)


# --- rendering -------------------------------------------------------------


def test_render_text() -> None:
    run = _run_args(fmt="text")
    out = cli._render(_FakeRuntime.result, run)
    assert out == "hello from fake"


def test_render_json_envelope() -> None:
    run = _run_args(fmt="json")
    out = cli._render(_FakeRuntime.result, run)
    parsed = json.loads(out)
    assert parsed == {
        "provider": "opencode-zen",
        "model": "gpt-5-mini",
        "text": "hello from fake",
        "structured": None,
        "finish": "stop",
        "cost_usd": 0.0012,
    }


# --- run command exit codes ------------------------------------------------


def test_run_happy_path(fake_provider: type[_FakeRuntime], capsys: Any) -> None:
    code = cli.main(["run", "--provider", "opencode-zen", "--prompt", "hi", "--model", "m"])
    assert code == cli.EXIT_OK
    assert fake_provider.last_init == {"model": "m"}
    assert fake_provider.closed is True
    assert "hello from fake" in capsys.readouterr().out


def test_run_writes_output_file(fake_provider: type[_FakeRuntime], tmp_path: Any) -> None:
    dest = tmp_path / "out.txt"
    code = cli.main(["run", "--provider", "x", "--prompt", "hi", "--output-file", str(dest)])
    assert code == cli.EXIT_OK
    assert dest.read_text(encoding="utf-8").strip() == "hello from fake"


def test_run_unknown_provider(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    def boom(provider: str) -> Any:
        raise ValueError("No airframe adapter serves provider 'nope'")

    monkeypatch.setattr(airframe, "runtime_for", boom)
    code = cli.main(["run", "--provider", "nope", "--prompt", "hi"])
    assert code == cli.EXIT_PROVIDER
    assert "No airframe adapter" in capsys.readouterr().err


def test_run_missing_extra(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    def boom(provider: str) -> Any:
        raise ImportError("requires the 'aioboto3' package")

    monkeypatch.setattr(airframe, "runtime_for", boom)
    code = cli.main(["run", "--provider", "bedrock", "--prompt", "hi"])
    assert code == cli.EXIT_PROVIDER
    assert "aioboto3" in capsys.readouterr().err


def test_run_auth_error(fake_provider: type[_FakeRuntime], capsys: Any) -> None:
    fake_provider.raises = RuntimeAuthError("bad key")
    code = cli.main(["run", "--provider", "x", "--prompt", "hi"])
    assert code == cli.EXIT_AUTH
    assert fake_provider.closed is True  # close() still ran
    assert "authentication failed" in capsys.readouterr().err


def test_run_runtime_error(fake_provider: type[_FakeRuntime], capsys: Any) -> None:
    fake_provider.raises = RuntimeTransientError("503")
    code = cli.main(["run", "--provider", "x", "--prompt", "hi"])
    assert code == cli.EXIT_RUNTIME
    assert "RuntimeTransientError" in capsys.readouterr().err


def test_run_no_prompt(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    monkeypatch.setattr(cli.sys, "stdin", _FakeStdin("", tty=True))
    code = cli.main(["run", "--provider", "x"])
    assert code == cli.EXIT_UNEXPECTED
    assert "no prompt given" in capsys.readouterr().err


# --- providers command -----------------------------------------------------


def test_providers_text(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    monkeypatch.setattr(
        airframe, "list_providers", lambda *, installed_only: ["claude", "bedrock"]
    )
    code = cli.main(["providers"])
    assert code == cli.EXIT_OK
    assert capsys.readouterr().out.split() == ["claude", "bedrock"]


def test_providers_json_all(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_list(*, installed_only: bool) -> list[str]:
        captured["installed_only"] = installed_only
        return ["claude"]

    monkeypatch.setattr(airframe, "list_providers", fake_list)
    code = cli.main(["providers", "--all", "--format", "json"])
    assert code == cli.EXIT_OK
    assert captured["installed_only"] is False
    assert json.loads(capsys.readouterr().out) == ["claude"]


# --- helpers ---------------------------------------------------------------


class _FakeStdin(io.StringIO):
    def __init__(self, text: str, *, tty: bool = False) -> None:
        super().__init__(text)
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def _ns(**kwargs: Any) -> Any:
    import argparse

    return argparse.Namespace(**kwargs)


def _run_args(*, fmt: str = "text") -> cli._RunArgs:
    return cli._RunArgs(
        provider="opencode-zen",
        model="gpt-5-mini",
        prompt="hi",
        system=None,
        timeout=600.0,
        fmt=fmt,
        output_file=None,
    )
