""":mod:`airframe.testing` — shared conformance contracts for adapter authors.

Importable test functions that exercise the airframe protocol against
any adapter. Modelled on
`SQLAlchemy's testing.suite <https://docs.sqlalchemy.org/en/20/dialects/>`_
pattern: third-party adapter authors do ::

    # tests/test_my_adapter_conformance.py
    import pytest
    from airframe.testing.contracts import (
        test_close_is_idempotent,
        test_unwrap_returns_self,
        test_supports_returns_bool,
        # ...
    )
    from airframe_adapters_together import TogetherRuntime

    @pytest.fixture
    def adapter_runtime():
        return TogetherRuntime(api_key="...")

…and run ``pytest`` against that file. Pytest collects the imported
test functions into the adapter author's test module; each one calls
the ``adapter_runtime`` fixture and asserts a contract on the airframe
protocol. The fixture name ``adapter_runtime`` is the convention.

No pytest plugin, no entry-point auto-discovery, no separate
distribution — just an importable submodule. Pytest moves from the
test dependency group into an optional ``airframe-agents[testing]``
extra so adapter authors can ``pip install airframe-agents[testing]``
and pick up the contracts.

Structural contracts (Phase 0 + Phase 1–5 capability-vs-API
agreement) live in :mod:`airframe.testing.contracts` and run in
default unit-test mode — no network, no auth. They cover the
"declared capability matches the gate's behaviour" surface for
every Phase 0–5 feature.

Behavioural contracts that require live vendor credentials live in
:mod:`airframe.testing.integration`. They're parametrized over
provider id and gated by ``pytest.mark.integration`` so the default
suite stays passing without credentials. Run them with
``pytest -m integration`` once the relevant adapter's auth chain is
satisfied (e.g. ``ANTHROPIC_API_KEY`` for ``claude``,
``GITHUB_TOKEN`` for ``github-copilot``).
"""

from __future__ import annotations

__all__ = ["contracts", "integration"]
