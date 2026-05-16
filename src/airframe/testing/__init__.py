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

The Phase 0 contracts are *structural* (no network, no auth — they
run in default unit-test mode). Behavioural / integration contracts
(401 ⇒ ``RuntimeAuthError``; successful call ⇒ ``input_tokens > 0``;
``schema=`` round-trip) require live vendor credentials and will
land alongside Phase 1's streaming/multi-turn integration tests in
:mod:`airframe.testing.integration`.
"""

from __future__ import annotations

__all__ = ["contracts"]
