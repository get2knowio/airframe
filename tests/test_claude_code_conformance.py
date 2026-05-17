"""Conformance contract suite for :class:`ClaudeCodeRuntime`.

Imports every test function from :mod:`airframe.testing.contracts`
and runs them against a locally-constructed adapter. Pytest collects
the imported test functions into this module via the standard
"import + pytest discovers names starting with ``test_``" mechanism
— same pattern SQLAlchemy uses in ``sqlalchemy.testing.suite``.

This is the in-tree mirror of what a third-party adapter author
would write. Treat the structure as the canonical example.
"""

from __future__ import annotations

import pytest

from airframe.adapters.claude_code import ClaudeCodeRuntime

# Importing the test functions makes pytest collect them here, scoped
# to the ``adapter_runtime`` fixture defined below.
from airframe.testing.contracts import (  # noqa: F401
    test_close_is_idempotent,
    test_close_on_fresh_runtime,
    test_plain_text_execute_path_is_wired,
    test_session_cancel_when_idle_is_noop,
    test_session_close_is_idempotent,
    test_session_close_on_fresh_session_is_safe,
    test_session_factory_returns_agent_session,
    test_session_resume_not_implemented_until_feature_flips,
    test_session_stream_is_async_generator,
    test_supports_accepts_model_kwarg,
    test_supports_is_idempotent,
    test_supports_returns_bool_for_every_feature,
    test_supports_structured_output_json_schema_is_true,
    test_unwrap_returns_self,
    test_unwrap_unrelated_type_raises_typeerror,
    test_validate_binding_returns_bool,
)


@pytest.fixture
def adapter_runtime() -> ClaudeCodeRuntime:
    """A ClaudeCodeRuntime instance — no auth required to construct.

    The runtime defers all SDK import / auth resolution to the first
    ``execute()`` call. The Phase 0 conformance contracts are
    structural and don't trigger that path, so a default-constructed
    runtime is sufficient.
    """
    return ClaudeCodeRuntime()
