"""Conformance contract suite for :class:`CodexRuntime`."""

from __future__ import annotations

import pytest

from airframe.adapters.codex import CodexRuntime
from airframe.testing.contracts import (  # noqa: F401
    test_close_is_idempotent,
    test_close_on_fresh_runtime,
    test_plain_text_execute_path_is_wired,
    test_supports_accepts_model_kwarg,
    test_supports_is_idempotent,
    test_supports_returns_bool_for_every_feature,
    test_supports_structured_output_json_schema_is_true,
    test_unwrap_returns_self,
    test_unwrap_unrelated_type_raises_typeerror,
    test_validate_binding_returns_bool,
)


@pytest.fixture
def adapter_runtime() -> CodexRuntime:
    return CodexRuntime()
