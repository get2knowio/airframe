"""Conformance contract suite for :class:`OpenCodeZenRuntime`.

Also exercises the :class:`OpenAICompatibleRuntime` base class — the
contracts here pass through to the base methods. A future
``Together`` / ``Groq`` / ``Fireworks`` adapter will get the same
coverage by writing a near-identical conformance file with its own
constructor in the fixture.
"""

from __future__ import annotations

import pytest

from airframe.adapters.opencode_zen import OpenCodeZenRuntime
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
def adapter_runtime() -> OpenCodeZenRuntime:
    # Dummy key satisfies ``OpenAICompatibleRuntime`` construction;
    # the structural contracts don't make any HTTP calls.
    return OpenCodeZenRuntime(api_key="dummy-key-for-conformance")
