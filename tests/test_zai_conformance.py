"""Conformance contract suite for :class:`ZaiAnthropicRuntime`.

Imports every test function from :mod:`airframe.testing.contracts`
and runs them against a locally-constructed adapter. Pytest collects
the imported test functions into this module via the standard
"import + pytest discovers names starting with ``test_``" mechanism
— same pattern SQLAlchemy uses in ``sqlalchemy.testing.suite``.

``ZaiAnthropicRuntime`` inherits ``ClaudeCodeRuntime``'s harness but
declares a narrower ``SUPPORTED_FEATURES``. Running the full suite
against it is the point: most contracts are of the
"behaviour agrees with the capability flag" shape, so they assert the
*opposite* branch here than they do for the parent — that a declined
feature declines cleanly rather than half-working.
"""

from __future__ import annotations

import pytest

from airframe.adapters.zai import ZaiAnthropicRuntime

# Importing the test functions makes pytest collect them here, scoped
# to the ``adapter_runtime`` fixture defined below.
from airframe.testing.contracts import (  # noqa: F401
    test_close_is_idempotent,
    test_close_on_fresh_runtime,
    test_count_tokens_agrees_with_supports_flag,
    test_emittable_hook_kinds_subset_of_eight_literals,
    test_plain_text_execute_path_is_wired,
    test_runtime_result_has_rate_limit_field,
    test_runtime_result_has_reasoning_field,
    test_runtime_transient_error_carries_rate_limit_attr,
    test_session_accepts_cache_kwarg,
    test_session_accepts_metadata_kwarg,
    test_session_accepts_slash_commands_kwarg,
    test_session_cancel_when_idle_is_noop,
    test_session_close_is_idempotent,
    test_session_close_on_fresh_session_is_safe,
    test_session_execute_signature_accepts_budget_kwargs,
    test_session_execute_signature_accepts_thinking_kwarg,
    test_session_factory_returns_agent_session,
    test_session_max_budget_usd_declined_when_budget_usd_cap_false,
    test_session_max_turns_declined_when_budget_turn_cap_false,
    test_session_mcp_servers_kwarg_agrees_with_transport_capabilities,
    test_session_native_tools_ignores_foreign_raw_tool,
    test_session_native_tools_kwarg_agrees_with_tools_native_capability,
    test_session_on_event_agrees_with_lifecycle_hooks_capability,
    test_session_on_permission_agrees_with_permission_callback_capability,
    test_session_polymorphic_prompt_declined_when_vision_false,
    test_session_rejects_wrong_provider_options_namespace,
    test_session_resume_not_implemented_until_feature_flips,
    test_session_stream_is_async_generator,
    test_session_thinking_kwarg_declined_when_capability_false,
    test_session_tools_kwarg_agrees_with_tools_function_capability,
    test_supported_native_tools_agrees_with_tools_native_capability,
    test_supports_accepts_model_kwarg,
    test_supports_is_idempotent,
    test_supports_returns_bool_for_every_feature,
    test_supports_structured_output_json_schema_is_true,
    test_unwrap_returns_self,
    test_unwrap_unrelated_type_raises_typeerror,
    test_validate_binding_returns_bool,
)


@pytest.fixture
def adapter_runtime() -> ZaiAnthropicRuntime:
    """A ZaiAnthropicRuntime instance — no auth required to construct.

    Auth resolution is deferred to ``_subprocess_env()``, which runs on
    first connect. The structural conformance contracts never reach it,
    so a default-constructed runtime is sufficient and no ``ZAI_API_KEY``
    is needed to collect this module.
    """
    return ZaiAnthropicRuntime()
