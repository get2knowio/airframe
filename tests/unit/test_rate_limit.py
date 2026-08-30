"""Unit tests for the cross-vendor rate-limit telemetry surface.

The :mod:`airframe.rate_limit` types are simple frozen dataclasses;
the interesting logic is the OpenAI-style header parser in
:mod:`airframe.adapters.openai_compatible`. The parser handles
``x-ratelimit-*`` headers in OpenAI's ``"1h2m3s"`` /
``"42ms"`` / ``"6.5s"`` / bare-integer duration formats plus a
``retry-after`` header for 429 responses.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from airframe.adapters.openai_compatible import (
    _duration_string_to_seconds,
    _parse_openai_rate_limit_headers,
)
from airframe.errors import RuntimeTransientError
from airframe.rate_limit import RateLimitInfo, RateLimitWindow


def test_rate_limit_info_default_is_empty() -> None:
    info = RateLimitInfo()
    assert info.windows == ()
    assert info.raw == {}


def test_rate_limit_window_minimal_construction() -> None:
    w = RateLimitWindow(name="requests")
    assert w.name == "requests"
    assert w.remaining is None
    assert w.limit is None
    assert w.utilization is None
    assert w.reset_at is None
    assert w.retry_after_seconds is None
    assert w.status is None


def test_runtime_transient_error_accepts_rate_limit_kwarg() -> None:
    info = RateLimitInfo(windows=(RateLimitWindow(name="requests", remaining=0),))
    err = RuntimeTransientError("throttled", rate_limit=info)
    assert err.rate_limit is info
    # Backwards-compatible default — existing call sites can omit it.
    legacy = RuntimeTransientError("network")
    assert legacy.rate_limit is None


def test_duration_string_parses_compound_units() -> None:
    assert _duration_string_to_seconds("1h2m3s") == 3723.0
    assert _duration_string_to_seconds("6m0s") == 360.0
    assert _duration_string_to_seconds("42ms") == 0.042
    assert _duration_string_to_seconds("6.5s") == 6.5
    assert _duration_string_to_seconds("10") == 10.0  # bare seconds
    assert _duration_string_to_seconds(7.5) == 7.5  # numeric pass-through
    assert _duration_string_to_seconds("") is None
    assert _duration_string_to_seconds(None) is None


def test_parse_openai_rate_limit_headers_returns_none_when_absent() -> None:
    assert _parse_openai_rate_limit_headers(None) is None
    assert _parse_openai_rate_limit_headers({}) is None
    assert _parse_openai_rate_limit_headers({"content-type": "application/json"}) is None


def test_parse_openai_rate_limit_headers_extracts_both_windows() -> None:
    info = _parse_openai_rate_limit_headers(
        {
            "x-ratelimit-limit-requests": "10000",
            "x-ratelimit-remaining-requests": "9999",
            "x-ratelimit-reset-requests": "6m0s",
            "x-ratelimit-limit-tokens": "2000000",
            "x-ratelimit-remaining-tokens": "1999500",
            "x-ratelimit-reset-tokens": "1s",
        }
    )
    assert info is not None
    by_name = {w.name: w for w in info.windows}
    assert set(by_name) == {"requests", "tokens"}
    req = by_name["requests"]
    assert req.limit == 10000
    assert req.remaining == 9999
    assert req.reset_at is not None
    delta = req.reset_at - datetime.now(tz=UTC)
    # 6m0s ⇒ 360s; allow a generous slop window for test scheduling.
    assert timedelta(seconds=350) < delta < timedelta(seconds=370)
    tok = by_name["tokens"]
    assert tok.limit == 2000000
    assert tok.remaining == 1999500


def test_parse_openai_rate_limit_headers_skips_window_when_all_three_absent() -> None:
    info = _parse_openai_rate_limit_headers(
        {
            "x-ratelimit-limit-requests": "100",
            # tokens window deliberately omitted
        }
    )
    assert info is not None
    assert len(info.windows) == 1
    assert info.windows[0].name == "requests"
    assert info.windows[0].limit == 100
    assert info.windows[0].remaining is None


def test_parse_openai_rate_limit_headers_propagates_retry_after() -> None:
    info = _parse_openai_rate_limit_headers(
        {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-reset-requests": "30s",
            "retry-after": "12",
        }
    )
    assert info is not None
    (window,) = info.windows
    assert window.retry_after_seconds == 12.0
    assert window.remaining == 0
