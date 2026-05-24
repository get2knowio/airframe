"""Cross-vendor rate-limit telemetry.

Every vendor SDK exposes some structured quota data. Today airframe
collapses all of it into an opaque :class:`RuntimeTransientError` —
the typed information that Claude's ``RateLimitInfo`` or OpenAI's
``x-ratelimit-*`` headers carry gets dropped on the floor. This
module names the cross-vendor shape so consumers can write
budget-aware retry, surface "you have X requests left this hour" UX,
or feed quota dashboards without ``unwrap()``-ing per adapter.

Two surfaces carry a :class:`RateLimitInfo`:

* :attr:`airframe.protocol.RuntimeResult.rate_limit` — populated when
  the vendor returned quota data on a successful call.
* :attr:`airframe.errors.RuntimeTransientError.rate_limit` — populated
  when the call was throttled and the error carried quota data.

Adapters declaring :data:`~airframe.features.Feature.RATE_LIMIT_TELEMETRY`
must populate one or the other when the vendor surfaced data; they
may always leave both ``None`` on a turn that produced no rate-limit
signal. Adapters that don't declare the feature leave both ``None``
unconditionally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class RateLimitWindow:
    """One quota window — e.g. requests/minute, tokens/5h.

    Vendor shapes are not uniform. OpenAI returns
    ``remaining`` + ``limit`` per window (requests + tokens); Claude
    returns ``utilization`` (0.0–1.0) per named window
    (``"five_hour"``, ``"seven_day"``, etc.); some vendors only return
    a ``Retry-After`` on 429 and no quota at all. Every field except
    :attr:`name` is therefore optional — consumers should treat any
    specific field as a hint and check ``is not None`` before using
    it.

    Attributes:
        name: Vendor-supplied window identifier. Conventional values
            include ``"requests"`` / ``"tokens"`` (OpenAI), ``"rpm"`` /
            ``"tpm"`` (some compat vendors), ``"five_hour"`` /
            ``"seven_day"`` / ``"seven_day_opus"`` /
            ``"seven_day_sonnet"`` / ``"overage"`` (Claude). Adapters
            preserve the vendor's name; consumers branching on it
            should expect vendor-specific values.
        remaining: Quota units left in this window, when the vendor
            reports it. ``None`` when not exposed (Claude reports
            utilisation instead of remaining).
        limit: Total quota units for this window, when the vendor
            reports it. ``None`` when not exposed.
        utilization: Fraction of the window consumed (0.0 ≤ x ≤ 1.0)
            when the vendor reports it. ``None`` when the vendor only
            reports ``remaining`` / ``limit``.
        reset_at: When this window resets to full. ``None`` when not
            reported.
        retry_after_seconds: Server-suggested wait before retrying.
            Typically populated on throttle responses (HTTP 429); may
            be ``None`` on a successful response that still carries
            window quota.
        status: ``"allowed"`` — request served normally;
            ``"allowed_warning"`` — served but near the limit;
            ``"rejected"`` — throttled. ``None`` for vendors that
            don't differentiate.
    """

    name: str
    remaining: int | None = None
    limit: int | None = None
    utilization: float | None = None
    reset_at: datetime | None = None
    retry_after_seconds: float | None = None
    status: Literal["allowed", "allowed_warning", "rejected"] | None = None


@dataclass(frozen=True, slots=True)
class RateLimitInfo:
    """Vendor-agnostic rate-limit snapshot.

    Wraps zero or more :class:`RateLimitWindow` instances plus the
    vendor's untyped raw payload for escape-hatch use. An instance
    with empty :attr:`windows` and empty :attr:`raw` is legal but
    uninformative — callers should treat
    ``RateLimitInfo(windows=())`` the same as ``None``.

    Attributes:
        windows: Per-window quota state, in vendor-supplied order.
            OpenAI typically emits two (requests + tokens); Claude
            emits one per active rate-limit type; some vendors emit
            none and only populate :attr:`raw`.
        raw: The vendor's untyped payload (headers dict, SDK event
            dict, etc.) for consumers that need to reach a field the
            typed surface doesn't yet expose. Treated as opaque by
            airframe; the shape is vendor-specific and unstable.
    """

    windows: tuple[RateLimitWindow, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)


__all__ = ["RateLimitInfo", "RateLimitWindow"]
