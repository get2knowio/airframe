"""Per-request metadata — abuse-detection / attribution / audit tags.

Every vendor SDK accepts a free-form per-request tag for abuse
detection and per-tenant usage attribution. Consumers running
multi-tenant agent UIs all reinvent this per adapter today — the
:class:`RequestMetadata` namespace lets them pass one typed object
that adapters forward into whichever vendor channel they support
(or silently drop, when no channel exists).

**Contract — silent-ignore, not a capability gate.** Unlike most
features, passing ``metadata=`` to an adapter that doesn't expose a
metadata channel does *not* raise
:class:`~airframe.errors.UnsupportedFeatureError`. The tag is pure
observation — the call's correctness doesn't depend on the tag
reaching the vendor. Consumers who *care* whether the tag actually
propagated check
:data:`~airframe.features.Feature.REQUEST_METADATA` via
:meth:`~airframe.protocol.AgentRuntime.supports` first; consumers
who just want best-effort attribution can pass ``metadata=`` and
forget about it.

This mirrors the existing ``persona=`` kwarg on
:meth:`~airframe.protocol.AgentRuntime.execute` — informational,
adapter-may-honour-may-not, no capability gate.

Per-adapter mapping (today):

* **OpenAI-compatible** — ``user_id`` → ``user=``; ``tags`` →
  ``metadata=`` (chat.completions native field); ``request_id`` →
  ``extra_headers={"X-Request-ID": ...}``.
* **Claude Agent SDK** — ``user_id`` → ``ClaudeAgentOptions.user``.
  ``tags`` / ``request_id`` silently dropped (no agent-SDK channel).
* **Bedrock / Copilot / Kimi / OpenCode** — silently dropped. The
  feature flag stays ``False`` until a real channel surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RequestMetadata:
    """Per-request observation tag forwarded to the vendor when possible.

    Attributes:
        user_id: Stable identifier for the end-user the call is on
            behalf of. Vendors use this for abuse detection and
            per-end-user rate limiting (OpenAI's ``user=`` parameter,
            Anthropic's ``metadata={"user_id": ...}``). ``None`` when
            the call isn't on behalf of a specific user.
        request_id: Caller-side correlation identifier for stitching
            this turn into application logs / traces. Forwarded as a
            ``X-Request-ID`` header on vendors that accept it; some
            vendors echo it back in their response headers.
        tags: Arbitrary string→string labels (tenant, environment,
            experiment cohort, etc.). Forwarded via the vendor's
            native ``metadata`` channel when one exists; values
            must be strings to match the lowest common denominator
            across vendors (OpenAI requires ``Dict[str, str]``).
    """

    user_id: str | None = None
    request_id: str | None = None
    tags: dict[str, str] | None = None


__all__ = ["RequestMetadata"]
