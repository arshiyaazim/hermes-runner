"""Pure Phase 3 AI routing contracts shared by runtime adapters.

This module selects no provider and performs no I/O.  It defines the semantic
boundary between business callers, policy, and repository-specific adapters so
those layers can be migrated independently behind feature gates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class Capability(str, Enum):
    TEXT_GENERATION = "text_generation"
    STRUCTURED_OUTPUT = "structured_output"
    LONG_CONTEXT = "long_context"
    VISION = "vision"
    AUDIO = "audio"


class CostClass(str, Enum):
    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"


_COST_RANK = {CostClass.LOW: 0, CostClass.STANDARD: 1, CostClass.HIGH: 2}


class LatencyClass(str, Enum):
    INTERACTIVE = "interactive"
    BACKGROUND = "background"
    BATCH = "batch"


class PrivacyClass(str, Enum):
    LOCAL_ONLY = "local_only"
    PRIVATE_GATEWAY = "private_gateway"
    APPROVED_EXTERNAL = "approved_external"


class TransportKind(str, Enum):
    LOCAL = "local"
    PRIVATE_GATEWAY = "private_gateway"
    DIRECT_PROVIDER = "direct_provider"


class FailureClass(str, Enum):
    AUTHENTICATION_FAILED = "authentication_failed"
    AUTHORIZATION_FAILED = "authorization_failed"
    QUOTA_EXHAUSTED = "quota_exhausted"
    RATE_LIMITED = "rate_limited"
    MODEL_UNAVAILABLE = "model_unavailable"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_OUTAGE = "provider_outage"
    TIMEOUT = "timeout"
    MALFORMED_RESPONSE = "malformed_response"
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    SAFETY_REFUSAL = "safety_refusal"
    INVALID_REQUEST = "invalid_request"
    POLICY_REJECTION = "policy_rejection"
    INTERNAL_APPLICATION_DEFECT = "internal_application_defect"
    UNKNOWN = "unknown"


class FailureAction(str, Enum):
    RETRY_THEN_NEXT = "retry_then_next"
    BOUNDED_RETRY_THEN_NEXT = "bounded_retry_then_next"
    NEXT_ROUTE = "next_route"
    NEXT_COMPATIBLE_ROUTE = "next_compatible_route"
    FAIL_CLOSED = "fail_closed"
    INTERNAL_ERROR = "internal_error"


class AttemptOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


class NoCompatibleRoute(LookupError):
    """Raised before invocation when no route satisfies the request."""


_IDENTIFIER_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$")
_SECRET_VALUE_PREFIXES = ("sk-", "bearer ", "basic ")


def _validate_identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a non-empty safe identifier")


@dataclass(frozen=True)
class InferenceRequest:
    """Model-independent request produced by business/conversation code."""

    workload: str
    required_capabilities: frozenset[Capability]
    max_cost_class: CostClass
    latency_class: LatencyClass
    required_context_tokens: int
    privacy_class: PrivacyClass
    fallback_eligible: bool
    retry_eligible: bool
    correlation_ref: str
    context_version: str

    def __post_init__(self) -> None:
        _validate_identifier(self.workload, "workload")
        if not self.required_capabilities:
            raise ValueError("required_capabilities must not be empty")
        if self.required_context_tokens < 0:
            raise ValueError("required_context_tokens must be non-negative")
        if not self.correlation_ref.strip():
            raise ValueError("correlation_ref must not be empty")
        _validate_identifier(self.context_version, "context_version")


@dataclass(frozen=True)
class RouteCandidate:
    """One non-secret, policy-approved provider/model transport candidate."""

    route_id: str
    provider: str
    model: str
    transport: TransportKind
    capabilities: frozenset[Capability]
    cost_class: CostClass
    latency_classes: frozenset[LatencyClass]
    max_context_tokens: int
    privacy_classes: frozenset[PrivacyClass]
    credential_ref: str

    def __post_init__(self) -> None:
        _validate_identifier(self.route_id, "route_id")
        _validate_identifier(self.provider, "provider")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must not be empty")
        if not self.capabilities:
            raise ValueError("capabilities must not be empty")
        if not self.latency_classes:
            raise ValueError("latency_classes must not be empty")
        if not self.privacy_classes:
            raise ValueError("privacy_classes must not be empty")
        if self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        credential_lower = (self.credential_ref or "").strip().lower()
        if (
            not _IDENTIFIER_RE.fullmatch(self.credential_ref or "")
            or credential_lower.startswith(_SECRET_VALUE_PREFIXES)
        ):
            raise ValueError("credential_ref must be a non-secret reference name")


def _incompatibilities(request: InferenceRequest, route: RouteCandidate) -> tuple[str, ...]:
    reasons: list[str] = []
    if not request.required_capabilities.issubset(route.capabilities):
        reasons.append("capability")
    if _COST_RANK[route.cost_class] > _COST_RANK[request.max_cost_class]:
        reasons.append("cost")
    if request.latency_class not in route.latency_classes:
        reasons.append("latency")
    if request.required_context_tokens > route.max_context_tokens:
        reasons.append("context")
    if request.privacy_class not in route.privacy_classes:
        reasons.append("privacy")
    return tuple(reasons)


def resolve_eligible_routes(
    request: InferenceRequest,
    routes: Iterable[RouteCandidate],
) -> tuple[RouteCandidate, ...]:
    """Filter in policy order, rejecting unsafe/incompatible downgrades.

    This performs compatibility filtering only.  Repository adapters own actual
    calls, and the retry executor owns transitions between returned candidates.
    """

    eligible: list[RouteCandidate] = []
    rejected: list[str] = []
    seen: set[str] = set()
    for route in routes:
        if route.route_id in seen:
            raise ValueError(f"duplicate route_id {route.route_id!r}")
        seen.add(route.route_id)
        reasons = _incompatibilities(request, route)
        if reasons:
            rejected.append(f"{route.route_id}: {','.join(reasons)}")
            continue
        eligible.append(route)
        if not request.fallback_eligible:
            break
    if not eligible:
        detail = "; ".join(rejected) if rejected else "no routes configured"
        raise NoCompatibleRoute(
            f"no compatible route for workload {request.workload!r}: {detail}"
        )
    return tuple(eligible)


def failure_action(
    failure_class: FailureClass,
    *,
    retry_eligible: bool,
) -> FailureAction:
    """Return the deterministic high-level action for a classified failure."""

    if failure_class in {
        FailureClass.AUTHENTICATION_FAILED,
        FailureClass.AUTHORIZATION_FAILED,
        FailureClass.SAFETY_REFUSAL,
        FailureClass.INVALID_REQUEST,
        FailureClass.POLICY_REJECTION,
    }:
        return FailureAction.FAIL_CLOSED
    if failure_class is FailureClass.INTERNAL_APPLICATION_DEFECT:
        return FailureAction.INTERNAL_ERROR
    if failure_class is FailureClass.MALFORMED_RESPONSE:
        return FailureAction.NEXT_COMPATIBLE_ROUTE
    if failure_class in {
        FailureClass.QUOTA_EXHAUSTED,
        FailureClass.MODEL_UNAVAILABLE,
        FailureClass.PROVIDER_UNAVAILABLE,
        FailureClass.CONTEXT_LENGTH_EXCEEDED,
        FailureClass.UNKNOWN,
    }:
        return FailureAction.NEXT_ROUTE
    if not retry_eligible:
        return FailureAction.NEXT_ROUTE
    if failure_class is FailureClass.RATE_LIMITED:
        return FailureAction.BOUNDED_RETRY_THEN_NEXT
    if failure_class in {FailureClass.TIMEOUT, FailureClass.PROVIDER_OUTAGE}:
        return FailureAction.RETRY_THEN_NEXT
    return FailureAction.NEXT_ROUTE


def build_attempt_audit(
    *,
    request: InferenceRequest,
    route: RouteCandidate,
    fallback_number: int,
    attempt_number: int,
    latency_ms: int,
    outcome: AttemptOutcome,
    failure_class: FailureClass | None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> dict[str, object]:
    """Build safe structured metadata; credential references are omitted."""

    if fallback_number < 0 or attempt_number < 1 or latency_ms < 0:
        raise ValueError("fallback_number, attempt_number or latency_ms is invalid")
    for value, name in ((input_tokens, "input_tokens"), (output_tokens, "output_tokens")):
        if value is not None and value < 0:
            raise ValueError(f"{name} must be non-negative")
    if outcome is AttemptOutcome.SUCCESS and failure_class is not None:
        raise ValueError("successful attempt cannot have a failure_class")
    if outcome is AttemptOutcome.FAILURE and failure_class is None:
        raise ValueError("failed attempt requires a failure_class")
    return {
        "event": "ai_routing_attempt",
        "correlation_ref": request.correlation_ref,
        "workload": request.workload,
        "context_version": request.context_version,
        "route_id": route.route_id,
        "transport": route.transport.value,
        "provider": route.provider,
        "model": route.model,
        "fallback_number": fallback_number,
        "attempt_number": attempt_number,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "outcome": outcome.value,
        "failure_class": failure_class.value if failure_class else None,
    }
