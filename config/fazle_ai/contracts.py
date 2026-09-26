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
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_FAILURE = "authorization_failure"
    QUOTA_CREDIT_EXHAUSTED = "quota_credit_exhausted"
    RATE_LIMITED = "rate_limited"
    PROVIDER_OR_MODEL_UNAVAILABLE = "provider_or_model_unavailable"
    PROVIDER_OUTAGE_OR_5XX = "provider_outage_or_5xx"
    TIMEOUT = "timeout"
    MALFORMED_PROVIDER_RESPONSE = "malformed_provider_response"
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    SAFETY_OR_PROVIDER_REFUSAL = "safety_or_provider_refusal"
    INVALID_REQUEST_OR_POLICY_REJECTION = "invalid_request_or_policy_rejection"
    INTERNAL_APPLICATION_DEFECT = "internal_application_defect"
    UNKNOWN_UNCLASSIFIED = "unknown_unclassified"

    # Transitional source-compatible aliases. Structured audits always emit
    # the canonical values above.
    AUTHENTICATION_FAILED = AUTHENTICATION_FAILURE
    AUTHORIZATION_FAILED = AUTHORIZATION_FAILURE
    QUOTA_EXHAUSTED = QUOTA_CREDIT_EXHAUSTED
    MODEL_UNAVAILABLE = PROVIDER_OR_MODEL_UNAVAILABLE
    PROVIDER_UNAVAILABLE = PROVIDER_OR_MODEL_UNAVAILABLE
    PROVIDER_OUTAGE = PROVIDER_OUTAGE_OR_5XX
    MALFORMED_RESPONSE = MALFORMED_PROVIDER_RESPONSE
    SAFETY_REFUSAL = SAFETY_OR_PROVIDER_REFUSAL
    INVALID_REQUEST = INVALID_REQUEST_OR_POLICY_REJECTION
    POLICY_REJECTION = INVALID_REQUEST_OR_POLICY_REJECTION
    UNKNOWN = UNKNOWN_UNCLASSIFIED


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
    allowed_workloads: frozenset[str]
    enabled: bool
    environments: frozenset[str]
    max_attempts_same_route: int = 2
    transient_backoff_s: float = 2.0
    retry_after_safe_maximum_s: float = 30.0
    provider_endpoints: tuple[str, ...] = ()
    provider_group: str | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.route_id, "route_id")
        _validate_identifier(self.provider, "provider")
        if self.provider_group is not None:
            _validate_identifier(self.provider_group, "provider_group")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must not be empty")
        if not self.capabilities:
            raise ValueError("capabilities must not be empty")
        if not self.latency_classes:
            raise ValueError("latency_classes must not be empty")
        if not self.privacy_classes:
            raise ValueError("privacy_classes must not be empty")
        if not self.allowed_workloads or not all(_IDENTIFIER_RE.fullmatch(v) for v in self.allowed_workloads):
            raise ValueError("allowed_workloads must contain safe identifiers")
        if not self.environments or not all(_IDENTIFIER_RE.fullmatch(v) for v in self.environments):
            raise ValueError("environments must contain safe identifiers")
        if self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        if self.max_attempts_same_route < 1 or self.max_attempts_same_route > 3:
            raise ValueError("max_attempts_same_route must be between 1 and 3")
        if self.transient_backoff_s < 0 or self.retry_after_safe_maximum_s < 0:
            raise ValueError("retry timing must be non-negative")
        endpoint_pattern = re.compile(r"^[a-z0-9][a-z0-9._-]*(?:/[a-z0-9][a-z0-9._-]*)?$")
        if (
            len(set(self.provider_endpoints)) != len(self.provider_endpoints)
            or any(not endpoint_pattern.fullmatch(value) for value in self.provider_endpoints)
            or (self.provider_endpoints and self.provider != "openrouter")
            or (self.provider == "openrouter" and len(self.provider_endpoints) != 1)
        ):
            raise ValueError("provider_endpoints must be unique OpenRouter endpoint slugs")
        credential_lower = (self.credential_ref or "").strip().lower()
        if (
            not _IDENTIFIER_RE.fullmatch(self.credential_ref or "")
            or credential_lower.startswith(_SECRET_VALUE_PREFIXES)
        ):
            raise ValueError("credential_ref must be a non-secret reference name")


def _incompatibilities(
    request: InferenceRequest, route: RouteCandidate, *, environment: str,
    available_credential_refs: frozenset[str] | None,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not route.enabled:
        reasons.append("disabled")
    if environment not in route.environments:
        reasons.append("environment")
    if request.workload not in route.allowed_workloads:
        reasons.append("workload")
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
    if available_credential_refs is not None and route.credential_ref not in available_credential_refs:
        reasons.append("credential")
    return tuple(reasons)


def resolve_eligible_routes(
    request: InferenceRequest,
    routes: Iterable[RouteCandidate],
    *,
    environment: str = "development",
    available_credential_refs: frozenset[str] | None = None,
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
        reasons = _incompatibilities(
            request, route, environment=environment,
            available_credential_refs=available_credential_refs,
        )
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
        FailureClass.UNKNOWN_UNCLASSIFIED,
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
    fallback_decision: FailureAction | None = None,
    diagnostic: str | None = None,
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
        "fallback_decision": fallback_decision.value if fallback_decision else None,
        "diagnostic": diagnostic,
    }
