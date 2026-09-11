"""config.fazle_ai

Non-secret AI routing policy and bounded retry/fallback logic for
hermes-runner and (future) other Fazle-AI workloads. All policy values
are version-controlled; no secret values are stored in this package.

See config/fazle-ai/README.md for usage.
"""

from __future__ import annotations

from config.fazle_ai.contracts import (
    AttemptOutcome,
    Capability,
    CostClass,
    FailureAction,
    FailureClass,
    InferenceRequest,
    LatencyClass,
    NoCompatibleRoute,
    PrivacyClass,
    RouteCandidate,
    TransportKind,
    build_attempt_audit,
    failure_action,
    resolve_eligible_routes,
)

from config.fazle_ai.policy_loader import (
    Policy,
    PolicyValidationError,
    RetryRule,
    SecretRef,
    WorkloadConfig,
    load_policy,
    load_workload,
    redact_for_log,
    validate_workload_against_policy,
)
from config.fazle_ai.retry_policy import (
    FailureClass as LegacyFailureClass,
    RetryDecision,
    decide_retry,
    execute_with_retries,
)
from config.fazle_ai.router import (
    ProviderResult,
    RoutingEngine,
    RoutingExhausted,
    RoutingResult,
)

__all__ = [
    "AttemptOutcome",
    "Capability",
    "CostClass",
    "FailureClass",
    "FailureAction",
    "InferenceRequest",
    "LatencyClass",
    "LegacyFailureClass",
    "NoCompatibleRoute",
    "Policy",
    "PolicyValidationError",
    "RetryDecision",
    "RetryRule",
    "PrivacyClass",
    "ProviderResult",
    "RouteCandidate",
    "RoutingEngine",
    "RoutingExhausted",
    "RoutingResult",
    "SecretRef",
    "TransportKind",
    "WorkloadConfig",
    "decide_retry",
    "build_attempt_audit",
    "execute_with_retries",
    "failure_action",
    "load_policy",
    "load_workload",
    "redact_for_log",
    "resolve_eligible_routes",
    "validate_workload_against_policy",
]
