"""Phase 3A shared semantic contract tests.

These tests are provider-free and runtime-free.  They define what every Fazle
AI caller/adapter must agree on before any existing production selector is
changed.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

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


def _request(**overrides) -> InferenceRequest:
    values = {
        "workload": "conversational_reply",
        "required_capabilities": frozenset({Capability.TEXT_GENERATION}),
        "max_cost_class": CostClass.STANDARD,
        "latency_class": LatencyClass.INTERACTIVE,
        "required_context_tokens": 8_000,
        "privacy_class": PrivacyClass.APPROVED_EXTERNAL,
        "fallback_eligible": True,
        "retry_eligible": True,
        "correlation_ref": "message:42",
        "context_version": "ctx-v1",
    }
    values.update(overrides)
    return InferenceRequest(**values)


def _route(route_id: str, **overrides) -> RouteCandidate:
    values = {
        "route_id": route_id,
        "provider": "omniroute",
        "model": "gemini/gemini-3.1-flash-lite",
        "transport": TransportKind.PRIVATE_GATEWAY,
        "capabilities": frozenset({Capability.TEXT_GENERATION, Capability.STRUCTURED_OUTPUT}),
        "cost_class": CostClass.LOW,
        "latency_classes": frozenset({LatencyClass.INTERACTIVE, LatencyClass.BACKGROUND}),
        "max_context_tokens": 32_000,
        "privacy_classes": frozenset({PrivacyClass.PRIVATE_GATEWAY, PrivacyClass.APPROVED_EXTERNAL}),
        "credential_ref": "omniroute_api_key",
        "allowed_workloads": frozenset({"conversational_reply"}),
        "enabled": True,
        "environments": frozenset({"development", "test", "production"}),
    }
    values.update(overrides)
    return RouteCandidate(**values)


def test_request_is_immutable_and_business_code_names_a_workload_not_a_model():
    request = _request()
    assert request.workload == "conversational_reply"
    assert not hasattr(request, "provider")
    assert not hasattr(request, "model")
    with pytest.raises(FrozenInstanceError):
        request.workload = "changed"  # type: ignore[misc]


def test_preferred_compatible_route_order_is_preserved():
    routes = (_route("preferred"), _route("fallback", provider="ollama-local", model="qwen3:8b"))
    assert resolve_eligible_routes(_request(), routes) == routes


def test_incompatible_capability_fallback_is_rejected():
    request = _request(required_capabilities=frozenset({Capability.VISION, Capability.TEXT_GENERATION}))
    route = _route("text-only", capabilities=frozenset({Capability.TEXT_GENERATION}))
    with pytest.raises(NoCompatibleRoute) as exc:
        resolve_eligible_routes(request, (route,))
    assert "capability" in str(exc.value)


def test_context_requirement_rejects_too_small_route():
    with pytest.raises(NoCompatibleRoute) as exc:
        resolve_eligible_routes(_request(required_context_tokens=64_000), (_route("short"),))
    assert "context" in str(exc.value)


def test_cost_ceiling_and_latency_requirement_are_enforced():
    expensive = _route("expensive", cost_class=CostClass.HIGH)
    slow = _route("slow", latency_classes=frozenset({LatencyClass.BACKGROUND}))
    with pytest.raises(NoCompatibleRoute) as exc:
        resolve_eligible_routes(_request(), (expensive, slow))
    assert "cost" in str(exc.value)
    assert "latency" in str(exc.value)


def test_privacy_restriction_rejects_external_route():
    external = _route(
        "external",
        transport=TransportKind.DIRECT_PROVIDER,
        privacy_classes=frozenset({PrivacyClass.APPROVED_EXTERNAL}),
    )
    with pytest.raises(NoCompatibleRoute) as exc:
        resolve_eligible_routes(_request(privacy_class=PrivacyClass.LOCAL_ONLY), (external,))
    assert "privacy" in str(exc.value)


def test_disabled_wrong_environment_and_wrong_workload_routes_are_rejected():
    routes = (
        _route("disabled", enabled=False),
        _route("staging", environments=frozenset({"staging"})),
        _route("other", allowed_workloads=frozenset({"structured_extraction"})),
    )
    with pytest.raises(NoCompatibleRoute) as exc:
        resolve_eligible_routes(_request(), routes, environment="development")
    assert all(word in str(exc.value) for word in ("disabled", "environment", "workload"))


def test_unavailable_credential_is_rejected_before_provider_call():
    with pytest.raises(NoCompatibleRoute) as exc:
        resolve_eligible_routes(
            _request(), (_route("missing"),),
            available_credential_refs=frozenset(),
        )
    assert "credential" in str(exc.value)


def test_fallback_disabled_returns_only_first_compatible_route():
    routes = (_route("preferred"), _route("compatible-fallback"))
    assert resolve_eligible_routes(_request(fallback_eligible=False), routes) == routes[:1]


@pytest.mark.parametrize(
    ("failure", "retry_eligible", "expected"),
    [
        (FailureClass.TIMEOUT, True, FailureAction.RETRY_THEN_NEXT),
        (FailureClass.PROVIDER_OUTAGE, True, FailureAction.RETRY_THEN_NEXT),
        (FailureClass.RATE_LIMITED, True, FailureAction.BOUNDED_RETRY_THEN_NEXT),
        (FailureClass.QUOTA_EXHAUSTED, True, FailureAction.NEXT_ROUTE),
        (FailureClass.MODEL_UNAVAILABLE, True, FailureAction.NEXT_ROUTE),
        (FailureClass.MALFORMED_RESPONSE, True, FailureAction.NEXT_COMPATIBLE_ROUTE),
        (FailureClass.AUTHENTICATION_FAILED, True, FailureAction.FAIL_CLOSED),
        (FailureClass.AUTHORIZATION_FAILED, True, FailureAction.FAIL_CLOSED),
        (FailureClass.SAFETY_REFUSAL, True, FailureAction.FAIL_CLOSED),
        (FailureClass.INTERNAL_APPLICATION_DEFECT, True, FailureAction.INTERNAL_ERROR),
        (FailureClass.TIMEOUT, False, FailureAction.NEXT_ROUTE),
    ],
)
def test_failure_taxonomy_has_deterministic_actions(failure, retry_eligible, expected):
    assert failure_action(failure, retry_eligible=retry_eligible) is expected


def test_attempt_audit_contains_required_safe_routing_fields():
    audit = build_attempt_audit(
        request=_request(),
        route=_route("preferred"),
        fallback_number=0,
        attempt_number=1,
        latency_ms=123,
        outcome=AttemptOutcome.SUCCESS,
        failure_class=None,
        input_tokens=50,
        output_tokens=12,
    )
    assert audit == {
        "event": "ai_routing_attempt",
        "correlation_ref": "message:42",
        "workload": "conversational_reply",
        "context_version": "ctx-v1",
        "route_id": "preferred",
        "transport": "private_gateway",
        "provider": "omniroute",
        "model": "gemini/gemini-3.1-flash-lite",
        "fallback_number": 0,
        "attempt_number": 1,
        "latency_ms": 123,
        "input_tokens": 50,
        "output_tokens": 12,
        "outcome": "success",
        "failure_class": None,
        "fallback_decision": None,
        "diagnostic": None,
    }
    assert "credential_ref" not in audit
    assert not any("key" in name or "secret" in name or "authorization" in name for name in audit)


def test_attempt_audit_rejects_invalid_ordinals_and_latency():
    with pytest.raises(ValueError):
        build_attempt_audit(
            request=_request(), route=_route("r"), fallback_number=-1,
            attempt_number=0, latency_ms=-1, outcome=AttemptOutcome.FAILURE,
            failure_class=FailureClass.TIMEOUT,
        )


def test_route_requires_non_secret_credential_reference_name():
    with pytest.raises(ValueError):
        _route("bad-secret", credential_ref="sk-this-is-a-secret-value")


def test_openrouter_route_requires_exactly_one_provider_endpoint():
    with pytest.raises(ValueError, match="provider_endpoints"):
        _route("openrouter-unpinned", provider="openrouter", provider_endpoints=())
    with pytest.raises(ValueError, match="provider_endpoints"):
        _route(
            "openrouter-multiple", provider="openrouter",
            provider_endpoints=("open-inference/fp8", "baidu/fp8"),
        )


def test_empty_workload_and_route_identifiers_are_rejected():
    with pytest.raises(ValueError):
        _request(workload=" ")
    with pytest.raises(ValueError):
        _route(" ")


def test_failure_taxonomy_uses_phase3b_canonical_values():
    assert FailureClass.AUTHENTICATION_FAILURE.value == "authentication_failure"
    assert FailureClass.QUOTA_CREDIT_EXHAUSTED.value == "quota_credit_exhausted"
    assert FailureClass.PROVIDER_OR_MODEL_UNAVAILABLE.value == "provider_or_model_unavailable"
    assert FailureClass.PROVIDER_OUTAGE_OR_5XX.value == "provider_outage_or_5xx"
    assert FailureClass.MALFORMED_PROVIDER_RESPONSE.value == "malformed_provider_response"
    assert FailureClass.SAFETY_OR_PROVIDER_REFUSAL.value == "safety_or_provider_refusal"
    assert FailureClass.INVALID_REQUEST_OR_POLICY_REJECTION.value == "invalid_request_or_policy_rejection"
    assert FailureClass.INTERNAL_APPLICATION_DEFECT.value == "internal_application_defect"
    assert FailureClass.UNKNOWN_UNCLASSIFIED.value == "unknown_unclassified"


def test_unknown_unclassified_is_fail_closed_not_silent_fallback():
    assert failure_action(
        FailureClass.UNKNOWN_UNCLASSIFIED, retry_eligible=True
    ) is FailureAction.FAIL_CLOSED
