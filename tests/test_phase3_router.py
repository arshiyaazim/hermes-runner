"""Phase 3B deterministic fallback executor tests (no provider calls)."""

from __future__ import annotations

import pytest

from config.fazle_ai.contracts import (
    Capability, CostClass, FailureClass, InferenceRequest, LatencyClass,
    PrivacyClass, RouteCandidate, TransportKind,
)
from config.fazle_ai.router import ProviderResult, RoutingEngine, RoutingExhausted
from config.fazle_ai.route_loader import load_routing_plan


def _request(**overrides):
    values = dict(
        workload="conversational_reply",
        required_capabilities=frozenset({Capability.TEXT_GENERATION}),
        max_cost_class=CostClass.STANDARD,
        latency_class=LatencyClass.INTERACTIVE,
        required_context_tokens=1000,
        privacy_class=PrivacyClass.APPROVED_EXTERNAL,
        fallback_eligible=True,
        retry_eligible=True,
        correlation_ref="message:7",
        context_version="ctx-v1",
    )
    values.update(overrides)
    return InferenceRequest(**values)


def _route(route_id, **overrides):
    values = dict(
        route_id=route_id, provider=route_id, model=f"{route_id}/model",
        transport=TransportKind.PRIVATE_GATEWAY,
        capabilities=frozenset({Capability.TEXT_GENERATION}),
        cost_class=CostClass.LOW,
        latency_classes=frozenset({LatencyClass.INTERACTIVE}),
        max_context_tokens=8000,
        privacy_classes=frozenset({PrivacyClass.APPROVED_EXTERNAL}),
        credential_ref=f"{route_id}_api_key",
        allowed_workloads=frozenset({"conversational_reply"}),
        enabled=True,
        environments=frozenset({"development", "test", "production"}),
    )
    values.update(overrides)
    return RouteCandidate(**values)


class ScriptedCall:
    def __init__(self, scripts):
        self.scripts = {name: list(results) for name, results in scripts.items()}
        self.calls = []

    def __call__(self, route, attempt_number):
        self.calls.append((route.route_id, attempt_number))
        return self.scripts[route.route_id].pop(0)


def _failed(kind, *, retry_after_s=None):
    return ProviderResult.failure(kind, retry_after_s=retry_after_s)


def test_preferred_route_success_returns_one_final_reply_and_audit():
    call = ScriptedCall({"preferred": [ProviderResult.success("hello", input_tokens=3, output_tokens=1)]})
    result = RoutingEngine((_route("preferred"),)).execute(_request(), call)
    assert result.content == "hello"
    assert call.calls == [("preferred", 1)]
    assert result.selected_route_id == "preferred"
    assert len(result.attempt_audits) == 1
    assert result.attempt_audits[0]["outcome"] == "success"


@pytest.mark.parametrize("failure", [FailureClass.QUOTA_EXHAUSTED, FailureClass.MODEL_UNAVAILABLE])
def test_permanent_capacity_failure_advances_without_retry(failure):
    call = ScriptedCall({
        "preferred": [_failed(failure)],
        "fallback": [ProviderResult.success("fallback")],
    })
    result = RoutingEngine((_route("preferred"), _route("fallback"))).execute(_request(), call)
    assert result.content == "fallback"
    assert call.calls == [("preferred", 1), ("fallback", 1)]

def test_quota_failure_skips_remaining_models_for_same_provider_group():
    call = ScriptedCall({
        "model-a": [_failed(FailureClass.QUOTA_EXHAUSTED)],
        "provider-2": [ProviderResult.success("fallback")],
    })
    result = RoutingEngine((
        _route("model-a", provider_group="provider-config-1"),
        _route("model-b", provider_group="provider-config-1"),
        _route("provider-2"),
    )).execute(_request(), call)
    assert result.content == "fallback"
    assert call.calls == [("model-a", 1), ("provider-2", 1)]


def test_timeout_retries_once_then_advances():
    call = ScriptedCall({
        "preferred": [_failed(FailureClass.TIMEOUT), _failed(FailureClass.TIMEOUT)],
        "fallback": [ProviderResult.success("ok")],
    })
    result = RoutingEngine((_route("preferred"), _route("fallback")), sleep=lambda _: None).execute(_request(), call)
    assert result.content == "ok"
    assert call.calls == [("preferred", 1), ("preferred", 2), ("fallback", 1)]


def test_route_retry_policy_can_forbid_same_route_retry():
    call = ScriptedCall({
        "preferred": [_failed(FailureClass.TIMEOUT)],
        "fallback": [ProviderResult.success("ok")],
    })
    RoutingEngine((
        _route("preferred", max_attempts_same_route=1), _route("fallback")
    )).execute(_request(), call)
    assert call.calls == [("preferred", 1), ("fallback", 1)]


def test_rate_limit_honors_only_one_bounded_retry():
    call = ScriptedCall({
        "preferred": [
            _failed(FailureClass.RATE_LIMITED, retry_after_s=0.01),
            _failed(FailureClass.RATE_LIMITED, retry_after_s=0.01),
        ],
        "fallback": [ProviderResult.success("ok")],
    })
    sleeps = []
    result = RoutingEngine((_route("preferred"), _route("fallback")), sleep=sleeps.append).execute(_request(), call)
    assert result.content == "ok"
    assert sleeps == [0.01]
    assert call.calls == [("preferred", 1), ("preferred", 2), ("fallback", 1)]


def test_unbounded_or_absent_retry_after_advances_immediately():
    for retry_after in (None, 120):
        call = ScriptedCall({
            "preferred": [_failed(FailureClass.RATE_LIMITED, retry_after_s=retry_after)],
            "fallback": [ProviderResult.success("ok")],
        })
        RoutingEngine((_route("preferred"), _route("fallback")), sleep=lambda _: None).execute(_request(), call)
        assert call.calls == [("preferred", 1), ("fallback", 1)]


def test_auth_failure_fails_closed_without_trying_another_credential():
    call = ScriptedCall({
        "preferred": [_failed(FailureClass.AUTHENTICATION_FAILED)],
        "fallback": [ProviderResult.success("must-not-run")],
    })
    with pytest.raises(RoutingExhausted) as exc:
        RoutingEngine((_route("preferred"), _route("fallback"))).execute(_request(), call)
    assert exc.value.failure_class is FailureClass.AUTHENTICATION_FAILED
    assert call.calls == [("preferred", 1)]


def test_malformed_response_uses_only_a_compatible_fallback():
    incompatible = _route("visionless", capabilities=frozenset({Capability.TEXT_GENERATION}))
    compatible = _route("vision", capabilities=frozenset({Capability.TEXT_GENERATION, Capability.VISION}))
    request = _request(required_capabilities=frozenset({Capability.TEXT_GENERATION, Capability.VISION}))
    call = ScriptedCall({"vision": [_failed(FailureClass.MALFORMED_RESPONSE)]})
    with pytest.raises(RoutingExhausted):
        RoutingEngine((incompatible, compatible)).execute(request, call)
    assert call.calls == [("vision", 1)]


def test_all_routes_exhausted_preserves_audits_without_synthesizing_reply():
    call = ScriptedCall({
        "one": [_failed(FailureClass.QUOTA_EXHAUSTED)],
        "two": [_failed(FailureClass.PROVIDER_UNAVAILABLE)],
    })
    with pytest.raises(RoutingExhausted) as exc:
        RoutingEngine((_route("one"), _route("two"))).execute(_request(), call)
    assert exc.value.content == ""
    assert len(exc.value.attempt_audits) == 2
    assert [a["fallback_number"] for a in exc.value.attempt_audits] == [0, 1]


@pytest.mark.parametrize(
    "failure",
    [
        FailureClass.INVALID_REQUEST_OR_POLICY_REJECTION,
        FailureClass.SAFETY_OR_PROVIDER_REFUSAL,
        FailureClass.INTERNAL_APPLICATION_DEFECT,
    ],
)
def test_non_fallback_failures_stop_on_first_route(failure):
    call = ScriptedCall({"one": [_failed(failure)], "two": [ProviderResult.success("bad")]})
    with pytest.raises(RoutingExhausted) as exc:
        RoutingEngine((_route("one"), _route("two"))).execute(_request(), call)
    assert exc.value.failure_class is failure
    assert call.calls == [("one", 1)]


def test_engine_skips_missing_credential_before_call():
    call = ScriptedCall({"two": [ProviderResult.success("ok")]})
    result = RoutingEngine(
        (_route("one"), _route("two")),
        available_credential_refs=frozenset({"two_api_key"}),
    ).execute(_request(), call)
    assert result.content == "ok"
    assert call.calls == [("two", 1)]


def test_failure_audit_records_fallback_decision():
    call = ScriptedCall({
        "one": [_failed(FailureClass.QUOTA_CREDIT_EXHAUSTED)],
        "two": [ProviderResult.success("ok")],
    })
    result = RoutingEngine((_route("one"), _route("two"))).execute(_request(), call)
    assert result.attempt_audits[0]["fallback_decision"] == "next_route"
    assert result.final_route_id == "two"


def test_retry_disabled_advances_after_timeout_without_second_attempt():
    call = ScriptedCall({
        "one": [_failed(FailureClass.TIMEOUT)],
        "two": [ProviderResult.success("ok")],
    })
    RoutingEngine((_route("one"), _route("two"))).execute(_request(retry_eligible=False), call)
    assert call.calls == [("one", 1), ("two", 1)]


def test_versioned_workload_policy_loads_request_defaults_and_explicit_routes():
    plan = load_routing_plan("config/fazle-ai/workloads/hermes-runner.yaml")
    request = plan.new_request(correlation_ref="session:abc", context_version="ctx-v2")
    assert request.workload == "administrative_reasoning"
    assert Capability.STRUCTURED_OUTPUT in request.required_capabilities
    assert plan.routes[0].route_id == "omniroute-gemini-3.1-flash-lite"
    assert plan.routes[0].model == "gemini/gemini-3.1-flash-lite"
    assert plan.routes[0].transport is TransportKind.DIRECT_PROVIDER
    assert plan.routes[-1].transport is TransportKind.LOCAL
    assert all(route.credential_ref for route in plan.routes)


def test_free_omniroute_route_is_primary_and_paid_openrouter_is_fallback_only():
    plan = load_routing_plan("config/fazle-ai/workloads/hermes-runner.yaml")
    primary = plan.routes[0]
    assert primary.provider == "omniroute"
    assert primary.credential_ref == "omniroute_api_key"
    assert primary.cost_class is CostClass.LOW
    assert all(
        route.provider == "openrouter"
        for route in plan.routes
        if route.route_id != primary.route_id and route.enabled
    )
    # The paid route must remain reachable, but only strictly after the free one.
    paid = [i for i, route in enumerate(plan.routes) if route.provider == "openrouter"]
    assert paid and min(paid) > 0


def test_every_policy_route_declares_compatibility_and_retry_ownership():
    import yaml

    with open("config/fazle-ai/workloads/hermes-runner.yaml", encoding="utf-8") as handle:
        routes = yaml.safe_load(handle)["routing_contract"]["routes"]
    required = {
        "route_id", "provider", "model", "transport", "capabilities",
        "cost_class", "latency_classes", "max_context_tokens",
        "privacy_classes", "credential_ref", "allowed_workloads", "enabled",
        "environments", "max_attempts_same_route", "transient_backoff_s",
        "retry_after_safe_maximum_s",
    }
    assert routes
    assert all(required.issubset(route) for route in routes)


def test_enabled_policy_rejects_route_metadata_missing_required_fields(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("version: '1.0'\nrouting_contract: {workload: x}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_routing_plan(str(path))
