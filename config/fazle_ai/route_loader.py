"""Load version-controlled Phase 3 request and route metadata."""
from __future__ import annotations

from dataclasses import dataclass

import yaml

from config.fazle_ai.contracts import (
    Capability, CostClass, InferenceRequest, LatencyClass, PrivacyClass,
    RouteCandidate, TransportKind,
)


@dataclass(frozen=True)
class RoutingPlan:
    request_defaults: dict
    routes: tuple[RouteCandidate, ...]

    def new_request(self, *, correlation_ref: str, context_version: str) -> InferenceRequest:
        return InferenceRequest(
            **self.request_defaults,
            correlation_ref=correlation_ref,
            context_version=context_version,
        )


def _enum_set(enum_type, values, field):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field} must be a non-empty list")
    try:
        return frozenset(enum_type(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}") from exc


def load_routing_plan(path: str) -> RoutingPlan:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    block = raw.get("routing_contract") if isinstance(raw, dict) else None
    if not isinstance(block, dict):
        raise ValueError("routing_contract must be an object")
    required = {
        "workload", "required_capabilities", "max_cost_class", "latency_class",
        "required_context_tokens", "privacy_class", "fallback_eligible",
        "retry_eligible", "routes",
    }
    if not required.issubset(block):
        raise ValueError("routing_contract is missing required fields")
    defaults = {
        "workload": block["workload"],
        "required_capabilities": _enum_set(Capability, block["required_capabilities"], "required_capabilities"),
        "max_cost_class": CostClass(block["max_cost_class"]),
        "latency_class": LatencyClass(block["latency_class"]),
        "required_context_tokens": int(block["required_context_tokens"]),
        "privacy_class": PrivacyClass(block["privacy_class"]),
        "fallback_eligible": block["fallback_eligible"],
        "retry_eligible": block["retry_eligible"],
    }
    routes = []
    route_required = {
        "route_id", "provider", "model", "transport", "capabilities",
        "cost_class", "latency_classes", "max_context_tokens",
        "privacy_classes", "credential_ref", "allowed_workloads", "enabled",
        "environments", "max_attempts_same_route", "transient_backoff_s",
        "retry_after_safe_maximum_s",
    }
    for item in block["routes"]:
        if not isinstance(item, dict) or not route_required.issubset(item):
            raise ValueError("routing route is missing explicit fields")
        routes.append(RouteCandidate(
            route_id=item["route_id"], provider=item["provider"], model=item["model"],
            transport=TransportKind(item["transport"]),
            capabilities=_enum_set(Capability, item["capabilities"], "capabilities"),
            cost_class=CostClass(item["cost_class"]),
            latency_classes=_enum_set(LatencyClass, item["latency_classes"], "latency_classes"),
            max_context_tokens=int(item["max_context_tokens"]),
            privacy_classes=_enum_set(PrivacyClass, item["privacy_classes"], "privacy_classes"),
            credential_ref=item["credential_ref"],
            allowed_workloads=frozenset(item["allowed_workloads"]),
            enabled=bool(item["enabled"]),
            environments=frozenset(item["environments"]),
            max_attempts_same_route=int(item["max_attempts_same_route"]),
            transient_backoff_s=float(item["transient_backoff_s"]),
            retry_after_safe_maximum_s=float(item["retry_after_safe_maximum_s"]),
        ))
    if not routes:
        raise ValueError("routing_contract.routes must not be empty")
    return RoutingPlan(defaults, tuple(routes))
