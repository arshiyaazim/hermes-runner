"""Read-only client for Core's validated Phase 3 route registry."""
from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config.fazle_ai.contracts import (
    Capability, CostClass, LatencyClass, PrivacyClass, RouteCandidate, TransportKind,
)

_TRANSPORTS = {
    "omniroute_gateway": TransportKind.PRIVATE_GATEWAY,
    "direct_provider": TransportKind.DIRECT_PROVIDER,
    "local_ollama": TransportKind.LOCAL,
    "hermes_runner_adapter": TransportKind.PRIVATE_GATEWAY,
}
_PROVIDER_ENV = {
    "omniroute": ("OMNIROUTE_API_KEY", "HERMES_CUSTOM_127_0_0_1_20128_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "minimax": ("MINIMAX_API_KEY",),
}


class RegistryUnavailable(RuntimeError):
    pass


def _json_request(url: str, bearer: str, *, body=None, timeout=10):
    encoded = None if body is None else json.dumps(body).encode()
    request = Request(url, data=encoded, headers={
        "Authorization": f"Bearer {bearer}", "Accept": "application/json",
        **({"Content-Type": "application/json"} if encoded is not None else {}),
    }, method="POST" if encoded is not None else "GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        raise RegistryUnavailable(f"registry request failed: {type(exc).__name__}") from None


def load_registry_routes(base_url: str, bearer: str, workload: str):
    payload = _json_request(
        f"{base_url.rstrip('/')}/api/assistant/ops/ai-policy/runner/{workload}", bearer,
    )
    if payload.get("source") != "database" or payload.get("workload") != workload:
        raise RegistryUnavailable("registry returned an invalid policy envelope")
    routes = []
    for item in payload.get("routes") or []:
        try:
            routes.append(RouteCandidate(
                route_id=item["route_id"], provider=item["provider"], model=item["model"],
                transport=_TRANSPORTS[item["transport"]],
                capabilities=frozenset(Capability(value) for value in item["capabilities"]),
                cost_class=CostClass(item["cost_class"]),
                latency_classes=frozenset(LatencyClass(value) for value in item["latency_classes"]),
                max_context_tokens=int(item["context_limit"]),
                privacy_classes=frozenset(PrivacyClass(value) for value in item["privacy_classes"]),
                credential_ref=item["credential_ref"], allowed_workloads=frozenset(item["allowed_workloads"]),
                enabled=bool(item["enabled"]), environments=frozenset(item["environments"]),
                max_attempts_same_route=int(item["max_attempts"]),
                transient_backoff_s=float(item["retry_backoff_seconds"]),
                retry_after_safe_maximum_s=30.0,
                provider_endpoints=tuple(item.get("provider_endpoints") or ()),
            ))
        except (KeyError, TypeError, ValueError):
            raise RegistryUnavailable("registry returned an invalid route") from None
    if not routes:
        raise RegistryUnavailable("registry returned no routes")
    return tuple(routes)


def resolve_route_credential(base_url: str, bearer: str, workload: str, route, env: dict):
    if route.credential_ref == "local_runtime":
        return dict(env)
    names = _PROVIDER_ENV.get(route.provider)
    if not names:
        raise RegistryUnavailable("route provider has no credential adapter")
    payload = _json_request(
        f"{base_url.rstrip('/')}/api/assistant/ops/ai-policy/resolve-secret", bearer,
        body={"consumer": "runner", "workload": workload, "credential_ref": route.credential_ref},
    )
    value = payload.get("secret_value")
    if not isinstance(value, str) or not value:
        raise RegistryUnavailable("registry credential is unavailable")
    result = dict(env)
    for name in names:
        result[name] = value
    return result
