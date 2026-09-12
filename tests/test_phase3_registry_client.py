import io
import json
from types import SimpleNamespace
from unittest.mock import patch

from config.fazle_ai import registry_client


class Response(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *_args): return False


def _route():
    return {
        "route_id": "omni", "provider": "omniroute", "model": "model-a",
        "transport": "omniroute_gateway", "capabilities": ["text_generation", "structured_output", "long_context"],
        "cost_class": "standard", "latency_classes": ["interactive"], "context_limit": 64000,
        "privacy_classes": ["approved_external"], "credential_ref": "omni_ref",
        "allowed_workloads": ["administrative_reasoning"], "enabled": True,
        "environments": ["production"], "max_attempts": 1, "retry_backoff_seconds": 0,
    }


def test_registry_policy_is_read_only_and_converted_to_runner_contract():
    response = {"source": "database", "workload": "administrative_reasoning", "routes": [_route()]}
    with patch.object(registry_client, "urlopen", return_value=Response(json.dumps(response).encode())) as call:
        routes = registry_client.load_registry_routes("http://core", "bearer", "administrative_reasoning")
    assert routes[0].model == "model-a" and routes[0].credential_ref == "omni_ref"
    assert call.call_args.args[0].get_method() == "GET"


def test_registry_credential_is_injected_only_into_selected_provider_environment():
    route = SimpleNamespace(provider="omniroute", credential_ref="omni_ref")
    with patch.object(registry_client, "urlopen", return_value=Response(b'{"secret_value":"value"}')):
        env = registry_client.resolve_route_credential(
            "http://core", "bearer", "administrative_reasoning", route, {"UNCHANGED": "yes"},
        )
    assert env["OMNIROUTE_API_KEY"] == "value"
    assert env["HERMES_CUSTOM_127_0_0_1_20128_API_KEY"] == "value"
    assert env["UNCHANGED"] == "yes"


def test_registry_failure_is_bounded_and_does_not_echo_bearer():
    with patch.object(registry_client, "urlopen", side_effect=TimeoutError("Bearer very-secret")):
        try:
            registry_client.load_registry_routes("http://core", "bearer-value", "administrative_reasoning")
        except Exception as exc:
            assert "bearer-value" not in str(exc) and "very-secret" not in str(exc)
        else:
            raise AssertionError("expected failure")
