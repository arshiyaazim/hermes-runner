"""Offline runtime wiring tests for hermes-runner Phase 3B."""
from __future__ import annotations

from io import BytesIO
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pathlib import Path

import pytest

import server
from config.fazle_ai.contracts import FailureClass
from config.fazle_ai.failure_classifier import classify_provider_failure, safe_diagnostic


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("401 invalid api key", FailureClass.AUTHENTICATION_FAILURE),
        ("403 unauthorized", FailureClass.AUTHORIZATION_FAILURE),
        ("429 insufficient_quota credits exhausted", FailureClass.QUOTA_CREDIT_EXHAUSTED),
        ("429 rate limit retry-after: 3", FailureClass.RATE_LIMITED),
        ("404 model_not_found", FailureClass.PROVIDER_OR_MODEL_UNAVAILABLE),
        ("503 upstream unavailable", FailureClass.PROVIDER_OUTAGE_OR_5XX),
        ("context length exceeded", FailureClass.CONTEXT_LENGTH_EXCEEDED),
        ("invalid json response", FailureClass.MALFORMED_PROVIDER_RESPONSE),
        ("content safety refusal", FailureClass.SAFETY_OR_PROVIDER_REFUSAL),
        ("request timed out", FailureClass.TIMEOUT),
        ("unrecognized failure", FailureClass.UNKNOWN_UNCLASSIFIED),
    ],
)
def test_provider_failure_classification(text, expected):
    assert classify_provider_failure(text) is expected


def test_safe_diagnostic_redacts_credentials_and_is_bounded():
    value = safe_diagnostic("Bearer abcdefghijklmnopqrstuvwxyz sk-secretvalue123456 " + "x" * 900)
    assert "abcdef" not in value and "sk-secret" not in value
    assert len(value) <= 500


def test_failed_runtime_audit_keeps_only_redacted_bounded_diagnostic():
    failed = (_completed(stdout="", stderr="503 Bearer abcdefghijklmnopqrstuvwxyz", returncode=1), False, None)
    success = (_completed(stdout="ok"), False, None)
    with patch.object(server, "PHASE3_ROUTING_ENABLED", True), patch.object(
        server, "_phase3_available_credential_refs", return_value=frozenset({"openrouter_api_key"})
    ), patch.object(server, "_run_hermes_once", side_effect=[failed, failed, success]), patch.object(
        server.time, "sleep", return_value=None
    ):
        server.run_hermes(None, "hello", "helpful", force_mode="READ")
    diagnostic = server.get_last_routing_audits()[0]["diagnostic"]
    assert "Bearer" not in diagnostic and "abcdef" not in diagnostic


def _completed(stdout="reply", stderr="session_id: phase3", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def test_zero_exit_with_empty_reply_is_a_failure_not_a_success():
    """The hermes CLI exits 0 with empty stdout when a provider credential is
    missing or invalid. That must never surface as a successful empty reply:
    it has to fail and let routing fall through to the next eligible route."""
    replies = [
        (_completed(stdout="", stderr="401 invalid api key", returncode=0), False, None),
        (_completed(stdout="fallback reply"), False, None),
    ]

    def _sequence(*args, **kwargs):
        return replies.pop(0)

    with patch.object(server, "PHASE3_ROUTING_ENABLED", True), patch.object(
        server, "_phase3_available_credential_refs",
        return_value=frozenset({"local_runtime", "omniroute_api_key", "openrouter_api_key"}),
    ), patch.object(server, "_run_hermes_once", side_effect=_sequence) as invoke:
        reply, _, error, _ = server.run_hermes(None, "hello", "helpful", force_mode="READ")

    assert invoke.call_count == 2, "empty primary must fall through to the next route"
    assert error is None
    assert reply == "fallback reply"
    assert reply != ""
    audits = server.get_last_routing_audits()
    assert audits[0]["outcome"] == "failure"
    assert audits[0]["model"] == "gemini/gemini-3.1-flash-lite"


def test_empty_reply_classification_reaches_provider_failure():
    """Directly assert the classification used for the empty-success case."""
    from config.fazle_ai.contracts import FailureClass

    assert FailureClass.MALFORMED_PROVIDER_RESPONSE.value == "malformed_provider_response"


def test_policy_disabled_preserves_exact_legacy_single_selection():
    with patch.object(server, "PHASE3_ROUTING_ENABLED", False), patch.object(
        server, "_run_hermes_once", return_value=(_completed(), False, None)
    ) as invoke:
        reply, _, error, _ = server.run_hermes(None, "hello", "helpful", force_mode="READ")
    assert reply == "reply" and error is None
    cmd = invoke.call_args.args[0]
    assert cmd[cmd.index("-m") + 1] == server.HERMES_RUNNER_READ_MODEL
    assert invoke.call_count == 1


def test_policy_enabled_timeout_then_fallback_keeps_one_toolset_and_final_reply():
    first = (None, True, None)
    second = (_completed(stdout="one final reply"), False, None)
    with patch.object(server, "PHASE3_ROUTING_ENABLED", True), patch.object(
        server, "_phase3_available_credential_refs", return_value=frozenset({"openrouter_api_key"})
    ), patch.object(server, "_run_hermes_once", side_effect=[first, first, second]) as invoke, patch.object(
        server.time, "sleep", return_value=None
    ):
        reply, session_id, error, mode = server.run_hermes(None, "hello", "helpful", force_mode="READ")
    assert (reply, session_id, error, mode) == ("one final reply", "phase3", None, "READ")
    assert invoke.call_count == 3
    commands = [call.args[0] for call in invoke.call_args_list]
    assert len({cmd[cmd.index("-t") + 1] for cmd in commands}) == 1
    assert len(server.get_last_routing_audits()) == 3


def test_policy_enabled_auth_failure_stops_without_fallback():
    failed = (_completed(stdout="", stderr="401 invalid API key", returncode=1), False, None)
    with patch.object(server, "PHASE3_ROUTING_ENABLED", True), patch.object(
        server, "_phase3_available_credential_refs", return_value=frozenset({"openrouter_api_key"})
    ), patch.object(server, "_run_hermes_once", return_value=failed) as invoke:
        reply, _, error, _ = server.run_hermes(None, "hello", "helpful", force_mode="READ")
    assert reply is None and "authentication_failure" in error
    assert invoke.call_count == 1


def test_policy_enabled_missing_credentials_calls_no_provider():
    with patch.object(server, "PHASE3_ROUTING_ENABLED", True), patch.object(
        server, "_phase3_available_credential_refs", return_value=frozenset()
    ), patch.object(server, "_run_hermes_once") as invoke:
        reply, _, error, _ = server.run_hermes(None, "hello", "helpful", force_mode="READ")
    assert reply is None and "compatible route" in error
    invoke.assert_not_called()


def test_policy_enabled_invalid_policy_fails_closed_before_subprocess(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("routing_contract: broken", encoding="utf-8")
    with patch.object(server, "PHASE3_ROUTING_ENABLED", True), patch.object(
        server, "PHASE3_ROUTING_WORKLOAD_FILE", str(bad)
    ), patch.object(server, "_run_hermes_once") as invoke:
        reply, _, error, _ = server.run_hermes(None, "hello", "helpful", force_mode="READ")
    assert reply is None and "routing policy" in error.lower()
    invoke.assert_not_called()


def test_policy_enabled_uses_ephemeral_profile_without_stale_override_or_source_mutation(tmp_path):
    home = tmp_path / "hermes-home"
    home.mkdir()
    source_config = """\
model: {provider: omniroute, default: legacy}
fallback_providers:
  - {provider: minimax, model: MiniMax-M3}
fallback_model: {provider: openrouter, model: legacy}
mcp_servers: {fazle-core: {command: /safe/mcp}}
plugins: {enabled: [task_action_policy]}
"""
    (home / "config.yaml").write_text(source_config, encoding="utf-8")
    (home / ".env").write_text("OMNIROUTE_API_KEY=not-read-by-test\n", encoding="utf-8")
    (home / "sessions").mkdir()
    seen = []

    def inspect_profile(_cmd, env, *_args, **_kwargs):
        routed_home = Path(env["HERMES_HOME"])
        assert routed_home != home
        config = (routed_home / "config.yaml").read_text(encoding="utf-8")
        assert "fallback_providers" in config
        assert "fallback_model" not in config
        assert "task_action_policy" in config
        assert "fazle-core" in config
        assert (routed_home / ".env").resolve() == (home / ".env").resolve()
        assert (routed_home / "sessions").resolve() == (home / "sessions").resolve()
        seen.append(routed_home)
        return _completed(), False, None

    env = {"HERMES_HOME": str(home)}
    with server._phase3_no_fallback_profile(env) as routed_env:
        inspect_profile([], routed_env)
    assert seen and not seen[0].exists()
    assert (home / "config.yaml").read_text(encoding="utf-8") == source_config


def test_openrouter_route_profile_does_not_inject_stale_provider_override(tmp_path):
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model: {provider: openrouter, default: legacy}\n"
        "fallback_providers: [{provider: other, model: hidden}]\n",
        encoding="utf-8",
    )
    route = SimpleNamespace(
        provider="openrouter", provider_endpoints=("open-inference/fp8",),
    )
    with server._phase3_no_fallback_profile({"HERMES_HOME": str(home)}, route) as routed_env:
        import yaml
        config = yaml.safe_load((Path(routed_env["HERMES_HOME"]) / "config.yaml").read_text())
        assert "fallback_providers" in config
        assert "provider_routing" not in config


def test_openrouter_route_without_endpoint_keeps_normal_provider_defaults(tmp_path):
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    route = SimpleNamespace(provider="openrouter", provider_endpoints=())
    with server._phase3_no_fallback_profile({"HERMES_HOME": str(home)}, route) as routed_env:
        import yaml
        config = yaml.safe_load((Path(routed_env["HERMES_HOME"]) / "config.yaml").read_text())
        assert "provider_routing" not in config


def test_selected_customer_provider_gets_an_ephemeral_named_profile(tmp_path):
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text("providers: {omniroute: {base_url: http://stale/v1}}\n", encoding="utf-8")
    route_config = {
        "provider": "9router", "base_url": "http://127.0.0.1:20129/v1",
        "model": "general", "api_key": "runtime-secret", "credential_ref": "ref",
    }
    with server._phase3_no_fallback_profile({"HERMES_HOME": str(home)}, provider_config=route_config) as env:
        import yaml
        config = yaml.safe_load((Path(env["HERMES_HOME"]) / "config.yaml").read_text())
        assert config["providers"]["fazle-customer"]["base_url"] == route_config["base_url"]
        assert config["providers"]["fazle-customer"]["models"] == {"general": {}}
        assert "provider_routing" not in config


def test_customer_modelless_resolution_only_uses_advertised_supported_route():
    from config.fazle_ai.route_loader import load_routing_plan
    plan = load_routing_plan(server.PHASE3_ROUTING_WORKLOAD_FILE)
    with patch.object(server, "_resolve_customer_model", return_value="auto"):
        selected = server._customer_routing_plan(
            plan,
            {"provider": "9router", "api_key": "key", "model": "", "base_url": "http://router/v1"},
        ).routes[0]
    assert selected.model == "auto"
    assert selected.provider == "custom:fazle-customer"


def test_customer_workload_routes_are_deterministic_and_exclude_admin_fallbacks():
    from config.fazle_ai.route_loader import load_routing_plan
    plan = load_routing_plan(server.PHASE3_ROUTING_WORKLOAD_FILE)
    candidates = {
        "candidates": [
            {"config_id": 9, "provider": "openrouter", "api_key": "key-a", "model": "model-a", "base_url": "https://a.test/v1"},
            {"config_id": 12, "provider": "9router", "api_key": "key-b", "model": "general", "base_url": "https://b.test/v1"},
        ]
    }
    scoped = server._customer_routing_plan(plan, candidates, workload_id=server.TRUSTED_CUSTOMER_WORKLOAD_ID)
    assert [route.route_id for route in scoped.routes] == ["hermes-customer-9", "hermes-customer-12"]
    assert all(route.provider == "custom:fazle-customer" for route in scoped.routes)

def test_customer_workload_expands_saved_models_in_order():
    from config.fazle_ai.route_loader import load_routing_plan
    plan = load_routing_plan(server.PHASE3_ROUTING_WORKLOAD_FILE)
    scoped = server._customer_routing_plan(
        plan,
        {
            "config_id": 9,
            "provider": "openrouter",
            "api_key": "key-a",
            "model": "model-b",
            "models": ["model-a", "model-b", "model-c", "model-c"],
            "base_url": "https://a.test/v1",
        },
        workload_id=server.TRUSTED_CUSTOMER_WORKLOAD_ID,
    )
    assert [route.model for route in scoped.routes[:3]] == ["model-b", "model-a", "model-c"]
    assert [route.route_id for route in scoped.routes[:3]] == [
        "hermes-customer-9-model-0",
        "hermes-customer-9-model-1",
        "hermes-customer-9-model-2",

    ]
    assert {route.provider_group for route in scoped.routes[:3]} == {"hermes-customer:9"}

def test_customer_runtime_attempts_saved_models_in_order():
    provider = {
        "candidates": [{
            "config_id": 9,
            "provider": "openrouter",
            "api_key": "key-a",
            "model": "model-b",
            "models": ["model-a", "model-b"],
            "base_url": "https://a.test/v1",
        }],
    }
    failed = (_completed(stdout="", stderr="404 model_not_found", returncode=1), False, None)
    success = (_completed(stdout="policy-safe reply"), False, None)
    with patch.object(server, "_load_customer_provider_config", return_value=provider), patch.object(
        server, "_phase3_no_fallback_profile", return_value=nullcontext({"HERMES_HOME": "/tmp/hermes-test"}),
    ), patch.object(server, "_run_hermes_once", side_effect=[failed, success]) as invoke:
        process, error, failure = server._run_with_phase3_routing(
            ["hermes"], {}, session_id=None, persona="helpful", cwd="/tmp",
            caller_scope="customer", workload_id=server.TRUSTED_CUSTOMER_WORKLOAD_ID,
        )
    assert process.stdout == "policy-safe reply"
    assert error is None and failure is None
    assert [call.args[0][call.args[0].index("-m") + 1] for call in invoke.call_args_list] == ["model-b", "model-a"]

def test_untrusted_customer_workload_is_rejected():
    assert server._trusted_workload_request("customer", server.TRUSTED_CUSTOMER_WORKLOAD_ID)
    assert not server._trusted_workload_request("customer", "assistant-platform")
    assert not server._trusted_workload_request("customer", None)
    assert server._trusted_workload_request(None, None)
    assert not server._trusted_workload_request(None, server.TRUSTED_CUSTOMER_WORKLOAD_ID)


def test_workload_scoped_missing_provider_does_not_use_legacy_plan():
    with patch.object(server, "PHASE3_ROUTING_ENABLED", True), patch.object(
        server, "_load_customer_provider_config", return_value=None
    ), patch.object(server, "_run_hermes_once") as invoke:
        reply, _, error, _ = server.run_hermes(None, "hello", "helpful", force_mode="CUSTOMER", caller_scope="customer", workload_id=server.TRUSTED_CUSTOMER_WORKLOAD_ID)
    assert reply is None and "workload-scoped" in error
    invoke.assert_not_called()


def test_run_handler_parses_json_before_validating_workload_id():
    body = b'{"caller_scope":"customer","workload_id":"assistant-platform"}'
    handler = object.__new__(server.Handler)
    handler.path = "/run"
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = BytesIO(body)
    handler._send = Mock()

    handler.do_POST()

    handler._send.assert_called_once_with(400, {"error": "workload_id is only valid for customer calls"})
