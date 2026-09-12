"""Offline runtime wiring tests for hermes-runner Phase 3B."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch
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


def test_policy_enabled_uses_ephemeral_no_fallback_profile_without_mutating_source(tmp_path):
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
        assert "fallback_providers" not in config
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


def test_openrouter_route_profile_pins_one_endpoint_and_denies_collection(tmp_path):
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
        assert "fallback_providers" not in config
        assert config["provider_routing"] == {
            "only": ["open-inference/fp8"],
            "require_parameters": True,
            "data_collection": "deny",
        }


def test_openrouter_route_without_endpoint_fails_closed(tmp_path):
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    route = SimpleNamespace(provider="openrouter", provider_endpoints=())
    with pytest.raises(RuntimeError, match="deterministic provider endpoint"):
        with server._phase3_no_fallback_profile({"HERMES_HOME": str(home)}, route):
            pass
