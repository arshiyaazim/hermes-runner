import unittest
from unittest.mock import MagicMock, patch

import server


class TestGatewayRouting(unittest.TestCase):
    def test_defaults_to_omniroute_for_hermes(self):
        cfg = server.resolve_gateway_target({
            "HERMES_GATEWAY_TARGET": "auto",
            "OMNIROUTE_BASE_URL": "http://127.0.0.1:20128/v1",
            "NINE_ROUTER_BASE_URL": "http://127.0.0.1:20129/v1",
        })
        self.assertEqual(cfg["target"], "omniroute")
        self.assertEqual(cfg["base_url"], "http://127.0.0.1:20128/v1")

    def test_rejects_recursive_gateway_topology(self):
        with self.assertRaises(ValueError):
            server.validate_gateway_topology({
                "omniroute": "http://127.0.0.1:20129/v1",
                "9router": "http://127.0.0.1:20128/v1",
            })

    def test_redacts_sensitive_headers(self):
        redacted = server.redact_sensitive_headers({
            "Authorization": "Bearer very-secret-token",
            "x-api-key": "top-secret",
            "Accept": "application/json",
        })
        self.assertEqual(redacted["Authorization"], "Bearer [REDACTED]")
        self.assertEqual(redacted["x-api-key"], "[REDACTED]")
        self.assertEqual(redacted["Accept"], "application/json")

    @patch("server.subprocess.run")
    def test_run_hermes_uses_selected_gateway_in_subprocess_env(self, mock_run):
        result = MagicMock()
        result.stdout = "reply"
        result.stderr = "session_id: gateway-session"
        result.returncode = 0
        mock_run.return_value = result

        with patch.dict(
            "os.environ",
            {"HERMES_GATEWAY_TARGET": "9router", "OMNIROUTE_BASE_URL": "http://127.0.0.1:20128/v1", "NINE_ROUTER_BASE_URL": "http://127.0.0.1:20129/v1"},
            clear=False,
        ):
            server.run_hermes(None, "hello", "helpful", force_mode="READ")

        env = mock_run.call_args.kwargs["env"]
        self.assertEqual(env["HERMES_GATEWAY_TARGET"], "9router")
        self.assertEqual(env["NINE_ROUTER_BASE_URL"], "http://127.0.0.1:20129/v1")
        self.assertEqual(env["OMNIROUTE_BASE_URL"], "http://127.0.0.1:20128/v1")

    @patch("server.subprocess.run")
    def test_run_hermes_keeps_default_gateway_compatible_legacy_path(self, mock_run):
        result = MagicMock()
        result.stdout = "reply"
        result.stderr = "session_id: legacy-session"
        result.returncode = 0
        mock_run.return_value = result

        with patch.dict("os.environ", {}, clear=False):
            server.run_hermes(None, "hello", "helpful", force_mode="READ")

        env = mock_run.call_args.kwargs["env"]
        self.assertEqual(env["HERMES_GATEWAY_TARGET"], "omniroute")
        self.assertEqual(env["OMNIROUTE_BASE_URL"], "http://127.0.0.1:20128/v1")

    def test_invalid_gateway_target_is_rejected(self):
        with self.assertRaises(ValueError):
            server.resolve_gateway_target({"HERMES_GATEWAY_TARGET": "invalid"})

    def test_provider_override_is_preserved_with_gateway_selection(self):
        cfg = server.apply_gateway_selection({
            "HERMES_GATEWAY_TARGET": "omniroute",
            "OMNIROUTE_BASE_URL": "http://127.0.0.1:20128/v1",
            "NINE_ROUTER_BASE_URL": "http://127.0.0.1:20129/v1",
            "HERMES_RUNNER_PROVIDER": "minimax",
        })
        self.assertEqual(cfg["HERMES_GATEWAY_TARGET"], "omniroute")
        self.assertEqual(cfg["HERMES_RUNNER_PROVIDER"], "minimax")


if __name__ == "__main__":
    unittest.main()
