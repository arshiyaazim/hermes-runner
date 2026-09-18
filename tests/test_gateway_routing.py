import unittest

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


if __name__ == "__main__":
    unittest.main()
