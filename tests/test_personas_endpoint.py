import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server


class TestBuildPersonasResponse(unittest.TestCase):
    """Task 3 (2026-08-16): server.build_personas_response() is the logic
    behind GET /personas, factored out of Handler.do_GET the same way
    read_mode_state()/_handle_audit() already are, so it's testable
    without a real HTTP server (matches this test suite's existing
    convention -- see test_audit_endpoint.py's own docstring)."""

    def test_returns_devoted_as_default(self):
        result = server.build_personas_response()
        self.assertEqual(result["default"], "devoted")
        self.assertEqual(result["default"], server.DEFAULT_PERSONA)

    def test_devoted_persona_present_with_correct_key(self):
        result = server.build_personas_response()
        keys = [p["key"] for p in result["personas"]]
        self.assertIn("devoted", keys)

    def test_every_persona_key_has_a_label(self):
        result = server.build_personas_response()
        for p in result["personas"]:
            self.assertIn("key", p)
            self.assertIn("label", p)
            self.assertTrue(p["label"])

    def test_response_contains_only_key_and_label_per_entry(self):
        """No extra fields -- specifically never the prompt text."""
        result = server.build_personas_response()
        for p in result["personas"]:
            self.assertEqual(set(p.keys()), {"key", "label"})

    def test_response_top_level_keys_are_closed(self):
        result = server.build_personas_response()
        self.assertEqual(set(result.keys()), {"personas", "default"})

    def test_no_persona_prompt_text_leaks_into_response(self):
        """The actual safety-critical assertion: every PERSONAS dict VALUE
        (the real prompt text sent to the model) must never appear
        anywhere in the /personas response -- only keys/labels."""
        result = server.build_personas_response()
        serialized = json.dumps(result)
        for prompt_text in server.PERSONAS.values():
            self.assertNotIn(prompt_text, serialized)
        # Spot-check a few distinctive substrings from actual prompt text
        # that would only appear if the wrong dict leaked through.
        self.assertNotIn("guard-dog", serialized)
        self.assertNotIn("Captain Hermes", serialized)
        self.assertNotIn("nya~", serialized)

    def test_no_system_preamble_text_leaks_into_response(self):
        result = server.build_personas_response()
        serialized = json.dumps(result)
        self.assertNotIn("BREAK_GLASS_APPROVED", serialized)
        self.assertNotIn("Boss", serialized)

    def test_persona_count_matches_personas_dict(self):
        result = server.build_personas_response()
        self.assertEqual(len(result["personas"]), len(server.PERSONAS))

    def test_adding_a_runner_persona_is_picked_up_without_code_elsewhere(self):
        """Regression guard for the actual point of Task 3: adding a
        persona to PERSONAS must show up here automatically -- if this
        ever required editing a second list, the drift this task removes
        would silently come back."""
        original = dict(server.PERSONAS)
        try:
            server.PERSONAS["test_temp_persona"] = "A temporary test persona."
            result = server.build_personas_response()
            keys = [p["key"] for p in result["personas"]]
            self.assertIn("test_temp_persona", keys)
        finally:
            server.PERSONAS.clear()
            server.PERSONAS.update(original)


class TestPersonasGetRouteAuthGuard(unittest.TestCase):
    """The /personas route in Handler.do_GET reuses the exact same
    Bearer-secret check already proven for /mode (same guard clause,
    copy-pasted, not a new auth mechanism) -- verified here by asserting
    the source contains that guard rather than a route with no auth
    check, since a raw socket-level 401 test would require spinning up a
    live server (this suite's existing convention avoids that -- see
    test_audit_endpoint.py, test_relay.py: neither tests the HTTP layer
    directly, only the logic functions behind it)."""

    def test_personas_route_guarded_by_runner_secret_like_mode_route(self):
        import inspect
        src = inspect.getsource(server.Handler.do_GET)
        # Both /mode and /personas branches must each contain their own
        # RUNNER_SECRET Bearer check -- not just be present once globally.
        personas_branch = src.split('self.path == "/personas"')[1]
        self.assertIn("RUNNER_SECRET", personas_branch.split("return self._send(200")[0])
        self.assertIn("401", personas_branch.split("return self._send(200")[0])


if __name__ == "__main__":
    unittest.main()
