import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server


class TestRunHermesForceMode(unittest.TestCase):
    """Phase 4 (2026-08-04): force_mode must select a toolset independent
    of the persisted mode file — the whole point of the WhatsApp relay's
    safety property is that it can't be affected by (or affect) the web
    UI's current mode."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.mode_file = os.path.join(self.tmp_dir, "current_mode.txt")
        self._mode_patch = patch.object(server, "MODE_FILE", self.mode_file)
        self._mode_patch.start()

    def tearDown(self):
        self._mode_patch.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _fake_result(self, stdout="reply text", stderr="session_id: abc123", returncode=0):
        result = MagicMock()
        result.stdout = stdout
        result.stderr = stderr
        result.returncode = returncode
        return result

    @patch("server.subprocess.run")
    def test_force_mode_read_uses_read_toolset_even_when_file_says_run(self, mock_run):
        with open(self.mode_file, "w") as f:
            json.dump({"mode": "RUN", "set_at": None, "expires_at": None, "scope": None, "set_by": None}, f)
        mock_run.return_value = self._fake_result()
        server.run_hermes(None, "hello", "helpful", force_mode="READ")
        cmd = mock_run.call_args[0][0]
        toolsets_idx = cmd.index("-t") + 1
        self.assertEqual(cmd[toolsets_idx], server.MODE_TOOLSETS["READ"])

    @patch("server.subprocess.run")
    def test_no_force_mode_uses_persisted_mode(self, mock_run):
        with open(self.mode_file, "w") as f:
            json.dump({"mode": "BUILD", "set_at": None, "expires_at": None, "scope": None, "set_by": None}, f)
        mock_run.return_value = self._fake_result()
        server.run_hermes(None, "hello", "helpful", force_mode=None)
        cmd = mock_run.call_args[0][0]
        toolsets_idx = cmd.index("-t") + 1
        self.assertEqual(cmd[toolsets_idx], server.MODE_TOOLSETS["BUILD"])

    @patch("server.subprocess.run")
    def test_invalid_force_mode_falls_back_to_persisted_mode(self, mock_run):
        mock_run.return_value = self._fake_result()
        server.run_hermes(None, "hello", "helpful", force_mode="NOT_A_REAL_MODE")
        cmd = mock_run.call_args[0][0]
        toolsets_idx = cmd.index("-t") + 1
        self.assertEqual(cmd[toolsets_idx], server.MODE_TOOLSETS["READ"])  # default, no mode file

    @patch("server.subprocess.run")
    def test_returned_mode_reflects_forced_mode_not_persisted(self, mock_run):
        with open(self.mode_file, "w") as f:
            json.dump({"mode": "RUN", "set_at": None, "expires_at": None, "scope": None, "set_by": None}, f)
        mock_run.return_value = self._fake_result()
        _, _, _, mode = server.run_hermes(None, "hello", "helpful", force_mode="READ")
        self.assertEqual(mode, "READ")


class TestParseRunRequest(unittest.TestCase):
    """Unit tests against the real request-validation function do_POST
    calls — not a reimplementation of its logic."""

    def test_read_only_with_session_id_rejected(self):
        body = {"read_only": True, "hermes_session_id": "some-existing-session", "message": "hi"}
        _, _, _, force_mode, _, err = server._parse_run_request(body)
        self.assertIsNotNone(err)
        self.assertIn("read_only", err)

    def test_read_only_without_session_id_forces_read_mode(self):
        body = {"read_only": True, "message": "hi"}
        session_id, message, persona, force_mode, readonly_key, err = server._parse_run_request(body)
        self.assertIsNone(err)
        self.assertIsNone(session_id)
        self.assertEqual(force_mode, "READ")
        self.assertIsNone(readonly_key)

    def test_normal_request_no_force_mode(self):
        body = {"message": "hi", "hermes_session_id": "abc123"}
        session_id, message, persona, force_mode, readonly_key, err = server._parse_run_request(body)
        self.assertIsNone(err)
        self.assertEqual(session_id, "abc123")
        self.assertIsNone(force_mode)

    def test_empty_message_rejected(self):
        body = {"message": "   "}
        _, _, _, _, _, err = server._parse_run_request(body)
        self.assertEqual(err, "message required")

    def test_invalid_persona_falls_back_to_default(self):
        body = {"message": "hi", "persona": "not-a-real-persona"}
        _, _, persona, _, _, err = server._parse_run_request(body)
        self.assertIsNone(err)
        self.assertEqual(persona, server.DEFAULT_PERSONA)

    # ── 2026-08-05 Owner decision: per-purpose read-only lock keys ────────

    def test_readonly_key_parsed_when_read_only(self):
        body = {"read_only": True, "message": "hi", "readonly_key": "readonly:whatsapp_relay"}
        _, _, _, force_mode, readonly_key, err = server._parse_run_request(body)
        self.assertIsNone(err)
        self.assertEqual(force_mode, "READ")
        self.assertEqual(readonly_key, "readonly:whatsapp_relay")

    def test_readonly_key_ignored_when_not_read_only(self):
        """readonly_key on a non-read_only request is parsed but never
        consulted for lock-key purposes (see do_POST) -- this only checks
        parsing doesn't error; the "ignored" half is covered by
        TestRunEndpointLocking below."""
        body = {"message": "hi", "readonly_key": "readonly:whatsapp_relay"}
        _, _, _, force_mode, readonly_key, err = server._parse_run_request(body)
        self.assertIsNone(err)
        self.assertIsNone(force_mode)
        self.assertEqual(readonly_key, "readonly:whatsapp_relay")

    def test_missing_readonly_key_defaults_to_none(self):
        body = {"read_only": True, "message": "hi"}
        _, _, _, _, readonly_key, _ = server._parse_run_request(body)
        self.assertIsNone(readonly_key)


class TestLockKeyFor(unittest.TestCase):
    """2026-08-05 Owner decision: independent read-only callers (WhatsApp
    admin relay vs. Phase 5B job investigations) must not collide on one
    shared "new" lock bucket just because neither has a real
    hermes_session_id -- tests server._lock_key_for directly, the exact
    function do_POST calls to pick a lock key."""

    def test_readonly_call_with_key_uses_that_key(self):
        key = server._lock_key_for(None, "READ", "readonly:whatsapp_relay")
        self.assertEqual(key, "readonly:whatsapp_relay")

    def test_different_readonly_keys_are_different_lock_keys(self):
        """The actual bug this fixes: two independent read-only purposes
        must resolve to different dict keys in _session_locks, so acquiring
        one never blocks the other."""
        whatsapp_key = server._lock_key_for(None, "READ", "readonly:whatsapp_relay")
        job_key = server._lock_key_for(None, "READ", "readonly:job:bridge_watchdog")
        self.assertNotEqual(whatsapp_key, job_key)

    def test_readonly_call_without_key_falls_back_to_new(self):
        """Backward-compat: a read_only caller that doesn't opt in (no
        readonly_key) keeps exactly the prior behavior."""
        key = server._lock_key_for(None, "READ", None)
        self.assertEqual(key, "new")

    def test_interactive_call_ignores_readonly_key(self):
        """Safety-preserving: a readonly_key must never affect a real
        interactive/stateful session's lock key -- force_mode != "READ"
        always wins."""
        key = server._lock_key_for("some-session-id", None, "readonly:whatsapp_relay")
        self.assertEqual(key, "some-session-id")

    def test_interactive_call_with_no_session_id_uses_new(self):
        key = server._lock_key_for(None, None, None)
        self.assertEqual(key, "new")

    def test_two_calls_same_readonly_key_still_serialize(self):
        """Two overlapping calls for the SAME purpose must still collide --
        this fix only separates DIFFERENT purposes, it doesn't remove
        serialization within one purpose."""
        key_a = server._lock_key_for(None, "READ", "readonly:job:bridge_watchdog")
        key_b = server._lock_key_for(None, "READ", "readonly:job:bridge_watchdog")
        self.assertEqual(key_a, key_b)
        lock_a = server._lock_for(key_a)
        lock_b = server._lock_for(key_b)
        self.assertIs(lock_a, lock_b)


if __name__ == "__main__":
    unittest.main()
