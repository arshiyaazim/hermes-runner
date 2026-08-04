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
        _, _, _, force_mode, err = server._parse_run_request(body)
        self.assertIsNotNone(err)
        self.assertIn("read_only", err)

    def test_read_only_without_session_id_forces_read_mode(self):
        body = {"read_only": True, "message": "hi"}
        session_id, message, persona, force_mode, err = server._parse_run_request(body)
        self.assertIsNone(err)
        self.assertIsNone(session_id)
        self.assertEqual(force_mode, "READ")

    def test_normal_request_no_force_mode(self):
        body = {"message": "hi", "hermes_session_id": "abc123"}
        session_id, message, persona, force_mode, err = server._parse_run_request(body)
        self.assertIsNone(err)
        self.assertEqual(session_id, "abc123")
        self.assertIsNone(force_mode)

    def test_empty_message_rejected(self):
        body = {"message": "   "}
        _, _, _, _, err = server._parse_run_request(body)
        self.assertEqual(err, "message required")

    def test_invalid_persona_falls_back_to_default(self):
        body = {"message": "hi", "persona": "not-a-real-persona"}
        _, _, persona, _, err = server._parse_run_request(body)
        self.assertIsNone(err)
        self.assertEqual(persona, server.DEFAULT_PERSONA)


if __name__ == "__main__":
    unittest.main()
