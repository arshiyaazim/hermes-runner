import datetime
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server


class ModeTTLTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.mode_file = os.path.join(self.tmp_dir, "current_mode.txt")
        self.audit_log = os.path.join(self.tmp_dir, "mode_audit.log")
        self._mode_patch = patch.object(server, "MODE_FILE", self.mode_file)
        self._audit_patch = patch.object(server, "AUDIT_LOG_FILE", self.audit_log)
        self._mode_patch.start()
        self._audit_patch.start()

    def tearDown(self):
        self._mode_patch.stop()
        self._audit_patch.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _audit_entries(self):
        if not os.path.isfile(self.audit_log):
            return []
        with open(self.audit_log) as f:
            return [json.loads(line) for line in f if line.strip()]


class TestPermanentDefault(ModeTTLTestBase):
    def test_no_file_defaults_to_read(self):
        state = server.read_mode_state()
        self.assertEqual(state["mode"], "READ")
        self.assertIsNone(state["expires_at"])

    def test_legacy_bare_word_file_treated_as_permanent(self):
        with open(self.mode_file, "w") as f:
            f.write("BUILD\n")
        state = server.read_mode_state()
        self.assertEqual(state["mode"], "BUILD")
        self.assertIsNone(state["expires_at"])
        self.assertFalse(state["expired"])


class TestTemporaryModes(ModeTTLTestBase):
    def test_temporary_build_mode_set(self):
        state = server.write_mode_state("BUILD", ttl_seconds=300, scope="TIME")
        self.assertEqual(state["mode"], "BUILD")
        self.assertIsNotNone(state["expires_at"])

    def test_temporary_run_mode_set(self):
        state = server.write_mode_state("RUN", ttl_seconds=120, scope="TIME")
        self.assertEqual(state["mode"], "RUN")
        read_back = server.read_mode_state()
        self.assertEqual(read_back["mode"], "RUN")
        self.assertGreater(read_back["seconds_remaining"], 0)

    def test_read_mode_never_gets_a_ttl(self):
        state = server.write_mode_state("READ", ttl_seconds=300)
        self.assertIsNone(state["expires_at"])


class TestExpiryRevert(ModeTTLTestBase):
    def test_expiry_reverts_to_read_and_persists(self):
        server.write_mode_state("BUILD", ttl_seconds=60)
        # Force expiry by rewriting the file with a past expires_at.
        past = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=5)).isoformat()
        with open(self.mode_file, "w") as f:
            json.dump({"mode": "BUILD", "set_at": past, "expires_at": past, "scope": "TIME", "set_by": "admin"}, f)

        state = server.read_mode_state()
        self.assertEqual(state["mode"], "READ")
        self.assertTrue(state["expired"])

        # Persisted, not just in-memory — a fresh read (simulating a
        # restart, since this is file-based with no cached state) must
        # also show READ, permanently (no lingering expires_at).
        second_read = server.read_mode_state()
        self.assertEqual(second_read["mode"], "READ")
        self.assertIsNone(second_read["expires_at"])

    def test_expired_mode_cannot_select_privileged_toolset(self):
        past = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=5)).isoformat()
        with open(self.mode_file, "w") as f:
            json.dump({"mode": "RUN", "set_at": past, "expires_at": past, "scope": "TIME", "set_by": "admin"}, f)
        effective_mode = server.read_current_mode()
        self.assertEqual(effective_mode, "READ")
        toolsets = server.MODE_TOOLSETS[effective_mode]
        self.assertNotIn("terminal", toolsets)

    def test_restart_simulation_state_survives(self):
        # "Restart" == nothing but the file persists; there is no other
        # process state to lose in this stdlib-only shim.
        server.write_mode_state("BUILD", ttl_seconds=3600)
        # Simulate a restart: nothing cached, just re-read the same file path.
        state_after_restart = server.read_mode_state()
        self.assertEqual(state_after_restart["mode"], "BUILD")


class TestTTLValidation(ModeTTLTestBase):
    def test_invalid_ttl_non_numeric_rejected(self):
        with self.assertRaises(ValueError):
            server.write_mode_state("BUILD", ttl_seconds="soon")

    def test_negative_ttl_rejected(self):
        with self.assertRaises(ValueError):
            server.write_mode_state("BUILD", ttl_seconds=-10)

    def test_zero_ttl_rejected(self):
        with self.assertRaises(ValueError):
            server.write_mode_state("BUILD", ttl_seconds=0)

    def test_excessively_long_ttl_rejected(self):
        with self.assertRaises(ValueError):
            server.write_mode_state("BUILD", ttl_seconds=999999)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            server.write_mode_state("SUPERADMIN")

    def test_invalid_scope_rejected(self):
        with self.assertRaises(ValueError):
            server.write_mode_state("BUILD", scope="FOREVER")

    def test_min_boundary_ttl_accepted(self):
        state = server.write_mode_state("BUILD", ttl_seconds=server.TTL_MIN_SECONDS)
        self.assertIsNotNone(state["expires_at"])

    def test_max_boundary_ttl_accepted(self):
        state = server.write_mode_state("BUILD", ttl_seconds=server.TTL_MAX_SECONDS)
        self.assertIsNotNone(state["expires_at"])


class TestScopes(ModeTTLTestBase):
    def test_task_scoped_gets_default_ttl_when_unspecified(self):
        state = server.write_mode_state("BUILD", scope="TASK")
        self.assertEqual(state["scope"], "TASK")
        self.assertIsNotNone(state["expires_at"])

    def test_session_scoped_gets_default_ttl_when_unspecified(self):
        state = server.write_mode_state("BUILD", scope="SESSION")
        self.assertEqual(state["scope"], "SESSION")
        self.assertIsNotNone(state["expires_at"])

    def test_time_scoped_with_explicit_ttl(self):
        state = server.write_mode_state("BUILD", ttl_seconds=600, scope="TIME")
        self.assertEqual(state["scope"], "TIME")


class TestConcurrency(ModeTTLTestBase):
    def test_concurrent_reads_near_expiry_do_not_raise_or_corrupt(self):
        past = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=1)).isoformat()
        with open(self.mode_file, "w") as f:
            json.dump({"mode": "RUN", "set_at": past, "expires_at": past, "scope": "TIME", "set_by": "admin"}, f)

        results = []
        errors = []

        def worker():
            try:
                results.append(server.read_mode_state()["mode"])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertTrue(all(m == "READ" for m in results))
        # File must end up in a single, valid, parseable state — not
        # corrupted by concurrent writers racing the revert.
        with open(self.mode_file) as f:
            final = json.load(f)
        self.assertEqual(final["mode"], "READ")


class TestFailSafe(ModeTTLTestBase):
    def test_corrupted_json_fails_closed_to_read(self):
        with open(self.mode_file, "w") as f:
            f.write("{not valid json at all")
        state = server.read_mode_state()
        self.assertEqual(state["mode"], "READ")

    def test_invalid_mode_value_in_file_fails_closed(self):
        with open(self.mode_file, "w") as f:
            json.dump({"mode": "SUPERADMIN", "expires_at": None}, f)
        state = server.read_mode_state()
        self.assertEqual(state["mode"], "READ")

    def test_corrupted_expiry_timestamp_fails_closed(self):
        with open(self.mode_file, "w") as f:
            json.dump({"mode": "RUN", "expires_at": "not-a-timestamp"}, f)
        state = server.read_mode_state()
        self.assertEqual(state["mode"], "READ")

    def test_unreadable_file_path_fails_closed(self):
        with patch.object(server, "MODE_FILE", "/nonexistent/dir/current_mode.txt"):
            state = server.read_mode_state()
            self.assertEqual(state["mode"], "READ")


class TestAuditLog(ModeTTLTestBase):
    def test_mode_change_is_logged(self):
        server.write_mode_state("BUILD", ttl_seconds=300, set_by="admin")
        entries = self._audit_entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["event"], "mode_change")
        self.assertEqual(entries[0]["to_mode"], "BUILD")

    def test_auto_revert_is_logged(self):
        past = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=5)).isoformat()
        with open(self.mode_file, "w") as f:
            json.dump({"mode": "RUN", "set_at": past, "expires_at": past, "scope": "TIME", "set_by": "admin"}, f)
        server.read_mode_state()
        entries = self._audit_entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["event"], "auto_revert_expired")
        self.assertEqual(entries[0]["from_mode"], "RUN")

    def test_multiple_changes_all_logged(self):
        server.write_mode_state("BUILD", ttl_seconds=300)
        server.write_mode_state("RUN", ttl_seconds=300)
        server.write_mode_state("READ")
        entries = self._audit_entries()
        self.assertEqual(len(entries), 3)


if __name__ == "__main__":
    unittest.main()
