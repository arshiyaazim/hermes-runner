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


class TestReadonlySessionContinuity(unittest.TestCase):
    """2026-08-13, Bridge1 Hermes Control Channel follow-up: a successful
    READ-mode turn registers its session_id as legitimately resumable
    under read_only; non-READ turns never do, and the registration is
    what _parse_run_request's allowlist check (see TestParseRunRequest)
    actually consults."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.mode_file = os.path.join(self.tmp_dir, "current_mode.txt")
        self._mode_patch = patch.object(server, "MODE_FILE", self.mode_file)
        self._mode_patch.start()
        server._readonly_originated_sessions.clear()

    def tearDown(self):
        self._mode_patch.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        server._readonly_originated_sessions.clear()

    def _fake_result(self, stdout="reply text", stderr="session_id: read-only-abc", returncode=0):
        result = MagicMock()
        result.stdout = stdout
        result.stderr = stderr
        result.returncode = returncode
        return result

    @patch("server.subprocess.run")
    def test_successful_read_mode_call_registers_session(self, mock_run):
        mock_run.return_value = self._fake_result()
        _, session_id, error, mode = server.run_hermes(None, "hello", "helpful", force_mode="READ")
        self.assertIsNone(error)
        self.assertEqual(mode, "READ")
        self.assertTrue(server._is_readonly_originated_session(session_id))

    @patch("server.subprocess.run")
    def test_build_mode_call_does_not_register_session(self, mock_run):
        mock_run.return_value = self._fake_result(stderr="session_id: build-mode-session")
        server.run_hermes(None, "hello", "helpful", force_mode="BUILD")
        self.assertFalse(server._is_readonly_originated_session("build-mode-session"))

    @patch("server.subprocess.run")
    def test_second_read_only_call_can_resume_first_call_own_session(self, mock_run):
        """The actual continuity story end to end: call 1 (no session) ->
        registers a session; call 2 passes that session back under
        read_only and _parse_run_request accepts it (not rejected)."""
        mock_run.return_value = self._fake_result(stderr="session_id: turn-one-session")
        _, session_id, error, _ = server.run_hermes(None, "first turn", "helpful", force_mode="READ")
        self.assertIsNone(error)

        body = {"read_only": True, "hermes_session_id": session_id, "message": "second turn"}
        parsed_session, _, _, force_mode, _, _, err = server._parse_run_request(body)
        self.assertIsNone(err)
        self.assertEqual(force_mode, "READ")
        self.assertEqual(parsed_session, session_id)

    @patch("server.subprocess.run", side_effect=OSError("no such file or directory"))
    def test_subprocess_launch_failure_returns_clean_error_not_exception(self, mock_run):
        """2026-08-13: previously an OSError (or any non-timeout exception)
        from subprocess.run() propagated fully uncaught through
        run_hermes() -- indistinguishable, from the caller's side, from
        the 2 unexplained hangs found in the 2026-08-13 capability
        assessment. Now caught, logged (see subprocess_diag.log), and
        surfaced as a normal (reply=None, error=str) result."""
        reply, session_id, error, mode = server.run_hermes(None, "hello", "helpful", force_mode="READ")
        self.assertIsNone(reply)
        self.assertIsNotNone(error)
        self.assertIn("OSError", error)


class TestRunHermesWhatsAppAdminModelOverride(unittest.TestCase):
    """2026-08-12: the WhatsApp Admin relay's readonly_key must select
    MiniMax-M3/minimax instead of HERMES_RUNNER_MODEL/PROVIDER's default —
    and every other caller (no readonly_key, or a different one, e.g.
    Phase 5B's "readonly:job:<name>") must be completely unaffected."""

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
    def test_whatsapp_relay_readonly_key_selects_minimax(self, mock_run):
        mock_run.return_value = self._fake_result()
        server.run_hermes(
            None, "hello", "helpful", force_mode="READ",
            readonly_key="readonly:whatsapp_relay",
        )
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[cmd.index("-m") + 1], server.HERMES_RUNNER_WHATSAPP_ADMIN_MODEL)
        self.assertEqual(cmd[cmd.index("--provider") + 1], server.HERMES_RUNNER_WHATSAPP_ADMIN_PROVIDER)
        self.assertEqual(server.HERMES_RUNNER_WHATSAPP_ADMIN_MODEL, "MiniMax-M3")
        self.assertEqual(server.HERMES_RUNNER_WHATSAPP_ADMIN_PROVIDER, "minimax")

    @patch("server.subprocess.run")
    def test_no_readonly_key_keeps_default_model(self, mock_run):
        mock_run.return_value = self._fake_result()
        server.run_hermes(None, "hello", "helpful", force_mode="READ")
        cmd = mock_run.call_args[0][0]
        if server.HERMES_RUNNER_MODEL:
            self.assertEqual(cmd[cmd.index("-m") + 1], server.HERMES_RUNNER_MODEL)
        self.assertNotIn(server.HERMES_RUNNER_WHATSAPP_ADMIN_MODEL, cmd)

    @patch("server.subprocess.run")
    def test_different_readonly_key_keeps_default_model(self, mock_run):
        """Phase 5B alert-investigation jobs use readonly:job:<name> — must
        NOT be swept into the WhatsApp-relay-only override."""
        mock_run.return_value = self._fake_result()
        server.run_hermes(
            None, "hello", "helpful", force_mode="READ",
            readonly_key="readonly:job:bridge_watchdog",
        )
        cmd = mock_run.call_args[0][0]
        if server.HERMES_RUNNER_MODEL:
            self.assertEqual(cmd[cmd.index("-m") + 1], server.HERMES_RUNNER_MODEL)
        self.assertNotIn(server.HERMES_RUNNER_WHATSAPP_ADMIN_MODEL, cmd)

    @patch("server.subprocess.run")
    def test_customer_scope_keeps_default_model(self, mock_run):
        """caller_scope="customer" (hermes_dispatch.py) never sends this
        readonly_key — confirm it's unaffected too."""
        mock_run.return_value = self._fake_result()
        server.run_hermes(
            None, "hello", "helpful", force_mode="CUSTOMER", caller_scope="customer",
        )
        cmd = mock_run.call_args[0][0]
        if server.HERMES_RUNNER_MODEL:
            self.assertEqual(cmd[cmd.index("-m") + 1], server.HERMES_RUNNER_MODEL)
        self.assertNotIn(server.HERMES_RUNNER_WHATSAPP_ADMIN_MODEL, cmd)


class TestRunHermesWhatsAppAdminToolsetOverride(unittest.TestCase):
    """2026-08-20 (Owner-directed): the WhatsApp Admin relay's readonly_key
    must guarantee the RUN toolset (file/code_execution/terminal available)
    regardless of whatever mode is currently persisted -- WhatsApp must not
    depend on the Owner visiting the website to flip a dropdown. This does
    NOT touch current_mode.txt itself (the web UI's own dropdown is
    unaffected) and every other caller (no readonly_key, or a different one)
    must see no change at all."""

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
    def test_whatsapp_relay_gets_run_toolset_even_when_mode_file_says_read(self, mock_run):
        with open(self.mode_file, "w") as f:
            json.dump({"mode": "READ", "set_at": None, "expires_at": None, "scope": None, "set_by": None}, f)
        mock_run.return_value = self._fake_result()
        server.run_hermes(
            None, "fix the bug", "devoted", readonly_key="readonly:whatsapp_relay",
        )
        cmd = mock_run.call_args[0][0]
        toolsets_idx = cmd.index("-t") + 1
        self.assertEqual(cmd[toolsets_idx], server.MODE_TOOLSETS["RUN"])

    @patch("server.subprocess.run")
    def test_whatsapp_relay_toolset_override_does_not_touch_mode_file(self, mock_run):
        with open(self.mode_file, "w") as f:
            json.dump({"mode": "READ", "set_at": None, "expires_at": None, "scope": None, "set_by": None}, f)
        mock_run.return_value = self._fake_result()
        server.run_hermes(
            None, "fix the bug", "devoted", readonly_key="readonly:whatsapp_relay",
        )
        with open(self.mode_file) as f:
            self.assertEqual(json.load(f)["mode"], "READ")  # untouched -- web UI dropdown unaffected

    @patch("server.subprocess.run")
    def test_different_readonly_key_keeps_persisted_toolset(self, mock_run):
        """Phase 5B alert-investigation jobs (readonly:job:<name>) must NOT
        be swept into the WhatsApp-relay-only toolset override."""
        with open(self.mode_file, "w") as f:
            json.dump({"mode": "READ", "set_at": None, "expires_at": None, "scope": None, "set_by": None}, f)
        mock_run.return_value = self._fake_result()
        server.run_hermes(
            None, "hello", "helpful", readonly_key="readonly:job:bridge_watchdog",
        )
        cmd = mock_run.call_args[0][0]
        toolsets_idx = cmd.index("-t") + 1
        self.assertEqual(cmd[toolsets_idx], server.MODE_TOOLSETS["READ"])

    @patch("server.subprocess.run")
    def test_no_readonly_key_keeps_persisted_toolset(self, mock_run):
        with open(self.mode_file, "w") as f:
            json.dump({"mode": "BUILD", "set_at": None, "expires_at": None, "scope": None, "set_by": None}, f)
        mock_run.return_value = self._fake_result()
        server.run_hermes(None, "hello", "helpful")
        cmd = mock_run.call_args[0][0]
        toolsets_idx = cmd.index("-t") + 1
        self.assertEqual(cmd[toolsets_idx], server.MODE_TOOLSETS["BUILD"])


class TestRunHermesBuildModeOverride(unittest.TestCase):
    """2026-08-20: BUILD/RUN mode conversations (elevated agentic coding
    work — where authorize_build/authorize_action actually get called) must
    select HERMES_RUNNER_BUILD_MODEL/PROVIDER instead of
    HERMES_RUNNER_MODEL/PROVIDER's default, mirroring the WhatsApp Admin
    override immediately above. READ and CUSTOMER mode must stay
    completely unaffected — the whole point is a narrow, mode-scoped
    override, not a blast-radius-widening change to the general default."""

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
    def test_build_mode_selects_build_override(self, mock_run):
        mock_run.return_value = self._fake_result()
        server.run_hermes(None, "hello", "helpful", force_mode="BUILD")
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[cmd.index("-m") + 1], server.HERMES_RUNNER_BUILD_MODEL)
        self.assertEqual(cmd[cmd.index("--provider") + 1], server.HERMES_RUNNER_BUILD_PROVIDER)
        self.assertEqual(server.HERMES_RUNNER_BUILD_MODEL, "MiniMax-M3")
        self.assertEqual(server.HERMES_RUNNER_BUILD_PROVIDER, "minimax")

    @patch("server.subprocess.run")
    def test_run_mode_selects_build_override(self, mock_run):
        mock_run.return_value = self._fake_result()
        server.run_hermes(None, "hello", "helpful", force_mode="RUN")
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[cmd.index("-m") + 1], server.HERMES_RUNNER_BUILD_MODEL)
        self.assertEqual(cmd[cmd.index("--provider") + 1], server.HERMES_RUNNER_BUILD_PROVIDER)

    @patch("server.subprocess.run")
    def test_read_mode_keeps_default_model_not_build_override(self, mock_run):
        mock_run.return_value = self._fake_result()
        server.run_hermes(None, "hello", "helpful", force_mode="READ")
        cmd = mock_run.call_args[0][0]
        if server.HERMES_RUNNER_MODEL:
            self.assertEqual(cmd[cmd.index("-m") + 1], server.HERMES_RUNNER_MODEL)
        if server.HERMES_RUNNER_MODEL != server.HERMES_RUNNER_BUILD_MODEL:
            self.assertNotIn(server.HERMES_RUNNER_BUILD_MODEL, cmd)

    @patch("server.subprocess.run")
    def test_customer_scope_unaffected_by_build_override(self, mock_run):
        mock_run.return_value = self._fake_result()
        server.run_hermes(
            None, "hello", "helpful", force_mode="CUSTOMER", caller_scope="customer",
        )
        cmd = mock_run.call_args[0][0]
        if server.HERMES_RUNNER_MODEL:
            self.assertEqual(cmd[cmd.index("-m") + 1], server.HERMES_RUNNER_MODEL)

    @patch("server.subprocess.run")
    def test_whatsapp_admin_override_takes_precedence_over_build_override(self, mock_run):
        """readonly_key="readonly:whatsapp_relay" always pairs with
        force_mode="READ" in practice (do_POST's own rule), so this is a
        belt-and-suspenders check that the two conditions never fight even
        if force_mode were somehow BUILD/RUN alongside that readonly_key."""
        mock_run.return_value = self._fake_result()
        server.run_hermes(
            None, "hello", "helpful", force_mode="BUILD",
            readonly_key="readonly:whatsapp_relay",
        )
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[cmd.index("-m") + 1], server.HERMES_RUNNER_WHATSAPP_ADMIN_MODEL)


class TestParseRunRequest(unittest.TestCase):
    """Unit tests against the real request-validation function do_POST
    calls — not a reimplementation of its logic."""

    def test_read_only_with_unknown_session_id_rejected(self):
        """2026-08-13: read_only + hermes_session_id is now conditionally
        allowed (see _is_readonly_originated_session), but the original
        2026-08-04 protection is unchanged for any session_id this server
        did NOT itself already return from a prior read_only call --
        including an interactive/BUILD/RUN session someone might guess or
        copy from elsewhere."""
        body = {"read_only": True, "hermes_session_id": "some-existing-session", "message": "hi"}
        _, _, _, force_mode, _, _, err = server._parse_run_request(body)
        self.assertIsNotNone(err)
        self.assertIsNone(force_mode)

    def test_read_only_with_own_originated_session_id_allowed(self):
        """The one new case: a session_id this server itself registered
        via _register_readonly_session (i.e. actually returned from a
        prior read_only call) may now be resumed under read_only too."""
        server._register_readonly_session("own-readonly-session")
        try:
            body = {"read_only": True, "hermes_session_id": "own-readonly-session", "message": "hi"}
            session_id, _, _, force_mode, _, _, err = server._parse_run_request(body)
            self.assertIsNone(err)
            self.assertEqual(force_mode, "READ")
            self.assertEqual(session_id, "own-readonly-session")
        finally:
            server._readonly_originated_sessions.pop("own-readonly-session", None)

    def test_readonly_session_allowlist_expires(self):
        """TTL backstop: an entry older than _READONLY_SESSION_TTL_SECONDS
        is pruned lazily and no longer resumable under read_only."""
        server._register_readonly_session("stale-readonly-session")
        try:
            with server._readonly_session_lock:
                server._readonly_originated_sessions["stale-readonly-session"] = (
                    server.time.monotonic() - server._READONLY_SESSION_TTL_SECONDS - 1
                )
            self.assertFalse(server._is_readonly_originated_session("stale-readonly-session"))
            body = {"read_only": True, "hermes_session_id": "stale-readonly-session", "message": "hi"}
            _, _, _, force_mode, _, _, err = server._parse_run_request(body)
            self.assertIsNotNone(err)
            self.assertIsNone(force_mode)
        finally:
            server._readonly_originated_sessions.pop("stale-readonly-session", None)

    def test_read_only_without_session_id_forces_read_mode(self):
        body = {"read_only": True, "message": "hi"}
        session_id, message, persona, force_mode, readonly_key, caller_scope, err = server._parse_run_request(body)
        self.assertIsNone(err)
        self.assertIsNone(session_id)
        self.assertEqual(force_mode, "READ")
        self.assertIsNone(readonly_key)

    def test_normal_request_no_force_mode(self):
        body = {"message": "hi", "hermes_session_id": "abc123"}
        session_id, message, persona, force_mode, readonly_key, caller_scope, err = server._parse_run_request(body)
        self.assertIsNone(err)
        self.assertEqual(session_id, "abc123")
        self.assertIsNone(force_mode)

    def test_empty_message_rejected(self):
        body = {"message": "   "}
        _, _, _, _, _, _, err = server._parse_run_request(body)
        self.assertEqual(err, "message required")

    def test_invalid_persona_falls_back_to_default(self):
        body = {"message": "hi", "persona": "not-a-real-persona"}
        _, _, persona, _, _, _, err = server._parse_run_request(body)
        self.assertIsNone(err)
        self.assertEqual(persona, server.DEFAULT_PERSONA)

    # ── 2026-08-05 Owner decision: per-purpose read-only lock keys ────────

    def test_readonly_key_parsed_when_read_only(self):
        body = {"read_only": True, "message": "hi", "readonly_key": "readonly:whatsapp_relay"}
        _, _, _, force_mode, readonly_key, caller_scope, err = server._parse_run_request(body)
        self.assertIsNone(err)
        self.assertEqual(force_mode, "READ")
        self.assertEqual(readonly_key, "readonly:whatsapp_relay")

    def test_readonly_key_ignored_when_not_read_only(self):
        """readonly_key on a non-read_only request is parsed but never
        consulted for lock-key purposes (see do_POST) -- this only checks
        parsing doesn't error; the "ignored" half is covered by
        TestRunEndpointLocking below."""
        body = {"message": "hi", "readonly_key": "readonly:whatsapp_relay"}
        _, _, _, force_mode, readonly_key, caller_scope, err = server._parse_run_request(body)
        self.assertIsNone(err)
        self.assertIsNone(force_mode)
        self.assertEqual(readonly_key, "readonly:whatsapp_relay")

    def test_missing_readonly_key_defaults_to_none(self):
        body = {"read_only": True, "message": "hi"}
        _, _, _, _, readonly_key, _, _ = server._parse_run_request(body)
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


class TestSchemaProbeFailureHaltObservability(unittest.TestCase):
    """2026-08-16: the deferred-call validation gap this hard-stop guards
    against is closed and tested in hermes-agent (ae4551f2c); the
    underlying model habit that trips it (repeatedly observed against
    opencode_dispatch specifically) is not fixed. hermes-agent's own
    run_agent.py logs a structured "[schema_probe_failure_halt]
    tool=... count=..." warning via stdlib logging, which reaches this
    subprocess's stderr -- this reuses the existing subprocess_diag.log
    (_append_diag) sink to make that durably queryable, no new log file."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.diag_file = os.path.join(self.tmp_dir, "subprocess_diag.log")
        self._diag_patch = patch.object(server, "DIAG_LOG_FILE", self.diag_file)
        self._diag_patch.start()

    def tearDown(self):
        self._diag_patch.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _read_diag_events(self):
        if not os.path.exists(self.diag_file):
            return []
        with open(self.diag_file) as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_marker_in_stderr_recorded_as_diag_event(self):
        server._record_schema_probe_halts(
            "some other log line\n"
            "[schema_probe_failure_halt] tool=opencode_dispatch count=3\n",
            "sess-1",
        )
        events = [e for e in self._read_diag_events() if e["event"] == "schema_probe_failure_halt"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["tool"], "opencode_dispatch")
        self.assertEqual(events[0]["count"], 3)
        self.assertEqual(events[0]["session"], "sess-1")

    def test_no_marker_records_nothing(self):
        server._record_schema_probe_halts("a normal successful run, no halts here\n", "sess-2")
        events = [e for e in self._read_diag_events() if e["event"] == "schema_probe_failure_halt"]
        self.assertEqual(events, [])

    def test_no_session_id_uses_new(self):
        server._record_schema_probe_halts(
            "[schema_probe_failure_halt] tool=send_whatsapp_message count=3\n", None,
        )
        events = [e for e in self._read_diag_events() if e["event"] == "schema_probe_failure_halt"]
        self.assertEqual(events[0]["session"], "new")

    def test_multiple_markers_in_one_run_all_recorded(self):
        """A single turn can trip more than one tool's halt (e.g. the model
        retries a different tool after the first hard-stop) -- every
        occurrence must be captured, not just the first."""
        server._record_schema_probe_halts(
            "[schema_probe_failure_halt] tool=opencode_dispatch count=3\n"
            "[schema_probe_failure_halt] tool=send_whatsapp_message count=3\n",
            "sess-3",
        )
        events = [e for e in self._read_diag_events() if e["event"] == "schema_probe_failure_halt"]
        self.assertEqual(len(events), 2)
        self.assertEqual({e["tool"] for e in events}, {"opencode_dispatch", "send_whatsapp_message"})

    @patch("server.subprocess.run")
    def test_run_hermes_wires_stderr_scan_automatically(self, mock_run):
        """Regression guard: confirms _run_subprocess_once (called by
        run_hermes) actually scans real subprocess stderr, not just that
        _record_schema_probe_halts works in isolation above."""
        result = MagicMock()
        result.stdout = "some reply"
        result.stderr = "session_id: abc123\n[schema_probe_failure_halt] tool=opencode_dispatch count=3\n"
        result.returncode = 0
        mock_run.return_value = result

        server.run_hermes(None, "hello", "helpful")

        events = [e for e in self._read_diag_events() if e["event"] == "schema_probe_failure_halt"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["tool"], "opencode_dispatch")


if __name__ == "__main__":
    unittest.main()
