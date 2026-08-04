import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server
import audit_tools


class TestHandleAudit(unittest.TestCase):
    """/audit request dispatch (2026-08-04, Chat audit toolkit wiring --
    see assistant-platform/proposal_chat_audit_toolkit_20260804.md). Tests
    server._handle_audit directly (not the raw HTTP layer), matching this
    file's existing convention for _parse_run_request/run_hermes."""

    def test_unknown_tool_returns_400(self):
        status, payload = server._handle_audit({"tool": "audit_delete_everything", "args": {}})
        self.assertEqual(status, 400)
        self.assertIn("unknown audit tool", payload["error"])

    def test_missing_tool_key_returns_400(self):
        status, payload = server._handle_audit({"args": {}})
        self.assertEqual(status, 400)
        self.assertIn("unknown audit tool", payload["error"])

    def test_non_dict_args_returns_400(self):
        status, payload = server._handle_audit({"tool": "audit_search_code", "args": "not-a-dict"})
        self.assertEqual(status, 400)
        self.assertIn("args must be an object", payload["error"])

    def test_known_tool_is_invoked_with_args_and_result_passed_through(self):
        fake = unittest.mock.MagicMock(return_value={"matches": ["hit1", "hit2"]})
        with patch.dict(audit_tools.AUDIT_TOOLS, {"audit_search_code": fake}):
            status, payload = server._handle_audit(
                {"tool": "audit_search_code", "args": {"query": "foo", "root": "fazle-core", "max_results": 5}}
            )
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"matches": ["hit1", "hit2"]})
        fake.assert_called_once_with(query="foo", root="fazle-core", max_results=5)

    def test_missing_args_defaults_to_empty_dict(self):
        fake = unittest.mock.MagicMock(return_value={"changes": []})
        with patch.dict(audit_tools.AUDIT_TOOLS, {"audit_git_status": fake}):
            status, payload = server._handle_audit({"tool": "audit_git_status"})
        self.assertEqual(status, 200)
        fake.assert_called_once_with()

    def test_bad_argument_name_returns_400_not_500(self):
        status, payload = server._handle_audit(
            {"tool": "audit_search_code", "args": {"not_a_real_kwarg": "x"}}
        )
        self.assertEqual(status, 400)
        self.assertIn("invalid arguments", payload["error"])

    def test_tool_exception_returns_500_not_a_crash(self):
        fake = unittest.mock.MagicMock(side_effect=RuntimeError("disk on fire"))
        with patch.dict(audit_tools.AUDIT_TOOLS, {"audit_search_code": fake}):
            status, payload = server._handle_audit({"tool": "audit_search_code", "args": {"query": "x"}})
        self.assertEqual(status, 500)
        self.assertIn("disk on fire", payload["error"])

    def test_all_seven_approved_tools_are_registered(self):
        expected = {
            "audit_search_code", "audit_search_docs", "audit_search_kb",
            "audit_search_logs", "audit_read_file", "audit_git_status",
            "audit_recent_commits",
        }
        self.assertEqual(set(audit_tools.AUDIT_TOOLS.keys()), expected)

    def test_whatsapp_lookup_deliberately_not_exposed(self):
        """Scope guard: audit_lookup_whatsapp_messages is explicitly out of
        the Owner-approved 7-tool scope (DB-backed, not filesystem-backed --
        see audit_tools.py's module docstring)."""
        self.assertNotIn("audit_lookup_whatsapp_messages", audit_tools.AUDIT_TOOLS)


class TestAuditToolsPathSafety(unittest.TestCase):
    """Spot-check the ported safety properties actually made it across --
    full coverage of this logic already exists in fazle-mcp's own test
    suite; these are a guard against a silent divergence during the port."""

    def test_resolve_in_root_rejects_env_files(self):
        _, err = audit_tools._resolve_in_root("fazle-core", ".env")
        self.assertIsNotNone(err)

    def test_resolve_in_root_rejects_path_traversal(self):
        _, err = audit_tools._resolve_in_root("assistant-platform", "../core/knowledge_base")
        self.assertIsNotNone(err)

    def test_resolve_in_root_rejects_unknown_root(self):
        _, err = audit_tools._resolve_in_root("not-a-real-root")
        self.assertIsNotNone(err)

    def test_redact_line_masks_secret_and_phone(self):
        redacted = audit_tools._redact_line("token=abc123 phone=01712345678")
        self.assertNotIn("abc123", redacted)
        self.assertNotIn("01712345678", redacted)


if __name__ == "__main__":
    unittest.main()
