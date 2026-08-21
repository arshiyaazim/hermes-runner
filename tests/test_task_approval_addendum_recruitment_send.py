"""Regression test — recruitment-contact NL authorization (2026-08-21
closure pass, Task 3).

Live-reproduced finding: given an explicit, unambiguous Owner instruction
naming a specific, already-resolved candidate ("এই ৮৮০১৯৬৬০১১৬০৯
candidate-এর সাথে যোগাযোগ কর..."), Hermes composed a correct message but
still asked "আপনি কি এই message টা approve করেন?" instead of sending
directly via send_whatsapp_message(confirm=true) — the exact tool that was
already designed for this ("confirm=true) after the admin has directly
instructed this send", see fazle-mcp/send_whatsapp_tools.py). Root cause:
TASK_APPROVAL_ADDENDUM's NL->tool-call mapping table had entries for
'fix it'/'commit it'/'deploy it' etc. but none for a recruitment/contact
instruction, so it fell through to SYSTEM_PREAMBLE's generic "every
state-changing action needs its own yes/no" rule.

This only asserts the prompt text itself is present and well-formed — a
live model-behavior assertion isn't feasible in a unit test; the actual
end-to-end verification is documented in the closure-pass report (real
hermes-runner /run calls, same session, before/after this fix).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server


class TestRecruitmentContactAddendum(unittest.TestCase):
    def test_addendum_maps_contact_instruction_to_direct_send(self):
        text = server.TASK_APPROVAL_ADDENDUM
        self.assertIn("send_whatsapp_message", text)
        self.assertIn("confirm=true", text)
        self.assertIn("যোগাযোগ কর", text)
        # No second round-trip for this specific, scoped case.
        self.assertIn("no draft_whatsapp_reply, no second", text)

    def test_addendum_still_requires_a_specific_named_recipient(self):
        text = server.TASK_APPROVAL_ADDENDUM
        self.assertIn("does NOT cover a vague/broad instruction", text)
        self.assertIn("contact interested candidates", text)

    def test_addendum_still_requires_verified_facts_not_fabrication(self):
        text = server.TASK_APPROVAL_ADDENDUM
        self.assertIn("never invented", text)
        self.assertIn("never a fabricated", text)

    def test_full_preamble_still_forbids_destructive_shortcuts(self):
        # Sanity check this addition sits alongside, not instead of, the
        # existing hard boundary — never weakened by this change.
        text = server.TASK_APPROVAL_ADDENDUM
        self.assertIn("NEVER authorizes rm -rf", text)

    def test_autonomy_addendum_carves_out_the_same_named_case(self):
        # AUTONOMY_ADDENDUM independently asserts "Tier C/D writes ...
        # need explicit confirmation" -- a THIRD competing source found
        # live (round 2 of the fix: SYSTEM_PREAMBLE + TASK_APPROVAL_
        # ADDENDUM alone still weren't enough, Hermes explicitly reasoned
        # the authorization existed and still asked "yes" anyway).
        text = server.AUTONOMY_ADDENDUM
        self.assertIn("except the specific, named recruitment-contact case", text)

    def test_system_preamble_carves_out_the_named_case_from_the_generic_yesno_rule(self):
        # Live-reproduced: the addendum entry alone wasn't enough — Hermes
        # still asked "আপনি কি এই message টা approve করেন?" because
        # SYSTEM_PREAMBLE's own "every state-changing action needs its own
        # yes/no" rule reads as universal. This is the explicit carve-out.
        text = server.SYSTEM_PREAMBLE
        self.assertIn("UNLESS", text)
        self.assertIn("Boss's instruction IS the yes", text)


if __name__ == "__main__":
    unittest.main()
