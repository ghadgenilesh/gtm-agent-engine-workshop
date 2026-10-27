import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

with patch.dict(os.environ, {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") or "test-key",
    "LANGSMITH_TRACING": "false",
}):
    from gtm_agent import gtm_agent as agent


class SendProspectEmailTests(unittest.TestCase):
    def send(self, prospect):
        return agent.send_prospect_email.func(
            prospect=prospect,
            subject="Introduction",
            body="Hello",
            runtime=SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}}),
        )

    def test_authoritative_disqualification_blocks_send(self):
        record = {"disqualified": True, "email": "authoritative@example.com"}
        for fields in ({}, {"disqualified": False}):
            with self.subTest(fields=fields), patch.object(
                agent.data_service, "get_prospect_record", return_value=record
            ), patch.object(agent.uuid, "uuid4") as message_id:
                result = self.send({
                    "prospect_id": "LEAD-50001",
                    "email": "model@example.com",
                    **fields,
                })
                self.assertEqual(result, {
                    "status": "blocked",
                    "reason": "disqualified",
                    "to": "authoritative@example.com",
                })
                message_id.assert_not_called()

    def test_eligible_and_unresolved_prospects_keep_send_behavior(self):
        cases = (
            ({"prospect_id": "LEAD-70001"}, {"disqualified": False}),
            ({"prospect_id": "missing"}, None),
            ({}, None),
        )
        for fields, record in cases:
            with self.subTest(fields=fields), patch.object(
                agent.data_service, "get_prospect_record", return_value=record
            ):
                result = self.send({"name": "Prospect", "email": "prospect@example.com", **fields})
                self.assertEqual(result, {
                    "status": "sent",
                    "message_id": result["message_id"],
                    "to": "prospect@example.com",
                    "to_name": "Prospect",
                    "from": "ava.mills@northpoint.com",
                    "from_name": "Ava Mills",
                    "subject": "Introduction",
                    "body": "Hello",
                })
                self.assertRegex(result["message_id"], r"^msg-[0-9a-f]{12}$")

    def test_missing_email_still_fails(self):
        self.assertEqual(self.send({}), {
            "status": "failed", "error": "Prospect record has no email address."
        })

    @unittest.skipUnless(os.environ.get("OPENAI_API_KEY"), "Requires model credentials")
    def test_agent_reports_blocked_send_without_retry(self):
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
            result = agent.run_agent(
                "Send an introductory email to prospect LEAD-50001.",
                user_id="rep_amills",
            )
        sends = [
            json.loads(message.content) for message in result["messages"]
            if message.type == "tool" and message.name == "send_prospect_email"
        ]
        self.assertEqual(len(sends), 1)
        self.assertEqual(sends[0]["status"], "blocked")
        self.assertEqual(sends[0]["reason"], "disqualified")
        self.assertNotIn("message_id", sends[0])
        reply = result["reply"].lower()
        self.assertIn("disqualified", reply)
        self.assertRegex(reply, r"not sent|wasn't sent|did not send|didn't send|haven't sent|has not been sent")


if __name__ == "__main__":
    unittest.main()
