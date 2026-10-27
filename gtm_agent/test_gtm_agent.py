import json
import os
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import data_service
    from gtm_agent import gtm_agent as agent


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "name": "Test Prospect",
            "email": "prospect@example.com",
            "company": "Example",
            "disqualified": False,
            "engagement_history": [{"channel": "Webinar"}],
            "account_details": [{"industry": "Software"}],
            "tech_stack": ["Python"],
            "annual_revenue": 1000000,
            "billing_qualification": {
                "tax_id": "test-tax-id",
                "date_of_birth": "test-dob",
                "card_on_file": "test-card",
                "credit_check_ref": "test-credit-ref",
            },
            "future_private_field": "private",
        }
        self.offering = {
            "required_tech_stack": ["Python"],
            "min_annual_revenue": 100000,
            "description": "Example offering",
        }
        self.enterContext(patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}))
        self.enterContext(patch.dict(data_service.PROSPECTS, {"TEST": self.record}))
        self.enterContext(patch.dict(data_service._PROFILES, {}, clear=True))
        self.scoring = self.enterContext(patch.object(agent, "_scoring_llm"))
        self.scoring.invoke.return_value = Mock(model_dump=lambda: {"score": 75})

    def assert_no_billing(self, value):
        serialized = json.dumps(value)
        for key, secret in self.record["billing_qualification"].items():
            self.assertNotIn(key, serialized)
            self.assertNotIn(secret, serialized)
        self.assertNotIn("billing_qualification", serialized)

    def test_contact_contains_only_approved_fields(self):
        result = agent.get_prospect.invoke({"prospect_id": "TEST"})
        self.assertEqual(result, {"found": True, "prospect": {
            "prospect_id": "TEST",
            "name": "Test Prospect",
            "email": "prospect@example.com",
            "company": "Example",
            "disqualified": False,
        }})
        self.assert_no_billing(result)

    def test_built_and_cached_profiles_remain_safe_through_scoring(self):
        for cached in (False, True):
            with self.subTest(cached=cached):
                data_service._PROFILES.clear()
                if cached:
                    data_service.save_profile_to_db("TEST", {"prospect_id": "TEST", **deepcopy(self.record)})
                result = agent.build_prospect_profile.invoke({"prospect_id": "TEST"})
                self.assertTrue(result["found"])
                self.assert_no_billing(result)
                self.assert_no_billing(data_service.get_profile_from_db("TEST"))
                profile = result["prospect_profile"]
                self.assertEqual(profile["annual_revenue"], 1000000)
                self.assertEqual(profile["engagement_history"], self.record["engagement_history"])
                arguments = {"prospect_profile": profile, "offering": self.offering}
                self.assert_no_billing(arguments)
                self.assertEqual(agent.score_prospect.invoke(arguments), {"score": 75})
                self.assert_no_billing(self.scoring.invoke.call_args.args[0])
                self.assertIn("billing_qualification", self.record)

    def test_scoring_filters_billing_from_supplied_profiles(self):
        for prospect_id in (None, "TEST"):
            with self.subTest(prospect_id=prospect_id):
                profile = deepcopy(self.record)
                if prospect_id:
                    profile["prospect_id"] = prospect_id
                agent.score_prospect.invoke({"prospect_profile": profile, "offering": self.offering})
                messages = self.scoring.invoke.call_args.args[0]
                self.assert_no_billing(messages)
                supplied = json.loads(messages[1]["content"].split("Prospect profile:\n", 1)[1])
                self.assertEqual(supplied, {k: v for k, v in profile.items() if k != "billing_qualification"})


if __name__ == "__main__":
    unittest.main()
