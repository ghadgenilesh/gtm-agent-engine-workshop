import json
import os
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

from gtm_agent import data_service

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent


class UpdateProspectInfoTests(unittest.TestCase):
    def setUp(self):
        self.prospect_id = "LEAD-39002"
        records = patch.object(data_service, "PROSPECTS", deepcopy(data_service.PROSPECTS))
        records.start()
        self.addCleanup(records.stop)
        profiles = patch.object(data_service, "_PROFILES", {})
        profiles.start()
        self.addCleanup(profiles.stop)
        tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        tracing.start()
        self.addCleanup(tracing.stop)

    def test_update_persists_technology_and_preserves_return_shape(self):
        original = list(data_service.fetch_tech_stack(self.prospect_id))

        result = data_service.update_prospect_info(self.prospect_id, "Terraform")

        expected = original + ["Terraform"]
        self.assertEqual(result, {"updated": True, "found": True, "tech_stack": expected})
        self.assertEqual(data_service.fetch_tech_stack(self.prospect_id), expected)
        data_service.update_prospect_info(self.prospect_id, "Terraform")
        self.assertEqual(data_service.fetch_tech_stack(self.prospect_id), expected)

    def test_cached_profile_reflects_updated_technology(self):
        profile = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertNotIn("Terraform", profile["prospect_profile"]["tech_stack"])

        data_service.update_prospect_info(self.prospect_id, "Terraform")
        rebuilt = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})

        self.assertTrue(rebuilt["found"])
        self.assertIn("Terraform", rebuilt["prospect_profile"]["tech_stack"])

    def test_unknown_prospect_is_not_updated(self):
        result = data_service.update_prospect_info("unknown", "Terraform")

        self.assertEqual(result, {"updated": False, "found": False})
        self.assertIsNone(data_service.get_prospect_record("unknown"))

    def test_scoring_receives_saved_stack_after_update(self):
        profile = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        offering = data_service.get_offering("OFFER-10004")
        gtm_agent.update_prospect_info.invoke({
            "prospect_id": self.prospect_id, "technology": "Terraform",
        })
        scorer = Mock()
        with patch.object(gtm_agent, "_scoring_llm", scorer):
            gtm_agent.score_prospect.invoke({
                "prospect_profile": profile["prospect_profile"], "offering": offering,
            })

        messages = scorer.invoke.call_args.args[0]
        saved_profile = json.loads(messages[1]["content"].split("\n\nProspect profile:\n")[1])
        self.assertEqual(saved_profile["tech_stack"], data_service.fetch_tech_stack(self.prospect_id))
        self.assertFalse(set(offering["required_tech_stack"]) - set(saved_profile["tech_stack"]))


if __name__ == "__main__":
    unittest.main()
