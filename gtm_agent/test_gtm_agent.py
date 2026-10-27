import json
import unittest
from unittest.mock import patch

with patch.dict("os.environ", {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent as agent


class ProspectScoringTests(unittest.TestCase):
    def setUp(self):
        self.prospect_id = next(iter(agent.data_service.PROSPECTS))
        self.offering_id = next(iter(agent.data_service.OFFERINGS))
        self.cache = patch.object(agent.data_service, "_PROFILES", {})
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.model = patch.object(agent, "_scoring_llm")
        self.scoring = self.model.start()
        self.addCleanup(self.model.stop)
        self.result = agent.ProspectScore(
            score=75,
            justification="Good revenue fit with some required technologies missing.",
            rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=90, tech_stack_match=60, segment_fit=75,
            ),
        )
        self.scoring.invoke.return_value = self.result

    def score(self, **overrides):
        return agent.score_prospect.invoke({
            "prospect_id": self.prospect_id,
            "offering_id": self.offering_id,
            **overrides,
        })

    def test_score_with_only_ids_loads_records_and_returns_structured_score(self):
        self.assertEqual(self.score(), self.result.model_dump())
        messages = self.scoring.invoke.call_args.args[0]
        offering_text, profile_text = messages[1]["content"].split(
            "\n\nProspect profile:\n"
        )
        self.assertEqual(
            json.loads(offering_text.removeprefix("Offering:\n")),
            agent.data_service.get_offering(self.offering_id),
        )
        profile = json.loads(profile_text)
        self.assertEqual(profile["prospect_id"], self.prospect_id)
        self.assertEqual(profile, agent.data_service._PROFILES[self.prospect_id])

    def test_score_refreshes_cached_tech_stack(self):
        cached = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        stale_stack = cached["prospect_profile"]["tech_stack"]
        refreshed_stack = ["new technology"]
        with patch.object(agent.data_service, "fetch_tech_stack", return_value=refreshed_stack):
            self.assertEqual(self.score(), self.result.model_dump())
        profile_text = self.scoring.invoke.call_args.args[0][1]["content"].split(
            "\n\nProspect profile:\n"
        )[1]
        self.assertEqual(json.loads(profile_text)["tech_stack"], refreshed_stack)
        self.assertEqual(cached["prospect_profile"]["tech_stack"], stale_stack)

    def test_unknown_offering_returns_existing_error(self):
        self.assertEqual(self.score(offering_id="unknown"), {
            "score": None, "error": "Cannot score without a valid offering."
        })
        self.scoring.invoke.assert_not_called()

    def test_invalid_offerings_return_existing_error(self):
        offering = agent.data_service.get_offering(self.offering_id)
        for field, invalid in (
            ("required_tech_stack", []),
            ("min_annual_revenue", None),
            ("description", ""),
        ):
            with self.subTest(field=field), patch.dict(
                agent.data_service.OFFERINGS,
                {"invalid": {**offering, field: invalid}},
            ):
                self.assertEqual(self.score(offering_id="invalid"), {
                    "score": None, "error": "Cannot score without a valid offering."
                })
        self.scoring.invoke.assert_not_called()

    def test_unknown_prospect_returns_not_found_error(self):
        self.assertEqual(self.score(prospect_id="unknown"), {
            "score": None, "error": "Prospect not found."
        })
        self.scoring.invoke.assert_not_called()

    def test_profile_build_enriches_persists_and_reuses_cache(self):
        enrichment = {
            "engagement_history": ["recent engagement"],
            "account_details": {"annual_revenue": 123456},
            "tech_stack": ["fresh technology"],
        }
        with patch.object(
            agent.data_service, "fetch_engagement_history",
            return_value=enrichment["engagement_history"],
        ), patch.object(
            agent.data_service, "fetch_account_details",
            return_value=enrichment["account_details"],
        ), patch.object(
            agent.data_service, "fetch_tech_stack", return_value=enrichment["tech_stack"],
        ):
            result = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        record = agent.data_service.get_prospect_record(self.prospect_id)
        self.assertEqual(result, {
            "prospect_profile": {"prospect_id": self.prospect_id, **record, **enrichment},
            "found": True,
        })
        self.assertEqual(agent.data_service._PROFILES[self.prospect_id], result["prospect_profile"])
        with patch.object(agent.data_service, "get_prospect_record", return_value=None):
            self.assertEqual(
                agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id}),
                result,
            )
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "unknown"}), {
            "prospect_profile": None, "found": False,
        })

    def test_tool_exposes_id_arguments(self):
        schema = agent.score_prospect.args_schema.model_json_schema()
        self.assertEqual(set(schema["properties"]), {"prospect_id", "offering_id"})
        self.assertEqual(set(schema["required"]), {"prospect_id", "offering_id"})
        for name in schema["properties"]:
            self.assertEqual(schema["properties"][name]["type"], "string")
            self.assertIn(name, agent.score_prospect.description)
