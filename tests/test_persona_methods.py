"""Tests for persona prompt builders and question loading."""

from __future__ import annotations

import unittest
from pathlib import Path

from simulation.persona_methods import (
    build_prompts_for_persona_variant,
    build_prompts_method_a,
    build_prompts_naive,
    build_prompts_production_persona,
)
from simulation.questions_io import load_questions, question_ids


class TestPersonaMethods(unittest.TestCase):
    def setUp(self) -> None:
        self.fake_row = {
            "npi": "1234567890",
            "specialty": "Internal Medicine",
            "organization_name": "123 Main St",
            "city": "Houston",
            "state": "TX",
            "gender": "M",
            "credentials": "M.D.",
            "claims_2022": 1200.0,
            "beneficiaries_2022": 200.0,
            "diabetes_share_2022": 0.25,
            "glp1_penetration_2022": 0.15,
            "branded_share_2022": 0.4,
            "drug_diversity_2022": 8,
            "has_tirzepatide_2022": 0,
            "total_payments_2022": 100.0,
            "novo_nordisk_payments": 0.0,
            "eli_lilly_payments": 50.0,
            "has_research_payments": 0,
            "adoption_archetype": "Conservative_PCP",
            "pharma_engagement_tier": "Low_Engagement",
            "geo_cluster": "TX_Houston",
            "network_group": "123 Main St",
        }
        self.questions = load_questions(Path(__file__).resolve().parents[1] / "simulation" / "questions.yaml")
        self.q1 = self.questions[0]

    def test_question_ids_stable(self) -> None:
        ids = question_ids(self.questions)
        self.assertEqual(len(ids), 6)
        self.assertEqual(ids[0], "f_q1_tirzepatide_12m")

    def test_method_a_non_empty(self) -> None:
        s, u = build_prompts_method_a(self.fake_row, self.q1)
        self.assertTrue(s.strip())
        self.assertTrue(u.strip())
        self.assertIn("f_q1_", u)

    def test_production_persona_no_2022_tirz_line(self) -> None:
        s, u = build_prompts_production_persona(self.fake_row, self.q1)
        self.assertIn("2022 baseline", u)
        self.assertNotIn("Any tirzepatide claims in 2022", u)
        self.assertIn("f_q1_", u)

    def test_naive_non_empty(self) -> None:
        s, u = build_prompts_naive(self.fake_row, self.q1)
        self.assertIn("Internal Medicine", u)
        self.assertIn("f_q1_", u)

    def test_variant_production_and_a_use_method_a(self) -> None:
        sa, ua = build_prompts_for_persona_variant("production", "A", self.fake_row, self.q1)
        sb, ub = build_prompts_for_persona_variant("a", "A", self.fake_row, self.q1)
        self.assertIn("2022 baseline", ua)
        self.assertIn("Any tirzepatide claims in 2022", ub)


if __name__ == "__main__":
    unittest.main()
