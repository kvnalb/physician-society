"""Tests for claims-derived pseudo-labels (forward hold-out battery)."""

from __future__ import annotations

import unittest

from eval.behavioral_labels import pseudo_label_for_question


class TestBehavioralLabels(unittest.TestCase):
    def test_f_q1_tirzepatide(self) -> None:
        self.assertEqual(
            pseudo_label_for_question(
                {"has_tirzepatide_2023": 0, "tirzepatide_claims_2023": 0.0},
                "f_q1_tirzepatide_12m",
            ),
            "f_q1_expect_unlikely",
        )
        self.assertEqual(
            pseudo_label_for_question(
                {"has_tirzepatide_2023": 1, "tirzepatide_claims_2023": 5.0},
                "f_q1_tirzepatide_12m",
            ),
            "f_q1_expect_active",
        )
        self.assertEqual(
            pseudo_label_for_question(
                {"has_tirzepatide_2023": 1, "tirzepatide_claims_2023": 1.0},
                "f_q1_tirzepatide_12m",
            ),
            "f_q1_expect_selective",
        )

    def test_f_q2_glp1_traj(self) -> None:
        self.assertEqual(
            pseudo_label_for_question(
                {"glp1_penetration_2022": 0.10, "glp1_penetration_2023": 0.20},
                "f_q2_glp1_trajectory",
            ),
            "f_q2_up",
        )
        self.assertEqual(
            pseudo_label_for_question(
                {"glp1_penetration_2022": 0.10, "glp1_penetration_2023": 0.11},
                "f_q2_glp1_trajectory",
            ),
            "f_q2_flat",
        )

    def test_f_q5_missing_claims(self) -> None:
        self.assertIsNone(
            pseudo_label_for_question({"claims_2022": 0, "claims_2023": 100}, "f_q5_panel_scale_trajectory"),
        )


if __name__ == "__main__":
    unittest.main()
