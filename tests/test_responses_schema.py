"""Survey response schema helpers."""

from __future__ import annotations

import unittest

from simulation.responses_schema import (
    RESPONSE_ROW_SCHEMA_VERSION,
    flatten_survey_rows,
    is_v2_survey_row,
    responses_filename_for_model,
)


class TestResponsesSchema(unittest.TestCase):
    def test_filename_sanitizes_model(self) -> None:
        self.assertEqual(
            responses_filename_for_model("zai-org/GLM-5.1"),
            "responses__zai-org_GLM-5.1.jsonl",
        )

    def test_is_v2(self) -> None:
        row = {
            "schema_version": RESPONSE_ROW_SCHEMA_VERSION,
            "npi": "1",
            "method_a": {"q1": {"option_id": "a", "reasoning": ""}},
        }
        self.assertTrue(is_v2_survey_row(row))
        self.assertFalse(is_v2_survey_row({"npi": "1", "question_id": "q1", "method": "method_a"}))

    def test_flatten_ignores_legacy_method_b_block(self) -> None:
        rows = [
            {
                "schema_version": RESPONSE_ROW_SCHEMA_VERSION,
                "npi": "100",
                "method_a": {"q1": {"option_id": "x", "reasoning": "r1"}},
                "method_b": {"q1": {"option_id": "y", "reasoning": "r2"}},
            }
        ]
        flat = flatten_survey_rows(rows)
        self.assertEqual(len(flat), 1)
        self.assertEqual(
            {(r["method"], r["question_id"], r["parsed_option"]) for r in flat},
            {("method_a", "q1", "x")},
        )

    def test_flatten_single_method(self) -> None:
        rows = [
            {
                "schema_version": RESPONSE_ROW_SCHEMA_VERSION,
                "npi": "100",
                "method_a": {"q1": {"option_id": "x", "reasoning": "r1"}},
            }
        ]
        flat = flatten_survey_rows(rows)
        self.assertEqual(len(flat), 1)
        self.assertEqual(flat[0]["method"], "method_a")


if __name__ == "__main__":
    unittest.main()
