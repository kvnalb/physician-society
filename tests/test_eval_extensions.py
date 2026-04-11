"""Eval bundle: survey marginals, coherence, instrument health."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.coherence_rules import coherence_violations_for_npi, compute_persona_coherence
from eval.instrument_health import compute_instrument_health
from eval.metrics import compute_survey_marginals


class TestEvalExtensions(unittest.TestCase):
    def test_compute_survey_marginals(self) -> None:
        rows = [
            {"npi": "1", "question_id": "q1", "method": "method_a", "parsed_option": "a1"},
            {"npi": "2", "question_id": "q1", "method": "method_a", "parsed_option": "a1"},
            {"npi": "3", "question_id": "q1", "method": "method_a", "parsed_option": "b1"},
        ]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "r.jsonl"
            with open(p, "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            qpath = Path(td) / "q.yaml"
            qpath.write_text(
                "questions:\n"
                "  - question_id: q1\n"
                "    text: t\n"
                "    options:\n"
                "      - {option_id: a1, label: A}\n"
                "      - {option_id: b1, label: B}\n",
                encoding="utf-8",
            )
            out = compute_survey_marginals(p, questions_yaml=qpath)
            pq = out["per_question"]["q1"]
            self.assertEqual(pq["n_responses"], 3)
            self.assertEqual(pq["response_distribution"], {"a1": 2, "b1": 1})

    def test_coherence_q4_requires_q3_yes(self) -> None:
        v = coherence_violations_for_npi(
            {"q3_tirzepatide_prescribed": "q3_no_unlikely", "q4_tirzepatide_adoption_speed": "q4_early"},
            method_label="method_a",
            npi="x",
        )
        self.assertTrue(any(x["rule_id"] == "no_prescribe_implies_q4_na" for x in v))

    def test_instrument_health_v2(self) -> None:
        row = {
            "schema_version": 2,
            "npi": "1",
            "method_a": {"q1": {"option_id": "a", "reasoning": ""}},
            "latency_ms_by_method": {"method_a": 100},
            "survey_error_by_method": {"method_a": None},
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "r.jsonl"
            p.write_text(json.dumps(row) + "\n", encoding="utf-8")
            qpath = Path(td) / "q.yaml"
            qpath.write_text(
                "questions:\n  - question_id: q1\n    text: t\n    options:\n      - {option_id: a, label: A}\n      - {option_id: b, label: B}\n",
                encoding="utf-8",
            )
            h = compute_instrument_health(p, questions_yaml=qpath)
            self.assertEqual(h["n_v2_rows"], 1)
            self.assertEqual(h["latency_ms"]["n_calls_with_latency"], 1)
            self.assertEqual(h["v2_expected_answer_cells"], 1)
            self.assertEqual(h["missing_answer_cell_rate"], 0.0)

    def test_instrument_health_v2_empty_block_counts_missing(self) -> None:
        """Failed joint survey (empty ``method_a``) must count all questions as missing."""
        row = {
            "schema_version": 2,
            "npi": "9",
            "method_a": {},
            "latency_ms_by_method": {"method_a": 50},
            "survey_error_by_method": {"method_a": "json_decode:Expecting value: line 1 column 1 (char 0)"},
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "r.jsonl"
            p.write_text(json.dumps(row) + "\n", encoding="utf-8")
            qpath = Path(td) / "q.yaml"
            qpath.write_text(
                "questions:\n"
                "  - question_id: q1\n    text: t\n    options:\n      - {option_id: a, label: A}\n"
                "  - question_id: q2\n    text: t2\n    options:\n      - {option_id: x, label: X}\n",
                encoding="utf-8",
            )
            h = compute_instrument_health(p, questions_yaml=qpath)
            self.assertEqual(h["v2_missing_question_cells"], 2)
            self.assertEqual(h["v2_expected_answer_cells"], 2)
            self.assertEqual(h["missing_answer_cell_rate"], 1.0)
            self.assertEqual(h["flat_cells_missing_option"], 2)

    def test_compute_persona_coherence_empty(self) -> None:
        pc = compute_persona_coherence([], question_ids=["q3_tirzepatide_prescribed", "q4_tirzepatide_adoption_speed"])
        self.assertEqual(pc["n_method_blocks_checked"], 0)

    def test_instrument_health_single_method_block(self) -> None:
        row = {
            "schema_version": 2,
            "npi": "1",
            "method_a": {"q1": {"option_id": "a", "reasoning": ""}},
            "latency_ms_by_method": {"method_a": 100},
            "survey_error_by_method": {"method_a": None},
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "r.jsonl"
            p.write_text(json.dumps(row) + "\n", encoding="utf-8")
            qpath = Path(td) / "q.yaml"
            qpath.write_text(
                "questions:\n  - question_id: q1\n    text: t\n    options:\n      - {option_id: a, label: A}\n",
                encoding="utf-8",
            )
            h = compute_instrument_health(p, questions_yaml=qpath)
            self.assertEqual(h["n_v2_rows"], 1)
            self.assertEqual(h["latency_ms"]["n_calls_with_latency"], 1)
            self.assertEqual(h["v2_missing_question_cells"], 0)
            self.assertEqual(h["v2_expected_answer_cells"], 1)
            self.assertEqual(h["missing_answer_cell_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
