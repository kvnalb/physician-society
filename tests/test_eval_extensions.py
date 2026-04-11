"""Eval bundle: method marginal divergence, coherence, instrument health."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.coherence_rules import coherence_violations_for_npi, compute_persona_coherence
from eval.instrument_health import compute_instrument_health
from eval.metrics import compute_survey_agreement


class TestEvalExtensions(unittest.TestCase):
    def test_survey_js_method_ab(self) -> None:
        rows = [
            {"npi": "1", "question_id": "q1", "method": "method_a", "parsed_option": "a1"},
            {"npi": "1", "question_id": "q1", "method": "method_b", "parsed_option": "b1"},
            {"npi": "2", "question_id": "q1", "method": "method_a", "parsed_option": "a1"},
            {"npi": "2", "question_id": "q1", "method": "method_b", "parsed_option": "a1"},
        ]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "r.jsonl"
            with open(p, "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            # Minimal questions file
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
            out = compute_survey_agreement(p, questions_yaml=qpath)
            pq = out["per_question"]["q1"]
            self.assertIsNotNone(pq.get("js_method_ab_marginal"))
            self.assertIsNotNone(pq.get("tv_method_ab_marginal"))

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
            "method_b": {"q1": {"option_id": "b", "reasoning": ""}},
            "latency_ms_by_method": {"method_a": 100, "method_b": 120},
            "survey_error_by_method": {"method_a": None, "method_b": None},
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
            self.assertEqual(h["latency_ms"]["n_calls_with_latency"], 2)

    def test_compute_persona_coherence_empty(self) -> None:
        pc = compute_persona_coherence([], question_ids=["q3_tirzepatide_prescribed", "q4_tirzepatide_adoption_speed"])
        self.assertEqual(pc["n_method_blocks_checked"], 0)


if __name__ == "__main__":
    unittest.main()
