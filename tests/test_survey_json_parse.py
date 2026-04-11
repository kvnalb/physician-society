"""Tests for resilient multi-question survey JSON parsing."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from simulation.llm_client import parse_survey_json
from simulation.questions_io import load_questions


class TestSurveyJsonParse(unittest.TestCase):
    def setUp(self) -> None:
        self.qpath = Path(__file__).resolve().parents[1] / "simulation" / "questions.yaml"
        self.questions = load_questions(self.qpath)

    def test_full_parse_unchanged(self) -> None:
        answers = {
            q.question_id: {"option_id": q.options[0].option_id, "reasoning": "x"}
            for q in self.questions
        }
        raw = json.dumps({"answers": answers})
        out, err = parse_survey_json(raw, self.questions)
        self.assertIsNone(err)
        self.assertEqual(len(out), len(self.questions))

    def test_partial_map_from_valid_json(self) -> None:
        """When document parses but only some questions are present/valid, salvage those."""
        q0, q1 = self.questions[0], self.questions[1]
        raw = json.dumps(
            {
                "answers": {
                    q0.question_id: {"option_id": q0.options[0].option_id, "reasoning": "a"},
                    q1.question_id: {"option_id": q1.options[0].option_id, "reasoning": "b"},
                }
            }
        )
        out, err = parse_survey_json(raw, self.questions)
        self.assertTrue(err is not None and err.startswith("partial_survey_parse"))
        self.assertEqual(len(out), 2)
        self.assertIn(q0.question_id, out)
        self.assertIn(q1.question_id, out)

    def test_truncated_fence_salvages_leading_questions(self) -> None:
        """Unclosed ```json fence + truncated tail: still recover complete question objects."""
        q0 = self.questions[0]
        oid = q0.options[0].option_id
        # Opening fence only; JSON cuts off mid-stream after first full question object.
        raw = (
            "```json\n"
            '{"answers": {'
            f'"{q0.question_id}": {{"option_id": "{oid}", "reasoning": "Short."}}, '
            f'"{self.questions[1].question_id}": {{"option_id": "{self.questions[1].options[0].option_id}", '
            '"reasoning": "Truncated mid string without closing'
        )
        out, err = parse_survey_json(raw, self.questions)
        self.assertGreaterEqual(len(out), 1, out)
        self.assertIn(q0.question_id, out)
        self.assertEqual(out[q0.question_id]["option_id"], oid)

    def test_salvage_scoped_to_answers_object(self) -> None:
        """Question ids appearing only outside ``answers`` should not be picked up."""
        q0 = self.questions[0]
        oid = q0.options[0].option_id
        raw = (
            "Prologue mentions "
            + json.dumps({q0.question_id: "not the cell"})
            + ' {"answers": {'
            + f'"{q0.question_id}": {{"option_id": "{oid}", "reasoning": ""}}'
            + "}}\n"
        )
        out, err = parse_survey_json(raw, self.questions)
        self.assertIn(q0.question_id, out)
        self.assertEqual(out[q0.question_id]["option_id"], oid)

    def test_loads_from_temp_yaml_two_questions(self) -> None:
        """Minimal YAML for deterministic partial expectations."""
        with tempfile.TemporaryDirectory() as td:
            qpath = Path(td) / "q.yaml"
            qpath.write_text(
                "questions:\n"
                "  - question_id: qa\n    text: t\n    options:\n      - {option_id: a1, label: A}\n"
                "  - question_id: qb\n    text: t2\n    options:\n      - {option_id: b1, label: B}\n",
                encoding="utf-8",
            )
            qs = load_questions(qpath)
            raw = '{"answers": {"qa": {"option_id": "a1", "reasoning": ""}}}'
            out, err = parse_survey_json(raw, qs)
            self.assertEqual(err, "partial_survey_parse:1/2")
            self.assertEqual(list(out.keys()), ["qa"])


if __name__ == "__main__":
    unittest.main()
