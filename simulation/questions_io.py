"""Load survey questions from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass(frozen=True)
class QuestionOption:
    option_id: str
    label: str


@dataclass(frozen=True)
class Question:
    question_id: str
    text: str
    options: List[QuestionOption]
    ground_truth_field: str | None = None


def load_questions(path: Path | None = None) -> List[Question]:
    if path is None:
        path = Path(__file__).resolve().parent / "questions.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    out: List[Question] = []
    for item in raw.get("questions", []):
        opts = [
            QuestionOption(option_id=o["option_id"], label=o["label"])
            for o in item["options"]
        ]
        out.append(
            Question(
                question_id=item["question_id"],
                text=str(item["text"]).strip(),
                options=opts,
                ground_truth_field=item.get("ground_truth_field"),
            )
        )
    return out


def question_ids(questions: List[Question]) -> List[str]:
    return [q.question_id for q in questions]


def validate_option_id(question: Question, option_id: str) -> bool:
    return any(o.option_id == option_id for o in question.options)


def format_question_block(q: Question) -> str:
    lines = [q.text.strip(), "", "Choose exactly one option by returning ONLY its option_id on the first line."]
    lines.append("")
    for o in q.options:
        lines.append(f"- {o.option_id}: {o.label}")
    lines.extend(
        [
            "",
            "Output format:",
            "Line 1: the option_id only (no punctuation, no quotes).",
            "Optional lines 2+: brief clinical rationale (one short paragraph).",
        ]
    )
    return "\n".join(lines)
