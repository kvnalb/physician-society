"""Survey agreement and descriptive behavioral summaries."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sklearn.metrics import cohen_kappa_score

from simulation.questions_io import load_questions


def _load_jsonl(path: Path) -> List[dict[str, Any]]:
    rows: List[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def compute_survey_agreement(
    responses_jsonl: Path,
    questions_yaml: Path | None = None,
) -> dict[str, Any]:
    rows = _load_jsonl(responses_jsonl)
    questions = load_questions(questions_yaml)

    by_pair: Dict[Tuple[str, str], Dict[str, str]] = {}
    for r in rows:
        opt = r.get("parsed_option")
        if not opt or r.get("error"):
            continue
        npi = str(r["npi"])
        qid = str(r["question_id"])
        method = str(r["method"])
        key = (npi, qid)
        if key not in by_pair:
            by_pair[key] = {}
        by_pair[key][method] = opt

    per_question: Dict[str, Any] = {}
    kappas: List[float] = []

    for q in questions:
        ya: List[str] = []
        yb: List[str] = []
        dist_a: Counter[str] = Counter()
        dist_b: Counter[str] = Counter()
        for (npi, qid), methods in by_pair.items():
            if qid != q.question_id:
                continue
            a = methods.get("method_a")
            b = methods.get("method_b")
            if a is None or b is None:
                continue
            ya.append(a)
            yb.append(b)
            dist_a[a] += 1
            dist_b[b] += 1

        per_question[q.question_id] = {
            "n_paired": len(ya),
            "method_a_dist": dict(dist_a),
            "method_b_dist": dict(dist_b),
        }
        if len(ya) >= 2 and len(set(ya + yb)) > 0:
            try:
                k = float(cohen_kappa_score(ya, yb))
                per_question[q.question_id]["cohen_kappa"] = k
                kappas.append(k)
            except ValueError:
                per_question[q.question_id]["cohen_kappa"] = None
        else:
            per_question[q.question_id]["cohen_kappa"] = None

    mean_kappa = sum(kappas) / len(kappas) if kappas else None
    if mean_kappa is None:
        stability = "Insufficient paired Method A/B responses to assess agreement."
    elif mean_kappa < 0.2:
        stability = "Methods A and B show weak agreement (low Cohen's kappa on average)."
    elif mean_kappa < 0.5:
        stability = "Methods A and B show moderate agreement on average."
    else:
        stability = "Methods A and B show relatively strong agreement on average."

    return {
        "method_agreement_kappa_mean": mean_kappa,
        "per_question": per_question,
        "stability": stability,
    }


def compute_metrics_bundle(
    responses_jsonl: Path,
    *,
    questions_yaml: Path | None = None,
) -> dict[str, Any]:
    return {
        "survey": compute_survey_agreement(responses_jsonl, questions_yaml),
    }
