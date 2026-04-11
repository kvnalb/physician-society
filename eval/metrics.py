"""Hold-out pseudo-label alignment, simulated marginals, distribution distance, coherence."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.special import rel_entr

from eval.behavioral_labels import RULES_VERSION, pseudo_label_for_question
from eval.coherence_rules import compute_persona_coherence
from eval.instrument_health import compute_instrument_health, load_raw_response_rows
from simulation.questions_io import load_questions
from simulation.responses_schema import flatten_survey_rows


def _load_jsonl(path: Path) -> List[dict[str, Any]]:
    rows: List[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_jsonl_flat(path: Path) -> List[dict[str, Any]]:
    return flatten_survey_rows(_load_jsonl(path))


def compute_survey_marginals(
    responses_jsonl: Path,
    questions_yaml: Path | None = None,
) -> dict[str, Any]:
    """
    Per-question counts of simulated answers (single stream; flattened ``method_a`` rows only).
    """
    rows = _load_jsonl_flat(responses_jsonl)
    questions = load_questions(questions_yaml)
    per_question: Dict[str, Any] = {}

    for q in questions:
        dist: Counter[str] = Counter()
        for r in rows:
            if str(r.get("method", "")).lower() != "method_a":
                continue
            if r.get("question_id") != q.question_id:
                continue
            opt = r.get("parsed_option")
            if not opt or r.get("error"):
                continue
            dist[str(opt)] += 1
        n = int(sum(dist.values()))
        per_question[q.question_id] = {
            "n_responses": n,
            "response_distribution": dict(dist),
        }

    return {
        "note": "Single-stream simulated marginals (v2 ``method_a`` survey block only).",
        "per_question": per_question,
    }


def _js_divergence(p: Dict[str, float], q: Dict[str, float], keys: List[str]) -> float:
    """Jensen–Shannon divergence (natural log) on aligned probability vectors."""
    eps = 1e-12
    pv = np.array([p.get(k, 0.0) + eps for k in keys], dtype=float)
    qv = np.array([q.get(k, 0.0) + eps for k in keys], dtype=float)
    pv = pv / pv.sum()
    qv = qv / qv.sum()
    m = 0.5 * (pv + qv)
    kl_pm = float(np.sum(rel_entr(pv, m)))
    kl_qm = float(np.sum(rel_entr(qv, m)))
    return 0.5 * (kl_pm + kl_qm)


def _tv_distance(p: Dict[str, float], q: Dict[str, float], keys: List[str]) -> float:
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def compute_behavioral_alignment(
    responses_jsonl: Path,
    cohort_path: Path,
    *,
    questions_yaml: Path | None = None,
) -> dict[str, Any]:
    """
    Compare LLM picks (``method_a`` rows) to claims-derived pseudo-labels for the same NPI (hold-out fields).
    """
    questions = load_questions(questions_yaml)
    cohort_df = pd.read_csv(cohort_path, sep="\t", low_memory=False, dtype={"npi": str})
    cohort_by_npi = {str(r["npi"]): r for _, r in cohort_df.iterrows()}

    rows = _load_jsonl_flat(responses_jsonl)
    want_method = "method_a"

    per_q: Dict[str, Any] = {}
    accuracies: List[float] = []

    for q in questions:
        qid = q.question_id
        option_ids = [o.option_id for o in q.options]
        correct = 0
        total = 0
        pred_dist: Counter[str] = Counter()
        gold_dist: Counter[str] = Counter()

        for r in rows:
            if str(r.get("method", "")).lower() != want_method:
                continue
            if r.get("question_id") != qid:
                continue
            pred = r.get("parsed_option")
            if not pred or r.get("error"):
                continue
            npi = str(r["npi"])
            crow = cohort_by_npi.get(npi)
            if crow is None:
                continue
            gold = pseudo_label_for_question(crow, qid)
            if gold is None:
                continue
            total += 1
            pred_dist[pred] += 1
            gold_dist[gold] += 1
            if pred == gold:
                correct += 1

        acc = correct / total if total else None
        if acc is not None:
            accuracies.append(acc)

        p_norm = {k: pred_dist[k] / total for k in option_ids} if total else {}
        g_norm = {k: gold_dist[k] / total for k in option_ids} if total else {}
        js = _js_divergence(p_norm, g_norm, option_ids) if total else None
        tv = _tv_distance(p_norm, g_norm, option_ids) if total else None

        per_q[qid] = {
            "n_labeled": total,
            "accuracy": acc,
            "js_divergence_marginal": js,
            "tv_distance_marginal": tv,
            "pred_distribution": dict(pred_dist),
            "pseudo_label_distribution": dict(gold_dist),
        }

    mean_acc = sum(accuracies) / len(accuracies) if accuracies else None
    return {
        "rules_version": RULES_VERSION,
        "mean_accuracy_over_labeled_questions": mean_acc,
        "per_question": per_q,
        "note": (
            "Pseudo-labels use **post-2022** Part D (and related) fields in the cohort TSV, not human surveys. "
            "Production personas must **exclude** those fields from the LLM prompt when interpreting alignment."
        ),
    }


def _distribution_quality_from_holdout(behavioral: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Summarize cohort-level JS/TV between simulated answers and hold-out pseudo marginals."""
    if not behavioral or not behavioral.get("per_question"):
        return {
            "pillar": "Simulated vs Part D hold-out pseudo marginals (per question).",
            "mean_js_sim_vs_holdout": None,
            "mean_tv_sim_vs_holdout": None,
            "per_question": {},
        }
    pq = behavioral["per_question"]
    rows = []
    js_vals: List[float] = []
    tv_vals: List[float] = []
    for qid, v in pq.items():
        js = v.get("js_divergence_marginal")
        tv = v.get("tv_distance_marginal")
        if js is not None:
            js_vals.append(float(js))
        if tv is not None:
            tv_vals.append(float(tv))
        rows.append(
            {
                "question_id": qid,
                "js_sim_vs_holdout": js,
                "tv_sim_vs_holdout": tv,
                "n_labeled": v.get("n_labeled"),
            }
        )
    return {
        "pillar": (
            "Cohort-level distance between **simulated** answer histograms and **hold-out** "
            "pseudo-label histograms built from post-2022 Medicare fields."
        ),
        "mean_js_sim_vs_holdout": sum(js_vals) / len(js_vals) if js_vals else None,
        "mean_tv_sim_vs_holdout": sum(tv_vals) / len(tv_vals) if tv_vals else None,
        "per_question": {r["question_id"]: r for r in rows},
    }


def compute_metrics_bundle(
    responses_jsonl: Path,
    *,
    questions_yaml: Path | None = None,
    cohort_path: Path | None = None,
) -> dict[str, Any]:
    survey_marginals = compute_survey_marginals(responses_jsonl, questions_yaml)
    behavioral: Optional[dict[str, Any]] = None
    if cohort_path is not None and cohort_path.is_file():
        behavioral = compute_behavioral_alignment(
            responses_jsonl,
            cohort_path,
            questions_yaml=questions_yaml,
        )
    out: Dict[str, Any] = {
        "survey_marginals": survey_marginals,
        "instrument_health": compute_instrument_health(responses_jsonl, questions_yaml=questions_yaml),
        "distribution_quality": _distribution_quality_from_holdout(behavioral),
    }
    questions = load_questions(questions_yaml)
    qids = [q.question_id for q in questions]
    out["persona_coherence"] = compute_persona_coherence(load_raw_response_rows(responses_jsonl), question_ids=qids)
    if behavioral is not None:
        out["behavioral_alignment"] = behavioral
    return out
