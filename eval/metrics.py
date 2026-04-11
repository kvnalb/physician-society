"""Survey agreement, behavioral pseudo-label alignment, and distribution distance."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import rel_entr
from sklearn.metrics import cohen_kappa_score

from eval.behavioral_labels import RULES_VERSION, pseudo_label_for_question
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


def compute_survey_agreement(
    responses_jsonl: Path,
    questions_yaml: Path | None = None,
) -> dict[str, Any]:
    rows = _load_jsonl_flat(responses_jsonl)
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
    method_for_alignment: str = "method_a",
) -> dict[str, Any]:
    """
    Compare LLM picks to claims-derived pseudo-labels for the same NPI.
    method_for_alignment: 'method_a' or 'method_b' (filters JSONL rows).
    """
    questions = load_questions(questions_yaml)
    cohort_df = pd.read_csv(cohort_path, sep="\t", low_memory=False, dtype={"npi": str})
    cohort_by_npi = {str(r["npi"]): r for _, r in cohort_df.iterrows()}

    rows = _load_jsonl_flat(responses_jsonl)
    want_method = method_for_alignment.strip().lower()
    if want_method not in ("method_a", "method_b"):
        raise ValueError("method_for_alignment must be method_a or method_b")

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
        "method_for_alignment": want_method,
        "mean_accuracy_over_labeled_questions": mean_acc,
        "per_question": per_q,
        "note": "Pseudo-labels are claims-derived heuristics, not human survey responses.",
    }


def compute_metrics_bundle(
    responses_jsonl: Path,
    *,
    questions_yaml: Path | None = None,
    cohort_path: Path | None = None,
    method_for_alignment: str = "method_a",
) -> dict[str, Any]:
    out: Dict[str, Any] = {
        "survey": compute_survey_agreement(responses_jsonl, questions_yaml),
    }
    if cohort_path is not None and cohort_path.is_file():
        out["behavioral_alignment"] = compute_behavioral_alignment(
            responses_jsonl,
            cohort_path,
            questions_yaml=questions_yaml,
            method_for_alignment=method_for_alignment,
        )
    return out
