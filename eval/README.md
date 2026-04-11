# Evaluation bundle

- **`metrics.py`**: **hold-out `behavioral_alignment`** (per-NPI match vs pseudo-labels from **post-2022** Part D fields), **`distribution_quality`** (cohort-level JS/TV between simulated answers and pseudo-label histograms), **`survey_marginals`** (per-item simulated answer histograms from the **method_a** stream), **`instrument_health`**, **`persona_coherence`**.
- **`behavioral_labels.py`**: pseudo-label mapping for the **forward** survey (`RULES_VERSION`).
- **`coherence_rules.py`**: deterministic cross-item checks (legacy q3/q4 rules apply only when those items exist).
- **`instrument_health.py`**: parse/coverage/latency summaries from response JSONL (v2 **method_a** block).
- **`run_eval.py`**: CLI to write `metrics.json`.

## Optional external reference marginals (deferred)

To benchmark **synthetic marginals vs a human or syndicated tracker** (Artificial Societies–style “distribution accuracy”), add a CSV such as:

- `question_id`, `option_id`, `reference_share` (non-negative, summing to 1 per question)

and wire a future CLI flag in `run_eval.py`. A column template is described in [`reference_marginals.example.md`](reference_marginals.example.md). **Not used by default** to avoid scope and licensing creep.
