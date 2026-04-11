# Evaluation bundle

- **`metrics.py`**: survey agreement (κ, method A vs B marginal JS/TV), `instrument_health`, `distribution_quality`, `persona_coherence` (cross-item rules), and optional `behavioral_alignment` vs claims-derived pseudo-labels.
- **`behavioral_labels.py`**: pseudo-label mapping (`RULES_VERSION`).
- **`coherence_rules.py`**: deterministic cross-item checks on parsed options.
- **`instrument_health.py`**: parse/coverage/latency summaries from response JSONL.
- **`run_eval.py`**: CLI to write `metrics.json`.

## Optional external reference marginals (deferred)

To benchmark **synthetic marginals vs a human or syndicated tracker** (Artificial Societies–style “distribution accuracy”), add a CSV such as:

- `question_id`, `option_id`, `reference_share` (non-negative, summing to 1 per question)

and wire a future CLI flag in `run_eval.py`. A column template is described in [`reference_marginals.example.md`](reference_marginals.example.md). **Not used by default** to avoid scope and licensing creep.
