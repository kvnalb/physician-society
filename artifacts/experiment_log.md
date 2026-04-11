# Experiment log: persona variants vs eval

Committed summary of batch runs. Each run directory includes **`run_manifest.json`** (CLI config snapshot) and, after `eval.run_eval`, **`metrics.json`** beside `responses.jsonl` (same eval blob can also be copied to `artifacts/runs/<run_id>_metrics.json`). Streamlit can switch snapshots via **Eval metrics snapshot**. Pseudo-label rules: `eval/behavioral_labels.py` (`RULES_VERSION`).

| run_id | persona_variant | methods | NPIs | wall_time | model | mean_kappa | mean_behavioral_acc (method_a) | notes |
|--------|-----------------|---------|------|-----------|-------|------------|--------------------------------|-------|
| v_offline_100 | offline_seed | A+B | 100 | — | offline_seed | (see metrics) | (see metrics) | Deterministic smoke; not LLM. Replace rows after real API runs. |
| v0_naive | naive | A only | 100 | | gpt-4o-mini | n/a | | Fill after: `run_batch --run-id v0_naive --persona-variant naive --method A` |
| v1_b | b | A only | 100 | | gpt-4o-mini | n/a | | `... --persona-variant b --method A` |
| v2_ab | ab | A+B | 100 | | gpt-4o-mini | | | `... --run-id v2_ab --persona-variant ab --method both` |
| v3_a_numeric | a_numeric | A+B | 100 | | gpt-4o-mini | | | Rich A + numeric guardrails |

## Commands

```bash
# Example: full grid cell (requires API key + cohort TSV)
python -m simulation.run_batch --run-id v2_ab --persona-variant ab --method both --concurrency 12

python -m eval.run_eval \
  --responses-file data/output/runs/v2_ab/responses.jsonl \
  --output artifacts/runs/v2_ab_metrics.json

# Inspect one persona
python -m simulation.persona_query --npi <NPI> --run-id v2_ab
```

Refresh this table after each run (mean_kappa from `survey.method_agreement_kappa_mean`, mean_behavioral_acc from `behavioral_alignment.mean_accuracy_over_labeled_questions`).
