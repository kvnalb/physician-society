# Retest / self-replication stability

Use this when you want a **cheap** analogue of “self-replication” style checks (without recruiting a human panel):

1. Run the batch twice with the **same** cohort slice, model, persona variant, and temperature (ideally **temperature 0**), but **different** `--run-id` (so outputs land in separate directories).
2. Compare flattened `(npi, question_id, method)` cells with:

```bash
cd /path/to/physician-society
python scripts/compare_runs_stability.py \
  --run-a data/output/runs/run_a/responses__MODEL.jsonl \
  --run-b data/output/runs/run_b/responses__MODEL.jsonl
```

The script prints **item agreement rate** and a split by `method_a` / `method_b`. Interpret as **stochastic stability** of the joint survey completion, not human accuracy.

For **order sensitivity**, run one batch with default question order and one with `--shuffle-questions --shuffle-seed 1`, then compare the same way on a small `--limit-npis` subset.
