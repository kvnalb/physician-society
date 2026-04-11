# Optional external reference marginals (future)

If you license or obtain human / syndicated reference shares per question, use a CSV with columns:

`question_id`, `option_id`, `reference_share` (non-negative, sum to 1.0 per `question_id`).

Example rows (not real data):

```csv
question_id,option_id,reference_share
q2_glp1_panel_share,q2_lt5,0.12
q2_glp1_panel_share,q2_5_15,0.28
```

Wire a future `run_eval` flag to compare synthetic marginals to this file. **Not used by default.**
