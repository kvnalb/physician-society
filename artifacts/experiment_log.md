# Experiment log (template)

| run_id | persona_variant | NPIs | wall_time | model | mean_js_holdout | mean_behavioral_acc | notes |
|--------|-----------------|------|-----------|-------|-----------------|---------------------|-------|
| v0_naive | naive | 100 | | gpt-4o-mini | | | `python -m simulation.run_batch --run-id v0_naive --persona-variant naive` |
| v1_production | production | 100 | | gpt-4o-mini | | | Default 2022-only persona |
| v2_rich_a | a | 100 | | gpt-4o-mini | | | Rich persona with 2022 tirzepatide line |

Refresh this table after each run (`distribution_quality.mean_js_sim_vs_holdout`, `behavioral_alignment.mean_accuracy_over_labeled_questions`).
