# Final Result Inventory

Generated from the local repo state on branch `kiro` at commit `cc09242`, plus pulled artifacts and logs from:

- `user@140.123.102.157:/home/user/bitoguard-hackathon`
- `user@140.123.102.5:/home/user/aws/bitoguard-hackathon`
- `lab308@140.123.102.164:/home/lab308/bitoguard-hackathon`

Numbers below are taken from actual artifacts where available. Anything not backed by a final artifact is marked `partial`, `failed`, `running`, or `uncertain`.

| experiment name | branch / commit | artifact path | F1 | AP | threshold | rows | positives | status | notes |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| nested-CV final aggregate | runtime checkout `157 main@d505e4c` ; local snapshot `origin/main@e0c90fe` | [nested_oof_metrics.json](/Users/oscartsao/Developer/bitoguard-hackathon/bitoguard_core/artifacts/official_features/nested_hpo/nested_oof_metrics.json) | 0.3636 | 0.3222 | 0.18 | 51017 | 1640 | complete | pooled honest nested-CV result |
| hybrid primary locked eval | runtime checkout on `157` likely `main@d505e4c` (runtime commit inferred, uncertain); local snapshot `kiro@cc09242` | [hybrid_primary_locked_eval_report.json](/Users/oscartsao/Developer/bitoguard-hackathon/bitoguard_core/artifacts/official_features/hybrid_primary_locked_eval/hybrid_primary_locked_eval_report.json) | 0.3729 | 0.3192 | 0.17 | 51017 | 1640 | complete | current best mainline result |
| GraphSAGE 5-fold follow-up (`157`) | `157 main@d505e4c` | [graphsage_ablation_5fold_report.json](/Users/oscartsao/Developer/bitoguard-hackathon/bitoguard_core/artifacts/official_features/graphsage_ablation_5fold_157/graphsage_ablation_5fold_report.json) | 0.3645 | 0.3131 | per-fold | 51017 | 1640 | complete | mean over 5 folds; std F1 0.0213 |
| GraphSAGE 5-fold follow-up (`102.5`) | `102.5 autoresearch/overnight-20260326@c130428` | [graphsage_ablation_5fold_report.json](/Users/oscartsao/Developer/bitoguard-hackathon/bitoguard_core/artifacts/official_features/graphsage_ablation_5fold_1025_final/graphsage_ablation_5fold_report.json) | 0.3722 | 0.3170 | per-fold | 51017 | 1640 | complete | mean over 5 folds; std F1 0.0152 |
| XGB-only nested ablation (`157`) | `157 main@d505e4c` | [xgb_only_nested_ablation_157_report.json](/Users/oscartsao/Developer/bitoguard-hackathon/bitoguard_core/artifacts/official_features/xgb_only_nested_ablation_157_report.json) | 0.3687 | n/a | n/a | n/a | n/a | partial | final artifact exists, but reports `mean_best_f1` from inner objective only, not outer-honest final metric |
| XGB-only nested ablation (`102.5`) | `102.5 autoresearch/overnight-20260326@c130428` | [xgb_only_nested_ablation_1025_report.json](/Users/oscartsao/Developer/bitoguard-hackathon/bitoguard_core/artifacts/official_features/xgb_only_nested_ablation_1025_report.json) | 0.3771 | n/a | n/a | n/a | n/a | partial | final artifact exists, but reports `mean_best_f1` from inner objective only, not outer-honest final metric |
| XGB-only nested ablation (`164`) | runtime checkout / commit uncertain | remote log only: `/home/lab308/xgb_only_nested_ablation.log` | 0.3702 | n/a | n/a | n/a | n/a | failed | got through outer fold 0, then CatBoost GPU OOM on next fold |
| baseline-XGB bias measurement (`164`) | runtime checkout / commit uncertain | remote report only: `/home/lab308/bitoguard-hackathon/bitoguard_core/artifacts_xgb_baseline_bias/xgb_baseline_honest_report.json` | 0.3640 | 0.3207 | 0.1664 | 51017 | 1640 | complete | useful sensitivity check; not the same protocol as nested-CV |
| hybrid + Base C(GraphSAGE) locked eval | `157 main@d505e4c` ; local snapshot `kiro@cc09242` | [hybrid_primary_locked_eval_with_graphsage_report.json](/Users/oscartsao/Developer/bitoguard-hackathon/bitoguard_core/artifacts/official_features/hybrid_primary_locked_eval_with_graphsage/hybrid_primary_locked_eval_with_graphsage_report.json) | 0.3765 | 0.3194 | 0.1735 | 51017 | 1640 | complete | direct apples-to-apples mainline check: hybrid pipeline with Base C restored |
| XGB grid nested ablation (`102.5`, 5-fold targeted check) | `102.5 autoresearch/overnight-20260326@c130428` ; local snapshot `kiro@cc09242` | [summary.json](/Users/oscartsao/Developer/bitoguard-hackathon/bitoguard_core/artifacts/xgb_grid_nested_ablation/summary.json) | 0.3742 | n/a | n/a | n/a | n/a | complete (5-fold targeted check) | fold winners disagree: `(8,0.05)`, `(5,0.07)`, `(6,0.07)`, `(6,0.03)`, `(6,0.02)`; useful for pair selection only |
| provisional mainline submission | `157 main@d505e4c` ; local snapshot `kiro@cc09242` | [submission_provisional_graphsage_mainline.json](/Users/oscartsao/Developer/bitoguard-hackathon/bitoguard_core/artifacts/final_submission_current_mainline/submission_provisional_graphsage_mainline.json) | n/a | n/a | 0.1735 | 12753 | n/a | complete | generated from current mainline `hybrid + Base C(GraphSAGE)` config; provisional only |

## Notes

- For experiments executed on remote hosts without a committed runtime snapshot, the branch / commit is taken from the current remote checkout when verified. When the exact runtime commit could not be proven after the fact, it is marked `uncertain`.
- `XGB-only nested ablation` reports are **not directly comparable** to the final nested / hybrid / GraphSAGE runs because the saved output is `mean_best_f1` from the inner tuning objective, not a final outer-fold honest metric with threshold / rows / positives.
- `XGB grid nested ablation` is a cleaner targeted check than the old XGB-only Optuna runs, but it is still a **parameter-selection tool**, not a final apples-to-apples hybrid result.
