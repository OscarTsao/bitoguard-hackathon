# GraphSAGE Reproducibility Audit

Goal: explain why the 5-fold GraphSAGE follow-up on `157` produced `mean F1 = 0.3645` while `102.5` produced `mean F1 = 0.3722`.

## Compared runs

- `157`: [graphsage_ablation_5fold_report.json](/Users/oscartsao/Developer/bitoguard-hackathon/bitoguard_core/artifacts/official_features/graphsage_ablation_5fold_157/graphsage_ablation_5fold_report.json)
- `102.5`: [graphsage_ablation_5fold_report.json](/Users/oscartsao/Developer/bitoguard-hackathon/bitoguard_core/artifacts/official_features/graphsage_ablation_5fold_1025_final/graphsage_ablation_5fold_report.json)

## Summary

The two runs are **not fully reproducible**. The strongest confirmed reason is that they were not executed from the same repo checkout and not against byte-identical input artifacts.

## Audit checklist

| item | `157` | `102.5` | status | notes |
|---|---|---|---|---|
| same branch | `main` | `autoresearch/overnight-20260326` | confirmed different | this alone is enough to treat the comparison as non-reproducible |
| same commit | `d505e4cd70d37579213b39686d46b8cba4ad9a06` | `c13042856c93167cfbcf16ad1cececdd700acf56` | confirmed different | codebase differed at runtime |
| same frozen params | yes, report contents match | yes, report contents match | confirmed same | CatBoost / LGBM / XGB frozen params in both logs match exactly |
| same primary split artifact | SHA256 `9e769406...` | SHA256 `fe178183...` | confirmed different | primary split parquet differs across machines |
| same feature snapshot | SHA256 `f468d8b7...` | SHA256 `e578fed2...` | confirmed different | official user feature parquet differs across machines |
| same GraphSAGE script shape | same call sites found | same call sites found | mostly same | both scripts call `train_graphsage_model(..., hidden_dim=128)` for inner and final passes |
| same GraphSAGE hyperparams | `hidden_dim=128`, `PRIMARY_GRAPH_MAX_EPOCHS`, `FINAL_GRAPH_MIN_EPOCHS` | same | confirmed same at script level | grep showed identical call signatures |
| same Python stack | Python 3.12.3 | Python 3.12.0 (conda-forge) | confirmed different | minor runtime stack difference |
| same package versions | numpy 2.4.3, pandas 2.3.2, catboost 1.2.10, lightgbm 4.6.0, xgboost 3.2.0, torch 2.11.0+cpu | same package versions for these libs | mostly same | main numerical libs match; Python build differs |
| same GPU / driver | RTX 5090, driver 580.126.09 | RTX 5090, driver 580.126.09 | confirmed same | hardware class matches |
| same artifact paths | `/home/user/bitoguard-hackathon/...` | `/home/user/aws/bitoguard-hackathon/...` | confirmed different | separate worktrees and artifact dirs |
| fallback / skipped stage | no explicit fallback evidence in final report | repeated “less than 75% GPU memory available” warnings | uncertain impact | warnings suggest different memory pressure on `102.5` |
| same protocol | 5-fold GraphSAGE follow-up with honest selection | same | confirmed same high-level protocol | protocol matched, but inputs/checkouts differed |

## Most likely difference sources ranked

1. **Different repo commits / branches**
   - Confirmed. `157` ran from `main@d505e4c`, `102.5` ran from `autoresearch/overnight-20260326@c130428`.
   - This is the highest-confidence cause of divergence.

2. **Different feature snapshot and split artifacts**
   - Confirmed by SHA256 mismatch for:
     - `official_primary_transductive_split_full.parquet`
     - `official_user_features_full.parquet`
   - This means the runs did not consume identical precomputed inputs.

3. **Different artifact/worktree roots**
   - Confirmed.
   - Increases risk that “same named file” is not the same content.

4. **Different runtime environment at the Python build level**
   - Confirmed (`3.12.3` vs `3.12.0 conda-forge`).
   - Lower-likelihood than the branch / artifact differences, but still a source of drift.

5. **GPU memory pressure on `102.5`**
   - Confirmed warnings in the `102.5` log.
   - Impact is plausible but not proven.

6. **GraphSAGE nondeterminism**
   - Still possible.
   - But not the first explanation to chase, because stronger confirmed differences already exist.

## Confirmed findings

- The two GraphSAGE 5-fold follow-ups were **not run from the same checkout**.
- The two runs **did not use byte-identical primary split or feature snapshot artifacts**.
- The frozen tabular params were the same.
- The nominal GraphSAGE call pattern (`hidden_dim=128`, same epoch constants) was the same.

## Still uncertain

- Whether GPU memory pressure on `102.5` improved or degraded GraphSAGE quality.
- Whether any subtle PyTorch nondeterminism contributed materially after controlling for artifact differences.
- Whether the differing feature snapshots reflect benign regeneration drift or a substantive data / cutoff mismatch.

## Audit conclusion

The `157` vs `102.5` GraphSAGE gap **cannot** be treated as a clean model-performance comparison. The runs were not reproducible because both the code checkout and precomputed artifacts differed. The current evidence is enough to say:

- `102.5`'s better GraphSAGE result is **interesting**
- but it is **not yet trustworthy as a decisive upgrade signal**

Before promoting GraphSAGE into the mainline, the minimum bar should be one clean rerun from the **same commit** and **same artifact snapshot** on both machines, or a single authoritative rerun in one controlled environment.
