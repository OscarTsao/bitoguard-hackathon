# BitoGuard Improvement Plan

> **Context**: Hackathon AML detection system for BitoPro exchange.
> Current best competition F1 = 0.3682 (threshold 0.1492).
> The model architecture (5-branch stacker) is saturated.
> The ceiling is **data quality + feature signal**, not model capacity.

All Python runs from `bitoguard_core/` with `PYTHONPATH=.`.
Competition pipeline: `bitoguard_core/official/`
Production pipeline: `bitoguard_core/models/stacker.py`

---

## Task Status

| Task | Status | Description |
|------|--------|-------------|
| 1 | ✅ DONE | `features/dormancy.py` — dormancy_score feature |
| 2 | ✅ DONE | `scripts/fix_graph_a7.py` — purge placeholder device edges |
| 3 | ✅ DONE | `features/event_ngram_features.py` — AML bigrams, trigrams, transition entropy |
| 4 | ✅ DONE | `features/statistical_features.py` — Benford chi2, amount entropy, burst score |
| 5 | ✅ DONE | `models/hpo_meta.py` — Optuna HPO for meta-learner |
| 6 | ✅ DONE | Wired all new features into `official/features.py` and `features/registry.py` |
| 7 | ✅ DONE | `tests/test_new_features.py` — 32/32 tests passing |

---

## Task 1: Dormancy-aware training split

### Problem
Some blacklisted users may have zero/near-zero behavioral activity. Explicit dormancy
feature lets the model use it directly rather than re-learning from scattered zero columns.

### Implementation
`bitoguard_core/features/dormancy.py`:
- `compute_dormancy_score(df)` — fraction of BEHAVIORAL_COLUMNS that are zero [0,1]
- `is_dormant(df, threshold=1.0)` — boolean Series
- `split_dormant_active(df)` — returns (dormant, active) tuple

Handles both production (`twd_dep_count`) and official (`twd_total_count`) column names.

### Validation
```bash
cd bitoguard_core && PYTHONPATH=. python -m pytest tests/test_new_features.py::TestDormancy -v
```

---

## Task 2: A7 graph cleanup — purge placeholder device edges

### Problem
Device ID `dev_cfcd208495d565ef66e7dff9f98764da` (MD5("0")) links ~78% of users
into one giant connected component. Makes `shared_device_count`, `component_size`,
`blacklist_1hop_count`, `blacklist_2hop_count` all useless.

### Implementation
`bitoguard_core/scripts/fix_graph_a7.py`:
- Pre-checks top device nodes by user count
- Calls `rebuild_entity_edges(store)` with placeholder filtering
- Falls back to direct SQL deletion if rebuild fails
- Post-checks to verify no placeholder devices remain

### Usage
```bash
cd bitoguard_core && PYTHONPATH=. python scripts/fix_graph_a7.py
```
**Note**: Only has effect if DuckDB has canonical.entity_edges data.
The official competition pipeline uses parquet files and already handles
this via MAX_IP_ENTITY_USERS=200, MAX_WALLET_ENTITY_USERS=200 caps.

---

## Task 3: Event sequence n-gram features

### Problem
Existing sequence features are aggregated statistics. They discard event ordering.
Money laundering has distinctive temporal signatures: `deposit → swap → withdraw`.

### Implementation
`bitoguard_core/features/event_ngram_features.py`:
- 7 AML-relevant bigrams (e.g., FD→SB, FD→CW, SB→CW)
- 4 AML-relevant trigrams (e.g., FD→SB→CW full layering chain)
- `seq_transition_entropy` — high = chaotic, low = repetitive/bot-like
- `seq_longest_streak` — longest consecutive same-type run
- `seq_inflow_outflow_ratio`, `seq_outflow_fraction`, `seq_n_unique_types`

Handles both column name conventions:
- Production: `occurred_at`, `direction`
- Official: `created_at`, `kind_label`

### Validation
```bash
cd bitoguard_core && PYTHONPATH=. python -m pytest tests/test_new_features.py::TestEventNgrams -v
```

---

## Task 4: Statistical features — Benford's law, entropy, burst detection

### Problem
Current features are mostly aggregations (sum, count, ratio). Three high-signal
feature families were missing.

### Implementation
`bitoguard_core/features/statistical_features.py`:
- `fiat_benford_chi2` / `crypto_benford_chi2` / `trade_benford_chi2`
- `fiat_amount_entropy` / `crypto_amount_entropy` / `trade_amount_entropy`
- `fiat_round_ratio` / `crypto_round_ratio` / `trade_benford_chi2`
- `fiat_burst_score` / `crypto_burst_score` / `trade_burst_score`
- `fiat_inter_event_cv` / `crypto_inter_event_cv`

Handles both `occurred_at` (production) and `created_at` (official) time columns.

### Validation
```bash
cd bitoguard_core && PYTHONPATH=. python -m pytest tests/test_new_features.py::TestStatisticalFeatures -v
```

---

## Task 5: Optuna HPO on meta-learner class weights and threshold

### Implementation
`bitoguard_core/models/hpo_meta.py`:
- `optimize_meta_learner(oof_matrix, y_true, n_trials=200)` — Bayesian search
- Searches: pos_weight [1,50], C [0.01,10], calibration [isotonic/sigmoid/none], threshold [0.05,0.60]
- Uses 5-fold CV on OOF matrix to avoid overfitting
- `build_optimized_meta_learner()` — convenience wrapper returns fitted model + threshold

### Usage
```bash
BITOGUARD_HPO=1 cd bitoguard_core && PYTHONPATH=. python models/stacker.py
```

### Validation
```bash
cd bitoguard_core && PYTHONPATH=. python -c "
import numpy as np; from models.hpo_meta import optimize_meta_learner
X = np.random.randn(1000, 5); y = (X[:, 0] + X[:, 1] > 0.5).astype(int)
r = optimize_meta_learner(X, y, n_trials=20)
print(f'HPO best_f1={r[\"best_f1\"]:.4f}')
"
```

---

## Task 6: Wire all new features into official pipeline

### Changes made
- `official/features.py`: Added dormancy_score + event n-gram + statistical features
  at end of `build_official_features()` before return
- `features/registry.py`: Added imports and `compute_event_ngram_features`,
  `compute_statistical_features` to `module_entries` list; `dormancy_score` added
  after final fillna

Both integrations use try/except to be best-effort (pipeline won't fail if module errors).

### Validation
```bash
cd bitoguard_core && PYTHONPATH=. python -c "
from features.dormancy import compute_dormancy_score
from features.event_ngram_features import compute_event_ngram_features
from features.statistical_features import compute_statistical_features
print('All new feature modules import successfully')
"
```

---

## Task 7: Tests

### File
`bitoguard_core/tests/test_new_features.py` — 32 tests, all passing

### Run
```bash
cd bitoguard_core && PYTHONPATH=. python -m pytest tests/test_new_features.py -v
```

---

## Experiment Results (GPU experiments)

| Experiment | AP (OOF) | F1 (OOF) | vs Base A | Notes |
|-----------|---------|---------|----------|-------|
| Base A (CatBoost 4-seed) | 0.2974 | 0.3682 (blend) | baseline | Current best |
| TabNet | 0.1872 | 0.1693 | -0.1102 AP | Failed — below blend threshold |
| SMOTE | 0.2179 | — | -0.0175 AP | Hurts AP, raises F1 via threshold artefact |
| BorderlineSMOTE | 0.2206 | — | -0.0148 AP | Same — not beneficial |
| Temporal gap features | TBD | TBD | TBD | Running (CPU) |

**Key insight**: TabNet significantly underperforms CatBoost on this dataset. SMOTE reduces
AP (precision) while raising apparent F1 only via threshold shift — not a real improvement.

---

## Next Steps

1. **Run official pipeline** with new features → measure actual F1 impact
2. **Run temporal gap features** (CPU) → check AP vs Base A
3. **HPO on meta-learner** → may improve threshold selection
4. If AP improvement found → incorporate into blend

## Execution Commands

```bash
cd bitoguard_core

# Fix graph (only if DuckDB has data)
PYTHONPATH=. python scripts/fix_graph_a7.py

# Run tests
PYTHONPATH=. python -m pytest tests/test_new_features.py -v

# Run official pipeline with new features
PYTHONPATH=. python official/pipeline.py

# Optional: HPO on meta-learner
BITOGUARD_HPO=1 PYTHONPATH=. python models/stacker.py
```
