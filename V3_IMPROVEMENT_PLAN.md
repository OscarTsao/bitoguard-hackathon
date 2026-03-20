# BitoGuard V3 — Definitive Improvement Plan

> **Status**: In progress.  Current F1 = 0.3682 → Target F1 ≈ 0.42–0.46
>
> Pipeline: `bitoguard_core/official/`
> All Python: `cd bitoguard_core && PYTHONPATH=.`

---

## Task Status

| # | Task | Status | Expected ΔF1 |
|---|------|--------|--------------|
| 1 | Self-training pseudo-label expansion | ✅ DONE | +0.02~0.06 |
| 2 | Multi-scale PPR (alpha=0.05/0.20/0.50) | ✅ DONE (v45) | +0.02~0.04 |
| 3 | Temporal co-occurrence edges | ✅ DONE (v47) | +0.01~0.02 |
| 4 | Edge weight HPO via Optuna | ✅ DONE | +0.01~0.03 |
| 5 | GraphSAGE 3-layer + hidden_dim=96 + patience=12 | ✅ DONE (v47) | +0.005~0.02 |
| 6 | Non-linear stacker comparison | pending | +0.005~0.015 |
| 7 | Optuna threshold + calibration HPO | pending | +0.01~0.03 |
| 8 | Run full pipeline + self-training + measure | pending | — |

---

## Task 1: Self-training (pseudo-label expansion) — HIGHEST ROI

### Problem

PPR propagation only reaches users within graph distance of the 1,608 known positives. Users in disconnected components or far regions get zero propagation signal. Self-training expands the seed set iteratively.

### Implementation

`bitoguard_core/official/self_training.py`

- `expand_with_pseudo_labels(dataset, oof_path, threshold=0.70, max_new=200)` — loads OOF predictions, selects pseudo-positives from predict_label users above threshold
- `run_fast_self_training(n_rounds=2, threshold=0.70, max_new=200)` — efficient self-training that only retrains transductive features + Base B (keeps Base A/D/E fixed)
- Leakage guard: OOF F1 always measured against ORIGINAL labels only

### Validation

```bash
cd bitoguard_core && PYTHONPATH=. python -c "
from official.self_training import expand_with_pseudo_labels
print('Self-training module imported OK')
"
```

---

## Task 2: Multi-scale PPR (ALREADY DONE in v45)

Multi-scale PPR with alpha=0.05 (long-range), 0.20 (medium), 0.50 (local) already implemented.
Negative seed propagation was tried in v30 and halved AP (0.1013→0.0509) — NOT re-adding.

---

## Task 3: Temporal co-occurrence edges (DONE in v47)

15-minute time bucket edges connecting users who transact in same window.
Edge types: `temporal_small` (2-5 users, weight=0.30), `temporal_medium` (6-30 users, weight=0.15).

---

## Task 4: Edge weight HPO via Optuna (DONE)

- `graph_dataset.build_transductive_graph(dataset, edge_weights=None)` now accepts custom weights dict
- `bitoguard_core/official/hpo_edge_weights.py` — Optuna HPO for 6 edge weights
  - Searches: relation [0.3, 2.0], wallet_small [0.1, 1.5], wallet_medium [0.05, 1.0], ip_small [0.1, 1.5], ip_medium [0.05, 0.8], temporal [0.05, 1.0]

### Validation

```bash
cd bitoguard_core && PYTHONPATH=. python -c "
from official.graph_dataset import build_transductive_graph
from official.hpo_edge_weights import DEFAULT_WEIGHTS
print(f'Default weights: {DEFAULT_WEIGHTS}')
print('PASS')
"
```

---

## Task 5: GraphSAGE 3-layer + hidden_dim=96 (DONE in v47)

- 3 aggregation layers (captures 3-hop neighborhood)
- hidden_dim: 64 → 96 (+50% capacity)
- patience: 8 → 12 (more training time)
- Both `train_graphsage_model` and `predict_graph_model` updated

---

## Task 6: Non-linear stacker

Add GBM stacker comparison. Currently `_fit_catboost_stacker` exists but is not used in default `use_blend=True` path. Evaluate CatBoost depth=3 stacker vs BlendEnsemble on each run.

---

## Task 7: Optuna threshold + calibration HPO

Create `official/hpo_threshold.py` — Optuna search over (calibration method, threshold) jointly optimizing bootstrap-mean F1.

---

## Task 8: Run full pipeline + measure

```bash
# Run pipeline with all improvements
cd bitoguard_core
BITOGUARD_AWS_EVENT_CLEAN_DIR=data/aws_event/clean PYTHONPATH=. python -m official.pipeline

# After pipeline completes, run self-training
BITOGUARD_AWS_EVENT_CLEAN_DIR=data/aws_event/clean PYTHONPATH=. python -m official.self_training

# Check results
PYTHONPATH=. python -c "
import json
from pathlib import Path
vr = Path('artifacts/reports/official_validation_report.json')
if vr.exists():
    r = json.loads(vr.read_text())
    print(json.dumps(r.get('primary_transductive_oof_metrics', {}), indent=2))
"
```

---

## Key facts

- Data: `bitoguard_core/data/aws_event/clean/*.parquet`
- Run from `bitoguard_core/` with `PYTHONPATH=.`
- Set `BITOGUARD_AWS_EVENT_CLEAN_DIR=data/aws_event/clean` when running pipeline
- Negative seeds: PERMANENTLY REMOVED (v30, halved AP from 0.1013→0.0509)
- GraphSAGE best epoch ≈ 15-25 with patience=12
