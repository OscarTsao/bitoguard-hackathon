# BitoGuard Repo Refactor Design

**Date**: 2026-03-16
**Status**: Approved
**Goal**: Make v2 stacker canonical, convert to proper Python package, fix P0 alert bugs, and clean documentation/scripts sprawl — optimizing for both demo readiness and long-term maintainability.

---

## Architecture Overview

### What Changes

```
bitoguard_core/                   →  proper installable Python package
├── models/train.py               →  DELETED (v1 LightGBM, OOM-killed at 2.55M rows)
├── models/validate.py            →  DELETED (fails with single snapshot date)
├── models/score.py               →  score_latest_snapshot_v2() becomes the only path
│   ├── score_latest_snapshot()   →  DELETED (v1, broken defaults)
│   ├── _build_model_version()    →  DELETED (dead code)
│   └── dormancy_score()          →  DELETED (never called)
├── models/stacker.py             →  import fixed (no longer imports private fn)
├── config.py                     →  m1_enabled=True, m3_enabled=True, m4_enabled=False
├── api/main.py                   →  calls score_latest_snapshot_v2()
├── pipeline/refresh_live.py      →  calls score_latest_snapshot_v2()
└── pyproject.toml                →  NEW — makes package installable

docs/ (65 files → 8 files)
scripts/ (23 files → ~10 files)
root-level status .md files       →  DELETED (6 generated artifacts)
```

### What Stays the Same

- All 85 tests pass (no behavior changes in tasks 1–2)
- DuckDB schema unchanged
- Frontend unchanged
- Makefile command names unchanged (`make train`, `make score`, etc.)
- 6-module conceptual architecture (M1–M6) unchanged
- `BITOGUARD_GRAPH_FEATURES_TRUSTED_ONLY` safety invariant preserved

### Post-Refactor Pipeline (canonical)

```
make sync      → pipeline/sync.py         (unchanged)
make features  → features/build_features.py (unchanged)
make train     → models/stacker.py        (was: models/train.py)
make score     → models/score.py          (score_latest_snapshot_v2 only)
make drift     → services/drift.py        (unchanged)
```

---

## Task 1: P0 Bug Fixes

**Why first**: System currently generates zero alerts in production. M1 and M3 are off by default; M4 crashes on schema mismatch. This must ship before anything else.

### Changes

**`bitoguard_core/config.py`** — flip three module defaults:
```python
# Before
m1_enabled: bool = False
m3_enabled: bool = False
m4_enabled: bool = True

# After
m1_enabled: bool = True
m3_enabled: bool = True
m4_enabled: bool = False  # explicitly disabled until retrained on v2 schema
```

**`bitoguard_core/models/score.py`** — recalibrate alert thresholds:
```python
# Before (unreachable — max risk_score ≈ 57 with M1+M3)
bins = [-1, 35, 60, 80, 100]

# After
bins = [-1, 20, 50, 70, 100]

# Risk score ceiling (do not revert thresholds):
# With M1+M3 weights [0.35, 0.45] normalized, max risk_score ≈ 57.
# Thresholds must be below 57 to produce any alerts.
```

**Verification**: `make score` → query `SELECT COUNT(*) FROM ops.alerts WHERE risk_level != 'low'` returns > 0.

---

## Task 2: Python Packaging

**Why**: PYTHONPATH=. convention breaks IDE imports, makes Dockerfiles fragile, and signals "not production-ready."

### New Files

**`bitoguard_core/pyproject.toml`**:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "bitoguard"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []

[tool.setuptools.packages.find]
where = ["."]
include = ["bitoguard*", "models*", "features*", "services*",
           "pipeline*", "api*", "db*"]
```

**Empty `__init__.py`** added to:
- `bitoguard_core/__init__.py`
- `bitoguard_core/models/__init__.py`
- `bitoguard_core/features/__init__.py`
- `bitoguard_core/services/__init__.py`
- `bitoguard_core/pipeline/__init__.py`
- `bitoguard_core/api/__init__.py`
- `bitoguard_core/db/__init__.py`

### Modified Files

**`Makefile`** — add `pip install -e bitoguard_core/` to `make setup`, remove `PYTHONPATH=.` prefixes from all targets.

**`bitoguard_core/Dockerfile.processing`** and **`bitoguard_core/Dockerfile.training`** — add `RUN pip install -e .` after `COPY`.

**Import statements**: No changes needed — `from models.xxx import` works once installed with `pip install -e .` from within `bitoguard_core/`.

---

## Task 3: Delete v1, Wire v2

**Why**: v1 OOM-kills at 2.55M rows. v2 stacker (OOF AUC 0.9495) is demonstrably better and must be the path called by the API.

### Deleted Files
- `bitoguard_core/models/train.py`
- `bitoguard_core/models/validate.py`

### Modified Files

**`bitoguard_core/models/stacker.py`** — fix private import coupling:
```python
# Before
from models.train_catboost import _load_v2_training_dataset, _CAT_FEATURE_NAMES

# After
from models.common import load_v2_training_dataset, CAT_FEATURE_NAMES
```
(Move `_load_v2_training_dataset` → `load_v2_training_dataset` and `_CAT_FEATURE_NAMES` → `CAT_FEATURE_NAMES` into `models/common.py`)

**`bitoguard_core/models/score.py`** — remove v1 path and dead code:
- Delete `score_latest_snapshot()` function
- Delete `_build_model_version()` function
- Delete `dormancy_score()` function
- Rename `score_latest_snapshot_v2` → `score_latest_snapshot` for clean API

**`bitoguard_core/api/main.py`** — update call site:
```python
# Before
result = score_latest_snapshot()

# After
result = score_latest_snapshot()  # now points to v2 implementation
```

**`bitoguard_core/pipeline/refresh_live.py`** — same call-site update.

**`Makefile`**:
```makefile
train: cd bitoguard_core && python -m models.stacker
```

---

## Task 4: Scripts Cleanup

### Deleted Scripts (redundant/empty/overlapping)
- `scripts/launch-5fold-scriptmode.sh` — empty (1 line)
- `scripts/deploy-and-launch-5fold.sh` — duplicate
- `scripts/deploy-and-train.sh` — duplicate
- `scripts/deploy-infrastructure-first.sh` — subset of deploy-ml-pipeline
- `scripts/deploy-sagemaker-features.sh` — overlaps deploy-sagemaker-only
- `scripts/deploy-sagemaker-only.sh` — overlaps deploy-ml-pipeline
- `scripts/quick-deploy-and-run.sh` — duplicate shortcut
- `scripts/local-train-5fold.sh` — redundant with `make train`
- `scripts/launch-5fold-training.sh` — redundant with `make train`

### Deleted Root-Level Status Docs (generated artifacts)
- `DEPLOYMENT_CHECKLIST.md`
- `DEPLOYMENT_READY.md`
- `SAGEMAKER_DEPLOYMENT_READY.md`
- `SAGEMAKER_READY.md`
- `STACKER_IMPLEMENTATION_COMPLETE.md`
- `WORKSHOP_DEPLOYMENT.md`

### Kept Scripts (~10 with distinct purpose)
- `scripts/deploy-ml-pipeline.sh` — canonical full deployment
- `scripts/launch-sagemaker-direct.sh` — direct SageMaker (restricted env)
- `scripts/launch-5fold-sagemaker.sh` — SageMaker 5-fold
- `scripts/test-api.sh` — API smoke testing
- Remaining scripts with unique, non-overlapping purpose

---

## Task 5: Docs Cleanup

### Kept (8 files)
| File | Reason |
|------|--------|
| `README.md` | Root entry point — updated for v2 pipeline |
| `CLAUDE.md` | AI assistant instructions — updated |
| `docs/GRAPH_TRUST_BOUNDARY.md` | Critical: device placeholder bug warning |
| `docs/GRAPH_RECOVERY_PLAN.md` | Future work roadmap |
| `docs/ML_PIPELINE_SUMMARY.md` | Best existing pipeline overview |
| `docs/SAGEMAKER_DEPLOYMENT_GUIDE.md` | Single canonical deployment doc |
| `docs/superpowers/specs/` | Design specs (this directory) |
| `postman_collection.json` | API testing (kept at root) |

### Deleted
All other files in `docs/` — 20+ AWS/SageMaker overlapping guides, implementation summaries, quick-start variants, redundant checklists.

`README.md` and `CLAUDE.md` updated to reflect:
- v2 stacker as canonical training path
- `pip install -e bitoguard_core/` in setup instructions
- Corrected module default states (M1+M3 on, M4 off)

---

## Task 6: Test Coverage

### New Test: Alert Integration Guard
**`bitoguard_core/tests/test_smoke.py`** — add:
```python
def test_alerts_generated_after_scoring():
    """Guards against threshold miscalibration causing zero alerts."""
    store = DuckDBStore(read_only=True)
    count = store.fetchone("SELECT COUNT(*) FROM ops.alerts WHERE risk_level != 'low'")[0]
    assert count > 0, (
        f"Zero non-low alerts detected. "
        f"Check: (1) m1_enabled/m3_enabled in config, "
        f"(2) alert threshold bins in score.py"
    )
```

### New Test: M4 Schema Guard
**`bitoguard_core/tests/test_model_pipeline.py`** — add:
```python
def test_iforest_schema_matches_v2_features_if_loaded():
    """If an IsolationForest model exists, its encoded_columns must match
    the current v2 feature schema. Fails loudly instead of silently zeroing."""
    from models.common import model_dir, feature_columns
    iforest_metas = list(model_dir().glob("iforest_*.json"))
    if not iforest_metas:
        pytest.skip("No IsolationForest model found — skip schema check")
    meta = json.loads(iforest_metas[-1].read_text())
    expected = set(feature_columns())
    actual = set(meta.get("encoded_columns", []))
    missing = expected - actual
    extra = actual - expected
    assert not missing and not extra, (
        f"IsolationForest schema mismatch. "
        f"Missing: {missing}, Extra: {extra}. "
        f"Retrain with: make train-iforest"
    )
```

---

## Iteration Path (Post-Refactor)

This refactor establishes the foundation. The next iteration cycle adds:

1. **Branch C** (graph propagation features in stacker) — single file change in `stacker.py`
2. **M4 reactivation** — retrain IsolationForest on v2 features, negatives-only
3. **Analyst feedback loop** — wire `ops.case_actions` to retraining trigger
4. **Prediction stability monitoring** — score distribution percentiles logged per run

Each is a self-contained addition, not surgery.
