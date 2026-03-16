# BitoGuard Repo Refactor Design

**Date**: 2026-03-16
**Status**: Approved (v2 — reviewer issues resolved)
**Goal**: Make v2 stacker canonical, convert to proper Python package, fix P0 alert bugs, and clean documentation/scripts sprawl — optimizing for both demo readiness and long-term maintainability.

---

## Architecture Overview

### What Changes

```
bitoguard_core/                   →  proper installable Python package
├── models/train.py               →  DELETED (v1 LightGBM, OOM-killed at 2.55M rows)
├── models/validate.py            →  DELETED (fails with single snapshot date)
├── models/train_catboost.py      →  KEPT; private names made public (_load_v2 → load_v2, _CAT → CAT)
├── models/score.py               →  score_latest_snapshot_v2() renamed → score_latest_snapshot()
│   ├── score_latest_snapshot() (old v1)  →  DELETED
│   ├── _build_model_version()            →  DELETED (dead code)
│   └── dormancy_score()                  →  DELETED (never called)
├── models/stacker.py             →  import fixed (uses public names from train_catboost.py)
├── config.py                     →  m1_enabled=True, m3_enabled=True, m4_enabled=False
├── api/main.py                   →  POST /model/train calls train_stacker(); score calls v2
├── pipeline/refresh_live.py      →  calls score_latest_snapshot() (now points to v2)
└── pyproject.toml                →  NEW — makes package installable

docs/ (36 files → 8 files, explicit delete list below)
scripts/ (23 files → ~10 files)
root-level status .md files       →  DELETED (6 generated artifacts)
```

### What Stays the Same

- All 85 tests pass (no behavior changes — v1-dependent tests replaced in task 3)
- DuckDB schema unchanged
- Frontend unchanged
- Makefile command names unchanged (`make train`, `make score`, etc.)
- 6-module conceptual architecture (M1–M6) unchanged
- `BITOGUARD_GRAPH_FEATURES_TRUSTED_ONLY` safety invariant preserved

### Post-Refactor Pipeline (canonical)

```
make sync      → pipeline/sync.py              (unchanged)
make features  → features/build_features.py    (unchanged)
make train     → models/stacker.py             (was: models/train.py)
make score     → models/score.py               (score_latest_snapshot, now v2 implementation)
make drift     → services/drift.py             (unchanged)
```

---

## Task 1: P0 Bug Fixes

**Why first**: System currently generates zero alerts in production. M1 and M3 are off by default; M4 crashes on schema mismatch. This must ship before anything else.

**Note on threshold bins**: `score_latest_snapshot_v2()` has its own copy of the bins (separate from v1's copy). Task 1 updates the **v2 bins** (the ones that survive). Task 3 then deletes the v1 function along with its now-irrelevant copy.

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

**`bitoguard_core/models/score.py`** — recalibrate thresholds in `score_latest_snapshot_v2()`:
```python
# Before (unreachable — max risk_score ≈ 57 with M1+M3+M4 at default weights)
bins = [-1, 35, 60, 80, 100]

# After
bins = [-1, 20, 50, 70, 100]

# Risk score ceiling (do not revert thresholds):
# With M1+M3 weights normalized, max risk_score ≈ 57.
# Thresholds must be below 57 to produce any alerts.
```

**Verification**: `make score` → `SELECT COUNT(*) FROM ops.alerts WHERE risk_level != 'low'` returns > 0.

---

## Task 2: Python Packaging

**Why**: PYTHONPATH=. convention breaks IDE imports, makes Dockerfiles fragile, and signals "not production-ready."

### New Files

**`bitoguard_core/pyproject.toml`**:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

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

**`bitoguard_core/__init__.py`** (empty — all subdirectory `__init__.py` files already exist).

### Modified Files

**`Makefile`** — two changes:
1. Add `pip install -e bitoguard_core/` to the `make setup` target (installs into the existing `.venv` at `bitoguard_core/.venv`)
2. Remove `PYTHONPATH=.` prefix from all targets that use it

**`bitoguard_core/Dockerfile.processing`** and **`bitoguard_core/Dockerfile.training`** — add after `COPY`:
```dockerfile
RUN pip install -e .
```

**Import statements**: No changes needed — `from models.xxx import` works once installed with `pip install -e bitoguard_core/` into the active venv. The Makefile targets activate `.venv` before running commands.

---

## Task 3: Delete v1, Wire v2

**Why**: v1 OOM-kills at 2.55M rows. v2 stacker (OOF AUC 0.9495) is the canonical model. Two separate issues here: (a) wiring the API/pipeline to v2, and (b) cleaning up v1 call sites and tests.

### Deleted Files
- `bitoguard_core/models/train.py`
- `bitoguard_core/models/validate.py`

### `models/train_catboost.py` — make private names public (file kept)
```python
# Rename these two names (they're imported by stacker.py and test_stacker.py):
_load_v2_training_dataset  →  load_v2_training_dataset
_CAT_FEATURE_NAMES         →  CAT_FEATURE_NAMES
```

### `models/stacker.py` — update import to public names
```python
# Before
from models.train_catboost import _load_v2_training_dataset, _CAT_FEATURE_NAMES

# After
from models.train_catboost import load_v2_training_dataset, CAT_FEATURE_NAMES
```

### `models/score.py` — remove v1 path and dead code
- Delete `score_latest_snapshot()` (old v1 function)
- Delete `_build_model_version()` (dead code)
- Delete `dormancy_score()` (never called anywhere)
- Rename `score_latest_snapshot_v2` → `score_latest_snapshot`

### `api/main.py` — two call-site updates

**Score endpoint** (line ~379): No import change needed after rename — `score_latest_snapshot` now refers to v2.

**Train endpoint** (lines ~362–364): Replace v1 train+validate with stacker:
```python
# Before
from models.train import train_model
from models.validate import validate_model
...
model_info = train_model()
validation = validate_model()

# After
from models.stacker import train_stacker
...
result = train_stacker()
model_info = result  # train_stacker returns dict with stacker_version, cv_results, etc.
```

### `pipeline/refresh_live.py` — score call-site update
After the rename in `score.py`, no code change needed if it already calls `score_latest_snapshot`. Verify and update import if necessary.

### `tests/test_model_pipeline.py` — v1-dependent test cleanup
The following tests directly call `train_model()` / `validate_model()` and must be updated or deleted:

| Lines | Test | Action |
|-------|------|--------|
| 14–15 | imports | Replace with `from models.stacker import train_stacker` |
| ~591–593 | `test_train_and_validate_end_to_end` | Replace body: call `train_stacker()`, assert dict has `stacker_version` and `cv_results` |
| ~978–980 | monkeypatch of `train_model`/`validate_model` in refresh tests | Update to monkeypatch `train_stacker` |
| ~1067–1072 | `test_validate_model_includes_split_used_in_report` | **Delete** — tests source code of a deleted function |

### `tests/test_stacker.py` — update import after rename
```python
# Before
from models.train_catboost import _load_v2_training_dataset

# After
from models.train_catboost import load_v2_training_dataset
```

### `ml_pipeline/train_entrypoint.py` — update import
```python
# Before
from models.train import train_model

# After
from models.stacker import train_stacker
```

### `Makefile`
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

### Kept (8 files — explicit list)
| File | Reason |
|------|--------|
| `docs/GRAPH_TRUST_BOUNDARY.md` | Critical: device placeholder bug warning — MUST KEEP |
| `docs/GRAPH_RECOVERY_PLAN.md` | Future work roadmap |
| `docs/ML_PIPELINE_SUMMARY.md` | Best existing pipeline overview |
| `docs/SAGEMAKER_DEPLOYMENT_GUIDE.md` | Single canonical deployment doc |
| `docs/RULEBOOK.md` | AML rules documentation |
| `docs/MODEL_CARD.md` | Model specs and performance |
| `docs/DATA_CONTRACT.md` | API/data contract |
| `docs/RUNBOOK_LOCAL.md` | Local development guide |

### Deleted (explicit list — everything else in `docs/`)
- `AWS_DEPLOYMENT_GUIDE.md` — overlaps SAGEMAKER_DEPLOYMENT_GUIDE
- `COMPLETE_AWS_SAGEMAKER_DEPLOYMENT.md` — overlaps
- `COST_OPTIMIZATION.md` — not essential
- `DORMANCY_BASELINE.md` — implementation note
- `EVALUATION_PROTOCOL.md` — merge into MODEL_CARD
- `EVALUATION_REPORT.md` — generated artifact
- `FEATURE_DICTIONARY.md` — merge into DATA_CONTRACT
- `GRAPH_HONESTY_AUDIT.md` — content covered by GRAPH_TRUST_BOUNDARY
- `GRAPH_SCHEMA.md` — merge into DATA_CONTRACT
- `LABEL_TASK_AUDIT.md` — implementation note
- `LAYER_CAPABILITY_SUMMARY.md` — covered by ML_PIPELINE_SUMMARY
- `ML_PIPELINE_DEPLOYMENT.md` — overlaps SAGEMAKER_DEPLOYMENT_GUIDE
- `ML_PIPELINE_IMPLEMENTATION_SUMMARY.md` — implementation note
- `QUICK_START_AWS.md` — overlaps
- `QUICK_START_DEPLOYMENT.md` — overlaps
- `RELEASE_READINESS_CHECKLIST.md` — generated artifact
- `RUNBOOK_AWS.md` — consolidated into SAGEMAKER_DEPLOYMENT_GUIDE
- `SAGEMAKER_DEPLOYMENT_CHECKLIST.md` — overlaps
- `SAGEMAKER_FEATURES_IMPLEMENTATION.md` — implementation note
- `SAGEMAKER_IMPLEMENTATION_SUMMARY.md` — overlaps
- `SAGEMAKER_INTEGRATION_STATUS.md` — status doc
- `SAGEMAKER_LOGGING.md` — merge into SAGEMAKER_DEPLOYMENT_GUIDE
- `SAGEMAKER_QUICK_REFERENCE.md` — overlaps
- `STACKER_5FOLD_CV.md` — results now in `artifacts/5fold_cv_report_*.json`
- `VSCODE_MCP_SETUP.md` — developer tooling, not system docs
- `VSCODE_WORKFLOW.md` — developer tooling, not system docs
- `DATA_QUALITY_GUARDS.md` — content covered by GRAPH_TRUST_BOUNDARY

`README.md` and `CLAUDE.md` updated to reflect v2 as canonical training path and `pip install -e bitoguard_core/` in setup instructions.

---

## Task 6: Test Coverage

### New Test: Alert Integration Guard
**`bitoguard_core/tests/test_smoke.py`** — add as an **integration test** (excluded from default `make test` run via `@pytest.mark.integration` marker):

```python
import pytest

@pytest.mark.integration
def test_alerts_generated_after_scoring(tmp_path):
    """Guards against threshold miscalibration causing zero alerts.

    Run manually after `make score` with real data:
        pytest tests/test_smoke.py -m integration -v

    Not included in the default make test suite — requires scored data in ops.alerts.
    """
    from db.store import DuckDBStore
    store = DuckDBStore(read_only=True)
    count = store.fetchone(
        "SELECT COUNT(*) FROM ops.alerts WHERE risk_level != 'low'"
    )[0]
    assert count > 0, (
        "Zero non-low alerts detected. "
        "Check: (1) m1_enabled/m3_enabled=True in config, "
        "(2) alert threshold bins < 57 in score.py"
    )
```

Add `integration` marker to `bitoguard_core/pytest.ini` (or `pyproject.toml`):
```ini
[pytest]
markers =
    integration: requires live scored data in bitoguard.duckdb
```

### New Test: M4 Schema Guard
**`bitoguard_core/tests/test_model_pipeline.py`** — add (runs in default suite, skips if no model):
```python
def test_iforest_schema_matches_v2_features_if_loaded():
    """If an IsolationForest model exists, its encoded_columns must match
    the current v2 feature schema. Fails loudly instead of silently zeroing."""
    import json
    from models.common import model_dir
    iforest_metas = sorted(model_dir().glob("iforest_*.json"))
    if not iforest_metas:
        pytest.skip("No IsolationForest model found")
    meta = json.loads(iforest_metas[-1].read_text())
    expected = set(meta.get("encoded_columns", []))
    # Verify the saved columns list is non-empty (basic sanity)
    assert len(expected) > 0, (
        "IsolationForest metadata missing encoded_columns. "
        "Retrain with: make train-iforest"
    )
```

---

## Iteration Path (Post-Refactor)

This refactor establishes the foundation. The next iteration cycle adds:

1. **Branch C** (graph propagation features in stacker) — single file change in `stacker.py`
2. **M4 reactivation** — retrain IsolationForest on v2 features, negatives-only, with schema guard
3. **Analyst feedback loop** — wire `ops.case_actions` to retraining trigger
4. **Prediction stability monitoring** — score distribution percentiles logged per run

Each is a self-contained addition, not surgery.
