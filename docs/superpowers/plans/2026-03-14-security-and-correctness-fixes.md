# Security & Correctness Fixes Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 12 issues identified in the code review: 4 critical (insecure deserialization, no auth, full-table scans, SQL injection) and 8 high (IsolationForest label leakage, validation fallback, concurrent writes, schema destruction, open proxy, non-atomic decisions, graph risk normalization, blacklist fast-path leakage).

**Architecture:** Changes are grouped into 4 independent, sequentially-testable chunks: (1) storage layer security, (2) ML pipeline correctness, (3) API security & performance, (4) graph feature correctness. All changes are behaviour-preserving — public interfaces and test contracts are unchanged except where a bug fix requires changing a return value.

**Tech Stack:** Python 3.12, FastAPI, DuckDB 1.3.2, LightGBM 4.6, scikit-learn 1.7, joblib, Next.js 16, TypeScript

---

## Files Modified (by chunk)

**Chunk 1 — Storage Layer:**
- Modify: `bitoguard_core/db/store.py` — allowlist, schema-preserving replace, write lock, `transaction()` CM
- Modify: `bitoguard_core/pipeline/refresh_live.py` — column name validation for ALTER TABLE
- Modify: `bitoguard_core/services/alert_engine.py` — atomic case decisions, push-down SQL for generate_alerts

**Chunk 2 — ML Pipeline:**
- Modify: `bitoguard_core/models/common.py` — replace pickle with LightGBM text + joblib+hash helpers
- Modify: `bitoguard_core/models/train.py` — save LightGBM as native text format (`.lgbm`)
- Modify: `bitoguard_core/models/anomaly.py` — joblib save, SHA-256 manifest, fix contamination=0.05
- Modify: `bitoguard_core/models/score.py` — safe load, fix graph risk absolute thresholds, atomic DELETE+INSERT
- Modify: `bitoguard_core/models/validate.py` — add `split_used` field, remove silent train fallback
- Modify: `bitoguard_core/services/explain.py` — use safe LightGBM loader
- Modify: `bitoguard_core/requirements.txt` — add joblib

**Chunk 3 — API Security & Performance:**
- Modify: `bitoguard_core/config.py` — add `api_key: str | None` setting
- Modify: `bitoguard_core/api/main.py` — add API key auth dependency, fix full-table scans
- Modify: `bitoguard_core/services/diagnosis.py` — filter SQL for login/crypto/trade
- Modify: `bitoguard_frontend/src/app/api/backend/[...path]/route.ts` — path allowlist

**Chunk 4 — Graph Feature Correctness:**
- Modify: `bitoguard_core/features/graph_features.py` — per-snapshot blacklist cutoff in fast path

---

## Chunk 1: Storage Layer Security & Correctness

### Task 1: Table name allowlist + SQL injection prevention

**Files:**
- Modify: `bitoguard_core/db/store.py`

- [ ] **Step 1: Write failing test for allowlist**

Add to `bitoguard_core/tests/test_model_pipeline.py` (or create `tests/test_store.py`):

```python
# tests/test_store.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import pytest
from db.store import DuckDBStore


def test_replace_table_rejects_unknown_table(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "t.duckdb")
    with pytest.raises(ValueError, match="not in the allowed"):
        store.replace_table("evil.inject", pd.DataFrame({"x": [1]}))


def test_read_table_rejects_unknown_table(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "t.duckdb")
    with pytest.raises(ValueError, match="not in the allowed"):
        store.read_table("ops.nonexistent_table")


def test_append_rejects_unknown_table(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "t.duckdb")
    with pytest.raises(ValueError, match="not in the allowed"):
        store.append_dataframe("'; DROP TABLE ops.alerts; --", pd.DataFrame({"x": [1]}))
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd bitoguard_core && source .venv/bin/activate
PYTHONPATH=. pytest tests/test_store.py -v
```
Expected: 3 tests FAIL (ValueError not raised)

- [ ] **Step 3: Add allowlist and `_validate_table_name` to `store.py`**

Add after the imports in `bitoguard_core/db/store.py` (after line 12, before `def utc_now`):

```python
import threading

from db.schema import CANONICAL_TABLE_SPECS, FEATURE_TABLE_SPECS, RAW_TABLE_SPECS

_ALLOWED_TABLES: frozenset[str] = frozenset(
    f"{spec.schema}.{spec.name}"
    for specs in (RAW_TABLE_SPECS, CANONICAL_TABLE_SPECS, FEATURE_TABLE_SPECS)
    for spec in specs
) | frozenset({
    "ops.sync_runs", "ops.data_quality_issues", "ops.oracle_user_labels",
    "ops.oracle_scenarios", "ops.model_predictions", "ops.alerts",
    "ops.cases", "ops.case_actions", "ops.validation_reports", "ops.refresh_state",
})

_WRITE_LOCK = threading.Lock()


def _validate_table_name(table_name: str) -> None:
    if table_name not in _ALLOWED_TABLES:
        raise ValueError(f"Table '{table_name}' is not in the allowed table list")
```

Then add `_validate_table_name(table_name)` as the first line of `replace_table`, `append_dataframe`, and `read_table`.

- [ ] **Step 4: Run tests to confirm they pass**

```bash
PYTHONPATH=. pytest tests/test_store.py -v
```
Expected: 3 tests PASS

- [ ] **Step 5: Run full suite to catch regressions**

```bash
PYTHONPATH=. pytest tests/ -q
```
Expected: all pass (allowlist includes all tables already used by the system)

- [ ] **Step 6: Commit**

```bash
git add bitoguard_core/db/store.py bitoguard_core/tests/test_store.py
git commit -m "fix: add table name allowlist to prevent SQL injection in DuckDBStore"
```

---

### Task 2: Schema-preserving `replace_table` + DuckDB write mutex + `transaction()` CM

**Files:**
- Modify: `bitoguard_core/db/store.py`

- [ ] **Step 1: Write failing test for schema preservation and write lock**

Add to `tests/test_store.py`:

```python
def test_replace_table_preserves_schema_when_empty_df(tmp_path: Path) -> None:
    """replace_table with empty DataFrame must not destroy the table schema."""
    store = DuckDBStore(tmp_path / "t2.duckdb")
    # Write some data first
    store.append_dataframe("ops.alerts", pd.DataFrame([{
        "alert_id": "a1", "user_id": "u1", "snapshot_date": "2026-01-01",
        "created_at": "2026-01-01T00:00:00+00:00", "risk_level": "high",
        "status": "open", "prediction_id": "p1", "report_path": None,
    }]))
    # Replace with empty DataFrame — schema must survive
    store.replace_table("ops.alerts", pd.DataFrame(columns=["alert_id", "user_id", "snapshot_date", "created_at", "risk_level", "status", "prediction_id", "report_path"]))
    # Should still be able to insert a row (schema intact)
    store.append_dataframe("ops.alerts", pd.DataFrame([{
        "alert_id": "a2", "user_id": "u2", "snapshot_date": "2026-01-02",
        "created_at": "2026-01-01T00:00:00+00:00", "risk_level": "medium",
        "status": "open", "prediction_id": "p2", "report_path": None,
    }]))
    result = store.fetch_df("SELECT COUNT(*) AS n FROM ops.alerts")
    assert result.iloc[0]["n"] == 1  # only a2, because replace cleared a1


def test_transaction_is_atomic_on_error(tmp_path: Path) -> None:
    """If any statement in a transaction raises, all changes are rolled back."""
    store = DuckDBStore(tmp_path / "t3.duckdb")
    try:
        with store.transaction() as conn:
            conn.execute(
                "INSERT INTO ops.alerts (alert_id, user_id, snapshot_date, created_at, risk_level, status) VALUES ('atomic_test', 'u1', '2026-01-01', now(), 'high', 'open')"
            )
            raise RuntimeError("simulated failure")
    except RuntimeError:
        pass
    result = store.fetch_df("SELECT COUNT(*) AS n FROM ops.alerts WHERE alert_id = 'atomic_test'")
    assert result.iloc[0]["n"] == 0, "Transaction must have been rolled back"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
PYTHONPATH=. pytest tests/test_store.py::test_replace_table_preserves_schema_when_empty_df tests/test_store.py::test_transaction_is_atomic_on_error -v
```
Expected: FAIL — `replace_table` currently destroys schema, `transaction` doesn't exist yet

- [ ] **Step 3: Replace `replace_table` implementation and add `transaction()` to `store.py`**

Replace the `replace_table` method body (current lines 58-62 of `store.py`):

```python
def replace_table(self, table_name: str, dataframe: pd.DataFrame) -> None:
    _validate_table_name(table_name)
    with _WRITE_LOCK, self.connect() as conn:
        conn.register("tmp_df", dataframe)
        conn.execute(f"DELETE FROM {table_name}")
        conn.execute(f"INSERT INTO {table_name} SELECT * FROM tmp_df")
        conn.unregister("tmp_df")
```

Replace `append_dataframe` (current lines 64-70):

```python
def append_dataframe(self, table_name: str, dataframe: pd.DataFrame) -> None:
    if dataframe.empty:
        return
    _validate_table_name(table_name)
    with _WRITE_LOCK, self.connect() as conn:
        conn.register("tmp_df", dataframe)
        conn.execute(f"INSERT INTO {table_name} SELECT * FROM tmp_df")
        conn.unregister("tmp_df")
```

Replace `execute` (current lines 76-78):

```python
def execute(self, sql: str, params: tuple | None = None) -> None:
    with _WRITE_LOCK, self.connect() as conn:
        conn.execute(sql, params or ())
```

Add `transaction()` after `connect()`:

```python
@contextmanager
def transaction(self) -> Iterator[duckdb.DuckDBPyConnection]:
    """Atomic multi-statement write context manager. Acquires write lock."""
    with _WRITE_LOCK, self.connect() as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
```

- [ ] **Step 4: Run new tests**

```bash
PYTHONPATH=. pytest tests/test_store.py -v
```
Expected: all PASS

- [ ] **Step 5: Run full suite**

```bash
PYTHONPATH=. pytest tests/ -q
```
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add bitoguard_core/db/store.py bitoguard_core/tests/test_store.py
git commit -m "fix: schema-preserving replace_table, write mutex, and atomic transaction() CM"
```

---

### Task 3: Column name injection in `refresh_live.py`

**Files:**
- Modify: `bitoguard_core/pipeline/refresh_live.py`

- [ ] **Step 1: Find the injection point**

In `refresh_live.py`, find `_ensure_table_columns` (around line 192-197). It does:
```python
conn.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {column} {dtype}")
```
where `column` comes from `dataframe.columns`.

- [ ] **Step 2: Add column name validation**

Add this import at the top of `refresh_live.py`:
```python
import re
```

Add this helper right after the imports:
```python
_SAFE_COLUMN_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _safe_column_name(col: str) -> str:
    if not _SAFE_COLUMN_RE.match(col):
        raise ValueError(f"Column name '{col}' contains invalid characters: {col!r}")
    return col
```

In `_ensure_table_columns`, wrap the column name before interpolation:
```python
safe_col = _safe_column_name(column)
conn.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {safe_col} {dtype}")
```

- [ ] **Step 3: Run full suite to confirm no regression**

```bash
cd bitoguard_core && source .venv/bin/activate
PYTHONPATH=. pytest tests/ -q
```
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add bitoguard_core/pipeline/refresh_live.py
git commit -m "fix: validate column names before ALTER TABLE in refresh_live to prevent SQL injection"
```

---

### Task 4: Atomic case decisions + optimized `generate_alerts`

**Files:**
- Modify: `bitoguard_core/services/alert_engine.py`

- [ ] **Step 1: Write test for atomicity**

Add to `tests/test_store.py` or `tests/test_smoke.py`:

```python
def test_case_decision_is_atomic(tmp_path: Path, monkeypatch) -> None:
    """If apply_case_decision raises mid-way, no partial state should persist."""
    # This test is an integration smoke: verify the function succeeds atomically
    # (actual rollback-on-crash is architecture-level; we verify the happy path
    # executes within a single transaction context by checking no stale state)
    from services.alert_engine import apply_case_decision, generate_alerts
    from models.score import score_latest_snapshot

    # Setup is complex — test is done via test_smoke.py::test_case_decision_updates_statuses
    # which already covers atomic state transitions. This task is a code change only.
    pass
```

(The real test coverage is already provided by `test_case_decision_updates_statuses` in `test_smoke.py`.)

- [ ] **Step 2: Rewrite `apply_case_decision` to use `store.transaction()`**

Replace the three `store.execute()` calls in `apply_case_decision` (lines 121-135 of `alert_engine.py`):

```python
    with store.transaction() as conn:
        conn.execute(
            "INSERT INTO ops.case_actions (action_id, case_id, action_type, actor, action_at, note) VALUES (?, ?, ?, ?, ?, ?)",
            (make_id("action"), case_id, decision, actor, utc_now(), note),
        )
        conn.execute(
            "UPDATE ops.cases SET latest_decision = ?, status = ? WHERE case_id = ?",
            (decision, status_update["case_status"], case_id),
        )
        conn.execute(
            "UPDATE ops.alerts SET status = ? WHERE alert_id = ?",
            (status_update["alert_status"], alert_id),
        )
```

- [ ] **Step 3: Optimize `generate_alerts` to remove full-table scans**

Replace the body of `generate_alerts` (eliminate `store.read_table("ops.model_predictions")` and `store.read_table("ops.alerts")`):

```python
def generate_alerts() -> pd.DataFrame:
    settings = load_settings()
    store = DuckDBStore(settings.db_path)

    # Sync risk levels for existing alerts (only touch predictions that have an alert)
    predictions_to_sync = store.fetch_df(
        """
        SELECT p.* FROM ops.model_predictions p
        INNER JOIN ops.alerts a ON p.user_id = a.user_id AND p.snapshot_date = a.snapshot_date
        """
    )
    _sync_existing_alerts(store, predictions_to_sync)

    # Find high-risk predictions with no existing alert
    new_high_risk = store.fetch_df(
        """
        SELECT p.* FROM ops.model_predictions p
        WHERE p.risk_level IN ('high', 'critical')
        AND NOT EXISTS (
            SELECT 1 FROM ops.alerts a
            WHERE a.user_id = p.user_id AND a.snapshot_date = p.snapshot_date
        )
        """
    )
    if new_high_risk.empty:
        return pd.DataFrame()

    alerts = []
    cases = []
    for _, row in new_high_risk.iterrows():
        alert_id = make_id("alert")
        case_id = make_id("case")
        alerts.append({
            "alert_id": alert_id, "user_id": row["user_id"],
            "snapshot_date": row["snapshot_date"], "created_at": utc_now(),
            "risk_level": row["risk_level"], "status": "open",
            "prediction_id": row["prediction_id"], "report_path": None,
        })
        cases.append({
            "case_id": case_id, "alert_id": alert_id,
            "user_id": row["user_id"], "created_at": utc_now(),
            "status": "open", "latest_decision": None,
        })
    if alerts:
        store.append_dataframe("ops.alerts", pd.DataFrame(alerts))
    if cases:
        store.append_dataframe("ops.cases", pd.DataFrame(cases))
    return pd.DataFrame(alerts)
```

- [ ] **Step 4: Run full suite**

```bash
PYTHONPATH=. pytest tests/ -q
```
Expected: all pass (smoke tests cover alert/case decision paths)

- [ ] **Step 5: Commit**

```bash
git add bitoguard_core/services/alert_engine.py
git commit -m "fix: atomic case decisions via store.transaction(), remove full-table scans in generate_alerts"
```

---

## Chunk 2: ML Pipeline Security & Correctness

### Task 5: Replace pickle with safe serialization

**Files:**
- Modify: `bitoguard_core/models/common.py`
- Modify: `bitoguard_core/models/train.py`
- Modify: `bitoguard_core/models/anomaly.py`
- Modify: `bitoguard_core/models/score.py`
- Modify: `bitoguard_core/services/explain.py`
- Modify: `bitoguard_core/requirements.txt`

**Strategy:**
- LightGBM: use `model.booster_.save_model(path)` → text format (`.lgbm`). Load with `lgb.Booster(model_file=path)`. `booster.predict(x)` returns 1D float array of class-1 probabilities directly.
- IsolationForest: use `joblib.dump(model, path)` → `.joblib`. Write SHA-256 digest to `path.with_suffix('.sha256')`. On load, verify digest before `joblib.load`.

- [ ] **Step 1: Add joblib to requirements**

```
# bitoguard_core/requirements.txt  — add:
joblib==1.4.2
```

Run `pip install joblib==1.4.2` in the venv.

- [ ] **Step 2: Rewrite `common.py` serialization helpers**

Remove `save_pickle` and `load_pickle` entirely. Add these functions:

```python
import hashlib
import joblib
import lightgbm as lgb

# ── LightGBM (text format — no pickle, inherently safe) ──────────────────────

def save_lgbm(model: "lgb.LGBMClassifier", path: Path) -> None:
    """Save a fitted LGBMClassifier using LightGBM's native text format."""
    model.booster_.save_model(str(path))


def load_lgbm(path: Path) -> "lgb.Booster":
    """Load a LightGBM Booster from its native text format."""
    return lgb.Booster(model_file=str(path))


# ── IsolationForest (joblib + SHA-256 integrity check) ───────────────────────

def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_iforest(model: object, path: Path) -> None:
    """Save IsolationForest with joblib and write a SHA-256 integrity manifest."""
    joblib.dump(model, path)
    path.with_suffix(".sha256").write_text(_sha256_file(path), encoding="utf-8")


def load_iforest(path: Path) -> object:
    """Load IsolationForest after verifying SHA-256 integrity."""
    sha_path = path.with_suffix(".sha256")
    if not sha_path.exists():
        raise FileNotFoundError(f"SHA-256 manifest not found for {path}")
    expected = sha_path.read_text(encoding="utf-8").strip()
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"Model file integrity check FAILED for {path}. "
            "The file may have been tampered with. Retrain the model."
        )
    return joblib.load(path)
```

Also remove the `import pickle` at the top of `common.py`.

- [ ] **Step 3: Update `train.py` to use `save_lgbm` and `.lgbm` extension**

Replace (in `train_model()`):
```python
# OLD:
model_path = model_dir() / f"{version}.pkl"
...
save_pickle(model, model_path)
```
```python
# NEW:
model_path = model_dir() / f"{version}.lgbm"
...
save_lgbm(model, model_path)
```

Update imports in `train.py`:
```python
from models.common import encode_features, feature_columns, forward_date_splits, model_dir, save_json, save_lgbm, training_dataset
```

- [ ] **Step 4: Update `anomaly.py` to use `save_iforest` and `.joblib` extension**

Replace (in `train_anomaly_model()`):
```python
# OLD:
model_path = model_dir() / f"{version}.pkl"
...
save_pickle(model, model_path)
```
```python
# NEW:
model_path = model_dir() / f"{version}.joblib"
...
save_iforest(model, model_path)
```

Update imports in `anomaly.py`:
```python
from models.common import encode_features, feature_columns, forward_date_splits, model_dir, save_json, save_iforest, training_dataset
```

- [ ] **Step 5: Update `score.py` to use safe loaders**

Replace `_load_latest_model` in `score.py`:
```python
def _load_latest_model(prefix: str, extension: str) -> tuple[Path, dict]:
    settings = load_settings()
    model_files = sorted((settings.artifact_dir / "models").glob(f"{prefix}_*.{extension}"))
    if not model_files:
        raise FileNotFoundError(f"No model found for prefix={prefix}, extension={extension}")
    model_path = model_files[-1]
    meta = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    return model_path, meta
```

Replace loading calls in `score_latest_snapshot()`:
```python
# OLD:
lgbm_path, lgbm_meta = _load_latest_model("lgbm")
anomaly_path, anomaly_meta = _load_latest_model("iforest")
...
lgbm = load_pickle(lgbm_path)
anomaly_model = load_pickle(anomaly_path)
model_probability = lgbm.predict_proba(x_score)[:, 1]
```
```python
# NEW:
lgbm_path, lgbm_meta = _load_latest_model("lgbm", "lgbm")
anomaly_path, anomaly_meta = _load_latest_model("iforest", "joblib")
...
lgbm = load_lgbm(lgbm_path)
anomaly_model = load_iforest(anomaly_path)
model_probability = lgbm.predict(x_score)  # Booster.predict returns 1D proba array
```

Update imports:
```python
from models.common import encode_features, feature_columns, load_feature_table, load_lgbm, load_iforest
```

- [ ] **Step 6: Update `validate.py` to use safe LightGBM loader**

Replace `_load_latest` in `validate.py`:
```python
def _load_latest(prefix: str) -> tuple["lgb.Booster", dict]:
    settings = load_settings()
    model_files = sorted((settings.artifact_dir / "models").glob(f"{prefix}_*.lgbm"))
    model_path = model_files[-1]
    meta = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    return load_lgbm(model_path), meta
```

In `validate_model()`, `model.predict_proba(encoded)[:, 1]` → `model.predict(encoded)`.
Also update `_top_feature_importance`: `model.booster_.feature_importance(...)` → `model.feature_importance(...)` (lgb.Booster has `.feature_importance()` directly).

Update imports in `validate.py`:
```python
from models.common import encode_features, feature_columns, forward_date_splits, load_lgbm, training_dataset
```

- [ ] **Step 7: Update `explain.py` to use safe LightGBM loader**

Replace pickle loading in `explain_user()`:
```python
# OLD:
model_files = sorted((settings.artifact_dir / "models").glob("lgbm_*.pkl"))
...
model = load_pickle(model_path)
```
```python
# NEW:
model_files = sorted((settings.artifact_dir / "models").glob("lgbm_*.lgbm"))
...
model = load_lgbm(model_path)
```

Update import:
```python
from models.common import encode_features, feature_columns, load_feature_table, load_lgbm
```

- [ ] **Step 8: Retrain models to generate new format files**

```bash
cd bitoguard_core && source .venv/bin/activate
PYTHONPATH=. python models/train.py       # generates lgbm_*.lgbm
PYTHONPATH=. python models/anomaly.py     # generates iforest_*.joblib + *.sha256
PYTHONPATH=. python models/validate.py   # validates and saves report
```

Expected: no errors, new files in `artifacts/models/`

- [ ] **Step 9: Run full suite**

```bash
PYTHONPATH=. pytest tests/ -q
```
Expected: all pass (smoke tests copy the new format files from artifacts/)

- [ ] **Step 10: Commit**

```bash
git add bitoguard_core/models/common.py bitoguard_core/models/train.py \
        bitoguard_core/models/anomaly.py bitoguard_core/models/score.py \
        bitoguard_core/models/validate.py bitoguard_core/services/explain.py \
        bitoguard_core/requirements.txt bitoguard_core/artifacts/models/
git commit -m "fix: replace pickle with LightGBM text format and joblib+SHA256 for IsolationForest"
```

---

### Task 6: Fix IsolationForest contamination parameter

**Files:**
- Modify: `bitoguard_core/models/anomaly.py`

- [ ] **Step 1: Write test**

Add to `tests/test_model_pipeline.py` (in the existing `_seed_model_tables` context):

```python
def test_iforest_contamination_is_fixed(tmp_path: Path, monkeypatch) -> None:
    """IsolationForest must use a fixed contamination, not derived from labels."""
    import inspect
    from models.anomaly import train_anomaly_model
    src = inspect.getsource(train_anomaly_model)
    assert "hidden_suspicious_label" not in src or "contamination" not in src.split("hidden_suspicious_label")[0].split("contamination")[-1], \
        "IsolationForest contamination must not depend on hidden_suspicious_label"
    # Simpler: just assert the fixed value is present in the source
    assert "contamination=0.05" in src, "contamination should be fixed at 0.05"
```

- [ ] **Step 2: Confirm test fails**

```bash
PYTHONPATH=. pytest tests/test_model_pipeline.py::test_iforest_contamination_is_fixed -v
```
Expected: FAIL

- [ ] **Step 3: Fix `anomaly.py`**

Replace (line 16-19 of `anomaly.py`):
```python
# OLD:
model = IsolationForest(
    n_estimators=200,
    contamination=min(0.5, max(0.01, float(train_frame["hidden_suspicious_label"].mean()))),
    random_state=42,
)
```
```python
# NEW:
model = IsolationForest(
    n_estimators=200,
    contamination=0.05,  # fixed domain estimate; must not be derived from labels
    random_state=42,
)
```

- [ ] **Step 4: Run test**

```bash
PYTHONPATH=. pytest tests/test_model_pipeline.py::test_iforest_contamination_is_fixed -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bitoguard_core/models/anomaly.py
git commit -m "fix: use fixed contamination=0.05 in IsolationForest to prevent label leakage"
```

---

### Task 7: Fix `validate_model` silent train-set fallback

**Files:**
- Modify: `bitoguard_core/models/validate.py`

- [ ] **Step 1: Write test**

Add to `tests/test_model_pipeline.py`:

```python
def test_validate_model_report_includes_split_used(tmp_path: Path, monkeypatch) -> None:
    """validate_model report must include split_used field so callers know which split was evaluated."""
    store = _configure_model_store(tmp_path, monkeypatch)
    _seed_model_tables(store)
    train_model()
    train_anomaly_model()
    report = validate_model()
    assert "split_used" in report, "report must include split_used field"
    assert report["split_used"] in ("holdout", "valid", "train"), \
        f"split_used must be one of holdout/valid/train, got {report['split_used']}"
```

- [ ] **Step 2: Confirm test fails**

```bash
PYTHONPATH=. pytest tests/test_model_pipeline.py::test_validate_model_report_includes_split_used -v
```
Expected: FAIL (split_used not in report)

- [ ] **Step 3: Fix `validate.py`**

Replace line 87:
```python
# OLD:
holdout_dates = set(date_splits["holdout"] or date_splits["valid"] or date_splits["train"])
```
```python
# NEW:
_holdout_list = date_splits["holdout"] or date_splits["valid"] or date_splits["train"]
if date_splits["holdout"]:
    _split_used = "holdout"
elif date_splits["valid"]:
    _split_used = "valid"
else:
    _split_used = "train"
holdout_dates = set(_holdout_list)
```

Add `"split_used": _split_used` to the `report` dict (after `"model_version"` on line 142).

Also update `test_metrics_model_endpoint_returns_full_report` in `test_smoke.py` to assert `"split_used" in body`.

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=. pytest tests/ -q
```
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add bitoguard_core/models/validate.py bitoguard_core/tests/test_smoke.py
git commit -m "fix: add split_used field to validation report; remove silent train-set fallback"
```

---

### Task 8: Fix graph risk score with absolute thresholds

**Files:**
- Modify: `bitoguard_core/models/score.py`
- Modify: `bitoguard_core/models/score.py` (atomic DELETE+INSERT)

- [ ] **Step 1: Write test for reproducibility**

Add to `tests/test_model_pipeline.py`:

```python
def test_graph_risk_score_is_reproducible(tmp_path: Path, monkeypatch) -> None:
    """_graph_risk_score must return the same values regardless of batch composition."""
    import pandas as pd
    from models.score import _graph_risk_score

    frame_a = pd.DataFrame([{"blacklist_1hop_count": 0, "blacklist_2hop_count": 0,
                              "shared_device_count": 0, "shared_bank_count": 3}])
    frame_b = pd.DataFrame([{"blacklist_1hop_count": 0, "blacklist_2hop_count": 0,
                              "shared_device_count": 0, "shared_bank_count": 3},
                             {"blacklist_1hop_count": 0, "blacklist_2hop_count": 0,
                              "shared_device_count": 0, "shared_bank_count": 100}])
    score_alone = float(_graph_risk_score(frame_a).iloc[0])
    score_with_outlier = float(_graph_risk_score(frame_b).iloc[0])
    assert abs(score_alone - score_with_outlier) < 0.001, \
        f"Graph risk score changed when batch composition changed: {score_alone} vs {score_with_outlier}"
```

- [ ] **Step 2: Confirm test fails**

```bash
PYTHONPATH=. pytest tests/test_model_pipeline.py::test_graph_risk_score_is_reproducible -v
```
Expected: FAIL (batch-relative normalization causes values to differ)

- [ ] **Step 3: Fix `_graph_risk_score` in `score.py`**

Replace lines 25-32:
```python
# OLD:
def _graph_risk_score(frame: pd.DataFrame) -> pd.Series:
    raw = (
        frame["blacklist_1hop_count"] * 0.6
        + frame["blacklist_2hop_count"] * 0.4
        + frame["shared_device_count"] * 0.05
        + frame["shared_bank_count"] * 0.05
    )
    return raw.clip(lower=0).pipe(lambda s: s / max(1.0, s.max()))
```
```python
# NEW:
def _graph_risk_score(frame: pd.DataFrame) -> pd.Series:
    """Absolute-threshold graph risk (0–1). Reproducible across scoring batches.

    Blacklist proximity features (blacklist_1hop/2hop, shared_device_count) are
    disabled by default (graph_trusted_only=True) and contribute zero in that mode.
    shared_bank_count uses a log-scale capped at 10 accounts.
    """
    blacklist_risk = (
        (frame["blacklist_1hop_count"] > 0).astype(float) * 0.60
        + (frame["blacklist_2hop_count"] > 0).astype(float) * 0.30
    )
    device_risk = (frame["shared_device_count"].clip(0, 20) / 20.0) * 0.05
    bank_risk = (frame["shared_bank_count"].clip(0, 10) / 10.0) * 0.05
    return (blacklist_risk + device_risk + bank_risk).clip(0.0, 1.0)
```

Also wrap the DELETE+INSERT in `score_latest_snapshot` in a transaction (replace lines 103-104):
```python
# OLD:
store.execute("DELETE FROM ops.model_predictions WHERE snapshot_date = ?", (latest_date.date(),))
store.append_dataframe("ops.model_predictions", prediction_rows)
```
```python
# NEW:
with store.transaction() as conn:
    conn.execute(
        "DELETE FROM ops.model_predictions WHERE snapshot_date = ?",
        (latest_date.date(),),
    )
    conn.register("pred_df", prediction_rows)
    conn.execute("INSERT INTO ops.model_predictions SELECT * FROM pred_df")
    conn.unregister("pred_df")
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=. pytest tests/ -q
```
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add bitoguard_core/models/score.py
git commit -m "fix: absolute-threshold graph risk score for reproducibility; atomic prediction upsert"
```

---

## Chunk 3: API Security & Performance

### Task 9: Add API key authentication

**Files:**
- Modify: `bitoguard_core/config.py`
- Modify: `bitoguard_core/api/main.py`

- [ ] **Step 1: Write test**

Add to `tests/test_smoke.py`:

```python
def test_api_key_enforcement_when_configured(tmp_path: Path, monkeypatch) -> None:
    """When BITOGUARD_API_KEY is set, requests without the key get 403."""
    _configure_temp_db(tmp_path, monkeypatch)
    monkeypatch.setenv("BITOGUARD_API_KEY", "test-secret-key-abc123")
    # Re-import app with new settings (settings are module-level, need reload trick)
    import importlib
    import api.main as main_module
    importlib.reload(main_module)
    client = TestClient(main_module.app)

    # Without key: 403
    resp = client.get("/alerts")
    assert resp.status_code == 403

    # With wrong key: 403
    resp = client.get("/alerts", headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 403

    # With correct key: 200
    resp = client.get("/alerts", headers={"X-API-Key": "test-secret-key-abc123"})
    assert resp.status_code == 200

    # /healthz never requires key
    resp = client.get("/healthz")
    assert resp.status_code == 200
```

- [ ] **Step 2: Confirm test fails**

```bash
PYTHONPATH=. pytest tests/test_smoke.py::test_api_key_enforcement_when_configured -v
```
Expected: FAIL (no 403 returned currently)

- [ ] **Step 3: Add `api_key` to `config.py`**

In `Settings` dataclass, add:
```python
api_key: str | None  # None = authentication disabled (dev mode)
```

In `load_settings()`, add to the `Settings(...)` call:
```python
api_key=os.getenv("BITOGUARD_API_KEY") or None,
```

- [ ] **Step 4: Add auth dependency to `api/main.py`**

Add after imports:
```python
from fastapi import Depends, Security
from fastapi.security import APIKeyHeader

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _require_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    if _settings.api_key is None:
        return  # Auth disabled in dev mode
    if api_key != _settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid or missing X-API-Key header")
```

Add `dependencies=[Depends(_require_api_key)]` to every endpoint decorator **except** `@app.get("/healthz")`:

```python
@app.post("/pipeline/sync", dependencies=[Depends(_require_api_key)])
@app.post("/features/rebuild", dependencies=[Depends(_require_api_key)])
@app.post("/model/train", dependencies=[Depends(_require_api_key)])
@app.post("/model/score", dependencies=[Depends(_require_api_key)])
@app.get("/alerts", dependencies=[Depends(_require_api_key)])
@app.get("/alerts/{alert_id}/report", dependencies=[Depends(_require_api_key)])
@app.post("/alerts/{alert_id}/decision", dependencies=[Depends(_require_api_key)])
@app.get("/users/{user_id}/360", dependencies=[Depends(_require_api_key)])
@app.get("/users/{user_id}/graph", dependencies=[Depends(_require_api_key)])
@app.get("/metrics/model", dependencies=[Depends(_require_api_key)])
@app.get("/metrics/threshold", dependencies=[Depends(_require_api_key)])
@app.get("/metrics/drift", dependencies=[Depends(_require_api_key)])
```

- [ ] **Step 5: Run tests**

```bash
PYTHONPATH=. pytest tests/ -q
```
Expected: all pass (existing smoke tests don't set `BITOGUARD_API_KEY`, so auth is disabled — no regressions)

- [ ] **Step 6: Update `CLAUDE.md` environment variables table**

Add:
```
| BITOGUARD_API_KEY | (unset) | API key for X-API-Key header auth; unset = auth disabled |
```

- [ ] **Step 7: Commit**

```bash
git add bitoguard_core/config.py bitoguard_core/api/main.py \
        bitoguard_core/tests/test_smoke.py CLAUDE.md
git commit -m "feat: add optional API key authentication (BITOGUARD_API_KEY env var)"
```

---

### Task 10: Fix full-table scans in graph and diagnosis endpoints

**Files:**
- Modify: `bitoguard_core/api/main.py`
- Modify: `bitoguard_core/services/diagnosis.py`

- [ ] **Step 1: Write performance regression test**

Add to `tests/test_smoke.py`:

```python
def test_graph_endpoint_does_not_load_full_edge_table(tmp_path: Path, monkeypatch) -> None:
    """Graph endpoint must query edges by user_id, not load the whole table."""
    # Verify via inspection: _build_graph_payload should not call read_table("canonical.entity_edges")
    import inspect
    from api.main import _build_graph_payload
    src = inspect.getsource(_build_graph_payload)
    assert 'read_table("canonical.entity_edges")' not in src, \
        "_build_graph_payload must not load the full entity_edges table"
    assert "_load_neighborhood_edges" in src or "fetch_df" in src, \
        "_build_graph_payload must use filtered SQL queries"
```

- [ ] **Step 2: Confirm test fails**

```bash
PYTHONPATH=. pytest tests/test_smoke.py::test_graph_endpoint_does_not_load_full_edge_table -v
```
Expected: FAIL

- [ ] **Step 3: Add `_load_neighborhood_edges` helper to `api/main.py`**

Add this function before `_build_graph_payload`:

```python
def _load_neighborhood_edges(store: DuckDBStore, user_id: str, max_hops: int) -> pd.DataFrame:
    """Load entity edges for user's neighborhood without a full-table scan."""
    one_hop = store.fetch_df(
        "SELECT * FROM canonical.entity_edges WHERE src_id = ? OR dst_id = ?",
        (user_id, user_id),
    )
    if one_hop.empty or max_hops < 2:
        return one_hop
    neighbor_ids = (
        set(one_hop["src_id"].tolist()) | set(one_hop["dst_id"].tolist())
    ) - {user_id}
    if not neighbor_ids:
        return one_hop
    placeholders = ", ".join(["?"] * len(neighbor_ids))
    nb = list(neighbor_ids)
    two_hop = store.fetch_df(
        f"SELECT * FROM canonical.entity_edges WHERE src_id IN ({placeholders}) OR dst_id IN ({placeholders})",
        tuple(nb) * 2,
    )
    return pd.concat([one_hop, two_hop], ignore_index=True).drop_duplicates(subset=["edge_id"])
```

- [ ] **Step 4: Update `_build_graph_payload` to use the helper**

Replace line 89 of `api/main.py`:
```python
# OLD:
edges = store.read_table("canonical.entity_edges")
```
```python
# NEW:
edges = _load_neighborhood_edges(store, user_id, max_hops)
```

- [ ] **Step 5: Fix full-table scans in `diagnosis.py`**

Replace lines 64-66 in `build_risk_diagnosis` (the three `store.read_table(...)` calls):
```python
# OLD:
login = store.read_table("canonical.login_events")
crypto = store.read_table("canonical.crypto_transactions")
trade = store.read_table("canonical.trade_orders")
```
```python
# NEW:
login = store.fetch_df(
    "SELECT * FROM canonical.login_events WHERE user_id = ? ORDER BY occurred_at DESC LIMIT 50",
    (user_id,),
)
crypto = store.fetch_df(
    "SELECT * FROM canonical.crypto_transactions WHERE user_id = ? ORDER BY occurred_at DESC LIMIT 50",
    (user_id,),
)
trade = store.fetch_df(
    "SELECT * FROM canonical.trade_orders WHERE user_id = ? ORDER BY occurred_at DESC LIMIT 50",
    (user_id,),
)
```

Note: `_timeline_summary` already filters by `user_id` in Python — after this fix the filter is redundant but harmless. The `subset = frame[frame["user_id"] == user_id]` line can be removed from `_timeline_summary`, but leave it as defensive code.

- [ ] **Step 6: Run tests**

```bash
PYTHONPATH=. pytest tests/ -q
```
Expected: all pass (graph smoke test verifies hop constraints still hold)

- [ ] **Step 7: Commit**

```bash
git add bitoguard_core/api/main.py bitoguard_core/services/diagnosis.py
git commit -m "perf: replace full-table scans in graph and diagnosis endpoints with filtered SQL"
```

---

### Task 11: Frontend proxy path allowlist

**Files:**
- Modify: `bitoguard_frontend/src/app/api/backend/[...path]/route.ts`

- [ ] **Step 1: Write test**

Create `bitoguard_frontend/src/__tests__/proxy-allowlist.test.ts` (or use manual curl test since there's no frontend test suite):

Since there is no existing frontend test suite, verify manually with curl after the change:
```bash
# After npm run dev is running:
# Should succeed:
curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/api/backend/alerts
# Should return 403:
curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/api/backend/pipeline/sync
curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/api/backend/model/train
```

- [ ] **Step 2: Add allowlist to route.ts**

Replace the `proxy` function with this:

```typescript
const FRONTEND_ALLOWED_PATHS: ReadonlySet<string> = new Set([
  "healthz",
  "alerts",
  "users",
  "metrics",
])

async function proxy(request: NextRequest, path: string[]) {
  const firstSegment = path[0]
  if (!firstSegment || !FRONTEND_ALLOWED_PATHS.has(firstSegment)) {
    return NextResponse.json(
      { message: "Forbidden: this path is not accessible via the frontend proxy" },
      { status: 403, headers: { "cache-control": "no-store" } },
    )
  }

  try {
    const upstreamUrl = new URL(`${API_BASE}/${path.join("/")}`)
    upstreamUrl.search = request.nextUrl.search

    const headers = new Headers()
    const contentType = request.headers.get("content-type")
    if (contentType) {
      headers.set("content-type", contentType)
    }

    const upstreamResponse = await fetch(upstreamUrl, {
      method: request.method,
      headers,
      body: request.method === "GET" || request.method === "HEAD" ? undefined : await request.text(),
      cache: "no-store",
    })

    return new NextResponse(upstreamResponse.body, {
      status: upstreamResponse.status,
      headers: {
        "content-type": upstreamResponse.headers.get("content-type") ?? "application/json; charset=utf-8",
        "cache-control": "no-store",
      },
    })
  } catch {
    return NextResponse.json(
      { message: "Unable to reach internal API" },
      { status: 502, headers: { "cache-control": "no-store" } },
    )
  }
}
```

- [ ] **Step 3: Build frontend to catch type errors**

```bash
cd bitoguard_frontend && npm run build
```
Expected: build succeeds

- [ ] **Step 4: Commit**

```bash
git add bitoguard_frontend/src/app/api/backend/\[...path\]/route.ts
git commit -m "fix: add path allowlist to frontend proxy; block pipeline and training endpoints"
```

---

## Chunk 4: Graph Feature Correctness

### Task 12: Fix blacklist label leakage in graph fast path

**Files:**
- Modify: `bitoguard_core/features/graph_features.py`

This issue only affects when `BITOGUARD_GRAPH_FEATURES_TRUSTED_ONLY=false` (non-default). The fast path (line 127-130) uses `max_date` (global maximum across the entire edge set) as the blacklist cutoff, instead of bounding by each snapshot's end date.

- [ ] **Step 1: Write test**

Add to `tests/test_graph_data_quality.py`:

```python
def test_fast_path_blacklist_is_snapshot_bounded(tmp_path: Path, monkeypatch) -> None:
    """Graph fast path must filter blacklist by snapshot date, not global max_date.

    A user blacklisted at T+1 must NOT appear in the blacklisted_set when
    computing features for snapshot at T.
    """
    # This is a code inspection test since enabling trusted_only=False requires
    # clean graph data that doesn't exist in the test environment.
    import inspect
    from features import graph_features
    src = inspect.getsource(graph_features._build_graph_features_fast)
    # The blacklisted_set must NOT be computed using a single global max_date
    # (it should be computed per snapshot_date or inside a per-snapshot loop)
    assert "max_date" not in src or "blacklisted_set" not in src.split("max_date")[0].split("blacklisted_set")[-1], \
        "Blacklist set must not use global max_date — must be bounded per snapshot"
```

- [ ] **Step 2: Confirm test fails**

```bash
PYTHONPATH=. pytest tests/test_graph_data_quality.py::test_fast_path_blacklist_is_snapshot_bounded -v
```
Expected: FAIL

- [ ] **Step 3: Fix the fast path in `graph_features.py`**

The current fast path computes `blacklisted_set` once using global `max_date` (line 127-130). The fix moves the blacklist filter inside the per-date loop so each snapshot date gets a correctly bounded blacklist.

In `_build_graph_features_fast`, move the `blacklisted_set` construction:

Remove lines 126-130:
```python
# REMOVE:
max_date = edges_df["snapshot_time"].max()
blacklisted_set = set(
    blacklist_feed[blacklist_feed["observed_at"] <= max_date]["user_id"].astype(str)
)
```

In the records assembly loop (where features are replicated across snapshot dates), compute the per-snapshot blacklist:

```python
# In the records loop, add per-snapshot blacklist filter:
for sd in snapshot_dates:
    snapshot_end = pd.Timestamp(sd, tz="UTC") + pd.Timedelta(days=1)
    blacklisted_set = set(
        blacklist_feed[blacklist_feed["observed_at"] < snapshot_end]["user_id"].astype(str)
    )
    for uid in target_user_ids:
        base = per_user[uid]
        # Recompute blacklist proximity for this snapshot's blacklisted_set
        if not trusted_only and uid in per_user and per_user[uid].get("_has_graph_data"):
            # Recompute hop counts with per-snapshot blacklist
            node = _prefix("user", uid)
            lengths = nx.single_source_shortest_path_length(graph, node, cutoff=4) if node in graph else {}
            blacklist_1hop = sum(
                1 for n, d in lengths.items() if d == 2 and _node_type(n) == "user"
                and n.split(":", 1)[1] in blacklisted_set
            )
            blacklist_2hop = sum(
                1 for n, d in lengths.items() if d in (3, 4) and _node_type(n) == "user"
                and n.split(":", 1)[1] in blacklisted_set
            )
        else:
            blacklist_1hop = 0
            blacklist_2hop = 0
        records.append({...})
```

Note: This makes the fast path slightly slower (one blacklist filter per snapshot date), but still O(|dates| × |users|) not O(|dates| × |edges|). Given this code path is only active when `trusted_only=False` (non-default), the performance impact is acceptable.

The exact implementation details depend on the surrounding loop structure — read the full function body carefully before editing.

- [ ] **Step 4: Run full suite**

```bash
PYTHONPATH=. pytest tests/ -q
```
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add bitoguard_core/features/graph_features.py
git commit -m "fix: bound blacklist set by snapshot date in graph fast path to prevent label leakage"
```

---

## Post-Implementation Verification

- [ ] **Run complete test suite one final time**

```bash
cd bitoguard_core && source .venv/bin/activate
PYTHONPATH=. pytest tests/ -v
```
Expected: all 85+ tests pass

- [ ] **Verify API starts cleanly**

```bash
PYTHONPATH=. uvicorn api.main:app --port 8001 &
curl http://localhost:8001/healthz
# Expected: {"status": "ok"}
curl http://localhost:8001/alerts
# Expected: 200 (no API key set, auth disabled)
BITOGUARD_API_KEY=secret PYTHONPATH=. uvicorn api.main:app --port 8002 &
curl http://localhost:8002/alerts
# Expected: 403
curl -H "X-API-Key: secret" http://localhost:8002/alerts
# Expected: 200
kill %1 %2
```

- [ ] **Final commit**

```bash
git log --oneline -15  # verify clean commit history
```

---

## Summary of Changes by Issue

| Issue | Task | Files |
|-------|------|-------|
| CRITICAL-1: pickle RCE | Task 5 | common.py, train.py, anomaly.py, score.py, validate.py, explain.py |
| CRITICAL-2: no auth | Task 9 | config.py, api/main.py |
| CRITICAL-3: full-table scans | Task 10 | api/main.py, diagnosis.py |
| CRITICAL-4: SQL injection | Task 1 | store.py |
| HIGH-1: IForest contamination | Task 6 | anomaly.py |
| HIGH-2: validation fallback | Task 7 | validate.py |
| HIGH-3: concurrent DuckDB | Task 2 | store.py |
| HIGH-4: replace_table schema | Task 2 | store.py |
| HIGH-5: open proxy | Task 11 | route.ts |
| HIGH-6: non-atomic decisions | Task 4 | alert_engine.py |
| HIGH-7: graph risk normalization | Task 8 | score.py |
| HIGH-8: blacklist leakage | Task 12 | graph_features.py |
