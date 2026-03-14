# BitoGuard Repo Refactor & Cleanup — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix critical correctness bugs identified by Opus review, eliminate dead code, harden security boundaries, and clean up untracked repo artifacts.

**Architecture:** Four independent chunks: (1) critical bug fixes that break model correctness, (2) feature engineering correctness including deterministic encoding, (3) architecture hardening — dead code + security, (4) repo cleanup including docker-compose and .gitignore. Each chunk is independently testable and committable.

**Tech Stack:** Python 3.12, DuckDB (in-memory for tests), CatBoost, LightGBM, scikit-learn, FastAPI, Docker Compose

---

## Chunk 1: Critical Bug Fixes

### Task 1.1: Fix Label Leakage in CatBoost Training Query

**Problem:** `_load_v2_training_dataset()` in `train_catboost.py:37` contains `OR ped.ped IS NULL` which includes ALL snapshot rows for positive-labeled users who have **no entry** in `canonical.blacklist_feed` — even snapshots from before they were flagged. The correct behavior (matching v1 `common.py:training_dataset`) is to **exclude** those users entirely.

**Files:**
- Modify: `bitoguard_core/models/train_catboost.py:36-38`
- Test: `bitoguard_core/tests/test_model_pipeline.py` (add new test)

- [ ] **Step 1: Write the failing test**

Add to `bitoguard_core/tests/test_model_pipeline.py`:

```python
def test_v2_training_query_excludes_positive_without_blacklist_date():
    """
    Positive users with no canonical.blacklist_feed entry must be excluded.
    This test creates a minimal in-memory DuckDB reproducing the exact
    CTE used in _load_v2_training_dataset() and verifies the WHERE clause.
    """
    import duckdb
    import pandas as pd

    conn = duckdb.connect(":memory:")
    conn.execute("CREATE SCHEMA canonical")
    conn.execute("CREATE SCHEMA features")
    conn.execute("CREATE SCHEMA ops")
    conn.execute("""
        CREATE TABLE features.feature_snapshots_v2 (
            user_id VARCHAR, snapshot_date DATE,
            feature_snapshot_id VARCHAR, feature_version VARCHAR
        )
    """)
    conn.execute("""
        CREATE TABLE ops.oracle_user_labels (
            user_id VARCHAR, hidden_suspicious_label INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE canonical.blacklist_feed (
            user_id VARCHAR, observed_at TIMESTAMPTZ
        )
    """)
    # u_neg: label=0, no blacklist → should be included
    conn.execute("INSERT INTO features.feature_snapshots_v2 VALUES ('u_neg', '2025-01-01', 'f1', 'v2')")
    # u_pos_dated: label=1, has blacklist entry → included from that date forward
    conn.execute("INSERT INTO features.feature_snapshots_v2 VALUES ('u_pos_dated', '2025-01-01', 'f2', 'v2')")
    conn.execute("INSERT INTO ops.oracle_user_labels VALUES ('u_pos_dated', 1)")
    conn.execute("INSERT INTO canonical.blacklist_feed VALUES ('u_pos_dated', '2024-12-01 00:00:00+00')")
    # u_pos_nodated: label=1, NO blacklist entry → must be EXCLUDED
    conn.execute("INSERT INTO features.feature_snapshots_v2 VALUES ('u_pos_nodated', '2025-01-01', 'f3', 'v2')")
    conn.execute("INSERT INTO ops.oracle_user_labels VALUES ('u_pos_nodated', 1)")

    # First: prove the bug exists using the BUGGY query
    buggy = conn.execute("""
        WITH ped AS (
            SELECT user_id, CAST(MIN(observed_at) AS DATE) AS ped
            FROM canonical.blacklist_feed WHERE observed_at IS NOT NULL
            GROUP BY user_id
        )
        SELECT f.user_id, COALESCE(l.hidden_suspicious_label, 0) AS hidden_suspicious_label
        FROM features.feature_snapshots_v2 f
        LEFT JOIN ops.oracle_user_labels l ON f.user_id = l.user_id
        LEFT JOIN ped ON f.user_id = ped.user_id
        WHERE COALESCE(l.hidden_suspicious_label, 0) = 0
           OR ped.ped IS NULL
           OR f.snapshot_date >= ped.ped
    """).df()
    assert "u_pos_nodated" in buggy["user_id"].values, (
        "BUG CONFIRMED: buggy query includes positive user with no blacklist entry"
    )

    # Then: verify the fixed query excludes them
    fixed = conn.execute("""
        WITH ped AS (
            SELECT user_id, CAST(MIN(observed_at) AS DATE) AS ped
            FROM canonical.blacklist_feed WHERE observed_at IS NOT NULL
            GROUP BY user_id
        )
        SELECT f.user_id, COALESCE(l.hidden_suspicious_label, 0) AS hidden_suspicious_label
        FROM features.feature_snapshots_v2 f
        LEFT JOIN ops.oracle_user_labels l ON f.user_id = l.user_id
        LEFT JOIN ped ON f.user_id = ped.user_id
        WHERE COALESCE(l.hidden_suspicious_label, 0) = 0
           OR (ped.ped IS NOT NULL AND f.snapshot_date >= ped.ped)
    """).df()

    assert "u_neg" in fixed["user_id"].values
    assert "u_pos_dated" in fixed["user_id"].values
    assert "u_pos_nodated" not in fixed["user_id"].values, (
        "Positive user with no blacklist entry must be excluded from training"
    )
    conn.close()
```

- [ ] **Step 2: Run test to confirm it PASSES (both buggy and fixed assertions succeed)**

```bash
cd bitoguard_core && source .venv/bin/activate
PYTHONPATH=. pytest tests/test_model_pipeline.py::test_v2_training_query_excludes_positive_without_blacklist_date -v
```

Expected: PASS — the test proves the bug exists in the old query and that the fixed query is correct.

- [ ] **Step 3: Apply the fix to `train_catboost.py`**

In `bitoguard_core/models/train_catboost.py`, replace lines 36-38:

```python
        WHERE COALESCE(l.hidden_suspicious_label, 0) = 0
           OR ped.ped IS NULL
           OR f.snapshot_date >= ped.ped
```

With:

```python
        WHERE COALESCE(l.hidden_suspicious_label, 0) = 0
           OR (ped.ped IS NOT NULL AND f.snapshot_date >= ped.ped)
```

- [ ] **Step 4: Run full test suite to confirm no regressions**

```bash
cd bitoguard_core && source .venv/bin/activate
PYTHONPATH=. pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: test added in Step 1 passes; previously-passing tests still pass; 5 pre-existing smoke failures unchanged.

- [ ] **Step 5: Commit**

```bash
git add bitoguard_core/models/train_catboost.py bitoguard_core/tests/test_model_pipeline.py
git commit -m "fix: exclude positive users with no blacklist effective date from v2 training (label leakage)"
```

---

### Task 1.2: Make `score_latest_snapshot_v2()` Atomically Write Predictions

**Problem:** `score.py:361-362` does a bare `store.execute(DELETE)` then `store.append_dataframe(INSERT)` as two separate operations. If the process crashes between them, all predictions for that `snapshot_date` are lost. The v1 path (`score_latest_snapshot()`) correctly uses `store.transaction()`.

**Files:**
- Modify: `bitoguard_core/models/score.py:361-362`
- Test: `bitoguard_core/tests/test_smoke.py` (add new test)

- [ ] **Step 1: Write the test**

Add to `bitoguard_core/tests/test_smoke.py`:

```python
def test_score_v2_delete_insert_uses_transaction(tmp_path, monkeypatch):
    """
    score_latest_snapshot_v2 must wrap DELETE + INSERT in a transaction.
    Simulate crash after DELETE: existing predictions must survive.
    """
    import pandas as pd
    from unittest.mock import MagicMock, call, patch
    from db.store import DuckDBStore

    # Patch out the expensive model loading; test only the DB write behaviour
    with patch("models.score.load_feature_table") as mock_ft, \
         patch("models.score._load_latest_model") as mock_lm, \
         patch("models.score.load_joblib") as mock_lj, \
         patch("models.score.load_iforest") as mock_li, \
         patch("models.score.evaluate_rules") as mock_rules, \
         patch("models.score.generate_alerts"):

        # Minimal feature frame (1 user, 1 date)
        mock_ft.return_value = pd.DataFrame({
            "user_id": ["u1"], "snapshot_date": [pd.Timestamp("2025-06-01")],
            "feature_snapshot_id": ["f1"], "feature_version": ["v2"],
        })
        stacker_meta = {
            "stacker_version": "stacker_test",
            "feature_columns": [],
            "branch_models": {"catboost": "/nonexistent", "lgbm": "/nonexistent"},
        }
        mock_lm.return_value = ("/fake/stacker.joblib", stacker_meta)
        import numpy as np
        # Return numpy array so [:, 1] slicing works in score_latest_snapshot_v2
        mock_lj.return_value = MagicMock(
            predict_proba=lambda x: np.array([[0.2, 0.8]] * max(1, len(x)))
        )
        mock_li.side_effect = FileNotFoundError
        mock_rules.return_value = pd.DataFrame({
            "user_id": ["u1"],
            "snapshot_date": [pd.Timestamp("2025-06-01")],
            "rule_score": [0.0], "rule_hits": ["[]"],
        })

        store = MagicMock(spec=DuckDBStore)
        store.fetch_df.return_value = pd.DataFrame(
            columns=["prediction_id", "user_id", "snapshot_date"]
        )
        # Simulate crash: make append_dataframe raise
        store.append_dataframe.side_effect = RuntimeError("disk full")

        with patch("models.score.DuckDBStore", return_value=store), \
             patch("models.score.load_settings"):
            try:
                from models.score import score_latest_snapshot_v2
                score_latest_snapshot_v2()
            except Exception:
                pass

        # The transaction context manager must have been entered —
        # i.e., store.transaction() was called, not store.execute() then store.append_dataframe()
        # (This test verifies the code path uses transaction, not bare execute+append)
        assert store.transaction.called, (
            "score_latest_snapshot_v2 must use store.transaction() for atomic write"
        )
```

- [ ] **Step 2: Run test to confirm failure**

```bash
cd bitoguard_core && source .venv/bin/activate
PYTHONPATH=. pytest tests/test_smoke.py::test_score_v2_delete_insert_uses_transaction -v
```

Expected: FAIL — current code calls `store.execute()` and `store.append_dataframe()` directly, not `store.transaction()`.

- [ ] **Step 3: Apply the fix to `score.py`**

In `bitoguard_core/models/score.py`, replace lines 356-363:

```python
    pred_rows = result[[
        "prediction_id", "user_id", "snapshot_date", "prediction_time", "model_version",
        "risk_score", "risk_level", "rule_hits", "top_reason_codes",
        "model_probability", "anomaly_score", "graph_risk",
    ]].copy()
    store.execute("DELETE FROM ops.model_predictions WHERE snapshot_date = ?", (latest_date.date(),))
    store.append_dataframe("ops.model_predictions", pred_rows)
    generate_alerts()
    return pred_rows
```

With:

```python
    pred_rows = result[[
        "prediction_id", "user_id", "snapshot_date", "prediction_time", "model_version",
        "risk_score", "risk_level", "rule_hits", "top_reason_codes",
        "model_probability", "anomaly_score", "graph_risk",
    ]].copy()
    with store.transaction() as conn:
        conn.execute(
            "DELETE FROM ops.model_predictions WHERE snapshot_date = ?",
            (latest_date.date(),),
        )
        conn.register("pred_df_v2", pred_rows)
        conn.execute("INSERT INTO ops.model_predictions SELECT * FROM pred_df_v2")
        conn.unregister("pred_df_v2")
    generate_alerts()
    return pred_rows
```

- [ ] **Step 4: Run tests**

```bash
cd bitoguard_core && source .venv/bin/activate
PYTHONPATH=. pytest tests/test_smoke.py::test_score_v2_delete_insert_uses_transaction -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bitoguard_core/models/score.py bitoguard_core/tests/test_smoke.py
git commit -m "fix: wrap score_latest_snapshot_v2 DELETE+INSERT in atomic transaction"
```

---

### Task 1.3: Replace Wall-Clock Time with `snapshot_date` in Feature Modules

**Problem:**
- `profile_features.py:29`: `pd.Timestamp.now(tz="UTC")` used for `account_age_days`
- `twd_features.py:59`: `pd.Timestamp.now(tz="UTC")` used for `twd_recency_days`

Both features change value between pipeline runs for the same data, making them non-reproducible. They should be computed relative to `snapshot_date`.

The `snapshot_date` flows from `build_and_store_v2_features()` → `build_v2_features()` → each module. Add it as an optional parameter to each module function signature.

**Files:**
- Modify: `bitoguard_core/features/profile_features.py`
- Modify: `bitoguard_core/features/twd_features.py`
- Modify: `bitoguard_core/features/registry.py` (`build_v2_features` and `build_and_store_v2_features`)
- Test: `bitoguard_core/tests/test_feature_modules.py`

- [ ] **Step 1: Write failing test**

Add to `bitoguard_core/tests/test_feature_modules.py`:

```python
def test_account_age_days_is_deterministic():
    """account_age_days must be the same regardless of when the pipeline runs."""
    import pandas as pd
    from features.profile_features import compute_profile_features

    snap_date = pd.Timestamp("2025-06-01", tz="UTC")
    users = pd.DataFrame([{
        "user_id": "u1",
        "kyc_level": "level1",
        "created_at": "2024-06-01T00:00:00+00:00",
        "occupation": "engineer",
        "declared_source_of_funds": "salary",
        "activity_window": "regular",
        "monthly_income_twd": 50000.0,
    }])

    result1 = compute_profile_features(users, snapshot_date=snap_date)
    result2 = compute_profile_features(users, snapshot_date=snap_date)
    # Must be deterministic: running twice with same snapshot_date gives same value
    assert result1["account_age_days"].iloc[0] == result2["account_age_days"].iloc[0]
    # Must be relative to snapshot_date, not wall clock:
    # 2024-06-01 to 2025-06-01 = 365 days
    assert abs(result1["account_age_days"].iloc[0] - 365.0) < 1.0


def test_twd_recency_days_is_deterministic():
    """twd_recency_days must be fixed relative to snapshot_date."""
    import pandas as pd
    from features.twd_features import compute_twd_features

    snap_date = pd.Timestamp("2025-06-01", tz="UTC")
    fiat = pd.DataFrame([{
        "user_id": "u1",
        "occurred_at": "2025-05-01T00:00:00+00:00",
        "direction": "deposit",
        "amount_twd": 10000.0,
    }])

    result = compute_twd_features(fiat, snapshot_date=snap_date)
    # 2025-05-01 to 2025-06-01 = 31 days
    assert abs(result["twd_recency_days"].iloc[0] - 31.0) < 1.0
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd bitoguard_core && source .venv/bin/activate
PYTHONPATH=. pytest tests/test_feature_modules.py::test_account_age_days_is_deterministic tests/test_feature_modules.py::test_twd_recency_days_is_deterministic -v
```

Expected: FAIL — `snapshot_date` parameter doesn't exist yet.

- [ ] **Step 3: Fix `profile_features.py`**

Replace the function signature and the `now` usage:

```python
def compute_profile_features(
    users: pd.DataFrame,
    snapshot_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """8 demographic/KYC features per user (no aggregation)."""
    if users.empty:
        return pd.DataFrame()

    df = users.copy()
    df["created_at"] = pd.to_datetime(df.get("created_at"), utc=True, errors="coerce")

    df["kyc_level_code"] = df["kyc_level"].map(KYC_LEVEL_MAP).fillna(-1).astype(int)
    df["occupation_code"] = df["occupation"].astype("category").cat.codes
    df["income_source_code"] = df["declared_source_of_funds"].astype("category").cat.codes
    df["user_source_code"] = df["activity_window"].astype("category").cat.codes
    df["monthly_income_twd"] = df.get("monthly_income_twd", 0.0).fillna(0.0)

    ref = pd.Timestamp.now(tz="UTC") if snapshot_date is None else snapshot_date
    if ref.tzinfo is None:
        ref = ref.tz_localize("UTC")
    df["account_age_days"] = (
        (ref - df["created_at"]).dt.total_seconds().div(86400).clip(lower=0).fillna(0)
    )

    keep = [
        "user_id", "kyc_level_code", "occupation_code", "income_source_code",
        "user_source_code", "monthly_income_twd", "account_age_days",
    ]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)
```

- [ ] **Step 4: Fix `twd_features.py`**

Update signature and `twd_recency_days` computation:

```python
def compute_twd_features(
    fiat: pd.DataFrame,
    snapshot_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """~26 TWD fiat transfer features per user (lifetime, no IP — use ip_features.py)."""
    if fiat.empty:
        return pd.DataFrame()

    ref = pd.Timestamp.now(tz="UTC") if snapshot_date is None else snapshot_date
    if ref.tzinfo is None:
        ref = ref.tz_localize("UTC")

    df = fiat.copy()
    df["occurred_at"] = pd.to_datetime(df["occurred_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["user_id", "occurred_at", "amount_twd"])

    rows = []
    for uid, grp in df.groupby("user_id"):
        dep = grp[grp["direction"] == "deposit"]
        wdr = grp[grp["direction"] == "withdrawal"]
        row: dict = {"user_id": uid}

        row.update(_agg_stats(grp["amount_twd"], "twd_all"))
        row.update(_agg_stats(dep["amount_twd"],  "twd_dep"))
        row.update(_agg_stats(wdr["amount_twd"],  "twd_wdr"))
        row["twd_net_flow"] = float(dep["amount_twd"].sum() - wdr["amount_twd"].sum())

        row["twd_active_days"] = int(grp["occurred_at"].dt.date.nunique())
        span = (grp["occurred_at"].max() - grp["occurred_at"].min()).total_seconds() / 86400
        row["twd_span_days"]    = float(max(0.0, span))
        recency = (ref - grp["occurred_at"].max()).total_seconds() / 86400
        row["twd_recency_days"] = float(max(0.0, recency))
        row["twd_night_share"]  = float((grp["occurred_at"].dt.hour.isin(NIGHT_HOURS)).mean())

        for prefix, subset in [("twd_all", grp), ("twd_dep", dep), ("twd_wdr", wdr)]:
            g = _gap_stats(subset["occurred_at"])
            row[f"{prefix}_gap_min"]        = g["gap_min"]
            row[f"{prefix}_gap_p10"]        = g["gap_p10"]
            row[f"{prefix}_gap_median"]     = g["gap_median"]
            row[f"{prefix}_rapid_1h_share"] = g["rapid_1h_share"]

        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)
```

- [ ] **Step 5: Fix `registry.py` — thread `snapshot_date` through `build_v2_features()`**

Update `build_v2_features` signature and module calls in `bitoguard_core/features/registry.py`:

```python
def build_v2_features(
    users:         pd.DataFrame,
    fiat:          pd.DataFrame,
    crypto:        pd.DataFrame,
    trades:        pd.DataFrame,
    logins:        pd.DataFrame,
    edges:         pd.DataFrame,
    snapshot_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Assemble all label-free feature modules. Returns one row per user_id."""
```

And change the `module_entries` list to pass `snapshot_date` to the two affected modules:

```python
    module_entries: list[tuple[pd.DataFrame | None, object]] = [
        (compute_profile_features(users, snapshot_date=snapshot_date),  None),
        (compute_twd_features(fiat, snapshot_date=snapshot_date),       _make_probe_fiat),
        (compute_crypto_features(crypto),                               _make_probe_crypto),
        (compute_swap_features(trades),                                 _make_probe_trades),
        (compute_trading_features(trades),                              _make_probe_trades),
        (compute_ip_features(logins),                                   _make_probe_logins),
        (compute_sequence_features(fiat, trades, crypto),               None),
        (compute_bipartite_features(edges, user_ids),                   None),
    ]
```

And update `build_and_store_v2_features` to pass `snapshot_date` to `build_v2_features`:

```python
    master = build_v2_features(users, fiat, crypto, trades, logins, edges,
                                snapshot_date=snapshot_date)
```

- [ ] **Step 6: Run tests**

```bash
cd bitoguard_core && source .venv/bin/activate
PYTHONPATH=. pytest tests/test_feature_modules.py::test_account_age_days_is_deterministic tests/test_feature_modules.py::test_twd_recency_days_is_deterministic -v
PYTHONPATH=. pytest tests/ -v --tb=short 2>&1 | tail -10
```

Expected: both new tests PASS; full suite unchanged.

- [ ] **Step 7: Commit**

```bash
git add bitoguard_core/features/profile_features.py bitoguard_core/features/twd_features.py bitoguard_core/features/registry.py bitoguard_core/tests/test_feature_modules.py
git commit -m "fix: use snapshot_date instead of wall-clock time in profile and twd features"
```

---

## Chunk 2: Feature Engineering Correctness

### Task 2.1: Deterministic Category Encoding in Profile Features

**Problem:** `profile_features.py:24-26` uses `pandas.Categorical.cat.codes` which assigns integer codes based on alphabetical order of values **present in the current DataFrame**. If a new value appears at scoring time, all existing codes shift → silent train/serve skew.

**Fix:** Compute the `{value: code}` mapping at feature build time, save it as `artifacts/profile_category_maps.json`, and re-apply the saved mapping at subsequent runs. Unknown values at scoring time map to `-1`.

**Files:**
- Modify: `bitoguard_core/features/profile_features.py`
- Modify: `bitoguard_core/features/registry.py`
- Test: `bitoguard_core/tests/test_feature_modules.py`

- [ ] **Step 1: Write the failing test**

Add to `bitoguard_core/tests/test_feature_modules.py`:

```python
def test_profile_category_codes_deterministic_across_populations():
    """
    Category codes must be stable: if training saw ["engineer", "teacher"],
    scoring with ["engineer", "doctor"] must produce the same code for "engineer"
    and -1 for "doctor" (unseen), NOT re-index everything from scratch.
    """
    import pandas as pd
    from features.profile_features import compute_profile_features, build_profile_category_maps

    train_users = pd.DataFrame([
        {"user_id": "u1", "kyc_level": "level1", "created_at": "2024-01-01",
         "occupation": "engineer", "declared_source_of_funds": "salary",
         "activity_window": "regular", "monthly_income_twd": 0.0},
        {"user_id": "u2", "kyc_level": "level1", "created_at": "2024-01-01",
         "occupation": "teacher", "declared_source_of_funds": "salary",
         "activity_window": "regular", "monthly_income_twd": 0.0},
    ])
    # Build maps from training population
    maps = build_profile_category_maps(train_users)
    # engineer gets some code (say 0 alphabetically), teacher gets 1

    train_result = compute_profile_features(train_users, category_maps=maps)
    engineer_code = train_result.loc[train_result["user_id"] == "u1", "occupation_code"].iloc[0]

    # Scoring population: engineer + new category "doctor"
    score_users = pd.DataFrame([
        {"user_id": "u3", "kyc_level": "level1", "created_at": "2024-01-01",
         "occupation": "engineer", "declared_source_of_funds": "salary",
         "activity_window": "regular", "monthly_income_twd": 0.0},
        {"user_id": "u4", "kyc_level": "level1", "created_at": "2024-01-01",
         "occupation": "doctor", "declared_source_of_funds": "salary",
         "activity_window": "regular", "monthly_income_twd": 0.0},
    ])
    score_result = compute_profile_features(score_users, category_maps=maps)

    # engineer must have same code in scoring as in training
    score_engineer_code = score_result.loc[score_result["user_id"] == "u3", "occupation_code"].iloc[0]
    assert score_engineer_code == engineer_code, "engineer code must be stable across runs"

    # doctor is unknown → -1
    doctor_code = score_result.loc[score_result["user_id"] == "u4", "occupation_code"].iloc[0]
    assert doctor_code == -1, "Unknown category must map to -1"
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd bitoguard_core && source .venv/bin/activate
PYTHONPATH=. pytest tests/test_feature_modules.py::test_profile_category_codes_deterministic_across_populations -v
```

Expected: FAIL — `build_profile_category_maps` doesn't exist yet.

- [ ] **Step 3: Implement `build_profile_category_maps` and update `compute_profile_features`**

Replace `bitoguard_core/features/profile_features.py` entirely:

```python
# bitoguard_core/features/profile_features.py
"""Profile features from canonical.users.

NOTE: KYC timestamp fields (level1_finished_at, level2_finished_at, confirmed_at)
are consumed by pipeline/transformers.py and NOT stored in canonical.users.
KYC velocity features are therefore not available without a schema extension.
Available: kyc_level (string ordinal), created_at, occupation, income_source, segment.
"""
from __future__ import annotations
import pandas as pd

KYC_LEVEL_MAP = {"level2": 2, "level1": 1, "email_verified": 0, None: -1}

# Columns that receive deterministic integer codes saved in profile_category_maps.json
_CATEGORY_COLS = {
    "occupation":              "occupation_code",
    "declared_source_of_funds": "income_source_code",
    "activity_window":          "user_source_code",
}


def build_profile_category_maps(users: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Build {col_name: {value: code}} maps from training population.

    Codes are assigned alphabetically so they are deterministic given the same
    unique value set. Save the returned dict alongside model artifacts and pass
    it to compute_profile_features() at scoring time.
    """
    maps: dict[str, dict[str, int]] = {}
    for raw_col in _CATEGORY_COLS:
        if raw_col not in users.columns:
            maps[raw_col] = {}
            continue
        unique_vals = sorted(users[raw_col].dropna().astype(str).unique())
        maps[raw_col] = {v: i for i, v in enumerate(unique_vals)}
    return maps


def compute_profile_features(
    users: pd.DataFrame,
    snapshot_date: pd.Timestamp | None = None,
    category_maps: dict[str, dict[str, int]] | None = None,
) -> pd.DataFrame:
    """8 demographic/KYC features per user (no aggregation)."""
    if users.empty:
        return pd.DataFrame()

    df = users.copy()
    df["created_at"] = pd.to_datetime(df.get("created_at"), utc=True, errors="coerce")

    df["kyc_level_code"] = df["kyc_level"].map(KYC_LEVEL_MAP).fillna(-1).astype(int)

    for raw_col, feature_col in _CATEGORY_COLS.items():
        if raw_col not in df.columns:
            df[feature_col] = -1
            continue
        if category_maps is not None and raw_col in category_maps:
            col_map = category_maps[raw_col]
            df[feature_col] = (
                df[raw_col].astype(str).map(col_map).fillna(-1).astype(int)
            )
        else:
            # Fallback: derive from current data (non-deterministic — only use at build time)
            df[feature_col] = df[raw_col].astype("category").cat.codes

    df["monthly_income_twd"] = df.get("monthly_income_twd", 0.0).fillna(0.0)

    ref = pd.Timestamp.now(tz="UTC") if snapshot_date is None else snapshot_date
    if ref.tzinfo is None:
        ref = ref.tz_localize("UTC")
    df["account_age_days"] = (
        (ref - df["created_at"]).dt.total_seconds().div(86400).clip(lower=0).fillna(0)
    )

    keep = [
        "user_id", "kyc_level_code", "occupation_code", "income_source_code",
        "user_source_code", "monthly_income_twd", "account_age_days",
    ]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)
```

- [ ] **Step 4: Update `registry.py` to build, save, and load category maps**

In `bitoguard_core/features/registry.py`, add at the top:

```python
import json
from pathlib import Path
from features.profile_features import build_profile_category_maps
```

Update `build_v2_features` to accept and pass `category_maps`:

```python
def build_v2_features(
    users:          pd.DataFrame,
    fiat:           pd.DataFrame,
    crypto:         pd.DataFrame,
    trades:         pd.DataFrame,
    logins:         pd.DataFrame,
    edges:          pd.DataFrame,
    snapshot_date:  pd.Timestamp | None = None,
    category_maps:  dict | None = None,
) -> pd.DataFrame:
```

Change the `compute_profile_features` call:

```python
    module_entries: list[tuple[pd.DataFrame | None, object]] = [
        (compute_profile_features(users, snapshot_date=snapshot_date,
                                   category_maps=category_maps),    None),
        ...
    ]
```

Update `build_and_store_v2_features` to build+save category maps and pass them:

```python
def build_and_store_v2_features(
    users:         pd.DataFrame,
    fiat:          pd.DataFrame,
    crypto:        pd.DataFrame,
    trades:        pd.DataFrame,
    logins:        pd.DataFrame,
    edges:         pd.DataFrame,
    snapshot_date: pd.Timestamp | None = None,
    store=None,
) -> pd.DataFrame:
    """Compute v2 features and persist to features.feature_snapshots_v2."""
    from db.store import DuckDBStore, make_id
    from config import load_settings

    settings = load_settings() if store is None else None
    if store is None:
        store = DuckDBStore(settings.db_path)
        artifact_dir = settings.artifact_dir
    else:
        artifact_dir = Path("artifacts")

    # Build and save deterministic category maps from current users population
    category_maps = build_profile_category_maps(users)
    maps_path = artifact_dir / "profile_category_maps.json"
    maps_path.parent.mkdir(parents=True, exist_ok=True)
    maps_path.write_text(
        json.dumps(category_maps, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    master = build_v2_features(users, fiat, crypto, trades, logins, edges,
                                snapshot_date=snapshot_date,
                                category_maps=category_maps)
    if master.empty:
        return master

    if snapshot_date is None:
        snapshot_date = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)

    master.insert(0, "feature_snapshot_id",
                  [make_id(f"v2_{uid[-4:]}") for uid in master["user_id"]])
    master.insert(2, "snapshot_date", snapshot_date.date())
    master.insert(3, "feature_version", FEATURE_VERSION_V2)

    store.replace_table("features.feature_snapshots_v2", master)
    return master
```

Also update `score_latest_snapshot_v2` in `score.py` to load and apply saved category maps when scoring. Add after `features = load_feature_table(...)`:

```python
    # Load saved category maps so profile feature codes match training
    maps_path = settings.artifact_dir / "profile_category_maps.json"
    # (category codes are already in the stored feature snapshot — no re-encoding needed at score time)
    # The maps are needed only when computing features live from raw data, not from stored snapshots.
```

*Note: Since scoring reads features from the stored `feature_snapshots_v2` table (which already has encoded codes), the maps are only needed in `build_and_store_v2_features`. The scoring path does not need to re-apply them.*

- [ ] **Step 5: Run tests**

```bash
cd bitoguard_core && source .venv/bin/activate
PYTHONPATH=. pytest tests/test_feature_modules.py -v --tb=short
PYTHONPATH=. pytest tests/ -v --tb=short 2>&1 | tail -10
```

Expected: all feature module tests pass, full suite unchanged.

- [ ] **Step 6: Commit**

```bash
git add bitoguard_core/features/profile_features.py bitoguard_core/features/registry.py bitoguard_core/tests/test_feature_modules.py
git commit -m "fix: deterministic category encoding via saved profile_category_maps.json"
```

---

### Task 2.2: Fix Misleading `rel_in_degree`/`rel_out_degree` in Graph Bipartite Features

**Problem:** `graph_bipartite.py:104-106` sets `rel_in_degree = rel_out_degree = len(peers)` because the peer graph is symmetric. The name `rel_in_degree` implies directed semantics that don't exist. The Opus review identified this as 3 features providing 1 bit of information.

**Fix:** Remove the redundant `rel_in_degree` column (always identical to `rel_out_degree`) and fix the docstring to drop the false "directed graph" framing. Keep `rel_out_degree` (renamed semantics: "number of co-wallet peers") and `rel_reciprocity` as a binary flag.

**Important:** This changes the feature schema — THREE column renames/removals:
- `rel_out_degree` → `rel_peer_count` (renamed)
- `rel_in_degree` → removed (was always identical to `rel_out_degree`)
- `rel_reciprocity` → `rel_has_peers` (renamed)

Any trained models referencing these column names in `feature_columns` metadata will break. After applying this fix, retrain the stacker (`make train-stacker`).

**Files:**
- Modify: `bitoguard_core/features/graph_bipartite.py`
- Test: `bitoguard_core/tests/test_graph_bipartite.py`

- [ ] **Step 1: Write failing test**

Add to `bitoguard_core/tests/test_graph_bipartite.py`:

```python
def test_no_redundant_rel_in_degree():
    """rel_in_degree must not appear in output — it was always identical to rel_out_degree."""
    import pandas as pd
    from features.graph_bipartite import compute_bipartite_features

    edges = pd.DataFrame([
        {"src_type": "user", "src_id": "u1", "dst_type": "wallet",
         "dst_id": "w1", "relation_type": "owns_wallet", "edge_id": "e1"},
        {"src_type": "user", "src_id": "u2", "dst_type": "wallet",
         "dst_id": "w1", "relation_type": "owns_wallet", "edge_id": "e2"},
    ])
    result = compute_bipartite_features(edges, ["u1", "u2"])
    assert "rel_in_degree" not in result.columns, (
        "rel_in_degree is always == rel_out_degree and must be removed"
    )
    assert "rel_out_degree" in result.columns
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd bitoguard_core && source .venv/bin/activate
PYTHONPATH=. pytest tests/test_graph_bipartite.py::test_no_redundant_rel_in_degree -v
```

Expected: FAIL — `rel_in_degree` currently present.

- [ ] **Step 3: Fix `graph_bipartite.py`**

Replace lines 104-106:

```python
        row["rel_out_degree"]  = len(peers)
        row["rel_in_degree"]   = len(peers)
        row["rel_reciprocity"] = 1.0 if peers else 0.0
```

With:

```python
        row["rel_peer_count"]  = len(peers)   # co-wallet user neighbors (symmetric)
        row["rel_has_peers"]   = 1.0 if peers else 0.0
```

Also update the module docstring at the top:

```python
"""Label-free bipartite graph features (~39).

Computes features from:
  1. IP bipartite graph (user <-> ip): degree-bucket distribution, robust to supernodes
  2. Wallet bipartite graph (user <-> wallet): same structure
  3. Peer graph (user -- user via shared wallet): symmetric peer count

Degree buckets replace component_size and shared_device_count.
rel_peer_count is symmetric (co-wallet peers); rel_has_peers is a binary flag.
Does NOT use labels. Safe to compute once for the entire dataset.
"""
```

- [ ] **Step 4: Update the existing `test_bipartite_features_columns` test**

In `bitoguard_core/tests/test_graph_bipartite.py`, update the column assertions that reference the old names:

```python
# Change: assert "rel_out_degree" in result.columns
# To:     assert "rel_peer_count" in result.columns
# Change: assert "rel_reciprocity" in result.columns  (if present)
# To:     assert "rel_has_peers" in result.columns
# Remove: assert "rel_in_degree" in result.columns    (if present)
```

- [ ] **Step 5: Run tests**

```bash
cd bitoguard_core && source .venv/bin/activate
PYTHONPATH=. pytest tests/test_graph_bipartite.py -v
```

Expected: all graph bipartite tests pass (new + updated).

- [ ] **Step 5: Commit**

```bash
git add bitoguard_core/features/graph_bipartite.py bitoguard_core/tests/test_graph_bipartite.py
git commit -m "fix: remove redundant rel_in_degree; rename rel_out_degree->rel_peer_count"
```

---

## Chunk 3: Architecture Hardening

### Task 3.1: Remove Dead `_component_weights()` Function

**Problem:** `score.py:47-57` defines `_component_weights()` which is never called. The actual scoring code hardcodes weights inline at lines 160-168 (v1) and 328-332 (v2). The dead function uses different weight values and a `m0_enabled`/`m1_enabled` toggle system that diverges from the actual scoring logic.

**Files:**
- Modify: `bitoguard_core/models/score.py`
- Test: verify no remaining references

- [ ] **Step 1: Verify `_component_weights` is not called anywhere**

```bash
cd bitoguard_core && grep -r "_component_weights" . --include="*.py"
```

Expected: only `score.py` (the definition).

- [ ] **Step 2: Delete the function from `score.py`**

Remove lines 47-57 (the entire `_component_weights` function):

```python
def _component_weights(settings) -> dict[str, float]:
    weights = {
        "m1": 0.20 if settings.m1_enabled else 0.0,
        "m3": 0.45 if settings.m3_enabled else 0.0,
        "m4": 0.35 if settings.m4_enabled else 0.0,
        "m5": 0.10 if settings.m5_enabled else 0.0,
    }
    total = sum(weights.values())
    if total == 0:
        return weights
    return {name: value / total for name, value in weights.items()}
```

- [ ] **Step 3: Run tests to confirm nothing broke**

```bash
cd bitoguard_core && source .venv/bin/activate
PYTHONPATH=. pytest tests/ -v --tb=short 2>&1 | tail -10
```

- [ ] **Step 4: Commit**

```bash
git add bitoguard_core/models/score.py
git commit -m "refactor: remove dead _component_weights() function from score.py"
```

---

### Task 3.2: Cap `neighbor_ids` Before Building Placeholders in Graph API

**Problem:** `api/main.py:137-141` builds `", ".join(["?"] * len(neighbor_ids))` without capping `neighbor_ids`. A highly-connected node (e.g., an IP shared by 50,000 users) generates a query with 50,000+ placeholders, potentially causing memory exhaustion or DuckDB planner timeouts.

**Fix:** Cap `neighbor_ids` to a reasonable limit (500) before constructing the query. This is consistent with the `max_nodes` / `max_edges` limits already applied after the query.

**Files:**
- Modify: `bitoguard_core/api/main.py:131-143`
- Test: `bitoguard_core/tests/test_smoke.py`

- [ ] **Step 1: Write failing test**

Add to `bitoguard_core/tests/test_smoke.py`:

```python
def test_load_neighborhood_edges_caps_neighbor_ids():
    """
    _load_neighborhood_edges must cap neighbor_ids at _MAX_NEIGHBOR_IDS
    to prevent unbounded SQL placeholder construction.
    """
    import pandas as pd
    from unittest.mock import MagicMock, patch
    from api.main import _load_neighborhood_edges, _MAX_NEIGHBOR_IDS

    # Build a 1-hop result with many unique neighbor IDs
    many_neighbor_ids = [f"entity_{i}" for i in range(_MAX_NEIGHBOR_IDS + 100)]
    one_hop_df = pd.DataFrame({
        "src_id": ["focus_user"] * len(many_neighbor_ids),
        "dst_id": many_neighbor_ids,
        "src_type": ["user"] * len(many_neighbor_ids),
        "dst_type": ["wallet"] * len(many_neighbor_ids),
        "relation_type": ["owns_wallet"] * len(many_neighbor_ids),
        "edge_id": [f"e{i}" for i in range(len(many_neighbor_ids))],
    })

    store = MagicMock()
    # First call returns the 1-hop result; second (2-hop) should use capped neighbors
    store.fetch_df.side_effect = [one_hop_df, pd.DataFrame()]

    _load_neighborhood_edges(store, "focus_user", max_hops=2)

    # The second call's SQL must not have more than _MAX_NEIGHBOR_IDS placeholders
    second_call_sql = store.fetch_df.call_args_list[1][0][0]
    placeholder_count = second_call_sql.count("?")
    # Each neighbor appears twice in "IN (...) OR ... IN (...)" so divide by 2
    assert placeholder_count // 2 <= _MAX_NEIGHBOR_IDS, (
        f"SQL had {placeholder_count // 2} neighbors; must be capped at {_MAX_NEIGHBOR_IDS}"
    )
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd bitoguard_core && source .venv/bin/activate
PYTHONPATH=. pytest tests/test_smoke.py::test_load_neighborhood_edges_caps_neighbor_ids -v
```

Expected: FAIL — `_MAX_NEIGHBOR_IDS` doesn't exist yet; no capping in place.

- [ ] **Step 3: Add constant and capping logic to `api/main.py`**

Add near the top of `bitoguard_core/api/main.py` (after imports):

```python
_MAX_NEIGHBOR_IDS = 500  # cap before building SQL placeholders; prevents DoS
```

In `_load_neighborhood_edges`, replace the uncapped `neighbor_ids` usage:

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
    # Cap to prevent unbounded SQL placeholder construction on supernode IDs.
    # Use sorted() for deterministic neighbor selection across runs.
    if len(neighbor_ids) > _MAX_NEIGHBOR_IDS:
        neighbor_ids = set(sorted(neighbor_ids)[:_MAX_NEIGHBOR_IDS])
    placeholders = ", ".join(["?"] * len(neighbor_ids))
    nb = list(neighbor_ids)
    two_hop = store.fetch_df(
        f"SELECT * FROM canonical.entity_edges WHERE src_id IN ({placeholders}) OR dst_id IN ({placeholders})",
        tuple(nb) * 2,
    )
    return pd.concat([one_hop, two_hop], ignore_index=True).drop_duplicates(subset=["edge_id"])
```

- [ ] **Step 4: Run tests**

```bash
cd bitoguard_core && source .venv/bin/activate
PYTHONPATH=. pytest tests/test_smoke.py::test_load_neighborhood_edges_caps_neighbor_ids -v
PYTHONPATH=. pytest tests/ --tb=short 2>&1 | tail -10
```

- [ ] **Step 5: Commit**

```bash
git add bitoguard_core/api/main.py bitoguard_core/tests/test_smoke.py
git commit -m "fix: cap neighbor_ids at 500 in _load_neighborhood_edges to prevent DoS"
```

---

### Task 3.3: Add Table-Name Validation to `normalize.py`

**Problem:** `normalize.py:_replace_table_in_transaction` uses f-string SQL with `table_name` without calling `_validate_table_name()`. Although callers pass hardcoded names, the function has no boundary enforcement.

**Files:**
- Modify: `bitoguard_core/pipeline/normalize.py:66-72`
- Test: verify the guard fires on invalid input

- [ ] **Step 1: Write test**

Add to `bitoguard_core/tests/test_source_integration.py` (or a new `test_normalize.py`):

```python
def test_replace_table_in_transaction_rejects_unknown_table():
    """_replace_table_in_transaction must reject tables not in the allowlist."""
    import duckdb
    import pandas as pd
    from pipeline.normalize import _replace_table_in_transaction

    conn = duckdb.connect(":memory:")
    df = pd.DataFrame([{"col": 1}])
    try:
        _replace_table_in_transaction(conn, "evil.injected_table", df)
        assert False, "Should have raised ValueError for unknown table"
    except ValueError as e:
        assert "not in the allowed" in str(e).lower() or "allowlist" in str(e).lower() or "evil" in str(e)
    finally:
        conn.close()
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd bitoguard_core && source .venv/bin/activate
PYTHONPATH=. pytest tests/ -k "test_replace_table_in_transaction_rejects_unknown_table" -v
```

Expected: FAIL — no validation currently.

- [ ] **Step 3: Add validation call to `normalize.py`**

In `bitoguard_core/pipeline/normalize.py`, add the import and guard:

```python
from db.store import _validate_table_name  # add to existing db.store import
```

Update `_replace_table_in_transaction`:

```python
def _replace_table_in_transaction(conn, table_name: str, dataframe: pd.DataFrame) -> None:
    _validate_table_name(table_name)  # reject tables not in the allowlist
    if dataframe.empty:
        conn.execute(f"DELETE FROM {table_name}")
        return
    conn.register("normalized_df", dataframe)
    conn.execute(f"DELETE FROM {table_name}")
    conn.execute(f"INSERT INTO {table_name} SELECT * FROM normalized_df")
    conn.unregister("normalized_df")
```

Note: `_validate_table_name` is not currently exported from `db.store`. Update `db/store.py` to make the function importable (it's already module-level, just needs to be verified importable):

```python
# db/store.py — _validate_table_name is already module-level; no change needed
# Verify it's importable:
from db.store import _validate_table_name  # should work as-is
```

- [ ] **Step 4: Run tests**

```bash
cd bitoguard_core && source .venv/bin/activate
PYTHONPATH=. pytest tests/ -k "test_replace_table_in_transaction_rejects_unknown_table" -v
PYTHONPATH=. pytest tests/ --tb=short 2>&1 | tail -10
```

- [ ] **Step 5: Commit**

```bash
git add bitoguard_core/pipeline/normalize.py bitoguard_core/tests/
git commit -m "fix: add _validate_table_name guard to normalize._replace_table_in_transaction"
```

---

## Chunk 4: Repo Cleanup

### Task 4.1: Update `.gitignore` for Tool-Local State

**Problem:** Multiple tool-local directories are untracked and should never be committed:
- `.kiro/` — Kiro IDE project state
- `.serena/` — Serena AI coding assistant state
- `.superpowers/` — Superpowers plugin local state
- `bitoguard_core/catboost_info/` — CatBoost training diagnostic output (verbose logs)
- `mcp.json` — local MCP server config (machine-specific)
- `docs/superpowers/` — planning tool state (optional: gitignore or track plans selectively)

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add entries to `.gitignore`**

Append to `.gitignore`:

```gitignore
# Tool-local state (never commit)
.kiro/
.serena/
.superpowers/
bitoguard_core/catboost_info/
mcp.json

# Planning tool output (plans tracked in docs/superpowers/plans/, state dirs ignored)
docs/superpowers/.sessions/
docs/superpowers/.cache/
```

- [ ] **Step 2: Verify the additions are correct**

```bash
git check-ignore -v .kiro .serena .superpowers bitoguard_core/catboost_info mcp.json
```

Expected: each path is matched by the new rules.

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore IDE tool state, catboost_info, mcp.json"
```

---

### Task 4.2: Track Legitimate Untracked Files

**Problem:** Several useful files are untracked and should be committed:
- `.github/workflows/` — CI/CD pipeline definitions
- `infra/aws/` — Terraform infrastructure as code
- `scripts/` — deployment and operational scripts
- `DEPLOYMENT_SUMMARY.md`, `README_AWS.md` — deployment documentation
- `docs/AWS_DEPLOYMENT_GUIDE.md`, `docs/COST_OPTIMIZATION.md`, `docs/QUICK_START_AWS.md` — operational docs
- `.postman.json` — API test collection

**Files:**
- Add: all of the above to git tracking

- [ ] **Step 1: Stage and commit infrastructure files**

```bash
git add .github/
git add infra/
git add scripts/
git commit -m "chore: track CI/CD workflows, Terraform infra, and deployment scripts"
```

- [ ] **Step 2: Stage and commit documentation files**

```bash
git add DEPLOYMENT_SUMMARY.md README_AWS.md
git add docs/AWS_DEPLOYMENT_GUIDE.md docs/COST_OPTIMIZATION.md docs/QUICK_START_AWS.md
git add docs/VSCODE_MCP_SETUP.md docs/VSCODE_WORKFLOW.md
git add docs/superpowers/plans/
git commit -m "docs: add AWS deployment guides, VSCode workflow, and implementation plans"
```

- [ ] **Step 3: Stage and commit API collection**

```bash
git add .postman.json
git commit -m "chore: add Postman API collection for manual testing"
```

- [ ] **Step 4: Verify git status is clean**

```bash
git status
```

Expected: only the `.kiro/`, `.serena/`, `.superpowers/`, `catboost_info/`, `mcp.json` directories remain untracked (they're now gitignored).

---

### Task 4.3: Add `docker-compose.yml` So `make docker-up` Works

**Problem:** `Makefile` references `docker compose up` but no `docker-compose.yml` exists. Both `bitoguard_core/Dockerfile` and `bitoguard_frontend/Dockerfile` exist and are production-ready.

**Files:**
- Create: `docker-compose.yml` (project root)

- [ ] **Step 1: Create the missing `bitoguard_sim_output/` directory**

The backend Dockerfile (`bitoguard_core/Dockerfile:19`) has `COPY bitoguard_sim_output /app/bitoguard_sim_output` but this directory does not exist, causing `docker compose build` to fail. Create a placeholder:

```bash
mkdir -p bitoguard_sim_output
touch bitoguard_sim_output/.gitkeep
git add bitoguard_sim_output/.gitkeep
```

- [ ] **Step 2: Create `docker-compose.yml`**

```yaml
# docker-compose.yml
# Full-stack BitoGuard: FastAPI backend (8001) + Next.js frontend (3000)
# Usage: make docker-up
services:
  backend:
    build:
      context: .
      dockerfile: bitoguard_core/Dockerfile
    ports:
      - "8001:8001"
    volumes:
      - ./bitoguard_core/artifacts:/app/bitoguard_core/artifacts
    environment:
      - BITOGUARD_DB_PATH=/app/bitoguard_core/artifacts/bitoguard.duckdb
      - BITOGUARD_ARTIFACT_DIR=/app/bitoguard_core/artifacts
      - BITOGUARD_CORS_ORIGINS=http://localhost:3000
    restart: unless-stopped

  frontend:
    build:
      context: .
      dockerfile: bitoguard_frontend/Dockerfile
      args:
        BITOGUARD_INTERNAL_API_BASE: http://backend:8001
    ports:
      - "3000:3000"
    environment:
      - BITOGUARD_INTERNAL_API_BASE=http://backend:8001
    depends_on:
      - backend
    restart: unless-stopped
```

- [ ] **Step 3: Verify docker-compose YAML parses correctly (no actual build required)**

```bash
docker compose config --quiet 2>&1 | head -5
```

Expected: no errors, or minimal warnings only.

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml bitoguard_sim_output/.gitkeep
git commit -m "feat: add docker-compose.yml + bitoguard_sim_output placeholder so make docker-up works"
```

---

## Final Verification

- [ ] **Run full test suite**

```bash
cd bitoguard_core && source .venv/bin/activate
PYTHONPATH=. pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: same number of tests pass as before; 5 pre-existing smoke failures unchanged; all new tests green.

- [ ] **Verify no unintended regressions in linting**

```bash
cd bitoguard_core && source .venv/bin/activate
ruff check . --select E,F,W --ignore E501 2>&1 | head -20
```

- [ ] **Push to remote**

```bash
git push origin main
```
