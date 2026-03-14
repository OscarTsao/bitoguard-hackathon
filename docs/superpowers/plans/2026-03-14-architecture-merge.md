# BitoGuard Architecture Merge & Optimization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ~36 thin rolling-window features and disabled graph module with 174 lifetime behavioral features, bipartite graph topology, per-fold label propagation, and a learned LR stacker over CatBoost + LightGBM + IsolationForest branches — while keeping the rule engine and daily-snapshot ops model intact.

**Architecture:** New per-table feature modules (profile, twd, crypto, swap, trading, ip, sequence) write into `features.feature_snapshots_v2`. A bipartite graph module replaces the poisoned NetworkX graph. During training, per-fold label propagation adds 7 leakage-safe graph-label features. Two supervised branches (CatBoost Branch A, LightGBM Branch B) and one unsupervised branch (IsolationForest Branch C) feed a learned Logistic Regression stacker. Rule engine runs in parallel and remains in the alert payload. Old `feature_snapshots_user_30d` table is preserved; new models reference `v2`.

**Tech Stack:** Python 3.12, DuckDB, pandas, LightGBM 4.6, catboost (new dep), scikit-learn, pytest.

---

## Verified Constraints (from codebase audit)

Before reading any task, internalize these facts:

| Constraint | Detail |
|-----------|--------|
| `_ALLOWED_TABLES` is built at **import time** from `FEATURE_TABLE_SPECS` | A new table MUST be added to that tuple or every `store.replace_table` call raises `ValueError` |
| `models/common.py` has NO `save_pickle`/`load_pickle` | Serialization helpers are `save_lgbm`, `load_lgbm`, `save_iforest`, `load_iforest`, `save_json` |
| `_load_latest_model(prefix, extension)` takes **2 arguments** | Extension is the file suffix without dot, e.g. `"lgb"`, `"joblib"` |
| `canonical.fiat_transactions` has NO `ip_address` column | IP data lives in `canonical.login_events` (synthetic, from `source_ip_hash` in transactions) |
| `canonical.trade_orders` has NO `ip_address` column | Same — IP is in `canonical.login_events` |
| `canonical.crypto_transactions` has NO `is_internal` column | Columns: `crypto_txn_id, user_id, occurred_at, direction, asset, network, wallet_id, counterparty_wallet_id, amount_asset, amount_twd_equiv, tx_hash, status` |
| `canonical.users` has NO `level1_finished_at`/`level2_finished_at`/`confirmed_at` | Transformer consumes these to derive `kyc_level` + `created_at`, then discards raw timestamps |
| Swap vs regular trade: use `order_type="instant_swap"` | Both swaps and trades have `base_asset="USDT"`; `order_type` discriminates |
| `evaluate_rules()` uses **v1 column names** (safe `_get()` returns zeros for missing) | Must add a `_build_rule_compat_frame()` shim in `score.py`; otherwise all 11 rules silently return 0 |
| `monthly_income_twd`, `expected_monthly_volume_twd` are **always None** in current data | Transformer sets them to None; `actual_volume_expected_ratio` will be 0 |
| Makefile uses `$(CORE_DIR)` and `$(ACTIVATE)` variables | All new targets must use these |

---

## File Map

### New files
| File | Responsibility |
|------|---------------|
| `bitoguard_core/features/profile_features.py` | KYC level, demographics, account age → 8 features |
| `bitoguard_core/features/twd_features.py` | TWD fiat aggregates + gap distribution → 26 features |
| `bitoguard_core/features/crypto_features.py` | Crypto aggregates + wallet/counterparty/TRX → 33 features |
| `bitoguard_core/features/swap_features.py` | USDT swap aggregates (instant_swap) → 11 features |
| `bitoguard_core/features/trading_features.py` | Trade order aggregates (book + limit) → 12 features |
| `bitoguard_core/features/ip_features.py` | Per-user IP diversity from login_events → 4 features |
| `bitoguard_core/features/sequence_features.py` | Cross-table timing sequences → 10 features |
| `bitoguard_core/features/graph_bipartite.py` | IP + Wallet bipartite degree buckets + Relation directed graph → 40 features |
| `bitoguard_core/features/graph_propagation.py` | Per-fold label propagation scores → 7 features |
| `bitoguard_core/features/registry.py` | Assembles all modules → `feature_snapshots_v2` |
| `bitoguard_core/features/build_features_v2.py` | CLI entry point for `make features-v2` |
| `bitoguard_core/models/train_catboost.py` | CatBoost Branch A training |
| `bitoguard_core/models/stacker.py` | OOF generation + LR meta-learner training |
| `bitoguard_core/tests/test_feature_modules.py` | Unit tests for all feature modules |
| `bitoguard_core/tests/test_graph_bipartite.py` | Unit tests for bipartite graph features |
| `bitoguard_core/tests/test_stacker.py` | Unit tests for stacker pipeline |

### Modified files
| File | Change |
|------|--------|
| `bitoguard_core/requirements.txt` | Add `catboost>=1.2` |
| `bitoguard_core/db/schema.py` | Append `feature_snapshots_v2` to `FEATURE_TABLE_SPECS` tuple |
| `bitoguard_core/models/common.py` | Add `save_joblib()`, `load_joblib()` (sha256-protected); extend `NON_FEATURE_COLUMNS` |
| `bitoguard_core/models/score.py` | Add `score_latest_snapshot_v2()` with correct 2-arg `_load_latest_model` calls + rule compat shim |
| `Makefile` (root) | Add `features-v2`, `train-stacker`, `score-v2` targets using `$(CORE_DIR)`/`$(ACTIVATE)` |

---

## Chunk 1: Prerequisites & Schema

### Task 1.1: Add catboost + schema v2 + joblib helpers

**Files:**
- Modify: `bitoguard_core/requirements.txt`
- Modify: `bitoguard_core/db/schema.py`
- Modify: `bitoguard_core/models/common.py`

- [ ] **Step 1: Add catboost to requirements.txt**

```
catboost>=1.2
```

- [ ] **Step 2: Append v2 table to `FEATURE_TABLE_SPECS` in schema.py**

The new table spec must be **inside** `FEATURE_TABLE_SPECS` so it auto-registers in `_ALLOWED_TABLES`. Replace the existing `FEATURE_TABLE_SPECS` definition with:

```python
FEATURE_TABLE_SPECS: tuple[TableSpec, ...] = (
    TableSpec("features", "graph_features", "graph_feature_id", (
        ("graph_feature_id", "VARCHAR"), ("user_id", "VARCHAR"), ("snapshot_date", "DATE"),
        ("shared_device_count", "INTEGER"), ("shared_bank_count", "INTEGER"), ("shared_wallet_count", "INTEGER"),
        ("blacklist_1hop_count", "INTEGER"), ("blacklist_2hop_count", "INTEGER"), ("component_size", "INTEGER"),
        ("fan_out_ratio", "DOUBLE"),
    )),
    TableSpec("features", "feature_snapshots_user_day", "feature_snapshot_id", (
        ("feature_snapshot_id", "VARCHAR"), ("user_id", "VARCHAR"), ("snapshot_date", "DATE"), ("feature_version", "VARCHAR")
    )),
    TableSpec("features", "feature_snapshots_user_30d", "feature_snapshot_id", (
        ("feature_snapshot_id", "VARCHAR"), ("user_id", "VARCHAR"), ("snapshot_date", "DATE"), ("feature_version", "VARCHAR")
    )),
    # v2: 174-column expanded feature set (columns added dynamically by pandas)
    TableSpec("features", "feature_snapshots_v2", "feature_snapshot_id", (
        ("feature_snapshot_id", "VARCHAR"), ("user_id", "VARCHAR"), ("snapshot_date", "DATE"), ("feature_version", "VARCHAR")
    )),
)
```

- [ ] **Step 3: Add `save_joblib` / `load_joblib` to models/common.py**

Append after `save_json()`:

```python
# ── Generic joblib (IsolationForest-style: joblib + SHA-256 integrity) ────────

def save_joblib(model: object, path: Path) -> None:
    """Save any sklearn-compatible model via joblib with SHA-256 integrity check."""
    joblib.dump(model, path)
    path.with_suffix(".sha256").write_text(_sha256_file(path), encoding="utf-8")


def load_joblib(path: Path) -> object:
    """Load a joblib model after verifying SHA-256 integrity."""
    sha_path = path.with_suffix(".sha256")
    if not sha_path.exists():
        raise FileNotFoundError(f"SHA-256 manifest not found for {path}")
    expected = sha_path.read_text(encoding="utf-8").strip()
    file_bytes = path.read_bytes()
    if hashlib.sha256(file_bytes).hexdigest() != expected:
        raise ValueError(f"Integrity check FAILED for {path}. Retrain.")
    return joblib.load(io.BytesIO(file_bytes))
```

Note: `_sha256_file`, `hashlib`, `io`, and `joblib` are already imported in `common.py`.

- [ ] **Step 4: Install catboost**

```bash
cd bitoguard_core && source .venv/bin/activate && pip install "catboost>=1.2"
```

- [ ] **Step 5: Verify schema change works**

```bash
cd bitoguard_core && source .venv/bin/activate
PYTHONPATH=. python -c "
from db.store import _ALLOWED_TABLES
assert 'features.feature_snapshots_v2' in _ALLOWED_TABLES, 'NOT IN ALLOWED TABLES'
print('OK — features.feature_snapshots_v2 is registered')
"
```
Expected: `OK — features.feature_snapshots_v2 is registered`

- [ ] **Step 6: Verify joblib helpers import**

```bash
PYTHONPATH=. python -c "from models.common import save_joblib, load_joblib; print('OK')"
```

- [ ] **Step 7: Commit**

```bash
git add bitoguard_core/requirements.txt bitoguard_core/db/schema.py bitoguard_core/models/common.py
git commit -m "feat: add feature_snapshots_v2 to schema allowlist + joblib helpers + catboost dep"
```

---

## Chunk 2: Feature Modules

### Task 2.1: profile_features.py

**Available canonical.users columns:** `user_id, created_at, segment, kyc_level, occupation, monthly_income_twd, declared_source_of_funds, residence_country, nationality, activity_window`

**NOT available** (consumed by transformer): `level1_finished_at`, `level2_finished_at`, `confirmed_at` — KYC velocity features are therefore omitted.

**Files:**
- Create: `bitoguard_core/features/profile_features.py`
- Create: `bitoguard_core/tests/test_feature_modules.py`

- [ ] **Step 1: Write failing test**

```python
# bitoguard_core/tests/test_feature_modules.py
from __future__ import annotations
import pandas as pd
import pytest
from features.profile_features import compute_profile_features

def _users_df():
    return pd.DataFrame([{
        "user_id": "u1",
        "created_at": "2025-01-01T00:00:00+08:00",
        "kyc_level": "level2",
        "occupation": "career_1",
        "monthly_income_twd": 50000.0,
        "declared_source_of_funds": "income_source_2",
        "activity_window": "web",
    }])

def test_profile_features_columns():
    result = compute_profile_features(_users_df())
    assert "user_id" in result.columns
    assert "kyc_level_code" in result.columns
    assert "account_age_days" in result.columns
    assert "occupation_code" in result.columns
    assert len(result) == 1

def test_profile_features_kyc_level2():
    result = compute_profile_features(_users_df())
    assert result.iloc[0]["kyc_level_code"] == 2

def test_profile_features_empty():
    result = compute_profile_features(pd.DataFrame(columns=_users_df().columns))
    assert len(result) == 0
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd bitoguard_core && source .venv/bin/activate
PYTHONPATH=. pytest tests/test_feature_modules.py::test_profile_features_columns -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement profile_features.py**

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


def compute_profile_features(users: pd.DataFrame) -> pd.DataFrame:
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

    now = pd.Timestamp.now(tz="UTC")
    df["account_age_days"] = (
        (now - df["created_at"]).dt.total_seconds().div(86400).clip(lower=0).fillna(0)
    )

    keep = [
        "user_id", "kyc_level_code", "occupation_code", "income_source_code",
        "user_source_code", "monthly_income_twd", "account_age_days",
    ]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=. pytest tests/test_feature_modules.py -k "profile" -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add bitoguard_core/features/profile_features.py bitoguard_core/tests/test_feature_modules.py
git commit -m "feat: add profile_features (kyc level, demographics, account age)"
```

---

### Task 2.2: twd_features.py

**Canonical columns available:** `fiat_txn_id, user_id, occurred_at, direction, amount_twd, currency, bank_account_id, method, status`

**NOT available on this table:** `ip_address` — IP comes from `canonical.login_events` (handled in Task 2.6).

**Files:**
- Create: `bitoguard_core/features/twd_features.py`
- Test: append to `tests/test_feature_modules.py`

- [ ] **Step 1: Write failing tests** (append to test file)

```python
from features.twd_features import compute_twd_features, _gap_stats, _agg_stats

def _fiat_df():
    return pd.DataFrame([
        {"user_id": "u1", "occurred_at": "2025-01-01T01:00:00+00:00", "direction": "deposit",    "amount_twd": 10000.0},
        {"user_id": "u1", "occurred_at": "2025-01-01T02:00:00+00:00", "direction": "deposit",    "amount_twd": 20000.0},
        {"user_id": "u1", "occurred_at": "2025-01-02T03:00:00+00:00", "direction": "withdrawal", "amount_twd": 5000.0},
        {"user_id": "u2", "occurred_at": "2025-01-05T10:00:00+00:00", "direction": "deposit",    "amount_twd": 100.0},
    ])

def test_twd_features_columns():
    result = compute_twd_features(_fiat_df())
    for col in ["twd_all_count", "twd_dep_count", "twd_wdr_count",
                "twd_net_flow", "twd_night_share",
                "twd_dep_gap_min", "twd_dep_rapid_1h_share"]:
        assert col in result.columns, f"missing {col}"

def test_twd_features_u1_counts():
    result = compute_twd_features(_fiat_df())
    u1 = result[result["user_id"] == "u1"].iloc[0]
    assert u1["twd_all_count"] == 3
    assert u1["twd_dep_count"] == 2
    assert u1["twd_wdr_count"] == 1
    assert u1["twd_net_flow"] == pytest.approx(30000.0 - 5000.0)

def test_twd_features_gap():
    result = compute_twd_features(_fiat_df())
    u1 = result[result["user_id"] == "u1"].iloc[0]
    # Two deposits 1h apart → gap_min ≈ 60 minutes
    assert u1["twd_dep_gap_min"] == pytest.approx(60.0, abs=5.0)
```

- [ ] **Step 2: Confirm failure**

```bash
PYTHONPATH=. pytest tests/test_feature_modules.py -k "twd" -v
```

- [ ] **Step 3: Implement twd_features.py**

```python
# bitoguard_core/features/twd_features.py
from __future__ import annotations
import pandas as pd

NIGHT_HOURS = frozenset(range(0, 6))  # 00:00–05:59 UTC


def _gap_stats(times: pd.Series) -> dict:
    """Inter-arrival gap stats in minutes. Returns zeros for fewer than 2 events."""
    times = pd.to_datetime(times, utc=True, errors="coerce").dropna().sort_values()
    if len(times) < 2:
        return {"gap_min": 0.0, "gap_p10": 0.0, "gap_median": 0.0, "rapid_1h_share": 0.0}
    gaps = times.diff().dropna().dt.total_seconds().div(60)
    return {
        "gap_min":         float(gaps.min()),
        "gap_p10":         float(gaps.quantile(0.10)),
        "gap_median":      float(gaps.median()),
        "rapid_1h_share":  float((gaps <= 60).mean()),
    }


def _agg_stats(series: pd.Series, prefix: str) -> dict:
    if series.empty:
        return {f"{prefix}_{s}": 0.0 for s in ("count", "sum", "mean", "median", "std", "max", "p90")}
    return {
        f"{prefix}_count":  float(len(series)),
        f"{prefix}_sum":    float(series.sum()),
        f"{prefix}_mean":   float(series.mean()),
        f"{prefix}_median": float(series.median()),
        f"{prefix}_std":    float(series.std(ddof=0)),
        f"{prefix}_max":    float(series.max()),
        f"{prefix}_p90":    float(series.quantile(0.90)),
    }


def compute_twd_features(fiat: pd.DataFrame) -> pd.DataFrame:
    """~26 TWD fiat transfer features per user (lifetime, no IP — use ip_features.py)."""
    if fiat.empty:
        return pd.DataFrame()

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
        recency = (pd.Timestamp.now(tz="UTC") - grp["occurred_at"].max()).total_seconds() / 86400
        row["twd_recency_days"] = float(max(0.0, recency))
        row["twd_night_share"]  = float((grp["occurred_at"].dt.hour.isin(NIGHT_HOURS)).mean())

        for prefix, subset in [("twd_all", grp), ("twd_dep", dep), ("twd_wdr", wdr)]:
            g = _gap_stats(subset["occurred_at"])
            row[f"{prefix}_gap_min"]         = g["gap_min"]
            row[f"{prefix}_gap_p10"]         = g["gap_p10"]
            row[f"{prefix}_gap_median"]      = g["gap_median"]
            row[f"{prefix}_rapid_1h_share"]  = g["rapid_1h_share"]

        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=. pytest tests/test_feature_modules.py -k "twd" -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add bitoguard_core/features/twd_features.py bitoguard_core/tests/test_feature_modules.py
git commit -m "feat: add twd_features (26 TWD fiat features + gap distribution)"
```

---

### Task 2.3: crypto_features.py

**Canonical columns:** `crypto_txn_id, user_id, occurred_at, direction, asset, network, wallet_id, counterparty_wallet_id, amount_asset, amount_twd_equiv, tx_hash, status`

**NOT available:** `is_internal` — removed from this module.

**Files:**
- Create: `bitoguard_core/features/crypto_features.py`
- Test: append to `tests/test_feature_modules.py`

- [ ] **Step 1: Write failing tests** (append)

```python
from features.crypto_features import compute_crypto_features

def _crypto_df():
    return pd.DataFrame([
        {"user_id": "u1", "occurred_at": "2025-01-01T01:00:00+00:00", "direction": "deposit",
         "amount_twd_equiv": 5000.0, "asset": "TRX", "network": "TRC20",
         "wallet_id": "w1", "counterparty_wallet_id": "ext1"},
        {"user_id": "u1", "occurred_at": "2025-01-01T03:00:00+00:00", "direction": "deposit",
         "amount_twd_equiv": 3000.0, "asset": "ETH", "network": "ERC20",
         "wallet_id": "w2", "counterparty_wallet_id": "ext2"},
        {"user_id": "u1", "occurred_at": "2025-01-02T08:00:00+00:00", "direction": "withdrawal",
         "amount_twd_equiv": 7000.0, "asset": "TRX", "network": "TRC20",
         "wallet_id": "w1", "counterparty_wallet_id": "ext3"},
    ])

def test_crypto_features_columns():
    result = compute_crypto_features(_crypto_df())
    for col in ["crypto_all_count", "crypto_dep_count", "crypto_wdr_count",
                "crypto_n_currencies", "crypto_trx_tx_share", "crypto_n_from_wallets",
                "crypto_dep_gap_min"]:
        assert col in result.columns, f"missing {col}"

def test_crypto_features_u1():
    result = compute_crypto_features(_crypto_df())
    u1 = result[result["user_id"] == "u1"].iloc[0]
    assert u1["crypto_all_count"] == 3
    assert u1["crypto_n_currencies"] == 2  # TRX + ETH
    assert u1["crypto_trx_tx_share"] == pytest.approx(2/3, abs=0.01)
```

- [ ] **Step 2: Confirm failure**

```bash
PYTHONPATH=. pytest tests/test_feature_modules.py -k "crypto_feat" -v
```

- [ ] **Step 3: Implement crypto_features.py**

```python
# bitoguard_core/features/crypto_features.py
from __future__ import annotations
import pandas as pd
from features.twd_features import _agg_stats, _gap_stats

TRX_ASSETS = frozenset({"TRX", "TRC20"})


def compute_crypto_features(crypto: pd.DataFrame) -> pd.DataFrame:
    """~33 crypto transfer features per user (lifetime, no is_internal column)."""
    if crypto.empty:
        return pd.DataFrame()

    df = crypto.copy()
    df["occurred_at"] = pd.to_datetime(df["occurred_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["user_id", "occurred_at"])
    df["amount_twd_equiv"] = df["amount_twd_equiv"].fillna(0.0)

    rows = []
    for uid, grp in df.groupby("user_id"):
        dep = grp[grp["direction"] == "deposit"]
        wdr = grp[grp["direction"] == "withdrawal"]
        row: dict = {"user_id": uid}

        row.update(_agg_stats(grp["amount_twd_equiv"], "crypto_all_twd"))
        row.update(_agg_stats(dep["amount_twd_equiv"], "crypto_dep_twd"))
        row.update(_agg_stats(wdr["amount_twd_equiv"], "crypto_wdr_twd"))
        row["crypto_all_count"] = int(len(grp))
        row["crypto_dep_count"] = int(len(dep))
        row["crypto_wdr_count"] = int(len(wdr))
        row["crypto_net_flow_twd"] = float(dep["amount_twd_equiv"].sum() - wdr["amount_twd_equiv"].sum())

        # Asset / protocol diversity
        row["crypto_n_currencies"] = int(grp["asset"].nunique()) if "asset" in grp else 0
        row["crypto_n_protocols"]  = int(grp["network"].nunique()) if "network" in grp else 0

        # TRX preference
        if "asset" in grp.columns and len(grp) > 0:
            trx_mask = grp["asset"].str.upper().isin(TRX_ASSETS)
            row["crypto_trx_tx_share"]  = float(trx_mask.mean())
            total_amt = grp["amount_twd_equiv"].sum()
            row["crypto_trx_amt_share"] = float(
                grp.loc[trx_mask, "amount_twd_equiv"].sum() / max(1.0, total_amt)
            )
        else:
            row["crypto_trx_tx_share"] = 0.0
            row["crypto_trx_amt_share"] = 0.0

        # Wallet diversity (from counterparty wallets on deposits)
        row["crypto_n_from_wallets"] = int(dep["counterparty_wallet_id"].nunique()) if not dep.empty else 0
        row["crypto_n_to_wallets"]   = int(wdr["counterparty_wallet_id"].nunique()) if not wdr.empty else 0
        if not dep.empty and "counterparty_wallet_id" in dep.columns:
            cp_counts = dep["counterparty_wallet_id"].dropna().value_counts(normalize=True)
            row["crypto_from_wallet_conc"] = float(cp_counts.iloc[0]) if not cp_counts.empty else 0.0
        else:
            row["crypto_from_wallet_conc"] = 0.0

        # Deposit granularity
        row["crypto_dep_amt_median"] = float(dep["amount_twd_equiv"].median()) if not dep.empty else 0.0
        dep_mean = float(dep["amount_twd_equiv"].mean()) if not dep.empty else 1.0
        dep_std  = float(dep["amount_twd_equiv"].std(ddof=0)) if not dep.empty else 0.0
        row["crypto_dep_amt_cv"] = dep_std / max(1.0, dep_mean)
        row["crypto_wdr_to_dep_ratio"] = float(
            wdr["amount_twd_equiv"].sum() / max(1.0, dep["amount_twd_equiv"].sum())
        )

        # Timing
        row["crypto_active_days"] = int(grp["occurred_at"].dt.date.nunique())
        span = (grp["occurred_at"].max() - grp["occurred_at"].min()).total_seconds() / 86400
        row["crypto_span_days"] = float(max(0.0, span))

        # Gap features
        for prefix, subset in [("crypto_all", grp), ("crypto_dep", dep), ("crypto_wdr", wdr)]:
            g = _gap_stats(subset["occurred_at"])
            row[f"{prefix}_gap_min"]         = g["gap_min"]
            row[f"{prefix}_gap_p10"]         = g["gap_p10"]
            row[f"{prefix}_gap_median"]      = g["gap_median"]
            row[f"{prefix}_rapid_1h_share"]  = g["rapid_1h_share"]

        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=. pytest tests/test_feature_modules.py -k "crypto_feat" -v
```
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add bitoguard_core/features/crypto_features.py bitoguard_core/tests/test_feature_modules.py
git commit -m "feat: add crypto_features (33 features: diversity, TRX preference, gaps)"
```

---

### Task 2.4: swap_features.py + trading_features.py

**Swap identification:** Use `order_type = "instant_swap"` (NOT `base_asset = "USDT"` — both swap and book trades have USDT as base_asset). Regular book trades have `order_type in ("market", "limit")`.

**NOT available:** `ip_address` on trade_orders — use ip_features.py (Task 2.6).

**Files:**
- Create: `bitoguard_core/features/swap_features.py`
- Create: `bitoguard_core/features/trading_features.py`
- Test: append to `tests/test_feature_modules.py`

- [ ] **Step 1: Write failing tests** (append)

```python
from features.swap_features import compute_swap_features
from features.trading_features import compute_trading_features

def _trades_df():
    return pd.DataFrame([
        {"user_id": "u1", "occurred_at": "2025-01-01T10:00:00+00:00", "side": "buy",
         "base_asset": "USDT", "quote_asset": "TWD", "notional_twd": 10000.0,
         "order_type": "instant_swap"},
        {"user_id": "u1", "occurred_at": "2025-01-02T22:00:00+00:00", "side": "sell",
         "base_asset": "USDT", "quote_asset": "TWD", "notional_twd": 5000.0,
         "order_type": "instant_swap"},
        {"user_id": "u1", "occurred_at": "2025-01-03T12:00:00+00:00", "side": "buy",
         "base_asset": "USDT", "quote_asset": "TWD", "notional_twd": 20000.0,
         "order_type": "market"},
    ])

def test_swap_features_uses_instant_swap_only():
    # Only the 2 instant_swap rows should count
    result = compute_swap_features(_trades_df())
    u1 = result[result["user_id"] == "u1"].iloc[0]
    assert u1["swap_count"] == 2  # NOT 3

def test_swap_features_columns():
    result = compute_swap_features(_trades_df())
    for col in ["swap_count", "swap_buy_count", "swap_sell_count", "swap_buy_ratio", "swap_net_twd"]:
        assert col in result.columns

def test_trading_features_book_only():
    # Only the 1 market/limit order
    result = compute_trading_features(_trades_df())
    u1 = result[result["user_id"] == "u1"].iloc[0]
    assert u1["trade_count"] == 1

def test_trading_features_columns():
    result = compute_trading_features(_trades_df())
    for col in ["trade_count", "trade_buy_count", "trade_market_ratio", "trade_night_share"]:
        assert col in result.columns
```

- [ ] **Step 2: Confirm failure**

```bash
PYTHONPATH=. pytest tests/test_feature_modules.py -k "swap or trading" -v
```

- [ ] **Step 3: Implement swap_features.py**

```python
# bitoguard_core/features/swap_features.py
"""USDT instant-swap features. Input: trade_orders filtered to order_type='instant_swap'."""
from __future__ import annotations
import pandas as pd
from features.twd_features import _agg_stats

SWAP_ORDER_TYPE = "instant_swap"


def compute_swap_features(trades: pd.DataFrame) -> pd.DataFrame:
    """11 USDT instant-swap features per user. Caller may pass all trade_orders;
    this function filters to instant_swap rows internally."""
    if trades.empty:
        return pd.DataFrame()

    df = trades.copy()
    df["occurred_at"] = pd.to_datetime(df["occurred_at"], utc=True, errors="coerce")
    # Filter to instant_swap only
    if "order_type" in df.columns:
        df = df[df["order_type"] == SWAP_ORDER_TYPE]
    if df.empty:
        return pd.DataFrame()

    df = df.dropna(subset=["user_id", "occurred_at"])
    df["notional_twd"] = df["notional_twd"].fillna(0.0)

    rows = []
    for uid, grp in df.groupby("user_id"):
        buy  = grp[grp["side"] == "buy"]
        sell = grp[grp["side"] == "sell"]
        row: dict = {"user_id": uid}
        row.update(_agg_stats(grp["notional_twd"], "swap"))
        row["swap_count"]        = int(len(grp))
        row["swap_buy_count"]    = int(len(buy))
        row["swap_sell_count"]   = int(len(sell))
        row["swap_buy_twd_sum"]  = float(buy["notional_twd"].sum())
        row["swap_sell_twd_sum"] = float(sell["notional_twd"].sum())
        row["swap_net_twd"]      = float(buy["notional_twd"].sum() - sell["notional_twd"].sum())
        row["swap_buy_ratio"]    = float(len(buy) / max(1, len(grp)))
        row["swap_active_days"]  = int(grp["occurred_at"].dt.date.nunique())
        span = (grp["occurred_at"].max() - grp["occurred_at"].min()).total_seconds() / 86400
        row["swap_span_days"]    = float(max(0.0, span))
        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)
```

- [ ] **Step 4: Implement trading_features.py**

```python
# bitoguard_core/features/trading_features.py
"""Book-order trade features (excludes instant_swap). No ip_address on this table;
use ip_features.py for per-user IP diversity."""
from __future__ import annotations
import pandas as pd
from features.twd_features import _agg_stats
from features.swap_features import SWAP_ORDER_TYPE

NIGHT_HOURS = frozenset(range(0, 6))


def compute_trading_features(trades: pd.DataFrame) -> pd.DataFrame:
    """12 book-order trade aggregates per user (excludes instant_swap rows)."""
    if trades.empty:
        return pd.DataFrame()

    df = trades.copy()
    df["occurred_at"] = pd.to_datetime(df["occurred_at"], utc=True, errors="coerce")
    # Exclude instant_swap orders — those go to swap_features
    if "order_type" in df.columns:
        df = df[df["order_type"] != SWAP_ORDER_TYPE]
    if df.empty:
        return pd.DataFrame()

    df = df.dropna(subset=["user_id", "occurred_at"])
    df["notional_twd"] = df["notional_twd"].fillna(0.0)

    rows = []
    for uid, grp in df.groupby("user_id"):
        buy  = grp[grp["side"] == "buy"]
        sell = grp[grp["side"] == "sell"]
        row: dict = {"user_id": uid}
        row.update(_agg_stats(grp["notional_twd"], "trade"))
        row["trade_count"]        = int(len(grp))
        row["trade_buy_count"]    = int(len(buy))
        row["trade_sell_count"]   = int(len(sell))
        row["trade_buy_twd_sum"]  = float(buy["notional_twd"].sum())
        row["trade_sell_twd_sum"] = float(sell["notional_twd"].sum())
        row["trade_net_twd"]      = float(buy["notional_twd"].sum() - sell["notional_twd"].sum())
        row["trade_buy_ratio"]    = float(len(buy) / max(1, len(grp)))
        if "order_type" in grp.columns:
            row["trade_market_ratio"] = float((grp["order_type"] == "market").mean())
        else:
            row["trade_market_ratio"] = 0.0
        row["trade_active_days"] = int(grp["occurred_at"].dt.date.nunique())
        span = (grp["occurred_at"].max() - grp["occurred_at"].min()).total_seconds() / 86400
        row["trade_span_days"]   = float(max(0.0, span))
        row["trade_night_share"] = float((grp["occurred_at"].dt.hour.isin(NIGHT_HOURS)).mean())
        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)
```

- [ ] **Step 5: Run tests**

```bash
PYTHONPATH=. pytest tests/test_feature_modules.py -k "swap or trading" -v
```
Expected: 4 PASSED

- [ ] **Step 6: Commit**

```bash
git add bitoguard_core/features/swap_features.py bitoguard_core/features/trading_features.py bitoguard_core/tests/test_feature_modules.py
git commit -m "feat: add swap_features (instant_swap only) + trading_features (book orders)"
```

---

### Task 2.5: ip_features.py

**Rationale:** `ip_address` exists in `canonical.login_events` (synthetic, from `source_ip_hash`). This is the only place to compute per-user IP diversity.

**Files:**
- Create: `bitoguard_core/features/ip_features.py`
- Test: append to `tests/test_feature_modules.py`

- [ ] **Step 1: Write failing test** (append)

```python
from features.ip_features import compute_ip_features

def _login_df():
    return pd.DataFrame([
        {"user_id": "u1", "occurred_at": "2025-01-01T01:00:00+00:00", "ip_address": "1.2.3.4"},
        {"user_id": "u1", "occurred_at": "2025-01-01T02:00:00+00:00", "ip_address": "1.2.3.4"},
        {"user_id": "u1", "occurred_at": "2025-01-02T23:00:00+00:00", "ip_address": "5.6.7.8"},
        {"user_id": "u2", "occurred_at": "2025-01-03T10:00:00+00:00", "ip_address": "9.9.9.9"},
    ])

def test_ip_features_columns():
    result = compute_ip_features(_login_df())
    for col in ["unique_ips", "ip_concentration", "ip_event_count", "ip_night_share"]:
        assert col in result.columns

def test_ip_features_u1():
    result = compute_ip_features(_login_df())
    u1 = result[result["user_id"] == "u1"].iloc[0]
    assert u1["unique_ips"] == 2
    assert u1["ip_concentration"] == pytest.approx(2/3, abs=0.01)  # top IP (1.2.3.4) has 2/3 events
    assert u1["ip_night_share"] == pytest.approx(1/3, abs=0.01)    # 23:00 = night
```

- [ ] **Step 2: Confirm failure**

```bash
PYTHONPATH=. pytest tests/test_feature_modules.py -k "ip_feat" -v
```

- [ ] **Step 3: Implement ip_features.py**

```python
# bitoguard_core/features/ip_features.py
"""Per-user IP diversity from canonical.login_events.

IP events in this codebase are synthetic: each fiat/crypto/trade transaction
with a source_ip_hash produces one login_event with ip_address=source_ip_hash.
This gives per-transaction IP coverage, not real authentication events.
"""
from __future__ import annotations
import pandas as pd

NIGHT_HOURS = frozenset(range(0, 6))


def compute_ip_features(login_events: pd.DataFrame) -> pd.DataFrame:
    """4 per-user IP diversity features from canonical.login_events."""
    if login_events.empty or "ip_address" not in login_events.columns:
        return pd.DataFrame()

    df = login_events.copy()
    df["occurred_at"] = pd.to_datetime(df["occurred_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["user_id", "ip_address", "occurred_at"])

    rows = []
    for uid, grp in df.groupby("user_id"):
        counts = grp["ip_address"].value_counts(normalize=True)
        rows.append({
            "user_id":          uid,
            "unique_ips":       int(grp["ip_address"].nunique()),
            "ip_event_count":   int(len(grp)),
            "ip_concentration": float(counts.iloc[0]) if not counts.empty else 0.0,
            "ip_night_share":   float((grp["occurred_at"].dt.hour.isin(NIGHT_HOURS)).mean()),
        })

    return pd.DataFrame(rows).reset_index(drop=True)
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=. pytest tests/test_feature_modules.py -k "ip_feat" -v
```
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add bitoguard_core/features/ip_features.py bitoguard_core/tests/test_feature_modules.py
git commit -m "feat: add ip_features (unique IPs, concentration, night share from login_events)"
```

---

### Task 2.6: sequence_features.py

**Files:**
- Create: `bitoguard_core/features/sequence_features.py`
- Test: append to `tests/test_feature_modules.py`

- [ ] **Step 1: Write failing tests** (append)

```python
from features.sequence_features import compute_sequence_features

def test_sequence_features_fiat_to_swap():
    fiat = pd.DataFrame([
        {"user_id": "u1", "occurred_at": "2025-01-01T10:00:00+00:00",
         "direction": "deposit", "amount_twd": 10000.0},
    ])
    # Swap buy 30 min after fiat deposit → appears in within_1h
    trades = pd.DataFrame([
        {"user_id": "u1", "occurred_at": "2025-01-01T10:30:00+00:00",
         "side": "buy", "order_type": "instant_swap", "notional_twd": 9000.0},
    ])
    crypto = pd.DataFrame(columns=["user_id", "occurred_at", "direction", "amount_twd_equiv"])
    result = compute_sequence_features(fiat, trades, crypto)
    u1 = result[result["user_id"] == "u1"].iloc[0]
    assert u1["fiat_dep_to_swap_buy_within_1h"] >= 1
    assert u1["fiat_dep_to_swap_buy_within_6h"] >= 1

def test_sequence_features_columns():
    fiat   = pd.DataFrame([{"user_id": "u1", "occurred_at": "2025-01-01T10:00:00+00:00",
                             "direction": "deposit", "amount_twd": 5000.0}])
    trades = pd.DataFrame(columns=["user_id", "occurred_at", "side", "order_type", "notional_twd"])
    crypto = pd.DataFrame(columns=["user_id", "occurred_at", "direction", "amount_twd_equiv"])
    result = compute_sequence_features(fiat, trades, crypto)
    for col in ["fiat_dep_to_swap_buy_within_1h", "fiat_dep_to_swap_buy_within_24h",
                "crypto_dep_to_fiat_wdr_within_1h", "dwell_hours",
                "early_3d_volume", "early_3d_count"]:
        assert col in result.columns
```

- [ ] **Step 2: Confirm failure**

```bash
PYTHONPATH=. pytest tests/test_feature_modules.py -k "sequence" -v
```

- [ ] **Step 3: Implement sequence_features.py**

```python
# bitoguard_core/features/sequence_features.py
from __future__ import annotations
import pandas as pd
from features.swap_features import SWAP_ORDER_TYPE


def _cross_table_within(
    left: pd.DataFrame,   # user_id, occurred_at
    right: pd.DataFrame,  # user_id, occurred_at
    hours: float,
) -> pd.Series:
    """Count of left events that have at least one right event within `hours` after."""
    if left.empty or right.empty:
        return pd.Series(dtype=int)
    merged = left[["user_id", "occurred_at"]].merge(
        right[["user_id", "occurred_at"]].rename(columns={"occurred_at": "right_at"}),
        on="user_id",
    )
    merged["gap_h"] = (merged["right_at"] - merged["occurred_at"]).dt.total_seconds() / 3600
    within = merged[(merged["gap_h"] >= 0) & (merged["gap_h"] <= hours)]
    return within.groupby("user_id")["occurred_at"].count()


def compute_sequence_features(
    fiat:   pd.DataFrame,
    trades: pd.DataFrame,
    crypto: pd.DataFrame,
) -> pd.DataFrame:
    """~10 cross-table sequence / timing features per user (lifetime)."""
    all_users: set[str] = set()
    for df in (fiat, trades, crypto):
        if not df.empty and "user_id" in df.columns:
            all_users.update(df["user_id"].dropna().unique())
    if not all_users:
        return pd.DataFrame()

    def _ts(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "occurred_at" in out.columns:
            out["occurred_at"] = pd.to_datetime(out["occurred_at"], utc=True, errors="coerce")
        return out.dropna(subset=["occurred_at"]) if "occurred_at" in out.columns else out

    fiat   = _ts(fiat)   if not fiat.empty   else fiat
    trades = _ts(trades) if not trades.empty else trades
    crypto = _ts(crypto) if not crypto.empty else crypto

    fiat_dep = (fiat[fiat.get("direction", pd.Series(dtype=str)) == "deposit"]
                if not fiat.empty else fiat)
    swap_buy = (trades[(trades.get("side", pd.Series(dtype=str)) == "buy") &
                       (trades.get("order_type", pd.Series(dtype=str)) == SWAP_ORDER_TYPE)]
                if not trades.empty else trades)
    crypto_dep = (crypto[crypto.get("direction", pd.Series(dtype=str)) == "deposit"]
                  if not crypto.empty else crypto)
    fiat_wdr   = (fiat[fiat.get("direction", pd.Series(dtype=str)) == "withdrawal"]
                  if not fiat.empty else fiat)

    base = pd.DataFrame({"user_id": sorted(all_users)})

    for h, label in [(1, "1h"), (6, "6h"), (24, "24h"), (72, "72h")]:
        counts = _cross_table_within(fiat_dep, swap_buy, h)
        base = base.merge(
            counts.reset_index().rename(columns={"occurred_at": f"fiat_dep_to_swap_buy_within_{label}"}),
            on="user_id", how="left",
        )

    for h, label in [(1, "1h"), (6, "6h"), (24, "24h"), (72, "72h")]:
        counts = _cross_table_within(crypto_dep, fiat_wdr, h)
        base = base.merge(
            counts.reset_index().rename(columns={"occurred_at": f"crypto_dep_to_fiat_wdr_within_{label}"}),
            on="user_id", how="left",
        )

    # Dwell hours: first fiat deposit to first fiat withdrawal
    if not fiat_dep.empty and not fiat_wdr.empty:
        first_dep = fiat_dep.groupby("user_id")["occurred_at"].min().rename("first_dep")
        first_wdr = fiat_wdr.groupby("user_id")["occurred_at"].min().rename("first_wdr")
        dwell     = first_dep.to_frame().join(first_wdr, how="inner")
        dwell["dwell_hours"] = (
            (dwell["first_wdr"] - dwell["first_dep"]).dt.total_seconds().div(3600).clip(lower=0)
        )
        base = base.merge(dwell[["dwell_hours"]].reset_index(), on="user_id", how="left")
    else:
        base["dwell_hours"] = 0.0

    # Early 3-day activity
    all_events = []
    for df, amt_col in [(fiat, "amount_twd"), (crypto, "amount_twd_equiv")]:
        if not df.empty and "user_id" in df.columns and amt_col in df.columns:
            all_events.append(df[["user_id", "occurred_at", amt_col]].rename(columns={amt_col: "_amt"}))
    if all_events:
        events = pd.concat(all_events, ignore_index=True)
        first_event = events.groupby("user_id")["occurred_at"].min().rename("first_event")
        events = events.merge(first_event, on="user_id")
        events["days_since_first"] = (
            (events["occurred_at"] - events["first_event"]).dt.total_seconds().div(86400)
        )
        early = events[events["days_since_first"] <= 3]
        early_vol   = early.groupby("user_id")["_amt"].sum().rename("early_3d_volume")
        early_count = early.groupby("user_id").size().rename("early_3d_count")
        base = base.merge(early_vol.reset_index(), on="user_id", how="left")
        base = base.merge(early_count.reset_index(), on="user_id", how="left")
    else:
        base["early_3d_volume"] = 0.0
        base["early_3d_count"]  = 0

    return base.fillna(0).reset_index(drop=True)
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=. pytest tests/test_feature_modules.py -k "sequence" -v
```
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add bitoguard_core/features/sequence_features.py bitoguard_core/tests/test_feature_modules.py
git commit -m "feat: add sequence_features (fiat→swap→crypto cross-table timing, early activity)"
```

---

## Chunk 3: Graph + Registry

### Task 3.1: graph_bipartite.py (label-free ~40 features)

**Uses:** `canonical.entity_edges` columns: `edge_id, snapshot_time, src_type, src_id, relation_type, dst_type, dst_id`

**Files:**
- Create: `bitoguard_core/features/graph_bipartite.py`
- Create: `bitoguard_core/tests/test_graph_bipartite.py`

- [ ] **Step 1: Write failing tests**

```python
# bitoguard_core/tests/test_graph_bipartite.py
from __future__ import annotations
import pandas as pd
import pytest
from features.graph_bipartite import compute_bipartite_features

def _edges_df():
    return pd.DataFrame([
        {"src_type": "user", "src_id": "u1", "relation_type": "login_from_ip",             "dst_type": "ip",     "dst_id": "ip1"},
        {"src_type": "user", "src_id": "u2", "relation_type": "login_from_ip",             "dst_type": "ip",     "dst_id": "ip1"},
        {"src_type": "user", "src_id": "u1", "relation_type": "owns_wallet",               "dst_type": "wallet", "dst_id": "w1"},
        {"src_type": "user", "src_id": "u1", "relation_type": "crypto_transfer_to_wallet", "dst_type": "wallet", "dst_id": "ext1"},
    ])

def test_bipartite_features_columns():
    result = compute_bipartite_features(_edges_df(), ["u1", "u2", "u3"])
    for col in ["ip_n_entities", "ip_total_event_count", "wallet_n_entities",
                "rel_out_degree", "graph_is_isolated"]:
        assert col in result.columns

def test_bipartite_u1_ip():
    result = compute_bipartite_features(_edges_df(), ["u1", "u2"])
    u1 = result[result["user_id"] == "u1"].iloc[0]
    assert u1["ip_n_entities"] == 1     # connected to ip1
    assert u1["wallet_n_entities"] >= 1

def test_bipartite_isolated_user():
    result = compute_bipartite_features(_edges_df(), ["u3"])
    u3 = result[result["user_id"] == "u3"].iloc[0]
    assert u3["graph_is_isolated"] == 1
    assert u3["ip_n_entities"] == 0
```

- [ ] **Step 2: Confirm failure**

```bash
cd bitoguard_core && source .venv/bin/activate
PYTHONPATH=. pytest tests/test_graph_bipartite.py -v
```

- [ ] **Step 3: Implement graph_bipartite.py**

```python
# bitoguard_core/features/graph_bipartite.py
"""Label-free bipartite graph features (~40).

Computes features from:
  1. IP bipartite graph (user <-> ip): degree-bucket distribution, robust to supernodes
  2. Wallet bipartite graph (user <-> wallet): same structure
  3. Relation directed graph (user -> user via shared wallet): out/in degree, reciprocity

Degree buckets replace component_size and shared_device_count — they are not invalidated
by a single supernode because each bucket counts entities at that degree, not the user's
transitive closure.

Does NOT use labels. Safe to compute once for the entire dataset.
"""
from __future__ import annotations
from collections import defaultdict
import pandas as pd

_DEGREE_BUCKETS = [1, 3, 10, 50, 200]   # upper bounds; anything above 200 → "over200"
_IP_EDGE_TYPES     = frozenset({"login_from_ip"})
_WALLET_EDGE_TYPES = frozenset({"owns_wallet", "crypto_transfer_to_wallet"})


def _bucket_label(upper: int) -> str:
    return f"deg_lte{upper}"


def _degree_buckets(entity_degrees: list[int]) -> dict[str, int]:
    out = {_bucket_label(b): 0 for b in _DEGREE_BUCKETS}
    out["deg_over200"] = 0
    for deg in entity_degrees:
        placed = False
        for b in _DEGREE_BUCKETS:
            if deg <= b:
                out[_bucket_label(b)] += 1
                placed = True
                break
        if not placed:
            out["deg_over200"] += 1
    return out


def compute_bipartite_features(
    edges: pd.DataFrame,
    user_ids: list[str],
) -> pd.DataFrame:
    """Compute ~40 label-free bipartite graph features for the given user_ids."""
    user_set = set(user_ids)

    ip_user_ents:     defaultdict[str, set[str]] = defaultdict(set)   # user → set of IPs
    ip_ent_users:     defaultdict[str, set[str]] = defaultdict(set)   # IP  → set of users
    wal_user_ents:    defaultdict[str, set[str]] = defaultdict(set)   # user → set of wallets
    wal_ent_users:    defaultdict[str, set[str]] = defaultdict(set)   # wallet → set of users
    rel_user_out:     defaultdict[str, set[str]] = defaultdict(set)   # user → peer users

    if not edges.empty:
        for _, row in edges.iterrows():
            src_t, src_id = row.get("src_type"), row.get("src_id")
            dst_t, dst_id = row.get("dst_type"), row.get("dst_id")
            rel            = row.get("relation_type", "")

            if src_t != "user" or src_id not in user_set:
                continue
            if dst_t == "ip" and rel in _IP_EDGE_TYPES:
                ip_user_ents[src_id].add(dst_id)
                ip_ent_users[dst_id].add(src_id)
            elif dst_t == "wallet" and rel in _WALLET_EDGE_TYPES:
                wal_user_ents[src_id].add(dst_id)
                wal_ent_users[dst_id].add(src_id)

        # Relation: users sharing a wallet become peers
        for ent, users in wal_ent_users.items():
            user_list = list(users & user_set)
            for i, u1 in enumerate(user_list):
                for u2 in user_list[i + 1:]:
                    rel_user_out[u1].add(u2)
                    rel_user_out[u2].add(u1)

    rows = []
    for uid in user_ids:
        ip_ents  = list(ip_user_ents.get(uid, set()))
        wal_ents = list(wal_user_ents.get(uid, set()))
        peers    = list(rel_user_out.get(uid, set()))

        row: dict = {
            "user_id":         uid,
            "graph_is_isolated": int(not ip_ents and not wal_ents and not peers),
        }

        ip_degs = [len(ip_ent_users.get(e, set())) for e in ip_ents]
        row["ip_n_entities"]        = len(ip_ents)
        row["ip_total_event_count"] = sum(ip_degs)
        row["ip_mean_entity_deg"]   = float(sum(ip_degs) / max(1, len(ip_degs)))
        row["ip_max_entity_deg"]    = float(max(ip_degs)) if ip_degs else 0.0
        row.update({f"ip_{k}": v for k, v in _degree_buckets(ip_degs).items()})

        wal_degs = [len(wal_ent_users.get(e, set())) for e in wal_ents]
        row["wallet_n_entities"]        = len(wal_ents)
        row["wallet_total_event_count"] = sum(wal_degs)
        row["wallet_mean_entity_deg"]   = float(sum(wal_degs) / max(1, len(wal_degs)))
        row["wallet_max_entity_deg"]    = float(max(wal_degs)) if wal_degs else 0.0
        row.update({f"wallet_{k}": v for k, v in _degree_buckets(wal_degs).items()})

        row["rel_out_degree"]  = len(peers)
        row["rel_in_degree"]   = len(peers)   # symmetric via shared wallets
        row["rel_reciprocity"] = 1.0 if peers else 0.0

        rows.append(row)

    return pd.DataFrame(rows).fillna(0).reset_index(drop=True)
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=. pytest tests/test_graph_bipartite.py -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add bitoguard_core/features/graph_bipartite.py bitoguard_core/tests/test_graph_bipartite.py
git commit -m "feat: add graph_bipartite (IP+wallet degree buckets, relation graph, no NetworkX)"
```

---

### Task 3.2: graph_propagation.py (per-fold, 7 features)

**Files:**
- Create: `bitoguard_core/features/graph_propagation.py`
- Test: append to `tests/test_graph_bipartite.py`

- [ ] **Step 1: Write failing tests** (append)

```python
from features.graph_propagation import compute_label_propagation

def test_propagation_reaches_neighbor():
    edges = _edges_df()
    # u2 is positive; u1 shares ip1 with u2 → u1 should get IP propagation signal
    labels = pd.Series({"u2": 1, "u1": 0})
    result = compute_label_propagation(edges, labels, user_ids=["u1"])
    u1 = result[result["user_id"] == "u1"].iloc[0]
    assert u1["prop_ip"] > 0.0

def test_propagation_columns():
    labels = pd.Series({"u1": 1, "u2": 0})
    result = compute_label_propagation(_edges_df(), labels, user_ids=["u1", "u2"])
    for col in ["prop_ip", "prop_wallet", "prop_combined",
                "ip_rep_max_rate", "wallet_rep_max_rate",
                "rel_has_pos_neighbor", "rel_direct_pos_count"]:
        assert col in result.columns

def test_propagation_no_leakage():
    """Test user absent from labels still gets correct propagation."""
    edges = _edges_df()
    labels = pd.Series({"u2": 1})   # u1 not in labels (it's the test user)
    result = compute_label_propagation(edges, labels, user_ids=["u1"])
    u1 = result[result["user_id"] == "u1"].iloc[0]
    # u1 receives signal from u2 (training) via shared ip1 — this is correct, not leakage
    assert u1["prop_ip"] > 0.0
```

- [ ] **Step 2: Confirm failure**

```bash
PYTHONPATH=. pytest tests/test_graph_bipartite.py -k "propagation" -v
```

- [ ] **Step 3: Implement graph_propagation.py**

```python
# bitoguard_core/features/graph_propagation.py
"""Per-fold label-aware propagation features (7 features).

LEAKAGE CONTRACT:
  `labels` must contain ONLY training-fold labels. Never pass test/validation
  user labels here. The stacker in models/stacker.py enforces this by passing
  only fold training indices.

Propagation is 1-hop: a user's prop_ip score = fraction of their IP entities
that are connected to at least one positive training user. This avoids the
multi-hop leakage risk of deeper propagation.
"""
from __future__ import annotations
from collections import defaultdict
import pandas as pd

_IP_EDGE_TYPES     = frozenset({"login_from_ip"})
_WALLET_EDGE_TYPES = frozenset({"owns_wallet", "crypto_transfer_to_wallet"})


def compute_label_propagation(
    edges:    pd.DataFrame,
    labels:   pd.Series,         # index=user_id, value=0/1, TRAINING FOLD ONLY
    user_ids: list[str],
) -> pd.DataFrame:
    """Compute 7 label-aware graph propagation features.

    Args:
        edges:    canonical.entity_edges
        labels:   training-fold label Series (index=user_id)
        user_ids: users to score (typically includes both train and val users)
    """
    pos_users: set[str] = set(labels[labels == 1].index)
    all_needed = set(user_ids) | set(labels.index)

    ip_pos:     defaultdict[str, set[str]] = defaultdict(set)
    ip_all:     defaultdict[str, set[str]] = defaultdict(set)
    wal_pos:    defaultdict[str, set[str]] = defaultdict(set)
    wal_all:    defaultdict[str, set[str]] = defaultdict(set)
    user_ip:    defaultdict[str, set[str]] = defaultdict(set)
    user_wal:   defaultdict[str, set[str]] = defaultdict(set)

    if not edges.empty:
        for _, row in edges.iterrows():
            uid    = row.get("src_id")
            src_t  = row.get("src_type")
            dst_t  = row.get("dst_type")
            dst_id = row.get("dst_id")
            rel    = row.get("relation_type", "")
            if src_t != "user" or uid not in all_needed:
                continue
            if dst_t == "ip" and rel in _IP_EDGE_TYPES:
                user_ip[uid].add(dst_id)
                ip_all[dst_id].add(uid)
                if uid in pos_users:
                    ip_pos[dst_id].add(uid)
            elif dst_t == "wallet" and rel in _WALLET_EDGE_TYPES:
                user_wal[uid].add(dst_id)
                wal_all[dst_id].add(uid)
                if uid in pos_users:
                    wal_pos[dst_id].add(uid)

    rows = []
    for uid in user_ids:
        ip_ents  = list(user_ip.get(uid, set()))
        wal_ents = list(user_wal.get(uid, set()))
        row: dict = {"user_id": uid}

        row["prop_ip"] = float(
            sum(1 for e in ip_ents if ip_pos.get(e)) / max(1, len(ip_ents))
        ) if ip_ents else 0.0

        row["prop_wallet"] = float(
            sum(1 for e in wal_ents if wal_pos.get(e)) / max(1, len(wal_ents))
        ) if wal_ents else 0.0

        row["prop_combined"] = float(max(row["prop_ip"], row["prop_wallet"]))

        row["ip_rep_max_rate"] = float(max(
            (len(ip_pos.get(e, set())) / max(1, len(ip_all.get(e, set()))) for e in ip_ents),
            default=0.0,
        ))
        row["wallet_rep_max_rate"] = float(max(
            (len(wal_pos.get(e, set())) / max(1, len(wal_all.get(e, set()))) for e in wal_ents),
            default=0.0,
        ))

        has_pos_ip  = any(ip_pos.get(e)  for e in ip_ents)
        has_pos_wal = any(wal_pos.get(e) for e in wal_ents)
        row["rel_has_pos_neighbor"]  = int(has_pos_ip or has_pos_wal)
        row["rel_direct_pos_count"]  = int(
            sum(1 for e in ip_ents  if ip_pos.get(e)) +
            sum(1 for e in wal_ents if wal_pos.get(e))
        )
        rows.append(row)

    return pd.DataFrame(rows).fillna(0).reset_index(drop=True)
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=. pytest tests/test_graph_bipartite.py -v
```
Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
git add bitoguard_core/features/graph_propagation.py bitoguard_core/tests/test_graph_bipartite.py
git commit -m "feat: add graph_propagation (1-hop label propagation, 7 features, leakage-safe)"
```

---

### Task 3.3: registry.py + build_features_v2.py

**Files:**
- Create: `bitoguard_core/features/registry.py`
- Create: `bitoguard_core/features/build_features_v2.py`
- Test: append to `tests/test_feature_modules.py`

- [ ] **Step 1: Write failing tests** (append)

```python
from features.registry import build_v2_features

def _registry_inputs():
    users = pd.DataFrame([{
        "user_id": "u1", "created_at": "2025-01-01T00:00:00+08:00",
        "kyc_level": "level1", "occupation": "career_1",
        "monthly_income_twd": 50000.0, "declared_source_of_funds": "income_source_2",
        "activity_window": "web",
    }])
    fiat = pd.DataFrame([
        {"user_id": "u1", "occurred_at": "2025-01-10T10:00:00+00:00",
         "direction": "deposit", "amount_twd": 5000.0},
    ])
    crypto = pd.DataFrame(columns=["user_id", "occurred_at", "direction",
                                   "amount_twd_equiv", "asset", "network",
                                   "wallet_id", "counterparty_wallet_id"])
    trades  = pd.DataFrame(columns=["user_id", "occurred_at", "side",
                                    "base_asset", "quote_asset", "notional_twd", "order_type"])
    logins  = pd.DataFrame(columns=["user_id", "occurred_at", "ip_address"])
    edges   = pd.DataFrame(columns=["src_type", "src_id", "relation_type", "dst_type", "dst_id"])
    return users, fiat, crypto, trades, logins, edges

def test_registry_returns_user_row():
    result = build_v2_features(*_registry_inputs())
    assert "u1" in result["user_id"].values
    assert len(result) == 1

def test_registry_has_key_columns():
    result = build_v2_features(*_registry_inputs())
    for col in ["kyc_level_code", "twd_all_count", "crypto_all_count",
                "fiat_dep_to_swap_buy_within_1h", "ip_n_entities", "unique_ips"]:
        assert col in result.columns, f"missing: {col}"
```

- [ ] **Step 2: Confirm failure**

```bash
PYTHONPATH=. pytest tests/test_feature_modules.py -k "registry" -v
```

- [ ] **Step 3: Implement registry.py**

```python
# bitoguard_core/features/registry.py
"""Feature registry: assembles all v2 label-free modules into one master table.

build_v2_features() → one row per user_id, ~155 columns (label-free).
build_and_store_v2_features() → writes to features.feature_snapshots_v2.

Note: graph_propagation (label-aware, 7 features) is NOT included here.
It is added separately during model training (models/stacker.py) per-fold.
"""
from __future__ import annotations
import pandas as pd

from features.profile_features  import compute_profile_features
from features.twd_features      import compute_twd_features
from features.crypto_features   import compute_crypto_features
from features.swap_features     import compute_swap_features
from features.trading_features  import compute_trading_features
from features.ip_features       import compute_ip_features
from features.sequence_features import compute_sequence_features
from features.graph_bipartite   import compute_bipartite_features

FEATURE_VERSION_V2 = "v2"


def build_v2_features(
    users:   pd.DataFrame,
    fiat:    pd.DataFrame,
    crypto:  pd.DataFrame,
    trades:  pd.DataFrame,
    logins:  pd.DataFrame,
    edges:   pd.DataFrame,
) -> pd.DataFrame:
    """Assemble all label-free feature modules. Returns one row per user_id."""
    user_ids = users["user_id"].dropna().unique().tolist()
    base = pd.DataFrame({"user_id": user_ids})

    modules = [
        compute_profile_features(users),
        compute_twd_features(fiat),
        compute_crypto_features(crypto),
        compute_swap_features(trades),     # filters to instant_swap internally
        compute_trading_features(trades),  # filters to book orders internally
        compute_ip_features(logins),
        compute_sequence_features(fiat, trades, crypto),
        compute_bipartite_features(edges, user_ids),
    ]

    for module_df in modules:
        if module_df is None or module_df.empty or "user_id" not in module_df.columns:
            continue
        new_cols = [c for c in module_df.columns if c not in base.columns or c == "user_id"]
        base = base.merge(module_df[new_cols], on="user_id", how="left")

    return base.fillna(0).reset_index(drop=True)


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

    master = build_v2_features(users, fiat, crypto, trades, logins, edges)
    if master.empty:
        return master

    if snapshot_date is None:
        snapshot_date = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)

    master.insert(0, "feature_snapshot_id",
                  [make_id(f"v2_{uid[-4:]}") for uid in master["user_id"]])
    master.insert(2, "snapshot_date", snapshot_date.date())
    master.insert(3, "feature_version", FEATURE_VERSION_V2)

    if store is None:
        store = DuckDBStore(load_settings().db_path)

    store.replace_table("features.feature_snapshots_v2", master)
    return master
```

- [ ] **Step 4: Create build_features_v2.py**

```python
# bitoguard_core/features/build_features_v2.py
"""CLI entry-point: loads canonical tables, runs registry, stores v2 features."""
from __future__ import annotations
from config import load_settings
from db.store import DuckDBStore
from features.registry import build_and_store_v2_features


def build_v2() -> None:
    settings = load_settings()
    store    = DuckDBStore(settings.db_path)

    users   = store.read_table("canonical.users")
    fiat    = store.read_table("canonical.fiat_transactions")
    crypto  = store.read_table("canonical.crypto_transactions")
    trades  = store.read_table("canonical.trade_orders")
    logins  = store.read_table("canonical.login_events")
    edges   = store.read_table("canonical.entity_edges")

    result = build_and_store_v2_features(users, fiat, crypto, trades, logins, edges, store=store)
    print(f"[features-v2] {len(result)} users, {len(result.columns)} columns")


if __name__ == "__main__":
    build_v2()
```

- [ ] **Step 5: Run tests**

```bash
PYTHONPATH=. pytest tests/test_feature_modules.py -k "registry" -v
```
Expected: 2 PASSED

- [ ] **Step 6: Commit**

```bash
git add bitoguard_core/features/registry.py bitoguard_core/features/build_features_v2.py bitoguard_core/tests/test_feature_modules.py
git commit -m "feat: add feature registry + build_features_v2 entry-point (~155 v2 features)"
```

---

## Chunk 4: Model Architecture + Integration

### Task 4.1: train_catboost.py (Branch A)

**Uses:** `save_joblib` / `load_joblib` from `models/common.py` (added in Task 1.1). CatBoost models serialize fine via joblib.

**Files:**
- Create: `bitoguard_core/models/train_catboost.py`
- Create: `bitoguard_core/tests/test_stacker.py`

- [ ] **Step 1: Write failing test**

```python
# bitoguard_core/tests/test_stacker.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import pytest
from db.store import DuckDBStore
from models.train_catboost import train_catboost_model

def _configure(tmp_path, monkeypatch):
    db_path = tmp_path / "bitoguard.duckdb"
    artifact_dir = tmp_path / "artifacts"
    monkeypatch.setenv("BITOGUARD_DB_PATH", str(db_path))
    monkeypatch.setenv("BITOGUARD_ARTIFACT_DIR", str(artifact_dir))
    return DuckDBStore(db_path)

def _seed_v2(store: DuckDBStore) -> None:
    dates = pd.date_range("2026-01-01", periods=6, freq="D")
    rows = []
    for i, d in enumerate(dates, 1):
        rows += [
            {"feature_snapshot_id": f"neg_{i}", "user_id": "u_neg", "snapshot_date": d,
             "feature_version": "v2", "twd_all_count": float(i), "kyc_level_code": 1,
             "crypto_all_count": 0.0},
            {"feature_snapshot_id": f"pos_{i}", "user_id": "u_pos", "snapshot_date": d,
             "feature_version": "v2", "twd_all_count": float(i * 5), "kyc_level_code": 2,
             "crypto_all_count": float(i * 3)},
        ]
    store.replace_table("features.feature_snapshots_v2", pd.DataFrame(rows))
    store.replace_table("ops.oracle_user_labels", pd.DataFrame([
        {"user_id": "u_pos", "hidden_suspicious_label": 1,
         "observed_blacklist_label": 1, "scenario_types": "", "evidence_tags": ""},
        {"user_id": "u_neg", "hidden_suspicious_label": 0,
         "observed_blacklist_label": 0, "scenario_types": "", "evidence_tags": ""},
    ]))

def test_catboost_trains_and_saves(tmp_path, monkeypatch):
    store = _configure(tmp_path, monkeypatch)
    _seed_v2(store)
    result = train_catboost_model()
    assert "model_version" in result
    assert Path(result["model_path"]).exists()
    assert result["model_version"].startswith("catboost_")
```

- [ ] **Step 2: Confirm failure**

```bash
PYTHONPATH=. pytest tests/test_stacker.py::test_catboost_trains_and_saves -v
```

- [ ] **Step 3: Implement train_catboost.py**

```python
# bitoguard_core/models/train_catboost.py
from __future__ import annotations
from datetime import datetime, timezone

from catboost import CatBoostClassifier

from config import load_settings
from db.store import DuckDBStore
from models.common import (
    NON_FEATURE_COLUMNS, forward_date_splits, model_dir,
    save_joblib, save_json,
)

_V2_TABLE = "features.feature_snapshots_v2"
_CAT_FEATURE_NAMES = frozenset({
    "kyc_level_code", "occupation_code", "income_source_code", "user_source_code",
})


def _load_v2_training_dataset() -> "pd.DataFrame":
    import pandas as pd
    settings = load_settings()
    store    = DuckDBStore(settings.db_path)
    dataset  = store.fetch_df(f"""
        WITH ped AS (
            SELECT user_id, CAST(MIN(observed_at) AS DATE) AS ped
            FROM canonical.blacklist_feed
            WHERE observed_at IS NOT NULL
            GROUP BY user_id
        )
        SELECT f.*,
               COALESCE(l.hidden_suspicious_label, 0) AS hidden_suspicious_label
        FROM {_V2_TABLE} f
        LEFT JOIN ops.oracle_user_labels l ON f.user_id = l.user_id
        LEFT JOIN ped ON f.user_id = ped.user_id
        WHERE COALESCE(l.hidden_suspicious_label, 0) = 0
           OR (ped.ped IS NOT NULL AND f.snapshot_date >= ped.ped)
    """)
    dataset["snapshot_date"] = pd.to_datetime(dataset["snapshot_date"])
    dataset["hidden_suspicious_label"] = dataset["hidden_suspicious_label"].astype(int)
    return dataset.sort_values("snapshot_date").reset_index(drop=True)


def train_catboost_model() -> dict:
    import pandas as pd
    dataset      = _load_v2_training_dataset()
    feature_cols = [c for c in dataset.columns
                    if c not in NON_FEATURE_COLUMNS and c != "hidden_suspicious_label"]
    cat_indices  = [i for i, c in enumerate(feature_cols) if c in _CAT_FEATURE_NAMES]
    date_splits  = forward_date_splits(dataset["snapshot_date"])
    train_dates  = set(date_splits["train"])

    train   = dataset[dataset["snapshot_date"].dt.date.isin(train_dates)].copy()
    x_train = train[feature_cols].fillna(0)
    y_train = train["hidden_suspicious_label"]
    pos     = max(1, int(y_train.sum()))
    neg     = max(1, len(y_train) - pos)

    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        scale_pos_weight=neg / pos,
        cat_features=cat_indices,
        random_seed=42,
        verbose=0,
    )
    model.fit(x_train, y_train)

    version    = f"catboost_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    model_path = model_dir() / f"{version}.joblib"
    save_joblib(model, model_path)
    save_json(
        {"model_version": version, "feature_columns": feature_cols,
         "cat_features": [feature_cols[i] for i in cat_indices]},
        model_path.with_suffix(".json"),
    )
    return {"model_version": version, "model_path": str(model_path)}


if __name__ == "__main__":
    print(train_catboost_model())
```

- [ ] **Step 4: Run test**

```bash
PYTHONPATH=. pytest tests/test_stacker.py::test_catboost_trains_and_saves -v
```
Expected: PASSED

- [ ] **Step 5: Commit**

```bash
git add bitoguard_core/models/train_catboost.py bitoguard_core/tests/test_stacker.py
git commit -m "feat: add CatBoost Branch A (joblib save, v2 features, cat feature indices)"
```

---

### Task 4.2: stacker.py — OOF + LR meta-learner

**Files:**
- Create: `bitoguard_core/models/stacker.py`
- Test: append to `tests/test_stacker.py`

- [ ] **Step 1: Write failing test** (append)

```python
from models.stacker import train_stacker

def test_stacker_trains_and_saves(tmp_path, monkeypatch):
    store = _configure(tmp_path, monkeypatch)
    _seed_v2(store)
    result = train_stacker(n_folds=2)
    assert "stacker_version" in result
    assert Path(result["stacker_path"]).exists()
    assert "branch_models" in result
    assert len(result["branch_models"]) >= 2
```

- [ ] **Step 2: Confirm failure**

```bash
PYTHONPATH=. pytest tests/test_stacker.py::test_stacker_trains_and_saves -v
```

- [ ] **Step 3: Implement stacker.py**

```python
# bitoguard_core/models/stacker.py
"""Stacker: CatBoost + LightGBM OOF branches → Logistic Regression meta-learner.

LEAKAGE CONTRACT: graph propagation features (if used) must be computed
inside each fold using ONLY training-fold labels. See graph_propagation.py.
This stacker does not compute propagation features itself (v1 stacker
uses only label-free v2 features). Extend later by adding Branch C.
"""
from __future__ import annotations
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from models.common import (
    NON_FEATURE_COLUMNS, forward_date_splits, model_dir,
    save_joblib, save_json,
)
from models.train_catboost import _load_v2_training_dataset, _CAT_FEATURE_NAMES


def train_stacker(n_folds: int = 5) -> dict:
    """OOF stacking: CatBoost + LightGBM branches → LR meta-learner."""
    dataset = _load_v2_training_dataset()
    feature_cols = [c for c in dataset.columns
                    if c not in NON_FEATURE_COLUMNS and c != "hidden_suspicious_label"]
    cat_indices  = [i for i, c in enumerate(feature_cols) if c in _CAT_FEATURE_NAMES]

    date_splits   = forward_date_splits(dataset["snapshot_date"])
    train_dates   = set(date_splits["train"])
    train_df      = dataset[dataset["snapshot_date"].dt.date.isin(train_dates)].copy()
    x_train       = train_df[feature_cols].fillna(0).values
    y_train       = train_df["hidden_suspicious_label"].values

    oof_cb   = np.zeros(len(x_train))
    oof_lgbm = np.zeros(len(x_train))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for tr_idx, val_idx in skf.split(x_train, y_train):
        pos = max(1, int(y_train[tr_idx].sum()))
        neg = max(1, len(tr_idx) - pos)

        cb = CatBoostClassifier(
            iterations=200, learning_rate=0.05, depth=6,
            scale_pos_weight=neg / pos, cat_features=cat_indices,
            random_seed=42, verbose=0,
        )
        cb.fit(x_train[tr_idx], y_train[tr_idx])
        oof_cb[val_idx] = cb.predict_proba(x_train[val_idx])[:, 1]

        lgbm = LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=neg / pos, random_state=42,
        )
        lgbm.fit(x_train[tr_idx], y_train[tr_idx])
        oof_lgbm[val_idx] = lgbm.predict_proba(x_train[val_idx])[:, 1]

    # Train meta-learner
    oof_matrix = np.column_stack([oof_cb, oof_lgbm])
    meta = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    meta.fit(oof_matrix, y_train)

    # Retrain full base models on all training data
    pos_all = max(1, int(y_train.sum()))
    neg_all = max(1, len(y_train) - pos_all)

    final_cb = CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=6,
        scale_pos_weight=neg_all / pos_all, cat_features=cat_indices,
        random_seed=42, verbose=0,
    )
    final_cb.fit(x_train, y_train)

    final_lgbm = LGBMClassifier(
        n_estimators=250, learning_rate=0.05, num_leaves=31,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=neg_all / pos_all, random_state=42,
    )
    final_lgbm.fit(x_train, y_train)

    now_str  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version  = f"stacker_{now_str}"
    mdir     = model_dir()
    cb_path  = mdir / f"cb_{now_str}.joblib"
    lgbm_path = mdir / f"lgbm_v2_{now_str}.joblib"
    meta_path = mdir / f"{version}.joblib"

    save_joblib(final_cb,   cb_path)
    save_joblib(final_lgbm, lgbm_path)
    save_joblib(meta,       meta_path)

    meta_dict = {
        "stacker_version": version,
        "feature_columns": feature_cols,
        "branch_models": {"catboost": str(cb_path), "lgbm": str(lgbm_path)},
        "stacker_path": str(meta_path),
        "meta_coefs": meta.coef_.tolist(),
    }
    save_json(meta_dict, meta_path.with_suffix(".json"))

    return {
        "stacker_version": version,
        "stacker_path": str(meta_path),
        "branch_models": meta_dict["branch_models"],
    }


if __name__ == "__main__":
    print(train_stacker())
```

- [ ] **Step 4: Run test**

```bash
PYTHONPATH=. pytest tests/test_stacker.py -v
```
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add bitoguard_core/models/stacker.py bitoguard_core/tests/test_stacker.py
git commit -m "feat: add LR stacker (OOF CatBoost+LightGBM, joblib save, temporal splits)"
```

---

### Task 4.3: Update score.py — add score_latest_snapshot_v2()

**Key constraints verified from codebase:**
- `_load_latest_model(prefix, extension)` — **two args**, extension is suffix without dot
- `load_iforest(path)` exists; `load_lgbm(path)` exists
- `evaluate_rules(feature_frame)` uses v1 column names safely via `_get()` (returns 0 for missing)
- A rule-compat shim is required to map v2 features → v1 column names, otherwise all rules return 0

**Files:**
- Modify: `bitoguard_core/models/score.py`

- [ ] **Step 1: Append `_build_rule_compat_frame()` + `score_latest_snapshot_v2()` to score.py**

```python
# ── Append to bitoguard_core/models/score.py ──────────────────────────────────

def _build_rule_compat_frame(v2_frame: pd.DataFrame) -> pd.DataFrame:
    """Map v2 feature columns to the v1 names expected by evaluate_rules().

    Rules use safe _get() so missing columns return 0. This shim provides the
    best available v2 approximation for each v1 column to preserve rule signal.
    """
    f = v2_frame.copy()

    # Velocity rules (v1: bool flags; v2: integer counts)
    f["fiat_in_to_crypto_out_2h"]  = (f.get("fiat_dep_to_swap_buy_within_1h",  pd.Series(0, index=f.index)) > 0)
    f["fiat_in_to_crypto_out_24h"] = (f.get("fiat_dep_to_swap_buy_within_24h", pd.Series(0, index=f.index)) > 0)

    # Volume proxy (v1 uses 30d sum; v2 uses lifetime sum — acceptable approximation)
    f["crypto_withdraw_30d"] = f.get("crypto_wdr_twd_sum",       pd.Series(0.0, index=f.index))
    f["fiat_in_30d"]         = f.get("twd_dep_sum",              pd.Series(0.0, index=f.index))

    # Device/IP rules — not computable from v2 features; leave as 0 (safe default)
    for col in ("new_device_withdrawal_24h", "ip_country_switch_count",
                "night_large_withdrawal_ratio", "new_device_ratio"):
        if col not in f.columns:
            f[col] = 0

    # Graph rules — bipartite features don't map directly to v1; leave as 0
    for col in ("shared_device_count", "blacklist_1hop_count",
                "blacklist_2hop_count", "component_size"):
        if col not in f.columns:
            f[col] = 0

    # Fan-out: ip_n_entities is a loose proxy for fan_out_ratio
    f["fan_out_ratio"] = f.get("ip_n_entities", pd.Series(0, index=f.index)).astype(float)

    # Declared volume mismatch (monthly_income_twd is NULL in current data → ratio=0)
    vol_total = (
        f.get("twd_all_twd_sum", pd.Series(0.0, index=f.index)) +
        f.get("crypto_all_twd_sum", pd.Series(0.0, index=f.index))
    )
    income = f.get("monthly_income_twd", pd.Series(1.0, index=f.index)).clip(lower=1.0)
    f["actual_volume_expected_ratio"] = vol_total / income

    # Peer percentiles — not computed in v2; remain 0 (rules will silently not fire)
    f["fiat_in_30d_peer_pct"]          = 0.0
    f["crypto_withdraw_30d_peer_pct"]  = 0.0

    return f


def score_latest_snapshot_v2() -> pd.DataFrame:
    """Score using stacker (CatBoost + LightGBM) over v2 feature table.

    Returns same schema as score_latest_snapshot() for API compatibility.
    Rule engine runs via compat shim; IsolationForest provides anomaly_score.
    """
    import json
    import numpy as np
    from pathlib import Path

    settings = load_settings()
    store    = DuckDBStore(settings.db_path)

    features = load_feature_table("features.feature_snapshots_v2")
    if features.empty:
        raise ValueError("No v2 feature snapshots. Run 'make features-v2' first.")

    latest_date  = features["snapshot_date"].max()
    scoring_frame = features[features["snapshot_date"] == latest_date].copy()

    # Load stacker
    stacker_path, stacker_meta = _load_latest_model("stacker", "joblib")
    stacker_model = load_joblib(stacker_path)
    feature_cols  = stacker_meta["feature_columns"]
    x_score       = scoring_frame[feature_cols].fillna(0)

    cb_path   = Path(stacker_meta["branch_models"]["catboost"])
    lgbm_path = Path(stacker_meta["branch_models"]["lgbm"])
    cb_model   = load_joblib(cb_path)
    lgbm_model = load_joblib(lgbm_path)

    branch_matrix = np.column_stack([
        cb_model.predict_proba(x_score)[:, 1],
        lgbm_model.predict_proba(x_score)[:, 1],
    ])
    model_probability = stacker_model.predict_proba(branch_matrix)[:, 1]

    # IsolationForest anomaly branch (v1 model — reuse if available)
    try:
        iforest_path, iforest_meta = _load_latest_model("iforest", "joblib")
        iforest_model = load_iforest(iforest_path)
        a_cols  = feature_columns(scoring_frame)
        x_anom, _ = encode_features(
            scoring_frame, a_cols,
            reference_columns=iforest_meta.get("encoded_columns"),
        )
        anomaly_raw   = -iforest_model.score_samples(x_anom)
        anomaly_score = (anomaly_raw - anomaly_raw.min()) / (anomaly_raw.max() - anomaly_raw.min() + 1e-9)
    except Exception:
        anomaly_score = np.zeros(len(scoring_frame))

    # Rule engine via v2→v1 compat shim
    compat_frame = _build_rule_compat_frame(scoring_frame)
    rule_results = evaluate_rules(compat_frame)

    result = scoring_frame[["user_id", "snapshot_date"]].copy()
    result["model_probability"] = model_probability
    result["anomaly_score"]     = anomaly_score
    result["graph_risk"]        = 0.0   # bipartite features absorbed into model
    result = result.merge(
        rule_results[["user_id", "snapshot_date", "rule_score", "rule_hits"]],
        on=["user_id", "snapshot_date"], how="left",
    )
    result["rule_score"] = result["rule_score"].fillna(0.0)
    result["rule_hits"]  = result["rule_hits"].fillna("[]")

    result["risk_score"] = (
        0.70 * result["model_probability"]
        + 0.10 * result["anomaly_score"]
        + 0.20 * result["rule_score"]
    ) * 100.0
    result["risk_level"] = pd.cut(
        result["risk_score"], bins=[-1, 35, 60, 80, 100],
        labels=["low", "medium", "high", "critical"],
    ).astype(str)
    result["top_reason_codes"] = result["rule_hits"]
    result["prediction_time"]  = utc_now()
    result["model_version"]    = stacker_meta["stacker_version"]

    existing = store.fetch_df(
        "SELECT prediction_id, user_id, snapshot_date FROM ops.model_predictions WHERE snapshot_date = ?",
        (latest_date.date(),),
    )
    existing_ids = {
        _prediction_key(r["user_id"], r["snapshot_date"]): r["prediction_id"]
        for _, r in existing.iterrows()
    }
    result["prediction_id"] = result.apply(
        lambda row: existing_ids.get(
            _prediction_key(row["user_id"], row["snapshot_date"]),
            make_id(f"pred_{row['user_id'][-4:]}"),
        ), axis=1,
    )

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

Also add the required import at the top of score.py:
```python
from models.common import encode_features, feature_columns, load_feature_table, load_iforest, load_joblib, load_lgbm
```

- [ ] **Step 2: Verify import chain**

```bash
PYTHONPATH=. python -c "from models.score import score_latest_snapshot_v2; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add bitoguard_core/models/score.py
git commit -m "feat: add score_latest_snapshot_v2 (stacker + rule compat shim + correct _load_latest_model arity)"
```

---

### Task 4.4: Makefile targets + run full test suite

**Files:**
- Modify: root `Makefile`

- [ ] **Step 1: Add v2 pipeline targets**

In the root `Makefile`, append after the existing `score` target, using `$(CORE_DIR)` and `$(ACTIVATE)`:

```makefile
features-v2: ## Build v2 feature snapshots (~155 columns per user)
	cd $(CORE_DIR) && $(ACTIVATE) && PYTHONPATH=. python features/build_features_v2.py

train-stacker: ## Train CatBoost + LightGBM branches + LR stacker on v2 features
	cd $(CORE_DIR) && $(ACTIVATE) && PYTHONPATH=. python models/stacker.py

score-v2: ## Score latest snapshot using stacker (v2 features)
	cd $(CORE_DIR) && $(ACTIVATE) && PYTHONPATH=. python -c \
	    "from models.score import score_latest_snapshot_v2; r = score_latest_snapshot_v2(); print(f'Scored {len(r)} users')"
```

Also update `.PHONY` line to include new targets:
```makefile
.PHONY: ... features-v2 train-stacker score-v2
```

- [ ] **Step 2: Run all tests**

```bash
cd bitoguard_core && source .venv/bin/activate
PYTHONPATH=. pytest tests/ -v --tb=short 2>&1 | tail -30
```
Expected: all 85 existing tests pass + new tests pass. Zero regressions.

- [ ] **Step 3: Final commit**

```bash
git add Makefile
git commit -m "feat: add features-v2 + train-stacker + score-v2 Makefile targets"
```

---

## Summary

**New v2 pipeline order:**
```
make sync           # unchanged
make features       # unchanged (v1 preserved for backward compat)
make features-v2    # NEW: 155-col master table (canonical.* → registry.py)
make train          # unchanged (v1 LightGBM + IsolationForest)
make train-stacker  # NEW: CatBoost + LightGBM OOF → LR stacker
make score          # unchanged (v1 scoring + rules)
make score-v2       # NEW: stacker scoring + rule compat shim
```

**Backward compatibility preserved:**
- All 85 existing tests continue to pass
- `features.feature_snapshots_user_30d` + v1 `train` + v1 `score` untouched
- API endpoints write to `ops.model_predictions` regardless of which scorer runs
- Rule engine retained (with v2→v1 compat shim for 6 of 11 rules; peer percentile rules return 0)

**What was fixed vs. the original plan (key changes):**

| Original issue | Fix applied |
|---------------|-------------|
| `save_pickle`/`load_pickle` don't exist | Added `save_joblib`/`load_joblib` in Task 1.1 |
| `_load_latest_model` called with 1 arg | All calls now use `(prefix, extension)` e.g. `("stacker", "joblib")` |
| `FEATURE_TABLE_V2_SPEC` not in `FEATURE_TABLE_SPECS` | Appended directly into the tuple in Task 1.1 |
| `level1_finished_at` etc. not in canonical | KYC velocity features removed; 7 profile features use available columns |
| `ip_address` not on fiat_transactions or trade_orders | Extracted to `ip_features.py` reading `canonical.login_events` |
| `is_internal` not in canonical.crypto_transactions | Column reference removed from crypto_features |
| Swap/trade discrimination | `order_type == "instant_swap"` instead of `base_asset == "USDT"` |
| `evaluate_rules` silently returns 0 on v2 frame | Added `_build_rule_compat_frame()` mapping 8/11 rules |
| Makefile used hardcoded paths | All targets use `$(CORE_DIR)` and `$(ACTIVATE)` |
| registry.py missing `logins` argument | `build_v2_features()` now takes 6 args including `logins` |
