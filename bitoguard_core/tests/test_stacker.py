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
    users = [
        ("u_neg1", 0, 1, 0.0),
        ("u_neg2", 0, 1, 0.0),
        ("u_neg3", 0, 1, 0.0),
        ("u_pos1", 1, 2, 3.0),
        ("u_pos2", 1, 2, 4.0),
        ("u_pos3", 1, 2, 5.0),
    ]
    rows = []
    for i, d in enumerate(dates, 1):
        for uid, _label, kyc, crypto_base in users:
            rows.append({
                "feature_snapshot_id": f"{uid}_{i}",
                "user_id": uid,
                "snapshot_date": d,
                "feature_version": "v2",
                "twd_all_count": float(i) * (5.0 if "pos" in uid else 1.0),
                "kyc_level_code": kyc,
                "crypto_all_count": crypto_base * i,
            })
    store.replace_table("features.feature_snapshots_v2", pd.DataFrame(rows))
    labels = []
    for uid, label, _, _ in users:
        labels.append({
            "user_id": uid,
            "hidden_suspicious_label": label,
            "observed_blacklist_label": label,
            "scenario_types": "",
            "evidence_tags": "",
        })
    store.replace_table("ops.oracle_user_labels", pd.DataFrame(labels))
    # Provide blacklist_feed entries for positive users so the leakage-safe
    # WHERE clause (ped.ped IS NOT NULL AND snapshot_date >= ped.ped) includes them.
    blacklist_rows = []
    for uid, label, _, _ in users:
        if label == 1:
            blacklist_rows.append({
                "blacklist_entry_id": f"bl_{uid}",
                "user_id": uid,
                "observed_at": pd.Timestamp("2025-12-31T00:00:00Z"),
                "source": "oracle",
                "reason_code": "test",
                "is_active": True,
            })
    store.replace_table("canonical.blacklist_feed", pd.DataFrame(blacklist_rows))


def test_catboost_trains_and_saves(tmp_path, monkeypatch):
    store = _configure(tmp_path, monkeypatch)
    _seed_v2(store)
    result = train_catboost_model()
    assert "model_version" in result
    assert Path(result["model_path"]).exists()
    assert result["model_version"].startswith("catboost_")


from models.stacker import train_stacker


def test_stacker_trains_and_saves(tmp_path, monkeypatch):
    store = _configure(tmp_path, monkeypatch)
    _seed_v2(store)
    result = train_stacker(n_folds=2, n_estimators=10)
    assert "stacker_version" in result
    assert Path(result["stacker_path"]).exists()
    assert "branch_models" in result
    assert len(result["branch_models"]) >= 2


def test_stacker_no_user_leakage(tmp_path, monkeypatch):
    """Verify that no user_id appears in both train and val within any OOF fold."""
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold
    from models.common import NON_FEATURE_COLUMNS, forward_date_splits
    from models.train_catboost import load_v2_training_dataset

    store = _configure(tmp_path, monkeypatch)
    _seed_v2(store)

    dataset = load_v2_training_dataset()
    feature_cols = [c for c in dataset.columns
                    if c not in NON_FEATURE_COLUMNS and c != "hidden_suspicious_label"]
    date_splits = forward_date_splits(dataset["snapshot_date"])
    train_dates = set(date_splits["train"])
    train_df = dataset[dataset["snapshot_date"].dt.date.isin(train_dates)].copy()
    groups = train_df["user_id"].reset_index(drop=True).values
    x_train_df = train_df[feature_cols].fillna(0).reset_index(drop=True)
    y_train = train_df["hidden_suspicious_label"].values

    sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
    for fold_idx, (tr_idx, val_idx) in enumerate(sgkf.split(x_train_df, y_train, groups=groups)):
        train_users = set(groups[tr_idx])
        val_users = set(groups[val_idx])
        overlap = train_users & val_users
        assert overlap == set(), (
            f"Fold {fold_idx}: user(s) {overlap} appear in both train and val sets"
        )
        assert len(np.unique(y_train[tr_idx])) == 2, (
            f"Fold {fold_idx}: training set must contain both classes"
        )
