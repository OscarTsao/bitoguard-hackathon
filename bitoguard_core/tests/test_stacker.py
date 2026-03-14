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
