from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastapi.testclient import TestClient

from api.main import app
from features.build_anomaly_features import build_anomaly_feature_snapshots
from features.build_features import build_feature_snapshots
from features.graph_features import build_graph_features
from models.anomaly import train_anomaly_model
from models.anomaly_common import apply_anomaly_model, load_anomaly_source_table, load_user_cohort_frame
from models.common import load_pickle
from tests.test_smoke import _configure_temp_db


@pytest.mark.integration
def test_anomaly_feature_snapshots_are_numeric_and_id_free(tmp_path: Path, monkeypatch) -> None:
    _configure_temp_db(tmp_path, monkeypatch)
    build_graph_features()
    build_feature_snapshots()
    frame = build_anomaly_feature_snapshots()

    assert not frame.empty
    model_columns = [column for column in frame.columns if column not in {"feature_snapshot_id", "user_id", "snapshot_date", "feature_version"}]
    assert all(not column.endswith("_id") for column in model_columns)
    assert "fiat_txn_id" not in model_columns
    assert all(str(frame[column].dtype) in {"bool", "int64", "float64"} for column in model_columns)


@pytest.mark.integration
def test_anomaly_percentile_scoring_is_order_stable(tmp_path: Path, monkeypatch) -> None:
    _configure_temp_db(tmp_path, monkeypatch)
    build_graph_features()
    build_feature_snapshots()
    build_anomaly_feature_snapshots()
    model_info = train_anomaly_model()

    meta = json.loads(Path(model_info["meta_path"]).read_text(encoding="utf-8"))
    model = load_pickle(Path(model_info["model_path"]))
    frame = load_anomaly_source_table()
    latest_date = frame["snapshot_date"].max()
    latest = frame[frame["snapshot_date"] == latest_date].copy().reset_index(drop=True)
    shuffled = latest.sample(frac=1.0, random_state=7).reset_index(drop=True)
    cohorts = load_user_cohort_frame()

    _, original_scores = apply_anomaly_model(model, latest, cohorts, meta)
    _, shuffled_scores = apply_anomaly_model(model, shuffled, cohorts, meta)

    original_map = dict(zip(latest["user_id"], original_scores, strict=False))
    shuffled_map = dict(zip(shuffled["user_id"], shuffled_scores, strict=False))
    assert original_map.keys() == shuffled_map.keys()
    for user_id in original_map:
        assert original_map[user_id] == shuffled_map[user_id]


@pytest.mark.integration
def test_features_rebuild_returns_anomaly_rows(tmp_path: Path, monkeypatch) -> None:
    _configure_temp_db(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.post("/features/rebuild")

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_anomaly_rows"] > 0
