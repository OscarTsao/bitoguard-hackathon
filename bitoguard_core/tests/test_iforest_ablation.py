from __future__ import annotations

import json
from pathlib import Path

from features.build_anomaly_features import build_anomaly_feature_snapshots
from features.build_features import build_feature_snapshots
from features.graph_features import build_graph_features
from models.anomaly import train_anomaly_model
from models.ablate_iforest import run_iforest_ablation
from tests.test_smoke import _configure_temp_db


def test_iforest_ablation_writes_report(tmp_path: Path, monkeypatch) -> None:
    _configure_temp_db(tmp_path, monkeypatch)
    build_graph_features()
    build_feature_snapshots()
    build_anomaly_feature_snapshots()
    train_anomaly_model()
    report = run_iforest_ablation()

    assert report["data_summary"]["rows"] > 0
    assert report["data_summary"]["positive_rows"] > 0
    assert "primary_without_iforest" in report["results"]
    assert "primary_plus_iforest_candidate" in report["results"]
    assert "iforest_only" in report["results"]
    assert report["recommendation"]["verdict"] in {"eligible_for_primary_blend", "sidecar_only"}
    assert Path(report["report_path"]).exists()
    assert Path(report["score_path"]).exists()
    stored_report = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    assert stored_report["recommendation"]["gate_passed"] == report["recommendation"]["gate_passed"]
