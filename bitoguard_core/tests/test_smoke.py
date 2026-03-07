from __future__ import annotations

import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app
from config import load_settings
from models.score import score_latest_snapshot
from db.store import DuckDBStore


def test_settings_have_expected_defaults() -> None:
    settings = load_settings()
    assert settings.internal_api_port == 8001
    assert isinstance(settings.db_path, Path)


def _configure_temp_db(tmp_path: Path, monkeypatch) -> Path:
    source_root = Path(__file__).resolve().parents[1]
    source_artifacts = source_root / "artifacts"
    source_db = source_artifacts / "bitoguard.duckdb"
    target_db = tmp_path / "bitoguard.duckdb"
    target_artifacts = tmp_path / "artifacts"
    shutil.copy2(source_db, target_db)
    shutil.copytree(source_artifacts / "models", target_artifacts / "models")
    monkeypatch.setenv("BITOGUARD_DB_PATH", str(target_db))
    monkeypatch.setenv("BITOGUARD_ARTIFACT_DIR", str(target_artifacts))
    return target_db


def test_alert_report_includes_case_metadata(tmp_path: Path, monkeypatch) -> None:
    target_db = _configure_temp_db(tmp_path, monkeypatch)
    store = DuckDBStore(target_db)
    client = TestClient(app)
    alerts = client.get("/alerts", params={"page_size": 1})
    assert alerts.status_code == 200
    alert_id = alerts.json()["items"][0]["alert_id"]
    report_path = tmp_path / "artifacts" / "reports" / f"{alert_id}.json"
    before = store.fetch_df("SELECT report_path FROM ops.alerts WHERE alert_id = ?", (alert_id,))

    response = client.get(f"/alerts/{alert_id}/report")
    assert response.status_code == 200
    payload = response.json()
    assert "alert" in payload
    assert "case" in payload
    assert "case_actions" in payload
    assert payload["allowed_decisions"] == [
        "confirm_suspicious",
        "dismiss_false_positive",
        "escalate",
        "request_monitoring",
    ]
    assert not report_path.exists()

    alert_row = store.fetch_df("SELECT report_path FROM ops.alerts WHERE alert_id = ?", (alert_id,))
    assert alert_row.iloc[0]["report_path"] == before.iloc[0]["report_path"]


def test_case_decision_updates_statuses(tmp_path: Path, monkeypatch) -> None:
    _configure_temp_db(tmp_path, monkeypatch)
    client = TestClient(app)
    alerts = client.get("/alerts", params={"page_size": 1})
    alert_id = alerts.json()["items"][0]["alert_id"]

    decision = client.post(
        f"/alerts/{alert_id}/decision",
        json={"decision": "escalate", "actor": "tester", "note": "needs review"},
    )
    assert decision.status_code == 200
    payload = decision.json()
    assert payload["alert_status"] == "escalated"
    assert payload["case_status"] == "escalated"
    assert payload["latest_decision"] == "escalate"

    report = client.get(f"/alerts/{alert_id}/report")
    report_payload = report.json()
    assert report_payload["alert"]["status"] == "escalated"
    assert report_payload["case"]["status"] == "escalated"
    assert report_payload["case"]["latest_decision"] == "escalate"
    assert report_payload["case_actions"][0]["actor"] == "tester"


def test_graph_endpoint_supports_hop_summary(tmp_path: Path, monkeypatch) -> None:
    _configure_temp_db(tmp_path, monkeypatch)
    client = TestClient(app)
    alerts = client.get("/alerts", params={"page_size": 1})
    user_id = alerts.json()["items"][0]["user_id"]

    one_hop = client.get(f"/users/{user_id}/graph", params={"max_hops": 1})
    two_hop = client.get(f"/users/{user_id}/graph", params={"max_hops": 2})

    assert one_hop.status_code == 200
    assert two_hop.status_code == 200
    one_payload = one_hop.json()
    two_payload = two_hop.json()
    assert one_payload["focus_user_id"] == user_id
    assert two_payload["focus_user_id"] == user_id
    assert all(node["hop"] <= 1 for node in one_payload["nodes"])
    assert all(node["hop"] <= 2 for node in two_payload["nodes"])
    assert two_payload["summary"]["node_count"] >= one_payload["summary"]["node_count"]
    assert two_payload["summary"]["edge_count"] >= one_payload["summary"]["edge_count"]


def test_rescoring_preserves_alert_prediction_links(tmp_path: Path, monkeypatch) -> None:
    target_db = _configure_temp_db(tmp_path, monkeypatch)
    store = DuckDBStore(target_db)
    latest_snapshot = store.fetch_df("SELECT MAX(snapshot_date) AS snapshot_date FROM ops.model_predictions").iloc[0]["snapshot_date"]
    before = store.fetch_df(
        """
        SELECT alert_id, prediction_id
        FROM ops.alerts
        WHERE snapshot_date = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (latest_snapshot,),
    )
    assert not before.empty

    score_latest_snapshot()

    after = store.fetch_df(
        """
        SELECT alerts.alert_id, alerts.prediction_id, predictions.risk_score
        FROM ops.alerts AS alerts
        LEFT JOIN ops.model_predictions AS predictions ON alerts.prediction_id = predictions.prediction_id
        WHERE alerts.snapshot_date = ?
        ORDER BY alerts.created_at DESC
        LIMIT 1
        """,
        (latest_snapshot,),
    )
    assert after.iloc[0]["alert_id"] == before.iloc[0]["alert_id"]
    assert after.iloc[0]["prediction_id"] == before.iloc[0]["prediction_id"]
    assert after.iloc[0]["risk_score"] is not None
