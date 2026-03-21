from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from api.main import app
from config import load_settings
from db.store import DuckDBStore, make_id, utc_now


def _ensure_feature_columns(store: DuckDBStore) -> None:
    required_columns = (
        ("kyc_level", "VARCHAR"),
        ("occupation", "VARCHAR"),
        ("monthly_income_twd", "DOUBLE"),
        ("expected_monthly_volume_twd", "DOUBLE"),
        ("declared_source_of_funds", "VARCHAR"),
        ("segment", "VARCHAR"),
        ("shared_device_count", "DOUBLE"),
        ("shared_bank_count", "DOUBLE"),
        ("shared_wallet_count", "DOUBLE"),
        ("blacklist_1hop_count", "DOUBLE"),
        ("blacklist_2hop_count", "DOUBLE"),
        ("component_size", "DOUBLE"),
        ("fan_out_ratio", "DOUBLE"),
    )
    with store.transaction() as conn:
        existing_columns = conn.execute(
            "SELECT * FROM features.feature_snapshots_user_day LIMIT 0"
        ).df().columns.tolist()
        for column_name, column_type in required_columns:
            if column_name not in existing_columns:
                conn.execute(
                    f"ALTER TABLE features.feature_snapshots_user_day ADD COLUMN {column_name} {column_type}"
                )


def _ensure_alert_fixture(target_db: Path) -> None:
    store = DuckDBStore(target_db)
    alert_count = int(store.fetch_df("SELECT COUNT(*) AS n FROM ops.alerts").iloc[0]["n"])
    if alert_count > 0:
        return

    seed_user = store.fetch_df(
        """
        SELECT user_id, created_at, kyc_level, occupation, monthly_income_twd,
               expected_monthly_volume_twd, declared_source_of_funds, segment
        FROM canonical.users
        ORDER BY created_at ASC, user_id ASC
        LIMIT 1
        """
    )
    if seed_user.empty:
        with store.transaction() as conn:
            conn.execute(
                """
                INSERT INTO canonical.users (
                    user_id, created_at, segment, kyc_level, occupation,
                    monthly_income_twd, expected_monthly_volume_twd,
                    declared_source_of_funds, residence_country, residence_city,
                    nationality, activity_window
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "usr_smoke_0001",
                    "2026-03-01T00:00:00+00:00",
                    "retail",
                    "level_2",
                    "engineer",
                    120000.0,
                    80000.0,
                    "salary",
                    "TW",
                    "Taipei",
                    "TW",
                    "30d",
                ),
            )
        seed_user = store.fetch_df(
            """
            SELECT user_id, created_at, kyc_level, occupation, monthly_income_twd,
                   expected_monthly_volume_twd, declared_source_of_funds, segment
            FROM canonical.users
            ORDER BY created_at ASC, user_id ASC
            LIMIT 1
            """
        )
    assert not seed_user.empty
    user = seed_user.iloc[0]
    snapshot_date = (
        pd.Timestamp(user["created_at"]).date()
        if user["created_at"] is not None
        else pd.Timestamp("2026-03-01").date()
    )

    _ensure_feature_columns(store)
    feature_count = store.fetch_df(
        """
        SELECT COUNT(*) AS n
        FROM features.feature_snapshots_user_day
        WHERE user_id = ? AND snapshot_date = ?
        """,
        (user["user_id"], snapshot_date),
    ).iloc[0]["n"]
    if int(feature_count) == 0:
        with store.transaction() as conn:
            conn.execute(
                """
                INSERT INTO features.feature_snapshots_user_day (
                    feature_snapshot_id, user_id, snapshot_date, feature_version,
                    kyc_level, occupation, monthly_income_twd, expected_monthly_volume_twd,
                    declared_source_of_funds, segment, shared_device_count, shared_bank_count,
                    shared_wallet_count, blacklist_1hop_count, blacklist_2hop_count,
                    component_size, fan_out_ratio
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"fd_{user['user_id']}_{snapshot_date.isoformat()}",
                    user["user_id"],
                    snapshot_date,
                    "test-fixture",
                    user["kyc_level"],
                    user["occupation"],
                    user["monthly_income_twd"],
                    user["expected_monthly_volume_twd"],
                    user["declared_source_of_funds"],
                    user["segment"],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ),
            )

    prediction_count = int(store.fetch_df("SELECT COUNT(*) AS n FROM ops.model_predictions").iloc[0]["n"])
    if prediction_count == 0:
        settings = load_settings()
        model_files = sorted((settings.artifact_dir / "models").glob("lgbm_*.lgbm"))
        model_version = model_files[-1].stem if model_files else "lgbm_smoke_fixture"
        prediction_id = make_id("pred")
        prediction_time = utc_now()
        with store.transaction() as conn:
            conn.execute(
                """
                INSERT INTO ops.model_predictions (
                    prediction_id, user_id, snapshot_date, prediction_time, model_version,
                    risk_score, risk_level, rule_hits, top_reason_codes,
                    model_probability, anomaly_score, graph_risk
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction_id,
                    user["user_id"],
                    snapshot_date,
                    prediction_time,
                    model_version,
                    82.5,
                    "high",
                    json.dumps(["smoke_fixture_rule"]),
                    json.dumps(["smoke_fixture_rule"]),
                    0.82,
                    0.11,
                    0.0,
                ),
            )

    top_prediction = store.fetch_df(
        """
        SELECT prediction_id, user_id, snapshot_date, risk_level
        FROM ops.model_predictions
        ORDER BY risk_score DESC, prediction_time DESC
        LIMIT 1
        """
    )
    assert not top_prediction.empty
    prediction = top_prediction.iloc[0]
    alert_id = make_id("alert")
    case_id = make_id("case")
    created_at = utc_now()
    with store.transaction() as conn:
        conn.execute(
            """
            INSERT INTO ops.alerts (
                alert_id, user_id, snapshot_date, created_at, risk_level, status, prediction_id, report_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                alert_id,
                prediction["user_id"],
                prediction["snapshot_date"],
                created_at,
                prediction["risk_level"],
                "open",
                prediction["prediction_id"],
                None,
            ),
        )
        conn.execute(
            """
            INSERT INTO ops.cases (
                case_id, alert_id, user_id, created_at, status, latest_decision
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (case_id, alert_id, prediction["user_id"], created_at, "open", None),
        )


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
    validation_report = source_artifacts / "validation_report.json"
    if validation_report.exists():
        shutil.copy2(validation_report, target_artifacts / "validation_report.json")
    monkeypatch.setenv("BITOGUARD_DB_PATH", str(target_db))
    monkeypatch.setenv("BITOGUARD_ARTIFACT_DIR", str(target_artifacts))
    _ensure_alert_fixture(target_db)
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
    latest_snapshot = store.fetch_df(
        """
        SELECT MAX(alerts.snapshot_date) AS snapshot_date
        FROM ops.alerts AS alerts
        INNER JOIN ops.model_predictions AS predictions
            ON alerts.snapshot_date = predictions.snapshot_date
        """
    ).iloc[0]["snapshot_date"]
    assert latest_snapshot is not None
    before = store.fetch_df(
        """
        SELECT alert_id, prediction_id, predictions.risk_score
        FROM ops.alerts
        LEFT JOIN ops.model_predictions AS predictions USING (prediction_id)
        WHERE alerts.snapshot_date = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (latest_snapshot,),
    )
    assert not before.empty

    with store.transaction() as conn:
        conn.execute(
            """
            UPDATE ops.model_predictions
            SET risk_score = risk_score + 1.0,
                prediction_time = ?
            WHERE prediction_id = ?
            """,
            (utc_now(), before.iloc[0]["prediction_id"]),
        )

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
    assert after.iloc[0]["risk_score"] != before.iloc[0]["risk_score"]


def test_metrics_model_endpoint_returns_full_report(tmp_path: Path, monkeypatch) -> None:
    """GET /metrics/model returns a complete validation report with P@K and calibration."""
    _configure_temp_db(tmp_path, monkeypatch)
    client = TestClient(app)
    resp = client.get("/metrics/model")
    assert resp.status_code == 200
    body = resp.json()
    # Core metrics must be present
    for field in ("model_version", "precision", "recall", "f1", "fpr", "average_precision"):
        assert field in body, f"missing field: {field}"
    # Precision@K / Recall@K added in QUALITY-009
    assert "precision_at_k" in body, "precision_at_k missing from /metrics/model"
    assert "recall_at_k" in body, "recall_at_k missing from /metrics/model"
    assert isinstance(body["precision_at_k"], dict)
    # Calibration
    assert "calibration" in body
    assert "brier_score" in body["calibration"]
    # Feature importance
    assert "feature_importance_top20" in body
    assert isinstance(body["feature_importance_top20"], list)
    # Threshold sensitivity
    assert "threshold_sensitivity" in body
    assert len(body["threshold_sensitivity"]) > 0


def test_metrics_drift_endpoint_returns_health_status(tmp_path: Path, monkeypatch) -> None:
    """GET /metrics/drift returns a combined health report with feature drift, PSI, and staleness."""
    _configure_temp_db(tmp_path, monkeypatch)
    client = TestClient(app)
    resp = client.get("/metrics/drift")
    assert resp.status_code == 200
    body = resp.json()
    # Top-level keys
    assert "health_ok" in body, "missing health_ok in drift response"
    assert isinstance(body["health_ok"], bool)
    # Feature drift sub-report
    assert "feature_drift" in body, "missing feature_drift in drift response"
    feat = body["feature_drift"]
    for field in ("snapshot_from", "snapshot_to", "total_checked", "total_drifted", "health_ok", "drifted_features"):
        assert field in feat, f"missing field in feature_drift: {field}"
    assert isinstance(feat["drifted_features"], list)
    assert isinstance(feat["total_checked"], int)
    # Score PSI and model staleness may be None when no scoring runs / bundle exists
    assert "score_psi" in body
    assert "model_staleness" in body


def test_pipeline_sync_rejects_inverted_time_window(tmp_path: Path, monkeypatch) -> None:
    _configure_temp_db(tmp_path, monkeypatch)
    client = TestClient(app)
    resp = client.post(
        "/pipeline/sync",
        json={
            "full": False,
            "start_time": "2026-03-10T00:00:00+00:00",
            "end_time": "2026-03-09T00:00:00+00:00",
        },
    )
    assert resp.status_code == 422


def test_graph_endpoint_does_not_load_full_edge_table(tmp_path: Path, monkeypatch) -> None:
    """Graph endpoint must query edges by user_id, not load the whole table."""
    import inspect
    from api.main import _build_graph_payload
    src = inspect.getsource(_build_graph_payload)
    assert 'read_table("canonical.entity_edges")' not in src, \
        "_build_graph_payload must not load the full entity_edges table"
    assert "_load_neighborhood_edges" in src or "fetch_df" in src, \
        "_build_graph_payload must use filtered SQL queries"


def test_api_key_enforcement_when_configured(tmp_path: Path, monkeypatch) -> None:
    """When BITOGUARD_API_KEY is set, requests without the key get 401."""
    _configure_temp_db(tmp_path, monkeypatch)
    monkeypatch.setenv("BITOGUARD_API_KEY", "test-secret-key-abc123")
    from api.main import app
    client = TestClient(app)

    # Without key: 401
    resp = client.get("/alerts")
    assert resp.status_code == 401

    # With wrong key: 401
    resp = client.get("/alerts", headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401

    # With correct key: 200
    resp = client.get("/alerts", headers={"X-API-Key": "test-secret-key-abc123"})
    assert resp.status_code == 200

    # /healthz never requires key
    resp = client.get("/healthz")
    assert resp.status_code == 200


def test_score_v2_uses_transaction_for_db_write():
    """score_latest_snapshot must use store.transaction() not bare execute+append."""
    import numpy as np
    import pandas as pd
    from unittest.mock import MagicMock, patch

    with patch("models.score.load_feature_table") as mock_ft, \
         patch("models.score._load_latest_model") as mock_lm, \
         patch("models.score.load_joblib") as mock_lj, \
         patch("models.score.load_iforest") as mock_li, \
         patch("models.score.evaluate_rules") as mock_rules, \
         patch("models.score.generate_alerts"), \
         patch("models.score.DuckDBStore") as MockStore, \
         patch("models.score.load_settings"):

        mock_ft.return_value = pd.DataFrame({
            "user_id": ["u1"],
            "snapshot_date": [pd.Timestamp("2025-06-01")],
            "feature_snapshot_id": ["f1"],
            "feature_version": ["v2"],
        })
        stacker_meta = {
            "stacker_version": "stacker_test",
            "feature_columns": [],
            "branch_models": {"catboost": "/fake/cb.joblib", "lgbm": "/fake/lgbm.joblib"},
        }
        mock_lm.return_value = ("/fake/stacker.joblib", stacker_meta)
        mock_lj.return_value = MagicMock(
            predict_proba=lambda x: np.array([[0.2, 0.8]] * max(1, len(x)))
        )
        mock_li.side_effect = FileNotFoundError

        mock_rules.return_value = pd.DataFrame({
            "user_id": ["u1"],
            "snapshot_date": [pd.Timestamp("2025-06-01")],
            "rule_score": [0.0],
            "rule_hits": ["[]"],
        })

        store = MagicMock()
        store.fetch_df.return_value = pd.DataFrame(
            columns=["prediction_id", "user_id", "snapshot_date"]
        )
        # Context manager support for store.transaction()
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=MagicMock())
        ctx.__exit__ = MagicMock(return_value=False)
        store.transaction.return_value = ctx

        MockStore.return_value = store

        from models.score import score_latest_snapshot
        score_latest_snapshot()

        assert store.transaction.called, (
            "score_latest_snapshot must use store.transaction() for atomic write"
        )
        assert not store.execute.called or all(
            "DELETE" not in str(call) for call in store.execute.call_args_list
        ), "store.execute(DELETE) must not be called outside a transaction"




def test_load_neighborhood_edges_caps_neighbor_ids():
    """
    _load_neighborhood_edges must cap neighbor_ids at _MAX_NEIGHBOR_IDS
    to prevent unbounded SQL placeholder construction.
    """
    import pandas as pd
    from unittest.mock import MagicMock
    from api.main import _load_neighborhood_edges, _MAX_NEIGHBOR_IDS

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
    store.fetch_df.side_effect = [one_hop_df, pd.DataFrame()]

    _load_neighborhood_edges(store, "focus_user", max_hops=2)

    second_call_sql = store.fetch_df.call_args_list[1][0][0]
    placeholder_count = second_call_sql.count("?")
    assert placeholder_count // 2 <= _MAX_NEIGHBOR_IDS, (
        f"SQL had {placeholder_count // 2} neighbors; must be capped at {_MAX_NEIGHBOR_IDS}"
    )

@pytest.mark.integration
def test_alerts_generated_after_scoring() -> None:
    """Guards against threshold miscalibration causing zero alerts.

    This is an INTEGRATION test — it reads from the live bitoguard.duckdb.
    Run manually after `make score` with real data:

        pytest tests/test_smoke.py::test_alerts_generated_after_scoring -m integration -v

    Not included in the default `make test` suite (excluded via addopts in pyproject.toml).
    """
    from config import load_settings
    from db.store import DuckDBStore

    settings = load_settings()
    store = DuckDBStore(settings.db_path)
    df = store.fetch_df(
        "SELECT COUNT(*) AS cnt FROM ops.alerts WHERE risk_level IN ('medium', 'high', 'critical')"
    )
    count = int(df["cnt"].iloc[0]) if not df.empty else 0
    assert count > 0, (
        "Zero non-low alerts in ops.alerts. "
        "Check: (1) m1_enabled=True and m3_enabled=True in config.py defaults, "
        "(2) alert bins=[-1,20,50,70,100] in score.py score_latest_snapshot(). "
        "Run `make score` with real data first."
    )
