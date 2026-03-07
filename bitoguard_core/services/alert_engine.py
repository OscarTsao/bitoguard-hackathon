from __future__ import annotations

import pandas as pd

from config import load_settings
from db.store import DuckDBStore, make_id, utc_now

DECISION_STATUS_MAP = {
    "confirm_suspicious": {
        "case_status": "closed_confirmed",
        "alert_status": "confirmed_suspicious",
    },
    "dismiss_false_positive": {
        "case_status": "closed_dismissed",
        "alert_status": "dismissed_false_positive",
    },
    "escalate": {
        "case_status": "escalated",
        "alert_status": "escalated",
    },
    "request_monitoring": {
        "case_status": "monitoring",
        "alert_status": "monitoring",
    },
}


def _alert_key(user_id: str, snapshot_date: object) -> tuple[str, object]:
    return (user_id, pd.Timestamp(snapshot_date).date())


def _sync_existing_alerts(store: DuckDBStore, predictions: pd.DataFrame) -> None:
    if predictions.empty:
        return

    current_predictions = predictions[["user_id", "snapshot_date", "prediction_id", "risk_level"]].copy()
    with store.connect() as conn:
        conn.register("current_predictions", current_predictions)
        conn.execute(
            """
            UPDATE ops.alerts AS alerts
            SET
                prediction_id = current_predictions.prediction_id,
                risk_level = current_predictions.risk_level
            FROM current_predictions
            WHERE
                alerts.user_id = current_predictions.user_id
                AND alerts.snapshot_date = current_predictions.snapshot_date
            """
        )
        conn.unregister("current_predictions")


def generate_alerts() -> pd.DataFrame:
    settings = load_settings()
    store = DuckDBStore(settings.db_path)
    predictions = store.read_table("ops.model_predictions")
    if predictions.empty:
        return pd.DataFrame()

    _sync_existing_alerts(store, predictions)

    high_risk = predictions[predictions["risk_level"].isin(["high", "critical"])].copy()
    if high_risk.empty:
        return pd.DataFrame()

    existing = store.read_table("ops.alerts")
    existing_keys = set()
    if not existing.empty:
        existing_keys = {
            _alert_key(row["user_id"], row["snapshot_date"])
            for _, row in existing.iterrows()
        }

    alerts = []
    cases = []
    for _, row in high_risk.iterrows():
        key = _alert_key(row["user_id"], row["snapshot_date"])
        if key in existing_keys:
            continue
        alert_id = make_id("alert")
        case_id = make_id("case")
        alerts.append({
            "alert_id": alert_id,
            "user_id": row["user_id"],
            "snapshot_date": row["snapshot_date"],
            "created_at": utc_now(),
            "risk_level": row["risk_level"],
            "status": "open",
            "prediction_id": row["prediction_id"],
            "report_path": None,
        })
        cases.append({
            "case_id": case_id,
            "alert_id": alert_id,
            "user_id": row["user_id"],
            "created_at": utc_now(),
            "status": "open",
            "latest_decision": None,
        })
    if alerts:
        store.append_dataframe("ops.alerts", pd.DataFrame(alerts))
    if cases:
        store.append_dataframe("ops.cases", pd.DataFrame(cases))
    return pd.DataFrame(alerts)


def apply_case_decision(alert_id: str, decision: str, actor: str = "analyst", note: str = "") -> dict[str, str]:
    if decision not in DECISION_STATUS_MAP:
        raise ValueError(f"Unsupported decision: {decision}")
    settings = load_settings()
    store = DuckDBStore(settings.db_path)
    alert_row = store.fetch_df("SELECT * FROM ops.alerts WHERE alert_id = ?", (alert_id,))
    if alert_row.empty:
        raise ValueError(f"No alert found for alert_id={alert_id}")
    case_row = store.fetch_df("SELECT * FROM ops.cases WHERE alert_id = ?", (alert_id,))
    if case_row.empty:
        raise ValueError(f"No case found for alert_id={alert_id}")
    case_id = case_row.iloc[0]["case_id"]
    status_update = DECISION_STATUS_MAP[decision]
    store.execute(
        """
        INSERT INTO ops.case_actions (action_id, case_id, action_type, actor, action_at, note)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (make_id("action"), case_id, decision, actor, utc_now(), note),
    )
    store.execute(
        "UPDATE ops.cases SET latest_decision = ?, status = ? WHERE case_id = ?",
        (decision, status_update["case_status"], case_id),
    )
    store.execute(
        "UPDATE ops.alerts SET status = ? WHERE alert_id = ?",
        (status_update["alert_status"], alert_id),
    )
    return {
        "alert_id": alert_id,
        "decision": decision,
        "alert_status": status_update["alert_status"],
        "case_status": status_update["case_status"],
        "latest_decision": decision,
    }
