from __future__ import annotations

import json
from collections import deque
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pydantic import BaseModel

from config import load_settings
from db.store import DuckDBStore
from features.build_features import build_feature_snapshots
from features.graph_features import build_graph_features
from models.anomaly import train_anomaly_model
from models.score import score_latest_snapshot
from models.train import train_model
from models.validate import validate_model
from pipeline.sync import run_sync
from services.alert_engine import apply_case_decision
from services.diagnosis import build_risk_diagnosis


class SyncRequest(BaseModel):
    full: bool = True
    start_time: str | None = None
    end_time: str | None = None


class DecisionRequest(BaseModel):
    decision: str
    actor: str = "analyst"
    note: str = ""


app = FastAPI(title="BitoGuard Core API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_DECISIONS = [
    "confirm_suspicious",
    "dismiss_false_positive",
    "escalate",
    "request_monitoring",
]


def _row_to_dict(frame: pd.DataFrame) -> dict[str, Any] | None:
    if frame.empty:
        return None
    record = frame.iloc[0].to_dict()
    normalized = {}
    for key, value in record.items():
        if pd.isna(value):
            normalized[key] = None
        elif isinstance(value, pd.Timestamp):
            normalized[key] = value.isoformat()
        else:
            normalized[key] = value
    return normalized


def _records_to_dicts(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    records: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        normalized = {}
        for key, value in record.items():
            if pd.isna(value):
                normalized[key] = None
            elif isinstance(value, pd.Timestamp):
                normalized[key] = value.isoformat()
            else:
                normalized[key] = value
        records.append(normalized)
    return records


def _build_graph_payload(store: DuckDBStore, user_id: str, max_hops: int) -> dict[str, Any]:
    edges = store.read_table("canonical.entity_edges")
    focus_node_id = f"user:{user_id}"
    if edges.empty:
        return {
            "focus_user_id": user_id,
            "summary": {
                "node_count": 1,
                "edge_count": 0,
                "blacklist_neighbor_count": 0,
                "high_risk_neighbor_count": 0,
                "is_truncated": False,
            },
            "nodes": [{
                "id": focus_node_id,
                "type": "user",
                "label": user_id,
                "hop": 0,
                "is_focus": True,
                "risk_level": None,
                "is_known_blacklist": False,
            }],
            "edges": [],
        }

    graph_edges = edges.copy()
    graph_edges["src_node"] = graph_edges["src_type"] + ":" + graph_edges["src_id"]
    graph_edges["dst_node"] = graph_edges["dst_type"] + ":" + graph_edges["dst_id"]

    adjacency: dict[str, set[str]] = {}
    for _, row in graph_edges.iterrows():
        adjacency.setdefault(row["src_node"], set()).add(row["dst_node"])
        adjacency.setdefault(row["dst_node"], set()).add(row["src_node"])

    distances = {focus_node_id: 0}
    queue: deque[str] = deque([focus_node_id])
    while queue:
        current = queue.popleft()
        if distances[current] >= max_hops:
            continue
        for neighbor in sorted(adjacency.get(current, set())):
            if neighbor in distances:
                continue
            distances[neighbor] = distances[current] + 1
            queue.append(neighbor)

    included_nodes = set(distances.keys())
    graph_edges = graph_edges[
        graph_edges["src_node"].isin(included_nodes)
        & graph_edges["dst_node"].isin(included_nodes)
    ].copy()

    predictions = store.fetch_df(
        """
        SELECT user_id, risk_level
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY snapshot_date DESC, prediction_time DESC) AS rn
            FROM ops.model_predictions
        )
        WHERE rn = 1
        """
    )
    prediction_map = {
        row["user_id"]: row["risk_level"]
        for _, row in predictions.iterrows()
    }
    blacklist = store.fetch_df("SELECT user_id FROM canonical.blacklist_feed WHERE is_active = TRUE")
    blacklist_users = set(blacklist["user_id"].tolist())

    node_records = []
    for node_id, hop in distances.items():
        node_type, label = node_id.split(":", 1)
        risk_level = prediction_map.get(label) if node_type == "user" else None
        is_known_blacklist = node_type == "user" and label in blacklist_users
        node_records.append({
            "id": node_id,
            "type": node_type,
            "label": label,
            "hop": hop,
            "is_focus": node_id == focus_node_id,
            "risk_level": risk_level,
            "is_known_blacklist": is_known_blacklist,
        })

    node_records.sort(key=lambda item: (item["hop"], item["id"]))
    original_node_count = len(node_records)
    kept_node_records = node_records[:120]
    kept_node_ids = {item["id"] for item in kept_node_records}
    if focus_node_id not in kept_node_ids:
        kept_node_ids.add(focus_node_id)
        kept_node_records = [item for item in node_records if item["id"] in kept_node_ids]
        kept_node_records.sort(key=lambda item: (item["hop"], item["id"]))

    graph_edges = graph_edges[
        graph_edges["src_node"].isin(kept_node_ids)
        & graph_edges["dst_node"].isin(kept_node_ids)
    ].copy()
    graph_edges = graph_edges.sort_values("edge_id")
    original_edge_count = len(graph_edges)
    if original_edge_count > 240:
        graph_edges = graph_edges.head(240).copy()
    used_node_ids = set(graph_edges["src_node"]).union(set(graph_edges["dst_node"]))
    used_node_ids.add(focus_node_id)
    final_nodes = [item for item in kept_node_records if item["id"] in used_node_ids]
    final_nodes.sort(key=lambda item: (item["hop"], item["id"]))

    final_edges = graph_edges.apply(
        lambda row: {
            "id": row["edge_id"],
            "source": row["src_node"],
            "target": row["dst_node"],
            "relation_type": row["relation_type"],
        },
        axis=1,
    ).tolist()

    return {
        "focus_user_id": user_id,
        "summary": {
            "node_count": len(final_nodes),
            "edge_count": len(final_edges),
            "blacklist_neighbor_count": sum(
                1 for node in final_nodes if node["type"] == "user" and not node["is_focus"] and node["is_known_blacklist"]
            ),
            "high_risk_neighbor_count": sum(
                1
                for node in final_nodes
                if node["type"] == "user" and not node["is_focus"] and node["risk_level"] in {"high", "critical"}
            ),
            "is_truncated": original_node_count > 120 or original_edge_count > 240,
        },
        "nodes": final_nodes,
        "edges": final_edges,
    }


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/pipeline/sync")
def pipeline_sync(payload: SyncRequest) -> dict[str, Any]:
    sync_run_id = run_sync(
        full=payload.full,
        start_time=None if payload.start_time is None else __import__("datetime").datetime.fromisoformat(payload.start_time),
        end_time=None if payload.end_time is None else __import__("datetime").datetime.fromisoformat(payload.end_time),
    )
    return {"sync_run_id": sync_run_id}


@app.post("/features/rebuild")
def rebuild_features() -> dict[str, int]:
    graph_df = build_graph_features()
    day_df, rolling_df = build_feature_snapshots()
    return {
        "graph_feature_rows": len(graph_df),
        "user_day_rows": len(day_df),
        "user_30d_rows": len(rolling_df),
    }


@app.post("/model/train")
def model_train() -> dict[str, Any]:
    model_info = train_model()
    anomaly_info = train_anomaly_model()
    validation = validate_model()
    return {
        "model": model_info,
        "anomaly_model": anomaly_info,
        "validation_summary": {
            "precision": validation["precision"],
            "recall": validation["recall"],
            "f1": validation["f1"],
            "fpr": validation["fpr"],
        },
    }


@app.post("/model/score")
def model_score() -> dict[str, Any]:
    predictions = score_latest_snapshot()
    return {"rows": len(predictions), "high_risk": int((predictions["risk_level"].isin(["high", "critical"])).sum())}


@app.get("/alerts")
def list_alerts(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
    risk_level: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    settings = load_settings()
    store = DuckDBStore(settings.db_path)
    alerts = store.read_table("ops.alerts")
    if alerts.empty:
        return {"items": [], "page": page, "page_size": page_size, "total": 0, "has_next": False}
    predictions = store.fetch_df("SELECT user_id, snapshot_date, risk_score FROM ops.model_predictions")
    if not predictions.empty:
        alerts = alerts.merge(predictions, on=["user_id", "snapshot_date"], how="left")
    alerts["created_at"] = alerts["created_at"].astype(str)
    if risk_level is not None:
        alerts = alerts[alerts["risk_level"] == risk_level]
    if status is not None:
        alerts = alerts[alerts["status"] == status]
    alerts = alerts.sort_values("created_at", ascending=False)
    total = len(alerts)
    start = (page - 1) * page_size
    items = alerts.iloc[start:start + page_size].to_dict(orient="records")
    return {"items": items, "page": page, "page_size": page_size, "total": total, "has_next": start + page_size < total}


@app.get("/alerts/{alert_id}/report")
def alert_report(alert_id: str) -> dict[str, Any]:
    settings = load_settings()
    store = DuckDBStore(settings.db_path)
    alert = store.fetch_df("SELECT * FROM ops.alerts WHERE alert_id = ?", (alert_id,))
    if alert.empty:
        raise HTTPException(status_code=404, detail="alert not found")
    user_id = alert.iloc[0]["user_id"]
    report = build_risk_diagnosis(user_id)
    case = store.fetch_df("SELECT * FROM ops.cases WHERE alert_id = ? ORDER BY created_at DESC LIMIT 1", (alert_id,))
    case_actions = pd.DataFrame()
    if not case.empty:
        case_actions = store.fetch_df(
            """
            SELECT action_type, actor, action_at, note
            FROM ops.case_actions
            WHERE case_id = ?
            ORDER BY action_at DESC
            LIMIT 20
            """,
            (case.iloc[0]["case_id"],),
        )
    report["alert"] = _row_to_dict(alert[["alert_id", "status", "risk_level", "created_at"]])
    report["case"] = _row_to_dict(case[["case_id", "status", "latest_decision", "created_at"]]) if not case.empty else None
    report["case_actions"] = _records_to_dicts(case_actions)
    report["allowed_decisions"] = ALLOWED_DECISIONS
    return report


@app.get("/users/{user_id}/360")
def user_360(user_id: str) -> dict[str, Any]:
    settings = load_settings()
    store = DuckDBStore(settings.db_path)
    user = store.fetch_df("SELECT * FROM canonical.users WHERE user_id = ?", (user_id,))
    if user.empty:
        raise HTTPException(status_code=404, detail="user not found")
    prediction = store.fetch_df(
        "SELECT * FROM ops.model_predictions WHERE user_id = ? ORDER BY snapshot_date DESC LIMIT 1",
        (user_id,),
    )
    features = store.fetch_df(
        "SELECT * FROM features.feature_snapshots_user_day WHERE user_id = ? ORDER BY snapshot_date DESC LIMIT 1",
        (user_id,),
    )
    cases = store.fetch_df(
        "SELECT * FROM ops.cases WHERE user_id = ? ORDER BY created_at DESC LIMIT 5",
        (user_id,),
    )
    recent_login = store.fetch_df(
        "SELECT * FROM canonical.login_events WHERE user_id = ? ORDER BY occurred_at DESC LIMIT 10",
        (user_id,),
    )
    recent_crypto = store.fetch_df(
        "SELECT * FROM canonical.crypto_transactions WHERE user_id = ? ORDER BY occurred_at DESC LIMIT 10",
        (user_id,),
    )
    return {
        "user": user.iloc[0].to_dict(),
        "latest_prediction": None if prediction.empty else prediction.iloc[0].to_dict(),
        "latest_features": None if features.empty else features.iloc[0].to_dict(),
        "recent_login_events": recent_login.to_dict(orient="records"),
        "recent_crypto_transactions": recent_crypto.to_dict(orient="records"),
        "cases": cases.to_dict(orient="records"),
    }


@app.get("/users/{user_id}/graph")
def user_graph(user_id: str, max_hops: int = Query(default=2, ge=1, le=2)) -> dict[str, Any]:
    settings = load_settings()
    store = DuckDBStore(settings.db_path)
    return _build_graph_payload(store, user_id, max_hops=max_hops)


@app.post("/alerts/{alert_id}/decision")
def alert_decision(alert_id: str, payload: DecisionRequest) -> dict[str, str]:
    try:
        return apply_case_decision(alert_id, payload.decision, actor=payload.actor, note=payload.note)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/metrics/model")
def model_metrics() -> dict[str, Any]:
    settings = load_settings()
    store = DuckDBStore(settings.db_path)
    reports = store.read_table("ops.validation_reports")
    if reports.empty:
        raise HTTPException(status_code=404, detail="validation report not found")
    latest = reports.sort_values("created_at", ascending=False).iloc[0]
    return json.loads(latest["metrics_json"])


@app.get("/metrics/threshold")
def threshold_metrics() -> list[dict[str, Any]]:
    report = model_metrics()
    return report["threshold_sensitivity"]
