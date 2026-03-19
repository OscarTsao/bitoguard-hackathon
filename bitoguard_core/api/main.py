from __future__ import annotations

import hmac
import json
import os
import time
from collections import deque
from datetime import datetime
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
import pandas as pd
from pydantic import BaseModel, model_validator

from config import load_settings
from db.store import DuckDBStore
from features.build_anomaly_features import build_anomaly_feature_snapshots
from features.build_features import build_feature_snapshots
from features.graph_features import build_graph_features
from models.score import score_latest_snapshot
from models.score_official import ingest_official_scores
from models.stacker import train_stacker
from pipeline.sync import run_sync
from services.alert_engine import apply_case_decision
from services.diagnosis import build_risk_diagnosis
from services.drift import run_drift_check, run_score_drift_check
from services.model_monitor import check_model_staleness


class SyncRequest(BaseModel):
    full: bool = True
    start_time: datetime | None = None
    end_time: datetime | None = None

    @model_validator(mode="after")
    def validate_window(self) -> "SyncRequest":
        if self.full and (self.start_time is not None or self.end_time is not None):
            raise ValueError("start_time and end_time must be omitted when full=true")
        if self.start_time is not None and self.end_time is not None and self.start_time > self.end_time:
            raise ValueError("start_time must be less than or equal to end_time")
        return self


class DecisionRequest(BaseModel):
    decision: str
    actor: str = "analyst"
    note: str = ""


class PipelineRunRequest(BaseModel):
    enable_tuning: bool = False


_settings = load_settings()

app = FastAPI(title="BitoGuard Core API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_DECISIONS = [
    "confirm_suspicious",
    "dismiss_false_positive",
    "escalate",
    "request_monitoring",
]

_MAX_NEIGHBOR_IDS = 500

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _require_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    settings = load_settings()
    if settings.api_key is None:
        return  # Auth disabled in dev mode
    if api_key is None or not hmac.compare_digest(api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


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


def _load_validation_metrics(store: DuckDBStore) -> dict[str, Any]:
    settings = load_settings()
    fallback_path = settings.artifact_dir / "validation_report.json"
    fallback: dict[str, Any] = (
        json.loads(fallback_path.read_text(encoding="utf-8"))
        if fallback_path.exists() else {}
    )

    reports = store.read_table("ops.validation_reports")
    if not reports.empty:
        latest = reports.sort_values("created_at", ascending=False).iloc[0]
        payload = latest["metrics_json"]
        db_report: dict[str, Any] = json.loads(payload) if isinstance(payload, str) else payload
        # Back-fill any fields from the file report that are absent in the DB row.
        # This preserves newer fields (precision_at_k, calibration, feature_importance)
        # when the stored report was created before those fields were added.
        merged = {**fallback, **db_report}
        return merged

    if fallback:
        return fallback

    raise HTTPException(status_code=404, detail="validation report not found")


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
    neighbor_ids = set(sorted(neighbor_ids)[:_MAX_NEIGHBOR_IDS])
    placeholders = ", ".join(["?"] * len(neighbor_ids))
    nb = list(neighbor_ids)
    two_hop = store.fetch_df(
        f"SELECT * FROM canonical.entity_edges WHERE src_id IN ({placeholders}) OR dst_id IN ({placeholders})",
        tuple(nb) * 2,
    )
    return pd.concat([one_hop, two_hop], ignore_index=True).drop_duplicates(subset=["edge_id"])


def _build_graph_payload(store: DuckDBStore, user_id: str, max_hops: int, max_nodes: int = 120, max_edges: int = 240) -> dict[str, Any]:
    edges = _load_neighborhood_edges(store, user_id, max_hops)
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
    neighborhood_user_ids = sorted(
        node_id.split(":", 1)[1]
        for node_id in included_nodes
        if node_id.startswith("user:")
    )
    prediction_map: dict[str, str] = {}
    blacklist_users: set[str] = set()
    if neighborhood_user_ids:
        placeholders = ", ".join(["?"] * len(neighborhood_user_ids))
        predictions = store.fetch_df(
            f"""
            SELECT user_id, risk_level
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY snapshot_date DESC, prediction_time DESC) AS rn
                FROM ops.model_predictions
                WHERE user_id IN ({placeholders})
            )
            WHERE rn = 1
            """,
            tuple(neighborhood_user_ids),
        )
        prediction_map = {
            row["user_id"]: row["risk_level"]
            for _, row in predictions.iterrows()
        }
        blacklist = store.fetch_df(
            f"SELECT user_id FROM canonical.blacklist_feed WHERE is_active = TRUE AND user_id IN ({placeholders})",
            tuple(neighborhood_user_ids),
        )
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
    kept_node_records = node_records[:max_nodes]
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
    if original_edge_count > max_edges:
        graph_edges = graph_edges.head(max_edges).copy()
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
            "is_truncated": original_node_count > max_nodes or original_edge_count > max_edges,
        },
        "nodes": final_nodes,
        "edges": final_edges,
    }


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/pipeline/sync", dependencies=[Depends(_require_api_key)])
def pipeline_sync(payload: SyncRequest) -> dict[str, Any]:
    sync_run_id = run_sync(
        full=payload.full,
        start_time=payload.start_time,
        end_time=payload.end_time,
    )
    return {"sync_run_id": sync_run_id}


@app.post("/pipeline/run", dependencies=[Depends(_require_api_key)])
async def run_pipeline(request: PipelineRunRequest = None) -> dict[str, Any]:
    """Trigger the Step Functions ML pipeline.

    Body:
        enable_tuning: If True, runs full hyperparameter tuning (2-3h).
                       If False, uses best params from SSM (15-25 min).

    Ref: https://docs.aws.amazon.com/step-functions/latest/dg/tutorial-api-gateway.html
    """
    if request is None:
        request = PipelineRunRequest()
    arn = os.environ.get("BITOGUARD_STEP_FUNCTIONS_ARN")
    if not arn:
        raise HTTPException(
            status_code=500,
            detail="BITOGUARD_STEP_FUNCTIONS_ARN environment variable not set",
        )
    region = os.environ.get("AWS_REGION", "us-east-1")
    import boto3
    sfn = boto3.client("stepfunctions", region_name=region)
    execution_name = f"api-run-{int(time.time())}"
    resp = sfn.start_execution(
        stateMachineArn=arn,
        name=execution_name,
        input=json.dumps({"enable_tuning": request.enable_tuning}),
    )
    return {"execution_arn": resp["executionArn"]}


@app.post("/features/rebuild", dependencies=[Depends(_require_api_key)])
def rebuild_features() -> dict[str, int]:
    graph_df = build_graph_features()
    day_df, rolling_df = build_feature_snapshots()
    anomaly_df = build_anomaly_feature_snapshots()
    return {
        "graph_feature_rows": len(graph_df),
        "user_day_rows": len(day_df),
        "user_30d_rows": len(rolling_df),
        "user_anomaly_rows": len(anomaly_df),
    }


@app.post("/model/train", dependencies=[Depends(_require_api_key)])
def model_train() -> dict[str, Any]:
    result = train_stacker()
    return {
        "model": result["stacker_version"],
        "stacker_path": result["stacker_path"],
        "branch_models": result["branch_models"],
        "cv_results": result["cv_results"],
    }


@app.post("/model/score", dependencies=[Depends(_require_api_key)])
def model_score() -> dict[str, Any]:
    settings = load_settings()
    if settings.model_backend == "official":
        predictions = ingest_official_scores()
    else:
        predictions = score_latest_snapshot()
    return {"rows": len(predictions), "high_risk": int((predictions["risk_level"].isin(["high", "critical"])).sum())}


@app.get("/alerts", dependencies=[Depends(_require_api_key)])
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
    predictions = store.fetch_df("SELECT prediction_id, risk_score FROM ops.model_predictions")
    if not predictions.empty:
        alerts = alerts.merge(predictions, on="prediction_id", how="left")
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


@app.get("/alerts/{alert_id}", dependencies=[Depends(_require_api_key)])
def get_alert(alert_id: str) -> dict[str, Any]:
    settings = load_settings()
    store = DuckDBStore(settings.db_path)
    alert = store.fetch_df("SELECT * FROM ops.alerts WHERE alert_id = ? LIMIT 1", (alert_id,))
    if alert.empty:
        raise HTTPException(status_code=404, detail="alert not found")

    prediction = store.fetch_df(
        "SELECT risk_score FROM ops.model_predictions WHERE prediction_id = ? LIMIT 1",
        (alert.iloc[0]["prediction_id"],),
    )
    alert_row = alert.copy()
    alert_row["risk_score"] = None if prediction.empty else prediction.iloc[0]["risk_score"]
    alert_row["created_at"] = alert_row["created_at"].astype(str)
    return alert_row.iloc[0].to_dict()


@app.get("/alerts/{alert_id}/report", dependencies=[Depends(_require_api_key)])
def alert_report(alert_id: str) -> dict[str, Any]:
    settings = load_settings()
    store = DuckDBStore(settings.db_path)
    alert = store.fetch_df("SELECT * FROM ops.alerts WHERE alert_id = ?", (alert_id,))
    if alert.empty:
        raise HTTPException(status_code=404, detail="alert not found")
    alert_row = alert.iloc[0]
    user_id = alert_row["user_id"]
    try:
        report = build_risk_diagnosis(
            user_id,
            prediction_id=alert_row.get("prediction_id"),
            snapshot_date=alert_row.get("snapshot_date"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
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


@app.get("/users/{user_id}/360", dependencies=[Depends(_require_api_key)])
def user_360(user_id: str) -> dict[str, Any]:
    settings = load_settings()
    store = DuckDBStore(settings.db_path)
    user = store.fetch_df("SELECT * FROM canonical.users WHERE user_id = ?", (user_id,))
    if user.empty:
        raise HTTPException(status_code=404, detail="user not found")
    prediction = store.fetch_df(
        "SELECT * FROM ops.model_predictions WHERE user_id = ? ORDER BY snapshot_date DESC, prediction_time DESC LIMIT 1",
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


@app.get("/users/{user_id}/graph", dependencies=[Depends(_require_api_key)])
def user_graph(user_id: str, max_hops: int = Query(default=2, ge=1, le=2)) -> dict[str, Any]:
    settings = load_settings()
    store = DuckDBStore(settings.db_path)
    return _build_graph_payload(
        store, user_id, max_hops=max_hops,
        max_nodes=settings.graph_max_nodes,
        max_edges=settings.graph_max_edges,
    )


@app.post("/alerts/{alert_id}/decision", dependencies=[Depends(_require_api_key)])
def alert_decision(alert_id: str, payload: DecisionRequest) -> dict[str, str]:
    try:
        return apply_case_decision(alert_id, payload.decision, actor=payload.actor, note=payload.note)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/metrics/model", dependencies=[Depends(_require_api_key)])
def model_metrics() -> dict[str, Any]:
    settings = load_settings()
    store = DuckDBStore(settings.db_path)
    base = _load_validation_metrics(store)
    # Augment with OOF stacker metrics from the CV report (true generalization estimate,
    # leakage-free). The holdout report may have inflated metrics due to peer-percentile
    # feature leakage; OOF is the authoritative performance measure for production.
    #
    # Source priority:
    # 1. 5fold_cv_report_*.json — comprehensive report with oracle P@K and dataset stats
    # 2. models/cv_results_*.json — latest stacker CV results (updated after every retrain)
    cv_candidates = sorted(
        settings.artifact_dir.glob("5fold_cv_report_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if cv_candidates:
        try:
            cv_report = json.loads(cv_candidates[0].read_text(encoding="utf-8"))
            oof = cv_report.get("cv_evaluation", {}).get("oof_metrics", {})
            base["oof_metrics"] = oof
            base["oracle_precision_at_k"] = (
                cv_report.get("scoring_results", {}).get("oracle_precision_at_k", {})
            )
            base["dataset_stats"] = cv_report.get("dataset", {})
        except Exception:
            pass

    # Override oof_metrics with latest stacker cv_results if newer or 5fold report absent
    stacker_cv_candidates = sorted(
        (settings.artifact_dir / "models").glob("cv_results_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if stacker_cv_candidates:
        try:
            sc = json.loads(stacker_cv_candidates[0].read_text(encoding="utf-8"))
            # Map stacker cv_results "oof" → {"catboost": {auc, pr_auc}, "lgbm": ...,
            # "xgboost": ..., "stacker": ...} for the frontend OOF panel
            raw_oof = sc.get("oof", {})
            if raw_oof:
                if not cv_candidates or stacker_cv_candidates[0].stat().st_mtime > cv_candidates[0].stat().st_mtime:
                    base["oof_metrics"] = raw_oof
        except Exception:
            pass

    return base


@app.get("/metrics/threshold", dependencies=[Depends(_require_api_key)])
def threshold_metrics() -> list[dict[str, Any]]:
    store = DuckDBStore(load_settings().db_path)
    report = _load_validation_metrics(store)
    return report["threshold_sensitivity"]


@app.get("/metrics/drift", dependencies=[Depends(_require_api_key)])
def drift_metrics() -> dict[str, Any]:
    """Feature drift, score distribution PSI, and model staleness in one response."""
    settings = load_settings()
    result: dict[str, Any] = {}

    # Feature distribution drift
    result["feature_drift"] = run_drift_check().to_dict()

    # Score distribution PSI (between two most recent scoring runs)
    score_drift = run_score_drift_check()
    result["score_psi"] = score_drift.to_dict() if score_drift is not None else None

    # Model staleness
    bundle_path = settings.artifact_dir / "official_bundle.json"
    if bundle_path.exists():
        result["model_staleness"] = check_model_staleness(bundle_path).to_dict()
    else:
        result["model_staleness"] = None

    # Overall health: ok only if all components healthy
    feat_ok = result["feature_drift"].get("health_ok", True)
    psi_ok = score_drift.health_ok if score_drift is not None else True
    stale_ok = result["model_staleness"]["health_ok"] if result["model_staleness"] else True
    result["health_ok"] = feat_ok and psi_ok and stale_ok

    return result
