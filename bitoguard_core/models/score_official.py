"""Adapter: ingest pre-computed official pipeline scores into ops.model_predictions.

Run ``python -m official.score`` first to produce
``artifacts/predictions/official_predict_scores.parquet``.
Then call ``ingest_official_scores()`` to write them into DuckDB and generate alerts.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from config import load_settings
from db.store import DuckDBStore
from services.alert_engine import generate_alerts


def ingest_official_scores() -> pd.DataFrame:
    """Read official_predict_scores.parquet and write to ops.model_predictions.

    Idempotent: deletes existing rows for today's snapshot_date before inserting.
    """
    settings = load_settings()
    parquet_path = settings.artifact_dir / "predictions" / "official_predict_scores.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Official scores not found at {parquet_path}. "
            "Run `python -m official.score` first."
        )

    raw = pd.read_parquet(parquet_path)
    snapshot_date = date.today()
    prediction_time = datetime.now(timezone.utc).isoformat()

    rows: list[dict] = []
    for _, r in raw.iterrows():
        user_id = str(r["user_id"])
        prediction_id = f"pred_{user_id}_{snapshot_date.isoformat().replace('-', '')}"
        rule_hits = r.get("top_reason_codes", [])
        if isinstance(rule_hits, str):
            try:
                rule_hits = json.loads(rule_hits)
            except Exception:
                rule_hits = [rule_hits] if rule_hits else []
        rows.append({
            "prediction_id": prediction_id,
            "user_id": user_id,
            "snapshot_date": snapshot_date,
            "prediction_time": prediction_time,
            "model_version": "official_pipeline_v1",
            "risk_score": float(r.get("analyst_risk_score", 0.0)),
            "risk_level": str(r.get("risk_level", "low")),
            "rule_hits": json.dumps(rule_hits),
            "top_reason_codes": json.dumps(rule_hits),
            "model_probability": float(r.get("submission_probability", 0.0)),
            "anomaly_score": float(r.get("anomaly_score", 0.0)),
            "graph_risk": 0.0,
        })

    pred_frame = pd.DataFrame(rows)

    store = DuckDBStore(settings.db_path)
    with store.transaction() as conn:
        conn.execute(
            "DELETE FROM ops.model_predictions WHERE snapshot_date = ?",
            (snapshot_date,),
        )
        conn.register("pred_official", pred_frame)
        conn.execute("INSERT INTO ops.model_predictions SELECT * FROM pred_official")
        conn.unregister("pred_official")
    generate_alerts()
    return pred_frame
