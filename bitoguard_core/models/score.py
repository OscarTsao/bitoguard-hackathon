from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import load_settings
from db.store import DuckDBStore, make_id, utc_now
from models.common import encode_features, feature_columns, load_feature_table, load_pickle
from models.rule_engine import evaluate_rules
from services.alert_engine import generate_alerts


def _load_latest_model(prefix: str) -> tuple[Path, dict]:
    settings = load_settings()
    model_files = sorted((settings.artifact_dir / "models").glob(f"{prefix}_*.pkl"))
    if not model_files:
        raise FileNotFoundError(f"No model found for prefix={prefix}")
    model_path = model_files[-1]
    meta = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    return model_path, meta


def _graph_risk_score(frame: pd.DataFrame) -> pd.Series:
    raw = (
        frame["blacklist_1hop_count"] * 0.6
        + frame["blacklist_2hop_count"] * 0.4
        + frame["shared_device_count"] * 0.05
        + frame["shared_bank_count"] * 0.05
    )
    return raw.clip(lower=0).pipe(lambda s: s / max(1.0, s.max()))


def _prediction_key(user_id: str, snapshot_date: object) -> tuple[str, object]:
    return (user_id, pd.Timestamp(snapshot_date).date())


def score_latest_snapshot() -> pd.DataFrame:
    settings = load_settings()
    store = DuckDBStore(settings.db_path)
    features = load_feature_table("features.feature_snapshots_user_day")
    if features.empty:
        raise ValueError("No feature snapshots found. Run feature build first.")
    latest_date = features["snapshot_date"].max()
    scoring_frame = features[features["snapshot_date"] == latest_date].copy()
    rule_results = evaluate_rules(scoring_frame)

    lgbm_path, lgbm_meta = _load_latest_model("lgbm")
    anomaly_path, anomaly_meta = _load_latest_model("iforest")
    feature_cols = feature_columns(scoring_frame)
    x_score, _ = encode_features(scoring_frame, feature_cols, reference_columns=lgbm_meta["encoded_columns"])
    x_anomaly, _ = encode_features(scoring_frame, feature_cols, reference_columns=anomaly_meta["encoded_columns"])

    lgbm = load_pickle(lgbm_path)
    anomaly_model = load_pickle(anomaly_path)
    model_probability = lgbm.predict_proba(x_score)[:, 1]
    anomaly_raw = -anomaly_model.score_samples(x_anomaly)
    anomaly_score = (anomaly_raw - anomaly_raw.min()) / (anomaly_raw.max() - anomaly_raw.min() + 1e-9)
    graph_risk = _graph_risk_score(scoring_frame)

    result = scoring_frame[["user_id", "snapshot_date"]].copy()
    result["model_probability"] = model_probability
    result["anomaly_score"] = anomaly_score
    result["graph_risk"] = graph_risk
    result = result.merge(rule_results[["user_id", "snapshot_date", "rule_score", "rule_hits"]], on=["user_id", "snapshot_date"], how="left")
    result["risk_score"] = (
        0.35 * result["rule_score"]
        + 0.45 * result["model_probability"]
        + 0.10 * result["anomaly_score"]
        + 0.10 * result["graph_risk"]
    ) * 100.0
    result["risk_level"] = pd.cut(
        result["risk_score"],
        bins=[-1, 35, 60, 80, 100],
        labels=["low", "medium", "high", "critical"],
    ).astype(str)
    result["top_reason_codes"] = result["rule_hits"]
    result["prediction_time"] = utc_now()
    result["model_version"] = lgbm_meta["model_version"]

    existing_predictions = store.fetch_df(
        "SELECT prediction_id, user_id, snapshot_date FROM ops.model_predictions WHERE snapshot_date = ?",
        (latest_date.date(),),
    )
    existing_prediction_ids = {
        _prediction_key(row["user_id"], row["snapshot_date"]): row["prediction_id"]
        for _, row in existing_predictions.iterrows()
    }
    result["prediction_id"] = result.apply(
        lambda row: existing_prediction_ids.get(
            _prediction_key(row["user_id"], row["snapshot_date"]),
            make_id(f"pred_{row['user_id'][-4:]}"),
        ),
        axis=1,
    )

    prediction_rows = result[[
        "prediction_id", "user_id", "snapshot_date", "prediction_time", "model_version",
        "risk_score", "risk_level", "rule_hits", "top_reason_codes",
        "model_probability", "anomaly_score", "graph_risk",
    ]].copy()
    store.execute("DELETE FROM ops.model_predictions WHERE snapshot_date = ?", (latest_date.date(),))
    store.append_dataframe("ops.model_predictions", prediction_rows)
    generate_alerts()
    return prediction_rows


if __name__ == "__main__":
    print(score_latest_snapshot().head())
