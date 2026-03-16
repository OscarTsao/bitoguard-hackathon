from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_settings
from db.store import DuckDBStore, make_id, utc_now
from models.anomaly import build_raw_iforest_features
from models.common import encode_features, feature_columns, load_feature_table, load_iforest, load_joblib
from models.rule_engine import evaluate_rules
from services.alert_engine import generate_alerts


def _load_latest_model(prefix: str, extension: str) -> tuple[Path, dict]:
    settings = load_settings()
    model_files = sorted((settings.artifact_dir / "models").glob(f"{prefix}_*.{extension}"))
    if not model_files:
        raise FileNotFoundError(f"No model found for prefix={prefix}, extension={extension}")
    model_path = model_files[-1]
    meta = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    return model_path, meta


def _graph_risk_score(frame: pd.DataFrame) -> pd.Series:
    """Absolute-threshold graph risk (0–1). Reproducible across scoring batches.

    Blacklist proximity features (blacklist_1hop/2hop, shared_device_count) are
    disabled by default (graph_trusted_only=True) and contribute zero in that mode.
    shared_bank_count uses a linear scale capped at 10 accounts.
    """
    blacklist_risk = (
        (frame["blacklist_1hop_count"] > 0).astype(float) * 0.60
        + (frame["blacklist_2hop_count"] > 0).astype(float) * 0.30
    )
    device_risk = (frame["shared_device_count"].clip(0, 20) / 20.0) * 0.05
    bank_risk = (frame["shared_bank_count"].clip(0, 10) / 10.0) * 0.05
    return (blacklist_risk + device_risk + bank_risk).clip(0.0, 1.0)


def _prediction_key(user_id: str, snapshot_date: object) -> tuple[str, object]:
    return (user_id, pd.Timestamp(snapshot_date).date())


def _build_rule_compat_frame(v2_frame: pd.DataFrame) -> pd.DataFrame:
    """Map v2 feature columns to the v1 names expected by evaluate_rules().

    Rules use safe _get() so missing columns return 0. This shim provides the
    best available v2 approximation for each v1 column to preserve rule signal.
    """
    f = v2_frame.copy()

    # Velocity rules (v1: bool flags; v2: integer counts)
    f["fiat_in_to_crypto_out_2h"]  = (f.get("fiat_dep_to_swap_buy_within_1h",  pd.Series(0, index=f.index)) > 0)
    f["fiat_in_to_crypto_out_24h"] = (f.get("fiat_dep_to_swap_buy_within_24h", pd.Series(0, index=f.index)) > 0)

    # Volume proxy (v1 uses 30d sum; v2 uses lifetime sum — acceptable approximation)
    f["crypto_withdraw_30d"] = f.get("crypto_wdr_twd_sum",  pd.Series(0.0, index=f.index))
    f["fiat_in_30d"]         = f.get("twd_dep_sum",         pd.Series(0.0, index=f.index))

    # Device/IP rules — not computable from v2 features; leave as 0 (safe default)
    for col in ("new_device_withdrawal_24h", "ip_country_switch_count",
                "night_large_withdrawal_ratio", "new_device_ratio"):
        if col not in f.columns:
            f[col] = 0

    # Graph rules — bipartite features don't map directly to v1; leave as 0
    for col in ("shared_device_count", "blacklist_1hop_count",
                "blacklist_2hop_count", "component_size"):
        if col not in f.columns:
            f[col] = 0

    # Fan-out: ip_n_entities is a loose proxy for fan_out_ratio
    f["fan_out_ratio"] = f.get("ip_n_entities", pd.Series(0, index=f.index)).astype(float)

    # Declared volume mismatch (monthly_income_twd is NULL in current data → ratio=0)
    vol_total = (
        f.get("twd_all_twd_sum", pd.Series(0.0, index=f.index)) +
        f.get("crypto_all_twd_sum", pd.Series(0.0, index=f.index))
    )
    income = f.get("monthly_income_twd", pd.Series(1.0, index=f.index)).clip(lower=1.0)
    f["actual_volume_expected_ratio"] = vol_total / income

    # Peer percentiles — not computed in v2; remain 0 (rules will silently not fire)
    f["fiat_in_30d_peer_pct"]         = 0.0
    f["crypto_withdraw_30d_peer_pct"] = 0.0

    return f


def score_latest_snapshot() -> pd.DataFrame:
    """Score using stacker (CatBoost + LightGBM) over v2 feature table.

    Returns same schema as the former v1 score_latest_snapshot() for API compatibility.
    Rule engine runs via compat shim; IsolationForest provides anomaly_score.
    """
    settings = load_settings()
    store    = DuckDBStore(settings.db_path)

    features = load_feature_table("features.feature_snapshots_v2")
    if features.empty:
        raise ValueError("No v2 feature snapshots. Run 'make features-v2' first.")

    latest_date   = features["snapshot_date"].max()
    scoring_frame = features[features["snapshot_date"] == latest_date].copy()

    # Load stacker
    stacker_path, stacker_meta = _load_latest_model("stacker", "joblib")
    stacker_model = load_joblib(stacker_path)
    feature_cols  = stacker_meta["feature_columns"]
    x_score       = scoring_frame[feature_cols].fillna(0)

    cb_path   = Path(stacker_meta["branch_models"]["catboost"])
    lgbm_path = Path(stacker_meta["branch_models"]["lgbm"])
    cb_model   = load_joblib(cb_path)
    lgbm_model = load_joblib(lgbm_path)

    branch_preds = [
        cb_model.predict_proba(x_score)[:, 1],
        lgbm_model.predict_proba(x_score)[:, 1],
    ]

    # 3-branch stacker (optional XGBoost branch — present in v2 models)
    if "xgboost" in stacker_meta.get("branch_models", {}):
        xgb_path = Path(stacker_meta["branch_models"]["xgboost"])
        xgb_model = load_joblib(xgb_path)
        # XGBoost requires numeric-only features; encode categoricals as integer codes
        x_score_np = x_score.copy()
        for col in x_score_np.select_dtypes(include=["object", "category"]).columns:
            x_score_np[col] = pd.Categorical(x_score_np[col]).codes.astype("float32")
        branch_preds.append(xgb_model.predict_proba(x_score_np.values.astype("float32"))[:, 1])

    branch_matrix = np.column_stack(branch_preds)
    model_probability = stacker_model.predict_proba(branch_matrix)[:, 1]

    if settings.m4_enabled:
        try:
            iforest_path, iforest_meta = _load_latest_model("iforest", "joblib")
            iforest_model = load_iforest(iforest_path)
            raw_cols = iforest_meta.get("raw_feature_columns", [])
            if not raw_cols:
                raise ValueError("IsolationForest metadata missing 'raw_feature_columns'. Retrain with: make train-iforest")
            raw_df = build_raw_iforest_features(store, latest_date, scoring_frame["user_id"].tolist())
            x_anom = (
                raw_df.set_index("user_id")
                .reindex(scoring_frame["user_id"].values)
                .fillna(0)[raw_cols]
            )
            anomaly_raw   = -iforest_model.score_samples(x_anom.values)
            anomaly_score = (anomaly_raw - anomaly_raw.min()) / (anomaly_raw.max() - anomaly_raw.min() + 1e-9)
        except Exception:
            anomaly_score = np.zeros(len(scoring_frame))
    else:
        anomaly_score = np.zeros(len(scoring_frame))

    # Rule engine via v2→v1 compat shim
    compat_frame = _build_rule_compat_frame(scoring_frame)
    rule_results = evaluate_rules(compat_frame)

    result = scoring_frame[["user_id", "snapshot_date"]].copy()
    result["model_probability"] = model_probability
    result["anomaly_score"]     = anomaly_score
    result["graph_risk"]        = 0.0   # bipartite features absorbed into model
    result = result.merge(
        rule_results[["user_id", "snapshot_date", "rule_score", "rule_hits"]],
        on=["user_id", "snapshot_date"], how="left",
    )
    result["rule_score"] = result["rule_score"].fillna(0.0)
    result["rule_hits"]  = result["rule_hits"].fillna("[]")

    result["risk_score"] = (
        0.70 * result["model_probability"]
        + 0.10 * result["anomaly_score"]
        + 0.20 * result["rule_score"]
    ) * 100.0

    # Percentile-based risk tiering — the stacker outputs uncalibrated log-odds
    # so absolute thresholds are model-version-dependent. Capacity-based tiers
    # reflect real AML investigation throughput: top 1% = critical (≈ P@K ≥ 90%),
    # top 5% = high, top 20% = medium, rest = low.
    n = len(result)
    rank = result["risk_score"].rank(ascending=False, method="first")
    result["risk_level"] = np.select(
        [rank <= max(1, int(n * 0.01)),
         rank <= max(1, int(n * 0.05)),
         rank <= max(1, int(n * 0.20))],
        ["critical", "high", "medium"],
        default="low",
    )
    result["top_reason_codes"] = result["rule_hits"]
    result["prediction_time"]  = utc_now()
    result["model_version"]    = stacker_meta["stacker_version"]

    existing = store.fetch_df(
        "SELECT prediction_id, user_id, snapshot_date FROM ops.model_predictions WHERE snapshot_date = ?",
        (latest_date.date(),),
    )
    existing_ids = {
        _prediction_key(r["user_id"], r["snapshot_date"]): r["prediction_id"]
        for _, r in existing.iterrows()
    }
    result["prediction_id"] = result.apply(
        lambda row: existing_ids.get(
            _prediction_key(row["user_id"], row["snapshot_date"]),
            make_id(f"pred_{row['user_id'][-4:]}"),
        ), axis=1,
    )

    pred_rows = result[[
        "prediction_id", "user_id", "snapshot_date", "prediction_time", "model_version",
        "risk_score", "risk_level", "rule_hits", "top_reason_codes",
        "model_probability", "anomaly_score", "graph_risk",
    ]].copy()
    with store.transaction() as conn:
        conn.execute(
            "DELETE FROM ops.model_predictions WHERE snapshot_date = ?",
            (latest_date.date(),),
        )
        conn.register("pred_df_v2", pred_rows)
        conn.execute("INSERT INTO ops.model_predictions SELECT * FROM pred_df_v2")
        conn.unregister("pred_df_v2")
    generate_alerts()
    return pred_rows


if __name__ == "__main__":
    print(score_latest_snapshot().head())
