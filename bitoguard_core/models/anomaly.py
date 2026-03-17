from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from sklearn.ensemble import IsolationForest

from config import load_settings
from db.store import DuckDBStore
from models.anomaly_common import (
    fit_anomaly_transform_metadata,
    load_anomaly_source_table,
    load_user_cohort_frame,
)
from models.common import forward_date_splits, model_dir, save_iforest, save_json


_RAW_FEATURE_COLUMNS = [
    # Fiat activity (30-day window)
    "fiat_tx_count", "fiat_in_twd", "fiat_out_twd",
    "fiat_unique_banks", "fiat_max_single_twd",
    # Crypto activity (30-day window)
    "crypto_tx_count", "crypto_in_twd", "crypto_out_twd",
    "crypto_unique_wallets", "crypto_unique_assets",
    # Login activity (30-day window)
    "login_count", "login_unique_ips", "login_unique_devices",
    "login_vpn_count", "login_new_device_count", "login_geo_jump_count",
]


def build_raw_iforest_features(
    store: DuckDBStore,
    snapshot_date: pd.Timestamp,
    user_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Compute first-order raw aggregates from canonical tables (30-day window).

    These are intentionally minimal — count, sum, max only — so that
    IsolationForest detects volume/frequency anomalies that are orthogonal
    to the engineered features used by the supervised stacker.

    Returns one row per user_id with columns in _RAW_FEATURE_COLUMNS.
    Users with no activity in the window receive zeros.
    """
    window_start = snapshot_date - pd.Timedelta(days=30)

    uid_clause = ""
    params_base: tuple = (window_start, snapshot_date)
    if user_ids:
        placeholders = ", ".join("?" * len(user_ids))
        uid_clause = f"AND user_id IN ({placeholders})"
        params_base = (window_start, snapshot_date, *user_ids)

    fiat = store.fetch_df(f"""
        SELECT
            user_id,
            COUNT(*)                                                            AS fiat_tx_count,
            COALESCE(SUM(CASE WHEN direction='deposit'    THEN amount_twd END), 0) AS fiat_in_twd,
            COALESCE(SUM(CASE WHEN direction='withdrawal' THEN amount_twd END), 0) AS fiat_out_twd,
            COUNT(DISTINCT bank_account_id)                                     AS fiat_unique_banks,
            MAX(amount_twd)                                                     AS fiat_max_single_twd
        FROM canonical.fiat_transactions
        WHERE occurred_at >= ? AND occurred_at < ?
        {uid_clause}
        GROUP BY user_id
    """, params_base)

    crypto = store.fetch_df(f"""
        SELECT
            user_id,
            COUNT(*)                                                                      AS crypto_tx_count,
            COALESCE(SUM(CASE WHEN direction='deposit'    THEN amount_twd_equiv END), 0) AS crypto_in_twd,
            COALESCE(SUM(CASE WHEN direction='withdrawal' THEN amount_twd_equiv END), 0) AS crypto_out_twd,
            COUNT(DISTINCT counterparty_wallet_id)                                        AS crypto_unique_wallets,
            COUNT(DISTINCT asset)                                                         AS crypto_unique_assets
        FROM canonical.crypto_transactions
        WHERE occurred_at >= ? AND occurred_at < ?
        {uid_clause}
        GROUP BY user_id
    """, params_base)

    login = store.fetch_df(f"""
        SELECT
            user_id,
            COUNT(*)                       AS login_count,
            COUNT(DISTINCT ip_address)     AS login_unique_ips,
            COUNT(DISTINCT device_id)      AS login_unique_devices,
            SUM(CAST(is_vpn         AS INTEGER)) AS login_vpn_count,
            SUM(CAST(is_new_device  AS INTEGER)) AS login_new_device_count,
            SUM(CAST(is_geo_jump    AS INTEGER)) AS login_geo_jump_count
        FROM canonical.login_events
        WHERE occurred_at >= ? AND occurred_at < ?
        {uid_clause}
        GROUP BY user_id
    """, params_base)

    # Get the universe of user_ids to score
    if user_ids:
        base = pd.DataFrame({"user_id": user_ids})
    else:
        all_users = store.fetch_df("SELECT DISTINCT user_id FROM canonical.users")
        base = all_users[["user_id"]].copy()

    result = (
        base
        .merge(fiat,   on="user_id", how="left")
        .merge(crypto, on="user_id", how="left")
        .merge(login,  on="user_id", how="left")
        .fillna(0)
    )
    # Ensure all expected columns are present even if a table was empty
    for col in _RAW_FEATURE_COLUMNS:
        if col not in result.columns:
            result[col] = 0

    return result[["user_id"] + _RAW_FEATURE_COLUMNS]


def train_anomaly_model() -> dict:
    """Train IsolationForest using the anomaly_common 47-feature pipeline.

    Uses feature_snapshots_user_anomaly_30d (built by build_anomaly_feature_snapshots)
    as the training source so the model is compatible with apply_anomaly_model().
    Saves full transform metadata (clip_bounds, cohort_stats, global_stats,
    score_reference_quantiles) alongside the model.
    """
    import numpy as np

    from models.anomaly_common import (
        ANOMALY_MODEL_FEATURE_COLUMNS,
        REFERENCE_QUANTILES,
        transform_anomaly_source_frame,
    )

    source_frame = load_anomaly_source_table()
    if source_frame.empty:
        raise ValueError(
            "feature_snapshots_user_anomaly_30d is empty — run build_anomaly_feature_snapshots() first."
        )

    user_cohort_frame = load_user_cohort_frame()

    # Use latest snapshot date
    latest_date = source_frame["snapshot_date"].max()
    latest = source_frame[source_frame["snapshot_date"] == latest_date].copy().reset_index(drop=True)

    transform_meta = fit_anomaly_transform_metadata(latest, user_cohort_frame)
    X_frame = transform_anomaly_source_frame(latest, user_cohort_frame, transform_meta)
    X = X_frame.values
    print(f"IsolationForest training: {len(X):,} users, {X.shape[1]} features")

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,  # fixed domain estimate; must not be derived from labels
        random_state=42,
    )
    model.fit(X)

    # Compute reference quantile table for score-to-percentile mapping
    raw_scores = -model.score_samples(X)
    score_reference_quantiles = {
        "quantiles": REFERENCE_QUANTILES.tolist(),
        "values": np.quantile(raw_scores, REFERENCE_QUANTILES).tolist(),
    }

    version = f"iforest_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    model_path = model_dir() / f"{version}.joblib"
    meta_path  = model_dir() / f"{version}.json"
    save_iforest(model, model_path)

    save_json(
        {
            "model_version": version,
            "snapshot_date": str(latest_date.date() if hasattr(latest_date, "date") else latest_date),
            "n_training_users": int(len(X)),
            "score_reference_quantiles": score_reference_quantiles,
            **transform_meta,
        },
        meta_path,
    )
    print(f"Saved: {model_path}")
    return {
        "model_version": version,
        "model_path": str(model_path),
        "meta_path": str(meta_path),
    }


if __name__ == "__main__":
    print(train_anomaly_model())
