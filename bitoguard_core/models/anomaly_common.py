from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import load_settings
from db.store import DuckDBStore
from models.common import encode_features, feature_columns


ANOMALY_SOURCE_TABLE = "features.feature_snapshots_user_anomaly_30d"
ANOMALY_NON_FEATURE_COLUMNS = {
    "feature_snapshot_id",
    "user_id",
    "snapshot_date",
    "feature_version",
}
ANOMALY_LOG_FEATURES = {
    "fiat_in_1d": "fiat_in_1d_log",
    "fiat_in_7d": "fiat_in_7d_log",
    "fiat_in_30d": "fiat_in_30d_log",
    "fiat_out_7d": "fiat_out_7d_log",
    "fiat_out_30d": "fiat_out_30d_log",
    "trade_count_7d": "trade_count_7d_log",
    "trade_count_30d": "trade_count_30d_log",
    "trade_notional_7d": "trade_notional_7d_log",
    "trade_notional_30d": "trade_notional_30d_log",
    "crypto_withdraw_1d": "crypto_withdraw_1d_log",
    "crypto_withdraw_7d": "crypto_withdraw_7d_log",
    "crypto_withdraw_30d": "crypto_withdraw_30d_log",
    "login_count_1d": "login_count_1d_log",
    "login_count_7d": "login_count_7d_log",
    "login_count_30d": "login_count_30d_log",
    "unique_ip_count_7d": "unique_ip_count_7d_log",
    "unique_ip_count_30d": "unique_ip_count_30d_log",
    "unique_counterparty_wallet_count_30d": "unique_counterparty_wallet_count_30d_log",
    "avg_dwell_time": "avg_dwell_time_log",
    "large_deposit_withdraw_gap": "large_deposit_withdraw_gap_log",
    "geo_jump_count": "geo_jump_count_log",
    "ip_country_switch_count": "ip_country_switch_count_log",
    "shared_device_count": "shared_device_count_log",
    "shared_bank_count": "shared_bank_count_log",
    "shared_wallet_count": "shared_wallet_count_log",
    "blacklist_1hop_count": "blacklist_1hop_count_log",
    "blacklist_2hop_count": "blacklist_2hop_count_log",
    "component_size": "component_size_log",
}
ANOMALY_PASSTHROUGH_FEATURES = [
    "fiat_in_spike_ratio",
    "fiat_out_spike_ratio",
    "trade_notional_spike_ratio",
    "crypto_withdraw_spike_ratio",
    "login_spike_ratio",
    "new_counterparty_wallet_ratio_30d",
    "fiat_in_to_crypto_out_2h",
    "fiat_in_to_crypto_out_6h",
    "fiat_in_to_crypto_out_24h",
    "vpn_ratio",
    "new_device_ratio",
    "night_login_ratio",
    "night_large_withdrawal_ratio",
    "new_device_withdrawal_24h",
    "fan_out_ratio",
]
ANOMALY_PEER_FEATURE_NAMES = [
    "peer_trade_notional_robust_z",
    "peer_crypto_withdraw_robust_z",
    "peer_login_geo_robust_z",
    "peer_graph_connectivity_robust_z",
]
ANOMALY_MODEL_FEATURE_COLUMNS = list(ANOMALY_LOG_FEATURES.values()) + ANOMALY_PASSTHROUGH_FEATURES + ANOMALY_PEER_FEATURE_NAMES
COHORT_MIN_COUNT = 50
REFERENCE_QUANTILES = np.linspace(0.0, 1.0, 1001)


def has_transform_metadata(meta: dict[str, Any]) -> bool:
    required = {"feature_columns", "clip_bounds", "cohort_stats", "global_stats", "score_reference_quantiles"}
    return required.issubset(meta)


def load_anomaly_source_table() -> pd.DataFrame:
    settings = load_settings()
    store = DuckDBStore(settings.db_path)
    frame = store.read_table(ANOMALY_SOURCE_TABLE)
    if frame.empty:
        return frame
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"])
    return frame


def load_user_cohort_frame() -> pd.DataFrame:
    settings = load_settings()
    store = DuckDBStore(settings.db_path)
    users = store.read_table("canonical.users")[["user_id", "kyc_level", "segment"]].copy()
    users["kyc_level"] = users["kyc_level"].fillna("unknown").astype(str).str.strip().str.lower().replace("", "unknown")
    users["segment"] = users["segment"].fillna("unknown").astype(str).str.strip().str.lower().replace("", "unknown")
    return users.drop_duplicates(subset=["user_id"], keep="last")


def anomaly_training_dataset() -> pd.DataFrame:
    settings = load_settings()
    store = DuckDBStore(settings.db_path)
    features = load_anomaly_source_table()
    labels = store.read_table("ops.oracle_user_labels")
    if features.empty:
        return features
    dataset = features.merge(labels[["user_id", "hidden_suspicious_label", "scenario_types"]], on="user_id", how="left")
    dataset["hidden_suspicious_label"] = dataset["hidden_suspicious_label"].fillna(0).astype(int)
    dataset["scenario_types"] = dataset["scenario_types"].fillna("")
    return dataset


def _cohort_key(frame: pd.DataFrame) -> pd.Series:
    return frame["kyc_level"].fillna("unknown").astype(str) + "|" + frame["segment"].fillna("unknown").astype(str)


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _mad(series: pd.Series) -> float:
    centered = series - float(series.median())
    return float(centered.abs().median())


def _clip_bounds(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    bounds: dict[str, dict[str, float]] = {}
    for column in ANOMALY_LOG_FEATURES:
        series = _safe_numeric(frame[column]).clip(lower=0.0)
        lower = float(series.quantile(0.01))
        upper = float(series.quantile(0.99))
        if upper < lower:
            upper = lower
        bounds[column] = {"lower": lower, "upper": upper}
    return bounds


def _peer_metric_series(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "peer_trade_notional_robust_z": _safe_numeric(frame["trade_notional_30d"]).clip(lower=0.0),
        "peer_crypto_withdraw_robust_z": _safe_numeric(frame["crypto_withdraw_30d"]).clip(lower=0.0),
        "peer_login_geo_robust_z": (_safe_numeric(frame["geo_jump_count"]) + _safe_numeric(frame["ip_country_switch_count"])).clip(lower=0.0),
        "peer_graph_connectivity_robust_z": (
            _safe_numeric(frame["shared_device_count"])
            + _safe_numeric(frame["shared_bank_count"])
            + _safe_numeric(frame["shared_wallet_count"])
            + (_safe_numeric(frame["blacklist_1hop_count"]) * 2.0)
            + _safe_numeric(frame["blacklist_2hop_count"])
            + np.log1p(_safe_numeric(frame["component_size"]).clip(lower=0.0))
        ),
    }


def fit_anomaly_transform_metadata(source_frame: pd.DataFrame, user_cohort_frame: pd.DataFrame) -> dict[str, Any]:
    joined = source_frame[["user_id"]].merge(user_cohort_frame, on="user_id", how="left")
    joined["cohort_key"] = _cohort_key(joined)
    metric_values = _peer_metric_series(source_frame)
    global_stats: dict[str, dict[str, float | int]] = {}
    cohort_stats: dict[str, dict[str, dict[str, float | int]]] = {}
    for feature_name, values in metric_values.items():
        series = pd.Series(values, index=source_frame.index, dtype=float)
        global_stats[feature_name] = {
            "median": float(series.median()),
            "mad": _mad(series),
            "count": int(len(series)),
        }
        metric_frame = pd.DataFrame({
            "cohort_key": joined["cohort_key"],
            "value": series,
        })
        grouped = metric_frame.groupby("cohort_key")["value"]
        cohort_stats[feature_name] = {}
        for cohort_key, group in grouped:
            cohort_stats[feature_name][str(cohort_key)] = {
                "median": float(group.median()),
                "mad": _mad(group),
                "count": int(len(group)),
            }
    return {
        "feature_columns": ANOMALY_MODEL_FEATURE_COLUMNS,
        "clip_bounds": _clip_bounds(source_frame),
        "cohort_stats": cohort_stats,
        "global_stats": global_stats,
        "cohort_definition": {
            "fields": ["kyc_level", "segment"],
            "min_count": COHORT_MIN_COUNT,
        },
    }


def transform_anomaly_source_frame(
    source_frame: pd.DataFrame,
    user_cohort_frame: pd.DataFrame,
    meta: dict[str, Any],
) -> pd.DataFrame:
    cohort_min_count = int(meta.get("cohort_definition", {}).get("min_count", COHORT_MIN_COUNT))
    output = pd.DataFrame(index=source_frame.index)
    for source_column, feature_column in ANOMALY_LOG_FEATURES.items():
        bounds = meta["clip_bounds"][source_column]
        clipped = _safe_numeric(source_frame[source_column]).clip(lower=bounds["lower"], upper=bounds["upper"])
        output[feature_column] = np.log1p(clipped)

    for column in ANOMALY_PASSTHROUGH_FEATURES:
        output[column] = _safe_numeric(source_frame[column])

    joined = source_frame[["user_id"]].merge(user_cohort_frame, on="user_id", how="left")
    joined["cohort_key"] = _cohort_key(joined)
    metric_values = _peer_metric_series(source_frame)
    for feature_name, values in metric_values.items():
        global_stat = meta["global_stats"][feature_name]
        global_median = float(global_stat["median"])
        global_mad = float(global_stat["mad"])
        transformed: list[float] = []
        for cohort_key, value in zip(joined["cohort_key"], values, strict=False):
            cohort_stat = meta["cohort_stats"][feature_name].get(str(cohort_key), {})
            count = int(cohort_stat.get("count", 0))
            if count >= cohort_min_count:
                median = float(cohort_stat["median"])
                mad = float(cohort_stat["mad"])
            else:
                median = global_median
                mad = global_mad
            denom = max(mad, global_mad, 1e-6)
            transformed.append(float((float(value) - median) / denom))
        output[feature_name] = transformed

    output = output.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return output[meta["feature_columns"]].copy()


def apply_anomaly_model(
    model: object,
    source_frame: pd.DataFrame,
    user_cohort_frame: pd.DataFrame,
    meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    transformed = transform_anomaly_source_frame(source_frame, user_cohort_frame, meta)
    raw_scores = -model.score_samples(transformed)
    quantiles = np.asarray(meta["score_reference_quantiles"]["quantiles"], dtype=float)
    values = np.asarray(meta["score_reference_quantiles"]["values"], dtype=float)
    percentiles = np.interp(raw_scores, values, quantiles, left=0.0, right=1.0)
    return raw_scores, percentiles


def apply_legacy_anomaly_model(
    model: object,
    feature_frame: pd.DataFrame,
    meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    columns = feature_columns(feature_frame)
    encoded, _ = encode_features(feature_frame, columns, reference_columns=meta["encoded_columns"])
    raw_scores = -model.score_samples(encoded)
    normalized = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
    return raw_scores, normalized
