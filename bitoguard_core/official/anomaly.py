from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from official.common import RANDOM_SEED, encode_frame, feature_output_path, load_official_paths, load_pickle, save_json, save_pickle
from official.features import build_official_features
from official.graph_features import build_official_graph_features


OUTLIER_BASE_COLUMNS = [
    "twd_total_sum",
    "twd_withdraw_sum",
    "crypto_total_sum",
    "crypto_withdraw_sum",
    "order_total_sum",
    "swap_total_sum",
    "shared_ip_user_count",
    "shared_wallet_user_count",
    "relation_unique_counterparty_count",
    "crypto_unique_deposit_wallets",
    "crypto_unique_withdraw_wallets",
    "crypto_ext_ip_diversity",
]


def _load_training_frame(cutoff_tag: str) -> pd.DataFrame:
    feature_path = feature_output_path("official_user_features", cutoff_tag)
    graph_path = feature_output_path("official_graph_features", cutoff_tag)
    if not feature_path.exists():
        build_official_features(cutoff_tag=cutoff_tag)
    if not graph_path.exists():
        build_official_graph_features(cutoff_tag=cutoff_tag)
    frame = pd.read_parquet(feature_path).merge(pd.read_parquet(graph_path), on=["user_id", "snapshot_cutoff_at", "snapshot_cutoff_tag"], how="left")
    return frame


def _latest_anomaly_model() -> tuple[object, dict]:
    paths = load_official_paths()
    model_files = sorted(paths.model_dir.glob("official_iforest_*.pkl"))
    if not model_files:
        raise FileNotFoundError("No official_iforest model found")
    model_path = model_files[-1]
    meta = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    return load_pickle(model_path), meta


def score_anomaly_frame(frame: pd.DataFrame, model: object | None = None, meta: dict | None = None) -> pd.DataFrame:
    if model is None or meta is None:
        model, meta = _latest_anomaly_model()
    x_all, _ = encode_frame(frame, meta["feature_columns"], reference_columns=meta["encoded_columns"])
    raw_score = -model.score_samples(x_all)
    anomaly_score = (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min() + 1e-9)
    result = frame[["user_id", "snapshot_cutoff_at", "snapshot_cutoff_tag"]].copy()
    result["anomaly_score"] = anomaly_score
    for column in OUTLIER_BASE_COLUMNS:
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        log_series = np.log1p(series.clip(lower=0))
        median = float(log_series.median())
        mad = float((log_series - median).abs().median())
        denom = mad if mad > 1e-9 else 1.0
        result[f"{column}_robust_z"] = ((log_series - median) / denom).fillna(0.0)
        result[f"{column}_pct_rank"] = series.rank(method="average", pct=True).fillna(0.0)
    return result


def build_official_anomaly_features(cutoff_tag: str = "full") -> pd.DataFrame:
    frame = _load_training_frame(cutoff_tag)
    # v44: Add per-account-age features to capture concentrated activity patterns.
    # Per-age volume (total_value / account_age_days) is orthogonal to raw volume because
    # it normalizes by time, revealing sleeper fraudsters who move large amounts in a short
    # active window. Analysis: AUC 0.67-0.71 on labeled data, low corr with raw volumes (0.2).
    if "account_age_days" in frame.columns:
        _age_safe = frame["account_age_days"].fillna(0).clip(1).astype("float64")
        for _col, _feat_name in [
            ("crypto_total_sum", "crypto_volume_per_age"),
            ("crypto_withdraw_sum", "crypto_withdraw_per_age"),
            ("twd_total_sum", "twd_volume_per_age"),
            ("swap_total_sum", "swap_volume_per_age"),
        ]:
            if _col in frame.columns:
                frame[_feat_name] = (
                    np.log1p(frame[_col].fillna(0).clip(0).astype("float64") / _age_safe)
                ).clip(0, 15)
    # Extended feature list: original 12 + per-age ratios (when available).
    _per_age_cols = ["crypto_volume_per_age", "crypto_withdraw_per_age", "twd_volume_per_age", "swap_volume_per_age"]
    _extended_outlier_cols = OUTLIER_BASE_COLUMNS + [c for c in _per_age_cols if c in frame.columns]
    fit_frame = frame[frame["cohort"].isin(["train_only", "predict_only", "unlabeled_only"])].copy()
    # Use only the focused financial columns rather than all features.
    # IsolationForest with all ~200+ columns dilutes the anomaly signal —
    # the 12 OUTLIER_BASE_COLUMNS (+ per-age extensions) capture the core transaction-volume behaviour.
    feature_columns = [col for col in _extended_outlier_cols if col in frame.columns]
    x_fit, encoded_columns = encode_frame(fit_frame, feature_columns)
    contamination = max(0.01, float((fit_frame["status"] == 1).mean()))
    model = IsolationForest(
        n_estimators=250,
        contamination=contamination,
        random_state=RANDOM_SEED,
    )
    model.fit(x_fit)
    result = score_anomaly_frame(
        frame,
        model=model,
        meta={"feature_columns": feature_columns, "encoded_columns": encoded_columns},
    )

    # v49: Crypto-specific anomaly score — IsoForest on crypto features only.
    # FN analysis: missed positives have high crypto volume (35K vs 11K TN median)
    # but low overall anomaly_score (0.177 mean). The combined IsoForest dilutes
    # crypto signals with fiat/trading features. A crypto-focused model specifically
    # targets the "crypto-heavy non-structuring" FN fraud pattern.
    _crypto_outlier_cols = [
        "crypto_total_sum", "crypto_withdraw_sum",
        "crypto_unique_deposit_wallets", "crypto_unique_withdraw_wallets",
        "crypto_ext_ip_diversity",
    ]
    _crypto_per_age_cols = ["crypto_volume_per_age", "crypto_withdraw_per_age"]
    _crypto_feature_cols = [c for c in _crypto_outlier_cols + _crypto_per_age_cols if c in frame.columns]
    if len(_crypto_feature_cols) >= 3:
        try:
            _x_crypto_fit, _crypto_enc = encode_frame(fit_frame, _crypto_feature_cols)
            _x_crypto_all, _ = encode_frame(frame, _crypto_feature_cols, reference_columns=_crypto_enc)
            _crypto_model = IsolationForest(n_estimators=200, contamination=contamination, random_state=RANDOM_SEED + 1)
            _crypto_model.fit(_x_crypto_fit)
            _crypto_raw = -_crypto_model.score_samples(_x_crypto_all)
            result["crypto_anomaly_score"] = (
                (_crypto_raw - _crypto_raw.min()) / (_crypto_raw.max() - _crypto_raw.min() + 1e-9)
            )
        except Exception as _e:
            import logging as _logging
            _logging.getLogger(__name__).warning("crypto_anomaly_score failed: %s", _e)
            result["crypto_anomaly_score"] = 0.0
    else:
        result["crypto_anomaly_score"] = 0.0

    # v49: Segment-aware anomaly score — separate IsoForest for connected vs isolated users.
    # Connected users (sharing wallets/IPs with others) have different transaction distributions
    # than isolated users. Training separate models per segment improves calibration within
    # each population, potentially increasing anomaly AP for isolated-user FN detection.
    _connected_mask = (
        (frame.get("wallet_max_entity_user_count", pd.Series(0, index=frame.index)).fillna(0) > 1)
        | (frame.get("ip_max_entity_user_count", pd.Series(0, index=frame.index)).fillna(0) > 1)
        | (frame.get("relation_unique_counterparty_count", pd.Series(0, index=frame.index)).fillna(0) > 0)
    )
    result["anomaly_score_segmented"] = result["anomaly_score"].copy()
    for _seg_label, _seg_mask in [("connected", _connected_mask), ("isolated", ~_connected_mask)]:
        _seg_fit = fit_frame[_connected_mask.loc[fit_frame.index] if _seg_label == "connected" else ~_connected_mask.loc[fit_frame.index]]
        _seg_all = frame[_connected_mask if _seg_label == "connected" else ~_connected_mask]
        if len(_seg_fit) < 50 or len(_seg_all) == 0:
            continue
        try:
            _x_seg_fit, _seg_enc = encode_frame(_seg_fit, feature_columns)
            _x_seg_all, _ = encode_frame(_seg_all, feature_columns, reference_columns=_seg_enc)
            _seg_cont = max(0.01, float((_seg_fit["status"] == 1).mean()) if "status" in _seg_fit.columns else contamination)
            _seg_model = IsolationForest(n_estimators=200, contamination=_seg_cont, random_state=RANDOM_SEED + 2)
            _seg_model.fit(_x_seg_fit)
            _seg_raw = -_seg_model.score_samples(_x_seg_all)
            _seg_score = (_seg_raw - _seg_raw.min()) / (_seg_raw.max() - _seg_raw.min() + 1e-9)
            result.loc[_seg_all.index, "anomaly_score_segmented"] = _seg_score
        except Exception as _e:
            import logging as _logging
            _logging.getLogger(__name__).warning("anomaly_score_segmented[%s] failed: %s", _seg_label, _e)

    # LOF: local outlier factor — captures density-based anomalies IsoForest misses.
    # novelty=True required for score_samples() on unseen data.
    x_all, _ = encode_frame(frame, feature_columns, reference_columns=encoded_columns)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True)
    lof.fit(x_fit)
    lof_raw = -lof.score_samples(x_all)
    result["lof_score"] = (lof_raw - lof_raw.min()) / (lof_raw.max() - lof_raw.min() + 1e-9)

    # OCSVM: one-class SVM — captures non-linear decision boundaries.
    ocsvm = OneClassSVM(kernel="rbf", nu=min(contamination, 0.5), gamma="scale")
    ocsvm.fit(x_fit)
    ocsvm_raw = -ocsvm.score_samples(x_all)
    result["ocsvm_score"] = (ocsvm_raw - ocsvm_raw.min()) / (ocsvm_raw.max() - ocsvm_raw.min() + 1e-9)

    paths = load_official_paths()
    version = f"official_iforest_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    model_path = paths.model_dir / f"{version}.pkl"
    meta_path = paths.model_dir / f"{version}.json"
    save_pickle(model, model_path)
    save_json(
        {
            "model_version": version,
            "cutoff_tag": cutoff_tag,
            "feature_columns": feature_columns,
            "encoded_columns": encoded_columns,
            "contamination": contamination,
            "fit_row_count": int(len(fit_frame)),
            "fit_cohorts": ["train_only", "predict_only", "unlabeled_only"],
        },
        meta_path,
    )
    result.to_parquet(feature_output_path("official_anomaly_features", cutoff_tag), index=False)
    return result


def main() -> None:
    build_official_anomaly_features()


if __name__ == "__main__":
    main()
