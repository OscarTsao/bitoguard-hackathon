from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import IsolationForest

from features.build_anomaly_features import build_anomaly_feature_snapshots
from models.ablate_iforest import run_iforest_ablation
from models.anomaly_common import anomaly_training_dataset, fit_anomaly_transform_metadata, load_user_cohort_frame, transform_anomaly_source_frame
from models.common import feature_columns, model_dir, save_json, save_pickle


def train_anomaly_model() -> dict:
    dataset = anomaly_training_dataset().sort_values("snapshot_date").reset_index(drop=True)
    if dataset.empty:
        build_anomaly_feature_snapshots()
        dataset = anomaly_training_dataset().sort_values("snapshot_date").reset_index(drop=True)
    if dataset.empty:
        raise ValueError("No anomaly feature snapshots found. Run feature build first.")
    feature_cols = feature_columns(dataset)
    unique_dates = sorted(dataset["snapshot_date"].dt.date.unique())
    train_dates = set(unique_dates[:20])
    train_frame = dataset[dataset["snapshot_date"].dt.date.isin(train_dates)].copy()
    fit_frame = train_frame[train_frame["hidden_suspicious_label"].astype(int) != 1].copy()
    if fit_frame.empty:
        raise ValueError("No normal-like rows available to fit anomaly model.")

    user_cohorts = load_user_cohort_frame()
    transform_meta = fit_anomaly_transform_metadata(fit_frame, user_cohorts)
    x_train = transform_anomaly_source_frame(fit_frame, user_cohorts, transform_meta)
    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
    )
    model.fit(x_train)
    raw_scores = -model.score_samples(x_train)
    transform_meta["score_reference_quantiles"] = {
        "quantiles": [float(item) for item in np.linspace(0.0, 1.0, 1001)],
        "values": [float(item) for item in np.quantile(raw_scores, np.linspace(0.0, 1.0, 1001))],
    }
    version = f"iforest_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    model_path = model_dir() / f"{version}.pkl"
    meta_path = model_dir() / f"{version}.json"
    save_pickle(model, model_path)
    meta = {
        "model_version": version,
        "feature_columns": transform_meta["feature_columns"],
        "source_columns": feature_cols,
        "clip_bounds": transform_meta["clip_bounds"],
        "cohort_stats": transform_meta["cohort_stats"],
        "global_stats": transform_meta["global_stats"],
        "score_reference_quantiles": transform_meta["score_reference_quantiles"],
        "cohort_definition": transform_meta["cohort_definition"],
        "train_dates": sorted(str(item) for item in train_dates),
        "fit_row_count": int(len(fit_frame)),
        "excluded_suspicious_rows": int(train_frame["hidden_suspicious_label"].astype(int).eq(1).sum()),
    }
    save_json(meta, meta_path)
    ablation_summary: dict[str, object]
    try:
        ablation_report = run_iforest_ablation()
        ablation_summary = {
            "report_path": str(ablation_report["report_path"]),
            "score_path": str(ablation_report["score_path"]),
            "gate_verdict": ablation_report["recommendation"]["verdict"],
            "gate_passed": bool(ablation_report["recommendation"]["gate_passed"]),
        }
    except FileNotFoundError as exc:
        ablation_summary = {"status": "skipped", "reason": str(exc)}
    return {
        "model_version": version,
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "fit_row_count": meta["fit_row_count"],
        "feature_columns": meta["feature_columns"],
        "score_reference_quantiles": meta["score_reference_quantiles"],
        "ablation": ablation_summary,
    }


if __name__ == "__main__":
    print(train_anomaly_model())
