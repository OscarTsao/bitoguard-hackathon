from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score

from official.anomaly import score_anomaly_frame
from official.bundle import load_selected_bundle, save_selected_bundle
from official.calibration import choose_calibrator
from official.common import default_temporal_cutoff, encode_frame, feature_report_path, load_pickle, save_json
from official.features import build_official_features
from official.graph_features import build_official_graph_features
from official.rules import evaluate_official_rules
from official.thresholding import search_threshold
from official.train import _load_dataset


def _load_selected_model(bundle: dict[str, Any]) -> tuple[object, dict[str, Any]]:
    if bundle["selected_model"] != "lgbm":
        raise NotImplementedError(f"Selected model not yet supported: {bundle['selected_model']}")
    model_path = bundle["model_paths"]["lgbm"]
    model = load_pickle(Path(model_path))
    meta = json.loads(Path(model_path).with_suffix(".json").read_text(encoding="utf-8"))
    return model, meta


def _predict_raw_probability(frame: pd.DataFrame, bundle: dict[str, Any], model: object) -> np.ndarray:
    feature_columns = bundle["feature_columns_lgbm"]
    encoded_columns = bundle["encoded_columns_lgbm"]
    encoded, _ = encode_frame(frame, feature_columns, reference_columns=encoded_columns)
    return model.predict_proba(encoded)[:, 1]


def _classification_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    preds = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "fpr": float(fp / max(1, fp + tn)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _temporal_stress_metrics(bundle: dict[str, Any], model: object, calibrator: Any, threshold: float, split_frame: pd.DataFrame) -> dict[str, Any]:
    cutoff = default_temporal_cutoff()
    temporal_features = build_official_features(cutoff_ts=cutoff, cutoff_tag="temporal")
    temporal_graph = build_official_graph_features(cutoff_ts=cutoff, cutoff_tag="temporal")
    temporal = temporal_features.merge(temporal_graph, on=["user_id", "snapshot_cutoff_at", "snapshot_cutoff_tag"], how="left")
    anomaly = score_anomaly_frame(temporal).drop(columns=["snapshot_cutoff_at", "snapshot_cutoff_tag"])
    temporal = temporal.merge(anomaly, on="user_id", how="left")
    temporal = temporal.merge(evaluate_official_rules(temporal), on="user_id", how="left")
    holdout_ids = set(split_frame[split_frame["shadow_split"] == "shadow_holdout"]["user_id"].tolist())
    frame = temporal[(temporal["status"].notna()) & (temporal["user_id"].isin(holdout_ids))].copy()
    if frame.empty:
        return {"cutoff_at": cutoff.isoformat(), "shadow_holdout_rows": 0}
    raw_probability = _predict_raw_probability(frame, bundle, model)
    calibrated_probability = calibrator.predict(raw_probability)
    metrics = _classification_metrics(frame["status"].astype(int).to_numpy(), calibrated_probability, threshold)
    return {
        "cutoff_at": cutoff.isoformat(),
        "shadow_holdout_rows": int(len(frame)),
        **metrics,
    }


def validate_official_model() -> dict[str, Any]:
    dataset = _load_dataset("full")
    bundle = load_selected_bundle(require_ready=False)
    split_frame = pd.read_parquet(bundle["split_path"])
    oof_predictions = pd.read_parquet(bundle["oof_predictions_path"])
    shadow_predictions = pd.read_parquet(bundle["shadow_predictions_path"])

    model, _ = _load_selected_model(bundle)
    shadow_dev = shadow_predictions[shadow_predictions["shadow_split"] == "shadow_dev"].copy()
    shadow_holdout = shadow_predictions[shadow_predictions["shadow_split"] == "shadow_holdout"].copy()

    calibrator_report, calibrator = choose_calibrator(
        shadow_dev["raw_probability"].to_numpy(),
        shadow_dev["status"].astype(int).to_numpy(),
    )
    shadow_dev["submission_probability"] = calibrator.predict(shadow_dev["raw_probability"].to_numpy())
    threshold_report = search_threshold(
        shadow_dev["status"].astype(int).to_numpy(),
        shadow_dev["submission_probability"].to_numpy(),
        shadow_dev["strong_group_id"].to_numpy(),
    )
    selected_threshold = float(threshold_report["selected_threshold"])

    shadow_holdout["submission_probability"] = calibrator.predict(shadow_holdout["raw_probability"].to_numpy())
    shadow_holdout_metrics = _classification_metrics(
        shadow_holdout["status"].astype(int).to_numpy(),
        shadow_holdout["submission_probability"].to_numpy(),
        selected_threshold,
    )

    oof_predictions["submission_probability"] = calibrator.predict(oof_predictions["raw_probability"].to_numpy())
    core_oof_metrics = {
        "raw_average_precision": float(average_precision_score(oof_predictions["status"].astype(int), oof_predictions["raw_probability"])),
        "calibrated_average_precision": float(average_precision_score(oof_predictions["status"].astype(int), oof_predictions["submission_probability"])),
        "fold_count": int(oof_predictions["core_fold"].nunique()),
    }

    bundle["calibrator"] = calibrator_report
    bundle["selected_threshold"] = selected_threshold
    save_selected_bundle(bundle)

    report = {
        "bundle_version": bundle["bundle_version"],
        "selected_model": bundle["selected_model"],
        "selected_threshold": selected_threshold,
        "core_oof_metrics": core_oof_metrics,
        "calibrator": calibrator_report,
        "threshold_report": threshold_report,
        "shadow_holdout_metrics": shadow_holdout_metrics,
        "temporal_stress_test": _temporal_stress_metrics(bundle, model, calibrator, selected_threshold, split_frame),
    }
    save_json(report, feature_report_path("official_validation_report.json"))
    return report


def main() -> None:
    print(validate_official_model())


if __name__ == "__main__":
    main()
