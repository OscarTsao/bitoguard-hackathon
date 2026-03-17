from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score

from transductive_v1.calibration import select_best_calibration
from transductive_v1.common import bundle_path, feature_path, report_path, save_json
from transductive_v1.dataset import build_user_universe
from transductive_v1.graph_store import build_graph_store
from transductive_v1.label_free_features import build_label_free_user_features
from transductive_v1.primary_validation import build_primary_split
from transductive_v1.secondary_validation import build_secondary_group_split, iter_secondary_folds
from transductive_v1.train import _labeled_frame, run_primary_base_oof
from transductive_v1.stacking import build_stacker_oof
from transductive_v1.decision_rule import apply_rule


def _classification_metrics(labels: np.ndarray, probabilities: np.ndarray, preds: np.ndarray) -> dict[str, Any]:
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


def _metric_delta(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, float]:
    keys = ["precision", "recall", "f1", "fpr", "average_precision", "brier_score"]
    return {f"{key}_delta": float(primary[key] - secondary[key]) for key in keys}


def validate_transductive_v1(cutoff_tag: str = "full") -> dict[str, Any]:
    bundle = json.loads(bundle_path().read_text(encoding="utf-8"))
    primary_oof = pd.read_parquet(feature_path("primary_stack_oof", cutoff_tag))
    labels = primary_oof["status"].astype(int).to_numpy()
    calibration_report, calibrator, calibrated = select_best_calibration(primary_oof["stacker_raw_probability"].to_numpy(), labels)
    primary_oof["submission_probability"] = calibrated
    primary_preds = apply_rule(primary_oof["submission_probability"].to_numpy(), calibration_report["selected_rule"])
    primary_metrics = _classification_metrics(labels, primary_oof["submission_probability"].to_numpy(), primary_preds)

    label_free_frame = build_label_free_user_features(cutoff_tag=cutoff_tag, write_outputs=True)
    graph_store = build_graph_store(label_free_frame["user_id"].astype(int).tolist(), cutoff_tag=cutoff_tag, write_outputs=True)
    secondary_split = build_secondary_group_split(_labeled_frame(label_free_frame), graph_store, cutoff_tag=cutoff_tag, write_outputs=True)
    secondary_base_oof, _, _, _ = run_primary_base_oof(
        label_free_frame,
        graph_store,
        secondary_split.rename(columns={"secondary_fold": "primary_fold"})[["user_id", "status", "primary_fold"]],
        cutoff_tag=cutoff_tag,
        split_artifact_name="secondary_group_split",
        fold_column="primary_fold",
    )
    secondary_stack_oof, _, _ = build_stacker_oof(secondary_base_oof.rename(columns={"primary_fold": "secondary_fold"}), fold_column="secondary_fold")
    secondary_stack_oof["submission_probability"] = calibrator.predict(secondary_stack_oof["stacker_raw_probability"].to_numpy())
    secondary_preds = apply_rule(secondary_stack_oof["submission_probability"].to_numpy(), calibration_report["selected_rule"])
    secondary_metrics = _classification_metrics(
        secondary_stack_oof["status"].astype(int).to_numpy(),
        secondary_stack_oof["submission_probability"].to_numpy(),
        secondary_preds,
    )
    secondary_stack_oof.to_parquet(feature_path("secondary_stack_oof", cutoff_tag), index=False)

    bundle["calibrator"] = calibration_report
    bundle["decision_rule"] = calibration_report["selected_rule"]
    if calibration_report["selected_rule"]["rule_type"] == "threshold":
        bundle["selected_threshold"] = calibration_report["selected_rule"]["threshold"]
    else:
        bundle["selected_threshold"] = None
    bundle["secondary_stress_summary"] = secondary_metrics
    save_json(bundle, bundle_path())

    report = {
        "bundle_version": bundle["bundle_version"],
        "primary_validation_protocol": bundle["primary_validation_protocol"],
        "calibrator": calibration_report,
        "primary_transductive_oof_metrics": primary_metrics,
        "secondary_group_stress_metrics": secondary_metrics,
        "primary_vs_secondary_delta": _metric_delta(primary_metrics, secondary_metrics),
    }
    save_json(report, report_path("validation_report.json"))
    return report
