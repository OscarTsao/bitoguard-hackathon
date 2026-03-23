from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score

from official.bundle import load_selected_bundle, save_selected_bundle
from official.common import feature_output_path, feature_report_path, save_json
from official.graph_dataset import build_transductive_graph
from official.stacking import STACKER_FEATURE_COLUMNS, build_stacker_oof, choose_best_calibration_and_threshold
from official.train import PRIMARY_GRAPH_MAX_EPOCHS, _load_dataset, _label_free_feature_columns, run_transductive_oof_pipeline
from official.transductive_features import build_transductive_feature_frame
from official.transductive_validation import build_secondary_strict_splits


def _expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) using equal-width bins.

    ECE = Σ_b (|B_b| / N) × |acc(B_b) − conf(B_b)|

    where B_b is the set of predictions in bin b, acc(B_b) is the fraction of
    actual positives in that bin, and conf(B_b) is the mean predicted probability.
    A well-calibrated model has ECE ≈ 0.

    Args:
        labels: Binary ground truth (0/1).
        probabilities: Predicted probabilities in [0, 1].
        n_bins: Number of equal-width bins. Default 10.

    Returns:
        ECE value in [0, 1].
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probabilities >= lo) & (probabilities < hi)
        if not mask.any():
            continue
        bin_conf = float(probabilities[mask].mean())
        bin_acc = float(labels[mask].mean())
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


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
        "ece": float(_expected_calibration_error(labels, probabilities)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _metric_delta(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, float]:
    keys = ["precision", "recall", "f1", "fpr", "average_precision", "brier_score"]
    return {f"{key}_delta": float(primary[key] - secondary[key]) for key in keys}


def validate_official_model() -> dict[str, Any]:
    dataset = _load_dataset("full")
    bundle = load_selected_bundle(require_ready=False)
    primary_oof = pd.read_parquet(bundle["oof_predictions_path"])

    calibration_report, calibrator, primary_calibrated = choose_best_calibration_and_threshold(
        primary_oof["stacker_raw_probability"].to_numpy(),
        primary_oof["status"].astype(int).to_numpy(),
        primary_oof["primary_fold"].to_numpy(),
        use_pu_adjustment=False,
    )
    selected_threshold = float(calibration_report["selected_threshold"])
    primary_oof["submission_probability"] = primary_calibrated
    primary_metrics = _classification_metrics(
        primary_oof["status"].astype(int).to_numpy(),
        primary_oof["submission_probability"].to_numpy(),
        selected_threshold,
    )

    # Bug fix: don't load stale HPO params - use same defaults as primary
    catboost_params = None

    # Release GPU memory from primary OOF training before secondary OOF to
    # prevent CatBoost CUDA OOM (GPU is saturated after primary + final training).
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    import gc
    gc.collect()

    # v34: Allow secondary OOF to use GPU — cuda.empty_cache() + gc.collect()
    # above releases primary training memory. GPU reduces secondary OOF from 45+ min to ~5 min.
    # secondary uses same defaults as primary (bug fix)

    secondary_split = build_secondary_strict_splits(dataset, cutoff_tag="full", write_outputs=True)
    graph = build_transductive_graph(dataset)
    base_a_feature_columns = _label_free_feature_columns(dataset)
    base_b_feature_columns = bundle["feature_columns_base_b"]
    secondary_oof, _ = run_transductive_oof_pipeline(
        dataset,
        graph,
        secondary_split[["user_id", "secondary_fold"]].copy(),
        fold_column="secondary_fold",
        base_a_feature_columns=base_a_feature_columns,
        base_b_feature_columns=base_b_feature_columns,
        graph_max_epochs=PRIMARY_GRAPH_MAX_EPOCHS,
    )
    secondary_oof, _ = build_stacker_oof(
        secondary_oof,
        secondary_split[["user_id", "secondary_fold"]].copy(),
        fold_column="secondary_fold",
        use_blend=True,
    )
    secondary_oof["submission_probability"] = calibrator.predict(secondary_oof["stacker_raw_probability"].to_numpy())
    secondary_metrics = _classification_metrics(
        secondary_oof["status"].astype(int).to_numpy(),
        secondary_oof["submission_probability"].to_numpy(),
        selected_threshold,
    )
    secondary_oof_path = feature_output_path("official_secondary_oof_predictions", "full")
    secondary_oof.to_parquet(secondary_oof_path, index=False)

    bundle["calibrator"] = calibration_report
    bundle["selected_threshold"] = selected_threshold
    bundle["calibration_selection_basis"] = calibration_report["selection_basis"]
    bundle["secondary_stress_summary"] = {
        "secondary_oof_predictions_path": str(secondary_oof_path),
        "secondary_group_stress_metrics": secondary_metrics,
    }
    save_selected_bundle(bundle)

    report = {
        "bundle_version": bundle["bundle_version"],
        "selected_model": bundle["selected_model"],
        "selected_threshold": selected_threshold,
        "primary_validation_protocol": bundle["primary_validation_protocol"],
        "calibrator": calibration_report,
        "primary_transductive_oof_metrics": primary_metrics,
        "secondary_group_stress_metrics": secondary_metrics,
        "primary_vs_secondary_delta": _metric_delta(primary_metrics, secondary_metrics),
    }
    report["grouped_oof_metrics"] = report["primary_transductive_oof_metrics"]
    # Backward-compatible aliases for launch scripts that use short key names.
    report["primary_validation"] = primary_metrics
    report["secondary_stress"] = secondary_metrics
    save_json(report, feature_report_path("official_validation_report.json"))
    return report


def main() -> None:
    print(validate_official_model())


if __name__ == "__main__":
    main()
