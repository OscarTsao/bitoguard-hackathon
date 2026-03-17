from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import load_settings
from models.anomaly_common import anomaly_training_dataset, apply_anomaly_model, apply_legacy_anomaly_model, has_transform_metadata, load_user_cohort_frame
from models.common import encode_features, feature_columns, load_pickle, training_dataset
from models.rule_engine import evaluate_rules


PRIMARY_WITHOUT_IFOREST_WEIGHTS = {
    "rule_score": 0.35,
    "model_probability": 0.55,
    "graph_risk": 0.10,
}

PRIMARY_PLUS_IFOREST_CANDIDATE_WEIGHTS = {
    "rule_score": 0.35,
    "model_probability": 0.45,
    "anomaly_score": 0.10,
    "graph_risk": 0.10,
}

OPERATIONAL_THRESHOLDS = {
    "lgbm_only": 0.50,
    "iforest_only": 0.60,
    "primary_without_iforest": 0.60,
    "primary_plus_iforest_candidate": 0.60,
}

TOP_K_VALUES = [50, 100, 200, 500]


def _load_latest(prefix: str) -> tuple[object, dict[str, Any], Path]:
    settings = load_settings()
    model_files = sorted((settings.artifact_dir / "models").glob(f"{prefix}_*.pkl"))
    if not model_files:
        raise FileNotFoundError(f"No model found for prefix={prefix}")
    model_path = model_files[-1]
    meta = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    return load_pickle(model_path), meta, model_path


def _graph_risk_score(frame: pd.DataFrame) -> np.ndarray:
    raw = (
        frame["blacklist_1hop_count"] * 0.6
        + frame["blacklist_2hop_count"] * 0.4
        + frame["shared_device_count"] * 0.05
        + frame["shared_bank_count"] * 0.05
    )
    normalized = raw.clip(lower=0) / max(1.0, float(raw.max()))
    return normalized.to_numpy(dtype=float)


def _metric_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "fpr": float(fp / max(1, fp + tn)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "predicted_positive_rate": float(preds.mean()),
    }


def _best_threshold_summary(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    quantiles = np.linspace(0.0, 1.0, 201)
    thresholds = np.unique(np.quantile(scores, quantiles))
    best_summary: dict[str, float | int] | None = None
    best_rank: tuple[float, float, float] | None = None
    for threshold in thresholds:
        summary = _metric_at_threshold(labels, scores, float(threshold))
        rank = (
            float(summary["f1"]),
            float(summary["precision"]),
            float(summary["recall"]),
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_summary = summary
    assert best_summary is not None
    return best_summary


def _top_k_precision(labels: np.ndarray, scores: np.ndarray) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    ranking = np.argsort(-scores)
    for top_k in TOP_K_VALUES:
        if top_k > len(labels):
            continue
        top_labels = labels[ranking[:top_k]]
        rows.append(
            {
                "top_k": int(top_k),
                "positives": int(top_labels.sum()),
                "precision": float(top_labels.mean()),
            }
        )
    return rows


def _summarize_score(
    name: str,
    labels: np.ndarray,
    scores: np.ndarray,
    operational_threshold: float,
) -> dict[str, Any]:
    return {
        "name": name,
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "best_threshold": _best_threshold_summary(labels, scores),
        "operational_threshold": _metric_at_threshold(labels, scores, operational_threshold),
        "top_k_precision": _top_k_precision(labels, scores),
    }


def _recommendation(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidate_ap = float(results["primary_plus_iforest_candidate"]["average_precision"])
    baseline_ap = float(results["primary_without_iforest"]["average_precision"])
    iforest_ap = float(results["iforest_only"]["average_precision"])

    candidate_f1 = float(results["primary_plus_iforest_candidate"]["operational_threshold"]["f1"])
    baseline_f1 = float(results["primary_without_iforest"]["operational_threshold"]["f1"])

    ap_delta = candidate_ap - baseline_ap
    f1_delta = candidate_f1 - baseline_f1
    gate_passed = ap_delta >= 0.002 and f1_delta >= -0.005

    if gate_passed:
        verdict = "eligible_for_primary_blend"
        reason = "candidate_blend_meets_ap_and_f1_gate"
    elif iforest_ap < 0.05 and ap_delta <= 0.0:
        verdict = "sidecar_only"
        reason = "iforest_signal_is_weak_and_candidate_blend_does_not_improve"
    else:
        verdict = "sidecar_only"
        reason = "candidate_blend_does_not_clear_gate"

    return {
        "verdict": verdict,
        "reason": reason,
        "gate_passed": gate_passed,
        "delta_average_precision_vs_primary_without_iforest": ap_delta,
        "delta_operational_f1_vs_primary_without_iforest": f1_delta,
    }


def run_iforest_ablation() -> dict[str, Any]:
    standard_dataset = training_dataset().sort_values("snapshot_date").reset_index(drop=True)
    unique_dates = sorted(standard_dataset["snapshot_date"].dt.date.unique())
    holdout_dates = set(unique_dates[25:])
    holdout = standard_dataset[standard_dataset["snapshot_date"].dt.date.isin(holdout_dates)].copy().reset_index(drop=True)
    if holdout.empty:
        raise ValueError("Holdout frame is empty; cannot run iforest ablation.")

    lgbm_model, lgbm_meta, lgbm_path = _load_latest("lgbm")
    iforest_model, iforest_meta, iforest_path = _load_latest("iforest")

    features = feature_columns(holdout)
    x_lgbm, _ = encode_features(holdout, features, reference_columns=lgbm_meta["encoded_columns"])

    labels = holdout["hidden_suspicious_label"].astype(int).to_numpy()
    model_probability = lgbm_model.predict_proba(x_lgbm)[:, 1]
    if has_transform_metadata(iforest_meta):
        anomaly_dataset = anomaly_training_dataset().sort_values("snapshot_date").reset_index(drop=True)
        anomaly_holdout = anomaly_dataset[anomaly_dataset["snapshot_date"].dt.date.isin(holdout_dates)].copy().reset_index(drop=True)
        user_cohorts = load_user_cohort_frame()
        _, anomaly_score = apply_anomaly_model(iforest_model, anomaly_holdout, user_cohorts, iforest_meta)
        anomaly_frame = anomaly_holdout[["user_id", "snapshot_date"]].copy()
        anomaly_frame["iforest_only"] = anomaly_score
        score_frame = holdout[["user_id", "snapshot_date", "hidden_suspicious_label"]].copy().merge(
            anomaly_frame,
            on=["user_id", "snapshot_date"],
            how="left",
        )
        anomaly_score = score_frame["iforest_only"].fillna(0.0).to_numpy()
        legacy_mode = False
    else:
        _, anomaly_score = apply_legacy_anomaly_model(iforest_model, holdout, iforest_meta)
        score_frame = holdout[["user_id", "snapshot_date", "hidden_suspicious_label"]].copy()
        legacy_mode = True
    graph_risk = _graph_risk_score(holdout)
    rule_frame = evaluate_rules(holdout)
    rule_score = pd.to_numeric(rule_frame["rule_score"], errors="coerce").fillna(0.0).to_numpy()

    score_frame["lgbm_only"] = model_probability
    score_frame["iforest_only"] = anomaly_score
    score_frame["rule_score"] = rule_score
    score_frame["graph_risk"] = graph_risk
    score_frame["primary_without_iforest"] = (
        PRIMARY_WITHOUT_IFOREST_WEIGHTS["rule_score"] * rule_score
        + PRIMARY_WITHOUT_IFOREST_WEIGHTS["model_probability"] * model_probability
        + PRIMARY_WITHOUT_IFOREST_WEIGHTS["graph_risk"] * graph_risk
    )
    score_frame["primary_plus_iforest_candidate"] = (
        PRIMARY_PLUS_IFOREST_CANDIDATE_WEIGHTS["rule_score"] * rule_score
        + PRIMARY_PLUS_IFOREST_CANDIDATE_WEIGHTS["model_probability"] * model_probability
        + PRIMARY_PLUS_IFOREST_CANDIDATE_WEIGHTS["anomaly_score"] * anomaly_score
        + PRIMARY_PLUS_IFOREST_CANDIDATE_WEIGHTS["graph_risk"] * graph_risk
    )

    results = {
        name: _summarize_score(name, labels, score_frame[name].to_numpy(dtype=float), OPERATIONAL_THRESHOLDS[name])
        for name in OPERATIONAL_THRESHOLDS
    }

    positive_mask = labels == 1
    negative_mask = labels == 0
    report = {
        "artifacts": {
            "lgbm_model_path": str(lgbm_path),
            "iforest_model_path": str(iforest_path),
        },
        "data_summary": {
            "rows": int(len(holdout)),
            "positive_rows": int(labels.sum()),
            "negative_rows": int((labels == 0).sum()),
            "holdout_dates": [str(item) for item in sorted(holdout_dates)],
        },
        "weights": {
            "primary_without_iforest": PRIMARY_WITHOUT_IFOREST_WEIGHTS,
            "primary_plus_iforest_candidate": PRIMARY_PLUS_IFOREST_CANDIDATE_WEIGHTS,
        },
        "component_signal_summary": {
            "iforest_positive_mean": float(anomaly_score[positive_mask].mean()),
            "iforest_negative_mean": float(anomaly_score[negative_mask].mean()),
            "lgbm_positive_mean": float(model_probability[positive_mask].mean()),
            "lgbm_negative_mean": float(model_probability[negative_mask].mean()),
            "rule_positive_mean": float(rule_score[positive_mask].mean()),
            "rule_negative_mean": float(rule_score[negative_mask].mean()),
            "graph_positive_mean": float(graph_risk[positive_mask].mean()),
            "graph_negative_mean": float(graph_risk[negative_mask].mean()),
        },
        "legacy_mode": legacy_mode,
        "results": results,
        "recommendation": _recommendation(results),
    }

    settings = load_settings()
    report_dir = settings.artifact_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "iforest_ablation_report.json"
    score_path = report_dir / "iforest_ablation_scores.parquet"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    score_frame.to_parquet(score_path, index=False)
    report["report_path"] = str(report_path)
    report["score_path"] = str(score_path)
    return report


def main() -> None:
    print(json.dumps(run_iforest_ablation(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
