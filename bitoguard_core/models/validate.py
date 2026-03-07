from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, precision_recall_curve, recall_score

from config import load_settings
from db.store import DuckDBStore, make_id, utc_now
from models.common import encode_features, feature_columns, load_pickle, training_dataset


def _load_latest(prefix: str) -> tuple[object, dict]:
    settings = load_settings()
    model_files = sorted((settings.artifact_dir / "models").glob(f"{prefix}_*.pkl"))
    model_path = model_files[-1]
    meta = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    return load_pickle(model_path), meta


def validate_model() -> dict:
    settings = load_settings()
    store = DuckDBStore(settings.db_path)
    dataset = training_dataset().sort_values("snapshot_date").reset_index(drop=True)
    unique_dates = sorted(dataset["snapshot_date"].dt.date.unique())
    holdout_dates = set(unique_dates[25:])
    holdout = dataset[dataset["snapshot_date"].dt.date.isin(holdout_dates)].copy().reset_index(drop=True)

    model, meta = _load_latest("lgbm")
    feature_cols = feature_columns(holdout)
    encoded, _ = encode_features(holdout, feature_cols, reference_columns=meta["encoded_columns"])
    y_true = holdout["hidden_suspicious_label"].astype(int)
    probabilities = model.predict_proba(encoded)[:, 1]
    preds = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    fpr = fp / max(1, fp + tn)
    pr_precision, pr_recall, thresholds = precision_recall_curve(y_true, probabilities)

    threshold_report = []
    for threshold in [round(x, 2) for x in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]]:
        threshold_preds = (probabilities >= threshold).astype(int)
        threshold_report.append({
            "threshold": threshold,
            "precision": precision_score(y_true, threshold_preds, zero_division=0),
            "recall": recall_score(y_true, threshold_preds, zero_division=0),
            "f1": f1_score(y_true, threshold_preds, zero_division=0),
        })

    scenario_breakdown = []
    scenario_series = holdout["scenario_types"].fillna("").replace("", "clean")
    for scenario_name, frame in holdout.groupby(scenario_series):
        if frame.empty:
            continue
        idx = frame.index.to_list()
        scenario_preds = (probabilities[idx] >= 0.5).astype(int)
        scenario_breakdown.append({
            "scenario": scenario_name,
            "count": int(len(frame)),
            "precision": precision_score(frame["hidden_suspicious_label"], scenario_preds, zero_division=0),
            "recall": recall_score(frame["hidden_suspicious_label"], scenario_preds, zero_division=0),
        })

    report = {
        "model_version": meta["model_version"],
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "pr_curve": {
            "precision": pr_precision.tolist(),
            "recall": pr_recall.tolist(),
            "thresholds": thresholds.tolist(),
        },
        "threshold_sensitivity": threshold_report,
        "scenario_breakdown": scenario_breakdown,
    }
    report_path = settings.artifact_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    store.execute(
        """
        INSERT INTO ops.validation_reports (report_id, created_at, model_version, report_path, metrics_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (make_id("report"), utc_now(), meta["model_version"], str(report_path), json.dumps(report)),
    )
    return report


if __name__ == "__main__":
    print(validate_model())
