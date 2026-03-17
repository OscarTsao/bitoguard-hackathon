from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import load_settings
from hardware import describe_hardware, lightgbm_runtime_params
from models.common import encode_features, feature_columns, forward_date_splits, training_dataset


VALIDATION_SEEDS = (42, 52)
NEGATIVE_WEIGHTS = (1.0, 0.5, 0.25, 0.1, 0.05)


def _threshold_sweep(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    candidates = sorted({float(x) for x in np.quantile(probabilities, np.linspace(0.001, 0.999, 200))})
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        preds = (probabilities >= threshold).astype(int)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(score)
    return best_threshold, best_f1


def _topk_metrics(labels: np.ndarray, probabilities: np.ndarray, rate: float) -> dict[str, float]:
    k = max(1, int(round(len(labels) * rate)))
    topk_indices = np.argsort(-probabilities)[:k]
    preds = np.zeros_like(labels)
    preds[topk_indices] = 1
    return {
        "rate": float(rate),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


def _overlap_stats(train_users: set[Any], valid_users: set[Any], test_users: set[Any]) -> dict[str, int]:
    return {
        "train_users": len(train_users),
        "valid_users": len(valid_users),
        "test_users": len(test_users),
        "train_valid_overlap": len(train_users & valid_users),
        "train_test_overlap": len(train_users & test_users),
        "valid_test_overlap": len(valid_users & test_users),
    }


def _fit_lgbm(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    negative_weight: float = 1.0,
) -> dict[str, Any]:
    x_train, reference_columns = encode_features(train, features)
    x_valid, _ = encode_features(valid, features, reference_columns=reference_columns)
    x_test, _ = encode_features(test, features, reference_columns=reference_columns)

    y_train = train["hidden_suspicious_label"].astype(int).to_numpy()
    y_valid = valid["hidden_suspicious_label"].astype(int).to_numpy()
    y_test = test["hidden_suspicious_label"].astype(int).to_numpy()

    positive_count = max(1, int(y_train.sum()))
    negative_count = max(1, len(y_train) - positive_count)
    sample_weight = np.where(y_train == 1, negative_count / positive_count, negative_weight)

    model = LGBMClassifier(
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=-1,
        **lightgbm_runtime_params(),
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)

    valid_probabilities = model.predict_proba(x_valid)[:, 1]
    test_probabilities = model.predict_proba(x_test)[:, 1]
    selected_threshold, valid_best_f1 = _threshold_sweep(y_valid, valid_probabilities)
    threshold_preds = (test_probabilities >= selected_threshold).astype(int)
    oracle_threshold, oracle_f1 = _threshold_sweep(y_test, test_probabilities)

    return {
        "negative_weight": float(negative_weight),
        "rows": {"train": len(train), "valid": len(valid), "test": len(test)},
        "positives": {
            "train": int(y_train.sum()),
            "valid": int(y_valid.sum()),
            "test": int(y_test.sum()),
        },
        "valid_best_f1": float(valid_best_f1),
        "selected_threshold": float(selected_threshold),
        "test_average_precision": float(average_precision_score(y_test, test_probabilities)),
        "test_auc": float(roc_auc_score(y_test, test_probabilities)),
        "test_precision_at_selected_threshold": float(precision_score(y_test, threshold_preds, zero_division=0)),
        "test_recall_at_selected_threshold": float(recall_score(y_test, threshold_preds, zero_division=0)),
        "test_f1_at_selected_threshold": float(f1_score(y_test, threshold_preds, zero_division=0)),
        "test_oracle_threshold": float(oracle_threshold),
        "test_oracle_f1": float(oracle_f1),
        "topk": {
            "top_1pct": _topk_metrics(y_test, test_probabilities, 0.01),
            "top_5pct": _topk_metrics(y_test, test_probabilities, 0.05),
            "top_valid_rate": _topk_metrics(y_test, test_probabilities, float(y_valid.mean())),
        },
    }


def run_user_holdout_ablation() -> dict[str, Any]:
    print(f"[ablate_user_holdout] runtime: {describe_hardware()}")
    dataset = training_dataset().sort_values("snapshot_date").reset_index(drop=True)
    features = feature_columns(dataset)
    user_labels = dataset.groupby("user_id", as_index=False)["hidden_suspicious_label"].max()

    report: dict[str, Any] = {
        "dataset": {
            "rows": int(len(dataset)),
            "unique_users": int(dataset["user_id"].nunique()),
            "unique_dates": int(dataset["snapshot_date"].dt.date.nunique()),
            "positive_rows": int(dataset["hidden_suspicious_label"].sum()),
            "positive_users": int(user_labels["hidden_suspicious_label"].sum()),
        },
        "split_ablation": {},
        "user_holdout_negative_weight_sweep": [],
    }

    row_trainvalid, row_test = train_test_split(
        dataset,
        test_size=0.15,
        stratify=dataset["hidden_suspicious_label"],
        random_state=42,
    )
    row_train, row_valid = train_test_split(
        row_trainvalid,
        test_size=0.17647058823529413,
        stratify=row_trainvalid["hidden_suspicious_label"],
        random_state=42,
    )
    report["split_ablation"]["row_random"] = {
        "overlap": _overlap_stats(set(row_train["user_id"]), set(row_valid["user_id"]), set(row_test["user_id"])),
        "metrics": _fit_lgbm(row_train.reset_index(drop=True), row_valid.reset_index(drop=True), row_test.reset_index(drop=True), features),
    }

    splits = forward_date_splits(dataset["snapshot_date"])
    train_dates = set(splits["train"])
    valid_dates = set(splits["valid"])
    holdout_dates = set(splits["holdout"])
    time_train = dataset[dataset["snapshot_date"].dt.date.isin(train_dates)].reset_index(drop=True)
    time_valid = dataset[dataset["snapshot_date"].dt.date.isin(valid_dates)].reset_index(drop=True)
    time_test = dataset[dataset["snapshot_date"].dt.date.isin(holdout_dates)].reset_index(drop=True)
    report["split_ablation"]["forward_time"] = {
        "date_windows": {
            "train": [str(min(train_dates)), str(max(train_dates)), len(train_dates)],
            "valid": [str(min(valid_dates)), str(max(valid_dates)), len(valid_dates)],
            "holdout": [str(min(holdout_dates)), str(max(holdout_dates)), len(holdout_dates)],
        },
        "overlap": _overlap_stats(set(time_train["user_id"]), set(time_valid["user_id"]), set(time_test["user_id"])),
        "metrics": _fit_lgbm(time_train, time_valid, time_test, features),
    }

    for seed in VALIDATION_SEEDS:
        users_trainvalid, users_test = train_test_split(
            user_labels,
            test_size=0.15,
            stratify=user_labels["hidden_suspicious_label"],
            random_state=seed,
        )
        users_train, users_valid = train_test_split(
            users_trainvalid,
            test_size=0.17647058823529413,
            stratify=users_trainvalid["hidden_suspicious_label"],
            random_state=seed,
        )
        train_ids = set(users_train["user_id"])
        valid_ids = set(users_valid["user_id"])
        test_ids = set(users_test["user_id"])
        train = dataset[dataset["user_id"].isin(train_ids)].reset_index(drop=True)
        valid = dataset[dataset["user_id"].isin(valid_ids)].reset_index(drop=True)
        test = dataset[dataset["user_id"].isin(test_ids)].reset_index(drop=True)
        for negative_weight in NEGATIVE_WEIGHTS:
            report["user_holdout_negative_weight_sweep"].append({
                "seed": int(seed),
                "overlap": _overlap_stats(set(train["user_id"]), set(valid["user_id"]), set(test["user_id"])),
                **_fit_lgbm(train, valid, test, features, negative_weight=negative_weight),
            })

    sweep_frame = pd.DataFrame(report["user_holdout_negative_weight_sweep"])
    aggregate = (
        sweep_frame.groupby("negative_weight")
        .agg(
            mean_test_average_precision=("test_average_precision", "mean"),
            mean_test_auc=("test_auc", "mean"),
            mean_test_f1_at_selected_threshold=("test_f1_at_selected_threshold", "mean"),
            mean_test_oracle_f1=("test_oracle_f1", "mean"),
        )
        .reset_index()
        .sort_values("mean_test_average_precision", ascending=False)
    )
    report["user_holdout_negative_weight_summary"] = aggregate.to_dict(orient="records")

    settings = load_settings()
    report_path = settings.artifact_dir / "reports" / "user_holdout_ablation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


if __name__ == "__main__":
    result = run_user_holdout_ablation()
    print(json.dumps({
        "report_path": result["report_path"],
        "dataset": result["dataset"],
        "top_negative_weight_rows": result["user_holdout_negative_weight_summary"][:3],
    }, ensure_ascii=False, indent=2))
