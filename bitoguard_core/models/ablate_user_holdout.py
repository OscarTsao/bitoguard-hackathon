from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import load_settings
from hardware import (
    catboost_runtime_params,
    describe_hardware,
    hardware_profile,
    lightgbm_runtime_params,
    sklearn_n_jobs,
    xgboost_runtime_params,
)
from models.common import encode_features, feature_columns, forward_date_splits, training_dataset


VALIDATION_SEEDS = (42, 52)
NEGATIVE_WEIGHTS = (1.0, 0.5, 0.25, 0.1, 0.05)
MODEL_FAMILIES = ("lgbm", "xgboost", "catboost", "extratrees")
BOOSTING_FAMILIES = ("lgbm", "xgboost", "catboost")
QUEUE_RATE_CANDIDATES = (0.005, 0.01, 0.02, 0.05, 0.1)


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


def _topk_metrics(labels: np.ndarray, probabilities: np.ndarray, rate: float) -> dict[str, float | int]:
    normalized_rate = float(min(max(rate, 0.0), 1.0))
    preds = np.zeros_like(labels)
    if normalized_rate > 0.0:
        k = max(1, int(round(len(labels) * normalized_rate)))
        topk_indices = np.argsort(-probabilities)[:k]
        preds[topk_indices] = 1
    return {
        "rate": normalized_rate,
        "alerts": int(preds.sum()),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


def _safe_auc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, probabilities))


def _rank_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return pd.Series(probabilities).rank(method="average", pct=True).to_numpy(dtype=float)


def _blend_probabilities(
    probability_map: dict[str, dict[str, np.ndarray]],
    family_names: tuple[str, ...],
    strategy: str,
) -> tuple[np.ndarray, np.ndarray]:
    if strategy == "mean":
        valid_arrays = [probability_map[family]["valid"] for family in family_names]
        test_arrays = [probability_map[family]["test"] for family in family_names]
    elif strategy == "rank_mean":
        valid_arrays = [_rank_probabilities(probability_map[family]["valid"]) for family in family_names]
        test_arrays = [_rank_probabilities(probability_map[family]["test"]) for family in family_names]
    else:  # pragma: no cover - protected by internal call sites
        raise ValueError(f"Unsupported blend strategy: {strategy}")
    return np.mean(valid_arrays, axis=0), np.mean(test_arrays, axis=0)


def _overlap_stats(train_users: set[Any], valid_users: set[Any], test_users: set[Any]) -> dict[str, int]:
    return {
        "train_users": len(train_users),
        "valid_users": len(valid_users),
        "test_users": len(test_users),
        "train_valid_overlap": len(train_users & valid_users),
        "train_test_overlap": len(train_users & test_users),
        "valid_test_overlap": len(valid_users & test_users),
    }


def _prepare_split(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> dict[str, Any]:
    x_train, reference_columns = encode_features(train, features)
    x_valid, _ = encode_features(valid, features, reference_columns=reference_columns)
    x_test, _ = encode_features(test, features, reference_columns=reference_columns)

    return {
        "x_train": x_train.astype(np.float32),
        "x_valid": x_valid.astype(np.float32),
        "x_test": x_test.astype(np.float32),
        "y_train": train["hidden_suspicious_label"].astype(int).to_numpy(),
        "y_valid": valid["hidden_suspicious_label"].astype(int).to_numpy(),
        "y_test": test["hidden_suspicious_label"].astype(int).to_numpy(),
        "rows": {"train": len(train), "valid": len(valid), "test": len(test)},
        "positives": {
            "train": int(train["hidden_suspicious_label"].sum()),
            "valid": int(valid["hidden_suspicious_label"].sum()),
            "test": int(test["hidden_suspicious_label"].sum()),
        },
    }


def _sample_weight(y_train: np.ndarray, negative_weight: float) -> np.ndarray:
    positive_count = max(1, int(y_train.sum()))
    negative_count = max(1, len(y_train) - positive_count)
    return np.where(y_train == 1, negative_count / positive_count, negative_weight).astype(np.float32)


def _queue_rate_candidates(labels: np.ndarray) -> list[float]:
    prevalence = float(labels.mean()) if len(labels) else 0.0
    candidates = {
        *QUEUE_RATE_CANDIDATES,
        prevalence,
        prevalence / 2.0 if prevalence > 0 else 0.0,
        prevalence * 2.0 if prevalence > 0 else 0.0,
    }
    resolved = sorted({float(rate) for rate in candidates if 0.0 < rate < 0.25})
    return resolved or [0.01]


def _select_best_queue_rate(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    rows = [_topk_metrics(labels, probabilities, rate) for rate in _queue_rate_candidates(labels)]
    selected = max(rows, key=lambda row: (row["f1"], row["precision"], -row["rate"]))
    return {
        "selected_rate": float(selected["rate"]),
        "selected_metrics": selected,
        "candidate_metrics": rows,
    }


def _fit_model_probabilities(
    model_family: str,
    split: dict[str, Any],
    negative_weight: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    x_train = split["x_train"]
    x_valid = split["x_valid"]
    x_test = split["x_test"]
    y_train = split["y_train"]
    y_valid = split["y_valid"]
    weights = _sample_weight(y_train, negative_weight)

    start = time.perf_counter()
    if model_family == "lgbm":
        runtime_params = lightgbm_runtime_params()
        if runtime_params.get("device_type") == "gpu":
            # The LightGBM GPU path on this host is not stable on the honest
            # user-held-out benchmark. Keep GPU for XGBoost/CatBoost, but force
            # CPU here so the comparative sweep finishes reliably.
            runtime_params = {"n_jobs": hardware_profile().cpu_threads, "device_type": "cpu"}
        model_kwargs = dict(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbosity=-1,
            **runtime_params,
        )
        model = LGBMClassifier(**model_kwargs)
        try:
            model.fit(x_train, y_train, sample_weight=weights, eval_set=[(x_valid, y_valid)], eval_metric="binary_logloss")
        except Exception:
            if runtime_params.get("device_type") != "gpu":
                raise
            model = LGBMClassifier(
                **{
                    **model_kwargs,
                    "device_type": "cpu",
                    "n_jobs": hardware_profile().cpu_threads,
                }
            )
            model.fit(x_train, y_train, sample_weight=weights, eval_set=[(x_valid, y_valid)], eval_metric="binary_logloss")
    elif model_family == "xgboost":
        runtime_params = xgboost_runtime_params()
        model = XGBClassifier(
            n_estimators=350,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            objective="binary:logistic",
            eval_metric="logloss",
            **runtime_params,
        )
        try:
            model.fit(x_train, y_train, sample_weight=weights, eval_set=[(x_valid, y_valid)], verbose=False)
        except Exception:
            if runtime_params.get("device") != "cuda":
                raise
            model = XGBClassifier(
                n_estimators=350,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=42,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=hardware_profile().cpu_threads,
                tree_method="hist",
                device="cpu",
            )
            model.fit(x_train, y_train, sample_weight=weights, eval_set=[(x_valid, y_valid)], verbose=False)
    elif model_family == "catboost":
        runtime_params = catboost_runtime_params()
        model = CatBoostClassifier(
            iterations=350,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            **runtime_params,
        )
        try:
            model.fit(x_train, y_train, sample_weight=weights, eval_set=(x_valid, y_valid), use_best_model=False)
        except Exception:
            if runtime_params.get("task_type") != "GPU":
                raise
            model = CatBoostClassifier(
                iterations=350,
                learning_rate=0.05,
                depth=6,
                loss_function="Logloss",
                eval_metric="Logloss",
                random_seed=42,
                verbose=False,
                allow_writing_files=False,
                task_type="CPU",
                thread_count=hardware_profile().cpu_threads,
            )
            model.fit(x_train, y_train, sample_weight=weights, eval_set=(x_valid, y_valid), use_best_model=False)
    elif model_family == "extratrees":
        model = ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=sklearn_n_jobs(),
        )
        model.fit(x_train, y_train, sample_weight=weights)
    else:  # pragma: no cover - protected by MODEL_FAMILIES
        raise ValueError(f"Unsupported model family: {model_family}")

    fit_seconds = time.perf_counter() - start
    valid_probabilities = model.predict_proba(x_valid)[:, 1].astype(float)
    test_probabilities = model.predict_proba(x_test)[:, 1].astype(float)
    return valid_probabilities, test_probabilities, float(fit_seconds)


def _evaluate_probabilities(
    model_family: str,
    split: dict[str, Any],
    negative_weight: float,
    valid_probabilities: np.ndarray,
    test_probabilities: np.ndarray,
    fit_seconds: float,
) -> dict[str, Any]:
    y_valid = split["y_valid"]
    y_test = split["y_test"]
    selected_threshold, valid_best_f1 = _threshold_sweep(y_valid, valid_probabilities)
    threshold_preds = (test_probabilities >= selected_threshold).astype(int)
    oracle_threshold, oracle_f1 = _threshold_sweep(y_test, test_probabilities)
    valid_selected_alert_rate = float((valid_probabilities >= selected_threshold).mean())
    queue_policy = _select_best_queue_rate(y_valid, valid_probabilities)
    queue_metrics = _topk_metrics(y_test, test_probabilities, queue_policy["selected_rate"])

    return {
        "model_family": model_family,
        "negative_weight": float(negative_weight),
        "rows": split["rows"],
        "positives": split["positives"],
        "fit_seconds": float(fit_seconds),
        "valid_best_f1": float(valid_best_f1),
        "selected_threshold": float(selected_threshold),
        "valid_selected_alert_rate": valid_selected_alert_rate,
        "test_average_precision": float(average_precision_score(y_test, test_probabilities)),
        "test_auc": _safe_auc(y_test, test_probabilities),
        "test_precision_at_selected_threshold": float(precision_score(y_test, threshold_preds, zero_division=0)),
        "test_recall_at_selected_threshold": float(recall_score(y_test, threshold_preds, zero_division=0)),
        "test_f1_at_selected_threshold": float(f1_score(y_test, threshold_preds, zero_division=0)),
        "test_oracle_threshold": float(oracle_threshold),
        "test_oracle_f1": float(oracle_f1),
        "queue_policy": {
            "selected_rate": float(queue_policy["selected_rate"]),
            "valid_selected_f1": float(queue_policy["selected_metrics"]["f1"]),
            "test_precision": float(queue_metrics["precision"]),
            "test_recall": float(queue_metrics["recall"]),
            "test_f1": float(queue_metrics["f1"]),
            "candidate_rates": queue_policy["candidate_metrics"],
        },
        "topk": {
            "top_1pct": _topk_metrics(y_test, test_probabilities, 0.01),
            "top_5pct": _topk_metrics(y_test, test_probabilities, 0.05),
            "top_valid_positive_rate": _topk_metrics(y_test, test_probabilities, float(y_valid.mean())),
            "top_valid_alert_rate": _topk_metrics(y_test, test_probabilities, valid_selected_alert_rate),
        },
    }


def _run_model_family(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    model_family: str,
    negative_weight: float = 1.0,
    prepared_split: dict[str, Any] | None = None,
) -> dict[str, Any]:
    split = prepared_split or _prepare_split(train, valid, test, features)
    valid_probabilities, test_probabilities, fit_seconds = _fit_model_probabilities(
        model_family=model_family,
        split=split,
        negative_weight=negative_weight,
    )
    return _evaluate_probabilities(
        model_family=model_family,
        split=split,
        negative_weight=negative_weight,
        valid_probabilities=valid_probabilities,
        test_probabilities=test_probabilities,
        fit_seconds=fit_seconds,
    )


def _summarize_sweep(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []
    sweep_frame = pd.json_normalize(records, sep=".")
    aggregate = (
        sweep_frame.groupby(["model_family", "negative_weight"])
        .agg(
            mean_fit_seconds=("fit_seconds", "mean"),
            mean_test_average_precision=("test_average_precision", "mean"),
            mean_test_auc=("test_auc", "mean"),
            mean_test_f1_at_selected_threshold=("test_f1_at_selected_threshold", "mean"),
            mean_test_queue_f1=("queue_policy.test_f1", "mean"),
            mean_test_oracle_f1=("test_oracle_f1", "mean"),
            mean_top_1pct_f1=("topk.top_1pct.f1", "mean"),
            mean_top_5pct_f1=("topk.top_5pct.f1", "mean"),
            mean_top_valid_positive_rate_f1=("topk.top_valid_positive_rate.f1", "mean"),
            mean_top_valid_alert_rate_f1=("topk.top_valid_alert_rate.f1", "mean"),
            mean_queue_selected_rate=("queue_policy.selected_rate", "mean"),
        )
        .reset_index()
        .sort_values(
            ["mean_test_average_precision", "mean_test_queue_f1", "mean_top_valid_alert_rate_f1", "mean_test_f1_at_selected_threshold"],
            ascending=False,
        )
    )
    return aggregate.to_dict(orient="records")


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
            "feature_count": int(len(features)),
        },
        "model_families": list(MODEL_FAMILIES),
        "negative_weights": list(NEGATIVE_WEIGHTS),
        "split_ablation": {},
        "user_holdout_model_sweep": [],
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
        "metrics": _run_model_family(
            row_train.reset_index(drop=True),
            row_valid.reset_index(drop=True),
            row_test.reset_index(drop=True),
            features,
            model_family="lgbm",
        ),
    }
    print("[ablate_user_holdout] finished split ablation: row_random")

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
        "metrics": _run_model_family(time_train, time_valid, time_test, features, model_family="lgbm"),
    }
    print("[ablate_user_holdout] finished split ablation: forward_time")

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
        overlap = _overlap_stats(set(train["user_id"]), set(valid["user_id"]), set(test["user_id"]))
        prepared_split = _prepare_split(train, valid, test, features)

        for negative_weight in NEGATIVE_WEIGHTS:
            print(f"[ablate_user_holdout] seed={seed} negative_weight={negative_weight}")
            probability_map: dict[str, dict[str, np.ndarray]] = {}
            for model_family in MODEL_FAMILIES:
                print(f"[ablate_user_holdout] fitting {model_family}", flush=True)
                valid_probabilities, test_probabilities, fit_seconds = _fit_model_probabilities(
                    model_family=model_family,
                    split=prepared_split,
                    negative_weight=negative_weight,
                )
                probability_map[model_family] = {
                    "valid": valid_probabilities,
                    "test": test_probabilities,
                }
                report["user_holdout_model_sweep"].append({
                    "seed": int(seed),
                    "overlap": overlap,
                    **_evaluate_probabilities(
                        model_family=model_family,
                        split=prepared_split,
                        negative_weight=negative_weight,
                        valid_probabilities=valid_probabilities,
                        test_probabilities=test_probabilities,
                        fit_seconds=fit_seconds,
                    ),
                })

            blend_valid, blend_test = _blend_probabilities(probability_map, BOOSTING_FAMILIES, strategy="mean")
            report["user_holdout_model_sweep"].append({
                "seed": int(seed),
                "overlap": overlap,
                **_evaluate_probabilities(
                    model_family="blend_mean_boosting",
                    split=prepared_split,
                    negative_weight=negative_weight,
                    valid_probabilities=blend_valid,
                    test_probabilities=blend_test,
                    fit_seconds=0.0,
                ),
            })

            rank_blend_valid, rank_blend_test = _blend_probabilities(probability_map, BOOSTING_FAMILIES, strategy="rank_mean")
            report["user_holdout_model_sweep"].append({
                "seed": int(seed),
                "overlap": overlap,
                **_evaluate_probabilities(
                    model_family="blend_rank_boosting",
                    split=prepared_split,
                    negative_weight=negative_weight,
                    valid_probabilities=rank_blend_valid,
                    test_probabilities=rank_blend_test,
                    fit_seconds=0.0,
                ),
            })

    report["user_holdout_model_summary"] = _summarize_sweep(report["user_holdout_model_sweep"])
    report["user_holdout_negative_weight_summary"] = [
        row for row in report["user_holdout_model_summary"] if row["model_family"] == "lgbm"
    ]
    report["best_configuration"] = report["user_holdout_model_summary"][0] if report["user_holdout_model_summary"] else None
    if report["user_holdout_model_summary"]:
        summary_frame = pd.DataFrame(report["user_holdout_model_summary"])
        report["best_configurations_by_metric"] = {
            "average_precision": summary_frame.sort_values(
                ["mean_test_average_precision", "mean_test_queue_f1", "mean_top_1pct_f1"],
                ascending=False,
            ).head(5).to_dict(orient="records"),
            "queue_f1": summary_frame.sort_values(
                ["mean_test_queue_f1", "mean_test_average_precision", "mean_top_valid_alert_rate_f1"],
                ascending=False,
            ).head(5).to_dict(orient="records"),
            "selected_threshold_f1": summary_frame.sort_values(
                ["mean_test_f1_at_selected_threshold", "mean_test_average_precision"],
                ascending=False,
            ).head(5).to_dict(orient="records"),
            "top_1pct_f1": summary_frame.sort_values(
                ["mean_top_1pct_f1", "mean_test_average_precision"],
                ascending=False,
            ).head(5).to_dict(orient="records"),
        }
    else:
        report["best_configurations_by_metric"] = {}

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
        "best_configuration": result["best_configuration"],
        "top_configurations": result["user_holdout_model_summary"][:5],
    }, ensure_ascii=False, indent=2))
