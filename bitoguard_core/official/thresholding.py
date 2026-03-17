from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from official.common import RANDOM_SEED


def _metrics_at_threshold(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probabilities >= threshold).astype(int)
    precision = float(precision_score(labels, preds, zero_division=0))
    recall = float(recall_score(labels, preds, zero_division=0))
    f1 = float(f1_score(labels, preds, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    fpr = float(fp / max(1, fp + tn))
    predicted_positive_rate = float(preds.mean())
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "predicted_positive_rate": predicted_positive_rate,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def _group_bootstrap_f1(
    labels: np.ndarray,
    probabilities: np.ndarray,
    group_ids: np.ndarray | None,
    threshold: float,
    n_bootstrap: int,
) -> tuple[float, float]:
    preds = (probabilities >= threshold).astype(int)
    if group_ids is None:
        f1 = float(f1_score(labels, preds, zero_division=0))
        return f1, 0.0

    rng = np.random.default_rng(RANDOM_SEED)
    unique_groups = np.unique(group_ids)
    if len(unique_groups) <= 1:
        f1 = float(f1_score(labels, preds, zero_division=0))
        return f1, 0.0

    scores: list[float] = []
    for _ in range(n_bootstrap):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        mask = np.isin(group_ids, sampled_groups)
        if mask.sum() == 0:
            continue
        scores.append(float(f1_score(labels[mask], preds[mask], zero_division=0)))
    if not scores:
        f1 = float(f1_score(labels, preds, zero_division=0))
        return f1, 0.0
    return float(np.mean(scores)), float(np.std(scores))


def _threshold_candidates(probabilities: np.ndarray) -> list[float]:
    low_grid = np.arange(0.01, 0.30 + 1e-9, 0.01)
    high_grid = np.arange(0.35, 0.95 + 1e-9, 0.05)
    quantiles = np.quantile(probabilities, np.linspace(0.005, 0.995, 199))
    merged = sorted({round(float(x), 4) for x in np.concatenate([low_grid, high_grid, quantiles]) if 0.0 < float(x) < 1.0})
    return merged


def _meets_constraints(row: dict[str, float], constraints: dict[str, float | None]) -> bool:
    precision_min = constraints.get("precision_min")
    fpr_max = constraints.get("fpr_max")
    min_predicted_positive_rate = constraints.get("min_predicted_positive_rate")
    if precision_min is not None and row["precision"] < precision_min:
        return False
    if fpr_max is not None and row["fpr"] > fpr_max:
        return False
    if min_predicted_positive_rate is not None and row["predicted_positive_rate"] < min_predicted_positive_rate:
        return False
    return True


def search_threshold(
    y_true: np.ndarray,
    p_cal: np.ndarray,
    group_ids: np.ndarray | None,
    constraints: dict[str, float | None] | None = None,
    n_bootstrap: int = 50,
    positive_rate_sanity_bounds: tuple[float, float] = (0.005, 0.10),
) -> dict[str, Any]:
    labels = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(p_cal, dtype=float)
    groups = None if group_ids is None else np.asarray(group_ids)
    constraints = constraints or {
        "precision_min": None,
        "fpr_max": None,
        "min_predicted_positive_rate": None,
    }

    rows: list[dict[str, Any]] = []
    for threshold in _threshold_candidates(probabilities):
        metrics = _metrics_at_threshold(labels, probabilities, threshold)
        mean_f1, std_f1 = _group_bootstrap_f1(labels, probabilities, groups, threshold, n_bootstrap)
        row = {
            "threshold": float(threshold),
            **metrics,
            "bootstrap_mean_f1": mean_f1,
            "bootstrap_std_f1": std_f1,
        }
        row["feasible"] = _meets_constraints(row, constraints)
        rows.append(row)

    feasible = [row for row in rows if row["feasible"]]
    candidate_pool = feasible if feasible else rows
    best_mean = max(row["bootstrap_mean_f1"] for row in candidate_pool)
    plateau = [row for row in candidate_pool if row["bootstrap_mean_f1"] >= best_mean * 0.99]
    plateau.sort(key=lambda row: (-row["bootstrap_mean_f1"], row["bootstrap_std_f1"], row["fpr"], -row["precision"]))
    selected = plateau[0]
    sanity_min, sanity_max = positive_rate_sanity_bounds
    positive_rate = float(selected["predicted_positive_rate"])
    sanity = {
        "min_expected_predicted_positive_rate": float(sanity_min),
        "max_expected_predicted_positive_rate": float(sanity_max),
        "selected_predicted_positive_rate": positive_rate,
        "requires_manual_review": bool(positive_rate < sanity_min or positive_rate > sanity_max),
    }
    return {
        "selected_threshold": selected["threshold"],
        "selected_row": dict(selected),
        "constraints": constraints,
        "rows": rows,
        "predicted_positive_rate_sanity": sanity,
        "selection_basis": {
            "best_mean_f1": best_mean,
            "plateau_fraction": 0.99,
            "tie_breakers": ["bootstrap_std_f1", "fpr", "precision"],
            "used_feasible_subset": bool(feasible),
        },
    }
