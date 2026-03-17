from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def _metrics_at_predictions(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    precision = float(precision_score(labels, preds, zero_division=0))
    recall = float(recall_score(labels, preds, zero_division=0))
    f1 = float(f1_score(labels, preds, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": float(fp / max(1, fp + tn)),
        "predicted_positive_rate": float(preds.mean()),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def _threshold_candidates(probabilities: np.ndarray) -> list[float]:
    low_grid = np.arange(0.01, 0.30 + 1e-9, 0.01)
    high_grid = np.arange(0.35, 0.95 + 1e-9, 0.05)
    quantiles = np.quantile(probabilities, np.linspace(0.005, 0.995, 99))
    return sorted({round(float(x), 4) for x in np.concatenate([low_grid, high_grid, quantiles]) if 0.0 < float(x) < 1.0})


def _positive_rate_candidates() -> list[float]:
    return sorted({round(float(x), 4) for x in np.concatenate([np.arange(0.005, 0.05, 0.0025), np.arange(0.05, 0.10 + 1e-9, 0.005)])})


def select_best_rule(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    rows: list[dict[str, Any]] = []
    for threshold in _threshold_candidates(probabilities):
        preds = (probabilities >= threshold).astype(int)
        rows.append({"rule_type": "threshold", "threshold": float(threshold), **_metrics_at_predictions(labels, preds)})
    order = np.argsort(-probabilities)
    for positive_rate in _positive_rate_candidates():
        k = max(1, int(round(positive_rate * len(labels))))
        preds = np.zeros(len(labels), dtype=int)
        preds[order[:k]] = 1
        rows.append(
            {
                "rule_type": "positive_rate",
                "positive_rate": float(positive_rate),
                "top_k": int(k),
                **_metrics_at_predictions(labels, preds),
            }
        )

    best_f1 = max(row["f1"] for row in rows)
    plateau = [row for row in rows if row["f1"] >= best_f1 * 0.99]
    plateau.sort(key=lambda row: (-row["f1"], -row["precision"], row["fpr"], row["rule_type"] != "threshold"))
    selected = plateau[0]
    return {
        "selected_rule": dict(selected),
        "rows": rows,
        "selection_basis": {
            "plateau_fraction": 0.99,
            "tie_breakers": ["precision", "fpr", "prefer_threshold_if_equal"],
        },
    }


def apply_rule(probabilities: np.ndarray, rule: dict[str, Any]) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    rule_type = rule["rule_type"]
    if rule_type == "threshold":
        return (probabilities >= float(rule["threshold"])).astype(int)
    positive_rate = float(rule["positive_rate"])
    k = max(1, int(round(positive_rate * len(probabilities))))
    order = np.argsort(-probabilities)
    preds = np.zeros(len(probabilities), dtype=int)
    preds[order[:k]] = 1
    return preds
