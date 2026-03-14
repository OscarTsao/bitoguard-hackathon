from __future__ import annotations

import json

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
)

from config import load_settings
from db.store import DuckDBStore, make_id, utc_now
from models.common import encode_features, feature_columns, forward_date_splits, load_lgbm, training_dataset


def _load_latest(prefix: str) -> tuple[object, dict]:
    settings = load_settings()
    model_files = sorted((settings.artifact_dir / "models").glob(f"{prefix}_*.lgbm"))
    model_path = model_files[-1]
    meta = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    return load_lgbm(model_path), meta


def _precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of positives among the top-K scored users."""
    if k <= 0:
        return 0.0
    top_k_idx = np.argsort(scores)[::-1][:k]
    return float(y_true[top_k_idx].sum() / k)


def _recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of all positives captured in the top-K scored users."""
    total_pos = y_true.sum()
    if total_pos == 0:
        return 0.0
    if k <= 0:
        return 0.0
    top_k_idx = np.argsort(scores)[::-1][:k]
    return float(y_true[top_k_idx].sum() / total_pos)


def _top_feature_importance(model: object, encoded_columns: list[str], top_n: int = 20) -> list[dict]:
    """Extract LightGBM feature importances (gain-based)."""
    try:
        importances = model.feature_importance(importance_type="gain")
        pairs = sorted(
            zip(encoded_columns, importances.tolist()),
            key=lambda x: -x[1],
        )[:top_n]
        total = max(1.0, sum(imp for _, imp in pairs))
        return [
            {"feature": feat, "importance_gain": round(imp, 4), "importance_pct": round(100 * imp / total, 2)}
            for feat, imp in pairs
        ]
    except Exception:
        return []


def _calibration_summary(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> dict:
    """Compute calibration (reliability diagram) data."""
    try:
        fraction_of_pos, mean_predicted = calibration_curve(y_true, probabilities, n_bins=n_bins)
        brier = float(np.mean((probabilities - y_true) ** 2))
        return {
            "brier_score": round(brier, 6),
            "n_bins": n_bins,
            "bins": [
                {"mean_predicted": round(float(mp), 4), "fraction_positive": round(float(fp), 4)}
                for mp, fp in zip(mean_predicted, fraction_of_pos)
            ],
        }
    except Exception as exc:
        return {"error": str(exc)}


def validate_model() -> dict:
    settings = load_settings()
    store = DuckDBStore(settings.db_path)
    dataset = training_dataset().sort_values("snapshot_date").reset_index(drop=True)
    date_splits = forward_date_splits(dataset["snapshot_date"])
    # Determine which split to evaluate on — holdout is required; raise if missing
    holdout_dates_list = date_splits.get("holdout") or []
    if not holdout_dates_list:
        # Try valid as fallback but be explicit about it
        holdout_dates_list = date_splits.get("valid") or []
        split_used = "valid"
        if not holdout_dates_list:
            raise RuntimeError(
                "No holdout or valid split available for validation. "
                "The dataset may be too small to create temporal splits."
            )
    else:
        split_used = "holdout"
    holdout_dates = set(holdout_dates_list)
    holdout = dataset[dataset["snapshot_date"].dt.date.isin(holdout_dates)].copy().reset_index(drop=True)

    model, meta = _load_latest("lgbm")
    feature_cols = feature_columns(holdout)
    encoded, enc_cols = encode_features(holdout, feature_cols, reference_columns=meta["encoded_columns"])
    y_true = holdout["hidden_suspicious_label"].astype(int).to_numpy()
    probabilities = model.predict(encoded)
    preds = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    fpr = fp / max(1, fp + tn)
    pr_precision, pr_recall, thresholds = precision_recall_curve(y_true, probabilities)

    # ── Precision@K / Recall@K ────────────────────────────────────────────────
    n_pos = int(y_true.sum())
    k_values = [50, 100, 200, 500, n_pos]
    precision_at_k = {f"P@{k}": round(_precision_at_k(y_true, probabilities, k), 4) for k in k_values}
    recall_at_k = {f"R@{k}": round(_recall_at_k(y_true, probabilities, k), 4) for k in k_values}

    # ── Threshold sensitivity ─────────────────────────────────────────────────
    threshold_report = []
    for threshold in [round(x, 2) for x in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]]:
        threshold_preds = (probabilities >= threshold).astype(int)
        threshold_report.append({
            "threshold": threshold,
            "precision": precision_score(y_true, threshold_preds, zero_division=0),
            "recall": recall_score(y_true, threshold_preds, zero_division=0),
            "f1": f1_score(y_true, threshold_preds, zero_division=0),
        })

    # ── Scenario breakdown ────────────────────────────────────────────────────
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

    # ── Calibration ───────────────────────────────────────────────────────────
    calibration = _calibration_summary(y_true, probabilities)

    # ── Feature importance ────────────────────────────────────────────────────
    feature_importance = _top_feature_importance(model, enc_cols, top_n=20)

    report = {
        "model_version": meta["model_version"],
        "split_used": split_used,
        "holdout_rows": int(len(holdout)),
        "holdout_positives": n_pos,
        "holdout_negatives": int(len(holdout)) - n_pos,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "calibration": calibration,
        "feature_importance_top20": feature_importance,
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
    r = validate_model()
    # Print compact summary
    print(f"\nModel: {r['model_version']}")
    print(f"Holdout: {r['holdout_rows']} rows, {r['holdout_positives']} pos, {r['holdout_negatives']} neg")
    print(f"P={r['precision']:.4f}  R={r['recall']:.4f}  F1={r['f1']:.4f}  PR-AUC={r['average_precision']:.4f}  FPR={r['fpr']:.4f}")
    print(f"\nPrecision@K: {r['precision_at_k']}")
    print(f"Recall@K:    {r['recall_at_k']}")
    print(f"\nCalibration — Brier score: {r['calibration'].get('brier_score', 'N/A')}")
    print("\nTop-5 features by gain importance:")
    for feat in r["feature_importance_top20"][:5]:
        print(f"  {feat['feature']:<45} {feat['importance_pct']:>6.2f}%")
