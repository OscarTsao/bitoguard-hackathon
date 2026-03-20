"""Optuna HPO for calibration method + threshold — jointly optimizing bootstrap-mean F1.

Strategy:
  Given OOF raw probabilities + labels, search over:
    - calibration method: raw / sigmoid / beta / isotonic
    - PU learning adjustment: on / off
    - threshold: [0.05, 0.60] (dense grid via suggest_float)

  Objective: bootstrap-mean F1 (group-aware if group_ids provided).
  This extends the existing choose_best_calibration_and_threshold() from grid search
  to Optuna TPE, which can find sharper calibration+threshold combinations missed
  by the coarse grid in search_threshold().

Usage (standalone — run after pipeline produces OOF predictions):
  cd bitoguard_core
  BITOGUARD_AWS_EVENT_CLEAN_DIR=data/aws_event/clean \\
    PYTHONPATH=. python -m official.hpo_threshold \\
    --n-trials 200 --timeout 600

Can also be called programmatically:
  from official.hpo_threshold import run_threshold_hpo
  result = run_threshold_hpo(raw_probs, labels, group_ids, n_trials=200)
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

_CALIBRATION_NAMES = ["raw", "sigmoid", "beta", "isotonic"]


def _bootstrap_f1(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    group_ids: np.ndarray | None,
    n_bootstrap: int = 50,
    seed: int = 42,
) -> float:
    preds = (probs >= threshold).astype(int)
    rng = np.random.default_rng(seed)

    if group_ids is None:
        # Simple stratified bootstrap.
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        scores: list[float] = []
        for _ in range(n_bootstrap):
            pi = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            ni = rng.choice(neg_idx, size=min(len(neg_idx), len(pos_idx) * 10), replace=True)
            idx = np.concatenate([pi, ni])
            l, p = labels[idx], preds[idx]
            tp = int(((p == 1) & (l == 1)).sum())
            fp = int(((p == 1) & (l == 0)).sum())
            fn = int(((p == 0) & (l == 1)).sum())
            denom = 2 * tp + fp + fn
            scores.append(float(2 * tp / denom) if denom > 0 else 0.0)
        return float(np.mean(scores)) if scores else 0.0
    else:
        unique_groups = np.unique(group_ids)
        if len(unique_groups) <= 1:
            tp = int(((preds == 1) & (labels == 1)).sum())
            fp = int(((preds == 1) & (labels == 0)).sum())
            fn = int(((preds == 0) & (labels == 1)).sum())
            denom = 2 * tp + fp + fn
            return float(2 * tp / denom) if denom > 0 else 0.0
        scores = []
        for _ in range(n_bootstrap):
            sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            mask = np.isin(group_ids, sampled_groups)
            if mask.sum() == 0:
                continue
            l, p = labels[mask], preds[mask]
            tp = int(((p == 1) & (l == 1)).sum())
            fp = int(((p == 1) & (l == 0)).sum())
            fn = int(((p == 0) & (l == 1)).sum())
            denom = 2 * tp + fp + fn
            scores.append(float(2 * tp / denom) if denom > 0 else 0.0)
        return float(np.mean(scores)) if scores else 0.0


def run_threshold_hpo(
    raw_probabilities: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray | None = None,
    n_trials: int = 200,
    timeout: float | None = 600.0,
    n_bootstrap: int = 50,
) -> dict[str, Any]:
    """Run Optuna HPO over calibration + threshold to maximize bootstrap-mean F1.

    Args:
        raw_probabilities: OOF raw (uncalibrated) probabilities from the stacker.
        labels: Ground-truth binary labels (0/1).
        group_ids: Optional group IDs for group-aware bootstrap resampling.
        n_trials: Number of Optuna trials.
        timeout: Time budget in seconds.
        n_bootstrap: Number of bootstrap samples per evaluation.

    Returns:
        dict with keys: method, threshold, bootstrap_f1, pu_adjusted, c_estimate,
                        all_trials (list of trial dicts).
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as exc:
        raise ImportError("optuna required: pip install optuna") from exc

    from official.calibration import BetaCalibrator, IsotonicCalibrator, SigmoidCalibrator
    from official.stacking import IdentityCalibrator
    from models.pu_learning import estimate_c, pu_adjust

    calibrator_map = {
        "raw": IdentityCalibrator,
        "sigmoid": SigmoidCalibrator,
        "beta": BetaCalibrator,
        "isotonic": IsotonicCalibrator,
    }

    # Pre-fit all calibrators once (calibrator fitting is fast).
    calibrated_cache: dict[str, np.ndarray] = {}
    pu_cache: dict[str, tuple[np.ndarray, float]] = {}
    for name, cls in calibrator_map.items():
        cal = cls().fit(raw_probabilities, labels)
        calibrated = cal.predict(raw_probabilities)
        calibrated_cache[name] = calibrated
        c_est = estimate_c(calibrated, labels)
        pu_probs = pu_adjust(calibrated, c_est)
        pu_cache[name] = (pu_probs, float(c_est))

    def objective(trial: Any) -> float:
        method = trial.suggest_categorical("method", _CALIBRATION_NAMES)
        use_pu = trial.suggest_categorical("use_pu", [True, False])
        threshold = trial.suggest_float("threshold", 0.04, 0.65)

        if use_pu:
            probs, _ = pu_cache[method]
        else:
            probs = calibrated_cache[method]

        return _bootstrap_f1(labels, probs, threshold, group_ids, n_bootstrap)

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=min(30, n_trials // 4))
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Seed with the existing best calibration choices.
    study.enqueue_trial({"method": "isotonic", "use_pu": True, "threshold": 0.15})
    study.enqueue_trial({"method": "raw", "use_pu": False, "threshold": 0.10})
    study.enqueue_trial({"method": "beta", "use_pu": True, "threshold": 0.12})
    study.enqueue_trial({"method": "sigmoid", "use_pu": True, "threshold": 0.15})

    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best = study.best_trial
    method = best.params["method"]
    use_pu = best.params["use_pu"]
    threshold = best.params["threshold"]

    if use_pu:
        best_probs, c_est = pu_cache[method]
    else:
        best_probs = calibrated_cache[method]
        c_est = None

    best_f1 = _bootstrap_f1(labels, best_probs, threshold, group_ids, n_bootstrap)
    print(f"[hpo_threshold] Best: method={method} pu={use_pu} thr={threshold:.4f} F1={best_f1:.4f}")

    return {
        "method": method,
        "threshold": float(threshold),
        "bootstrap_f1": float(best_f1),
        "pu_adjusted": bool(use_pu),
        "c_estimate": float(c_est) if c_est is not None else None,
        "n_trials": len(study.trials),
        "best_probabilities": best_probs,
    }


def run_pipeline_threshold_hpo(n_trials: int = 200, timeout: float = 600.0) -> dict[str, Any]:
    """Run threshold HPO on saved OOF predictions from the official pipeline.

    Reads artifacts/official_features/oof_predictions_primary_fold.parquet
    (or falls back to the model bundle's OOF) and runs HPO.
    Saves result to artifacts/reports/hpo_threshold_report.json.
    """
    import pandas as pd
    from official.common import load_official_paths, save_json

    paths = load_official_paths()

    # Load OOF frame from primary split.
    oof_parquet = paths.feature_dir / "official_oof_predictions.parquet"
    if not oof_parquet.exists():
        raise FileNotFoundError(
            f"OOF file not found: {oof_parquet}. Run the pipeline first."
        )
    oof = pd.read_parquet(oof_parquet)

    required = {"stacker_raw_probability", "status"}
    if not required.issubset(set(oof.columns)):
        raise ValueError(f"OOF frame missing columns: {required - set(oof.columns)}")

    oof = oof.dropna(subset=["stacker_raw_probability", "status"])
    raw_probs = oof["stacker_raw_probability"].to_numpy(dtype=float)
    labels = oof["status"].astype(int).to_numpy()
    group_ids = oof["group_id"].to_numpy() if "group_id" in oof.columns else None

    result = run_threshold_hpo(raw_probs, labels, group_ids, n_trials=n_trials, timeout=timeout)

    # Remove numpy array from saved result (not JSON-serializable).
    save_result = {k: v for k, v in result.items() if k != "best_probabilities"}
    paths.report_dir.mkdir(parents=True, exist_ok=True)
    save_json(save_result, paths.report_dir / "hpo_threshold_report.json")
    print(f"[hpo_threshold] Saved to {paths.report_dir / 'hpo_threshold_report.json'}")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Threshold + calibration HPO")
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    result = run_pipeline_threshold_hpo(n_trials=args.n_trials, timeout=args.timeout)
    print(json.dumps({k: v for k, v in result.items() if k != "best_probabilities"}, indent=2))
