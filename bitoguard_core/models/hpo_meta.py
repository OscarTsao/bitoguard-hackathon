"""Bayesian HPO for the stacker meta-learner and decision threshold.

Optimizes:
  1. class_weight ratio for LogisticRegression meta-learner
  2. Calibration method (isotonic vs sigmoid vs raw)
  3. Decision threshold for F1

Uses Optuna with OOF predictions from the stacker to avoid retraining branches.

Usage:
    # Standalone:
    from models.hpo_meta import optimize_meta_learner
    result = optimize_meta_learner(oof_matrix, y_true, n_trials=200)

    # Via environment variable in stacker.py:
    BITOGUARD_HPO=1 PYTHONPATH=. python models/stacker.py
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


def optimize_meta_learner(
    oof_matrix: np.ndarray,
    y_true: np.ndarray,
    n_trials: int = 200,
    n_cv_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """Optimize meta-learner class_weight + calibration + threshold.

    Uses cross-validated F1 on the OOF matrix to avoid overfitting to a single split.

    Args:
        oof_matrix: (n_samples, n_branches) OOF predictions (raw probabilities or logits)
        y_true: binary labels
        n_trials: number of Optuna trials
        n_cv_folds: folds for inner cross-validation (default 5)
        random_state: random seed for reproducibility

    Returns:
        dict with best_class_weight, best_calibration, best_threshold, best_f1
    """
    if not HAS_OPTUNA:
        raise ImportError("optuna is required: pip install optuna>=3.0")

    from sklearn.model_selection import StratifiedKFold

    X = oof_matrix.astype(np.float64)
    y = y_true.astype(int)

    def objective(trial: optuna.Trial) -> float:
        pos_weight = trial.suggest_float("pos_weight", 1.0, 50.0, log=True)
        C = trial.suggest_float("C", 0.01, 10.0, log=True)
        calibration = trial.suggest_categorical("calibration", ["isotonic", "sigmoid", "none"])
        threshold = trial.suggest_float("threshold", 0.05, 0.60)

        cv_scores: list[float] = []
        skf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=random_state)
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            lr = LogisticRegression(
                C=C,
                max_iter=500,
                random_state=random_state,
                class_weight={0: 1, 1: pos_weight},
            )
            lr.fit(X_tr, y_tr)

            if calibration == "isotonic":
                cal = CalibratedClassifierCV(lr, method="isotonic", cv="prefit")
                cal.fit(X_tr, y_tr)
                probs = cal.predict_proba(X_val)[:, 1]
            elif calibration == "sigmoid":
                cal = CalibratedClassifierCV(lr, method="sigmoid", cv="prefit")
                cal.fit(X_tr, y_tr)
                probs = cal.predict_proba(X_val)[:, 1]
            else:
                probs = lr.predict_proba(X_val)[:, 1]

            preds = (probs >= threshold).astype(int)
            cv_scores.append(f1_score(y_val, preds, zero_division=0))

        return float(np.mean(cv_scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    print(f"\nHPO Best F1 (CV): {best.value:.4f}")
    print(f"  pos_weight: {best.params['pos_weight']:.2f}")
    print(f"  C: {best.params['C']:.4f}")
    print(f"  calibration: {best.params['calibration']}")
    print(f"  threshold: {best.params['threshold']:.4f}")

    return {
        "best_f1": best.value,
        "best_pos_weight": best.params["pos_weight"],
        "best_C": best.params["C"],
        "best_calibration": best.params["calibration"],
        "best_threshold": best.params["threshold"],
        "n_trials": n_trials,
        "study": study,
    }


def build_optimized_meta_learner(
    oof_matrix: np.ndarray,
    y_true: np.ndarray,
    n_trials: int = 200,
    random_state: int = 42,
):
    """Convenience wrapper: run HPO and return a fitted meta-learner with best params.

    Returns: (fitted_model, best_threshold, hpo_result)
    """
    result = optimize_meta_learner(oof_matrix, y_true, n_trials=n_trials, random_state=random_state)

    lr = LogisticRegression(
        C=result["best_C"],
        max_iter=500,
        random_state=random_state,
        class_weight={0: 1, 1: result["best_pos_weight"]},
    )
    lr.fit(oof_matrix, y_true)

    calibration = result["best_calibration"]
    if calibration == "isotonic":
        from sklearn.calibration import CalibratedClassifierCV
        model = CalibratedClassifierCV(lr, method="isotonic", cv="prefit")
        model.fit(oof_matrix, y_true)
    elif calibration == "sigmoid":
        from sklearn.calibration import CalibratedClassifierCV
        model = CalibratedClassifierCV(lr, method="sigmoid", cv="prefit")
        model.fit(oof_matrix, y_true)
    else:
        model = lr

    return model, result["best_threshold"], result
