from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score

from official.calibration import BetaCalibrator, IsotonicCalibrator, SigmoidCalibrator
from official.common import RANDOM_SEED, load_official_paths, save_pickle
from official.thresholding import search_threshold
from models.pu_learning import estimate_c, pu_adjust


class IdentityCalibrator:
    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "IdentityCalibrator":
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return np.asarray(probabilities, dtype=float)


# Base model probabilities + meta-features fed to the stacker.
STACKER_FEATURE_COLUMNS = [
    "base_a_probability",
    "base_c_s_probability",
    "base_b_probability",
    "base_c_probability",
    "base_d_probability",
    "base_e_probability",
    "rule_score",
    "anomaly_score",
    "max_base_probability",
    "std_base_probability",
    "base_a_x_anomaly",
    "base_a_x_rule",
    "base_a_x_cs",
    "base_d_x_cs",
    "base_e_x_cs",
    "base_a_x_e",
    "base_cs_x_anomaly",
    "base_b_x_cs",
    "cs_deficit",
    "base_cs_x_crypto_anomaly",
    "base_a_x_crypto_anomaly",
]

_BLEND_CANDIDATE_COLUMNS = [
    "base_a_probability",
    "base_c_s_probability",
    "base_b_probability",
    "base_c_probability",
    "base_d_probability",
    "base_e_probability",
    "anomaly_score",
    "base_cs_x_anomaly",
    "base_cs_x_crypto_anomaly",
    "anomaly_score_segmented",
]

_BASE_PROB_COLUMNS = [
    "base_a_probability", "base_c_s_probability", "base_b_probability",
    "base_c_probability", "base_d_probability", "base_e_probability",
]

# Minimum AP threshold to include a model in the blend.
_MIN_AP_FOR_BLEND = 0.08


class BlendEnsemble:
    """AP-weighted linear blend ensemble."""

    def __init__(
        self,
        weights: dict[str, float],
        isolated_weights: dict[str, float] | None = None,
    ) -> None:
        self.weights = {k: float(v) for k, v in weights.items() if float(v) > 0}
        self.isolated_weights = (
            {k: float(v) for k, v in isolated_weights.items() if float(v) > 0}
            if isolated_weights else None
        )

    def _apply_weights(self, X: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
        blend = np.zeros(len(X), dtype=float)
        total_weight = 0.0
        for col, w in weights.items():
            if col in X.columns:
                vals = pd.to_numeric(X[col], errors="coerce").fillna(0.0).to_numpy()
                blend += w * vals
                total_weight += w
        if total_weight > 0:
            blend = blend / total_weight
        return blend

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.isolated_weights is not None and "cs_deficit" in X.columns:
            is_isolated = pd.to_numeric(X["cs_deficit"], errors="coerce").fillna(0.0).to_numpy() > 0.05
            blend_connected = self._apply_weights(X, self.weights)
            blend_isolated = self._apply_weights(X, self.isolated_weights)
            blend = np.where(is_isolated, blend_isolated, blend_connected)
        else:
            blend = self._apply_weights(X, self.weights)
        blend = np.clip(blend, 0.0, 1.0)
        return np.column_stack([1.0 - blend, blend])


def _add_base_meta_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute max/std across base model probabilities and interaction features."""
    frame = frame.copy()
    available = [c for c in _BASE_PROB_COLUMNS if c in frame.columns]
    if available:
        frame["max_base_probability"] = frame[available].max(axis=1)
        frame["std_base_probability"] = frame[available].std(axis=1).fillna(0.0)
    else:
        frame["max_base_probability"] = 0.0
        frame["std_base_probability"] = 0.0

    a = pd.to_numeric(frame.get("base_a_probability", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    anomaly = pd.to_numeric(frame.get("anomaly_score", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    rule = pd.to_numeric(frame.get("rule_score", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    cs = pd.to_numeric(frame.get("base_c_s_probability", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    d = pd.to_numeric(frame.get("base_d_probability", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    e = pd.to_numeric(frame.get("base_e_probability", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    b = pd.to_numeric(frame.get("base_b_probability", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    crypto_anomaly = pd.to_numeric(
        frame.get("crypto_anomaly_score", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)

    frame["base_a_x_anomaly"] = (a * anomaly).astype(np.float32)
    frame["base_a_x_rule"] = (a * rule).astype(np.float32)
    frame["base_a_x_cs"] = (a * cs).astype(np.float32)
    frame["base_d_x_cs"] = (d * cs).astype(np.float32)
    frame["base_e_x_cs"] = (e * cs).astype(np.float32)
    frame["base_a_x_e"] = (a * e).astype(np.float32)
    frame["base_cs_x_anomaly"] = (cs * anomaly).astype(np.float32)
    frame["base_b_x_cs"] = (b * cs).astype(np.float32)
    frame["cs_deficit"] = (a - cs).astype(np.float32)
    frame["base_cs_x_crypto_anomaly"] = (cs * crypto_anomaly).astype(np.float32)
    frame["base_a_x_crypto_anomaly"] = (a * crypto_anomaly).astype(np.float32)
    return frame


def tune_blend_weights(frame: pd.DataFrame) -> dict[str, float]:
    """Grid-search blend weights on OOF predictions to maximize F1."""
    labeled = frame.dropna(subset=["status"])
    y = labeled["status"].astype(int).to_numpy()

    eligible: dict[str, np.ndarray] = {}
    for col in _BLEND_CANDIDATE_COLUMNS:
        if col not in labeled.columns:
            continue
        vals = pd.to_numeric(labeled[col], errors="coerce").fillna(0.0).to_numpy()
        if vals.std() < 1e-6:
            continue
        ap = float(average_precision_score(y, vals))
        if ap >= _MIN_AP_FOR_BLEND:
            eligible[col] = vals

    if not eligible:
        eligible = {
            col: pd.to_numeric(labeled[col], errors="coerce").fillna(0.0).to_numpy()
            for col in _BLEND_CANDIDATE_COLUMNS
            if col in labeled.columns
        }

    if len(eligible) == 1:
        return {list(eligible)[0]: 1.0}

    if len(eligible) > 5:
        top5 = sorted(eligible, key=lambda c: float(average_precision_score(y, eligible[c])), reverse=True)[:5]
        eligible = {c: eligible[c] for c in top5}

    cols = list(eligible)
    arrays = np.stack([eligible[c] for c in cols], axis=0)
    n = len(cols)

    if n <= 5:
        step = 0.05
    elif n == 6:
        step = 0.10
    else:
        step = 0.15
    parts = round(1.0 / step)

    def _integer_compositions(total: int, k: int) -> list[list[int]]:
        if k == 1:
            return [[total]]
        result: list[list[int]] = []
        for first in range(total + 1):
            for rest in _integer_compositions(total - first, k - 1):
                result.append([first] + rest)
        return result

    combos: list[list[float]] = [[v * step for v in comp] for comp in _integer_compositions(parts, n)]
    if not combos:
        return {col: 1.0 / n for col in cols}

    combo_mat = np.array(combos, dtype=np.float32)
    blend_scores = (combo_mat @ arrays).astype(np.float32)

    thresholds = np.arange(0.05, 0.90, 0.01)
    y_bool = (y == 1)
    pos_total = int(y_bool.sum())
    best_f1_per_combo = np.zeros(len(combo_mat), dtype=np.float32)

    for t in thresholds:
        pred = blend_scores >= t
        tp = pred[:, y_bool].sum(axis=1).astype(np.float32)
        pp = pred.sum(axis=1).astype(np.float32)
        fp = pp - tp
        fn = pos_total - tp
        denom_p = tp + fp
        denom_r = tp + fn
        prec = np.where(denom_p > 0, tp / denom_p, 0.0)
        rec = np.where(denom_r > 0, tp / denom_r, 0.0)
        denom_f = prec + rec
        f1 = np.where(denom_f > 0, 2.0 * prec * rec / denom_f, 0.0)
        np.maximum(best_f1_per_combo, f1, out=best_f1_per_combo)

    best_idx = int(best_f1_per_combo.argmax())
    best_w = combo_mat[best_idx].tolist()
    return {col: float(w) for col, w in zip(cols, best_w) if w > 1e-6}


def _best_f1(y: np.ndarray, scores: np.ndarray) -> float:
    best = 0.0
    for t in np.arange(0.05, 0.90, 0.02):
        f = float(f1_score(y, (scores >= t).astype(int), zero_division=0))
        if f > best:
            best = f
    return best


def _fit_catboost_stacker(frame: pd.DataFrame, feature_columns: list[str]) -> Any:
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        return fit_logistic_stacker(frame, feature_columns)

    y = frame["status"].astype(int)
    positives = max(1, int(y.sum()))
    negatives = max(1, len(y) - positives)
    class_weight_ratio = min(float(negatives) / positives, 5.0)

    model = CatBoostClassifier(
        depth=3,
        iterations=400,
        learning_rate=0.05,
        l2_leaf_reg=15.0,
        min_data_in_leaf=30,
        random_strength=0.5,
        class_weights=[1.0, class_weight_ratio],
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=RANDOM_SEED,
        verbose=False,
    )
    x = frame[feature_columns].copy()
    for c in feature_columns:
        if x[c].dtype == bool or str(x[c].dtype) == "bool":
            x[c] = x[c].astype(int)
    model.fit(x, y)
    return model


def _predict_stacker(model: Any, frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    x = frame[feature_columns].copy()
    for c in feature_columns:
        if x[c].dtype == bool or str(x[c].dtype) == "bool":
            x[c] = x[c].astype(int)
    return model.predict_proba(x)[:, 1]


def fit_logistic_stacker(frame: pd.DataFrame, feature_columns: list[str]) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(frame[feature_columns], frame["status"].astype(int))
    return model


def _auto_select_best_stacker(
    frame: pd.DataFrame,
    fold_column: str,
) -> tuple[pd.DataFrame, Any]:
    """Auto-select between BlendEnsemble and CatBoost stacker via OOF F1."""
    stacker_cols = [c for c in STACKER_FEATURE_COLUMNS if c in frame.columns]

    blend_weights = tune_blend_weights(frame)
    blend_model = BlendEnsemble(blend_weights)
    blend_frame = frame.copy()
    blend_frame["stacker_raw_probability"] = blend_model.predict_proba(blend_frame[stacker_cols])[:, 1]
    labeled_blend = blend_frame.dropna(subset=["status"])
    blend_f1 = _best_f1(
        labeled_blend["status"].astype(int).to_numpy(),
        labeled_blend["stacker_raw_probability"].to_numpy(),
    )
    print(f"[stacker] BlendEnsemble peak OOF F1={blend_f1:.4f}, weights={blend_weights}")

    try:
        oof_rows: list[pd.DataFrame] = []
        for fold_id in sorted(int(v) for v in frame[fold_column].dropna().unique()):
            valid_f = frame[frame[fold_column] == fold_id].copy()
            train_f = frame[frame[fold_column] != fold_id].copy()
            cb_model = _fit_catboost_stacker(train_f, stacker_cols)
            valid_f["stacker_raw_probability"] = _predict_stacker(cb_model, valid_f, stacker_cols)
            oof_rows.append(valid_f)
        cb_oof = pd.concat(oof_rows, ignore_index=True).sort_values("user_id").reset_index(drop=True)
        labeled_cb = cb_oof.dropna(subset=["status"])
        cb_f1 = _best_f1(
            labeled_cb["status"].astype(int).to_numpy(),
            labeled_cb["stacker_raw_probability"].to_numpy(),
        )
        cb_final = _fit_catboost_stacker(frame, stacker_cols)
        print(f"[stacker] CatBoost stacker peak OOF F1={cb_f1:.4f}")
    except Exception as exc:
        print(f"[stacker] CatBoost stacker failed: {exc}")
        cb_f1 = -1.0
        cb_oof = None
        cb_final = None

    if cb_f1 > blend_f1 and cb_oof is not None:
        print(f"[stacker] Selected CatBoost stacker (F1={cb_f1:.4f} > blend F1={blend_f1:.4f})")
        return cb_oof, cb_final
    else:
        print(f"[stacker] Selected BlendEnsemble (F1={blend_f1:.4f})")
        return blend_frame, blend_model


def build_stacker_oof(
    base_oof_frame: pd.DataFrame,
    split_frame: pd.DataFrame,
    fold_column: str = "primary_fold",
    use_blend: bool = True,
) -> tuple[pd.DataFrame, Any]:
    if fold_column in base_oof_frame.columns:
        frame = base_oof_frame.copy()
    else:
        frame = base_oof_frame.merge(split_frame[["user_id", fold_column]], on="user_id", how="left")
    if frame[fold_column].isna().any():
        raise ValueError(f"Missing fold assignments in {fold_column}")

    frame = _add_base_meta_features(frame)

    if use_blend:
        return _auto_select_best_stacker(frame, fold_column)

    # Fallback: fold-by-fold LR stacker
    available_cols = [c for c in STACKER_FEATURE_COLUMNS if c in frame.columns]
    oof_rows: list[pd.DataFrame] = []
    for fold_id in sorted(int(value) for value in frame[fold_column].dropna().unique()):
        valid_frame = frame[frame[fold_column] == fold_id].copy()
        train_frame = frame[frame[fold_column] != fold_id].copy()
        model = fit_logistic_stacker(train_frame, available_cols)
        valid_frame["stacker_raw_probability"] = model.predict_proba(valid_frame[available_cols])[:, 1]
        oof_rows.append(valid_frame)
    oof_frame = pd.concat(oof_rows, ignore_index=True).sort_values("user_id").reset_index(drop=True)
    final_model = fit_logistic_stacker(frame, available_cols)
    return oof_frame, final_model


CALIBRATION_CANDIDATES = {
    "raw": IdentityCalibrator,
    "sigmoid": SigmoidCalibrator,
    "beta": BetaCalibrator,
    "isotonic": IsotonicCalibrator,
}


def choose_best_calibration_and_threshold(
    raw_probabilities: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray | None,
    use_pu_adjustment: bool = True,
) -> tuple[dict[str, Any], Any, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    raw_probabilities = np.asarray(raw_probabilities, dtype=float)
    paths = load_official_paths()
    candidate_rows: list[dict[str, Any]] = []
    best_rank: tuple[float, float, float] | None = None
    best_payload: tuple[dict[str, Any], Any, np.ndarray] | None = None

    for method, builder in CALIBRATION_CANDIDATES.items():
        calibrator = builder().fit(raw_probabilities, labels)
        calibrated = calibrator.predict(raw_probabilities)

        if use_pu_adjustment:
            c_estimate = estimate_c(calibrated, labels)
            pu_calibrated = pu_adjust(calibrated, c_estimate)
        else:
            c_estimate = None
            pu_calibrated = calibrated

        threshold_report = search_threshold(labels, pu_calibrated, group_ids, beta=1.0)
        selected_row = threshold_report["selected_row"]
        ap = float(average_precision_score(labels, pu_calibrated))
        candidate_report = {
            "method": method,
            "average_precision": ap,
            "selected_threshold": float(threshold_report["selected_threshold"]),
            "selected_row": dict(selected_row),
            "threshold_report": threshold_report,
            "pu_c_estimate": float(c_estimate) if c_estimate is not None else None,
        }
        candidate_rows.append(candidate_report)
        rank = (
            float(selected_row["bootstrap_mean_f1"]),
            ap,
            -float(selected_row["fpr"]),
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_payload = (candidate_report, calibrator, pu_calibrated, c_estimate)

    assert best_payload is not None
    selected_report, calibrator, pu_calibrated, c_estimate = best_payload
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    calibrator_path = paths.model_dir / f"official_stacker_calibrator_{selected_report['method']}_{timestamp}.pkl"
    save_pickle(calibrator, calibrator_path)
    report = {
        "method": selected_report["method"],
        "average_precision": selected_report["average_precision"],
        "selected_threshold": selected_report["selected_threshold"],
        "selected_row": selected_report["selected_row"],
        "threshold_report": selected_report["threshold_report"],
        "calibrator_path": str(calibrator_path),
        "candidates": candidate_rows,
        "selection_basis": {
            "priority": ["best_bootstrap_mean_f1", "best_average_precision", "lowest_fpr"],
        },
        "pu_c_estimate": float(c_estimate) if c_estimate is not None else None,
        "pu_adjustment_enabled": use_pu_adjustment,
    }
    return report, calibrator, pu_calibrated


def save_stacker_model(model: Any, path: Path) -> None:
    save_pickle(model, path)
