from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from official.calibration import BetaCalibrator, IsotonicCalibrator, SigmoidCalibrator
from official.common import RANDOM_SEED, load_official_paths, save_pickle
from official.thresholding import search_threshold


class IdentityCalibrator:
    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "IdentityCalibrator":
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return np.asarray(probabilities, dtype=float)


STACKER_FEATURE_COLUMNS = [
    "base_a_probability",
    "base_b_probability",
    "base_c_probability",
    "rule_score",
    "anomaly_score",
]


CALIBRATION_CANDIDATES = {
    "raw": IdentityCalibrator,
    "sigmoid": SigmoidCalibrator,
    "beta": BetaCalibrator,
    "isotonic": IsotonicCalibrator,
}


def fit_logistic_stacker(frame: pd.DataFrame, feature_columns: list[str]) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(frame[feature_columns], frame["status"].astype(int))
    return model


def build_stacker_oof(
    base_oof_frame: pd.DataFrame,
    split_frame: pd.DataFrame,
    fold_column: str = "primary_fold",
) -> tuple[pd.DataFrame, LogisticRegression]:
    if fold_column in base_oof_frame.columns:
        frame = base_oof_frame.copy()
    else:
        frame = base_oof_frame.merge(split_frame[["user_id", fold_column]], on="user_id", how="left")
    if frame[fold_column].isna().any():
        raise ValueError(f"Missing fold assignments in {fold_column}")

    oof_rows: list[pd.DataFrame] = []
    for fold_id in sorted(int(value) for value in frame[fold_column].dropna().unique()):
        valid_frame = frame[frame[fold_column] == fold_id].copy()
        train_frame = frame[frame[fold_column] != fold_id].copy()
        model = fit_logistic_stacker(train_frame, STACKER_FEATURE_COLUMNS)
        valid_frame["stacker_raw_probability"] = model.predict_proba(valid_frame[STACKER_FEATURE_COLUMNS])[:, 1]
        oof_rows.append(valid_frame)
    oof_frame = pd.concat(oof_rows, ignore_index=True).sort_values("user_id").reset_index(drop=True)
    final_model = fit_logistic_stacker(frame, STACKER_FEATURE_COLUMNS)
    return oof_frame, final_model


def choose_best_calibration_and_threshold(
    raw_probabilities: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray | None,
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
        threshold_report = search_threshold(labels, calibrated, group_ids)
        selected_row = threshold_report["selected_row"]
        ap = float(average_precision_score(labels, calibrated))
        candidate_report = {
            "method": method,
            "average_precision": ap,
            "selected_threshold": float(threshold_report["selected_threshold"]),
            "selected_row": dict(selected_row),
            "threshold_report": threshold_report,
        }
        candidate_rows.append(candidate_report)
        rank = (
            float(selected_row["bootstrap_mean_f1"]),
            ap,
            -float(selected_row["fpr"]),
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_payload = (candidate_report, calibrator, calibrated)

    assert best_payload is not None
    selected_report, calibrator, calibrated = best_payload
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
    }
    return report, calibrator, calibrated


def save_stacker_model(model: LogisticRegression, path: Path) -> None:
    save_pickle(model, path)
