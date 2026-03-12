from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from official.common import RANDOM_SEED, load_official_paths, save_pickle


def _clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)


class SigmoidCalibrator:
    def __init__(self) -> None:
        self.model = LogisticRegression(random_state=RANDOM_SEED)

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "SigmoidCalibrator":
        clipped = _clip_probabilities(probabilities).reshape(-1, 1)
        self.model.fit(clipped, labels)
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        clipped = _clip_probabilities(probabilities).reshape(-1, 1)
        return self.model.predict_proba(clipped)[:, 1]


class BetaCalibrator:
    def __init__(self) -> None:
        self.model = LogisticRegression(random_state=RANDOM_SEED)

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "BetaCalibrator":
        clipped = _clip_probabilities(probabilities)
        features = np.column_stack([np.log(clipped), np.log1p(-clipped)])
        self.model.fit(features, labels)
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        clipped = _clip_probabilities(probabilities)
        features = np.column_stack([np.log(clipped), np.log1p(-clipped)])
        return self.model.predict_proba(features)[:, 1]


class IsotonicCalibrator:
    def __init__(self) -> None:
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        clipped = _clip_probabilities(probabilities)
        self.model.fit(clipped, labels)
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        clipped = _clip_probabilities(probabilities)
        return np.asarray(self.model.predict(clipped), dtype=float)


CALIBRATOR_BUILDERS = {
    "sigmoid": SigmoidCalibrator,
    "beta": BetaCalibrator,
    "isotonic": IsotonicCalibrator,
}


@dataclass(frozen=True)
class CalibrationResult:
    method: str
    brier_score: float
    average_precision: float
    average_precision_delta: float
    calibrator_path: str


def fit_sigmoid_calibrator(probabilities: np.ndarray, labels: np.ndarray) -> SigmoidCalibrator:
    return SigmoidCalibrator().fit(probabilities, labels)


def fit_beta_calibrator(probabilities: np.ndarray, labels: np.ndarray) -> BetaCalibrator:
    return BetaCalibrator().fit(probabilities, labels)


def fit_isotonic_calibrator(probabilities: np.ndarray, labels: np.ndarray) -> IsotonicCalibrator:
    return IsotonicCalibrator().fit(probabilities, labels)


def choose_calibrator(
    probabilities: np.ndarray,
    labels: np.ndarray,
    max_average_precision_drop: float = 0.01,
) -> tuple[dict[str, Any], Any]:
    clipped = _clip_probabilities(probabilities)
    labels = np.asarray(labels, dtype=int)
    raw_ap = float(average_precision_score(labels, clipped))
    paths = load_official_paths()

    candidate_reports: list[dict[str, Any]] = []
    best_report: dict[str, Any] | None = None
    best_calibrator: Any | None = None
    for method, builder in CALIBRATOR_BUILDERS.items():
        calibrator = builder().fit(clipped, labels)
        calibrated = _clip_probabilities(calibrator.predict(clipped))
        ap = float(average_precision_score(labels, calibrated))
        delta = raw_ap - ap
        report = {
            "method": method,
            "brier_score": float(brier_score_loss(labels, calibrated)),
            "average_precision": ap,
            "average_precision_delta": float(delta),
            "accepted": bool(delta <= max_average_precision_drop),
        }
        candidate_reports.append(report)
        if not report["accepted"]:
            continue
        if best_report is None or report["brier_score"] < best_report["brier_score"]:
            best_report = report
            best_calibrator = calibrator

    if best_report is None:
        best_report = next(report for report in candidate_reports if report["method"] == "sigmoid")
        best_calibrator = SigmoidCalibrator().fit(clipped, labels)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    calibrator_path = paths.model_dir / f"official_calibrator_{best_report['method']}_{timestamp}.pkl"
    save_pickle(best_calibrator, calibrator_path)
    selected_report = dict(best_report)
    selected_report["calibrator_path"] = str(calibrator_path)
    selected_report["raw_average_precision"] = raw_ap
    selected_report["candidates"] = [dict(report) for report in candidate_reports]
    return selected_report, best_calibrator
