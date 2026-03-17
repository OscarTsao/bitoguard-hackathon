from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from transductive_v1.common import RANDOM_SEED, model_path, save_pickle
from transductive_v1.decision_rule import select_best_rule


def _clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)


class IdentityCalibrator:
    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "IdentityCalibrator":
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return _clip_probabilities(probabilities)


class SigmoidCalibrator:
    def __init__(self) -> None:
        self.model = LogisticRegression(random_state=RANDOM_SEED)

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "SigmoidCalibrator":
        self.model.fit(_clip_probabilities(probabilities).reshape(-1, 1), labels)
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(_clip_probabilities(probabilities).reshape(-1, 1))[:, 1]


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
        self.model.fit(_clip_probabilities(probabilities), labels)
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(_clip_probabilities(probabilities)), dtype=float)


CALIBRATOR_BUILDERS = {
    "raw": IdentityCalibrator,
    "sigmoid": SigmoidCalibrator,
    "beta": BetaCalibrator,
    "isotonic": IsotonicCalibrator,
}


def select_best_calibration(
    raw_probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[str, Any], Any, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    raw_probabilities = np.asarray(raw_probabilities, dtype=float)
    candidate_rows: list[dict[str, Any]] = []
    best_rank: tuple[float, float, float] | None = None
    best_payload: tuple[dict[str, Any], Any, np.ndarray] | None = None
    for method, builder in CALIBRATOR_BUILDERS.items():
        calibrator = builder().fit(raw_probabilities, labels)
        calibrated = _clip_probabilities(calibrator.predict(raw_probabilities))
        decision_report = select_best_rule(calibrated, labels)
        selected_rule = decision_report["selected_rule"]
        ap = float(average_precision_score(labels, calibrated))
        candidate = {
            "method": method,
            "average_precision": ap,
            "decision_report": decision_report,
            "selected_rule": selected_rule,
        }
        candidate_rows.append(candidate)
        rank = (float(selected_rule["f1"]), ap, -float(selected_rule["fpr"]))
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_payload = (candidate, calibrator, calibrated)
    assert best_payload is not None
    selected, calibrator, calibrated = best_payload
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    calibrator_path = model_path(f"calibrator_{selected['method']}_{timestamp}.pkl")
    save_pickle(calibrator, calibrator_path)
    report = {
        "method": selected["method"],
        "average_precision": selected["average_precision"],
        "selected_rule": selected["selected_rule"],
        "decision_report": selected["decision_report"],
        "selection_basis": {
            "priority": ["best_f1", "best_average_precision", "lowest_fpr"],
        },
        "calibrator_path": str(calibrator_path),
        "candidates": [
            {
                "method": item["method"],
                "average_precision": item["average_precision"],
                "selected_rule": item["selected_rule"],
            }
            for item in candidate_rows
        ],
    }
    return report, calibrator, calibrated
