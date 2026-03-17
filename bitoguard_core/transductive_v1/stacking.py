from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from transductive_v1.common import RANDOM_SEED


CANDIDATE_STACKER_FEATURES = [
    "base_a_probability",
    "base_b_probability",
    "graph_risk_score",
    "rule_score",
    "anomaly_score",
    "projected_component_log_size",
    "connected_flag",
]


class PriorStacker:
    def __init__(self, positive_rate: float) -> None:
        self.positive_rate = float(min(max(positive_rate, 1e-6), 1 - 1e-6))

    def predict_proba(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows = len(frame)
        negative = 1.0 - self.positive_rate
        return np.column_stack([
            np.full(rows, negative, dtype=float),
            np.full(rows, self.positive_rate, dtype=float),
        ])


def resolve_stacker_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in CANDIDATE_STACKER_FEATURES if column in frame.columns]


def fit_stacker(frame: pd.DataFrame, feature_columns: list[str]) -> LogisticRegression:
    labels = frame["status"].astype(int)
    if labels.nunique() < 2:
        return PriorStacker(float(labels.mean()))  # type: ignore[return-value]
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(frame[feature_columns], labels)
    return model


def build_stacker_oof(base_oof_frame: pd.DataFrame, fold_column: str) -> tuple[pd.DataFrame, LogisticRegression, list[str]]:
    frame = base_oof_frame.copy()
    feature_columns = resolve_stacker_feature_columns(frame)
    rows = []
    for fold_id in sorted(int(value) for value in frame[fold_column].dropna().unique()):
        valid_frame = frame[frame[fold_column] == fold_id].copy()
        train_frame = frame[frame[fold_column] != fold_id].copy()
        model = fit_stacker(train_frame, feature_columns)
        valid_frame["stacker_raw_probability"] = model.predict_proba(valid_frame[feature_columns])[:, 1]
        rows.append(valid_frame)
    oof = pd.concat(rows, ignore_index=True).sort_values("user_id").reset_index(drop=True)
    final_model = fit_stacker(frame, feature_columns)
    return oof, final_model, feature_columns
