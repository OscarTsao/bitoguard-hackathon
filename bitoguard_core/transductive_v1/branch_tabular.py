from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from transductive_v1.common import RANDOM_SEED


CATBOOST_ITERATIONS = 300
CATBOOST_USED_RAM_LIMIT = "8gb"


def _resolved_catboost_iterations() -> int:
    return int(os.getenv("BITOGUARD_TV1_CATBOOST_ITERATIONS", str(CATBOOST_ITERATIONS)))


def _resolved_catboost_ram_limit() -> str:
    return os.getenv("BITOGUARD_TV1_CATBOOST_USED_RAM_LIMIT", CATBOOST_USED_RAM_LIMIT)


class PriorProbabilityModel:
    def __init__(self, positive_rate: float) -> None:
        self.positive_rate = float(min(max(positive_rate, 1e-6), 1 - 1e-6))

    def predict_proba(self, frame: pd.DataFrame) -> pd.DataFrame:
        negative = 1.0 - self.positive_rate
        rows = len(frame)
        return np.column_stack([
            np.full(rows, negative, dtype=float),
            np.full(rows, self.positive_rate, dtype=float),
        ])


@dataclass
class TabularFitResult:
    model_name: str
    model: Any
    feature_columns: list[str]
    validation_probabilities: list[float] | None


def fit_catboost(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame | None,
    feature_columns: list[str],
) -> TabularFitResult:
    usable_columns = [
        column
        for column in feature_columns
        if column in train_frame.columns and pd.to_numeric(train_frame[column], errors="coerce").nunique(dropna=True) > 1
    ]
    y_train = train_frame["status"].astype(int)
    positives = max(1, int(y_train.sum()))
    negatives = max(1, len(y_train) - positives)
    if not usable_columns:
        model = PriorProbabilityModel(positives / max(1, len(y_train)))
        validation_probabilities = None
        if valid_frame is not None and not valid_frame.empty:
            validation_probabilities = model.predict_proba(valid_frame)[:, 1].tolist()
        return TabularFitResult(
            model_name="prior_probability",
            model=model,
            feature_columns=[],
            validation_probabilities=validation_probabilities,
        )
    class_weights = [1.0, negatives / positives]
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=RANDOM_SEED,
        verbose=False,
        iterations=_resolved_catboost_iterations(),
        learning_rate=0.05,
        depth=6,
        class_weights=class_weights,
        used_ram_limit=_resolved_catboost_ram_limit(),
        allow_writing_files=False,
    )
    validation_probabilities: list[float] | None = None
    train_x = train_frame[usable_columns].copy()
    train_x[usable_columns] = train_x[usable_columns].astype(np.float32)
    if valid_frame is not None and not valid_frame.empty:
        y_valid = valid_frame["status"].astype(int)
        valid_x = valid_frame[usable_columns].copy()
        valid_x[usable_columns] = valid_x[usable_columns].astype(np.float32)
        model.fit(train_x, y_train, eval_set=(valid_x, y_valid), use_best_model=False)
        validation_probabilities = model.predict_proba(valid_x)[:, 1].tolist()
    else:
        model.fit(train_x, y_train)
    return TabularFitResult(
        model_name="catboost",
        model=model,
        feature_columns=usable_columns,
        validation_probabilities=validation_probabilities,
    )
