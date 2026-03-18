"""XGBoost Base E model fitting for the official pipeline.

XGBoost serves as a complementary tree-based model to CatBoost — it uses
a different regularization approach (column-block structure + GPU hist)
that can capture patterns CatBoost misses. Adding it to the stacker
provides ensemble diversity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from hardware import xgboost_runtime_params
from official.common import RANDOM_SEED, encode_frame
from official.modeling import ModelFitResult


def fit_xgboost(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame | None,
    feature_columns: list[str],
    params: dict[str, Any] | None = None,
    random_seed: int = RANDOM_SEED,
) -> ModelFitResult:
    """Fit XGBoost classifier with GPU support and early stopping.

    Parameters
    ----------
    train_frame : Training data with feature columns and 'status' label.
    valid_frame : Validation data (optional). If provided, early stopping is used.
    feature_columns : List of feature column names.
    params : Optional override params (depth, learning_rate, etc.).
    """
    x_train, encoded_columns = encode_frame(train_frame, feature_columns)
    y_train = train_frame["status"].astype(int)

    positives = max(1, int(y_train.sum()))
    negatives = max(1, len(y_train) - positives)
    # v41: Raise XGBoost scale_pos_weight cap from 10x to 15x.
    # Actual imbalance ~30x; 10x was leaving positives under-weighted.
    # CatBoost recovered with max_class_weight=10.6 (HPO best). XGBoost
    # handles class imbalance differently (direct gradient scaling vs tree-level),
    # so 15x provides stronger positive signal without gradient collapse risk.
    scale_pos_weight = min(float(negatives) / positives, 15.0)

    runtime_params = xgboost_runtime_params()
    p = params or {}

    model_kwargs = dict(
        n_estimators=p.get("n_estimators", 1500),
        max_depth=p.get("max_depth", 7),
        learning_rate=p.get("learning_rate", 0.05),
        subsample=p.get("subsample", 0.8),
        colsample_bytree=p.get("colsample_bytree", 0.8),
        reg_alpha=p.get("reg_alpha", 0.1),
        reg_lambda=p.get("reg_lambda", 5.0),
        min_child_weight=p.get("min_child_weight", 5),
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=p.get("random_state", random_seed),
        verbosity=0,
        early_stopping_rounds=100,
        **runtime_params,
    )

    validation_probabilities: list[float] | None = None
    if valid_frame is not None and not valid_frame.empty:
        x_valid, _ = encode_frame(valid_frame, feature_columns, reference_columns=encoded_columns)
        y_valid = valid_frame["status"].astype(int)
        model = XGBClassifier(**model_kwargs)
        try:
            model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
        except Exception:
            # GPU fallback
            if runtime_params.get("device") != "cuda":
                raise
            cpu_params = {**model_kwargs, "device": "cpu", "tree_method": "hist"}
            model = XGBClassifier(**cpu_params)
            model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
        validation_probabilities = model.predict_proba(x_valid)[:, 1].tolist()
    else:
        # No eval set — remove early_stopping_rounds
        model_kwargs.pop("early_stopping_rounds", None)
        model = XGBClassifier(**model_kwargs)
        try:
            model.fit(x_train, y_train)
        except Exception:
            if runtime_params.get("device") != "cuda":
                raise
            cpu_params = {**model_kwargs, "device": "cpu", "tree_method": "hist"}
            model = XGBClassifier(**cpu_params)
            model.fit(x_train, y_train)

    return ModelFitResult(
        model_name="xgboost",
        model=model,
        feature_columns=feature_columns,
        encoded_columns=encoded_columns,
        cat_features=None,
        validation_probabilities=validation_probabilities,
    )
