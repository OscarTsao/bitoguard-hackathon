from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from lightgbm import LGBMClassifier

from official.common import RANDOM_SEED, encode_frame


@dataclass
class ModelFitResult:
    model_name: str
    model: Any
    feature_columns: list[str]
    encoded_columns: list[str] | None
    cat_features: list[str] | None
    validation_probabilities: list[float] | None


def fit_lgbm(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame | None,
    feature_columns: list[str],
) -> ModelFitResult:
    x_train, encoded_columns = encode_frame(train_frame, feature_columns)
    y_train = train_frame["status"].astype(int)
    positives = max(1, int(y_train.sum()))
    negatives = max(1, len(y_train) - positives)
    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_SEED,
        scale_pos_weight=negatives / positives,
        verbosity=-1,
    )
    validation_probabilities: list[float] | None = None
    if valid_frame is not None and not valid_frame.empty:
        x_valid, _ = encode_frame(valid_frame, feature_columns, reference_columns=encoded_columns)
        y_valid = valid_frame["status"].astype(int)
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], eval_metric="binary_logloss")
        validation_probabilities = model.predict_proba(x_valid)[:, 1].tolist()
    else:
        model.fit(x_train, y_train)
    return ModelFitResult(
        model_name="lgbm",
        model=model,
        feature_columns=feature_columns,
        encoded_columns=encoded_columns,
        cat_features=None,
        validation_probabilities=validation_probabilities,
    )


def fit_catboost(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame | None,
    feature_columns: list[str],
) -> ModelFitResult:
    try:
        from catboost import CatBoostClassifier  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise ImportError("CatBoost is not installed. Install catboost to enable this path.") from exc

    cat_features = [
        column for column in feature_columns
        if pd.api.types.is_object_dtype(train_frame[column])
        or pd.api.types.is_string_dtype(train_frame[column])
        or pd.api.types.is_categorical_dtype(train_frame[column])
    ]
    y_train = train_frame["status"].astype(int)
    positives = max(1, int(y_train.sum()))
    negatives = max(1, len(y_train) - positives)
    class_weights = [1.0, negatives / positives]
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=RANDOM_SEED,
        verbose=False,
        class_weights=class_weights,
    )
    validation_probabilities: list[float] | None = None
    if valid_frame is not None and not valid_frame.empty:
        y_valid = valid_frame["status"].astype(int)
        model.fit(train_frame[feature_columns], y_train, cat_features=cat_features, eval_set=(valid_frame[feature_columns], y_valid), use_best_model=False)
        validation_probabilities = model.predict_proba(valid_frame[feature_columns])[:, 1].tolist()
    else:
        model.fit(train_frame[feature_columns], y_train, cat_features=cat_features)
    return ModelFitResult(
        model_name="catboost",
        model=model,
        feature_columns=feature_columns,
        encoded_columns=None,
        cat_features=cat_features,
        validation_probabilities=validation_probabilities,
    )
