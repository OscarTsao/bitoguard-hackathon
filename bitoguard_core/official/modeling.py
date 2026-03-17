from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from hardware import catboost_runtime_params, hardware_profile, lightgbm_runtime_params
from official.common import RANDOM_SEED, encode_frame


@dataclass
class ModelFitResult:
    model_name: str
    model: Any
    feature_columns: list[str]
    encoded_columns: list[str] | None
    cat_features: list[str] | None
    validation_probabilities: list[float] | None


def _pu_sample_weight(y_train: pd.Series, negative_weight: float) -> np.ndarray:
    labels = y_train.astype(int).to_numpy()
    positives = max(1, int(labels.sum()))
    negatives = max(1, len(labels) - positives)
    return np.where(labels == 1, negatives / positives, float(negative_weight)).astype(np.float32)


def fit_lgbm(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame | None,
    feature_columns: list[str],
    negative_weight: float = 1.0,
) -> ModelFitResult:
    x_train, encoded_columns = encode_frame(train_frame, feature_columns)
    y_train = train_frame["status"].astype(int)
    sample_weight = _pu_sample_weight(y_train, negative_weight)
    runtime_params = lightgbm_runtime_params()
    model_kwargs = dict(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_SEED,
        verbosity=-1,
        **runtime_params,
    )
    model = LGBMClassifier(**model_kwargs)
    validation_probabilities: list[float] | None = None
    x_valid = None
    y_valid = None
    if valid_frame is not None and not valid_frame.empty:
        x_valid, _ = encode_frame(valid_frame, feature_columns, reference_columns=encoded_columns)
        y_valid = valid_frame["status"].astype(int)
    try:
        if x_valid is not None and y_valid is not None:
            model.fit(x_train, y_train, sample_weight=sample_weight, eval_set=[(x_valid, y_valid)], eval_metric="binary_logloss")
            validation_probabilities = model.predict_proba(x_valid)[:, 1].tolist()
        else:
            model.fit(x_train, y_train, sample_weight=sample_weight)
    except Exception:
        # GPU may be unavailable at runtime despite detection; retry on CPU.
        if runtime_params.get("device_type") != "gpu":
            raise
        cpu_runtime = {"n_jobs": hardware_profile().cpu_threads}
        model = LGBMClassifier(**{
            **model_kwargs,
            **cpu_runtime,
            "device_type": "cpu",
        })
        if x_valid is not None and y_valid is not None:
            model.fit(x_train, y_train, sample_weight=sample_weight, eval_set=[(x_valid, y_valid)], eval_metric="binary_logloss")
            validation_probabilities = model.predict_proba(x_valid)[:, 1].tolist()
        else:
            model.fit(x_train, y_train, sample_weight=sample_weight)
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
    focal_gamma: float = 0.0,
    negative_weight: float = 1.0,
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
    _positives = max(1, int(y_train.sum()))
    _negatives = max(1, len(y_train) - _positives)
    # Pass class imbalance via class_weights in constructor rather than sample_weight in
    # fit(). CatBoost GPU handles class_weights more stably (avoids probability inflation
    # seen when sample_weight values are large). focal_gamma param accepted but ignored
    # (kept for API compatibility; CatBoost Logloss is used for all cases).
    class_weights = [float(negative_weight), _negatives / _positives]
    runtime_params = catboost_runtime_params()
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        class_weights=class_weights,
        random_seed=RANDOM_SEED,
        verbose=False,
        **runtime_params,
    )
    validation_probabilities: list[float] | None = None
    if valid_frame is not None and not valid_frame.empty:
        y_valid = valid_frame["status"].astype(int)
        try:
            model.fit(train_frame[feature_columns], y_train, cat_features=cat_features,
                      eval_set=(valid_frame[feature_columns], y_valid), use_best_model=False)
        except Exception:
            if runtime_params.get("task_type") != "GPU":
                raise
            model = CatBoostClassifier(
                loss_function="Logloss", eval_metric="Logloss",
                class_weights=class_weights,
                random_seed=RANDOM_SEED, verbose=False,
                task_type="CPU", thread_count=hardware_profile().cpu_threads,
            )
            model.fit(train_frame[feature_columns], y_train, cat_features=cat_features,
                      eval_set=(valid_frame[feature_columns], y_valid), use_best_model=False)
        validation_probabilities = model.predict_proba(valid_frame[feature_columns])[:, 1].tolist()
    else:
        try:
            model.fit(train_frame[feature_columns], y_train, cat_features=cat_features)
        except Exception:
            if runtime_params.get("task_type") != "GPU":
                raise
            model = CatBoostClassifier(
                loss_function="Logloss", eval_metric="Logloss",
                class_weights=class_weights,
                random_seed=RANDOM_SEED, verbose=False,
                task_type="CPU", thread_count=hardware_profile().cpu_threads,
            )
            model.fit(train_frame[feature_columns], y_train, cat_features=cat_features)
    # Sanity: warn if model outputs near-constant probabilities (indicates training failure)
    if validation_probabilities:
        _vp = np.array(validation_probabilities, dtype=float)
        if _vp.mean() < 1e-4 or _vp.max() < 0.01:
            import warnings
            warnings.warn(
                f"fit_catboost: validation probabilities collapsed "
                f"(mean={_vp.mean():.6f}, max={_vp.max():.6f}, focal_gamma={focal_gamma}). "
                "Check class_weights and focal_gamma interaction."
            )
    return ModelFitResult(
        model_name="catboost",
        model=model,
        feature_columns=feature_columns,
        encoded_columns=None,
        cat_features=cat_features,
        validation_probabilities=validation_probabilities,
    )
