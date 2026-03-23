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


def fit_lgbm(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame | None,
    feature_columns: list[str],
    random_seed: int = RANDOM_SEED,
) -> ModelFitResult:
    x_train, encoded_columns = encode_frame(train_frame, feature_columns)
    y_train = train_frame["status"].astype(int)
    positives = max(1, int(y_train.sum()))
    negatives = max(1, len(y_train) - positives)
    runtime_params = lightgbm_runtime_params()
    model_kwargs = dict(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_seed,
        scale_pos_weight=negatives / positives,
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
            model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], eval_metric="binary_logloss")
            validation_probabilities = model.predict_proba(x_valid)[:, 1].tolist()
        else:
            model.fit(x_train, y_train)
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
    focal_gamma: float = 0.0,
    catboost_params: dict | None = None,
    random_seed: int = RANDOM_SEED,
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
    # Use Logloss + class_weights for all cases. CatBoost's built-in Focal loss
    # causes "Logloss metric shouldn't have focal_gamma parameter" validation errors
    # in catboost 1.2.x when combined with AUC eval_metric. More importantly,
    # combining focal_gamma > 0 with class_weights double-counts class imbalance.
    # Solution: Logloss + class_weights handles imbalance cleanly.
    # focal_gamma param is accepted but ignored (kept for API compatibility).
    hp = dict(catboost_params or {})
    _max_cw = float(__import__("os").environ.get("CB_MAX_CLASS_WEIGHT", str(hp.pop("max_class_weight", 10.0))))
    _weight_ratio = min(negatives / positives, float(_max_cw))
    runtime_params = catboost_runtime_params()
    # Override task_type from catboost_params (e.g. force CPU for Base B).
    if "task_type" in hp:
        runtime_params = dict(runtime_params)
        runtime_params["task_type"] = hp.pop("task_type")
        if runtime_params["task_type"] == "CPU":
            runtime_params.pop("devices", None)
            runtime_params["thread_count"] = hardware_profile().cpu_threads
    _iters = int(hp.pop("iterations", 1500))
    _esr = int(hp.pop("early_stopping_rounds", 100))
    _depth = int(hp.pop("depth", 7))
    _lr = float(hp.pop("learning_rate", 0.05))
    _l2 = float(hp.pop("l2_leaf_reg", 3.0))
    _bc = int(hp.pop("border_count", 254))
    _base_kwargs: dict = dict(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=_iters,
        random_seed=random_seed,
        verbose=False,
        class_weights=[1.0, _weight_ratio],
        depth=_depth,
        learning_rate=_lr,
        l2_leaf_reg=_l2,
        border_count=_bc,
        **runtime_params,
    )
    model_kwargs = _base_kwargs
    model = CatBoostClassifier(**model_kwargs)
    validation_probabilities: list[float] | None = None
    train_x = train_frame[feature_columns]
    valid_x = None
    y_valid = None
    if valid_frame is not None and not valid_frame.empty:
        valid_x = valid_frame[feature_columns]
        y_valid = valid_frame["status"].astype(int)
    try:
        if valid_x is not None and y_valid is not None:
            model.fit(train_x, y_train, cat_features=cat_features, eval_set=(valid_x, y_valid), use_best_model=True, early_stopping_rounds=_esr)
            validation_probabilities = model.predict_proba(valid_x)[:, 1].tolist()
        else:
            model.fit(train_x, y_train, cat_features=cat_features)
    except Exception:
        # Retry on CPU when GPU training is not usable on this runtime.
        if runtime_params.get("task_type") != "GPU":
            raise
        cpu_runtime = {"task_type": "CPU", "thread_count": hardware_profile().cpu_threads}
        model = CatBoostClassifier(**{
            **model_kwargs,
            **cpu_runtime,
        })
        if valid_x is not None and y_valid is not None:
            model.fit(train_x, y_train, cat_features=cat_features, eval_set=(valid_x, y_valid), use_best_model=True, early_stopping_rounds=_esr)
            validation_probabilities = model.predict_proba(valid_x)[:, 1].tolist()
        else:
            model.fit(train_x, y_train, cat_features=cat_features)
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
