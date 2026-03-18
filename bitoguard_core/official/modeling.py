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
    random_seed: int = RANDOM_SEED,
) -> ModelFitResult:
    import lightgbm as lgb
    x_train, encoded_columns = encode_frame(train_frame, feature_columns)
    y_train = train_frame["status"].astype(int)
    sample_weight = _pu_sample_weight(y_train, negative_weight)
    runtime_params = lightgbm_runtime_params()
    # v41: LightGBM hyperparameter improvements.
    # - num_leaves: 31 → 127 — old value severely under-capacitated the model for ~160 features.
    #   num_leaves=31 ≈ depth-5 tree; 127 ≈ depth-7, matching CatBoost depth=9 capacity better.
    #   LightGBM leaf-wise growth with 127 leaves can capture complex interactions without
    #   the depth bottleneck (CatBoost uses level-wise growth which is fundamentally different).
    # - n_estimators: 400 → 2000 — with early stopping, optimal tree count is found automatically.
    #   400 was fixed and possibly underfit; GPU training makes 2000 trees affordable.
    # - early_stopping_rounds: 50 — prevents overfitting. Previously missing entirely!
    #   Without this, LightGBM trained exactly 400 trees regardless of validation performance.
    # - min_child_samples: 20 — leaf regularization (reduces overfitting on rare positive class).
    # - reg_lambda: 5.0 — L2 regularization matching XGBoost (XGBoost also uses reg_lambda=5.0).
    # Expected: Base D AP from 0.2705 → 0.285+ (better generalization on 160 features).
    model_kwargs = dict(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=127,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_samples=20,
        reg_lambda=5.0,
        random_state=random_seed,
        verbosity=-1,
        **runtime_params,
    )
    _early_stop_callbacks = [
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(period=-1),
    ]
    model = LGBMClassifier(**model_kwargs)
    validation_probabilities: list[float] | None = None
    x_valid = None
    y_valid = None
    if valid_frame is not None and not valid_frame.empty:
        x_valid, _ = encode_frame(valid_frame, feature_columns, reference_columns=encoded_columns)
        y_valid = valid_frame["status"].astype(int)
    try:
        if x_valid is not None and y_valid is not None:
            model.fit(x_train, y_train, sample_weight=sample_weight, eval_set=[(x_valid, y_valid)], callbacks=_early_stop_callbacks)
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
            model.fit(x_train, y_train, sample_weight=sample_weight, eval_set=[(x_valid, y_valid)], callbacks=_early_stop_callbacks)
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
    random_seed: int = RANDOM_SEED,
    catboost_params: dict | None = None,
) -> ModelFitResult:
    """Fit CatBoost classifier.

    Parameters
    ----------
    catboost_params : Optional dict of CatBoost hyperparameters (from HPO).
        Supported keys: depth, learning_rate, l2_leaf_reg, random_strength,
        bagging_temperature, border_count, min_data_in_leaf, max_class_weight.
        Any key not provided falls back to its default value.
    """
    try:
        from catboost import CatBoostClassifier  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise ImportError("CatBoost is not installed. Install catboost to enable this path.") from exc

    hp = catboost_params or {}
    cat_features = [
        column for column in feature_columns
        if pd.api.types.is_object_dtype(train_frame[column])
        or pd.api.types.is_string_dtype(train_frame[column])
        or pd.api.types.is_categorical_dtype(train_frame[column])
    ]
    y_train = train_frame["status"].astype(int)
    _positives = max(1, int(y_train.sum()))
    _negatives = max(1, len(y_train) - _positives)
    # Cap class weight ratio. Uncapped 30x ratio causes Base B CatBoost to
    # compress all probabilities near zero. Default cap is 10x.
    _max_cw = hp.get("max_class_weight", 10.0)
    _weight_ratio = min(float(_negatives) / _positives, _max_cw)
    class_weights = [float(negative_weight), _weight_ratio]
    runtime_params = catboost_runtime_params()
    # Allow catboost_params to override task_type (e.g. force CPU for secondary OOF
    # to avoid GPU OOM when GPU is already saturated from primary training).
    if "task_type" in hp:
        runtime_params = dict(runtime_params)
        runtime_params["task_type"] = hp["task_type"]
        if hp["task_type"] == "CPU":
            runtime_params.pop("devices", None)
            runtime_params["thread_count"] = hardware_profile().cpu_threads
    _depth = hp.get("depth", 7)
    _lr = hp.get("learning_rate", 0.05)
    _l2 = hp.get("l2_leaf_reg", 3.0)
    _rs = hp.get("random_strength", 1.0)
    _bt = hp.get("bagging_temperature", 1.0)
    _bc = hp.get("border_count", 254)
    _mdl = hp.get("min_data_in_leaf", 1)
    # iterations/early_stopping_rounds: match HPO budget (HPO uses 1500/100).
    # Without this, HPO params tuned at 1500 iter underperform at default 1000.
    _iters = hp.get("iterations", 1500)
    _esr = hp.get("early_stopping_rounds", 100)
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        class_weights=class_weights,
        random_seed=random_seed,
        verbose=False,
        iterations=_iters,
        depth=_depth,
        learning_rate=_lr,
        l2_leaf_reg=_l2,
        random_strength=_rs,
        bagging_temperature=_bt,
        border_count=_bc,
        min_data_in_leaf=_mdl,
        **runtime_params,
    )
    validation_probabilities: list[float] | None = None
    if valid_frame is not None and not valid_frame.empty:
        y_valid = valid_frame["status"].astype(int)
        try:
            model.fit(train_frame[feature_columns], y_train, cat_features=cat_features,
                      eval_set=(valid_frame[feature_columns], y_valid),
                      use_best_model=True, early_stopping_rounds=_esr)
        except Exception:
            if runtime_params.get("task_type") != "GPU":
                raise
            model = CatBoostClassifier(
                loss_function="Logloss", eval_metric="Logloss",
                class_weights=class_weights,
                random_seed=random_seed, verbose=False,
                iterations=_iters,
                depth=_depth, learning_rate=_lr,
                l2_leaf_reg=_l2, random_strength=_rs,
                bagging_temperature=_bt, border_count=_bc,
                min_data_in_leaf=_mdl,
                task_type="CPU", thread_count=hardware_profile().cpu_threads,
            )
            model.fit(train_frame[feature_columns], y_train, cat_features=cat_features,
                      eval_set=(valid_frame[feature_columns], y_valid),
                      use_best_model=True, early_stopping_rounds=_esr)
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
                random_seed=random_seed, verbose=False,
                iterations=_iters,
                depth=_depth, learning_rate=_lr,
                l2_leaf_reg=_l2, random_strength=_rs,
                bagging_temperature=_bt, border_count=_bc,
                min_data_in_leaf=_mdl,
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
