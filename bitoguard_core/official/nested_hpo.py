"""GPU-optimized nested CV + HPO pipeline.

Architecture: 3-stage pipeline per outer fold with dual-stream GPU/CPU parallelism.
  Stage 1: Pre-cache default model predictions on inner folds (~8 min)
  Stage 2: GPU stream (CatBoost→XGBoost HPO) ∥ CPU stream (LightGBM HPO) (~14 min)
  Stage 3: Final multi-seed training + inner-fold selection (~8 min)

Target: ~2h total for 5 outer folds on RTX 5090 + 24-core CPU + 128GB RAM.

Usage:
    cd bitoguard_core && source .venv/bin/activate

    # Quick test (5 trials, verify pipeline works):
    PYTHONPATH=. python -m official.nested_hpo --outer-fold 0 --n-trials 5 --inner-folds 2

    # Single fold:
    PYTHONPATH=. python -m official.nested_hpo --outer-fold 0 --n-trials 25 --inner-folds 2

    # All folds (recommended):
    PYTHONPATH=. python -m official.nested_hpo --all --n-trials 25 --inner-folds 2

    # Aggregate only:
    PYTHONPATH=. python -m official.nested_hpo --aggregate
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

try:
    import optuna
except ImportError:
    raise ImportError("optuna>=3.0 required. pip install 'optuna>=3.0'")

from hardware import (
    catboost_runtime_params,
    hardware_profile,
    xgboost_runtime_params,
)
from official.common import (
    RANDOM_SEED,
    encode_frame,
    load_official_paths,
    save_json,
)
from official.correct_and_smooth import correct_and_smooth
from official.graph_dataset import TransductiveGraph, build_transductive_graph
from official.inner_fold_selection import select_and_apply_inner_fold
from official.modeling import ModelFitResult
from official.modeling_xgb import fit_xgboost
from official.stacking import (
    STACKER_FEATURE_COLUMNS,
    BlendEnsemble,
    _add_base_meta_features,
    tune_blend_weights,
)
from official.train import (
    _label_frame,
    _label_free_feature_columns,
    _load_dataset,
)
from official.transductive_features import build_transductive_feature_frame
from official.transductive_validation import (
    PrimarySplitSpec,
    build_primary_transductive_splits,
    iter_fold_assignments,
)

# ── Constants ─────────────────────────────────────────────────────────────────

N_OUTER_FOLDS = 5
N_INNER_FOLDS = 2
N_TRIALS = 25
HPO_SEED = 42

FINAL_SEEDS_A = [42, 52, 62, 72]
FINAL_SEEDS_D = [42, 123, 456]
FINAL_SEEDS_E = [42, 123]

GPU_STREAM_CPU_THREADS = 4
CPU_STREAM_THREADS = 20

_NESTED_HPO_DIR_NAME = "nested_hpo"


# ── Utilities ─────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[nested_hpo {ts}] {msg}", flush=True)


def _nested_hpo_dir() -> Path:
    d = load_official_paths().feature_dir / _NESTED_HPO_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fold_dir(outer_fold: int) -> Path:
    d = _nested_hpo_dir() / f"fold_{outer_fold}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Search Spaces ─────────────────────────────────────────────────────────────

def _sample_catboost_params(trial: optuna.Trial) -> dict:
    return {
        "iterations": trial.suggest_int("iterations", 800, 2500, step=100),
        "depth": trial.suggest_int("depth", 5, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "border_count": trial.suggest_categorical("border_count", [32, 64, 128, 254]),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 8.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        "max_class_weight": trial.suggest_float("max_class_weight", 5.0, 25.0),
    }


def _sample_xgboost_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 800, 2500, step=100),
        "max_depth": trial.suggest_int("max_depth", 5, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50.0, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 30.0),
    }


def _sample_lgbm_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),
    }


_CB_BASELINE = {
    "iterations": 1500, "depth": 7, "learning_rate": 0.05,
    "l2_leaf_reg": 3.0, "border_count": 254, "random_strength": 1.0,
    "bagging_temperature": 1.0, "min_data_in_leaf": 1, "max_class_weight": 10.0,
}
_XGB_BASELINE = {
    "n_estimators": 1500, "max_depth": 7, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1,
    "reg_lambda": 5.0, "min_child_weight": 5.0,
}
_LGBM_BASELINE = {
    "n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31,
    "subsample": 0.9, "colsample_bytree": 0.9, "min_child_samples": 20,
    "reg_alpha": 0.001, "reg_lambda": 0.001,
}


def _make_pruner() -> optuna.pruners.MedianPruner:
    return optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=0, interval_steps=1,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _best_blend_f1(oof_frame: pd.DataFrame) -> float:
    """Best F1 from blend ensemble over all base model probabilities."""
    frame = _add_base_meta_features(oof_frame.copy())
    weights = tune_blend_weights(frame)
    blend = BlendEnsemble(weights)
    avail = [c for c in STACKER_FEATURE_COLUMNS if c in frame.columns]
    probs = blend.predict_proba(frame[avail])[:, 1]
    labels = frame["status"].astype(int).to_numpy()
    best = 0.0
    for t in np.arange(0.05, 0.50, 0.01):
        f = float(f1_score(labels, (probs >= t).astype(int), zero_division=0))
        if f > best:
            best = f
    return best


def _extract_scores(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {}
    for col in ("rule_score", "anomaly_score", "crypto_anomaly_score", "anomaly_score_segmented"):
        if col in frame.columns:
            out[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0).to_numpy()
        else:
            out[col] = np.zeros(len(frame))
    return out


def _assemble_oof(valid_frame, base_a, cs, base_b, base_d, base_e, scores):
    oof = valid_frame[["user_id", "status"]].copy()
    oof["base_a_probability"] = base_a
    oof["base_c_s_probability"] = cs
    oof["base_b_probability"] = base_b
    oof["base_c_probability"] = 0.0
    oof["base_d_probability"] = base_d
    oof["base_e_probability"] = base_e
    oof["rule_score"] = scores["rule_score"]
    oof["anomaly_score"] = scores["anomaly_score"]
    oof["crypto_anomaly_score"] = scores["crypto_anomaly_score"]
    oof["anomaly_score_segmented"] = scores["anomaly_score_segmented"]
    return oof


def _fit_catboost_with_params(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame | None,
    feature_columns: list[str],
    params: dict[str, Any] | None,
    random_seed: int,
    *,
    force_cpu: bool = False,
    thread_count: int | None = None,
) -> ModelFitResult:
    from catboost import CatBoostClassifier

    hp = dict(params or {})
    max_class_weight = float(hp.pop("max_class_weight", 10.0))
    requested_task_type = str(hp.pop("task_type", "")).upper()

    runtime_params = dict(catboost_runtime_params())
    if force_cpu or requested_task_type == "CPU":
        runtime_params = {
            "task_type": "CPU",
            "thread_count": thread_count or (os.cpu_count() or 1),
        }
    else:
        if thread_count is not None:
            runtime_params["thread_count"] = thread_count

    cat_features = [
        column for column in feature_columns
        if pd.api.types.is_object_dtype(train_frame[column])
        or pd.api.types.is_string_dtype(train_frame[column])
        or pd.api.types.is_categorical_dtype(train_frame[column])
    ]
    y_train = train_frame["status"].astype(int)
    positives = max(1, int(y_train.sum()))
    negatives = max(1, len(y_train) - positives)

    model_kwargs = dict(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=int(hp.pop("iterations", 1500)),
        random_seed=random_seed,
        verbose=False,
        class_weights=[1.0, min(negatives / positives, max_class_weight)],
        early_stopping_rounds=int(hp.pop("early_stopping_rounds", 100)),
        **hp,
        **runtime_params,
    )

    train_x = train_frame[feature_columns]
    valid_x = None
    y_valid = None
    validation_probabilities: list[float] | None = None
    if valid_frame is not None and not valid_frame.empty:
        valid_x = valid_frame[feature_columns]
        y_valid = valid_frame["status"].astype(int)

    model = CatBoostClassifier(**model_kwargs)
    try:
        if valid_x is not None and y_valid is not None:
            model.fit(
                train_x,
                y_train,
                cat_features=cat_features,
                eval_set=(valid_x, y_valid),
                use_best_model=True,
            )
            validation_probabilities = model.predict_proba(valid_x)[:, 1].tolist()
        else:
            model.fit(train_x, y_train, cat_features=cat_features)
    except Exception:
        if model_kwargs.get("task_type") != "GPU":
            raise
        cpu_kwargs = {
            **{k: v for k, v in model_kwargs.items() if k not in ("devices", "gpu_ram_part", "boosting_type")},
            "task_type": "CPU",
            "thread_count": thread_count or (os.cpu_count() or 1),
        }
        model = CatBoostClassifier(**cpu_kwargs)
        if valid_x is not None and y_valid is not None:
            model.fit(
                train_x,
                y_train,
                cat_features=cat_features,
                eval_set=(valid_x, y_valid),
                use_best_model=True,
            )
            validation_probabilities = model.predict_proba(valid_x)[:, 1].tolist()
        else:
            model.fit(train_x, y_train, cat_features=cat_features)

    return ModelFitResult(
        model_name="catboost",
        model=model,
        feature_columns=feature_columns,
        encoded_columns=None,
        cat_features=cat_features,
        validation_probabilities=validation_probabilities,
    )


def _fit_lgbm_cpu(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame | None,
    feature_columns: list[str],
    params: dict[str, Any] | None,
    random_seed: int,
    *,
    n_jobs: int,
) -> ModelFitResult:
    from lightgbm import LGBMClassifier

    hp = dict(params or {})
    x_train, encoded_columns = encode_frame(train_frame, feature_columns)
    y_train = train_frame["status"].astype(int)
    positives = max(1, int(y_train.sum()))
    negatives = max(1, len(y_train) - positives)

    model = LGBMClassifier(
        n_estimators=int(hp.get("n_estimators", 400)),
        learning_rate=float(hp.get("learning_rate", 0.05)),
        num_leaves=int(hp.get("num_leaves", 31)),
        subsample=float(hp.get("subsample", 0.9)),
        colsample_bytree=float(hp.get("colsample_bytree", 0.9)),
        min_child_samples=int(hp.get("min_child_samples", 20)),
        reg_alpha=float(hp.get("reg_alpha", 0.001)),
        reg_lambda=float(hp.get("reg_lambda", 0.001)),
        scale_pos_weight=negatives / positives,
        random_state=random_seed,
        verbosity=-1,
        device_type="cpu",
        n_jobs=n_jobs,
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


def _predict_fit_result(fit_result: ModelFitResult, frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.zeros(0, dtype=float)
    if fit_result.encoded_columns is not None:
        x_frame, _ = encode_frame(
            frame,
            fit_result.feature_columns,
            reference_columns=fit_result.encoded_columns,
        )
        return fit_result.model.predict_proba(x_frame)[:, 1]
    return fit_result.model.predict_proba(frame[fit_result.feature_columns])[:, 1]


def _mean_validation_probabilities(fits: list[ModelFitResult]) -> np.ndarray:
    return np.mean(
        [np.asarray(fit.validation_probabilities, dtype=float) for fit in fits],
        axis=0,
    )


def _cs_from_base_a_fits(
    fits: list[ModelFitResult],
    train_f: pd.DataFrame,
    valid_f: pd.DataFrame,
    unlabeled_f: pd.DataFrame | None,
    graph: TransductiveGraph,
    train_labels_dict: dict[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    base_a_train = np.mean([_predict_fit_result(fit, train_f) for fit in fits], axis=0)
    base_a_valid = _mean_validation_probabilities(fits)
    cs_base = dict(zip(train_f["user_id"].astype(int).tolist(), base_a_train.tolist()))
    cs_base.update(zip(valid_f["user_id"].astype(int).tolist(), base_a_valid.tolist()))
    if unlabeled_f is not None and not unlabeled_f.empty:
        base_a_unlabeled = np.mean([_predict_fit_result(fit, unlabeled_f) for fit in fits], axis=0)
        cs_base.update(zip(unlabeled_f["user_id"].astype(int).tolist(), base_a_unlabeled.tolist()))
    cs_result = correct_and_smooth(
        graph,
        train_labels_dict,
        cs_base,
        alpha_correct=0.5,
        alpha_smooth=0.5,
        n_correct_iter=50,
        n_smooth_iter=50,
    )
    cs_valid = np.array(
        [cs_result.get(int(uid), float(p)) for uid, p in zip(valid_f["user_id"].astype(int), base_a_valid)]
    )
    return base_a_valid, cs_valid


def _cs_from_base_a(model, train_f, valid_f, feature_columns, graph, train_labels_dict):
    """Compute C&S from Base A model predictions."""
    base_a_train = model.predict_proba(train_f[feature_columns])[:, 1]
    base_a_valid = model.predict_proba(valid_f[feature_columns])[:, 1]
    cs_base = dict(zip(train_f["user_id"].astype(int).tolist(), base_a_train.tolist()))
    cs_base.update(zip(valid_f["user_id"].astype(int).tolist(), base_a_valid.tolist()))
    cs_result = correct_and_smooth(
        graph, train_labels_dict, cs_base,
        alpha_correct=0.5, alpha_smooth=0.5,
        n_correct_iter=50, n_smooth_iter=50,
    )
    cs_valid = np.array([
        cs_result.get(int(uid), float(p))
        for uid, p in zip(valid_f["user_id"].astype(int), base_a_valid)
    ])
    return base_a_valid, cs_valid


# ── Inner Fold Construction ───────────────────────────────────────────────────

def _build_inner_folds(
    outer_train_frame: pd.DataFrame,
    feature_columns: list[str],
    graph: TransductiveGraph,
    dataset: pd.DataFrame,
    label_frame: pd.DataFrame,
    n_inner: int = N_INNER_FOLDS,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    labeled = outer_train_frame[outer_train_frame["status"].notna()].copy()
    y = labeled["status"].astype(int)
    splitter = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)

    folds = []
    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(labeled, y)):
        train_f = labeled.iloc[train_idx].copy()
        valid_f = labeled.iloc[valid_idx].copy()

        cat_features = [c for c in feature_columns
                        if pd.api.types.is_object_dtype(train_f[c])
                        or pd.api.types.is_string_dtype(train_f[c])]

        train_labels_dict = dict(zip(
            train_f["user_id"].astype(int).tolist(),
            train_f["status"].astype(float).tolist(),
        ))

        x_train_enc, enc_cols = encode_frame(train_f, feature_columns)
        x_valid_enc, _ = encode_frame(valid_f, feature_columns, reference_columns=enc_cols)

        inner_train_labels = label_frame[
            label_frame["user_id"].astype(int).isin(set(train_f["user_id"].astype(int)))
        ].copy()
        trans = build_transductive_feature_frame(graph, inner_train_labels)
        trans_cols = [c for c in trans.columns if c != "user_id"]

        relevant_ids = set(train_f["user_id"].astype(int)) | set(valid_f["user_id"].astype(int))
        relevant = dataset[dataset["user_id"].astype(int).isin(relevant_ids)].copy()
        with_td = relevant.merge(trans, on="user_id", how="left")
        with_td[trans_cols] = with_td[trans_cols].fillna(0.0)

        train_td = with_td[with_td["user_id"].astype(int).isin(set(train_f["user_id"].astype(int)))].copy()
        valid_td = with_td[with_td["user_id"].astype(int).isin(set(valid_f["user_id"].astype(int)))].copy()

        folds.append({
            "fold_id": fold_idx,
            "train_frame": train_f, "valid_frame": valid_f,
            "cat_features": cat_features,
            "train_labels_dict": train_labels_dict,
            "x_train_encoded": x_train_enc, "x_valid_encoded": x_valid_enc,
            "encoded_columns": enc_cols,
            "train_transductive": train_td, "valid_transductive": valid_td,
            "base_b_columns": feature_columns + trans_cols,
            "scores": _extract_scores(valid_f),
        })

    return folds


# ── Stage 1: Pre-cache ───────────────────────────────────────────────────────

def _build_precache(
    inner_folds_data: list[dict],
    graph: TransductiveGraph,
    feature_columns: list[str],
) -> tuple[list[dict], list[dict]]:
    """Train default models per inner fold, cache predictions for HPO.

    cb_cache[i]: fixed {base_b, base_d, base_e, scores} — for CatBoost HPO
    de_cache[i]: fixed {base_a, cs, base_b, base_d, base_e, scores} — for LGB/XGB HPO
    """
    cb_cache, de_cache = [], []

    for fd in inner_folds_data:
        _log(f"  Pre-cache inner fold {fd['fold_id']}...")
        t0 = time.time()
        train_f, valid_f = fd["train_frame"], fd["valid_frame"]
        y_tr, y_va = train_f["status"].astype(int), valid_f["status"].astype(int)
        pos, neg = max(1, int(y_tr.sum())), max(1, len(y_tr) - int(y_tr.sum()))

        base_a_fit = _fit_catboost_with_params(
            train_f,
            valid_f,
            feature_columns,
            _CB_BASELINE.copy(),
            HPO_SEED,
            thread_count=GPU_STREAM_CPU_THREADS,
        )
        base_a_valid, cs_valid = _cs_from_base_a_fits(
            [base_a_fit],
            train_f,
            valid_f,
            None,
            graph,
            fd["train_labels_dict"],
        )

        # Base B: Transductive CatBoost (CPU)
        bb_fit = _fit_catboost_with_params(
            fd["train_transductive"], fd["valid_transductive"], fd["base_b_columns"],
            {"task_type": "CPU", "l2_leaf_reg": 5.0},
            HPO_SEED,
            force_cpu=True,
            thread_count=CPU_STREAM_THREADS,
        )
        base_b_valid = np.asarray(bb_fit.validation_probabilities, dtype=float)

        # Base D: LightGBM default (CPU)
        base_d_fit = _fit_lgbm_cpu(
            train_f,
            valid_f,
            feature_columns,
            _LGBM_BASELINE.copy(),
            HPO_SEED,
            n_jobs=CPU_STREAM_THREADS,
        )
        base_d_valid = np.asarray(base_d_fit.validation_probabilities, dtype=float)

        # Base E: XGBoost default (GPU)
        base_e_fit = fit_xgboost(
            train_f,
            valid_f,
            feature_columns,
            params=_XGB_BASELINE.copy(),
            random_seed=HPO_SEED,
        )
        base_e_valid = np.asarray(base_e_fit.validation_probabilities, dtype=float)

        scores = fd["scores"]
        cb_cache.append({
            "base_b_probs": base_b_valid, "base_d_probs": base_d_valid,
            "base_e_probs": base_e_valid, **scores,
        })
        de_cache.append({
            "base_a_probs": base_a_valid, "cs_probs": cs_valid,
            "base_b_probs": base_b_valid, "base_d_probs": base_d_valid,
            "base_e_probs": base_e_valid, **scores,
        })
        _log(f"  Pre-cache fold {fd['fold_id']} done ({time.time() - t0:.0f}s)")

    return cb_cache, de_cache


# ── HPO Objectives ────────────────────────────────────────────────────────────

def _catboost_objective(
    trial: optuna.Trial,
    inner_folds_data: list[dict],
    cb_cache: list[dict],
    graph: TransductiveGraph,
    feature_columns: list[str],
    runtime_cb: dict,
) -> float:
    """Train Base A with trial params → recompute C&S → blend F1."""
    from catboost import CatBoostClassifier

    params = _sample_catboost_params(trial)
    max_cw = params.pop("max_class_weight")
    fold_f1s = []

    for i, fd in enumerate(inner_folds_data):
        train_f, valid_f = fd["train_frame"], fd["valid_frame"]
        y_tr = train_f["status"].astype(int)
        y_va = valid_f["status"].astype(int)
        pos, neg = max(1, int(y_tr.sum())), max(1, len(y_tr) - int(y_tr.sum()))
        cw = min(neg / pos, max_cw)

        model = CatBoostClassifier(
            loss_function="Logloss", eval_metric="Logloss",
            class_weights=[1.0, cw], random_seed=HPO_SEED, verbose=False,
            early_stopping_rounds=100, **params, **runtime_cb,
        )
        try:
            model.fit(train_f[feature_columns], y_tr, cat_features=fd["cat_features"],
                      eval_set=(valid_f[feature_columns], y_va), use_best_model=True)
        except Exception:
            if runtime_cb.get("task_type") != "GPU":
                return 0.0
            cpu_rt = {"task_type": "CPU", "thread_count": GPU_STREAM_CPU_THREADS}
            model = CatBoostClassifier(
                loss_function="Logloss", eval_metric="Logloss",
                class_weights=[1.0, cw], random_seed=HPO_SEED, verbose=False,
                early_stopping_rounds=100, **params, **cpu_rt,
            )
            try:
                model.fit(train_f[feature_columns], y_tr, cat_features=fd["cat_features"],
                          eval_set=(valid_f[feature_columns], y_va), use_best_model=True)
            except Exception:
                return 0.0

        base_a_valid, cs_valid = _cs_from_base_a(
            model, train_f, valid_f, feature_columns, graph, fd["train_labels_dict"]
        )
        oof = _assemble_oof(
            valid_f, base_a_valid, cs_valid,
            cb_cache[i]["base_b_probs"], cb_cache[i]["base_d_probs"],
            cb_cache[i]["base_e_probs"], cb_cache[i],
        )
        f1 = _best_blend_f1(oof)
        fold_f1s.append(f1)
        trial.report(f1, step=i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_f1s))


def _xgboost_objective(
    trial: optuna.Trial,
    inner_folds_data: list[dict],
    de_cache: list[dict],
    runtime_xgb: dict,
) -> float:
    """Train Base E with trial params → blend F1 (no C&S recompute)."""
    from xgboost import XGBClassifier

    params = _sample_xgboost_params(trial)
    fold_f1s = []

    for i, fd in enumerate(inner_folds_data):
        y_tr = fd["train_frame"]["status"].astype(int)
        y_va = fd["valid_frame"]["status"].astype(int)
        pos, neg = max(1, int(y_tr.sum())), max(1, len(y_tr) - int(y_tr.sum()))

        model = XGBClassifier(
            **params, scale_pos_weight=min(neg / pos, 15.0),
            objective="binary:logistic", eval_metric="logloss",
            random_state=HPO_SEED, verbosity=0, early_stopping_rounds=100,
            **runtime_xgb,
        )
        try:
            model.fit(fd["x_train_encoded"], y_tr,
                      eval_set=[(fd["x_valid_encoded"], y_va)], verbose=False)
        except Exception:
            if runtime_xgb.get("device") != "cuda":
                return 0.0
            model = XGBClassifier(
                **params, scale_pos_weight=min(neg / pos, 15.0),
                objective="binary:logistic", eval_metric="logloss",
                random_state=HPO_SEED, verbosity=0, early_stopping_rounds=100,
                tree_method="hist", device="cpu",
            )
            try:
                model.fit(fd["x_train_encoded"], y_tr,
                          eval_set=[(fd["x_valid_encoded"], y_va)], verbose=False)
            except Exception:
                return 0.0

        base_e_valid = model.predict_proba(fd["x_valid_encoded"])[:, 1]
        oof = _assemble_oof(
            fd["valid_frame"], de_cache[i]["base_a_probs"], de_cache[i]["cs_probs"],
            de_cache[i]["base_b_probs"], de_cache[i]["base_d_probs"],
            base_e_valid, de_cache[i],
        )
        f1 = _best_blend_f1(oof)
        fold_f1s.append(f1)
        trial.report(f1, step=i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_f1s))


def _lgbm_objective(
    trial: optuna.Trial,
    inner_folds_data: list[dict],
    de_cache: list[dict],
) -> float:
    """Train Base D with trial params on CPU → blend F1."""
    from lightgbm import LGBMClassifier

    params = _sample_lgbm_params(trial)
    fold_f1s = []

    for i, fd in enumerate(inner_folds_data):
        y_tr = fd["train_frame"]["status"].astype(int)
        y_va = fd["valid_frame"]["status"].astype(int)
        pos, neg = max(1, int(y_tr.sum())), max(1, len(y_tr) - int(y_tr.sum()))

        model = LGBMClassifier(
            **params, scale_pos_weight=neg / pos,
            random_state=HPO_SEED, verbosity=-1,
            n_jobs=CPU_STREAM_THREADS,
        )
        try:
            model.fit(fd["x_train_encoded"], y_tr,
                      eval_set=[(fd["x_valid_encoded"], y_va)], eval_metric="binary_logloss")
        except Exception:
            return 0.0

        base_d_valid = model.predict_proba(fd["x_valid_encoded"])[:, 1]
        oof = _assemble_oof(
            fd["valid_frame"], de_cache[i]["base_a_probs"], de_cache[i]["cs_probs"],
            de_cache[i]["base_b_probs"], base_d_valid,
            de_cache[i]["base_e_probs"], de_cache[i],
        )
        f1 = _best_blend_f1(oof)
        fold_f1s.append(f1)
        trial.report(f1, step=i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_f1s))


# ── Stage 2: Dual-Stream HPO ─────────────────────────────────────────────────

def _gpu_stream(cb_cache, de_cache, inner_folds_data, graph, feature_columns, fold_output, n_trials):
    """GPU: CatBoost HPO → XGBoost HPO (sequential, sharing GPU)."""
    runtime_cb = catboost_runtime_params()
    runtime_xgb = xgboost_runtime_params()

    # CatBoost
    _log("  [GPU] CatBoost HPO starting...")
    t0 = time.time()
    cb_study = optuna.create_study(
        direction="maximize", pruner=_make_pruner(),
        sampler=optuna.samplers.TPESampler(seed=HPO_SEED, n_startup_trials=5),
    )
    cb_study.enqueue_trial(_CB_BASELINE)
    cb_study.optimize(
        lambda trial: _catboost_objective(
            trial, inner_folds_data, cb_cache, graph, feature_columns, runtime_cb,
        ),
        n_trials=n_trials,
    )
    cb_el = time.time() - t0
    cb_pr = sum(1 for t in cb_study.trials if t.state == optuna.trial.TrialState.PRUNED)
    _log(f"  [GPU] CatBoost HPO done: F1={cb_study.best_value:.4f} "
         f"({len(cb_study.trials)} trials, {cb_pr} pruned, {cb_el:.0f}s)")
    save_json({"best_params": dict(cb_study.best_params), "best_f1": float(cb_study.best_value),
               "n_trials": len(cb_study.trials), "n_pruned": cb_pr, "elapsed_s": round(cb_el, 1),
               "all_trials": [{"n": t.number, "v": t.value, "p": t.params}
                              for t in cb_study.trials if t.value is not None]},
              fold_output / "catboost_study.json")

    # XGBoost
    _log("  [GPU] XGBoost HPO starting...")
    t0 = time.time()
    xgb_study = optuna.create_study(
        direction="maximize", pruner=_make_pruner(),
        sampler=optuna.samplers.TPESampler(seed=HPO_SEED, n_startup_trials=5),
    )
    xgb_study.enqueue_trial(_XGB_BASELINE)
    xgb_study.optimize(
        lambda trial: _xgboost_objective(trial, inner_folds_data, de_cache, runtime_xgb),
        n_trials=n_trials,
    )
    xgb_el = time.time() - t0
    xgb_pr = sum(1 for t in xgb_study.trials if t.state == optuna.trial.TrialState.PRUNED)
    _log(f"  [GPU] XGBoost HPO done: F1={xgb_study.best_value:.4f} "
         f"({len(xgb_study.trials)} trials, {xgb_pr} pruned, {xgb_el:.0f}s)")
    save_json({"best_params": dict(xgb_study.best_params), "best_f1": float(xgb_study.best_value),
               "n_trials": len(xgb_study.trials), "n_pruned": xgb_pr, "elapsed_s": round(xgb_el, 1),
               "all_trials": [{"n": t.number, "v": t.value, "p": t.params}
                              for t in xgb_study.trials if t.value is not None]},
              fold_output / "xgboost_study.json")

    return dict(cb_study.best_params), dict(xgb_study.best_params)


def _cpu_stream(de_cache, inner_folds_data, fold_output, n_trials):
    """CPU: LightGBM HPO (concurrent with GPU stream)."""
    _log("  [CPU] LightGBM HPO starting...")
    t0 = time.time()
    lgbm_study = optuna.create_study(
        direction="maximize", pruner=_make_pruner(),
        sampler=optuna.samplers.TPESampler(seed=HPO_SEED, n_startup_trials=5),
    )
    lgbm_study.enqueue_trial(_LGBM_BASELINE)
    lgbm_study.optimize(
        lambda trial: _lgbm_objective(trial, inner_folds_data, de_cache),
        n_trials=n_trials,
    )
    lgbm_el = time.time() - t0
    lgbm_pr = sum(1 for t in lgbm_study.trials if t.state == optuna.trial.TrialState.PRUNED)
    _log(f"  [CPU] LightGBM HPO done: F1={lgbm_study.best_value:.4f} "
         f"({len(lgbm_study.trials)} trials, {lgbm_pr} pruned, {lgbm_el:.0f}s)")
    save_json({"best_params": dict(lgbm_study.best_params), "best_f1": float(lgbm_study.best_value),
               "n_trials": len(lgbm_study.trials), "n_pruned": lgbm_pr, "elapsed_s": round(lgbm_el, 1),
               "all_trials": [{"n": t.number, "v": t.value, "p": t.params}
                              for t in lgbm_study.trials if t.value is not None]},
              fold_output / "lgbm_study.json")
    return dict(lgbm_study.best_params)


def _run_stage2(cb_cache, de_cache, inner_folds_data, graph, feature_columns, fold_output, n_trials):
    _log("Stage 2: Dual-stream HPO starting...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        gpu_f = pool.submit(_gpu_stream, cb_cache, de_cache, inner_folds_data,
                            graph, feature_columns, fold_output, n_trials)
        cpu_f = pool.submit(_cpu_stream, de_cache, inner_folds_data, fold_output, n_trials)
        best_cb, best_xgb = gpu_f.result()
        best_lgbm = cpu_f.result()
    _log(f"Stage 2 done ({time.time() - t0:.0f}s)")
    return best_cb, best_lgbm, best_xgb


# ── Stage 3: Final Training + Evaluation ─────────────────────────────────────

def _run_stage3(
    outer_fold: int,
    dataset: pd.DataFrame,
    graph: TransductiveGraph,
    label_frame: pd.DataFrame,
    feature_columns: list[str],
    outer_train_users: set[int],
    outer_valid_users: set[int],
    best_cb: dict, best_lgbm: dict, best_xgb: dict,
    seed: int, fold_output: Path,
) -> dict[str, Any]:
    _log("Stage 3: Final training starting...")
    t0 = time.time()

    cb_hpo = {**_CB_BASELINE, **best_cb}
    cb_hpo["early_stopping_rounds"] = 100

    # ── 3a: Inner 5-fold OOF on outer_train (1 seed per model) ──
    _log("  3a: Inner 5-fold OOF on outer_train...")
    outer_train_frame = dataset[dataset["user_id"].astype(int).isin(outer_train_users)].copy()
    outer_train_labels = label_frame[label_frame["user_id"].astype(int).isin(outer_train_users)].copy()

    oof_split = build_primary_transductive_splits(
        outer_train_frame[outer_train_frame["status"].notna()].copy(),
        cutoff_tag="full", spec=PrimarySplitSpec(n_splits=5, random_state=seed),
        write_outputs=False,
    )
    oof_rows: list[pd.DataFrame] = []

    for fold_id, inner_tr_users, inner_va_users in iter_fold_assignments(oof_split, "primary_fold"):
        inner_tr_labels = label_frame[label_frame["user_id"].astype(int).isin(inner_tr_users)].copy()
        trans = build_transductive_feature_frame(graph, inner_tr_labels)
        tc = [c for c in trans.columns if c != "user_id"]

        inner_ids = inner_tr_users | inner_va_users
        rel = dataset[dataset["user_id"].astype(int).isin(inner_ids)].copy()
        wtd = rel.merge(trans, on="user_id", how="left")
        wtd[tc] = wtd[tc].fillna(0.0)

        tr_lf = dataset[dataset["user_id"].astype(int).isin(inner_tr_users)].copy()
        va_lf = dataset[dataset["user_id"].astype(int).isin(inner_va_users)].copy()
        tr_td = wtd[wtd["user_id"].astype(int).isin(inner_tr_users)].copy()
        va_td = wtd[wtd["user_id"].astype(int).isin(inner_va_users)].copy()

        inner_other = dataset[~dataset["user_id"].astype(int).isin(inner_ids)].copy()

        # Base A × multi-seed
        ba_fits = [
            _fit_catboost_with_params(tr_lf, va_lf, feature_columns, cb_hpo.copy(), s)
            for s in FINAL_SEEDS_A
        ]
        ba_v, cs_v = _cs_from_base_a_fits(
            ba_fits,
            tr_lf,
            va_lf,
            inner_other,
            graph,
            dict(
                zip(
                    inner_tr_labels["user_id"].astype(int).tolist(),
                    inner_tr_labels["status"].astype(float).tolist(),
                )
            ),
        )

        # Base B
        bb = _fit_catboost_with_params(
            tr_td,
            va_td,
            feature_columns + tc,
            {"task_type": "CPU", "l2_leaf_reg": 5.0},
            HPO_SEED,
            force_cpu=True,
            thread_count=CPU_STREAM_THREADS,
        )

        # Base D × multi-seed (CPU only)
        bd_fits = [
            _fit_lgbm_cpu(tr_lf, va_lf, feature_columns, best_lgbm.copy(), s, n_jobs=CPU_STREAM_THREADS)
            for s in FINAL_SEEDS_D
        ]
        bd_v = _mean_validation_probabilities(bd_fits)

        # Base E × multi-seed
        be_fits = [
            fit_xgboost(tr_lf, va_lf, feature_columns, params=best_xgb, random_seed=s)
            for s in FINAL_SEEDS_E
        ]
        be_v = _mean_validation_probabilities(be_fits)

        ff = va_lf[["user_id", "status"]].copy()
        ff["primary_fold"] = fold_id
        ff["base_a_probability"] = ba_v
        ff["base_c_s_probability"] = cs_v
        ff["base_b_probability"] = np.asarray(bb.validation_probabilities, dtype=float)
        ff["base_c_probability"] = 0.0
        ff["base_d_probability"] = bd_v
        ff["base_e_probability"] = be_v
        for col in ("rule_score", "anomaly_score", "crypto_anomaly_score", "anomaly_score_segmented"):
            ff[col] = pd.to_numeric(va_lf[col], errors="coerce").fillna(0.0).to_numpy() if col in va_lf.columns else 0.0
        oof_rows.append(ff)

    train_oof = pd.concat(oof_rows, ignore_index=True).sort_values("user_id").reset_index(drop=True)
    _log(f"  3a done: inner OOF {len(train_oof)} rows ({time.time() - t0:.0f}s)")

    # ── 3b: Final models on full outer_train, predict outer_valid ──
    _log("  3b: Final multi-seed training...")
    t1 = time.time()

    train_lf = dataset[dataset["user_id"].astype(int).isin(outer_train_users)].copy()
    valid_lf = dataset[dataset["user_id"].astype(int).isin(outer_valid_users)].copy()

    final_trans = build_transductive_feature_frame(graph, outer_train_labels)
    ftc = [c for c in final_trans.columns if c != "user_id"]
    merged = dataset.merge(final_trans, on="user_id", how="left")
    merged[ftc] = merged[ftc].fillna(0.0)
    train_td = merged[merged["user_id"].astype(int).isin(outer_train_users)].copy()
    valid_td = merged[merged["user_id"].astype(int).isin(outer_valid_users)].copy()
    unlabeled = dataset[~dataset["user_id"].astype(int).isin(outer_train_users | outer_valid_users)].copy()

    # Base A × multi-seed
    fa_fits = [
        _fit_catboost_with_params(train_lf, valid_lf, feature_columns, cb_hpo.copy(), s)
        for s in FINAL_SEEDS_A
    ]
    cs_labels = dict(
        zip(
            outer_train_labels["user_id"].astype(int).tolist(),
            outer_train_labels["status"].astype(float).tolist(),
        )
    )
    ov_a, ov_cs = _cs_from_base_a_fits(
        fa_fits,
        train_lf,
        valid_lf,
        unlabeled,
        graph,
        cs_labels,
    )

    # Base B
    fb = _fit_catboost_with_params(
        train_td,
        valid_td,
        feature_columns + ftc,
        {"task_type": "CPU", "l2_leaf_reg": 5.0},
        HPO_SEED,
        force_cpu=True,
        thread_count=CPU_STREAM_THREADS,
    )

    # Base D × multi-seed (CPU only)
    fd_fits = [
        _fit_lgbm_cpu(train_lf, valid_lf, feature_columns, best_lgbm.copy(), s, n_jobs=CPU_STREAM_THREADS)
        for s in FINAL_SEEDS_D
    ]
    ov_d = _mean_validation_probabilities(fd_fits)

    # Base E × multi-seed
    fe_fits = [
        fit_xgboost(train_lf, valid_lf, feature_columns, params=best_xgb, random_seed=s)
        for s in FINAL_SEEDS_E
    ]
    ov_e = _mean_validation_probabilities(fe_fits)
    _log(f"  3b done ({time.time() - t1:.0f}s)")

    # ── 3c: Assemble outer_valid + inner-fold selection ──
    ov_pred = valid_lf[["user_id", "status"]].copy()
    ov_pred["outer_fold"] = outer_fold
    ov_pred["primary_fold"] = outer_fold
    ov_pred["base_a_probability"] = ov_a
    ov_pred["base_c_s_probability"] = ov_cs
    ov_pred["base_b_probability"] = np.asarray(fb.validation_probabilities, dtype=float)
    ov_pred["base_c_probability"] = 0.0
    ov_pred["base_d_probability"] = ov_d
    ov_pred["base_e_probability"] = ov_e
    for col in ("rule_score", "anomaly_score", "crypto_anomaly_score", "anomaly_score_segmented"):
        ov_pred[col] = pd.to_numeric(valid_lf[col], errors="coerce").fillna(0.0).to_numpy() if col in valid_lf.columns else 0.0

    stacker_cols = [c for c in STACKER_FEATURE_COLUMNS if c in train_oof.columns]
    selected, sel_meta = select_and_apply_inner_fold(
        train_oof, ov_pred, fold_column="primary_fold", stacker_feature_columns=stacker_cols,
    )

    labels = selected["status"].astype(int).to_numpy()
    probs = selected["submission_probability"].to_numpy()
    best_thr = float(sel_meta.get("selected_threshold", 0.10))
    fp = (probs >= best_thr).astype(int)
    best_f1 = float(f1_score(labels, fp, zero_division=0))
    selected["submission_pred"] = fp
    metrics = {
        "outer_fold": outer_fold,
        "n_valid": int(len(labels)),
        "n_positive": int(labels.sum()),
        "best_f1": float(best_f1),
        "best_threshold": float(best_thr),
        "precision": float(precision_score(labels, fp, zero_division=0)),
        "recall": float(recall_score(labels, fp, zero_division=0)),
        "auc_roc": float(roc_auc_score(labels, probs)) if labels.sum() > 0 else 0.0,
        "average_precision": float(average_precision_score(labels, probs)) if labels.sum() > 0 else 0.0,
        "selection_threshold": float(sel_meta.get("selected_threshold", 0)),
        "elapsed_s": round(time.time() - t0, 1),
    }
    selected.to_parquet(fold_output / "predictions.parquet", index=False)
    save_json(metrics, fold_output / "metrics.json")

    _log(f"Stage 3 done: F1={best_f1:.4f} P={metrics['precision']:.4f} "
         f"R={metrics['recall']:.4f} AP={metrics['average_precision']:.4f} ({time.time() - t0:.0f}s)")
    return metrics


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_outer_fold(
    outer_fold: int,
    dataset: pd.DataFrame,
    graph: TransductiveGraph,
    label_frame: pd.DataFrame,
    feature_columns: list[str],
    outer_split: pd.DataFrame,
    n_trials: int = N_TRIALS,
    n_inner_folds: int = N_INNER_FOLDS,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    t_start = time.time()
    fold_output = _fold_dir(outer_fold)
    if n_inner_folds != N_INNER_FOLDS:
        _log(f"Requested inner_folds={n_inner_folds}; using {N_INNER_FOLDS} for this runner")
        n_inner_folds = N_INNER_FOLDS
    _log(f"{'=' * 60}")
    _log(f"Outer fold {outer_fold} starting")
    _log(f"{'=' * 60}")

    assignments = iter_fold_assignments(outer_split, "primary_fold")
    target = None
    for fid, tr_u, va_u in assignments:
        if fid == outer_fold:
            target = (tr_u, va_u)
            break
    if target is None:
        raise ValueError(f"Outer fold {outer_fold} not found")
    outer_train_users, outer_valid_users = target
    outer_train_frame = dataset[dataset["user_id"].astype(int).isin(outer_train_users)].copy()
    _log(f"outer_train={len(outer_train_users)}, outer_valid={len(outer_valid_users)}")

    # Stage 1
    t1 = time.time()
    _log("Stage 1: Inner folds + pre-cache...")
    inner_folds = _build_inner_folds(
        outer_train_frame, feature_columns, graph, dataset, label_frame,
        n_inner=n_inner_folds, seed=seed,
    )
    cb_cache, de_cache = _build_precache(inner_folds, graph, feature_columns)
    s1 = time.time() - t1
    _log(f"Stage 1 done ({s1:.0f}s)")

    # Stage 2
    t2 = time.time()
    best_cb, best_lgbm, best_xgb = _run_stage2(
        cb_cache, de_cache, inner_folds, graph, feature_columns, fold_output, n_trials,
    )
    s2 = time.time() - t2
    save_json({"catboost": best_cb, "lightgbm": best_lgbm, "xgboost": best_xgb},
              fold_output / "best_params.json")

    # Stage 3
    t3 = time.time()
    metrics = _run_stage3(
        outer_fold, dataset, graph, label_frame, feature_columns,
        outer_train_users, outer_valid_users,
        best_cb, best_lgbm, best_xgb, seed, fold_output,
    )
    s3 = time.time() - t3

    timing = {"stage1_s": round(s1, 1), "stage2_s": round(s2, 1),
              "stage3_s": round(s3, 1), "total_s": round(time.time() - t_start, 1)}
    save_json(timing, fold_output / "timing.json")
    metrics["timing"] = timing
    _log(f"Outer fold {outer_fold} done: F1={metrics['best_f1']:.4f}, total={timing['total_s']:.0f}s")
    return metrics


# ── Aggregate ─────────────────────────────────────────────────────────────────

def aggregate_nested_results() -> dict[str, Any]:
    root = _nested_hpo_dir()
    fold_dirs = sorted(root.glob("fold_*"))
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories in {root}")

    all_preds, all_metrics = [], []
    for fd in fold_dirs:
        pp, mp = fd / "predictions.parquet", fd / "metrics.json"
        if not pp.exists() or not mp.exists():
            _log(f"  Skipping {fd.name}: missing files")
            continue
        all_preds.append(pd.read_parquet(pp))
        with open(mp, "r") as f:
            all_metrics.append(json.load(f))

    if not all_preds:
        raise FileNotFoundError("No valid fold results found")
    if len(all_metrics) < N_OUTER_FOLDS:
        _log(f"Warning: aggregating {len(all_metrics)} folds, expected {N_OUTER_FOLDS}")

    pooled = pd.concat(all_preds, ignore_index=True).sort_values("user_id").reset_index(drop=True)
    dup = pooled["user_id"].duplicated().sum()
    if dup > 0:
        _log(f"  Warning: {dup} duplicate user_ids")
        pooled = pooled.drop_duplicates(subset=["user_id"], keep="last")

    labels = pooled["status"].astype(int).to_numpy()
    probs = pooled["submission_probability"].to_numpy()

    # Median of per-fold thresholds (unbiased)
    thresholds = [m.get("best_threshold", 0.10) for m in all_metrics]
    pt = float(np.median(thresholds))
    fp = (probs >= pt).astype(int)

    pf_f1 = [m.get("best_f1", 0.0) for m in all_metrics]
    pf_auc = [m.get("auc_roc", 0.0) for m in all_metrics]
    pf_ap = [m.get("average_precision", 0.0) for m in all_metrics]

    result = {
        "n_folds": len(all_metrics),
        "n_total": int(len(labels)), "n_positive": int(labels.sum()),
        "pooled_f1": float(f1_score(labels, fp, zero_division=0)),
        "pooled_threshold": pt,
        "pooled_precision": float(precision_score(labels, fp, zero_division=0)),
        "pooled_recall": float(recall_score(labels, fp, zero_division=0)),
        "pooled_auc_roc": float(roc_auc_score(labels, probs)) if labels.sum() > 0 else 0.0,
        "pooled_average_precision": float(average_precision_score(labels, probs)) if labels.sum() > 0 else 0.0,
        "per_fold_f1": pf_f1,
        "mean_fold_f1": float(np.mean(pf_f1)), "std_fold_f1": float(np.std(pf_f1)),
        "per_fold_auc_roc": pf_auc, "mean_fold_auc_roc": float(np.mean(pf_auc)),
        "per_fold_ap": pf_ap, "mean_fold_ap": float(np.mean(pf_ap)),
        "per_fold_thresholds": thresholds,
        "per_fold_details": all_metrics,
    }
    pooled.to_parquet(root / "nested_oof_predictions.parquet", index=False)
    save_json(result, root / "nested_oof_metrics.json")

    _log(f"\n{'=' * 60}\n"
         f"  Nested CV Summary ({result['n_folds']} folds)\n"
         f"  Pooled F1: {result['pooled_f1']:.4f} (threshold={pt:.3f})\n"
         f"  Mean fold F1: {result['mean_fold_f1']:.4f} ± {result['std_fold_f1']:.4f}\n"
         f"  Pooled P={result['pooled_precision']:.4f} R={result['pooled_recall']:.4f}\n"
         f"  Pooled AUC-ROC={result['pooled_auc_roc']:.4f} AP={result['pooled_average_precision']:.4f}\n"
         f"{'=' * 60}")
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="GPU-optimized nested CV + HPO")
    parser.add_argument("--outer-fold", type=int, default=None, help="Run single outer fold (0-4)")
    parser.add_argument("--all", action="store_true", help="Run all 5 outer folds")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate fold results")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS, help=f"HPO trials per model (default: {N_TRIALS})")
    parser.add_argument("--inner-folds", type=int, default=N_INNER_FOLDS, help=f"Inner HPO folds (default: {N_INNER_FOLDS})")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help=f"Random seed (default: {RANDOM_SEED})")
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if args.aggregate:
        result = aggregate_nested_results()
        print(f"\nPooled F1: {result['pooled_f1']:.4f}")
        print(f"Mean fold F1: {result['mean_fold_f1']:.4f} ± {result['std_fold_f1']:.4f}")
        return

    if not args.all and args.outer_fold is None:
        parser.print_help()
        return

    _log("Loading dataset and building graph...")
    t0 = time.time()
    dataset = _load_dataset("full")
    graph = build_transductive_graph(dataset)
    label_frame = _label_frame(dataset)
    feature_columns = _label_free_feature_columns(dataset)
    outer_split = build_primary_transductive_splits(
        dataset, cutoff_tag="full", spec=PrimarySplitSpec(), write_outputs=False,
    )
    _log(f"Loaded: {len(dataset)} users, {len(feature_columns)} features ({time.time() - t0:.0f}s)")
    _log(f"Hardware: {hardware_profile()}")
    resolved_inner_folds = N_INNER_FOLDS
    if args.inner_folds != N_INNER_FOLDS:
        _log(f"Requested --inner-folds {args.inner_folds}; using {N_INNER_FOLDS}")

    if args.all:
        for fold_k in range(N_OUTER_FOLDS):
            try:
                run_outer_fold(
                    fold_k, dataset, graph, label_frame, feature_columns, outer_split,
                    n_trials=args.n_trials, n_inner_folds=resolved_inner_folds, seed=args.seed,
                )
            except Exception as e:
                _log(f"Outer fold {fold_k} FAILED: {e}")
                import traceback
                save_json(
                    {"outer_fold": fold_k, "error": str(e), "traceback": traceback.format_exc()},
                    _fold_dir(fold_k) / "error.json",
                )
                traceback.print_exc()
        _log("All folds done, aggregating...")
        agg = aggregate_nested_results()
        print(f"\n=== Final ===")
        print(f"Pooled F1: {agg['pooled_f1']:.4f}")
        print(f"Mean fold F1: {agg['mean_fold_f1']:.4f} ± {agg['std_fold_f1']:.4f}")
    else:
        result = run_outer_fold(
            args.outer_fold, dataset, graph, label_frame, feature_columns, outer_split,
            n_trials=args.n_trials, n_inner_folds=resolved_inner_folds, seed=args.seed,
        )
        print(f"\nOuter fold {args.outer_fold}: F1={result['best_f1']:.4f} "
              f"P={result['precision']:.4f} R={result['recall']:.4f}")


if __name__ == "__main__":
    main()
