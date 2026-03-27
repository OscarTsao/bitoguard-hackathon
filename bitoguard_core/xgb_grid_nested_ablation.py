from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import official.nested_hpo as nh
from official.correct_and_smooth import correct_and_smooth
from official.graph_dataset import build_transductive_graph
from official.inner_fold_selection import select_and_apply_inner_fold
from official.train import _load_dataset, _label_frame, _label_free_feature_columns
from official.transductive_features import build_transductive_feature_frame
from official.transductive_validation import PrimarySplitSpec, build_primary_transductive_splits, iter_fold_assignments
from official.validate import _classification_metrics


CB_FROZEN = {
    "iterations": 1200,
    "depth": 5,
    "learning_rate": 0.0376,
    "l2_leaf_reg": 2.597,
    "border_count": 64,
    "random_strength": 0.849,
    "bagging_temperature": 1.327,
    "min_data_in_leaf": 60,
    "max_class_weight": 7.365,
}

LGBM_FROZEN = {
    "n_estimators": 700,
    "learning_rate": 0.0618,
    "num_leaves": 38,
    "subsample": 0.845,
    "colsample_bytree": 0.595,
    "min_child_samples": 34,
    "reg_alpha": 0.391,
    "reg_lambda": 0.583,
}

XGB_FIXED = {
    "n_estimators": 1500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 5.0,
    "min_child_weight": 5.0,
}

XGB_GRID = {
    "max_depth": [4, 5, 6, 7, 8],
    "learning_rate": [0.02, 0.03, 0.05, 0.07, 0.10],
}


def _grid_dir() -> Path:
    root = nh.load_official_paths().artifact_dir / "xgb_grid_nested_ablation"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _best_blend_f1(oof_frame: pd.DataFrame) -> float:
    frame = nh._add_base_meta_features(oof_frame.copy())
    weights = nh.tune_blend_weights(frame)
    blend = nh.BlendEnsemble(weights)
    cols = [c for c in nh.STACKER_FEATURE_COLUMNS if c in frame.columns]
    probs = blend.predict_proba(frame[cols])[:, 1]
    labels = frame["status"].astype(int).to_numpy()
    best = 0.0
    for t in np.arange(0.05, 0.50, 0.01):
        preds = (probs >= t).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        pp = int((preds == 1).sum())
        pos = int((labels == 1).sum())
        precision = tp / max(1, pp)
        recall = tp / max(1, pos)
        if precision + recall > 0:
            best = max(best, 2.0 * precision * recall / (precision + recall))
    return float(best)


def _build_inner_folds(
    outer_train_frame: pd.DataFrame,
    feature_columns: list[str],
    graph,
    dataset: pd.DataFrame,
    label_frame: pd.DataFrame,
    n_inner: int = 2,
    seed: int = nh.RANDOM_SEED,
) -> list[dict]:
    splitter = __import__("sklearn.model_selection").model_selection.StratifiedKFold(
        n_splits=n_inner,
        shuffle=True,
        random_state=seed,
    )
    labeled = outer_train_frame[outer_train_frame["status"].notna()].copy()
    y = labeled["status"].astype(int)
    folds = []
    for fold_id, (train_idx, valid_idx) in enumerate(splitter.split(labeled, y)):
        train_f = labeled.iloc[train_idx].copy()
        valid_f = labeled.iloc[valid_idx].copy()
        train_labels = dict(zip(train_f["user_id"].astype(int).tolist(), train_f["status"].astype(float).tolist()))
        x_train_enc, enc_cols = nh.encode_frame(train_f, feature_columns)
        x_valid_enc, _ = nh.encode_frame(valid_f, feature_columns, reference_columns=enc_cols)
        fold_train_labels = label_frame[label_frame["user_id"].astype(int).isin(set(train_f["user_id"].astype(int)))].copy()
        trans = build_transductive_feature_frame(graph, fold_train_labels)
        trans_cols = [c for c in trans.columns if c != "user_id"]
        relevant_ids = set(train_f["user_id"].astype(int)) | set(valid_f["user_id"].astype(int))
        relevant = dataset[dataset["user_id"].astype(int).isin(relevant_ids)].copy()
        with_td = relevant.merge(trans, on="user_id", how="left")
        with_td[trans_cols] = with_td[trans_cols].fillna(0.0)
        train_td = with_td[with_td["user_id"].astype(int).isin(set(train_f["user_id"].astype(int)))].copy()
        valid_td = with_td[with_td["user_id"].astype(int).isin(set(valid_f["user_id"].astype(int)))].copy()
        folds.append(
            {
                "fold_id": fold_id,
                "train_frame": train_f,
                "valid_frame": valid_f,
                "train_labels_dict": train_labels,
                "x_train_encoded": x_train_enc,
                "x_valid_encoded": x_valid_enc,
                "train_transductive": train_td,
                "valid_transductive": valid_td,
                "base_b_columns": feature_columns + trans_cols,
                "scores": nh._extract_scores(valid_f),
            }
        )
    return folds


def _build_precache(inner_folds: list[dict], graph, feature_columns: list[str]) -> list[dict]:
    cache = []
    for fd in inner_folds:
        train_f, valid_f = fd["train_frame"], fd["valid_frame"]
        ba = nh._fit_catboost_with_params(train_f, valid_f, feature_columns, dict(CB_FROZEN), nh.HPO_SEED, thread_count=nh.GPU_STREAM_CPU_THREADS)
        ba_valid, cs_valid = nh._cs_from_base_a_fits([ba], train_f, valid_f, None, graph, fd["train_labels_dict"])
        bb = nh._fit_catboost_with_params(
            fd["train_transductive"],
            fd["valid_transductive"],
            fd["base_b_columns"],
            {"task_type": "CPU", "l2_leaf_reg": 5.0},
            nh.HPO_SEED,
            force_cpu=True,
            thread_count=nh.CPU_STREAM_THREADS,
        )
        bd = nh._fit_lgbm_cpu(train_f, valid_f, feature_columns, dict(LGBM_FROZEN), nh.HPO_SEED, n_jobs=nh.CPU_STREAM_THREADS)
        cache.append(
            {
                "base_a_probs": ba_valid,
                "cs_probs": cs_valid,
                "base_b_probs": np.asarray(bb.validation_probabilities, dtype=float),
                "base_d_probs": np.asarray(bd.validation_probabilities, dtype=float),
                **fd["scores"],
            }
        )
    return cache


def _objective(params: dict, inner_folds: list[dict], de_cache: list[dict]) -> float:
    vals = []
    runtime_xgb = nh.xgboost_runtime_params()
    from xgboost import XGBClassifier

    for i, fd in enumerate(inner_folds):
        y_tr = fd["train_frame"]["status"].astype(int)
        y_va = fd["valid_frame"]["status"].astype(int)
        pos, neg = max(1, int(y_tr.sum())), max(1, len(y_tr) - int(y_tr.sum()))
        model = XGBClassifier(
            **params,
            **runtime_xgb,
            scale_pos_weight=min(neg / pos, 15.0),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=nh.HPO_SEED,
            verbosity=0,
            early_stopping_rounds=100,
        )
        try:
            model.fit(fd["x_train_encoded"], y_tr, eval_set=[(fd["x_valid_encoded"], y_va)], verbose=False)
        except Exception:
            if runtime_xgb.get("device") != "cuda":
                return 0.0
            model = XGBClassifier(
                **params,
                tree_method="hist",
                device="cpu",
                scale_pos_weight=min(neg / pos, 15.0),
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=nh.HPO_SEED,
                verbosity=0,
                early_stopping_rounds=100,
            )
            model.fit(fd["x_train_encoded"], y_tr, eval_set=[(fd["x_valid_encoded"], y_va)], verbose=False)
        base_e_valid = model.predict_proba(fd["x_valid_encoded"])[:, 1]
        oof = nh._assemble_oof(
            fd["valid_frame"],
            de_cache[i]["base_a_probs"],
            de_cache[i]["cs_probs"],
            de_cache[i]["base_b_probs"],
            de_cache[i]["base_d_probs"],
            base_e_valid,
            de_cache[i],
        )
        vals.append(_best_blend_f1(oof))
    return float(np.mean(vals))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="*", default=[0, 1])
    args = parser.parse_args()

    dataset = _load_dataset("full")
    label_frame = _label_frame(dataset)
    feature_columns = _label_free_feature_columns(dataset)
    graph = build_transductive_graph(dataset)
    outer_split = build_primary_transductive_splits(dataset, cutoff_tag="full", spec=PrimarySplitSpec(), write_outputs=False)

    results = []
    for outer_fold, outer_train_users, outer_valid_users in [a for a in iter_fold_assignments(outer_split, "primary_fold") if a[0] in args.folds]:
        print(f"[xgb-grid] outer_fold={outer_fold} starting", flush=True)
        outer_train_frame = dataset[dataset["user_id"].astype(int).isin(outer_train_users)].copy()
        inner_folds = _build_inner_folds(outer_train_frame, feature_columns, graph, dataset, label_frame, n_inner=2, seed=nh.RANDOM_SEED)
        de_cache = _build_precache(inner_folds, graph, feature_columns)

        best_score = -1.0
        best_params = None
        rows = []
        for max_depth, learning_rate in itertools.product(XGB_GRID["max_depth"], XGB_GRID["learning_rate"]):
            params = {**XGB_FIXED, "max_depth": max_depth, "learning_rate": learning_rate}
            score = _objective(params, inner_folds, de_cache)
            row = {"max_depth": max_depth, "learning_rate": learning_rate, "score": score}
            rows.append(row)
            print(f"[xgb-grid] outer_fold={outer_fold} candidate={row}", flush=True)
            if score > best_score:
                best_score = score
                best_params = params

        fold_dir = _grid_dir() / f"fold_{outer_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "grid_scores.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        (fold_dir / "best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")
        report = {"outer_fold": outer_fold, "best_f1": best_score, "best_params": best_params}
        (fold_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        results.append(report)

    summary = {
        "outer_folds": args.folds,
        "reports": results,
        "mean_best_f1": float(np.mean([r["best_f1"] for r in results])) if results else None,
    }
    out = _grid_dir() / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[xgb-grid] saved={out}", flush=True)


if __name__ == "__main__":
    main()
