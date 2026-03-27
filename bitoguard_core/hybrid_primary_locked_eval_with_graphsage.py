from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from official import nested_hpo as nh
from official import train as train_mod
from official import modeling_xgb as xgb_mod
from official.graph_model import train_graphsage_model
from official.inner_fold_selection import honest_oof_evaluation
from official.train import (
    PRIMARY_GRAPH_MAX_EPOCHS,
    _load_dataset,
    _label_frame,
    _label_free_feature_columns,
)
from official.transductive_features import build_transductive_feature_frame
from official.transductive_validation import PrimarySplitSpec, build_primary_transductive_splits, iter_fold_assignments
from official.graph_dataset import build_transductive_graph
from official.validate import _classification_metrics


ROOT = Path(__file__).resolve().parent
NEST_ROOT = ROOT / "artifacts" / "official_features" / "nested_hpo"
OUT_DIR = ROOT / "artifacts" / "official_features" / "hybrid_primary_locked_eval_with_graphsage"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_OOF = OUT_DIR / "hybrid_primary_locked_eval_with_graphsage_oof.parquet"
OUT_REPORT = OUT_DIR / "hybrid_primary_locked_eval_with_graphsage_report.json"


def aggregate_params(model_key: str) -> dict:
    rows = []
    for path in sorted(NEST_ROOT.glob("fold_*/best_params.json")):
        data = json.loads(path.read_text())
        rows.append(data[model_key])
    if not rows:
        raise RuntimeError(f"no nested best_params for {model_key}")
    out = {}
    for key in rows[0].keys():
        vals = [row[key] for row in rows]
        if isinstance(vals[0], bool):
            out[key] = Counter(vals).most_common(1)[0][0]
        elif isinstance(vals[0], int) and not isinstance(vals[0], bool):
            out[key] = int(round(float(np.median(vals))))
        elif isinstance(vals[0], float):
            out[key] = float(np.median(vals))
        else:
            out[key] = Counter(vals).most_common(1)[0][0]
    return out


CB_PARAMS = aggregate_params("catboost")
LGBM_PARAMS = aggregate_params("lightgbm")
XGB_PARAMS = aggregate_params("xgboost")


_original_fit_xgb = xgb_mod.fit_xgboost


def fit_catboost_locked(train_frame, valid_frame, feature_columns, focal_gamma=0.0, catboost_params=None, random_seed=nh.RANDOM_SEED):
    params = dict(catboost_params or {})
    if params.get("task_type") == "CPU":
        return nh._fit_catboost_with_params(
            train_frame,
            valid_frame,
            feature_columns,
            params,
            random_seed,
            force_cpu=True,
            thread_count=nh.CPU_STREAM_THREADS,
        )
    return nh._fit_catboost_with_params(
        train_frame,
        valid_frame,
        feature_columns,
        dict(CB_PARAMS),
        random_seed,
        thread_count=nh.GPU_STREAM_CPU_THREADS,
    )


def fit_lgbm_locked(train_frame, valid_frame, feature_columns, random_seed=nh.RANDOM_SEED):
    return nh._fit_lgbm_cpu(
        train_frame,
        valid_frame,
        feature_columns,
        dict(LGBM_PARAMS),
        random_seed,
        n_jobs=nh.CPU_STREAM_THREADS,
    )


def fit_xgboost_locked(train_frame, valid_frame, feature_columns, params=None, random_seed=nh.RANDOM_SEED):
    return _original_fit_xgb(
        train_frame,
        valid_frame,
        feature_columns,
        params=dict(XGB_PARAMS),
        random_seed=random_seed,
    )


train_mod.fit_catboost = fit_catboost_locked
train_mod.fit_lgbm = fit_lgbm_locked
train_mod.fit_xgboost = fit_xgboost_locked


def build_primary_oof_with_graphsage(dataset: pd.DataFrame, graph, split_frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    label_frame = _label_frame(dataset)
    rows: list[pd.DataFrame] = []
    for fold_id, train_users, valid_users in iter_fold_assignments(split_frame, "primary_fold"):
        fold_train_labels = label_frame[label_frame["user_id"].astype(int).isin(train_users)].copy()
        transductive = build_transductive_feature_frame(graph, fold_train_labels)
        trans_cols = [c for c in transductive.columns if c != "user_id"]

        relevant_ids = train_users | valid_users
        relevant = dataset[dataset["user_id"].astype(int).isin(relevant_ids)].copy()
        with_td = relevant.merge(transductive, on="user_id", how="left")
        with_td[trans_cols] = with_td[trans_cols].fillna(0.0)

        train_lf = dataset[dataset["user_id"].astype(int).isin(train_users)].copy()
        valid_lf = dataset[dataset["user_id"].astype(int).isin(valid_users)].copy()
        train_td = with_td[with_td["user_id"].astype(int).isin(train_users)].copy()
        valid_td = with_td[with_td["user_id"].astype(int).isin(valid_users)].copy()
        unlabeled = dataset[~dataset["user_id"].astype(int).isin(relevant_ids)].copy()

        base_a_fits = [
            fit_catboost_locked(train_lf, valid_lf, feature_columns, random_seed=seed)
            for seed in nh.FINAL_SEEDS_A
        ]
        cs_labels = dict(zip(fold_train_labels["user_id"].astype(int).tolist(), fold_train_labels["status"].astype(float).tolist()))
        base_a_valid, cs_valid = nh._cs_from_base_a_fits(base_a_fits, train_lf, valid_lf, unlabeled, graph, cs_labels)

        base_b_fit = nh._fit_catboost_with_params(
            train_td,
            valid_td,
            feature_columns + trans_cols,
            {"task_type": "CPU", "l2_leaf_reg": 5.0},
            nh.HPO_SEED,
            force_cpu=True,
            thread_count=nh.CPU_STREAM_THREADS,
        )

        base_d_fits = [fit_lgbm_locked(train_lf, valid_lf, feature_columns, random_seed=seed) for seed in nh.FINAL_SEEDS_D]
        base_d_valid = nh._mean_validation_probabilities(base_d_fits)

        y_tr_xgb = train_lf["status"].astype(int)
        pos_xgb = max(1, int(y_tr_xgb.sum()))
        neg_xgb = max(1, len(y_tr_xgb) - int(y_tr_xgb.sum()))
        dtrain_xgb, dvalid_xgb = nh._build_xgb_qdmatrices(train_lf, valid_lf, feature_columns)
        base_e_probs = [
            nh._fit_xgboost_qdm(dtrain_xgb, dvalid_xgb, dict(XGB_PARAMS), seed, min(neg_xgb / pos_xgb, 15.0))
            for seed in nh.FINAL_SEEDS_E
        ]
        base_e_valid = np.mean(base_e_probs, axis=0)

        graph_fit = train_graphsage_model(
            graph,
            label_frame=label_frame,
            train_user_ids=train_users,
            valid_user_ids=valid_users,
            max_epochs=PRIMARY_GRAPH_MAX_EPOCHS,
            hidden_dim=128,
        )

        frame = valid_lf[["user_id", "status"]].copy()
        frame["primary_fold"] = fold_id
        frame["base_a_probability"] = base_a_valid
        frame["base_c_s_probability"] = cs_valid
        frame["base_b_probability"] = np.asarray(base_b_fit.validation_probabilities, dtype=float)
        frame["base_c_probability"] = np.asarray(graph_fit.validation_probabilities, dtype=float)
        frame["base_d_probability"] = base_d_valid
        frame["base_e_probability"] = base_e_valid
        for col in ("rule_score", "anomaly_score", "crypto_anomaly_score", "anomaly_score_segmented"):
            frame[col] = pd.to_numeric(valid_lf[col], errors="coerce").fillna(0.0).to_numpy() if col in valid_lf.columns else 0.0
        rows.append(frame)

    return pd.concat(rows, ignore_index=True).sort_values("user_id").reset_index(drop=True)


def main() -> None:
    dataset = _load_dataset("full")
    graph = build_transductive_graph(dataset)
    feature_columns = _label_free_feature_columns(dataset)
    primary_split = build_primary_transductive_splits(dataset, cutoff_tag="full", spec=PrimarySplitSpec(), write_outputs=False)

    primary_oof = build_primary_oof_with_graphsage(dataset, graph, primary_split, feature_columns)
    honest_oof, honest_fold_metas = honest_oof_evaluation(primary_oof, fold_column="primary_fold")
    honest_thresholds = [meta["selected_threshold"] for meta in honest_fold_metas]
    honest_threshold = float(np.median(honest_thresholds))
    metrics = _classification_metrics(
        honest_oof["status"].astype(int).to_numpy(),
        honest_oof["submission_probability"].to_numpy(),
        honest_threshold,
    )
    report = {
        "frozen_catboost_params": CB_PARAMS,
        "frozen_lightgbm_params": LGBM_PARAMS,
        "frozen_xgboost_params": XGB_PARAMS,
        "honest_threshold": honest_threshold,
        "honest_fold_thresholds": honest_thresholds,
        "metrics": metrics,
        "rows": int(len(honest_oof)),
        "positives": int(honest_oof["status"].astype(int).sum()),
    }
    honest_oof.to_parquet(OUT_OOF, index=False)
    OUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"[hybrid+graphsage] saved_report={OUT_REPORT}", flush=True)
    print(f"[hybrid+graphsage] saved_oof={OUT_OOF}", flush=True)


if __name__ == "__main__":
    main()
