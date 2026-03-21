"""Generate final submission from best configurable pipeline config.

This script trains final models on ALL labeled data using the best config,
then scores predict_only users and saves the submission CSV.

Usage:
    cd bitoguard_core
    PYTHONPATH=. python -m official.generate_submission
    # Or with custom threshold:
    BITOGUARD_THRESHOLD=0.18 PYTHONPATH=. python -m official.generate_submission
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from official.common import load_clean_table, load_official_paths
from official.community_features import build_community_features
from official.configurable_pipeline import _add_feature_eng_node_attrs
from official.correct_and_smooth import correct_and_smooth
from official.graph_dataset import build_transductive_graph
from official.modeling import fit_catboost, fit_lgbm
from official.modeling_xgb import fit_xgboost
from official.rules import evaluate_official_rules
from official.train import (
    _BASE_A_SEEDS,
    _BASE_D_SEEDS,
    _BASE_E_SEEDS,
    _label_frame,
    _label_free_feature_columns,
    _load_dataset,
    _prepare_base_frames,
    _transductive_feature_columns,
)
from official.transductive_features import build_transductive_feature_frame

# Best config (community + feature_eng - profile_sim - ppr)
BEST_CONFIG = {
    "community_features": True,
    "feature_eng_node_attrs": True,
    "pu_learning_loss": True,
    "multi_scale_ppr": False,
    "profile_similarity_edges": False,
    "graphsage_3layer": False,
}

# OOF-derived values (from best seed4 run: F1=0.3856 @ thr=0.18)
OOF_THRESHOLD = float(os.getenv("BITOGUARD_THRESHOLD", "0.18"))

# Blend weights from seg-blend (from seed4 log):
# Connected: {base_a:0.15, base_cs:0.25, base_e:0.05, cs_x_anomaly:0.55}
# Isolated:  {base_a:0.05, base_d:0.05, base_e:0.10, base_cs:0.15, cs_x_anomaly:0.65}
CONNECTED_WEIGHTS = {
    "base_a": 0.15, "base_cs": 0.25, "base_e": 0.05, "cs_x_anomaly": 0.55
}
ISOLATED_WEIGHTS = {
    "base_a": 0.05, "base_d": 0.05, "base_e": 0.10, "base_cs": 0.15, "cs_x_anomaly": 0.65
}
CS_DEFICIT_THRESHOLD = 0.05  # connected if base_a - base_cs <= 0.05


def generate_submission(output_path: str | None = None) -> pd.DataFrame:
    """Train final models on all labeled data and score predict_only users.

    Returns a DataFrame with columns [user_id, final_probability, flagged].
    """
    t0 = time.time()
    paths = load_official_paths()
    output_path = output_path or str(paths.prediction_dir / "configurable_submission.csv")

    print("[generate_submission] Loading dataset...")
    dataset = _load_dataset("full")

    # Apply rule features
    print("[generate_submission] Applying rules...")
    rule_df = evaluate_official_rules(dataset)
    _rule_cols_to_drop = [c for c in rule_df.columns if c != "user_id" and c in dataset.columns]
    if _rule_cols_to_drop:
        dataset = dataset.drop(columns=_rule_cols_to_drop)
    dataset = dataset.merge(rule_df, on="user_id", how="left")

    # Build graph
    print("[generate_submission] Building graph...")
    label_frame = _label_frame(dataset)
    graph = build_transductive_graph(dataset)

    # Community features (label-free only)
    if BEST_CONFIG.get("community_features"):
        try:
            comm_df = build_community_features(graph, label_frame)
            label_free_comm_cols = [
                c for c in comm_df.columns
                if c != "user_id" and not any(
                    kw in c for kw in ("pos_count", "pos_ratio", "high_risk", "ppr_sum")
                )
            ]
            if label_free_comm_cols:
                comm_safe = comm_df[["user_id"] + label_free_comm_cols]
                dataset = dataset.merge(comm_safe, on="user_id", how="left")
                for col in label_free_comm_cols:
                    if col in dataset.columns:
                        dataset[col] = dataset[col].fillna(0.0)
                print(f"[generate_submission] Community features: {label_free_comm_cols}")
        except Exception as e:
            print(f"[generate_submission] Community features failed: {e}")

    # Feature engineering (uses configurable_pipeline's implementation)
    if BEST_CONFIG.get("feature_eng_node_attrs"):
        try:
            dataset = _add_feature_eng_node_attrs(dataset)
            print("[generate_submission] Feature engineering applied")
        except Exception as e:
            print(f"[generate_submission] Feature engineering failed: {e}")

    # Build feature columns
    base_a_feature_columns = _label_free_feature_columns(dataset)
    trans_df = build_transductive_feature_frame(graph, label_frame)
    trans_cols = _transductive_feature_columns(trans_df)
    base_b_feature_columns = base_a_feature_columns + trans_cols

    # Split into labeled and predict_only
    labeled_user_ids = set(label_frame["user_id"].astype(int).tolist())
    predict_label_df = load_clean_table("predict_label")
    predict_user_ids = set(predict_label_df["user_id"].astype(int).tolist())

    label_free_frame, with_transductive_frame = _prepare_base_frames(dataset, trans_df)
    labeled_lf = label_free_frame[label_free_frame["user_id"].astype(int).isin(labeled_user_ids)].copy()
    labeled_trans = with_transductive_frame[with_transductive_frame["user_id"].astype(int).isin(labeled_user_ids)].copy()
    predict_lf = label_free_frame[label_free_frame["user_id"].astype(int).isin(predict_user_ids)].copy()
    predict_trans = with_transductive_frame[with_transductive_frame["user_id"].astype(int).isin(predict_user_ids)].copy()

    # PU learning parameters
    catboost_params: dict = {}
    if BEST_CONFIG.get("pu_learning_loss"):
        catboost_params["pu_negative_weight"] = 0.7

    print(f"[generate_submission] Training final models on {len(labeled_lf)} labeled users...")

    # Base A: multi-seed CatBoost on label-free features
    base_a_models = [
        fit_catboost(labeled_lf, None, base_a_feature_columns, focal_gamma=2.0,
                     random_seed=s, catboost_params=catboost_params).model
        for s in _BASE_A_SEEDS
    ]

    # Base D: multi-seed LightGBM
    base_d_models = [
        fit_lgbm(labeled_lf, None, base_a_feature_columns, random_seed=s).model
        for s in _BASE_D_SEEDS
    ]

    # Base E: multi-seed XGBoost
    base_e_models = [
        fit_xgboost(labeled_lf, None, base_a_feature_columns, random_seed=s).model
        for s in _BASE_E_SEEDS
    ]

    # Base B: CatBoost on transductive features (CPU, lower l2)
    base_b_params = {"task_type": "CPU", "l2_leaf_reg": 5.0}
    base_b_model = fit_catboost(labeled_trans, None, base_b_feature_columns,
                                focal_gamma=2.0, catboost_params=base_b_params).model

    print("[generate_submission] Scoring predict_only users...")

    # Score predict_only users
    base_a_probs = np.mean(
        [m.predict_proba(predict_lf[base_a_feature_columns])[:, 1] for m in base_a_models],
        axis=0,
    )
    base_d_probs = np.mean(
        [m.predict_proba(predict_lf[base_a_feature_columns]) for m in base_d_models],
        axis=0,
    )
    base_e_probs = np.mean(
        [m.predict_proba(predict_lf[base_a_feature_columns]) for m in base_e_models],
        axis=0,
    )
    base_b_probs = base_b_model.predict_proba(predict_trans[base_b_feature_columns])[:, 1]

    # C&S: compute for all users, extract predict_only
    all_base_probs: dict[int, float] = {}
    all_lf = label_free_frame.copy()
    all_base_a = np.mean(
        [m.predict_proba(all_lf[base_a_feature_columns].to_numpy())[:, 1] for m in base_a_models],
        axis=0,
    )
    for uid, prob in zip(all_lf["user_id"].astype(int), all_base_a):
        all_base_probs[int(uid)] = float(prob)

    train_labels: dict[int, float] = dict(zip(
        label_frame["user_id"].astype(int),
        label_frame["status"].astype(float),
    ))
    cs_result = correct_and_smooth(
        graph, train_labels, all_base_probs,
        alpha_correct=0.5, alpha_smooth=0.5,
        n_correct_iter=50, n_smooth_iter=50,
        restore_isolated_top_pct=0.0,
    )

    predict_ids = predict_lf["user_id"].astype(int).tolist()
    base_cs_probs = np.array(
        [cs_result.get(int(uid), all_base_probs.get(int(uid), float(ba)))
         for uid, ba in zip(predict_ids, base_a_probs)],
        dtype=float,
    )

    # Anomaly scores for predict_only
    anomaly_df = load_clean_table("features") if False else pd.DataFrame()  # skip for now
    anomaly_scores = predict_lf.get("anomaly_score", pd.Series(0.0, index=predict_lf.index)).values
    if "anomaly_score" in predict_lf.columns:
        anomaly_scores = predict_lf["anomaly_score"].fillna(0.0).values
    else:
        anomaly_scores = np.zeros(len(predict_ids))

    # cs_x_anomaly interaction
    cs_x_anomaly = base_cs_probs * anomaly_scores

    # cs_deficit for segment-aware blend
    cs_deficit = base_a_probs - base_cs_probs

    # Apply segment-aware blend
    final_probs = np.zeros(len(predict_ids))
    for i in range(len(predict_ids)):
        if cs_deficit[i] <= CS_DEFICIT_THRESHOLD:
            # Connected: use connected weights
            p = (CONNECTED_WEIGHTS["base_a"] * base_a_probs[i] +
                 CONNECTED_WEIGHTS["base_cs"] * base_cs_probs[i] +
                 CONNECTED_WEIGHTS["base_e"] * base_e_probs[i] +
                 CONNECTED_WEIGHTS["cs_x_anomaly"] * cs_x_anomaly[i])
        else:
            # Isolated: use isolated weights
            p = (ISOLATED_WEIGHTS["base_a"] * base_a_probs[i] +
                 ISOLATED_WEIGHTS["base_d"] * base_d_probs[i] +
                 ISOLATED_WEIGHTS["base_e"] * base_e_probs[i] +
                 ISOLATED_WEIGHTS["base_cs"] * base_cs_probs[i] +
                 ISOLATED_WEIGHTS["cs_x_anomaly"] * cs_x_anomaly[i])
        final_probs[i] = p

    # Apply threshold
    flagged = (final_probs >= OOF_THRESHOLD).astype(int)
    n_flagged = flagged.sum()

    result_df = pd.DataFrame({
        "user_id": predict_ids,
        "final_probability": final_probs,
        "flagged": flagged,
    })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    elapsed = time.time() - t0
    print(f"[generate_submission] DONE in {elapsed:.0f}s")
    print(f"[generate_submission] {n_flagged}/{len(predict_ids)} predict_only users flagged")
    print(f"[generate_submission] Threshold: {OOF_THRESHOLD:.4f}")
    print(f"[generate_submission] Output: {output_path}")
    return result_df


if __name__ == "__main__":
    generate_submission()
