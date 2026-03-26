"""Measure XGB HPO bias: compare E15 OOF F1 with HPO vs baseline XGB params.

This script reuses the hpo_xgboost.py fold cache approach:
  1. Pre-compute fixed models (Base A, B, D, C&S) per fold (~30-40 min)
  2. Train Base E with BASELINE params → compute blend F1
  3. Train Base E with HPO params → compute blend F1
  4. Report the bias = HPO_F1 - baseline_F1

Usage:
    cd bitoguard_core && source .venv/bin/activate
    PYTHONPATH=. python measure_xgb_bias.py
"""
import os
import sys
import time

os.environ.setdefault("DISABLE_TEMP_FEATURES", "1")
os.environ.setdefault("SKIP_GNN", "1")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from official.train import _load_dataset, _label_frame, _label_free_feature_columns
from official.train import _BASE_A_SEEDS, _BASE_D_SEEDS
from official.transductive_validation import build_primary_transductive_splits, iter_fold_assignments
from official.transductive_features import build_transductive_feature_frame
from official.graph_dataset import build_transductive_graph
from official.modeling_xgb import fit_xgboost
from official.stacking import tune_blend_weights
from official.correct_and_smooth import correct_and_smooth
from official.modeling import fit_catboost, fit_lgbm
from official.common import RANDOM_SEED

# ── Param sets ───────────────────────────────────────────────────────────────
XGB_BASELINE = {
    "n_estimators": 1500, "max_depth": 7, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1,
    "reg_lambda": 5.0, "min_child_weight": 5.0,
}
XGB_HPO = {
    "n_estimators": 1500, "max_depth": 6, "learning_rate": 0.0585,
    "subsample": 0.812, "colsample_bytree": 0.881, "reg_alpha": 0.061,
    "reg_lambda": 5.707, "min_child_weight": 5.185,
}
SEEDS = [42, 123]


def compute_blend_f1(oof: pd.DataFrame) -> tuple[float, float, dict]:
    """Compute best blend F1 from OOF frame."""
    weights = tune_blend_weights(oof)
    blend_prob = sum(oof[c] * w for c, w in weights.items() if c in oof.columns)
    labeled = oof[oof["status"].notna()]
    y = labeled["status"].astype(int).values
    probs = blend_prob[labeled.index].values

    best_f1, best_thr = 0.0, 0.10
    for thr in np.arange(0.04, 0.50, 0.005):
        pred = (probs >= thr).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)

    return best_f1, best_thr, weights


def main():
    t_start = time.time()

    # ── Step 1: Load dataset + build fold cache ──────────────────────────────
    print("[bias] Loading dataset...", flush=True)
    dataset = _load_dataset()
    label_frame = _label_frame(dataset)
    base_a_cols = _label_free_feature_columns(dataset)
    graph = build_transductive_graph(dataset)
    split_frame = build_primary_transductive_splits(dataset)

    print("[bias] Pre-computing fixed models per fold (Base A, B, D, C&S)...", flush=True)
    fold_cache = {}
    for fold_id, train_users, valid_users in iter_fold_assignments(split_frame, "primary_fold"):
        t_fold = time.time()
        train_lf = dataset[dataset["user_id"].astype(int).isin(train_users)].copy()
        valid_lf = dataset[dataset["user_id"].astype(int).isin(valid_users)].copy()

        # Base A (4 seeds)
        base_a_models = []
        for seed in _BASE_A_SEEDS:
            fit = fit_catboost(train_lf, valid_lf, base_a_cols, focal_gamma=2.0, random_seed=seed)
            base_a_models.append(fit)
        val_a_probs = np.mean([m.validation_probabilities for m in base_a_models], axis=0)

        # Base B (transductive)
        fold_train_labels = label_frame[label_frame["user_id"].astype(int).isin(train_users)]
        trans_feats = build_transductive_feature_frame(graph, fold_train_labels)
        trans_cols = [c for c in trans_feats.columns if c != "user_id"]
        train_trans = train_lf.merge(trans_feats, on="user_id", how="left")
        train_trans[trans_cols] = train_trans[trans_cols].fillna(0.0)
        valid_trans = valid_lf.merge(trans_feats, on="user_id", how="left")
        valid_trans[trans_cols] = valid_trans[trans_cols].fillna(0.0)
        base_b_cols = base_a_cols + trans_cols
        base_b_fit = fit_catboost(train_trans, valid_trans, base_b_cols, focal_gamma=2.0,
                                  catboost_params={"task_type": "CPU", "l2_leaf_reg": 5.0})

        # Base D (3 seeds)
        base_d_probs = []
        for seed in _BASE_D_SEEDS:
            fit_d = fit_lgbm(train_lf, valid_lf, base_a_cols, random_seed=seed)
            base_d_probs.append(fit_d.validation_probabilities)
        val_d_probs = np.mean(base_d_probs, axis=0)

        # C&S
        train_a_probs = np.mean(
            [m.model.predict_proba(train_lf[base_a_cols])[:, 1] for m in base_a_models], axis=0)
        cs_base = {}
        for uid, p in zip(train_lf["user_id"].astype(int), train_a_probs):
            cs_base[int(uid)] = float(p)
        for uid, p in zip(valid_lf["user_id"].astype(int), val_a_probs):
            cs_base[int(uid)] = float(p)
        all_labeled_ids = set(train_users) | set(valid_users)
        unlabeled = dataset[~dataset["user_id"].astype(int).isin(all_labeled_ids)]
        if len(unlabeled) > 0:
            ul_probs = np.mean(
                [m.model.predict_proba(unlabeled[base_a_cols])[:, 1] for m in base_a_models], axis=0)
            for uid, p in zip(unlabeled["user_id"].astype(int), ul_probs):
                cs_base[int(uid)] = float(p)
        cs_labels = dict(zip(fold_train_labels["user_id"].astype(int), fold_train_labels["status"].astype(float)))
        cs_result = correct_and_smooth(graph, cs_labels, cs_base, 0.5, 0.5, 50, 50)
        val_cs = np.array([cs_result.get(int(u), float(p)) for u, p in
                           zip(valid_lf["user_id"].astype(int), val_a_probs)], dtype=float)

        fold_cache[fold_id] = {
            "train_lf": train_lf, "valid_lf": valid_lf,
            "val_a": val_a_probs,
            "val_b": np.asarray(base_b_fit.validation_probabilities, dtype=float),
            "val_d": np.asarray(val_d_probs, dtype=float),
            "val_cs": val_cs,
            "valid_users": valid_users,
            "rule_score": pd.to_numeric(valid_lf["rule_score"], errors="coerce").fillna(0.0).values if "rule_score" in valid_lf.columns else np.zeros(len(valid_lf)),
            "anomaly_score": pd.to_numeric(valid_lf["anomaly_score"], errors="coerce").fillna(0.0).values if "anomaly_score" in valid_lf.columns else np.zeros(len(valid_lf)),
            "crypto_anomaly": pd.to_numeric(valid_lf["crypto_anomaly_score"], errors="coerce").fillna(0.0).values if "crypto_anomaly_score" in valid_lf.columns else np.zeros(len(valid_lf)),
        }
        print(f"  fold {fold_id} cached ({time.time()-t_fold:.0f}s)", flush=True)

    cache_time = time.time() - t_start
    print(f"\n[bias] Fold cache built in {cache_time:.0f}s", flush=True)

    # ── Step 2: Run Base E with both param sets ──────────────────────────────
    def run_base_e(params: dict, label: str) -> tuple[float, float, dict]:
        t0 = time.time()
        oof_frames = []
        for fold_id, cache in fold_cache.items():
            e_probs = []
            for seed in SEEDS:
                fit_e = fit_xgboost(cache["train_lf"], cache["valid_lf"], base_a_cols,
                                    random_seed=seed, params=params)
                e_probs.append(fit_e.validation_probabilities)
            val_e = np.mean(e_probs, axis=0)

            ff = cache["valid_lf"][["user_id", "status"]].copy()
            ff["primary_fold"] = fold_id
            ff["base_a_probability"] = cache["val_a"]
            ff["base_c_s_probability"] = cache["val_cs"]
            ff["base_b_probability"] = cache["val_b"]
            ff["base_c_probability"] = 0.0
            ff["base_d_probability"] = cache["val_d"]
            ff["base_e_probability"] = val_e
            ff["rule_score"] = cache["rule_score"]
            ff["anomaly_score"] = cache["anomaly_score"]
            ff["crypto_anomaly_score"] = cache["crypto_anomaly"]
            oof_frames.append(ff)

        oof = pd.concat(oof_frames, ignore_index=True)
        f1, thr, weights = compute_blend_f1(oof)
        elapsed = time.time() - t0
        print(f"\n[bias] {label}: F1={f1:.4f} thr={thr:.3f} ({elapsed:.0f}s)", flush=True)
        print(f"       weights={weights}", flush=True)
        return f1, thr, weights

    print("\n" + "="*60)
    print("[bias] Running Base E with BASELINE params...")
    print("="*60)
    f1_baseline, thr_baseline, w_baseline = run_base_e(XGB_BASELINE, "BASELINE")

    print("\n" + "="*60)
    print("[bias] Running Base E with HPO params...")
    print("="*60)
    f1_hpo, thr_hpo, w_hpo = run_base_e(XGB_HPO, "HPO")

    # ── Step 3: Report ───────────────────────────────────────────────────────
    bias = f1_hpo - f1_baseline
    total = time.time() - t_start

    print("\n" + "="*60)
    print(f"[RESULT] XGB HPO Bias Measurement")
    print(f"="*60)
    print(f"  F1 (baseline XGB):  {f1_baseline:.4f}  (thr={thr_baseline:.3f})")
    print(f"  F1 (HPO XGB):       {f1_hpo:.4f}  (thr={thr_hpo:.3f})")
    print(f"  Bias (HPO - base):  {bias:+.4f}")
    print(f"  Total time:         {total:.0f}s ({total/60:.1f} min)")
    print()
    if abs(bias) < 0.005:
        print("  → Bias is negligible (<0.005). HPO XGB F1 is trustworthy.")
    elif abs(bias) < 0.015:
        print("  → Small bias (0.005-0.015). HPO genuine gain likely real, minor optimistic bias.")
    else:
        print("  → Substantial bias (>0.015). Nested CV recommended for unbiased estimate.")


if __name__ == "__main__":
    main()
