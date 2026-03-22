"""XGBoost HPO: re-train only Base E per trial, keep Base A/B/D/C&S fixed."""
import os, sys
os.environ.setdefault("DISABLE_TEMP_FEATURES", "1")
os.environ.setdefault("SKIP_GNN", "1")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import optuna
from official.train import _load_dataset, _label_frame, _label_free_feature_columns
from official.transductive_validation import build_primary_transductive_splits
from official.transductive_features import build_transductive_feature_frame
from official.graph_dataset import build_transductive_graph
from official.modeling_xgb import fit_xgboost
from official.stacking import tune_blend_weights, build_stacker_oof, STACKER_FEATURE_COLUMNS
from official.correct_and_smooth import correct_and_smooth
from official.modeling import fit_catboost, fit_lgbm
from official.common import RANDOM_SEED

N_TRIALS = 60
SEEDS = [42, 123]

print("[HPO] Loading dataset...", flush=True)
dataset = _load_dataset()
label_frame = _label_frame(dataset)
base_a_cols = _label_free_feature_columns(dataset)
graph = build_transductive_graph(dataset)
split_frame = build_primary_transductive_splits(dataset)
labeled = dataset[dataset["status"].notna()]
labeled_ids = labeled["user_id"].astype(int).tolist()

# Pre-compute per-fold: Base A (4 seeds), Base B, Base D (3 seeds), C&S, transductive features
from official.train import _BASE_A_SEEDS, _BASE_D_SEEDS
from official.transductive_validation import iter_fold_assignments

print("[HPO] Pre-computing fixed models per fold...", flush=True)
fold_cache = {}
for fold_id, train_users, valid_users in iter_fold_assignments(split_frame, "primary_fold"):
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
        "val_a": val_a_probs, "val_b": np.asarray(base_b_fit.validation_probabilities, dtype=float),
        "val_d": np.asarray(val_d_probs, dtype=float), "val_cs": val_cs,
        "valid_users": valid_users,
        "rule_score": pd.to_numeric(valid_lf["rule_score"], errors="coerce").fillna(0.0).values if "rule_score" in valid_lf.columns else np.zeros(len(valid_lf)),
        "anomaly_score": pd.to_numeric(valid_lf["anomaly_score"], errors="coerce").fillna(0.0).values if "anomaly_score" in valid_lf.columns else np.zeros(len(valid_lf)),
        "crypto_anomaly": pd.to_numeric(valid_lf["crypto_anomaly_score"], errors="coerce").fillna(0.0).values if "crypto_anomaly_score" in valid_lf.columns else np.zeros(len(valid_lf)),
    }
    print(f"  fold {fold_id} cached", flush=True)

print("[HPO] Starting Optuna...", flush=True)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 800, 2500, step=100),
        "max_depth": trial.suggest_int("max_depth", 5, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 20.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 15.0),
    }
    
    oof_frames = []
    for fold_id, cache in fold_cache.items():
        # Re-train XGBoost with trial params
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
    weights = tune_blend_weights(oof)
    
    # Compute blend F1
    available = [c for c in weights if c in oof.columns]
    blend_prob = sum(oof[c] * w for c, w in weights.items() if c in oof.columns)
    labeled = oof[oof["status"].notna()]
    y = labeled["status"].astype(int).values
    probs = blend_prob[labeled.index].values
    
    best_f1 = 0
    for thr in np.arange(0.10, 0.40, 0.01):
        pred = (probs >= thr).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        best_f1 = max(best_f1, f1)
    
    return best_f1

study = optuna.create_study(direction="maximize", study_name="xgb_hpo")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\n[HPO] Best F1: {study.best_value:.4f}")
print(f"[HPO] Best params: {study.best_params}")
print(f"[HPO] vs E2 baseline (0.4380): {study.best_value - 0.4380:+.4f}")

# Save best params
import json
with open("artifacts/official_features/hpo_xgb_best.json", "w") as f:
    json.dump({"best_f1": study.best_value, "best_params": study.best_params}, f, indent=2)
print("[HPO] Saved to artifacts/official_features/hpo_xgb_best.json")
