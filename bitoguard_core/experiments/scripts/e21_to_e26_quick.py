"""E25a/E24/E23: Quick OOF-only experiments (no pipeline rerun needed)."""
import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

oof = pd.read_parquet("artifacts/official_features/official_oof_predictions.parquet")
print(f"OOF shape: {oof.shape}, columns: {list(oof.columns)}")
y = oof["status"].astype(int).values
probs = oof["stacker_raw_probability"].values

print("=" * 60)
print("E25a: F-beta threshold sweep")
print("=" * 60)
for beta in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]:
    best_thr, best_fb = 0, 0
    for thr in np.arange(0.05, 0.50, 0.002):
        preds = (probs >= thr).astype(int)
        fb = fbeta_score(y, preds, beta=beta, zero_division=0)
        if fb > best_fb:
            best_fb = fb
            best_thr = thr
    actual_f1 = f1_score(y, (probs >= best_thr).astype(int), zero_division=0)
    p = precision_score(y, (probs >= best_thr).astype(int), zero_division=0)
    r = recall_score(y, (probs >= best_thr).astype(int), zero_division=0)
    print(f"  beta={beta:.1f}: thr={best_thr:.4f} -> F1={actual_f1:.4f} P={p:.4f} R={r:.4f}")

print()
print("=" * 60)
print("E24: Level-2 stacking")
print("=" * 60)
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

level2_features = [c for c in oof.columns
                   if c.endswith("_probability") or c.endswith("_score")
                   or c.startswith("base_") or c.startswith("cs_")
                   or c in ["stacker_raw_probability", "max_base_probability",
                            "std_base_probability"]]
level2_features = [c for c in level2_features if c in oof.columns
                   and c not in ["submission_probability"]]
print(f"  Level 2 features ({len(level2_features)}): {level2_features}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for depth, n_est, reg, spw in [
    (3, 100, 5.0, 10.0),
    (4, 200, 3.0, 10.0),
    (2, 50, 10.0, 10.0),
    (3, 100, 5.0, 15.0),
    (3, 100, 5.0, 5.0),
    (3, 150, 5.0, 20.0),
    (5, 300, 1.0, 10.0),
]:
    l2_preds = np.zeros(len(oof))
    for fold, (ti, vi) in enumerate(skf.split(oof, y)):
        X_tr = oof.iloc[ti][level2_features].fillna(0)
        X_va = oof.iloc[vi][level2_features].fillna(0)
        clf = LGBMClassifier(n_estimators=n_est, max_depth=depth, learning_rate=0.05,
                             reg_lambda=reg, scale_pos_weight=spw, verbose=-1,
                             random_state=42)
        clf.fit(X_tr, y[ti])
        l2_preds[vi] = clf.predict_proba(X_va)[:, 1]
    best_f1, best_thr = 0, 0
    for t in np.arange(0.05, 0.80, 0.005):
        f1 = f1_score(y, (l2_preds >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = t
    print(f"  d={depth} n={n_est} reg={reg} spw={spw}: F1={best_f1:.4f} @ thr={best_thr:.3f}")

print()
print("=" * 60)
print("E23: Two-stage cascade")
print("=" * 60)
s2_features = [c for c in level2_features if c != "stacker_raw_probability"]

for stage1_recall_target in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
    stage1_thr = 0.05
    for thr in np.arange(0.30, 0.01, -0.005):
        rec = recall_score(y, (probs >= thr).astype(int))
        if rec >= stage1_recall_target:
            stage1_thr = thr
            break

    candidate_mask = probs >= stage1_thr
    n_candidates = candidate_mask.sum()
    if n_candidates < 50:
        continue
    pos_rate = y[candidate_mask].mean()

    s2_preds = np.zeros(len(oof))
    cand_indices = np.where(candidate_mask)[0]
    cand_y = y[candidate_mask]
    cand_X = oof.iloc[cand_indices][s2_features].fillna(0).values

    if len(np.unique(cand_y)) < 2:
        continue

    skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (ti, vi) in enumerate(skf2.split(cand_X, cand_y)):
        clf = LGBMClassifier(n_estimators=200, max_depth=4, verbose=-1,
                             scale_pos_weight=max(1, (1-pos_rate)/pos_rate),
                             random_state=42)
        clf.fit(cand_X[ti], cand_y[ti])
        s2_preds[cand_indices[vi]] = clf.predict_proba(cand_X[vi])[:, 1]

    best_f1, best_thr = 0, 0
    for t in np.arange(0.05, 0.80, 0.005):
        f1 = f1_score(y, (s2_preds >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = t
    print(f"  recall={stage1_recall_target:.2f}: cands={n_candidates} pos={pos_rate:.3f} "
          f"-> F1={best_f1:.4f} @ thr={best_thr:.3f}")

print("\n  --- Cascade with XGBoost stage 2 ---")
from xgboost import XGBClassifier
for stage1_recall_target in [0.65, 0.70, 0.75]:
    stage1_thr = 0.05
    for thr in np.arange(0.30, 0.01, -0.005):
        rec = recall_score(y, (probs >= thr).astype(int))
        if rec >= stage1_recall_target:
            stage1_thr = thr
            break
    candidate_mask = probs >= stage1_thr
    n_candidates = candidate_mask.sum()
    pos_rate = y[candidate_mask].mean()
    s2_preds = np.zeros(len(oof))
    cand_indices = np.where(candidate_mask)[0]
    cand_y = y[candidate_mask]
    cand_X = oof.iloc[cand_indices][s2_features].fillna(0).values
    if len(np.unique(cand_y)) < 2:
        continue
    skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (ti, vi) in enumerate(skf2.split(cand_X, cand_y)):
        clf = XGBClassifier(n_estimators=200, max_depth=4, verbosity=0,
                            scale_pos_weight=max(1, (1-pos_rate)/pos_rate),
                            random_state=42, use_label_encoder=False)
        clf.fit(cand_X[ti], cand_y[ti])
        s2_preds[cand_indices[vi]] = clf.predict_proba(cand_X[vi])[:, 1]
    best_f1, best_thr = 0, 0
    for t in np.arange(0.05, 0.80, 0.005):
        f1 = f1_score(y, (s2_preds >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = t
    print(f"  recall={stage1_recall_target:.2f}: cands={n_candidates} pos={pos_rate:.3f} "
          f"-> F1={best_f1:.4f} @ thr={best_thr:.3f} (XGB)")

print()
print("=" * 60)
print("E21: Feature Importance Analysis")
print("=" * 60)
import pickle
models_dir = "artifacts/models"
cb_files = sorted([f for f in os.listdir(models_dir) if "catboost_base_a" in f and f.endswith(".pkl")])
if cb_files:
    cb = pickle.load(open(os.path.join(models_dir, cb_files[0]), "rb"))
    imp = cb.model.get_feature_importance()
    names = cb.feature_columns
    zero_feats = [n for n, i in zip(names, imp) if i == 0]
    low_feats = [n for n, i in zip(names, imp) if 0 < i < 0.1]
    print(f"  Total features: {len(names)}")
    print(f"  Zero-importance: {len(zero_feats)}")
    print(f"  Near-zero (<0.1): {len(low_feats)}")
    if zero_feats:
        print(f"  Zero-imp features: {zero_feats}")
    sorted_idx = np.argsort(imp)[::-1]
    print(f"  Top 20 features:")
    for i in sorted_idx[:20]:
        print(f"    {names[i]}: {imp[i]:.2f}")

print()
print("=" * 60)
print("SUMMARY -- E15 baseline: F1=0.4418")
print("=" * 60)
