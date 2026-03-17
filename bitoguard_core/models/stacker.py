# bitoguard_core/models/stacker.py
"""Stacker: CatBoost + LightGBM + XGBoost + ExtraTrees OOF branches -> LR meta-learner + isotonic calibration.

Four-branch stacking:
  A: CatBoostClassifier (depth-6, scale_pos_weight)
  B: LGBMClassifier (num_leaves=63, min_child_samples=5, subsample/colsample)
  C: XGBClassifier (max_depth=6, gamma=0.1, scale_pos_weight)
  D: ExtraTreesClassifier (fully random splits for maximum diversity)

RandomForest was pruned: its meta-weight was -0.229 (actively hurting ensemble).
ET already provides the bagging-based diversity; RF added redundancy without lift.

OOF meta-learner: LogisticRegression on logit([P_A, P_B, P_C, P_D]) vectors.
Isotonic calibration: CalibratedClassifierCV(lr_meta, method='isotonic') fitted
on OOF predictions to map uncalibrated log-odds to true probabilities.
"""
from __future__ import annotations
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6
    _HAS_FROZEN = True
except ImportError:
    _HAS_FROZEN = False
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, precision_recall_curve,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold

from hardware import (
    catboost_runtime_params,
    describe_hardware,
    lightgbm_runtime_params,
    sklearn_n_jobs,
    xgboost_runtime_params,
)
from models.common import (
    NON_FEATURE_COLUMNS, forward_date_splits, model_dir,
    save_joblib, save_json,
)
from models.train_catboost import load_v2_training_dataset, CAT_FEATURE_NAMES
from features.graph_propagation import compute_label_propagation

# Per-fold label-aware propagation features (leakage-safe: built from training labels only)
_PROP_COLS = [
    "prop_ip", "prop_wallet", "prop_combined",
    "ip_rep_max_rate", "wallet_rep_max_rate",
    "rel_has_pos_neighbor", "rel_direct_pos_count",
    # New features from expanded graph_propagation (P1 port)
    "bfs_dist_1", "bfs_dist_2", "ppr_score",
    "pos_neighbor_count_relation", "pos_neighbor_count_wallet", "pos_neighbor_count_ip",
    "entity_wallet_max_seed_rate", "entity_ip_max_seed_rate",
    "component_seed_fraction", "component_seed_count",
]

def train_stacker(n_folds: int = 5, n_estimators: int = 500) -> dict:
    """OOF stacking: CatBoost + LightGBM branches -> Logistic Regression meta-learner."""
    print(f"[stacker] runtime: {describe_hardware()}")
    dataset = load_v2_training_dataset()
    feature_cols = [c for c in dataset.columns
                    if c not in NON_FEATURE_COLUMNS and c != "hidden_suspicious_label"]
    cat_indices  = [i for i, c in enumerate(feature_cols) if c in CAT_FEATURE_NAMES]

    date_splits   = forward_date_splits(dataset["snapshot_date"])
    train_dates   = set(date_splits["train"])
    train_df      = dataset[dataset["snapshot_date"].dt.date.isin(train_dates)].copy()
    train_df      = train_df.reset_index(drop=True)
    # Keep as DataFrame so CatBoost can handle categorical columns by name/index;
    # LightGBM receives the same DataFrame (it handles mixed types fine).
    groups        = train_df["user_id"].values
    x_train_df    = train_df[feature_cols].fillna(0).reset_index(drop=True)
    y_train       = train_df["hidden_suspicious_label"].values

    # Load entity edges for per-fold label propagation (leakage-safe 1-hop graph features).
    # Gracefully degrades to zeros if DuckDB is unavailable or table missing.
    try:
        from db.store import DuckDBStore
        from config import load_settings as _ls
        _store = DuckDBStore(_ls().db_path)
        entity_edges = _store.fetch_df("SELECT * FROM canonical.entity_edges")
        print(f"[propagation] Loaded {len(entity_edges):,} entity edges")
    except Exception as _e:
        entity_edges = pd.DataFrame()
        print(f"[propagation] Entity edges unavailable ({_e}); propagation features = 0")
    all_user_ids = train_df["user_id"].tolist()

    # Append propagation cols to feature list (numeric, no categorical encoding needed)
    feature_cols = feature_cols + _PROP_COLS
    cat_indices  = [i for i, c in enumerate(feature_cols) if c in CAT_FEATURE_NAMES]

    print(f"Training set: {len(y_train):,} samples, {int(y_train.sum()):,} positives ({y_train.mean():.2%})")
    print(f"Features: {len(feature_cols)} (+{len(_PROP_COLS)} propagation), Cat features: {len(cat_indices)}")

    oof_cb   = np.zeros(len(x_train_df))
    oof_lgbm = np.zeros(len(x_train_df))
    oof_xgb  = np.zeros(len(x_train_df))
    oof_et   = np.zeros(len(x_train_df))

    # XGBoost needs numeric-only data — encode categoricals as integer codes
    x_train_np = x_train_df.copy()
    for col in x_train_np.select_dtypes(include=["object", "category"]).columns:
        x_train_np[col] = pd.Categorical(x_train_np[col]).codes.astype("float32")
    x_train_np = x_train_np.values.astype("float32")

    fold_metrics: list[dict] = []
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold_i, (tr_idx, val_idx) in enumerate(sgkf.split(x_train_df, y_train, groups=groups), 1):
        pos = max(1, int(y_train[tr_idx].sum()))
        neg = max(1, len(tr_idx) - pos)
        print(f"\n[Fold {fold_i}/{n_folds}] train={len(tr_idx):,} val={len(val_idx):,} pos_rate={pos/len(tr_idx):.2%}")

        # ── Per-fold label propagation (leakage-safe) ──────────────────────────
        # Only training-fold labels are passed; validation users get scores based
        # on their graph proximity to training positives, with no label leakage.
        _fold_labels = pd.Series(
            y_train[tr_idx], index=train_df.iloc[tr_idx]["user_id"].values
        )
        _prop_df = compute_label_propagation(entity_edges, _fold_labels, all_user_ids)
        _prop_df = (
            _prop_df.set_index("user_id")
            .reindex(train_df["user_id"])
            .fillna(0)
            .reset_index(drop=True)
        )
        _prop_arr = _prop_df[_PROP_COLS].values.astype("float32")
        # Augmented DataFrames/arrays for this fold (base features + propagation)
        _x_fold_df = pd.concat(
            [x_train_df, pd.DataFrame(_prop_arr, columns=_PROP_COLS)], axis=1
        )
        _x_fold_np = np.hstack([x_train_np, _prop_arr])

        cb = CatBoostClassifier(
            iterations=n_estimators, learning_rate=0.05, depth=6,
            scale_pos_weight=neg / pos, cat_features=cat_indices,
            l2_leaf_reg=5, random_seed=42, verbose=0,
            **catboost_runtime_params(),
        )
        cb.fit(_x_fold_df.iloc[tr_idx], y_train[tr_idx])
        oof_cb[val_idx] = cb.predict_proba(_x_fold_df.iloc[val_idx])[:, 1]
        cb_auc = roc_auc_score(y_train[val_idx], oof_cb[val_idx])
        cb_ap  = average_precision_score(y_train[val_idx], oof_cb[val_idx])
        print(f"  CatBoost   AUC={cb_auc:.4f}  PR-AUC={cb_ap:.4f}")

        # Branch B: LGBM with larger leaves and stronger regularization for imbalanced data.
        # num_leaves=63 captures finer patterns; min_child_samples=5 allows small clusters
        # of positives to form their own leaf (critical at 2.5% prevalence).
        lgbm = LGBMClassifier(
            n_estimators=n_estimators, learning_rate=0.05, num_leaves=63,
            min_child_samples=5, reg_alpha=0.1, reg_lambda=1.0,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=neg / pos, random_state=42, verbose=-1,
            **lightgbm_runtime_params(),
        )
        lgbm.fit(_x_fold_df.iloc[tr_idx], y_train[tr_idx])
        oof_lgbm[val_idx] = lgbm.predict_proba(_x_fold_df.iloc[val_idx])[:, 1]
        lgbm_auc = roc_auc_score(y_train[val_idx], oof_lgbm[val_idx])
        lgbm_ap  = average_precision_score(y_train[val_idx], oof_lgbm[val_idx])
        print(f"  LightGBM   AUC={lgbm_auc:.4f}  PR-AUC={lgbm_ap:.4f}")

        # Branch C: XGBoost — different split-finding algorithm (exact) and
        # regularization (L1+L2); gamma=0.1 requires meaningful gain before splitting.
        xgb = XGBClassifier(
            n_estimators=n_estimators, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, gamma=0.1,
            reg_alpha=0.05, reg_lambda=1.0,
            scale_pos_weight=neg / pos, random_state=42,
            eval_metric="logloss", verbosity=0,
            **xgboost_runtime_params(),
        )
        xgb.fit(_x_fold_np[tr_idx], y_train[tr_idx])
        oof_xgb[val_idx] = xgb.predict_proba(_x_fold_np[val_idx])[:, 1]
        xgb_auc = roc_auc_score(y_train[val_idx], oof_xgb[val_idx])
        xgb_ap  = average_precision_score(y_train[val_idx], oof_xgb[val_idx])
        print(f"  XGBoost    AUC={xgb_auc:.4f}  PR-AUC={xgb_ap:.4f}")

        # Branch D: ExtraTrees — fully random split selection maximizes diversity
        # relative to gradient-boosted branches A/B/C, reducing ensemble variance
        # on the rare positive class where individual trees disagree most.
        et = ExtraTreesClassifier(
            n_estimators=max(50, n_estimators // 4), max_depth=10, min_samples_leaf=3,
            class_weight={0: 1, 1: neg / pos},
            random_state=42, n_jobs=sklearn_n_jobs(),
        )
        et.fit(_x_fold_np[tr_idx], y_train[tr_idx])
        oof_et[val_idx] = et.predict_proba(_x_fold_np[val_idx])[:, 1]
        et_auc = roc_auc_score(y_train[val_idx], oof_et[val_idx])
        et_ap  = average_precision_score(y_train[val_idx], oof_et[val_idx])
        print(f"  ExtraTrees AUC={et_auc:.4f}  PR-AUC={et_ap:.4f}")

        fold_metrics.append({
            "fold": fold_i,
            "n_train": int(len(tr_idx)),
            "n_val": int(len(val_idx)),
            "catboost":   {"auc": round(cb_auc, 4),   "pr_auc": round(cb_ap, 4)},
            "lgbm":       {"auc": round(lgbm_auc, 4), "pr_auc": round(lgbm_ap, 4)},
            "xgboost":    {"auc": round(xgb_auc, 4),  "pr_auc": round(xgb_ap, 4)},
            "extratrees": {"auc": round(et_auc, 4),   "pr_auc": round(et_ap, 4)},
        })

    # OOF metrics across all folds
    oof_cb_auc   = roc_auc_score(y_train, oof_cb)
    oof_lgbm_auc = roc_auc_score(y_train, oof_lgbm)
    oof_xgb_auc  = roc_auc_score(y_train, oof_xgb)
    oof_et_auc   = roc_auc_score(y_train, oof_et)
    oof_cb_ap    = average_precision_score(y_train, oof_cb)
    oof_lgbm_ap  = average_precision_score(y_train, oof_lgbm)
    oof_xgb_ap   = average_precision_score(y_train, oof_xgb)
    oof_et_ap    = average_precision_score(y_train, oof_et)

    print(f"\n{'='*55}")
    print(f"OOF CatBoost      AUC={oof_cb_auc:.4f}  PR-AUC={oof_cb_ap:.4f}")
    print(f"OOF LightGBM      AUC={oof_lgbm_auc:.4f}  PR-AUC={oof_lgbm_ap:.4f}")
    print(f"OOF XGBoost       AUC={oof_xgb_auc:.4f}  PR-AUC={oof_xgb_ap:.4f}")
    print(f"OOF ExtraTrees    AUC={oof_et_auc:.4f}  PR-AUC={oof_et_ap:.4f}")

    # 5-branch meta-learner with logit transform + isotonic calibration.
    # Logit transform (log-odds) makes the meta-feature space linear for LR:
    # if each branch is a calibrated probability estimate, the log-odds of each
    # branch are approximately additive — making LR in logit space optimal.
    # Clipping [1e-6, 1-1e-6] prevents -inf/+inf for saturated branch outputs.
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return np.log(p / (1.0 - p))

    oof_matrix = np.column_stack([_logit(oof_cb), _logit(oof_lgbm), _logit(oof_xgb), _logit(oof_et)])
    # class_weight={0:1, 1:15} ≈ sqrt(38) × base: expands score dynamic range without
    # over-weighting noisy positives. Fixes score compression (max was 0.3656 without this).
    base_meta = LogisticRegression(C=1.0, max_iter=500, random_state=42, class_weight={0: 1, 1: 15})
    base_meta.fit(oof_matrix, y_train)

    # Evaluate raw LR stacker
    oof_lr_preds = base_meta.predict_proba(oof_matrix)[:, 1]
    oof_lr_auc = roc_auc_score(y_train, oof_lr_preds)
    oof_lr_ap  = average_precision_score(y_train, oof_lr_preds)
    print(f"OOF Stacker (LR)      AUC={oof_lr_auc:.4f}  PR-AUC={oof_lr_ap:.4f}")

    # Isotonic calibration on OOF predictions (leave-one-fold calibration
    # is nested inside StratifiedGroupKFold via cv='prefit', which uses the
    # already-OOF predictions — correct since oof_matrix has no training leakage).
    meta = (CalibratedClassifierCV(FrozenEstimator(base_meta), method="isotonic")
            if _HAS_FROZEN else
            CalibratedClassifierCV(base_meta, method="isotonic", cv="prefit"))
    meta.fit(oof_matrix, y_train)
    oof_stacker_preds = meta.predict_proba(oof_matrix)[:, 1]
    oof_stacker_auc = roc_auc_score(y_train, oof_stacker_preds)
    oof_stacker_ap  = average_precision_score(y_train, oof_stacker_preds)
    print(f"OOF Stacker (calibrated) AUC={oof_stacker_auc:.4f}  PR-AUC={oof_stacker_ap:.4f}")
    print(f"  Score range: [{oof_stacker_preds.min():.4f}, {oof_stacker_preds.max():.4f}]  "
          f"p99={np.percentile(oof_stacker_preds, 99):.4f}")

    # ── OOF F1 / Precision / Recall at multiple thresholds (leakage-free) ─────
    prevalence = y_train.mean()
    prec_curve, rec_curve, thr_curve = precision_recall_curve(y_train, oof_stacker_preds)
    f1_curve = np.where(
        (prec_curve[:-1] + rec_curve[:-1]) > 0,
        2 * prec_curve[:-1] * rec_curve[:-1] / (prec_curve[:-1] + rec_curve[:-1]),
        0,
    )
    best_idx  = int(np.argmax(f1_curve))
    best_thr  = float(thr_curve[best_idx])
    best_f1   = float(f1_curve[best_idx])
    best_prec = float(prec_curve[best_idx])
    best_rec  = float(rec_curve[best_idx])

    # OOF F2 curve (beta=2: recall weighted 4x more than precision, appropriate for AML)
    f2_curve = np.where(
        (4 * prec_curve[:-1] + rec_curve[:-1]) > 0,
        (1 + 4) * prec_curve[:-1] * rec_curve[:-1] / (4 * prec_curve[:-1] + rec_curve[:-1]),
        0,
    )
    best_f2_idx = int(np.argmax(f2_curve))
    best_f2_thr = float(thr_curve[best_f2_idx])
    best_f2     = float(f2_curve[best_f2_idx])
    print(f"  Optimal-F2 @ threshold={best_f2_thr:.4f}: F2={best_f2:.4f}")

    print(f"\nOOF Threshold sweep (stacker calibrated):")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Flagged':>8}")
    for t in [0.10, 0.20, 0.30, 0.40, 0.50]:
        pred = (oof_stacker_preds >= t).astype(int)
        print(f"{t:>10.2f} "
              f"{precision_score(y_train, pred, zero_division=0):>10.4f} "
              f"{recall_score(y_train, pred, zero_division=0):>8.4f} "
              f"{f1_score(y_train, pred, zero_division=0):>8.4f} "
              f"{pred.sum():>8,}")
    print(f"  Optimal-F1 @ threshold={best_thr:.4f}: "
          f"P={best_prec:.4f}  R={best_rec:.4f}  F1={best_f1:.4f}  "
          f"Lift={best_prec/prevalence:.1f}x")
    tn, fp, fn, tp = confusion_matrix(y_train, (oof_stacker_preds >= best_thr).astype(int)).ravel()
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"{'='*55}")

    # ── Full-label propagation for final model retraining ─────────────────────
    # Use all training labels (no fold split) — appropriate since final models
    # are trained on all data and will be used for inference, not CV evaluation.
    _full_labels = pd.Series(y_train, index=train_df["user_id"].values)
    _full_prop_df = compute_label_propagation(entity_edges, _full_labels, all_user_ids)
    _full_prop_df = (
        _full_prop_df.set_index("user_id")
        .reindex(train_df["user_id"])
        .fillna(0)
        .reset_index(drop=True)
    )
    _full_prop_arr = _full_prop_df[_PROP_COLS].values.astype("float32")
    x_train_df_full = pd.concat(
        [x_train_df, pd.DataFrame(_full_prop_arr, columns=_PROP_COLS)], axis=1
    )
    x_train_np_full = np.hstack([x_train_np, _full_prop_arr])

    # Retrain final models on all training data with more iterations
    pos_all = max(1, int(y_train.sum()))
    neg_all = max(1, len(y_train) - pos_all)

    final_cb = CatBoostClassifier(
        iterations=n_estimators, learning_rate=0.05, depth=6,
        scale_pos_weight=neg_all / pos_all, cat_features=cat_indices,
        l2_leaf_reg=5, random_seed=42, verbose=0,
        **catboost_runtime_params(),
    )
    final_cb.fit(x_train_df_full, y_train)

    final_lgbm = LGBMClassifier(
        n_estimators=n_estimators, learning_rate=0.05, num_leaves=63,
        min_child_samples=5, reg_alpha=0.1, reg_lambda=1.0,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=neg_all / pos_all, random_state=42, verbose=-1,
        **lightgbm_runtime_params(),
    )
    final_lgbm.fit(x_train_df_full, y_train)

    final_xgb = XGBClassifier(
        n_estimators=n_estimators, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, gamma=0.1,
        reg_alpha=0.05, reg_lambda=1.0,
        scale_pos_weight=neg_all / pos_all, random_state=42,
        eval_metric="logloss", verbosity=0,
        **xgboost_runtime_params(),
    )
    final_xgb.fit(x_train_np_full, y_train)

    final_et = ExtraTreesClassifier(
        n_estimators=max(50, n_estimators // 2 + n_estimators // 4), max_depth=10, min_samples_leaf=3,
        class_weight={0: 1, 1: neg_all / pos_all},
        random_state=42, n_jobs=sklearn_n_jobs(),
    )
    final_et.fit(x_train_np_full, y_train)

    # Refit calibrated meta-learner on full training data using all four branch predictions.
    # Apply the same logit transform as the OOF meta-learner — the final model must
    # receive log-odds inputs at fit time to match what score.py sends at inference time.
    # Without this, the final meta-learner would be trained in probability space but
    # called in logit space, causing a systematic prediction error.
    full_branch_probs = np.column_stack([
        final_cb.predict_proba(x_train_df_full)[:, 1],
        final_lgbm.predict_proba(x_train_df_full)[:, 1],
        final_xgb.predict_proba(x_train_np_full)[:, 1],
        final_et.predict_proba(x_train_np_full)[:, 1],
    ])
    full_branch_matrix = np.column_stack([_logit(full_branch_probs[:, i]) for i in range(4)])
    final_base_meta = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    final_base_meta.fit(full_branch_matrix, y_train)
    final_meta = (CalibratedClassifierCV(FrozenEstimator(final_base_meta), method="isotonic")
                  if _HAS_FROZEN else
                  CalibratedClassifierCV(final_base_meta, method="isotonic", cv="prefit"))
    final_meta.fit(full_branch_matrix, y_train)

    now_str   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version   = f"stacker_{now_str}"
    mdir      = model_dir()
    cb_path   = mdir / f"cb_{now_str}.joblib"
    lgbm_path = mdir / f"lgbm_v2_{now_str}.joblib"
    xgb_path  = mdir / f"xgb_{now_str}.joblib"
    et_path   = mdir / f"et_{now_str}.joblib"
    meta_path = mdir / f"{version}.joblib"

    save_joblib(final_cb,   cb_path)
    save_joblib(final_lgbm, lgbm_path)
    save_joblib(final_xgb,  xgb_path)
    save_joblib(final_et,   et_path)
    save_joblib(final_meta, meta_path)

    # Compute OOF F1/precision/recall metrics for cv_results persistence
    _oof_thresh_sweep = []
    for _t in [0.10, 0.20, 0.30, 0.40, 0.50]:
        _pred = (oof_stacker_preds >= _t).astype(int)
        _oof_thresh_sweep.append({
            "threshold": _t,
            "precision": round(float(precision_score(y_train, _pred, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_train, _pred, zero_division=0)), 4),
            "f1":        round(float(f1_score(y_train, _pred, zero_division=0)), 4),
            "n_flagged": int(_pred.sum()),
        })
    _tn, _fp, _fn, _tp = confusion_matrix(
        y_train, (oof_stacker_preds >= best_thr).astype(int)
    ).ravel()

    cv_summary = {
        "n_folds": n_folds,
        "folds": fold_metrics,
        "oof": {
            "catboost":   {"auc": round(oof_cb_auc, 4),      "pr_auc": round(oof_cb_ap, 4)},
            "lgbm":       {"auc": round(oof_lgbm_auc, 4),    "pr_auc": round(oof_lgbm_ap, 4)},
            "xgboost":    {"auc": round(oof_xgb_auc, 4),     "pr_auc": round(oof_xgb_ap, 4)},
            "extratrees": {"auc": round(oof_et_auc, 4),      "pr_auc": round(oof_et_ap, 4)},
            "stacker":    {"auc": round(oof_stacker_auc, 4), "pr_auc": round(oof_stacker_ap, 4)},
        },
        "oof_threshold": {
            "optimal_threshold": round(best_thr, 4),
            "optimal_f1":        round(best_f1, 4),
            "optimal_f2_threshold": round(best_f2_thr, 4),
            "optimal_f2":            round(best_f2, 4),
            "optimal_precision": round(best_prec, 4),
            "optimal_recall":    round(best_rec, 4),
            "lift":              round(best_prec / max(prevalence, 1e-9), 2),
            "tp": int(_tp), "fp": int(_fp), "fn": int(_fn), "tn": int(_tn),
            "threshold_sweep": _oof_thresh_sweep,
        },
        "fold_mean": {
            "catboost_auc":   round(float(np.mean([f["catboost"]["auc"]   for f in fold_metrics])), 4),
            "catboost_std":   round(float(np.std( [f["catboost"]["auc"]   for f in fold_metrics])), 4),
            "lgbm_auc":       round(float(np.mean([f["lgbm"]["auc"]       for f in fold_metrics])), 4),
            "lgbm_std":       round(float(np.std( [f["lgbm"]["auc"]       for f in fold_metrics])), 4),
            "xgboost_auc":    round(float(np.mean([f["xgboost"]["auc"]    for f in fold_metrics])), 4),
            "xgboost_std":    round(float(np.std( [f["xgboost"]["auc"]    for f in fold_metrics])), 4),
            "extratrees_auc": round(float(np.mean([f["extratrees"]["auc"] for f in fold_metrics])), 4),
            "extratrees_std": round(float(np.std( [f["extratrees"]["auc"] for f in fold_metrics])), 4),
        },
    }
    print(f"\nFold mean CatBoost      AUC: {cv_summary['fold_mean']['catboost_auc']:.4f} ± {cv_summary['fold_mean']['catboost_std']:.4f}")
    print(f"Fold mean LightGBM      AUC: {cv_summary['fold_mean']['lgbm_auc']:.4f} ± {cv_summary['fold_mean']['lgbm_std']:.4f}")
    print(f"Fold mean XGBoost       AUC: {cv_summary['fold_mean']['xgboost_auc']:.4f} ± {cv_summary['fold_mean']['xgboost_std']:.4f}")
    print(f"Fold mean ExtraTrees    AUC: {cv_summary['fold_mean']['extratrees_auc']:.4f} ± {cv_summary['fold_mean']['extratrees_std']:.4f}")

    meta_dict = {
        "stacker_version": version,
        "feature_columns": feature_cols,
        "branch_models": {
            "catboost":   str(cb_path),
            "lgbm":       str(lgbm_path),
            "xgboost":    str(xgb_path),
            "extratrees": str(et_path),
        },
        "stacker_path": str(meta_path),
        "meta_coefs": base_meta.coef_.tolist(),
        "meta_input_transform": "logit",  # branches transformed to log-odds before LR
        "cv_results": cv_summary,
    }
    save_json(meta_dict, meta_path.with_suffix(".json"))

    # Also save a standalone CV results file for easy access
    cv_path = mdir / f"cv_results_{now_str}.json"
    save_json(cv_summary, cv_path)
    print(f"\nCV results saved to: {cv_path}")

    return {
        "stacker_version": version,
        "stacker_path":    str(meta_path),
        "branch_models":   meta_dict["branch_models"],
        "cv_results":      cv_summary,
    }


if __name__ == "__main__":
    print(train_stacker())
