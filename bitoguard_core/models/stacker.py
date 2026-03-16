# bitoguard_core/models/stacker.py
"""Stacker: CatBoost + LightGBM OOF branches -> Logistic Regression meta-learner.

OOF models train on the same label-free feature set as the deployed final models
so that OOF metrics reflect real-world generalization.

Note: Branch C (graph label propagation) causes self-referential leakage in the
OOF loop (training positives propagate their own labels back to themselves via
shared entities). Branch C wiring into score.py is a tracked follow-up task.
"""
from __future__ import annotations
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from models.common import (
    NON_FEATURE_COLUMNS, forward_date_splits, model_dir,
    save_joblib, save_json,
)
from models.train_catboost import load_v2_training_dataset, CAT_FEATURE_NAMES


def train_stacker(n_folds: int = 5) -> dict:
    """OOF stacking: CatBoost + LightGBM branches -> Logistic Regression meta-learner."""
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

    print(f"Training set: {len(y_train):,} samples, {int(y_train.sum()):,} positives ({y_train.mean():.2%})")
    print(f"Features: {len(feature_cols)}, Cat features: {len(cat_indices)}")

    oof_cb   = np.zeros(len(x_train_df))
    oof_lgbm = np.zeros(len(x_train_df))

    fold_metrics: list[dict] = []
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold_i, (tr_idx, val_idx) in enumerate(sgkf.split(x_train_df, y_train, groups=groups), 1):
        pos = max(1, int(y_train[tr_idx].sum()))
        neg = max(1, len(tr_idx) - pos)
        print(f"\n[Fold {fold_i}/{n_folds}] train={len(tr_idx):,} val={len(val_idx):,} pos_rate={pos/len(tr_idx):.2%}")

        # OOF models use the same label-free feature set as the deployed final models.
        # Branch C (label propagation) causes self-referential leakage in the OOF loop:
        # training positives propagate their labels through shared entities back to
        # themselves, creating a spurious prop_ip≈1.0 signal that vanishes at val time.
        # Branch C wiring into score.py is a separate TODO.
        cb = CatBoostClassifier(
            iterations=200, learning_rate=0.05, depth=6,
            scale_pos_weight=neg / pos, cat_features=cat_indices,
            random_seed=42, verbose=0,
        )
        cb.fit(x_train_df.iloc[tr_idx], y_train[tr_idx])
        oof_cb[val_idx] = cb.predict_proba(x_train_df.iloc[val_idx])[:, 1]
        cb_auc = roc_auc_score(y_train[val_idx], oof_cb[val_idx])
        cb_ap  = average_precision_score(y_train[val_idx], oof_cb[val_idx])
        print(f"  CatBoost   AUC={cb_auc:.4f}  PR-AUC={cb_ap:.4f}")

        lgbm = LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=neg / pos, random_state=42,
        )
        lgbm.fit(x_train_df.iloc[tr_idx], y_train[tr_idx])
        oof_lgbm[val_idx] = lgbm.predict_proba(x_train_df.iloc[val_idx])[:, 1]
        lgbm_auc = roc_auc_score(y_train[val_idx], oof_lgbm[val_idx])
        lgbm_ap  = average_precision_score(y_train[val_idx], oof_lgbm[val_idx])
        print(f"  LightGBM   AUC={lgbm_auc:.4f}  PR-AUC={lgbm_ap:.4f}")

        fold_metrics.append({
            "fold": fold_i,
            "n_train": int(len(tr_idx)),
            "n_val": int(len(val_idx)),
            "catboost": {"auc": round(cb_auc, 4), "pr_auc": round(cb_ap, 4)},
            "lgbm": {"auc": round(lgbm_auc, 4), "pr_auc": round(lgbm_ap, 4)},
        })

    # OOF metrics across all folds
    oof_cb_auc   = roc_auc_score(y_train, oof_cb)
    oof_lgbm_auc = roc_auc_score(y_train, oof_lgbm)
    oof_cb_ap    = average_precision_score(y_train, oof_cb)
    oof_lgbm_ap  = average_precision_score(y_train, oof_lgbm)

    print(f"\n{'='*50}")
    print(f"OOF CatBoost  AUC={oof_cb_auc:.4f}  PR-AUC={oof_cb_ap:.4f}")
    print(f"OOF LightGBM  AUC={oof_lgbm_auc:.4f}  PR-AUC={oof_lgbm_ap:.4f}")

    # Train meta-learner on OOF predictions
    oof_matrix = np.column_stack([oof_cb, oof_lgbm])
    meta = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    meta.fit(oof_matrix, y_train)
    oof_stacker_preds = meta.predict_proba(oof_matrix)[:, 1]
    oof_stacker_auc = roc_auc_score(y_train, oof_stacker_preds)
    oof_stacker_ap  = average_precision_score(y_train, oof_stacker_preds)
    print(f"OOF Stacker   AUC={oof_stacker_auc:.4f}  PR-AUC={oof_stacker_ap:.4f}")
    print(f"{'='*50}")

    # Retrain final models on all training data with more iterations
    pos_all = max(1, int(y_train.sum()))
    neg_all = max(1, len(y_train) - pos_all)

    final_cb = CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=6,
        scale_pos_weight=neg_all / pos_all, cat_features=cat_indices,
        random_seed=42, verbose=0,
    )
    final_cb.fit(x_train_df, y_train)

    final_lgbm = LGBMClassifier(
        n_estimators=250, learning_rate=0.05, num_leaves=31,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=neg_all / pos_all, random_state=42,
    )
    final_lgbm.fit(x_train_df, y_train)

    now_str   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version   = f"stacker_{now_str}"
    mdir      = model_dir()
    cb_path   = mdir / f"cb_{now_str}.joblib"
    lgbm_path = mdir / f"lgbm_v2_{now_str}.joblib"
    meta_path = mdir / f"{version}.joblib"

    save_joblib(final_cb,   cb_path)
    save_joblib(final_lgbm, lgbm_path)
    save_joblib(meta,       meta_path)

    cv_summary = {
        "n_folds": n_folds,
        "folds": fold_metrics,
        "oof": {
            "catboost": {"auc": round(oof_cb_auc, 4), "pr_auc": round(oof_cb_ap, 4)},
            "lgbm":     {"auc": round(oof_lgbm_auc, 4), "pr_auc": round(oof_lgbm_ap, 4)},
            "stacker":  {"auc": round(oof_stacker_auc, 4), "pr_auc": round(oof_stacker_ap, 4)},
        },
        "fold_mean": {
            "catboost_auc":  round(float(np.mean([f["catboost"]["auc"]  for f in fold_metrics])), 4),
            "catboost_std":  round(float(np.std( [f["catboost"]["auc"]  for f in fold_metrics])), 4),
            "lgbm_auc":      round(float(np.mean([f["lgbm"]["auc"]      for f in fold_metrics])), 4),
            "lgbm_std":      round(float(np.std( [f["lgbm"]["auc"]      for f in fold_metrics])), 4),
        },
    }
    print(f"\nFold mean CatBoost AUC: {cv_summary['fold_mean']['catboost_auc']:.4f} ± {cv_summary['fold_mean']['catboost_std']:.4f}")
    print(f"Fold mean LightGBM AUC: {cv_summary['fold_mean']['lgbm_auc']:.4f} ± {cv_summary['fold_mean']['lgbm_std']:.4f}")

    meta_dict = {
        "stacker_version": version,
        "feature_columns": feature_cols,
        "branch_models": {"catboost": str(cb_path), "lgbm": str(lgbm_path)},
        "stacker_path": str(meta_path),
        "meta_coefs": meta.coef_.tolist(),
        "cv_results": cv_summary,
    }
    save_json(meta_dict, meta_path.with_suffix(".json"))

    # Also save a standalone CV results file for easy access
    cv_path = mdir / f"cv_results_{now_str}.json"
    save_json(cv_summary, cv_path)
    print(f"\nCV results saved to: {cv_path}")

    return {
        "stacker_version": version,
        "stacker_path": str(meta_path),
        "branch_models": meta_dict["branch_models"],
        "cv_results": cv_summary,
    }


if __name__ == "__main__":
    print(train_stacker())
