# bitoguard_core/models/stacker.py
"""Stacker: CatBoost + LightGBM OOF branches -> Logistic Regression meta-learner.

LEAKAGE CONTRACT: graph propagation features (if used) must be computed
inside each fold using ONLY training-fold labels. See graph_propagation.py.
This stacker does not compute propagation features itself (v1 stacker
uses only label-free v2 features). Extend later by adding Branch C.
"""
from __future__ import annotations
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from models.common import (
    NON_FEATURE_COLUMNS, forward_date_splits, model_dir,
    save_joblib, save_json,
)
from models.train_catboost import _load_v2_training_dataset, _CAT_FEATURE_NAMES


def train_stacker(n_folds: int = 5) -> dict:
    """OOF stacking: CatBoost + LightGBM branches -> LR meta-learner."""
    dataset = _load_v2_training_dataset()
    feature_cols = [c for c in dataset.columns
                    if c not in NON_FEATURE_COLUMNS and c != "hidden_suspicious_label"]
    cat_indices  = [i for i, c in enumerate(feature_cols) if c in _CAT_FEATURE_NAMES]

    date_splits   = forward_date_splits(dataset["snapshot_date"])
    train_dates   = set(date_splits["train"])
    train_df      = dataset[dataset["snapshot_date"].dt.date.isin(train_dates)].copy()
    # Keep as DataFrame so CatBoost can handle categorical columns by name/index;
    # LightGBM receives the same DataFrame (it handles mixed types fine).
    groups        = train_df["user_id"].reset_index(drop=True).values
    x_train_df    = train_df[feature_cols].fillna(0).reset_index(drop=True)
    y_train       = train_df["hidden_suspicious_label"].values

    oof_cb   = np.zeros(len(x_train_df))
    oof_lgbm = np.zeros(len(x_train_df))

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for tr_idx, val_idx in sgkf.split(x_train_df, y_train, groups=groups):
        pos = max(1, int(y_train[tr_idx].sum()))
        neg = max(1, len(tr_idx) - pos)

        cb = CatBoostClassifier(
            iterations=200, learning_rate=0.05, depth=6,
            scale_pos_weight=neg / pos, cat_features=cat_indices,
            random_seed=42, verbose=0,
        )
        cb.fit(x_train_df.iloc[tr_idx], y_train[tr_idx])
        oof_cb[val_idx] = cb.predict_proba(x_train_df.iloc[val_idx])[:, 1]

        lgbm = LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=neg / pos, random_state=42,
        )
        lgbm.fit(x_train_df.iloc[tr_idx], y_train[tr_idx])
        oof_lgbm[val_idx] = lgbm.predict_proba(x_train_df.iloc[val_idx])[:, 1]

    # Train meta-learner on OOF predictions
    oof_matrix = np.column_stack([oof_cb, oof_lgbm])
    meta = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    meta.fit(oof_matrix, y_train)

    # Retrain full base models on all training data
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

    meta_dict = {
        "stacker_version": version,
        "feature_columns": feature_cols,
        "branch_models": {"catboost": str(cb_path), "lgbm": str(lgbm_path)},
        "stacker_path": str(meta_path),
        "meta_coefs": meta.coef_.tolist(),
    }
    save_json(meta_dict, meta_path.with_suffix(".json"))

    return {
        "stacker_version": version,
        "stacker_path": str(meta_path),
        "branch_models": meta_dict["branch_models"],
    }


if __name__ == "__main__":
    print(train_stacker())
