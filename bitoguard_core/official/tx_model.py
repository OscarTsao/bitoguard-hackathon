"""Transaction-level LightGBM classifier.

Training: User labels propagated to transactions (all tx from positive user = positive).
Prediction: Per-transaction probability -> aggregated to user-level score.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score


def train_tx_model_oof(
    tx_features: pd.DataFrame,
    feature_cols: list[str],
    labels: pd.DataFrame,
    split_frame: pd.DataFrame,
    fold_column: str = "primary_fold",
) -> pd.DataFrame:
    """Train transaction-level LightGBM 5-fold OOF, return user-level scores."""

    # Propagate user labels to transactions
    tx = tx_features.merge(labels[["user_id", "status"]], on="user_id", how="inner")
    tx["label"] = tx["status"].astype(int)

    # Get fold assignments (user-level -> transaction inherits user's fold)
    tx = tx.merge(split_frame[["user_id", fold_column]], on="user_id", how="left")

    user_results = {}

    folds = sorted(int(v) for v in tx[fold_column].dropna().unique())
    for fold_id in folds:
        train_tx = tx[tx[fold_column] != fold_id]
        valid_tx = tx[tx[fold_column] == fold_id]

        if len(train_tx) == 0 or len(valid_tx) == 0:
            continue

        n_pos = int(train_tx["label"].sum())
        n_neg = len(train_tx) - n_pos
        scale = min(n_neg / max(n_pos, 1), 20.0)

        model = LGBMClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale,
            verbose=-1,
            n_jobs=-1,
        )

        X_train = train_tx[feature_cols].values
        y_train = train_tx["label"].values
        X_valid = valid_tx[feature_cols].values

        model.fit(X_train, y_train)
        valid_tx = valid_tx.copy()
        valid_tx["tx_risk"] = model.predict_proba(X_valid)[:, 1]

        user_agg = valid_tx.groupby("user_id").agg(
            max_tx_risk=("tx_risk", "max"),
            mean_tx_risk=("tx_risk", "mean"),
            p95_tx_risk=("tx_risk", lambda x: np.percentile(x, 95)),
            n_high_risk_tx=("tx_risk", lambda x: (x > 0.5).sum()),
            n_tx=("tx_risk", "count"),
        ).reset_index()
        user_agg["high_risk_tx_ratio"] = user_agg["n_high_risk_tx"] / user_agg["n_tx"].clip(lower=1)

        fold_users = user_agg.merge(labels[["user_id", "status"]], on="user_id")
        ap = average_precision_score(fold_users["status"].astype(int), fold_users["max_tx_risk"])
        print(f"  [TX fold {fold_id}] AP(max_risk)={ap:.4f}, "
              f"n_tx_train={len(train_tx)}, n_tx_valid={len(valid_tx)}, "
              f"n_users_valid={len(user_agg)}")

        for _, row in user_agg.iterrows():
            user_results[int(row["user_id"])] = {
                "max_tx_risk": float(row["max_tx_risk"]),
                "mean_tx_risk": float(row["mean_tx_risk"]),
                "p95_tx_risk": float(row["p95_tx_risk"]),
                "high_risk_tx_ratio": float(row["high_risk_tx_ratio"]),
            }

    result = pd.DataFrame([
        {"user_id": uid, **scores}
        for uid, scores in user_results.items()
    ])

    if not result.empty:
        result_eval = result.merge(labels[["user_id", "status"]], on="user_id", how="inner")
        overall_ap = average_precision_score(result_eval["status"].astype(int), result_eval["max_tx_risk"])
        print(f"[TX model] Overall OOF AP(max_risk)={overall_ap:.4f}, {len(result)} users")

    return result
