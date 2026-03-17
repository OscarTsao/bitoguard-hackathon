# bitoguard_core/models/train_catboost.py
from __future__ import annotations
from datetime import datetime, timezone

from catboost import CatBoostClassifier

from config import load_settings
from db.store import DuckDBStore
from hardware import catboost_runtime_params, describe_hardware
from models.common import (
    NON_FEATURE_COLUMNS, forward_date_splits, model_dir,
    save_joblib, save_json,
)

_V2_TABLE = "features.feature_snapshots_v2"
CAT_FEATURE_NAMES = frozenset({
    "kyc_level_code", "occupation_code", "income_source_code", "user_source_code",
})


def load_v2_training_dataset() -> "pd.DataFrame":
    import pandas as pd
    settings = load_settings()
    store    = DuckDBStore(settings.db_path)
    dataset  = store.fetch_df(f"""
        WITH ped AS (
            SELECT user_id, CAST(MIN(observed_at) AS DATE) AS ped
            FROM canonical.blacklist_feed
            WHERE observed_at IS NOT NULL
            GROUP BY user_id
        )
        SELECT f.*,
               COALESCE(l.hidden_suspicious_label, 0) AS hidden_suspicious_label
        FROM {_V2_TABLE} f
        LEFT JOIN ops.oracle_user_labels l ON f.user_id = l.user_id
        LEFT JOIN ped ON f.user_id = ped.user_id
        WHERE COALESCE(l.hidden_suspicious_label, 0) = 0
           OR (ped.ped IS NOT NULL AND f.snapshot_date >= ped.ped)
    """)
    dataset["snapshot_date"] = pd.to_datetime(dataset["snapshot_date"])
    dataset["hidden_suspicious_label"] = dataset["hidden_suspicious_label"].astype(int)
    return dataset.sort_values("snapshot_date").reset_index(drop=True)


def train_catboost_model() -> dict:
    import pandas as pd
    print(f"[train_catboost_model] runtime: {describe_hardware()}")
    dataset      = load_v2_training_dataset()
    feature_cols = [c for c in dataset.columns
                    if c not in NON_FEATURE_COLUMNS and c != "hidden_suspicious_label"]
    cat_indices  = [i for i, c in enumerate(feature_cols) if c in CAT_FEATURE_NAMES]
    date_splits  = forward_date_splits(dataset["snapshot_date"])
    train_dates  = set(date_splits["train"])

    train   = dataset[dataset["snapshot_date"].dt.date.isin(train_dates)].copy()
    x_train = train[feature_cols].fillna(0)
    y_train = train["hidden_suspicious_label"]
    pos     = max(1, int(y_train.sum()))
    neg     = max(1, len(y_train) - pos)

    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        scale_pos_weight=neg / pos,
        cat_features=cat_indices,
        random_seed=42,
        verbose=0,
        **catboost_runtime_params(),
    )
    model.fit(x_train, y_train)

    version    = f"catboost_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    model_path = model_dir() / f"{version}.joblib"
    save_joblib(model, model_path)
    save_json(
        {"model_version": version, "feature_columns": feature_cols,
         "cat_features": [feature_cols[i] for i in cat_indices]},
        model_path.with_suffix(".json"),
    )
    return {"model_version": version, "model_path": str(model_path)}


if __name__ == "__main__":
    print(train_catboost_model())
