from __future__ import annotations

from datetime import datetime, timezone

from lightgbm import LGBMClassifier

from hardware import describe_hardware, lightgbm_runtime_params
from models.common import encode_features, feature_columns, model_dir, save_json, save_pickle, training_dataset


def train_model() -> dict:
    print(f"[train_model] runtime: {describe_hardware()}")
    dataset = training_dataset().sort_values("snapshot_date").reset_index(drop=True)
    feature_cols = feature_columns(dataset)
    unique_dates = sorted(dataset["snapshot_date"].dt.date.unique())
    train_dates = set(unique_dates[:20])
    valid_dates = set(unique_dates[20:25])
    holdout_dates = set(unique_dates[25:])

    train_frame = dataset[dataset["snapshot_date"].dt.date.isin(train_dates)].copy()
    valid_frame = dataset[dataset["snapshot_date"].dt.date.isin(valid_dates)].copy()
    holdout_frame = dataset[dataset["snapshot_date"].dt.date.isin(holdout_dates)].copy()

    x_train, encoded_columns = encode_features(train_frame, feature_cols)
    x_valid, _ = encode_features(valid_frame, feature_cols, reference_columns=encoded_columns)
    x_holdout, _ = encode_features(holdout_frame, feature_cols, reference_columns=encoded_columns)
    y_train = train_frame["hidden_suspicious_label"]
    y_valid = valid_frame["hidden_suspicious_label"]

    positives = max(1, int(y_train.sum()))
    negatives = max(1, len(y_train) - positives)
    model = LGBMClassifier(
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        scale_pos_weight=negatives / positives,
        **lightgbm_runtime_params(),
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="binary_logloss",
        callbacks=[],
    )

    version = f"lgbm_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    model_path = model_dir() / f"{version}.pkl"
    meta_path = model_dir() / f"{version}.json"
    save_pickle(model, model_path)
    save_json(
        {
            "model_version": version,
            "feature_columns": feature_cols,
            "encoded_columns": encoded_columns,
            "train_dates": sorted(str(d) for d in train_dates),
            "valid_dates": sorted(str(d) for d in valid_dates),
            "holdout_dates": sorted(str(d) for d in holdout_dates),
            "holdout_rows": len(x_holdout),
        },
        meta_path,
    )
    return {"model_version": version, "model_path": str(model_path), "meta_path": str(meta_path)}


if __name__ == "__main__":
    print(train_model())
