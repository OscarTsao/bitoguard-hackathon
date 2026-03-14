from __future__ import annotations

from datetime import datetime, timezone

from sklearn.ensemble import IsolationForest

from models.common import encode_features, feature_columns, forward_date_splits, model_dir, save_iforest, save_json, training_dataset


def train_anomaly_model() -> dict:
    dataset = training_dataset().sort_values("snapshot_date").reset_index(drop=True)
    feature_cols = feature_columns(dataset)
    train_dates = set(forward_date_splits(dataset["snapshot_date"])["train"])
    train_frame = dataset[dataset["snapshot_date"].dt.date.isin(train_dates)].copy()
    x_train, encoded_columns = encode_features(train_frame, feature_cols)
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,  # fixed domain estimate; must not be derived from labels
        random_state=42,
    )
    model.fit(x_train)
    version = f"iforest_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    model_path = model_dir() / f"{version}.joblib"
    meta_path = model_dir() / f"{version}.json"
    save_iforest(model, model_path)
    save_json(
        {
            "model_version": version,
            "feature_columns": feature_cols,
            "encoded_columns": encoded_columns,
            "train_dates": sorted(str(d) for d in train_dates),
        },
        meta_path,
    )
    return {"model_version": version, "model_path": str(model_path), "meta_path": str(meta_path)}


if __name__ == "__main__":
    print(train_anomaly_model())
