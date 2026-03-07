from __future__ import annotations

from datetime import datetime, timezone

from sklearn.ensemble import IsolationForest

from models.common import encode_features, feature_columns, model_dir, save_json, save_pickle, training_dataset


def train_anomaly_model() -> dict:
    dataset = training_dataset().sort_values("snapshot_date").reset_index(drop=True)
    feature_cols = feature_columns(dataset)
    unique_dates = sorted(dataset["snapshot_date"].dt.date.unique())
    train_dates = set(unique_dates[:20])
    train_frame = dataset[dataset["snapshot_date"].dt.date.isin(train_dates)].copy()
    x_train, encoded_columns = encode_features(train_frame, feature_cols)
    model = IsolationForest(
        n_estimators=200,
        contamination=max(0.01, float(train_frame["hidden_suspicious_label"].mean())),
        random_state=42,
    )
    model.fit(x_train)
    version = f"iforest_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    model_path = model_dir() / f"{version}.pkl"
    meta_path = model_dir() / f"{version}.json"
    save_pickle(model, model_path)
    save_json(
        {"model_version": version, "feature_columns": feature_cols, "encoded_columns": encoded_columns},
        meta_path,
    )
    return {"model_version": version, "model_path": str(model_path), "meta_path": str(meta_path)}


if __name__ == "__main__":
    print(train_anomaly_model())
