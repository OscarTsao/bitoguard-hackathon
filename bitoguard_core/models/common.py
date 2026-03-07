from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from config import load_settings
from db.store import DuckDBStore


NON_FEATURE_COLUMNS = {
    "feature_snapshot_id",
    "user_id",
    "snapshot_date",
    "feature_version",
}


def load_feature_table(table_name: str = "features.feature_snapshots_user_30d") -> pd.DataFrame:
    settings = load_settings()
    store = DuckDBStore(settings.db_path)
    frame = store.read_table(table_name)
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"])
    return frame


def training_dataset() -> pd.DataFrame:
    settings = load_settings()
    store = DuckDBStore(settings.db_path)
    features = load_feature_table("features.feature_snapshots_user_30d")
    labels = store.read_table("ops.oracle_user_labels")
    dataset = features.merge(labels[["user_id", "hidden_suspicious_label", "scenario_types"]], on="user_id", how="left")
    dataset["hidden_suspicious_label"] = dataset["hidden_suspicious_label"].fillna(0).astype(int)
    dataset["scenario_types"] = dataset["scenario_types"].fillna("")
    return dataset


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in NON_FEATURE_COLUMNS | {"hidden_suspicious_label", "scenario_types"}]


def encode_features(frame: pd.DataFrame, columns: list[str], reference_columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    encoded = pd.get_dummies(frame[columns].copy(), dummy_na=True)
    if reference_columns is not None:
        encoded = encoded.reindex(columns=reference_columns, fill_value=0)
        return encoded, reference_columns
    return encoded, list(encoded.columns)


def model_dir() -> Path:
    settings = load_settings()
    path = settings.artifact_dir / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_pickle(obj: Any, path: Path) -> None:
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
