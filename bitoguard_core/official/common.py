from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import load_settings


RANDOM_SEED = 42
PRIMARY_KEY_COLUMNS = {
    "user_info": "user_id",
    "train_label": "user_id",
    "predict_label": "user_id",
    "twd_transfer": "id",
    "crypto_transfer": "id",
    "usdt_swap": "id",
    "usdt_twd_trading": "id",
}
OFFICIAL_TABLES = tuple(PRIMARY_KEY_COLUMNS)
EVENT_TIME_COLUMNS = {
    "twd_transfer": "created_at",
    "crypto_transfer": "created_at",
    "usdt_swap": "created_at",
    "usdt_twd_trading": "updated_at",
}
VALIDATION_THRESHOLDS = [round(x, 2) for x in np.arange(0.30, 0.81, 0.05).tolist()]


@dataclass(frozen=True)
class OfficialPaths:
    raw_dir: Path
    clean_dir: Path
    artifact_dir: Path
    model_dir: Path
    report_dir: Path
    prediction_dir: Path
    feature_dir: Path
    bundle_path: Path


def load_official_paths() -> OfficialPaths:
    settings = load_settings()
    model_dir = settings.artifact_dir / "models"
    report_dir = settings.artifact_dir / "reports"
    prediction_dir = settings.artifact_dir / "predictions"
    feature_dir = settings.artifact_dir / "official_features"
    for path in (model_dir, report_dir, prediction_dir, feature_dir):
        path.mkdir(parents=True, exist_ok=True)
    return OfficialPaths(
        raw_dir=settings.aws_event_raw_dir,
        clean_dir=settings.aws_event_clean_dir,
        artifact_dir=settings.artifact_dir,
        model_dir=model_dir,
        report_dir=report_dir,
        prediction_dir=prediction_dir,
        feature_dir=feature_dir,
        bundle_path=settings.artifact_dir / "official_bundle.json",
    )


def require_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def load_clean_table(name: str) -> pd.DataFrame:
    paths = load_official_paths()
    return require_parquet(paths.clean_dir / f"{name}.parquet")


def load_raw_table(name: str) -> pd.DataFrame:
    paths = load_official_paths()
    return require_parquet(paths.raw_dir / f"{name}.parquet")


def save_json(data: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_pickle(obj: Any, path: Path) -> None:
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def to_utc_timestamp(value: object | None) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def encode_frame(
    frame: pd.DataFrame,
    columns: list[str],
    reference_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    encoded = pd.get_dummies(frame[columns].copy(), dummy_na=True)
    encoded.columns = _unique_feature_names([_sanitize_feature_name(column) for column in encoded.columns])
    if reference_columns is not None:
        encoded = encoded.reindex(columns=reference_columns, fill_value=0)
        return encoded, reference_columns
    return encoded, list(encoded.columns)


def _sanitize_feature_name(value: object) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", str(value))
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "feature"


def _unique_feature_names(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique: list[str] = []
    for column in columns:
        count = seen.get(column, 0)
        if count == 0:
            unique.append(column)
        else:
            unique.append(f"{column}_{count}")
        seen[column] = count + 1
    return unique


def list_event_cutoffs() -> tuple[pd.Timestamp, pd.Timestamp]:
    timestamps: list[pd.Series] = []
    for table_name, column_name in EVENT_TIME_COLUMNS.items():
        frame = load_clean_table(table_name)
        timestamps.append(pd.to_datetime(frame[column_name], utc=True, errors="coerce"))
    combined = pd.concat(timestamps, ignore_index=True).dropna()
    return combined.min(), combined.max()


def default_temporal_cutoff() -> pd.Timestamp:
    start, end = list_event_cutoffs()
    preferred = end - timedelta(days=7)
    if preferred <= start:
        span = end - start
        return start + (span * 0.75)
    return preferred


def feature_output_path(name: str, cutoff_tag: str = "full") -> Path:
    return load_official_paths().feature_dir / f"{name}_{cutoff_tag}.parquet"


def feature_report_path(name: str) -> Path:
    return load_official_paths().report_dir / name


def prediction_output_path(name: str) -> Path:
    return load_official_paths().prediction_dir / name


def bundle_file_path(path: Path | None = None) -> Path:
    if path is not None:
        return path
    return load_official_paths().bundle_path
