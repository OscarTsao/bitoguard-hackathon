# bitoguard_core/shared/aws_common.py
"""Shared utilities for official/ and transductive_v1/ pipelines.

Both pipelines process the same raw AWS event data (BitoPro) and share
identical data-loading helpers, serialization utilities, and feature-encoding
primitives. This module is the single source of truth for those components.

Re-exported from:
  official/common.py        (was duplicated)
  transductive_v1/common.py (was duplicated)
"""
from __future__ import annotations

import json
import pickle
import re
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import load_settings


RANDOM_SEED = 42

EVENT_TIME_COLUMNS = {
    "twd_transfer": "created_at",
    "crypto_transfer": "created_at",
    "usdt_swap": "created_at",
    "usdt_twd_trading": "updated_at",
}


def load_clean_table(name: str) -> pd.DataFrame:
    """Load a cleaned parquet table from the AWS event clean directory."""
    settings = load_settings()
    path = settings.aws_event_clean_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


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
    encoded.columns = _unique_feature_names(
        [_sanitize_feature_name(column) for column in encoded.columns]
    )
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
        unique.append(column if count == 0 else f"{column}_{count}")
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
