from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_settings

# ── Shared with transductive_v1 (single source of truth) ─────────────────────
from shared.aws_common import (  # noqa: F401  (re-exported for backward compat)
    RANDOM_SEED,
    EVENT_TIME_COLUMNS,
    load_clean_table,
    save_json,
    save_pickle,
    load_pickle,
    to_utc_timestamp,
    safe_ratio,
    encode_frame,
    list_event_cutoffs,
    default_temporal_cutoff,
)

# ── official/-specific constants ──────────────────────────────────────────────
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


def load_raw_table(name: str) -> pd.DataFrame:
    paths = load_official_paths()
    path = paths.raw_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


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
