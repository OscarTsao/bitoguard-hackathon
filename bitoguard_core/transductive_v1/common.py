from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import load_settings

# ── Shared with official/ (single source of truth) ───────────────────────────
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

# ── transductive_v1/-specific paths ──────────────────────────────────────────

@dataclass(frozen=True)
class TransductivePaths:
    root: Path
    feature_dir: Path
    model_dir: Path
    report_dir: Path
    prediction_dir: Path
    bundle_path: Path
    clean_dir: Path
    raw_dir: Path


def load_paths() -> TransductivePaths:
    settings = load_settings()
    root = settings.artifact_dir / "transductive_v1"
    feature_dir = root / "features"
    model_dir = root / "models"
    report_dir = root / "reports"
    prediction_dir = root / "predictions"
    for path in (root, feature_dir, model_dir, report_dir, prediction_dir):
        path.mkdir(parents=True, exist_ok=True)
    return TransductivePaths(
        root=root,
        feature_dir=feature_dir,
        model_dir=model_dir,
        report_dir=report_dir,
        prediction_dir=prediction_dir,
        bundle_path=root / "bundle.json",
        clean_dir=settings.aws_event_clean_dir,
        raw_dir=settings.aws_event_raw_dir,
    )


def feature_path(name: str, cutoff_tag: str = "full") -> Path:
    return load_paths().feature_dir / f"{name}_{cutoff_tag}.parquet"


def report_path(name: str) -> Path:
    return load_paths().report_dir / name


def model_path(name: str) -> Path:
    return load_paths().model_dir / name


def prediction_path(name: str) -> Path:
    return load_paths().prediction_dir / name


def bundle_path() -> Path:
    return load_paths().bundle_path
