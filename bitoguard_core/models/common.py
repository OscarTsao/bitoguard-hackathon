from __future__ import annotations

import hashlib
import io
import json
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
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
    dataset = store.fetch_df(
        """
        WITH positive_effective_dates AS (
            SELECT
                user_id,
                CAST(MIN(observed_at) AS DATE) AS positive_effective_date
            FROM canonical.blacklist_feed
            WHERE observed_at IS NOT NULL
            GROUP BY user_id
        )
        SELECT
            features.*,
            COALESCE(labels.hidden_suspicious_label, 0) AS hidden_suspicious_label,
            COALESCE(labels.scenario_types, '') AS scenario_types
        FROM features.feature_snapshots_user_30d AS features
        LEFT JOIN ops.oracle_user_labels AS labels ON features.user_id = labels.user_id
        LEFT JOIN positive_effective_dates AS positive_effective_dates ON features.user_id = positive_effective_dates.user_id
        WHERE COALESCE(labels.hidden_suspicious_label, 0) = 0
            OR (
                positive_effective_dates.positive_effective_date IS NOT NULL
                AND features.snapshot_date >= positive_effective_dates.positive_effective_date
            )
        """
    )
    dataset["snapshot_date"] = pd.to_datetime(dataset["snapshot_date"])
    dataset["hidden_suspicious_label"] = dataset["hidden_suspicious_label"].astype(int)
    dataset["scenario_types"] = dataset["scenario_types"].fillna("")
    return dataset


def forward_date_splits(snapshot_dates: pd.Series) -> dict[str, list[date]]:
    date_series = pd.Series(snapshot_dates).dropna()
    if date_series.empty:
        return {"train": [], "valid": [], "holdout": []}

    unique_dates = sorted(pd.to_datetime(date_series).dt.date.unique())
    total_dates = len(unique_dates)
    if total_dates == 1:
        return {"train": unique_dates, "valid": [], "holdout": []}
    if total_dates == 2:
        return {"train": unique_dates[:1], "valid": unique_dates[1:], "holdout": []}

    train_count = max(1, int(total_dates * 0.7))
    valid_count = max(1, int(total_dates * 0.15))
    while train_count + valid_count >= total_dates:
        if train_count > valid_count and train_count > 1:
            train_count -= 1
        elif valid_count > 1:
            valid_count -= 1
        else:
            train_count -= 1
    valid_end = train_count + valid_count
    return {
        "train": unique_dates[:train_count],
        "valid": unique_dates[train_count:valid_end],
        "holdout": unique_dates[valid_end:],
    }


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in NON_FEATURE_COLUMNS | {"hidden_suspicious_label", "scenario_types"}]


def encode_features(frame: pd.DataFrame, columns: list[str], reference_columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    # Cast object-dtype categorical columns to string before get_dummies to avoid
    # FutureWarning about Index.insert with object-dtype index
    subset = frame[columns].copy()
    for col in subset.select_dtypes(include="object").columns:
        subset[col] = subset[col].astype(str)
    encoded = pd.get_dummies(subset, dummy_na=True)
    if reference_columns is not None:
        encoded = encoded.reindex(columns=reference_columns, fill_value=0)
        return encoded, reference_columns
    return encoded, list(encoded.columns)


def model_dir() -> Path:
    settings = load_settings()
    path = settings.artifact_dir / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── LightGBM (text format — no pickle, inherently safe) ──────────────────────

def save_lgbm(model: "lgb.LGBMClassifier", path: Path) -> None:
    """Save a fitted LGBMClassifier using LightGBM's native text format."""
    model.booster_.save_model(str(path))


def load_lgbm(path: Path) -> "lgb.Booster":
    """Load a LightGBM Booster from its native text format."""
    return lgb.Booster(model_file=str(path))


# ── IsolationForest (joblib + SHA-256 integrity check) ───────────────────────

def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_iforest(model: object, path: Path) -> None:
    """Save IsolationForest with joblib and write a SHA-256 integrity manifest."""
    joblib.dump(model, path)
    path.with_suffix(".sha256").write_text(_sha256_file(path), encoding="utf-8")


def load_iforest(path: Path) -> object:
    """Load IsolationForest after verifying SHA-256 integrity.

    Reads the file once into memory to eliminate the TOCTOU window between
    the integrity check and deserialization.
    """
    sha_path = path.with_suffix(".sha256")
    if not sha_path.exists():
        raise FileNotFoundError(f"SHA-256 manifest not found for {path}")
    expected = sha_path.read_text(encoding="utf-8").strip()
    file_bytes = path.read_bytes()
    actual = hashlib.sha256(file_bytes).hexdigest()
    if actual != expected:
        raise ValueError(
            f"Model file integrity check FAILED for {path}. "
            "The file may have been tampered with. Retrain the model."
        )
    return joblib.load(io.BytesIO(file_bytes))


def save_json(data: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Generic joblib (IsolationForest-style: joblib + SHA-256 integrity) ────────

def save_joblib(model: object, path: Path) -> None:
    """Save any sklearn-compatible model via joblib with SHA-256 integrity check."""
    joblib.dump(model, path)
    path.with_suffix(".sha256").write_text(_sha256_file(path), encoding="utf-8")


def load_joblib(path: Path) -> object:
    """Load a joblib model after verifying SHA-256 integrity."""
    sha_path = path.with_suffix(".sha256")
    if not sha_path.exists():
        raise FileNotFoundError(f"SHA-256 manifest not found for {path}")
    expected = sha_path.read_text(encoding="utf-8").strip()
    file_bytes = path.read_bytes()
    if hashlib.sha256(file_bytes).hexdigest() != expected:
        raise ValueError(f"Integrity check FAILED for {path}. Retrain.")
    return joblib.load(io.BytesIO(file_bytes))


def load_pickle(path: Path) -> Any:
    """Load a pickle or joblib file. Alias for tests and ablation scripts."""
    if Path(path).suffix == ".joblib":
        import joblib
        return joblib.load(path)
    import pickle
    with open(path, "rb") as handle:
        return pickle.load(handle)
