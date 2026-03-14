# bitoguard_core/features/profile_features.py
"""Profile features from canonical.users.

NOTE: KYC timestamp fields (level1_finished_at, level2_finished_at, confirmed_at)
are consumed by pipeline/transformers.py and NOT stored in canonical.users.
KYC velocity features are therefore not available without a schema extension.
Available: kyc_level (string ordinal), created_at, occupation, income_source, segment.
"""
from __future__ import annotations
import pandas as pd

KYC_LEVEL_MAP = {"level2": 2, "level1": 1, "email_verified": 0, None: -1}

# Columns that receive deterministic integer codes saved in profile_category_maps.json
_CATEGORY_COLS = {
    "occupation":               "occupation_code",
    "declared_source_of_funds": "income_source_code",
    "activity_window":          "user_source_code",
}


def build_profile_category_maps(users: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Build {col_name: {value: code}} maps from training population.

    Codes are assigned alphabetically so they are deterministic given the same
    unique value set. Save the returned dict alongside model artifacts and pass
    it to compute_profile_features() at scoring time.
    """
    maps: dict[str, dict[str, int]] = {}
    for raw_col in _CATEGORY_COLS:
        if raw_col not in users.columns:
            maps[raw_col] = {}
            continue
        # dropna() intentionally excludes nulls from the map — null values
        # hit the .fillna(-1) path in compute_profile_features at scoring time.
        unique_vals = sorted(users[raw_col].dropna().astype(str).unique())
        maps[raw_col] = {v: i for i, v in enumerate(unique_vals)}
    return maps


def compute_profile_features(
    users: pd.DataFrame,
    snapshot_date: pd.Timestamp | None = None,
    category_maps: dict[str, dict[str, int]] | None = None,
) -> pd.DataFrame:
    """8 demographic/KYC features per user (no aggregation)."""
    if users.empty:
        return pd.DataFrame()

    df = users.copy()
    df["created_at"] = pd.to_datetime(df.get("created_at"), utc=True, errors="coerce")

    df["kyc_level_code"] = df["kyc_level"].map(KYC_LEVEL_MAP).fillna(-1).astype(int)

    for raw_col, feature_col in _CATEGORY_COLS.items():
        if raw_col not in df.columns:
            df[feature_col] = -1
            continue
        if category_maps is not None and raw_col in category_maps:
            col_map = category_maps[raw_col]
            df[feature_col] = (
                df[raw_col].astype(str).map(col_map).fillna(-1).astype(int)
            )
        else:
            # Fallback: derive from current data (non-deterministic — only use at build time)
            df[feature_col] = df[raw_col].astype("category").cat.codes

    df["monthly_income_twd"] = df.get("monthly_income_twd", 0.0).fillna(0.0)

    ref = pd.Timestamp.now(tz="UTC") if snapshot_date is None else snapshot_date
    if ref.tzinfo is None:
        ref = ref.tz_localize("UTC")
    df["account_age_days"] = (
        (ref - df["created_at"]).dt.total_seconds().div(86400).clip(lower=0).fillna(0)
    )

    keep = [
        "user_id", "kyc_level_code", "occupation_code", "income_source_code",
        "user_source_code", "monthly_income_twd", "account_age_days",
    ]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)
