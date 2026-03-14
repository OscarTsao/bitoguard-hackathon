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


def compute_profile_features(
    users: pd.DataFrame,
    snapshot_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """8 demographic/KYC features per user (no aggregation)."""
    if users.empty:
        return pd.DataFrame()

    df = users.copy()
    df["created_at"] = pd.to_datetime(df.get("created_at"), utc=True, errors="coerce")

    df["kyc_level_code"] = df["kyc_level"].map(KYC_LEVEL_MAP).fillna(-1).astype(int)
    df["occupation_code"] = df["occupation"].astype("category").cat.codes
    df["income_source_code"] = df["declared_source_of_funds"].astype("category").cat.codes
    df["user_source_code"] = df["activity_window"].astype("category").cat.codes
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
