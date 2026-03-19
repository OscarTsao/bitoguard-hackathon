"""Dormancy detection and cascade split utilities.

The current dataset has a critical artifact: a significant fraction of blacklisted
users have zero or near-zero behavioral activity. Any model trained on this data
may learn dormancy detection rather than suspicious behavior detection.

This module provides:
  1. compute_dormancy_score(df) — fraction of behavioral columns that are zero
  2. is_dormant(df) — deterministic dormancy classifier
  3. split_dormant_active(df) — partition a feature DataFrame into dormant/active subsets
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Behavioral columns that should be non-zero for an "active" user.
# These are from the v2 feature registry (features/registry.py).
# A user with ALL of these == 0 is classified as dormant.
BEHAVIORAL_COLUMNS = [
    "twd_dep_count", "twd_wdr_count", "twd_all_count",
    "crypto_dep_count", "crypto_wdr_count", "crypto_all_count",
    "swap_count", "trading_count",
    "ip_n_unique", "ip_n_sessions",
    "early_3d_count",
    # Official pipeline column variants
    "twd_total_count", "twd_deposit_count", "twd_withdraw_count",
    "crypto_total_count", "crypto_deposit_count", "crypto_withdraw_count",
    "swap_total_count",
]


def compute_dormancy_score(df: pd.DataFrame) -> pd.Series:
    """Return a float in [0, 1]: fraction of BEHAVIORAL_COLUMNS that are zero.

    1.0 = fully dormant (all behavioral columns present in df are 0).
    0.0 = fully active (no behavioral columns are 0).

    Columns not present in df are ignored (not counted as zero).
    """
    present = [c for c in BEHAVIORAL_COLUMNS if c in df.columns]
    if not present:
        return pd.Series(0.0, index=df.index, dtype="float32")
    zero_counts = (df[present].fillna(0) == 0).sum(axis=1)
    return (zero_counts / len(present)).astype("float32")


def is_dormant(df: pd.DataFrame, threshold: float = 1.0) -> pd.Series:
    """Return boolean Series: True if user is dormant (dormancy_score >= threshold).

    Default threshold=1.0 means ALL behavioral columns must be zero.
    Use threshold=0.9 for near-dormant detection (90%+ columns are zero).
    """
    return compute_dormancy_score(df) >= threshold


def split_dormant_active(
    df: pd.DataFrame,
    threshold: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into (dormant, active) subsets.

    Returns:
        dormant: rows where dormancy_score >= threshold
        active: rows where dormancy_score < threshold
    """
    mask = is_dormant(df, threshold)
    return df[mask].copy(), df[~mask].copy()
