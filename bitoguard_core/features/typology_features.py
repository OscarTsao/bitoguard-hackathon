# bitoguard_core/features/typology_features.py
"""FATF AML typology-specific features.

Derived entirely from columns already assembled in the v2 base frame.
All columns that might be absent are guarded with _safe() so the module
is safe to call on any partial frame.

Columns produced (6 float32 features):
  structuring_ratio        - many sub-threshold TWD deposits relative to volume
  dormancy_burst_score     - old account that suddenly burst in the last 7 days
  round_amount_proxy       - average deposit clusters near round TWD multiples
  multi_asset_layering     - swap activity x recent cashout ratio
  velocity_acceleration    - recent 7d volume vs older volume ratio
  same_day_cycle_proxy     - rapid fiat-in -> crypto-out cycling indicator
"""
from __future__ import annotations

import pandas as pd


_TYPOLOGY_COLS = [
    "structuring_ratio",
    "dormancy_burst_score",
    "round_amount_proxy",
    "multi_asset_layering",
    "velocity_acceleration",
    "same_day_cycle_proxy",
]


def _safe(base: pd.DataFrame, col: str, fill: float = 0.0) -> pd.Series:
    """Return base[col] if present, else a zero-filled Series aligned to base.index."""
    if col in base.columns:
        return base[col].fillna(fill)
    return pd.Series(fill, index=base.index, dtype="float32")


def compute_typology_features(base: pd.DataFrame) -> pd.DataFrame:
    """Compute 6 FATF typology proxy features from the assembled v2 base frame.

    Parameters
    ----------
    base : pd.DataFrame
        The merged v2 feature frame (must contain a ``user_id`` column).
        Any missing upstream column is treated as zero.

    Returns
    -------
    pd.DataFrame
        One row per user with columns ``user_id`` + the 6 typology features.
    """
    if base.empty or "user_id" not in base.columns:
        return pd.DataFrame(columns=["user_id"] + _TYPOLOGY_COLS)

    out = pd.DataFrame({"user_id": base["user_id"]})

    # 1. structuring_ratio
    # Proxy for deposit structuring: many small sub-50k TWD deposits
    # relative to total deposit volume. High value => smurfing signal.
    # Formula: clip(dep_count / (dep_sum/50_000 + 1), 0, 10) / 10  -> [0, 1]
    dep_count = _safe(base, "twd_dep_count")
    dep_sum   = _safe(base, "twd_dep_sum")
    raw_structuring = dep_count / (dep_sum / 50_000.0 + 1.0)
    out["structuring_ratio"] = (
        raw_structuring.clip(lower=0.0, upper=10.0) / 10.0
    ).astype("float32")

    # 2. dormancy_burst_score
    # Binary flag: old account (>90 days) with >50% of lifetime deposits
    # in the last 7 days. Dormant-then-burst = primary money-mule indicator.
    dep_count_7d = _safe(base, "twd_dep_7d_count")
    account_age  = _safe(base, "account_age_days")
    burst_condition = (
        (dep_count_7d > 0)
        & (account_age > 90)
        & (dep_count_7d / (dep_count + 1.0) > 0.5)
    )
    out["dormancy_burst_score"] = burst_condition.astype("float32")

    # 3. round_amount_proxy
    # If average TWD deposit is near a multiple of NT$10,000, score is high.
    # Structuring often uses identical round amounts.
    avg_deposit = dep_sum / (dep_count + 1.0)
    remainder   = avg_deposit % 10_000.0
    out["round_amount_proxy"] = (
        (1.0 - remainder / 10_000.0).clip(lower=0.0, upper=1.0)
    ).astype("float32")

    # 4. multi_asset_layering
    # High swap activity + high recent cashout = fiat->swap->crypto layering.
    # Formula: swap_count * xch_cashout_ratio_7d  clipped to [0, 10]
    swap_count           = _safe(base, "swap_count")
    xch_cashout_ratio_7d = _safe(base, "xch_cashout_ratio_7d")
    out["multi_asset_layering"] = (
        (swap_count * xch_cashout_ratio_7d).clip(lower=0.0, upper=10.0)
    ).astype("float32")

    # 5. velocity_acceleration
    # Recent 7d volume vs older baseline, detecting sudden burst.
    # Formula: (dep_sum_7d + 1) / (dep_sum_30d - dep_sum_7d + 1)  clipped to [0, 10]
    dep_sum_7d  = _safe(base, "twd_dep_7d_sum")
    dep_sum_30d = _safe(base, "twd_dep_30d_sum")
    older_volume = (dep_sum_30d - dep_sum_7d).clip(lower=0.0)
    out["velocity_acceleration"] = (
        (dep_sum_7d + 1.0) / (older_volume + 1.0)
    ).clip(lower=0.0, upper=10.0).astype("float32")

    # 6. same_day_cycle_proxy
    # Rapid fiat-in -> crypto-out same-day cycling.
    # Prefers fast_cashout_24h_count; falls back to sequence feature proxy.
    if "fast_cashout_24h_count" in base.columns:
        raw_cycle = _safe(base, "fast_cashout_24h_count")
    else:
        fiat_to_swap_24h   = _safe(base, "fiat_dep_to_swap_buy_within_24h")
        cashout_ratio_life = _safe(base, "xch_cashout_ratio_lifetime")
        raw_cycle = fiat_to_swap_24h * cashout_ratio_life
    out["same_day_cycle_proxy"] = (
        raw_cycle.clip(lower=0.0, upper=10.0)
    ).astype("float32")

    return out.reset_index(drop=True)
