from __future__ import annotations

"""Cross-channel correlation and temporal lag features.

AML patterns often involve coordinated behaviour across channels — e.g. a fiat
deposit followed immediately by a crypto withdrawal (classic layering).  This
module computes per-user lag/change-point proxies from the tabular feature
columns already present in the dataset snapshot.  No raw event tables are
required; all features are derived from pre-aggregated rolling-window columns.

Public API
----------
build_lag_features(dataset) -> DataFrame
    Accepts the full feature dataset (one row per user) and returns a
    user_id + lag_* DataFrame suitable for left-joining onto the main frame.

get_lag_feature_columns() -> list[str]
    Returns the canonical list of lag feature column names.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical list of output feature names
# ---------------------------------------------------------------------------

_LAG_FEATURE_COLUMNS: list[str] = [
    # Deposit-withdrawal velocity asymmetry
    "lag_dep_wd_velocity_ratio",
    "lag_recent_vs_career_velocity",
    # Cross-channel correlation proxies
    "lag_fiat_crypto_sync",
    "lag_deposit_to_withdrawal_days",
    "lag_channel_switching_score",
    # Change-point detection proxies
    "lag_activity_acceleration",
    "lag_amount_shift",
    "lag_burst_indicator",
    # Dormancy → activity pattern
    "lag_dormancy_burst",
    "lag_kyc_to_first_txn_speed",
    # Round-number and high-amount patterns
    "lag_round_amount_fiat_ratio",
    "lag_high_amount_flag",
]


def get_lag_feature_columns() -> list[str]:
    """Return the canonical list of all lag feature column names."""
    return list(_LAG_FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_EPS = 1e-8  # safe-division epsilon


def _col(ds: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    """Return a float64 Series for *name* or a constant Series on missing col."""
    if name in ds.columns:
        return pd.to_numeric(ds[name], errors="coerce").fillna(default)
    return pd.Series(default, index=ds.index, dtype="float64")


def _safe_div(num: pd.Series, denom: pd.Series) -> pd.Series:
    """Element-wise safe division, adding _EPS to denominator."""
    return num / (denom + _EPS)


# ---------------------------------------------------------------------------
# Feature group builders (each returns a Series or scalar-ready value)
# ---------------------------------------------------------------------------


def _dep_wd_velocity_ratio(ds: pd.DataFrame) -> pd.Series:
    """(deposits_7d / 7) / (withdrawals_30d / 30 + ε)

    A high ratio means fast recent deposits relative to slow cumulative
    withdrawals — classic smurfing / placement pattern.
    """
    # Prefer count columns; fall back to amount columns if available
    dep_7d = _col(ds, "twd_dep_count_7d") + _col(ds, "crypto_deposit_count_7d")
    wd_30d = _col(ds, "twd_wd_count_30d") + _col(ds, "crypto_withdraw_count_30d")

    daily_dep = dep_7d / 7.0
    daily_wd = wd_30d / 30.0

    return _safe_div(daily_dep, daily_wd).clip(0.0, 100.0).astype("float32")


def _recent_vs_career_velocity(ds: pd.DataFrame) -> pd.Series:
    """(7d deposit count / 7) / (career deposit count / account_age_days + ε)

    Detects sudden activity surges on accounts that were previously dormant or
    had low transaction rates throughout their lifetime.
    """
    dep_7d = _col(ds, "twd_dep_count_7d") + _col(ds, "crypto_deposit_count_7d")
    career_dep = (
        _col(ds, "career_dep_count")
        # fallback: use 30-day count if career column absent
        if "career_dep_count" in ds.columns
        else _col(ds, "twd_dep_count_30d") + _col(ds, "crypto_deposit_count_30d")
    )
    account_age = _col(ds, "account_age_days", default=1.0).clip(lower=1.0)

    recent_daily = dep_7d / 7.0
    career_daily = _safe_div(career_dep, account_age)

    return _safe_div(recent_daily, career_daily).clip(0.0, 200.0).astype("float32")


def _fiat_crypto_sync(ds: pd.DataFrame) -> pd.Series:
    """(twd_dep_count_7d × crypto_withdraw_count_7d) / (twd_total × crypto_total + ε)

    Measures co-occurrence of recent fiat inflows with crypto outflows.
    High values indicate the classic layering pattern: cash in → crypto out.
    """
    twd_dep_7d = _col(ds, "twd_dep_count_7d")
    crypto_wd_7d = _col(ds, "crypto_withdraw_count_7d")
    twd_total = _col(ds, "twd_total_count", default=1.0).clip(lower=1.0)
    crypto_total = _col(ds, "crypto_total_count", default=1.0).clip(lower=1.0)

    numerator = twd_dep_7d * crypto_wd_7d
    denominator = twd_total * crypto_total

    return _safe_div(numerator, denominator).clip(0.0, 1.0).astype("float32")


def _deposit_to_withdrawal_days(ds: pd.DataFrame) -> pd.Series:
    """Estimated days between a fiat deposit and the subsequent crypto withdrawal.

    Uses twd_deposit_to_crypto_median_hours if present, else falls back to an
    inverse-velocity proxy: users who deposit fast and withdraw fast get a low
    value (short lag — suspicious), while slow traders get a high value.
    """
    if "twd_deposit_to_crypto_median_hours" in ds.columns:
        hours = pd.to_numeric(ds["twd_deposit_to_crypto_median_hours"], errors="coerce").fillna(0.0)
        return (hours / 24.0).clip(0.0, 365.0).astype("float32")

    # Fallback proxy: 1 / (fiat_crypto_sync + ε)  — smaller sync score → longer lag
    sync = _fiat_crypto_sync(ds)
    return _safe_div(pd.Series(1.0, index=ds.index), sync.astype("float64")).clip(0.0, 365.0).astype("float32")


def _channel_switching_score(ds: pd.DataFrame) -> pd.Series:
    """Count of distinct channels (fiat / crypto / swap) the user has used.

    Score: 0 (none), 1, 2, or 3 channels.  High channel switching combined
    with large volumes is an AML indicator (spreading across products to avoid
    single-channel detection thresholds).
    """
    has_twd = (_col(ds, "twd_total_count") > 0).astype("float32")
    has_crypto = (_col(ds, "crypto_total_count") > 0).astype("float32")
    has_swap = (_col(ds, "swap_total_count") > 0).astype("float32")
    return (has_twd + has_crypto + has_swap).astype("float32")


def _activity_acceleration(ds: pd.DataFrame) -> pd.Series:
    """(total_count_7d / 7) / (total_count_30d / 30 + ε)

    A value > 1 means recent activity is faster than the 30-day average —
    potential acceleration leading up to an event (e.g. mass withdrawal).
    """
    count_7d = (
        _col(ds, "twd_dep_count_7d")
        + _col(ds, "twd_wd_count_7d")
        + _col(ds, "crypto_deposit_count_7d")
        + _col(ds, "crypto_withdraw_count_7d")
        + _col(ds, "swap_count_7d")
        + _col(ds, "trade_count_7d")
    )
    count_30d = (
        _col(ds, "twd_dep_count_30d")
        + _col(ds, "twd_wd_count_30d")
        + _col(ds, "crypto_deposit_count_30d")
        + _col(ds, "crypto_withdraw_count_30d")
        + _col(ds, "swap_count_30d")
        + _col(ds, "trade_count_30d")
    )

    daily_7d = count_7d / 7.0
    daily_30d = count_30d / 30.0

    return _safe_div(daily_7d, daily_30d).clip(0.0, 100.0).astype("float32")


def _amount_shift(ds: pd.DataFrame) -> pd.Series:
    """(avg_amount_7d) / (avg_amount_30d + ε)

    Detects a shift in transaction size between recent and historical windows.
    Sudden increase in average transaction size is a classic structuring signal.
    """
    # Use TWD as primary signal (most direct fiat-layering channel)
    sum_7d = _col(ds, "twd_dep_amount_7d") + _col(ds, "twd_wd_amount_7d")
    sum_30d = _col(ds, "twd_dep_amount_30d") + _col(ds, "twd_wd_amount_30d")
    count_7d = (
        _col(ds, "twd_dep_count_7d") + _col(ds, "twd_wd_count_7d")
    ).clip(lower=_EPS)
    count_30d = (
        _col(ds, "twd_dep_count_30d") + _col(ds, "twd_wd_count_30d")
    ).clip(lower=_EPS)

    avg_7d = _safe_div(sum_7d, count_7d)
    avg_30d = _safe_div(sum_30d, count_30d)

    return _safe_div(avg_7d, avg_30d).clip(0.0, 100.0).astype("float32")


def _burst_indicator(ds: pd.DataFrame) -> pd.Series:
    """Binary flag: 1 if activity_acceleration > 3 (sudden 3× burst).

    Threshold of 3× is chosen to flag outliers while ignoring normal weekly
    variance (legitimate traders might show 2× acceleration on active days).
    """
    accel = _activity_acceleration(ds)
    return (accel > 3.0).astype("float32")


def _dormancy_burst(ds: pd.DataFrame) -> pd.Series:
    """Dormancy-burst score combining account age and sudden activity.

    Prefers ``dormancy_score`` from the official feature pipeline when available
    (computed by event_ngram_features / statistical_features).  Falls back to a
    hand-crafted proxy: old account (>180d) × fast_cashout indicator.
    """
    if "dormancy_score" in ds.columns:
        s = pd.to_numeric(ds["dormancy_score"], errors="coerce").fillna(0.0)
        if not (s == 0).all():
            return s.clip(0.0, 1.0).astype("float32")
    # Fallback: binary — old account with fast cashout behaviour
    account_age = _col(ds, "account_age_days", default=0.0)
    fast_co = _col(ds, "fast_cashout_24h_count") + _col(ds, "fast_cashout_72h_count")
    old_account = (account_age > 180.0).astype(float)
    has_fast = (fast_co > 0).astype(float)
    return (old_account * has_fast).astype("float32")


def _kyc_to_first_txn_speed(ds: pd.DataFrame) -> pd.Series:
    """1 / (days_email_to_level1 + 1) × (has_transactions)

    Fast KYC completion followed by immediate transactions is a red flag
    (account factory / identity-for-hire behaviour).  Users who took a long
    time to verify and who have no transactions score near zero.
    Uses career-level transaction counts when windowed (90d) counts are absent.
    """
    kyc_days = _col(ds, "days_email_to_level1", default=365.0).clip(lower=0.0)
    # Use career counts as a robust fallback for 90d windowed counts
    total_career = (
        _col(ds, "twd_total_count")
        + _col(ds, "crypto_total_count")
        + _col(ds, "swap_total_count")
        + _col(ds, "order_total_count")
    )
    has_txns = (total_career > 0).astype(float)
    kyc_speed = 1.0 / (kyc_days + 1.0)
    return (kyc_speed * has_txns).clip(0.0, 1.0).astype("float32")


def _round_amount_fiat_ratio(ds: pd.DataFrame) -> pd.Series:
    """Fraction of fiat deposits that are round numbers.

    Round-number structuring (e.g. exactly 10,000 / 50,000 TWD) is a
    well-known indicator of placement-phase layering.

    Uses ``twd_round_10k_ratio`` (pre-computed) when available, else
    ``twd_round_amount_count / twd_deposit_count``, else 0.
    """
    if "twd_round_10k_ratio" in ds.columns:
        s = pd.to_numeric(ds["twd_round_10k_ratio"], errors="coerce").fillna(0.0)
        if not (s == 0).all():
            return s.clip(0.0, 1.0).astype("float32")
    if "twd_round_amount_count" in ds.columns:
        round_count = pd.to_numeric(ds["twd_round_amount_count"], errors="coerce").fillna(0.0)
        dep_count = _col(ds, "twd_deposit_count").clip(lower=_EPS)
        return _safe_div(round_count, dep_count).clip(0.0, 1.0).astype("float32")
    return pd.Series(0.0, index=ds.index, dtype="float32")


def _high_amount_flag(ds: pd.DataFrame) -> pd.Series:
    """1 if any single transaction is unusually large relative to user mean.

    Uses career max / career avg for fiat deposits (or crypto if fiat absent).
    A ratio > 10 indicates at least one transaction 10× larger than the user's
    own average — structuring or single large placement signal.
    """
    # Prefer windowed; fall back to career max/avg
    max_dep = _col(ds, "twd_dep_amount_max_30d")
    if (max_dep == 0).all():
        max_dep = _col(ds, "twd_deposit_max")
    if (max_dep == 0).all():
        max_dep = _col(ds, "crypto_deposit_max")

    avg_dep = _col(ds, "twd_dep_amount_avg_30d")
    if (avg_dep == 0).all():
        avg_dep = _col(ds, "twd_deposit_avg")
    if (avg_dep == 0).all():
        avg_dep = _col(ds, "crypto_deposit_avg")

    ratio = _safe_div(max_dep, avg_dep.clip(lower=_EPS))
    return (ratio > 10.0).astype("float32")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _resolve_column_aliases(ds: pd.DataFrame) -> pd.DataFrame:
    """Map actual official feature column names to expected alias names.

    The official pipeline generates career-level statistics only (e.g.
    ``twd_deposit_count``, ``crypto_withdraw_sum``).  Short-window columns
    (7d, 30d) exist in the schema but are null in the official feature set.

    Strategy:
    1. Try exact windowed columns first (correct if populated).
    2. Fall back to career-rate-based proxies: expected count over window =
       career_count / account_age_days × window_days.  This is a steady-state
       estimate; it allows the formulas to produce non-trivial values even
       without true windowed data.
    """
    ds = ds.copy()
    def _get(name: str, default: float = 0.0) -> pd.Series:
        if name in ds.columns:
            s = pd.to_numeric(ds[name], errors="coerce").fillna(default)
            if s.isna().all() or (s == 0).all():
                return pd.Series(default, index=ds.index, dtype="float64")
            return s
        return pd.Series(default, index=ds.index, dtype="float64")

    age_days = _get("account_age_days", default=1.0).clip(lower=1.0)

    # Career counts (always available)
    twd_dep = _get("twd_deposit_count")
    twd_wd = _get("twd_withdraw_count")
    twd_total = _get("twd_total_count")
    crypto_dep = _get("crypto_deposit_count")
    crypto_wd = _get("crypto_withdraw_count")
    crypto_total = _get("crypto_total_count")
    swap_total = _get("swap_total_count")
    order_total = _get("order_total_count")

    # ── Count aliases (try exact windowed → career-rate fallback) ───────────
    for period, days in [("7d", 7.0), ("30d", 30.0)]:
        # TWD deposit
        alias_dep = f"twd_dep_count_{period}"
        if alias_dep not in ds.columns:
            exact = _get(f"twd_total_{period}_count") - _get(f"twd_withdraw_{period}_count")
            ds[alias_dep] = exact.clip(lower=0.0).astype("float32")
            if (ds[alias_dep] == 0).all():
                # Career-rate proxy: expected count in `period`
                ds[alias_dep] = (twd_dep / age_days * days).clip(lower=0.0).astype("float32")

        alias_wd = f"twd_wd_count_{period}"
        if alias_wd not in ds.columns:
            exact = _get(f"twd_withdraw_{period}_count")
            ds[alias_wd] = exact.astype("float32")
            if (ds[alias_wd] == 0).all():
                ds[alias_wd] = (twd_wd / age_days * days).clip(lower=0.0).astype("float32")

        alias_cdep = f"crypto_deposit_count_{period}"
        if alias_cdep not in ds.columns:
            exact = _get(f"crypto_total_{period}_count") - _get(f"crypto_withdraw_{period}_count")
            ds[alias_cdep] = exact.clip(lower=0.0).astype("float32")
            if (ds[alias_cdep] == 0).all():
                ds[alias_cdep] = (crypto_dep / age_days * days).clip(lower=0.0).astype("float32")

        alias_cwd = f"crypto_withdraw_count_{period}"
        if alias_cwd not in ds.columns:
            exact = _get(f"crypto_withdraw_{period}_count")
            ds[alias_cwd] = exact.astype("float32")
            if (ds[alias_cwd] == 0).all():
                ds[alias_cwd] = (crypto_wd / age_days * days).clip(lower=0.0).astype("float32")

        if f"swap_count_{period}" not in ds.columns:
            exact = _get(f"swap_total_{period}_count")
            ds[f"swap_count_{period}"] = exact.astype("float32")
            if (ds[f"swap_count_{period}"] == 0).all():
                ds[f"swap_count_{period}"] = (swap_total / age_days * days).clip(lower=0.0).astype("float32")

        if f"trade_count_{period}" not in ds.columns:
            exact = _get(f"order_total_{period}_count")
            ds[f"trade_count_{period}"] = exact.astype("float32")
            if (ds[f"trade_count_{period}"] == 0).all():
                ds[f"trade_count_{period}"] = (order_total / age_days * days).clip(lower=0.0).astype("float32")

    # Career deposit count
    if "career_dep_count" not in ds.columns or (ds["career_dep_count"] == 0).all():
        ds["career_dep_count"] = (twd_dep + crypto_dep).astype("float32")

    # ── Amount aliases (try exact → career-rate fallback) ──────────────────
    twd_dep_sum = _get("twd_deposit_sum")
    twd_wd_sum = _get("twd_withdraw_sum")
    for period, days in [("7d", 7.0), ("30d", 30.0)]:
        alias_dep_amt = f"twd_dep_amount_{period}"
        alias_wd_amt = f"twd_wd_amount_{period}"
        if alias_dep_amt not in ds.columns:
            total_sum = _get(f"twd_total_{period}_sum")
            wd_count = _get(f"twd_withdraw_{period}_count")
            tot_count = _get(f"twd_total_{period}_count").clip(lower=_EPS)
            wd_frac = (wd_count / tot_count).clip(0.0, 1.0)
            ds[alias_wd_amt] = (total_sum * wd_frac).astype("float32")
            ds[alias_dep_amt] = (total_sum * (1.0 - wd_frac)).astype("float32")
            if (ds[alias_dep_amt] == 0).all():
                ds[alias_dep_amt] = (twd_dep_sum / age_days * days).astype("float32")
                ds[alias_wd_amt] = (twd_wd_sum / age_days * days).astype("float32")

    return ds


def build_lag_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-channel correlation and temporal lag features.

    Parameters
    ----------
    dataset:
        Full feature dataset, one row per user.  Must contain a ``user_id``
        column (Int64 type).  Any missing feature columns are handled
        gracefully by substituting zeros.

    Returns
    -------
    DataFrame with columns: user_id, lag_dep_wd_velocity_ratio, ...,
    lag_high_amount_flag.  One row per user.  All lag_* columns are float32.
    """
    try:
        if "user_id" not in dataset.columns:
            raise ValueError("dataset must contain a 'user_id' column")

        ds = _resolve_column_aliases(dataset.reset_index(drop=True))

        result = pd.DataFrame(index=ds.index)
        result["user_id"] = pd.array(
            pd.to_numeric(ds["user_id"], errors="coerce").fillna(0).astype(int).tolist(),
            dtype="Int64",
        )

        # ── 1. Deposit-withdrawal velocity asymmetry ──────────────────────
        result["lag_dep_wd_velocity_ratio"] = _dep_wd_velocity_ratio(ds)
        result["lag_recent_vs_career_velocity"] = _recent_vs_career_velocity(ds)

        # ── 2. Cross-channel correlation proxies ──────────────────────────
        result["lag_fiat_crypto_sync"] = _fiat_crypto_sync(ds)
        result["lag_deposit_to_withdrawal_days"] = _deposit_to_withdrawal_days(ds)
        result["lag_channel_switching_score"] = _channel_switching_score(ds)

        # ── 3. Change-point detection proxies ─────────────────────────────
        result["lag_activity_acceleration"] = _activity_acceleration(ds)
        result["lag_amount_shift"] = _amount_shift(ds)
        result["lag_burst_indicator"] = _burst_indicator(ds)

        # ── 4. Dormancy → activity ─────────────────────────────────────────
        result["lag_dormancy_burst"] = _dormancy_burst(ds)
        result["lag_kyc_to_first_txn_speed"] = _kyc_to_first_txn_speed(ds)

        # ── 5. Round-number and high-amount patterns ──────────────────────
        result["lag_round_amount_fiat_ratio"] = _round_amount_fiat_ratio(ds)
        result["lag_high_amount_flag"] = _high_amount_flag(ds)

        # Verify all expected columns are present
        for col in _LAG_FEATURE_COLUMNS:
            if col not in result.columns:
                result[col] = np.float32(0.0)

        # Final safety: replace any NaN/Inf that slipped through
        for col in _LAG_FEATURE_COLUMNS:
            result[col] = (
                pd.to_numeric(result[col], errors="coerce")
                .fillna(0.0)
                .astype("float32")
            )

        logger.info(
            "build_lag_features: produced %d rows × %d lag features",
            len(result),
            len(_LAG_FEATURE_COLUMNS),
        )
        return result[["user_id", *_LAG_FEATURE_COLUMNS]].reset_index(drop=True)

    except Exception as exc:
        logger.error("build_lag_features failed (%s); returning empty DataFrame", exc)
        empty = pd.DataFrame(columns=["user_id", *_LAG_FEATURE_COLUMNS])
        empty["user_id"] = pd.array([], dtype="Int64")
        for col in _LAG_FEATURE_COLUMNS:
            empty[col] = pd.Series(dtype="float32")
        return empty
