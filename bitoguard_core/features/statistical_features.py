"""Advanced statistical features: Benford's law, entropy, burst detection.

These features capture distribution-level anomalies that aggregate statistics
(sum, count, mean) cannot detect.

Column name handling:
  - fiat: 'occurred_at' or 'created_at' for time; 'amount_twd' for amounts
  - crypto: 'occurred_at' or 'created_at' for time; 'amount_twd_equiv' for amounts
  - trades: 'occurred_at', 'updated_at', or 'created_at' for time;
            'notional_twd' or 'trade_notional_twd' or 'twd_amount' for amounts
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Expected Benford's Law distribution for first digits 1-9
BENFORD_EXPECTED = np.array([np.log10(1 + 1 / d) for d in range(1, 10)])

_TIME_COLS = ("occurred_at", "created_at", "updated_at")
_FIAT_AMT_COLS = ("amount_twd",)
_CRYPTO_AMT_COLS = ("amount_twd_equiv",)
_TRADE_AMT_COLS = ("notional_twd", "trade_notional_twd", "twd_amount")
_STAT_FEATURE_SUFFIXES = (
    "benford_chi2",
    "amount_entropy",
    "round_ratio",
    "burst_score",
    "inter_event_cv",
)
_SOURCE_SPECS = (
    ("fiat", _FIAT_AMT_COLS),
    ("crypto", _CRYPTO_AMT_COLS),
    ("trade", _TRADE_AMT_COLS),
)
_STAT_FEATURE_COLUMNS = [
    f"{prefix}_{suffix}"
    for prefix, _ in _SOURCE_SPECS
    for suffix in _STAT_FEATURE_SUFFIXES
]


def _find_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _leading_digit(value: float) -> int | None:
    """Extract the leading digit (1-9) from a positive number."""
    if value <= 0 or not np.isfinite(value):
        return None
    s = f"{abs(value):.10f}".lstrip("0").replace(".", "")
    for ch in s:
        if ch.isdigit() and ch != "0":
            return int(ch)
    return None


def _benford_chi_squared(amounts: pd.Series) -> float:
    """Chi-squared statistic of leading digit distribution vs Benford's law.

    Higher values = more deviation from natural distribution.
    Structuring and round-number laundering produce high Benford deviation.
    """
    amounts = pd.to_numeric(amounts, errors="coerce").dropna()
    amounts = amounts[amounts > 0]
    if len(amounts) < 10:
        return 0.0

    observed = np.zeros(9)
    # Vectorized leading digit extraction via log10
    log_vals = np.log10(amounts.values)
    fractional = log_vals - np.floor(log_vals)
    leading = np.floor(10 ** fractional).astype(int)
    leading = leading[(leading >= 1) & (leading <= 9)]
    for d in range(1, 10):
        observed[d - 1] = (leading == d).sum()

    total = observed.sum()
    if total == 0:
        return 0.0
    observed_freq = observed / total
    chi_sq = float(np.sum((observed_freq - BENFORD_EXPECTED) ** 2 / (BENFORD_EXPECTED + 1e-10)))
    return chi_sq


def _amount_entropy(amounts: pd.Series, n_bins: int = 20) -> float:
    """Shannon entropy of the amount distribution.

    Low entropy = concentrated amounts (possible structuring).
    High entropy = diverse amounts (more natural).
    """
    amounts = pd.to_numeric(amounts, errors="coerce").dropna()
    if len(amounts) < 2:
        return 0.0
    hist, _ = np.histogram(amounts.values, bins=n_bins)
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _round_number_ratio(amounts: pd.Series, divisor: float = 1000.0) -> float:
    """Fraction of transactions with round-number amounts (divisible by divisor)."""
    amounts = pd.to_numeric(amounts, errors="coerce").dropna()
    if len(amounts) == 0:
        return 0.0
    round_count = ((amounts % divisor).abs() < 1.0).sum()
    return float(round_count / len(amounts))


def _burst_score(timestamps: pd.Series, window_hours: float = 24.0) -> float:
    """Maximum transaction density in any sliding window vs median daily baseline.

    High burst = sudden activity spike vs baseline (orchestrated behavior).
    """
    timestamps = pd.to_datetime(timestamps, utc=True, errors="coerce").dropna().sort_values()
    if len(timestamps) < 3:
        return float(len(timestamps))

    window_ns = int(window_hours * 3600 * 1e9)
    ts_ns = timestamps.values.astype(np.int64)
    right_idx = np.searchsorted(ts_ns, ts_ns + window_ns, side="right")
    counts = right_idx - np.arange(len(ts_ns))
    max_count = int(counts.max())

    date_counts = timestamps.dt.date.value_counts()
    median_daily = float(date_counts.median()) if len(date_counts) > 0 else 1.0
    return float(max_count / max(1.0, median_daily))


def _inter_event_cv(timestamps: pd.Series) -> float:
    """Coefficient of variation of inter-event times.

    Low CV = regular cadence (bot-like or scheduled).
    High CV = irregular (more human-like).
    """
    timestamps = pd.to_datetime(timestamps, utc=True, errors="coerce").dropna().sort_values()
    if len(timestamps) < 3:
        return 0.0
    diffs = timestamps.diff().dropna().dt.total_seconds()
    mean_diff = diffs.mean()
    if mean_diff <= 0:
        return 0.0
    return float(diffs.std() / mean_diff)


def _compute_user_stats(
    user_events: pd.DataFrame,
    amt_candidates: tuple[str, ...],
    prefix: str,
) -> dict:
    """Compute stats for one user's events."""
    row: dict = {}
    amt_col = _find_col(user_events, amt_candidates)
    time_col = _find_col(user_events, _TIME_COLS)

    amounts = (
        pd.to_numeric(user_events[amt_col], errors="coerce").dropna()
        if amt_col else pd.Series(dtype=float)
    )
    row[f"{prefix}_benford_chi2"] = _benford_chi_squared(amounts)
    row[f"{prefix}_amount_entropy"] = _amount_entropy(amounts)
    row[f"{prefix}_round_ratio"] = _round_number_ratio(amounts)

    if time_col:
        row[f"{prefix}_burst_score"] = _burst_score(user_events[time_col])
        row[f"{prefix}_inter_event_cv"] = _inter_event_cv(user_events[time_col])
    else:
        row[f"{prefix}_burst_score"] = 0.0
        row[f"{prefix}_inter_event_cv"] = 0.0

    return row


def _compute_user_stats_from_group(
    group: pd.DataFrame,
    amt_candidates: tuple[str, ...],
    prefix: str,
) -> pd.Series:
    """Adapter for groupby.apply that preserves the existing per-user logic."""
    result = _compute_user_stats(group.reset_index(drop=True), amt_candidates, prefix)
    return pd.Series(result)


def compute_statistical_features(
    fiat: pd.DataFrame,
    crypto: pd.DataFrame,
    trades: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Benford, entropy, burst, and timing features per user."""
    source_frames = {
        "fiat": fiat,
        "crypto": crypto,
        "trade": trades,
    }

    user_series = [
        df["user_id"].dropna()
        for df in source_frames.values()
        if not df.empty and "user_id" in df.columns
    ]
    if not user_series:
        return pd.DataFrame()

    all_users = pd.Index(pd.concat(user_series, ignore_index=True).unique()).sort_values()
    result = pd.DataFrame({"user_id": all_users})

    for prefix, amt_candidates in _SOURCE_SPECS:
        source_df = source_frames[prefix]
        if source_df.empty or "user_id" not in source_df.columns:
            continue

        stats = source_df.groupby("user_id", sort=True).apply(
            _compute_user_stats_from_group,
            amt_candidates=amt_candidates,
            prefix=prefix,
            include_groups=False,
        )
        result = result.merge(stats.reset_index(), on="user_id", how="left")

    ordered_columns = ["user_id", *_STAT_FEATURE_COLUMNS]
    return result.reindex(columns=ordered_columns, fill_value=0.0).fillna(0)
