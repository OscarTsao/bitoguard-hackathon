# bitoguard_core/features/ip_features.py
"""Per-user IP diversity from canonical.login_events.

IP events in this codebase are synthetic: each fiat/crypto/trade transaction
with a source_ip_hash produces one login_event with ip_address=source_ip_hash.
This gives per-transaction IP coverage, not real authentication events.
"""
from __future__ import annotations
import pandas as pd

NIGHT_HOURS = frozenset(range(22, 24))


def compute_ip_features(login_events: pd.DataFrame) -> pd.DataFrame:
    """4 per-user IP diversity features from canonical.login_events."""
    if login_events.empty or "ip_address" not in login_events.columns:
        return pd.DataFrame()

    df = login_events.copy()
    df["occurred_at"] = pd.to_datetime(df["occurred_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["user_id", "ip_address", "occurred_at"])

    # Vectorized aggregations — avoids O(n_users) Python for-loop overhead.
    g = df.groupby("user_id")
    unique_ips     = g["ip_address"].nunique()
    ip_event_count = g.size()

    # ip_concentration = share of most-common IP per user.
    # Computed via (user, ip) counts then max share per user — all vectorized.
    pair_counts = df.groupby(["user_id", "ip_address"]).size()
    ip_concentration = (pair_counts / pair_counts.groupby(level="user_id").sum()).groupby(level="user_id").max()

    # ip_night_share: fraction of events in NIGHT_HOURS (22-23 UTC).
    df["_is_night"] = df["occurred_at"].dt.hour.isin(NIGHT_HOURS)
    ip_night_share = g["_is_night"].mean()

    result = pd.DataFrame({
        "user_id":          unique_ips.index,
        "unique_ips":       unique_ips.values,
        "ip_event_count":   ip_event_count.values,
        "ip_concentration": ip_concentration.reindex(unique_ips.index).fillna(0.0).values,
        "ip_night_share":   ip_night_share.reindex(unique_ips.index).fillna(0.0).values,
    })
    return result.reset_index(drop=True)
