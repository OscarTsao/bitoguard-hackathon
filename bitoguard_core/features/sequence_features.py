# bitoguard_core/features/sequence_features.py
from __future__ import annotations
import numpy as np
import pandas as pd
from features.swap_features import SWAP_ORDER_TYPE


def _cross_table_within(
    left: pd.DataFrame,
    right: pd.DataFrame,
    hours: float,
) -> pd.Series:
    """Count (left, right) pairs per user where 0 ≤ right_at − left_at ≤ hours.

    Uses binary-search per user group: O((n+m)·log m) vs O(n·m) for the naïve
    merge approach.  For tables with >10 k rows this is typically 20-100× faster.
    """
    if left.empty or right.empty:
        return pd.Series(dtype=int)

    common = set(left["user_id"].unique()) & set(right["user_id"].unique())
    if not common:
        return pd.Series(dtype=int)

    left_f  = left.loc[left["user_id"].isin(common), ["user_id", "occurred_at"]].copy()
    right_f = right.loc[right["user_id"].isin(common), ["user_id", "occurred_at"]].copy()

    for df in (left_f, right_f):
        df["occurred_at"] = pd.to_datetime(df["occurred_at"], utc=True, errors="coerce")

    left_f  = left_f.dropna(subset=["occurred_at"]).sort_values(["user_id", "occurred_at"])
    right_f = right_f.dropna(subset=["occurred_at"]).sort_values(["user_id", "occurred_at"])

    # Work in int64 nanoseconds for fast numpy searchsorted
    right_f["_ts_ns"] = right_f["occurred_at"].astype("int64")
    max_ns = int(hours * 3_600_000_000_000)

    results: dict[str, int] = {}
    right_grouped = right_f.groupby("user_id", sort=False)["_ts_ns"]
    for uid, l_grp in left_f.groupby("user_id", sort=False):
        try:
            r_ns = right_grouped.get_group(uid).values
        except KeyError:
            continue
        l_ns = l_grp["occurred_at"].astype("int64").values
        count = int(np.sum(
            np.searchsorted(r_ns, l_ns + max_ns, side="right") -
            np.searchsorted(r_ns, l_ns, side="left")
        ))
        if count:
            results[uid] = count

    return pd.Series(results)


def compute_sequence_features(
    fiat:   pd.DataFrame,
    trades: pd.DataFrame,
    crypto: pd.DataFrame,
) -> pd.DataFrame:
    """~10 cross-table sequence / timing features per user (lifetime)."""
    all_users: set[str] = set()
    for df in (fiat, trades, crypto):
        if not df.empty and "user_id" in df.columns:
            all_users.update(df["user_id"].dropna().unique())
    if not all_users:
        return pd.DataFrame()

    def _ts(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "occurred_at" in out.columns:
            out["occurred_at"] = pd.to_datetime(out["occurred_at"], utc=True, errors="coerce")
        return out.dropna(subset=["occurred_at"]) if "occurred_at" in out.columns else out

    fiat   = _ts(fiat)   if not fiat.empty   else fiat
    trades = _ts(trades) if not trades.empty else trades
    crypto = _ts(crypto) if not crypto.empty else crypto

    fiat_dep = (fiat[fiat.get("direction", pd.Series(dtype=str)) == "deposit"]
                if not fiat.empty else fiat)
    swap_buy = (trades[(trades.get("side", pd.Series(dtype=str)) == "buy") &
                       (trades.get("order_type", pd.Series(dtype=str)) == SWAP_ORDER_TYPE)]
                if not trades.empty else trades)
    crypto_dep = (crypto[crypto.get("direction", pd.Series(dtype=str)) == "deposit"]
                  if not crypto.empty else crypto)
    fiat_wdr   = (fiat[fiat.get("direction", pd.Series(dtype=str)) == "withdrawal"]
                  if not fiat.empty else fiat)

    base = pd.DataFrame({"user_id": sorted(all_users)})

    def _merge_counts(base: pd.DataFrame, counts: pd.Series, col_name: str) -> pd.DataFrame:
        if counts.empty:
            base[col_name] = 0
            return base
        counts_df = counts.reset_index()
        counts_df.columns = ["user_id", col_name]
        return base.merge(counts_df, on="user_id", how="left")

    for h, label in [(1, "1h"), (6, "6h"), (24, "24h"), (72, "72h")]:
        counts = _cross_table_within(fiat_dep, swap_buy, h)
        base = _merge_counts(base, counts, f"fiat_dep_to_swap_buy_within_{label}")

    for h, label in [(1, "1h"), (6, "6h"), (24, "24h"), (72, "72h")]:
        counts = _cross_table_within(crypto_dep, fiat_wdr, h)
        base = _merge_counts(base, counts, f"crypto_dep_to_fiat_wdr_within_{label}")

    # Fiat pass-through: fiat deposit → fiat withdrawal (money mule signal)
    # Legitimate users rarely deposit and immediately withdraw fiat without any investment.
    for h, label in [(24, "24h"), (72, "72h")]:
        counts = _cross_table_within(fiat_dep, fiat_wdr, h)
        base = _merge_counts(base, counts, f"fiat_dep_to_fiat_wdr_within_{label}")

    # Dwell hours: first fiat deposit to first fiat withdrawal
    if not fiat_dep.empty and not fiat_wdr.empty:
        first_dep = fiat_dep.groupby("user_id")["occurred_at"].min().rename("first_dep")
        first_wdr = fiat_wdr.groupby("user_id")["occurred_at"].min().rename("first_wdr")
        dwell     = first_dep.to_frame().join(first_wdr, how="inner")
        dwell["dwell_hours"] = (
            (dwell["first_wdr"] - dwell["first_dep"]).dt.total_seconds().div(3600).clip(lower=0)
        )
        base = base.merge(dwell[["dwell_hours"]].reset_index(), on="user_id", how="left")
    else:
        base["dwell_hours"] = 0.0

    # Early 3-day activity
    all_events = []
    for df, amt_col in [(fiat, "amount_twd"), (crypto, "amount_twd_equiv")]:
        if not df.empty and "user_id" in df.columns and amt_col in df.columns:
            all_events.append(df[["user_id", "occurred_at", amt_col]].rename(columns={amt_col: "_amt"}))
    if all_events:
        events = pd.concat(all_events, ignore_index=True)
        first_event = events.groupby("user_id")["occurred_at"].min().rename("first_event")
        events = events.merge(first_event, on="user_id")
        events["days_since_first"] = (
            (events["occurred_at"] - events["first_event"]).dt.total_seconds().div(86400)
        )
        early = events[events["days_since_first"] <= 3]
        early_vol   = early.groupby("user_id")["_amt"].sum().rename("early_3d_volume")
        early_count = early.groupby("user_id").size().rename("early_3d_count")
        base = base.merge(early_vol.reset_index(), on="user_id", how="left")
        base = base.merge(early_count.reset_index(), on="user_id", how="left")
    else:
        base["early_3d_volume"] = 0.0
        base["early_3d_count"]  = 0

    return base.fillna(0).reset_index(drop=True)
