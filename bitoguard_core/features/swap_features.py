# bitoguard_core/features/swap_features.py
"""USDT instant-swap features. Input: trade_orders filtered to order_type='instant_swap'."""
from __future__ import annotations
import pandas as pd
from features.twd_features import _agg_stats

SWAP_ORDER_TYPE = "instant_swap"


def compute_swap_features(trades: pd.DataFrame) -> pd.DataFrame:
    """11 USDT instant-swap features per user. Caller may pass all trade_orders;
    this function filters to instant_swap rows internally."""
    if trades.empty:
        return pd.DataFrame()

    df = trades.copy()
    df["occurred_at"] = pd.to_datetime(df["occurred_at"], utc=True, errors="coerce")
    if "order_type" in df.columns:
        df = df[df["order_type"] == SWAP_ORDER_TYPE]
    if df.empty:
        return pd.DataFrame()

    df = df.dropna(subset=["user_id", "occurred_at"])
    df["notional_twd"] = df["notional_twd"].fillna(0.0)

    rows = []
    for uid, grp in df.groupby("user_id"):
        buy  = grp[grp["side"] == "buy"]
        sell = grp[grp["side"] == "sell"]
        row: dict = {"user_id": uid}
        row.update(_agg_stats(grp["notional_twd"], "swap"))
        row["swap_count"]        = int(len(grp))
        row["swap_buy_count"]    = int(len(buy))
        row["swap_sell_count"]   = int(len(sell))
        row["swap_buy_twd_sum"]  = float(buy["notional_twd"].sum())
        row["swap_sell_twd_sum"] = float(sell["notional_twd"].sum())
        row["swap_net_twd"]      = float(buy["notional_twd"].sum() - sell["notional_twd"].sum())
        row["swap_buy_ratio"]    = float(len(buy) / max(1, len(grp)))
        row["swap_active_days"]  = int(grp["occurred_at"].dt.date.nunique())
        span = (grp["occurred_at"].max() - grp["occurred_at"].min()).total_seconds() / 86400
        row["swap_span_days"]    = float(max(0.0, span))
        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)
