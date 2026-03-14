# bitoguard_core/features/trading_features.py
"""Book-order trade features (excludes instant_swap). No ip_address on this table;
use ip_features.py for per-user IP diversity."""
from __future__ import annotations
import pandas as pd
from features.twd_features import _agg_stats
from features.swap_features import SWAP_ORDER_TYPE

NIGHT_HOURS = frozenset(range(0, 6))


def compute_trading_features(trades: pd.DataFrame) -> pd.DataFrame:
    """12 book-order trade aggregates per user (excludes instant_swap rows)."""
    if trades.empty:
        return pd.DataFrame()

    df = trades.copy()
    df["occurred_at"] = pd.to_datetime(df["occurred_at"], utc=True, errors="coerce")
    if "order_type" in df.columns:
        df = df[df["order_type"] != SWAP_ORDER_TYPE]
    if df.empty:
        return pd.DataFrame()

    df = df.dropna(subset=["user_id", "occurred_at"])
    df["notional_twd"] = df["notional_twd"].fillna(0.0)

    rows = []
    for uid, grp in df.groupby("user_id"):
        buy  = grp[grp["side"] == "buy"]
        sell = grp[grp["side"] == "sell"]
        row: dict = {"user_id": uid}
        row.update(_agg_stats(grp["notional_twd"], "trade"))
        row["trade_count"]        = int(len(grp))
        row["trade_buy_count"]    = int(len(buy))
        row["trade_sell_count"]   = int(len(sell))
        row["trade_buy_twd_sum"]  = float(buy["notional_twd"].sum())
        row["trade_sell_twd_sum"] = float(sell["notional_twd"].sum())
        row["trade_net_twd"]      = float(buy["notional_twd"].sum() - sell["notional_twd"].sum())
        row["trade_buy_ratio"]    = float(len(buy) / max(1, len(grp)))
        if "order_type" in grp.columns:
            row["trade_market_ratio"] = float((grp["order_type"] == "market").mean())
        else:
            row["trade_market_ratio"] = 0.0
        row["trade_active_days"] = int(grp["occurred_at"].dt.date.nunique())
        span = (grp["occurred_at"].max() - grp["occurred_at"].min()).total_seconds() / 86400
        row["trade_span_days"]   = float(max(0.0, span))
        row["trade_night_share"] = float((grp["occurred_at"].dt.hour.isin(NIGHT_HOURS)).mean())
        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)
