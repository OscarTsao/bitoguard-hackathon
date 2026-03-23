"""Transaction-level feature engineering.

For each transaction, compute contextual features relative to:
- The user's own history (is this amount unusual? is the timing unusual?)
- Global baselines (is this a round amount? is it at night?)
- Sequential context (what happened before/after this transaction?)

Output: DataFrame with 1 row per transaction, columns = features + user_id + tx_id + label
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path


def build_tx_features(data_dir: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Build per-transaction feature vectors from all 4 event tables."""
    data_dir = Path(data_dir)

    twd = pd.read_parquet(data_dir / "twd_transfer.parquet")
    crypto = pd.read_parquet(data_dir / "crypto_transfer.parquet")
    swap = pd.read_parquet(data_dir / "usdt_swap.parquet")
    trade = pd.read_parquet(data_dir / "usdt_twd_trading.parquet")

    rows = []

    # TWD transfers
    df = twd.copy()
    df["tx_type"] = df["is_deposit"].map({True: 0, False: 1})
    df["amount"] = pd.to_numeric(df.get("amount_twd", pd.Series(0, index=df.index)), errors="coerce").fillna(0).abs()
    df["ts"] = pd.to_datetime(df["created_at"], utc=True)
    df["ip"] = df.get("source_ip_hash", pd.Series("", index=df.index)).fillna("")
    df["table"] = "twd"
    rows.append(df[["id", "user_id", "ts", "tx_type", "amount", "ip", "table"]])

    # Crypto transfers
    df = crypto.copy()
    kind = df.get("kind_label", pd.Series("deposit", index=df.index)).fillna("deposit").str.lower()
    internal = df.get("is_internal_transfer", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    deposit_mask = kind.str.contains("deposit", na=False)
    df["tx_type"] = 5  # default: external withdrawal
    df.loc[deposit_mask & internal, "tx_type"] = 2
    df.loc[deposit_mask & ~internal, "tx_type"] = 3
    df.loc[~deposit_mask & internal, "tx_type"] = 4
    df["amount"] = pd.to_numeric(df.get("amount_twd_equiv", df.get("amount_twd", pd.Series(0, index=df.index))), errors="coerce").fillna(0).abs()
    df["ts"] = pd.to_datetime(df["created_at"], utc=True)
    df["ip"] = df.get("source_ip_hash", pd.Series("", index=df.index)).fillna("")
    df["table"] = "crypto"
    rows.append(df[["id", "user_id", "ts", "tx_type", "amount", "ip", "table"]])

    # Swaps
    df = swap.copy()
    kind = df.get("kind_label", pd.Series("buy", index=df.index)).fillna("buy").str.lower()
    df["tx_type"] = kind.map(lambda x: 6 if "buy" in str(x) else 7)
    df["amount"] = pd.to_numeric(df.get("twd_amount", df.get("amount_twd", pd.Series(0, index=df.index))), errors="coerce").fillna(0).abs()
    df["ts"] = pd.to_datetime(df["created_at"], utc=True)
    df["ip"] = ""
    df["table"] = "swap"
    rows.append(df[["id", "user_id", "ts", "tx_type", "amount", "ip", "table"]])

    # Trades
    df = trade.copy()
    if "is_buy" in df.columns:
        df["tx_type"] = df["is_buy"].map({True: 8, False: 9})
    else:
        side = df.get("side_label", pd.Series("buy", index=df.index)).fillna("buy").str.lower()
        df["tx_type"] = side.map(lambda x: 8 if "buy" in str(x) else 9)
    df["amount"] = pd.to_numeric(df.get("trade_notional_twd", df.get("amount_twd", pd.Series(0, index=df.index))), errors="coerce").fillna(0).abs()
    ts_col = "updated_at" if "updated_at" in df.columns else "created_at"
    df["ts"] = pd.to_datetime(df[ts_col], utc=True)
    df["ip"] = df.get("source_ip_hash", pd.Series("", index=df.index)).fillna("")
    df["table"] = "trade"
    rows.append(df[["id", "user_id", "ts", "tx_type", "amount", "ip", "table"]])

    all_tx = pd.concat(rows, ignore_index=True)
    all_tx = all_tx.sort_values(["user_id", "ts"]).reset_index(drop=True)
    print(f"[tx_features] Unified {len(all_tx)} transactions from 4 tables")

    # Per-user statistics (for relative features)
    user_stats = all_tx.groupby("user_id").agg(
        user_mean_amount=("amount", "mean"),
        user_std_amount=("amount", "std"),
        user_median_amount=("amount", "median"),
        user_tx_count=("id", "count"),
    ).reset_index()
    user_stats["user_std_amount"] = user_stats["user_std_amount"].fillna(0)
    all_tx = all_tx.merge(user_stats, on="user_id", how="left")

    # 1. Amount features
    all_tx["log_amount"] = np.log1p(all_tx["amount"])
    all_tx["amount_zscore"] = np.where(
        all_tx["user_std_amount"] > 0,
        (all_tx["amount"] - all_tx["user_mean_amount"]) / all_tx["user_std_amount"],
        0.0
    )
    all_tx["amount_vs_median"] = np.where(
        all_tx["user_median_amount"] > 0,
        all_tx["amount"] / all_tx["user_median_amount"],
        0.0
    )

    # 2. Round amount (structuring detection)
    round_thresholds = np.array([10000, 20000, 30000, 50000, 100000, 150000, 200000, 300000, 500000])
    amounts = all_tx["amount"].values
    near_round = np.zeros(len(amounts), dtype=float)
    for rt in round_thresholds:
        near_round = np.minimum(near_round + (np.abs(amounts - rt) / max(rt, 1) < 0.02).astype(float), 1.0)
    all_tx["near_round_flag"] = np.where(amounts > 0, near_round, 0.0)
    all_tx["just_under_50k"] = ((amounts >= 45000) & (amounts < 50000)).astype(float)

    # 3. Time features
    all_tx["hour"] = all_tx["ts"].dt.tz_convert("Asia/Taipei").dt.hour
    all_tx["is_night"] = ((all_tx["hour"] >= 23) | (all_tx["hour"] < 5)).astype(float)
    all_tx["is_weekend"] = all_tx["ts"].dt.dayofweek.isin([5, 6]).astype(float)
    all_tx["hour_sin"] = np.sin(2 * np.pi * all_tx["hour"] / 24)
    all_tx["hour_cos"] = np.cos(2 * np.pi * all_tx["hour"] / 24)

    # 4. Sequential context (per-user)
    all_tx["prev_amount"] = all_tx.groupby("user_id")["amount"].shift(1).fillna(0)
    all_tx["prev_type"] = all_tx.groupby("user_id")["tx_type"].shift(1).fillna(-1)
    all_tx["delta_seconds"] = all_tx.groupby("user_id")["ts"].diff().dt.total_seconds().fillna(0)
    all_tx["delta_hours"] = (all_tx["delta_seconds"] / 3600).clip(0, 8760)
    all_tx["log_delta_hours"] = np.log1p(all_tx["delta_hours"])
    all_tx["same_amount_as_prev"] = (
        (all_tx["amount"] > 0) &
        (np.abs(all_tx["amount"] - all_tx["prev_amount"]) / all_tx["amount"].clip(lower=1) < 0.01)
    ).astype(float)

    # 5. IP features
    all_tx["prev_ip"] = all_tx.groupby("user_id")["ip"].shift(1).fillna("")
    all_tx["ip_changed"] = ((all_tx["ip"] != "") & (all_tx["prev_ip"] != "") &
                             (all_tx["ip"] != all_tx["prev_ip"])).astype(float)

    # 6. Type transition
    all_tx["is_deposit_type"] = all_tx["tx_type"].isin([0, 2, 3]).astype(float)
    all_tx["is_withdrawal_type"] = all_tx["tx_type"].isin([1, 4, 5]).astype(float)
    all_tx["is_conversion_type"] = all_tx["tx_type"].isin([6, 7, 8, 9]).astype(float)
    all_tx["prev_was_deposit"] = all_tx.groupby("user_id")["is_deposit_type"].shift(1).fillna(0)
    all_tx["deposit_then_convert"] = (
        all_tx["prev_was_deposit"] * all_tx["is_conversion_type"]
    )

    # 7. Position in user sequence
    all_tx["tx_position"] = all_tx.groupby("user_id").cumcount()
    all_tx["tx_position_pct"] = all_tx["tx_position"] / all_tx["user_tx_count"].clip(lower=1)

    feature_cols = [
        "tx_type", "log_amount", "amount_zscore", "amount_vs_median",
        "near_round_flag", "just_under_50k",
        "hour_sin", "hour_cos", "is_night", "is_weekend",
        "log_delta_hours", "same_amount_as_prev", "ip_changed",
        "is_deposit_type", "is_withdrawal_type", "is_conversion_type",
        "deposit_then_convert", "prev_type",
        "tx_position_pct",
    ]

    output = all_tx[["id", "user_id", "ts", "table"] + feature_cols].copy()
    print(f"[tx_features] {len(feature_cols)} features per transaction, {len(output)} rows total")
    return output, feature_cols
