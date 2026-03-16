# bitoguard_core/features/crypto_features.py
from __future__ import annotations
import pandas as pd
from features.twd_features import _agg_stats, _amount_entropy, _gap_stats

TRX_ASSETS = frozenset({"TRX", "TRC20"})


def compute_crypto_features(
    crypto: pd.DataFrame,
    snapshot_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """~43 crypto transfer features per user (lifetime + 7d/30d windows)."""
    if crypto.empty:
        return pd.DataFrame()

    ref = pd.Timestamp.now(tz="UTC") if snapshot_date is None else snapshot_date
    if ref.tzinfo is None:
        ref = ref.tz_localize("UTC")

    df = crypto.copy()
    df["occurred_at"] = pd.to_datetime(df["occurred_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["user_id", "occurred_at"])
    df["amount_twd_equiv"] = df["amount_twd_equiv"].fillna(0.0)

    rows = []
    for uid, grp in df.groupby("user_id"):
        dep = grp[grp["direction"] == "deposit"]
        wdr = grp[grp["direction"] == "withdrawal"]
        row: dict = {"user_id": uid}

        row.update(_agg_stats(grp["amount_twd_equiv"], "crypto_all_twd"))
        row.update(_agg_stats(dep["amount_twd_equiv"], "crypto_dep_twd"))
        row.update(_agg_stats(wdr["amount_twd_equiv"], "crypto_wdr_twd"))
        row["crypto_all_count"] = int(len(grp))
        row["crypto_dep_count"] = int(len(dep))
        row["crypto_wdr_count"] = int(len(wdr))
        row["crypto_net_flow_twd"] = float(dep["amount_twd_equiv"].sum() - wdr["amount_twd_equiv"].sum())

        row["crypto_n_currencies"] = int(grp["asset"].nunique()) if "asset" in grp else 0
        row["crypto_n_protocols"]  = int(grp["network"].nunique()) if "network" in grp else 0

        if "asset" in grp.columns and len(grp) > 0:
            trx_mask = grp["asset"].str.upper().isin(TRX_ASSETS)
            row["crypto_trx_tx_share"]  = float(trx_mask.mean())
            total_amt = grp["amount_twd_equiv"].sum()
            row["crypto_trx_amt_share"] = float(
                grp.loc[trx_mask, "amount_twd_equiv"].sum() / max(1.0, total_amt)
            )
        else:
            row["crypto_trx_tx_share"] = 0.0
            row["crypto_trx_amt_share"] = 0.0

        row["crypto_n_from_wallets"] = int(dep["counterparty_wallet_id"].nunique()) if not dep.empty else 0
        row["crypto_n_to_wallets"]   = int(wdr["counterparty_wallet_id"].nunique()) if not wdr.empty else 0
        if not dep.empty and "counterparty_wallet_id" in dep.columns:
            cp_counts = dep["counterparty_wallet_id"].dropna().value_counts(normalize=True)
            row["crypto_from_wallet_conc"] = float(cp_counts.iloc[0]) if not cp_counts.empty else 0.0
        else:
            row["crypto_from_wallet_conc"] = 0.0

        # Amount entropy for crypto withdrawals: low entropy = repeated same-amount withdrawals
        # (typical of automated layering); high entropy = naturally varied (legitimate trading).
        row["crypto_wdr_amt_entropy"] = _amount_entropy(wdr["amount_twd_equiv"]) if not wdr.empty else 0.0
        row["crypto_dep_amt_median"] = float(dep["amount_twd_equiv"].median()) if not dep.empty else 0.0
        dep_mean = float(dep["amount_twd_equiv"].mean()) if not dep.empty else 1.0
        dep_std  = float(dep["amount_twd_equiv"].std(ddof=0)) if not dep.empty else 0.0
        row["crypto_dep_amt_cv"] = dep_std / max(1.0, dep_mean)
        row["crypto_wdr_to_dep_ratio"] = float(
            wdr["amount_twd_equiv"].sum() / max(1.0, dep["amount_twd_equiv"].sum())
        )

        row["crypto_active_days"]   = int(grp["occurred_at"].dt.date.nunique())
        span = (grp["occurred_at"].max() - grp["occurred_at"].min()).total_seconds() / 86400
        row["crypto_span_days"]     = float(max(0.0, span))
        row["crypto_weekend_share"] = float((grp["occurred_at"].dt.dayofweek >= 5).mean())

        for prefix, subset in [("crypto_all", grp), ("crypto_dep", dep), ("crypto_wdr", wdr)]:
            g = _gap_stats(subset["occurred_at"])
            row[f"{prefix}_gap_min"]         = g["gap_min"]
            row[f"{prefix}_gap_p10"]         = g["gap_p10"]
            row[f"{prefix}_gap_median"]      = g["gap_median"]
            row[f"{prefix}_rapid_1h_share"]  = g["rapid_1h_share"]

        # --- Windowed velocity features (7d and 30d) ---
        # Crypto withdrawals shortly after fiat deposits = primary cash-out indicator
        wdr_7d  = wdr[wdr["occurred_at"] >= ref - pd.Timedelta(days=7)]
        wdr_30d = wdr[wdr["occurred_at"] >= ref - pd.Timedelta(days=30)]
        dep_7d  = dep[dep["occurred_at"] >= ref - pd.Timedelta(days=7)]

        row["crypto_wdr_7d_count"]  = int(len(wdr_7d))
        row["crypto_wdr_7d_sum"]    = float(wdr_7d["amount_twd_equiv"].sum())
        row["crypto_wdr_30d_count"] = int(len(wdr_30d))
        row["crypto_wdr_30d_sum"]   = float(wdr_30d["amount_twd_equiv"].sum())
        row["crypto_dep_7d_count"]  = int(len(dep_7d))
        row["crypto_dep_7d_sum"]    = float(dep_7d["amount_twd_equiv"].sum())

        # Burst ratio: recent 7-day crypto withdrawal vs. expected from lifetime average
        crypto_span = max(row["crypto_span_days"], 7.0)
        lifetime_daily_wdr = float(wdr["amount_twd_equiv"].sum()) / crypto_span
        row["crypto_wdr_burst_ratio"] = float(
            row["crypto_wdr_7d_sum"] / max(lifetime_daily_wdr * 7, 1.0)
        )

        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)
