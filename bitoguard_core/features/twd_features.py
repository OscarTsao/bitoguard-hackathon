# bitoguard_core/features/twd_features.py
from __future__ import annotations
import pandas as pd

import numpy as np

NIGHT_HOURS = frozenset(range(0, 6))  # 00:00–05:59 UTC


def _amount_entropy(amounts: pd.Series, n_bins: int = 20) -> float:
    """Shannon entropy of amount distribution (normalized to [0, 1]).

    Low entropy = all transactions at similar amounts (mule / structuring signal).
    High entropy = naturally varied amounts (typical legitimate user).
    """
    if len(amounts) < 2:
        return 0.0
    counts, _ = np.histogram(amounts.dropna(), bins=n_bins)
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)) / np.log(n_bins))


def _gap_stats(times: pd.Series) -> dict:
    """Inter-arrival gap stats in minutes. Returns zeros for fewer than 2 events."""
    times = pd.to_datetime(times, utc=True, errors="coerce").dropna().sort_values()
    if len(times) < 2:
        return {"gap_min": 0.0, "gap_p10": 0.0, "gap_median": 0.0, "rapid_1h_share": 0.0}
    gaps = times.diff().dropna().dt.total_seconds().div(60)
    return {
        "gap_min":         float(gaps.min()),
        "gap_p10":         float(gaps.quantile(0.10)),
        "gap_median":      float(gaps.median()),
        "rapid_1h_share":  float((gaps <= 60).mean()),
    }


def _agg_stats(series: pd.Series, prefix: str) -> dict:
    if series.empty:
        return {f"{prefix}_{s}": 0.0 for s in ("count", "sum", "mean", "median", "std", "max", "p90")}
    return {
        f"{prefix}_count":  float(len(series)),
        f"{prefix}_sum":    float(series.sum()),
        f"{prefix}_mean":   float(series.mean()),
        f"{prefix}_median": float(series.median()),
        f"{prefix}_std":    float(series.std(ddof=0)),
        f"{prefix}_max":    float(series.max()),
        f"{prefix}_p90":    float(series.quantile(0.90)),
    }


def compute_twd_features(
    fiat: pd.DataFrame,
    snapshot_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """~26 TWD fiat transfer features per user (lifetime, no IP — use ip_features.py)."""
    if fiat.empty:
        return pd.DataFrame()

    ref = pd.Timestamp.now(tz="UTC") if snapshot_date is None else snapshot_date
    if ref.tzinfo is None:
        ref = ref.tz_localize("UTC")

    df = fiat.copy()
    df["occurred_at"] = pd.to_datetime(df["occurred_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["user_id", "occurred_at", "amount_twd"])

    rows = []
    for uid, grp in df.groupby("user_id"):
        dep = grp[grp["direction"] == "deposit"]
        wdr = grp[grp["direction"] == "withdrawal"]
        row: dict = {"user_id": uid}

        row.update(_agg_stats(grp["amount_twd"], "twd_all"))
        row.update(_agg_stats(dep["amount_twd"],  "twd_dep"))
        row.update(_agg_stats(wdr["amount_twd"],  "twd_wdr"))
        row["twd_net_flow"] = float(dep["amount_twd"].sum() - wdr["amount_twd"].sum())

        row["twd_active_days"] = int(grp["occurred_at"].dt.date.nunique())
        span = (grp["occurred_at"].max() - grp["occurred_at"].min()).total_seconds() / 86400
        row["twd_span_days"]    = float(max(0.0, span))
        recency = (ref - grp["occurred_at"].max()).total_seconds() / 86400
        row["twd_recency_days"] = float(max(0.0, recency))
        row["twd_night_share"]   = float((grp["occurred_at"].dt.hour.isin(NIGHT_HOURS)).mean())
        # Weekend share: AML bots and money mules operate on weekends; legitimate users skew weekdays
        row["twd_weekend_share"] = float((grp["occurred_at"].dt.dayofweek >= 5).mean())

        for prefix, subset in [("twd_all", grp), ("twd_dep", dep), ("twd_wdr", wdr)]:
            g = _gap_stats(subset["occurred_at"])
            row[f"{prefix}_gap_min"]         = g["gap_min"]
            row[f"{prefix}_gap_p10"]         = g["gap_p10"]
            row[f"{prefix}_gap_median"]      = g["gap_median"]
            row[f"{prefix}_rapid_1h_share"]  = g["rapid_1h_share"]

        # --- Windowed velocity features (7d and 30d) ---
        # Burst: sudden activity spikes after dormancy are a primary AML signal
        dep_7d  = dep[dep["occurred_at"] >= ref - pd.Timedelta(days=7)]
        dep_30d = dep[dep["occurred_at"] >= ref - pd.Timedelta(days=30)]
        wdr_7d  = wdr[wdr["occurred_at"] >= ref - pd.Timedelta(days=7)]
        wdr_30d = wdr[wdr["occurred_at"] >= ref - pd.Timedelta(days=30)]

        row["twd_dep_7d_count"]  = int(len(dep_7d))
        row["twd_dep_7d_sum"]    = float(dep_7d["amount_twd"].sum())
        row["twd_wdr_7d_count"]  = int(len(wdr_7d))
        row["twd_wdr_7d_sum"]    = float(wdr_7d["amount_twd"].sum())
        row["twd_dep_30d_count"] = int(len(dep_30d))
        row["twd_dep_30d_sum"]   = float(dep_30d["amount_twd"].sum())
        row["twd_wdr_30d_count"] = int(len(wdr_30d))
        row["twd_wdr_30d_sum"]   = float(wdr_30d["amount_twd"].sum())

        # Burst ratio: 7-day activity vs. expected based on lifetime daily average
        # >1 means recent spike; heavily suspicious at >5x
        lifetime_dep_sum = float(dep["amount_twd"].sum())
        daily_avg_dep    = lifetime_dep_sum / max(row["twd_span_days"], 7.0)
        row["twd_dep_burst_ratio"] = float(
            row["twd_dep_7d_sum"] / max(daily_avg_dep * 7, 1.0)
        )

        # Amount entropy: normalized Shannon entropy over binned deposit amounts.
        # Low entropy → same amount repeated = structuring; High → naturally varied.
        row["twd_dep_amt_entropy"] = _amount_entropy(dep["amount_twd"]) if not dep.empty else 0.0

        # Structuring score: fraction of deposits clustering near round NT$ thresholds
        # NT$500,000 is a common AML reporting threshold; structuring deposits appear
        # just below (e.g. 480k-499k). Round-number ratio captures similar smurfing.
        if not dep.empty:
            amounts = dep["amount_twd"]
            # Fraction that are multiples of NT$10,000 (common structuring denomination)
            row["twd_dep_round_10k_ratio"]  = float((amounts % 10_000 < 1.0).mean())
            # Fraction just below NT$500k threshold (within 10% below)
            row["twd_dep_near_500k_ratio"]  = float(
                ((amounts >= 450_000) & (amounts < 500_000)).mean()
            )
        else:
            row["twd_dep_round_10k_ratio"] = 0.0
            row["twd_dep_near_500k_ratio"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)
