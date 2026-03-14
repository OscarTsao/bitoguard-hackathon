# bitoguard_core/features/twd_features.py
from __future__ import annotations
import pandas as pd

NIGHT_HOURS = frozenset(range(0, 6))  # 00:00–05:59 UTC


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
        row["twd_night_share"]  = float((grp["occurred_at"].dt.hour.isin(NIGHT_HOURS)).mean())

        for prefix, subset in [("twd_all", grp), ("twd_dep", dep), ("twd_wdr", wdr)]:
            g = _gap_stats(subset["occurred_at"])
            row[f"{prefix}_gap_min"]         = g["gap_min"]
            row[f"{prefix}_gap_p10"]         = g["gap_p10"]
            row[f"{prefix}_gap_median"]      = g["gap_median"]
            row[f"{prefix}_rapid_1h_share"]  = g["rapid_1h_share"]

        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)
