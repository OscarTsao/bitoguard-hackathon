"""Phase 2A: Temporal Pattern Features — Opus-Validated (2026-03-20).

23 label-free features across 5 categories, validated on 51,017 labeled users.
Signal strengths (individual AUC / AP on labeled subset):
  burst_max_amount_1h_log  AUC=0.662 AP=0.064  corr_existing=0.297  ← BEST
  burst_max_txns_1h        AUC=0.651 AP=0.056  corr_existing=0.512
  burst_dormancy_score     AUC=0.640 AP=0.053  corr_existing=0.575
  cycle_n_complete         AUC=0.629 AP=0.062  corr_existing=0.576
  cycle_velocity_score     AUC=0.626 AP=0.066  corr_existing=~0.57
  struct_above_50k_ratio   AUC=0.624 AP=0.043  corr_existing=0.128  ← MOST NOVEL
  burst_inter_deposit_cv   AUC=0.624 AP=0.050  corr_existing=~0.59
  cycle_fiat_to_swap_eff   AUC=0.622 AP=0.043  corr_existing=low
  cycle_has_complete       AUC=0.620 AP=0.049  corr_existing=mod
  struct_deposit_amt_gini  AUC=0.602 AP=0.043  corr_existing=mod
  struct_near_50k_count    AUC=0.600 AP=0.060  corr_existing=mod
  struct_near_50k_ratio    AUC=0.589 AP=0.039  corr_existing=0.128
  ip_unique_per_100_txns   AUC=0.580 AP=0.047  corr_existing=mod
  ip_max_in_24h            AUC=0.572 AP=0.042  corr_existing=mod

Skipped (validated as no signal): ip_change_at_deposit, fiat_to_crypto_median_h,
  weekend_ratio, swap_burst_30min, avg_cycle_efficiency.

Entry point: build_temporal_features(dataset, skip_layering=True) -> pd.DataFrame
All 63,770 users covered; users with 0 events get fillna=0.
Runtime: ~25-35 seconds total.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from official.common import load_clean_table

# Final validated feature columns (23 total, after alias deduplication)
TEMPORAL_FEATURE_COLUMNS: list[str] = [
    # A: Structuring (6)
    "struct_near_50k_ratio",
    "struct_near_100k_ratio",
    "struct_near_50k_count",
    "struct_above_50k_ratio",
    "struct_deposit_amt_gini",
    "struct_rapid_deposit_triplets",
    # B: Layering cycles (5)
    "cycle_n_complete",
    "cycle_has_complete",
    "cycle_velocity_score",
    "cycle_fiat_to_swap_eff",
    "cycle_swap_to_crypto_eff",
    # C: Dormancy-burst (5)
    "burst_max_txns_1h",
    "burst_max_amount_1h_log",
    "burst_dormancy_score",
    "burst_n_activity_peaks",
    "burst_inter_deposit_cv",
    # D: IP behavior (4)
    "ip_unique_per_100_txns",
    "ip_max_in_24h",
    "ip_deposit_change_ratio",
    "ip_crypto_diversity_per_txn",
    # E: Cross-channel timing (3)
    "timing_wallet_turnover",
    "timing_cross_channel_count",
    "timing_night_all_channel_ratio",
]


def build_temporal_features(
    dataset: pd.DataFrame,
    skip_layering: bool = True,
) -> pd.DataFrame:
    """Build 23 validated temporal features for all users.

    Parameters
    ----------
    dataset : pd.DataFrame
        Full user dataset (provides user_ids including predict_only).
    skip_layering : bool
        Unused — kept for backward-compatibility. All features are fast.

    Returns
    -------
    pd.DataFrame with columns [user_id] + TEMPORAL_FEATURE_COLUMNS.
    """
    t0 = time.time()
    all_user_ids = dataset["user_id"].astype(int).drop_duplicates()
    n_users = len(all_user_ids)
    uid_set = set(all_user_ids.tolist())
    print(f"[temporal_features] Building {len(TEMPORAL_FEATURE_COLUMNS)} features for {n_users} users...", flush=True)

    # ── Load tables ─────────────────────────────────────────────────────────
    twd = load_clean_table("twd_transfer").copy()
    crypto = load_clean_table("crypto_transfer").copy()
    swap = load_clean_table("usdt_swap").copy()
    trade = load_clean_table("usdt_twd_trading").copy()
    print(f"[temporal_features] Loaded: twd={len(twd)}, crypto={len(crypto)}, "
          f"swap={len(swap)}, trade={len(trade)}", flush=True)

    # Filter to known users
    twd = twd[twd["user_id"].isin(uid_set)].copy()
    crypto = crypto[crypto["user_id"].isin(uid_set)].copy()
    swap = swap[swap["user_id"].isin(uid_set)].copy()
    trade = trade[trade["user_id"].isin(uid_set)].copy()

    # Parse timestamps
    twd["created_at"] = pd.to_datetime(twd["created_at"], utc=True, errors="coerce")
    crypto["created_at"] = pd.to_datetime(crypto["created_at"], utc=True, errors="coerce")
    swap["created_at"] = pd.to_datetime(swap["created_at"], utc=True, errors="coerce")
    trade["updated_at"] = pd.to_datetime(trade.get("updated_at", pd.Series(dtype="object")), utc=True, errors="coerce")
    if "created_at" in trade.columns and "updated_at" not in trade.columns:
        trade["updated_at"] = pd.to_datetime(trade["created_at"], utc=True, errors="coerce")
    trade_ts_col = "updated_at" if "updated_at" in trade.columns else "created_at"

    base = all_user_ids.to_frame("user_id")

    # ── Category A: Structuring ──────────────────────────────────────────────
    deps = twd[twd["is_deposit"] == True].copy()
    deps = deps.dropna(subset=["amount_twd", "created_at"])
    deps = deps[deps["amount_twd"] > 0]

    if not deps.empty:
        amt = deps["amount_twd"]
        deps = deps.assign(
            near_50k=((amt >= 49500) & (amt <= 50500)).astype(float),
            near_100k=((amt >= 99000) & (amt <= 101000)).astype(float),
            above_50k=(amt >= 50000).astype(float),
        )
        struct_agg = deps.groupby("user_id").agg(
            _total=("amount_twd", "count"),
            _n50k=("near_50k", "sum"),
            _n100k=("near_100k", "sum"),
            _nabove=("above_50k", "sum"),
        )
        struct_agg["struct_near_50k_ratio"] = (struct_agg["_n50k"] / struct_agg["_total"]).clip(0, 1)
        struct_agg["struct_near_100k_ratio"] = (struct_agg["_n100k"] / struct_agg["_total"]).clip(0, 1)
        struct_agg["struct_near_50k_count"] = struct_agg["_n50k"].clip(0, 100).astype(int)
        struct_agg["struct_above_50k_ratio"] = (struct_agg["_nabove"] / struct_agg["_total"]).clip(0, 1)

        # Gini coefficient of deposit amounts
        def _gini(x: pd.Series) -> float:
            vals = np.sort(x.values)
            n = len(vals)
            total = vals.sum()
            if n < 2 or total == 0:
                return 0.0
            idx = np.arange(1, n + 1)
            return float((2 * np.dot(idx, vals) - (n + 1) * total) / (n * total))

        gini = deps.groupby("user_id")["amount_twd"].apply(_gini, include_groups=False).rename("struct_deposit_amt_gini")
        struct_agg = struct_agg.join(gini)
        struct_agg["struct_deposit_amt_gini"] = struct_agg["struct_deposit_amt_gini"].clip(0, 1)

        # Rapid deposit triplets: 3 deposits within 2 hours (vectorized with shift)
        dep_sorted = deps[["user_id", "created_at"]].sort_values(["user_id", "created_at"])
        dep_sorted = dep_sorted.assign(
            prev2=dep_sorted.groupby("user_id")["created_at"].shift(2)
        )
        dep_sorted["gap_h"] = (dep_sorted["created_at"] - dep_sorted["prev2"]).dt.total_seconds() / 3600
        triplets = (
            dep_sorted[dep_sorted["gap_h"] <= 2]
            .groupby("user_id")
            .size()
            .rename("struct_rapid_deposit_triplets")
        )
        struct_agg = struct_agg.join(triplets)
        struct_agg["struct_rapid_deposit_triplets"] = (
            struct_agg["struct_rapid_deposit_triplets"].fillna(0).clip(0, 50).astype(int)
        )

        feat_struct_cols = [
            "struct_near_50k_ratio", "struct_near_100k_ratio", "struct_near_50k_count",
            "struct_above_50k_ratio", "struct_deposit_amt_gini", "struct_rapid_deposit_triplets",
        ]
        feat_struct = base.merge(struct_agg[feat_struct_cols].reset_index(), on="user_id", how="left").fillna(0.0)
    else:
        feat_struct = base.assign(**{c: 0.0 for c in [
            "struct_near_50k_ratio", "struct_near_100k_ratio", "struct_near_50k_count",
            "struct_above_50k_ratio", "struct_deposit_amt_gini", "struct_rapid_deposit_triplets",
        ]})

    # ── Category B: Layering Cycles ─────────────────────────────────────────
    # n_fiat_deposits, n_swap_buys, n_crypto_withdrawals
    n_fiat_deps = (
        deps.groupby("user_id").size().rename("_n_fdep")
        if not deps.empty else pd.Series(dtype=int, name="_n_fdep")
    )
    # usdt_swap: kind_label values include "buy_usdt_with_twd" for buy side
    # Fallback: use is_deposit or kind_label upper-case contains "BUY"
    swap_buy_mask = swap["kind_label"].astype(str).str.lower().str.contains("buy")
    n_swap_buy = swap[swap_buy_mask].groupby("user_id").size().rename("_n_swap")
    crypto_wd = crypto[crypto["is_external_transfer"] == True]
    n_crypto_wd = crypto_wd.groupby("user_id").size().rename("_n_cwd")

    cycle_df = (
        base
        .merge(n_fiat_deps, on="user_id", how="left")
        .merge(n_swap_buy, on="user_id", how="left")
        .merge(n_crypto_wd, on="user_id", how="left")
        .fillna(0)
    )
    cycle_df["cycle_n_complete"] = np.minimum(
        np.minimum(cycle_df["_n_fdep"], cycle_df["_n_swap"]),
        cycle_df["_n_cwd"],
    ).clip(0, 200).astype(int)
    cycle_df["cycle_has_complete"] = (cycle_df["cycle_n_complete"] > 0).astype(int)

    # Span days for velocity
    ts_parts = []
    for df, tc in [(twd, "created_at"), (crypto, "created_at"), (swap, "created_at")]:
        if not df.empty and tc in df.columns:
            ts_parts.append(df[["user_id", tc]].dropna().rename(columns={tc: "ts"}))
    if ts_parts:
        all_ts = pd.concat(ts_parts, ignore_index=True)
        span = all_ts.groupby("user_id")["ts"].agg(span_days=lambda x: max(1.0, float(pd.Timedelta(x.max() - x.min()).total_seconds()) / 86400)).reset_index()
        cycle_df = cycle_df.merge(span, on="user_id", how="left")
    cycle_df["span_days"] = cycle_df.get("span_days", 1.0).fillna(1.0).clip(1.0, None)
    cycle_df["cycle_velocity_score"] = (
        np.log1p(cycle_df["cycle_n_complete"]) / (cycle_df["span_days"] + 1)
    ).clip(0, 2.0)

    # Cross-stage timing: fiat_deposit → next swap_buy (per user, merge_asof)
    def _stage_efficiency(left_df: pd.DataFrame, right_df: pd.DataFrame,
                          left_ts: str, right_ts: str) -> pd.Series:
        """1 / (median gap in hours from left event to next right event + 1).
        Uses np.searchsorted per user — no merge_asof timezone issues.
        """
        if left_df.empty or right_df.empty:
            return pd.Series(dtype="float32")
        lf = left_df[["user_id", left_ts]].dropna()
        rf = right_df[["user_id", right_ts]].dropna()
        # Build sorted int64 ns arrays per user for right side
        rf_by_user: dict = {
            uid: np.sort(g[right_ts].values.astype("int64"))
            for uid, g in rf.groupby("user_id")
        }
        result: dict = {}
        for uid, g in lf.groupby("user_id"):
            rt = rf_by_user.get(uid)
            if rt is None or len(rt) == 0:
                continue
            lt_vals = np.sort(g[left_ts].values.astype("int64"))
            gaps_ns = []
            for lt in lt_vals:
                idx = np.searchsorted(rt, lt)
                if idx < len(rt):
                    gaps_ns.append(rt[idx] - lt)
            if gaps_ns:
                median_h = float(np.median(gaps_ns)) / 3.6e12  # ns → hours
                result[uid] = float(1.0 / (median_h + 1.0))
        return pd.Series(result, dtype="float32")

    swap_buy_df = swap[swap_buy_mask][["user_id", "created_at"]].dropna()
    _eff_fs = _stage_efficiency(deps[["user_id", "created_at"]], swap_buy_df,
                                "created_at", "created_at")
    _eff_sc = _stage_efficiency(swap_buy_df, crypto_wd[["user_id", "created_at"]].dropna(),
                                "created_at", "created_at")
    # Series index is user_id integers — reset to get proper DataFrame columns
    eff_fs_df = _eff_fs.reset_index().rename(columns={"index": "user_id", 0: "cycle_fiat_to_swap_eff"})
    if "cycle_fiat_to_swap_eff" not in eff_fs_df.columns:
        eff_fs_df.columns = ["user_id", "cycle_fiat_to_swap_eff"]
    eff_sc_df = _eff_sc.reset_index().rename(columns={"index": "user_id", 0: "cycle_swap_to_crypto_eff"})
    if "cycle_swap_to_crypto_eff" not in eff_sc_df.columns:
        eff_sc_df.columns = ["user_id", "cycle_swap_to_crypto_eff"]

    cycle_df = cycle_df.merge(eff_fs_df, on="user_id", how="left")
    cycle_df = cycle_df.merge(eff_sc_df, on="user_id", how="left")
    cycle_df["cycle_fiat_to_swap_eff"] = cycle_df["cycle_fiat_to_swap_eff"].fillna(0.0)
    cycle_df["cycle_swap_to_crypto_eff"] = cycle_df["cycle_swap_to_crypto_eff"].fillna(0.0)

    feat_cycle = cycle_df[["user_id", "cycle_n_complete", "cycle_has_complete",
                            "cycle_velocity_score", "cycle_fiat_to_swap_eff", "cycle_swap_to_crypto_eff"]].copy()

    # ── Category C: Dormancy-Burst ───────────────────────────────────────────
    evts_parts = []
    for df, tc, ac in [
        (twd, "created_at", "amount_twd"),
        (crypto, "created_at", "amount_twd_equiv"),
        (swap, "created_at", "twd_amount"),
    ]:
        if not df.empty and tc in df.columns:
            sub = df[["user_id", tc]].dropna().rename(columns={tc: "ts"}).copy()
            if ac in df.columns:
                sub["amt"] = pd.to_numeric(df.loc[sub.index, ac], errors="coerce").fillna(0.0)
            else:
                sub["amt"] = 0.0
            evts_parts.append(sub)

    if evts_parts:
        all_evts = pd.concat(evts_parts, ignore_index=True).dropna(subset=["ts"])
        all_evts["hour"] = all_evts["ts"].dt.floor("h")

        # Hourly burst (txn count + amount)
        hourly = all_evts.groupby(["user_id", "hour"]).agg(
            _n=("ts", "count"),
            _amt=("amt", "sum"),
        )
        hourly_max = hourly.groupby("user_id").agg(
            burst_max_txns_1h=("_n", "max"),
            _max_amt=("_amt", "max"),
        )
        hourly_max["burst_max_amount_1h_log"] = np.log1p(hourly_max["_max_amt"]).clip(0, 20)
        hourly_max["burst_max_txns_1h"] = hourly_max["burst_max_txns_1h"].clip(0, 100)

        # Daily burst (dormancy score + peak days)
        all_evts["date"] = all_evts["ts"].dt.date
        daily = all_evts.groupby(["user_id", "date"]).size().reset_index(name="cnt")
        daily_stats = daily.groupby("user_id")["cnt"].agg(
            _max="max", _mean="mean", _count="count"
        )
        daily_stats["burst_dormancy_score"] = (
            daily_stats["_max"] / (daily_stats["_mean"] + 1e-6)
        ).clip(0, 100)
        peak_days = daily.groupby("user_id").apply(
            lambda g: int((g["cnt"] > g["cnt"].mean() * 3).sum()),
            include_groups=False,
        ).reset_index(name="burst_n_activity_peaks")
        daily_stats = daily_stats.reset_index().merge(peak_days, on="user_id", how="left")
        daily_stats["burst_n_activity_peaks"] = daily_stats["burst_n_activity_peaks"].fillna(0).clip(0, 30).astype(int)

        # Inter-deposit CV
        if not deps.empty:
            dep_ts_sorted = deps[["user_id", "created_at"]].dropna().sort_values(["user_id", "created_at"])

            def _cv(g: pd.Series) -> float:
                ts = g.values.astype("int64") / 1e9
                if len(ts) < 3:
                    return 0.0
                gaps = np.diff(ts)
                m = gaps.mean()
                return float(gaps.std() / (m + 1e-6)) if m > 0 else 0.0

            inter_cv = (
                dep_ts_sorted.groupby("user_id")["created_at"]
                .apply(_cv, include_groups=False)
                .reset_index(name="burst_inter_deposit_cv")
            )
            inter_cv["burst_inter_deposit_cv"] = inter_cv["burst_inter_deposit_cv"].clip(0, 10)
        else:
            inter_cv = pd.DataFrame({"user_id": pd.Series(dtype=int), "burst_inter_deposit_cv": pd.Series(dtype=float)})

        feat_burst = (
            base
            .merge(hourly_max[["burst_max_txns_1h", "burst_max_amount_1h_log", "_max_amt"]].reset_index(),
                   on="user_id", how="left")
            .merge(daily_stats[["user_id", "burst_dormancy_score", "burst_n_activity_peaks"]],
                   on="user_id", how="left")
            .merge(inter_cv, on="user_id", how="left")
            .fillna(0.0)
            .drop(columns=["_max_amt"], errors="ignore")
        )
    else:
        feat_burst = base.assign(**{c: 0.0 for c in [
            "burst_max_txns_1h", "burst_max_amount_1h_log",
            "burst_dormancy_score", "burst_n_activity_peaks", "burst_inter_deposit_cv",
        ]})

    # ── Category D: IP Behavior ──────────────────────────────────────────────
    ip_parts = []
    for df, tc in [(twd, "created_at"), (trade, trade_ts_col)]:
        if "source_ip_hash" in df.columns and not df.empty:
            sub = df[["user_id", tc, "source_ip_hash"]].dropna().copy()
            sub = sub[sub["source_ip_hash"].astype(str).str.strip().ne("")]
            sub = sub.rename(columns={tc: "ts"})
            ip_parts.append(sub)

    if ip_parts:
        ip_evts = pd.concat(ip_parts, ignore_index=True).dropna(subset=["ts"])

        ip_agg = ip_evts.groupby("user_id").agg(
            _uniq=("source_ip_hash", "nunique"),
            _total=("source_ip_hash", "count"),
        )
        ip_agg["ip_unique_per_100_txns"] = (
            ip_agg["_uniq"] / (ip_agg["_total"] + 1) * 100
        ).clip(0, 50)

        # Max unique IPs in any calendar day (using date)
        ip_evts["date"] = ip_evts["ts"].dt.date
        daily_ip = ip_evts.groupby(["user_id", "date"])["source_ip_hash"].nunique()
        max_daily_ip = daily_ip.groupby("user_id").max().rename("ip_max_in_24h").clip(0, 20).astype(int)

        feat_ip = (
            base
            .merge(ip_agg[["ip_unique_per_100_txns"]].reset_index(), on="user_id", how="left")
            .merge(max_daily_ip.reset_index(), on="user_id", how="left")
            .fillna(0.0)
        )
    else:
        feat_ip = base.assign(ip_unique_per_100_txns=0.0, ip_max_in_24h=0)

    # ip_deposit_change_ratio: fraction of consecutive TWD deposits with IP change
    if not deps.empty and "source_ip_hash" in deps.columns:
        dep_ip = deps[["user_id", "created_at", "source_ip_hash"]].dropna(subset=["source_ip_hash"]).copy()
        dep_ip = dep_ip.sort_values(["user_id", "created_at"])
        dep_ip["prev_ip"] = dep_ip.groupby("user_id")["source_ip_hash"].shift(1)
        dep_ip["changed"] = (dep_ip["source_ip_hash"] != dep_ip["prev_ip"]) & dep_ip["prev_ip"].notna()
        ip_change = dep_ip.groupby("user_id")["changed"].mean().rename("ip_deposit_change_ratio").clip(0, 1)
        feat_ip = feat_ip.merge(ip_change.reset_index(), on="user_id", how="left").fillna(0.0)
    else:
        feat_ip["ip_deposit_change_ratio"] = 0.0

    # ip_crypto_diversity_per_txn: unique IPs on crypto events / n_crypto_events
    if "source_ip_hash" in crypto.columns and not crypto.empty:
        crypto_ip = crypto[["user_id", "source_ip_hash"]].dropna(subset=["source_ip_hash"])
        crypto_ip = crypto_ip[crypto_ip["source_ip_hash"].astype(str).str.strip().ne("")]
        if not crypto_ip.empty:
            cr_ip_agg = crypto_ip.groupby("user_id").agg(
                _uniq=("source_ip_hash", "nunique"),
                _total=("source_ip_hash", "count"),
            )
            cr_ip_agg["ip_crypto_diversity_per_txn"] = (
                cr_ip_agg["_uniq"] / (cr_ip_agg["_total"] + 1)
            ).clip(0, 10)
            feat_ip = feat_ip.merge(
                cr_ip_agg[["ip_crypto_diversity_per_txn"]].reset_index(), on="user_id", how="left"
            ).fillna(0.0)
        else:
            feat_ip["ip_crypto_diversity_per_txn"] = 0.0
    else:
        feat_ip["ip_crypto_diversity_per_txn"] = 0.0

    # ── Category E: Cross-Channel Timing ────────────────────────────────────
    # timing_wallet_turnover: unique external wallets / span_days
    if not crypto_wd.empty and "to_wallet_hash" in crypto_wd.columns:
        wal_agg = crypto_wd.groupby("user_id")["to_wallet_hash"].nunique().rename("_n_wallets")
        timing_df = base.merge(wal_agg.reset_index(), on="user_id", how="left")
        if "span_days" in cycle_df.columns:
            timing_df = timing_df.merge(cycle_df[["user_id", "span_days"]], on="user_id", how="left")
            timing_df["timing_wallet_turnover"] = (
                timing_df["_n_wallets"].fillna(0) / (timing_df["span_days"].fillna(1) + 1)
            ).clip(0, 1)
        else:
            timing_df["timing_wallet_turnover"] = (
                timing_df["_n_wallets"].fillna(0) / 2.0
            ).clip(0, 1)
    else:
        timing_df = base.assign(timing_wallet_turnover=0.0)

    # timing_cross_channel_count: number of distinct channels used (0-4)
    _channel_uids = {
        "twd": set(twd["user_id"].dropna().astype(int).tolist()),
        "crypto": set(crypto["user_id"].dropna().astype(int).tolist()),
        "swap": set(swap["user_id"].dropna().astype(int).tolist()),
        "trade": set(trade["user_id"].dropna().astype(int).tolist()),
    }
    timing_df["timing_cross_channel_count"] = sum(
        timing_df["user_id"].astype(int).isin(uids).astype(int)
        for uids in _channel_uids.values()
    ).clip(0, 4)

    # timing_night_all_channel_ratio: fraction of events during Taiwan night (UTC 14-22)
    night_parts = []
    for df, tc in [(twd, "created_at"), (crypto, "created_at"), (swap, "created_at")]:
        if not df.empty and tc in df.columns:
            sub = df[["user_id", tc]].dropna().rename(columns={tc: "ts"})
            night_parts.append(sub)
    if night_parts:
        night_all = pd.concat(night_parts, ignore_index=True).dropna(subset=["ts"])
        night_all["is_night"] = night_all["ts"].dt.hour.between(14, 22).astype(float)
        night_ratio = (
            night_all.groupby("user_id")["is_night"].mean()
            .rename("timing_night_all_channel_ratio")
            .clip(0, 1)
        )
        timing_df = timing_df.merge(night_ratio.reset_index(), on="user_id", how="left").fillna(0.0)
    else:
        timing_df["timing_night_all_channel_ratio"] = 0.0

    feat_timing = timing_df[["user_id", "timing_wallet_turnover",
                             "timing_cross_channel_count", "timing_night_all_channel_ratio"]].copy()

    # ── Merge all categories ─────────────────────────────────────────────────
    result = feat_struct
    for df in [feat_cycle, feat_burst, feat_ip, feat_timing]:
        result = result.merge(df, on="user_id", how="left")

    result = result.fillna(0.0)
    result["user_id"] = result["user_id"].astype(int)

    # Ensure all expected columns exist
    for col in TEMPORAL_FEATURE_COLUMNS:
        if col not in result.columns:
            result[col] = 0.0

    # Final ordering
    result = result[["user_id"] + TEMPORAL_FEATURE_COLUMNS].copy()

    elapsed = time.time() - t0
    n_feat = len(TEMPORAL_FEATURE_COLUMNS)
    print(f"[temporal_features] Done: {n_feat} features, {elapsed:.1f}s", flush=True)
    return result


def get_temporal_feature_columns() -> list[str]:
    """Return the list of output feature column names."""
    return list(TEMPORAL_FEATURE_COLUMNS)
