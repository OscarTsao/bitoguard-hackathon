"""Phase 2A: Raw Event Sequence Features — pandas-only, no DL.

20 label-free features extracted directly from 707K raw event records.
These capture AML patterns invisible to 239-column aggregate statistics:

- Fine-grained timing: inter-deposit intervals, burst windows (30min)
- Structuring: identical-amount runs, near-identical deposit pairs
- Cross-channel chain timing: precise fiat→swap→crypto latency
- IP entropy: Shannon diversity of IP usage
- Wallet behavior: single-use wallets, HHI concentration
- Temporal patterns: night activity, weekend concentration, recency

Key distinction from temporal_features.py (23 features):
- temporal_features.py: window aggregates (7d/30d counts, CV of inter-deposit times)
- sequence_features.py: event-level patterns (min gap, burst 30min, identical-amount runs,
  exact chain latency in hours, IP entropy, wallet HHI)

Entry point: build_sequence_features(dataset) -> pd.DataFrame
Runtime target: < 60 seconds for 63,770 users.
All features are label-free; users with 0 events get 0.0.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from official.common import load_clean_table

SEQUENCE_FEATURE_COLUMNS: list[str] = [
    # Group 1: Fine-grained timing (4)
    "seq_min_inter_deposit_h",       # min hours between consecutive deposits
    "seq_median_inter_deposit_h",    # median inter-deposit hours
    "seq_burst_count_30min",         # max events in any 30-min window (all channels)
    "seq_burst_days_count",          # number of days with ≥3 transactions
    # Group 2: Structuring patterns (3)
    "seq_max_identical_amount_run",  # longest consecutive run of same-amount deposits
    "seq_n_near_identical_pairs",    # deposit pairs within 1% of each other in amount
    "seq_deposit_amount_cv",         # CV of deposit amounts (low = structured)
    # Group 3: Cross-channel chain timing (4)
    "seq_chain_min_h",               # min fiat→swap→crypto chain time (hours)
    "seq_chain_median_h",            # median chain time (hours)
    "seq_n_chains_48h",              # complete chains within 48h window
    "seq_swap_to_crypto_min_h",      # min swap_buy → crypto_withdraw hours
    # Group 4: IP entropy (2)
    "seq_ip_entropy",                # Shannon entropy of IP distribution
    "seq_ip_per_deposit",            # unique IPs per deposit event
    # Group 5: Wallet behavior (3)
    "seq_single_use_wallet_ratio",   # wallets used exactly once / total unique wallets
    "seq_wallet_hhi",                # Herfindahl–Hirschman Index of wallet usage
    "seq_new_wallet_first_week",     # unique new external wallets in first 7 days
    # Group 6: Temporal patterns (4)
    "seq_night_burst_ratio",         # fraction of txns 23:00–05:00 Asia/Taipei
    "seq_weekend_ratio",             # fraction of txns on weekends (Asia/Taipei)
    "seq_activity_span_days",        # days from first to last event
    "seq_recency_days",              # days since last event (from global max date)
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _max_in_window(ts_ns: np.ndarray, window_ns: int) -> int:
    """O(n) sliding-window max event count within `window_ns` nanoseconds."""
    n = len(ts_ns)
    if n == 0:
        return 0
    left = 0
    max_c = 1
    for right in range(1, n):
        while ts_ns[right] - ts_ns[left] > window_ns:
            left += 1
        max_c = max(max_c, right - left + 1)
    return max_c


def _max_identical_run(amounts_sorted: np.ndarray, bucket: float = 100.0) -> int:
    """Longest consecutive run of identical amounts (rounded to nearest `bucket`)."""
    if len(amounts_sorted) == 0:
        return 0
    bucketed = np.round(amounts_sorted / bucket).astype(np.int64)
    max_run = cur = 1
    for i in range(1, len(bucketed)):
        if bucketed[i] == bucketed[i - 1]:
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 1
    return max_run


def _near_identical_pairs(amounts: np.ndarray, tol: float = 0.01) -> int:
    """Count pairs of amounts within `tol` fraction of each other."""
    if len(amounts) < 2:
        return 0
    sorted_a = np.sort(amounts)
    count = 0
    for i in range(len(sorted_a) - 1):
        ref = sorted_a[i]
        if ref <= 0:
            continue
        j = i + 1
        while j < len(sorted_a) and sorted_a[j] <= ref * (1 + tol):
            count += 1
            j += 1
    return min(count, 500)  # cap to prevent extreme values


def _shannon_entropy(values: np.ndarray) -> float:
    """Shannon entropy (bits) of a sequence of labels."""
    if len(values) == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    probs = counts / counts.sum()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(-np.sum(probs * np.log2(probs + 1e-12)))


def _hhi(counts: np.ndarray) -> float:
    """Herfindahl–Hirschman Index: sum of squared market shares."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    fracs = counts / total
    return float(np.sum(fracs ** 2))


def _chain_timing_features(
    deps: pd.DataFrame,
    swap_buy: pd.DataFrame,
    crypto_wd: pd.DataFrame,
) -> pd.DataFrame:
    """Per-user cross-channel chain timing using searchsorted (avoids timezone issues).

    For each fiat deposit, finds the next swap_buy within 48h, then the next
    crypto_withdraw within 48h of that swap. Records chain hours and count.
    """
    _48h_ns = 48 * 3600 * int(1e9)

    # Build per-user sorted timestamp arrays (int64 nanoseconds)
    dep_by_user: dict[int, np.ndarray] = {}
    for uid, grp in deps.groupby("user_id"):
        ts = grp["created_at"].dropna().values.astype("int64")
        if len(ts) > 0:
            dep_by_user[int(uid)] = np.sort(ts)

    swap_by_user: dict[int, np.ndarray] = {}
    for uid, grp in swap_buy.groupby("user_id"):
        ts = grp["created_at"].dropna().values.astype("int64")
        if len(ts) > 0:
            swap_by_user[int(uid)] = np.sort(ts)

    wd_by_user: dict[int, np.ndarray] = {}
    for uid, grp in crypto_wd.groupby("user_id"):
        ts = grp["created_at"].dropna().values.astype("int64")
        if len(ts) > 0:
            wd_by_user[int(uid)] = np.sort(ts)

    sw_by_user: dict[int, np.ndarray] = {}
    for uid, grp in swap_buy.groupby("user_id"):
        ts = grp["created_at"].dropna().values.astype("int64")
        if len(ts) > 0:
            sw_by_user[int(uid)] = np.sort(ts)

    rows: list[dict] = []
    all_uids = set(dep_by_user) | set(swap_by_user) | set(wd_by_user)

    for uid in all_uids:
        d_arr = dep_by_user.get(uid)
        s_arr = swap_by_user.get(uid)
        w_arr = wd_by_user.get(uid)

        chain_hours: list[float] = []
        sw_to_wd_hours: list[float] = []

        if d_arr is not None and s_arr is not None and w_arr is not None:
            for d_ts in d_arr:
                # Find earliest swap buy after this deposit (within 48h)
                s_idx = np.searchsorted(s_arr, d_ts, side="right")
                if s_idx >= len(s_arr):
                    continue
                s_ts = s_arr[s_idx]
                if s_ts - d_ts > _48h_ns:
                    continue
                # Find earliest crypto withdrawal after this swap (within 48h)
                w_idx = np.searchsorted(w_arr, s_ts, side="right")
                if w_idx >= len(w_arr):
                    continue
                w_ts = w_arr[w_idx]
                if w_ts - s_ts > _48h_ns:
                    continue
                chain_h = float(w_ts - d_ts) / 3.6e12  # ns → hours
                sw_h = float(w_ts - s_ts) / 3.6e12
                chain_hours.append(chain_h)
                sw_to_wd_hours.append(sw_h)

        # swap→crypto min (even without full chain)
        min_sw_h = 0.0
        if s_arr is not None and w_arr is not None and len(s_arr) > 0 and len(w_arr) > 0:
            gaps = []
            for s_ts in s_arr:
                w_idx = np.searchsorted(w_arr, s_ts, side="right")
                if w_idx < len(w_arr):
                    gap_h = float(w_arr[w_idx] - s_ts) / 3.6e12
                    if gap_h <= 48.0:
                        gaps.append(gap_h)
            if gaps:
                min_sw_h = float(min(gaps))

        rows.append({
            "user_id": uid,
            "seq_chain_min_h": float(min(chain_hours)) if chain_hours else 0.0,
            "seq_chain_median_h": float(np.median(chain_hours)) if chain_hours else 0.0,
            "seq_n_chains_48h": len(chain_hours),
            "seq_swap_to_crypto_min_h": min_sw_h,
        })

    if not rows:
        return pd.DataFrame(columns=["user_id", "seq_chain_min_h", "seq_chain_median_h",
                                     "seq_n_chains_48h", "seq_swap_to_crypto_min_h"])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_sequence_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Build 20 raw-event sequence features for all users.

    Parameters
    ----------
    dataset : pd.DataFrame
        Full user dataset (provides user_ids including predict_only).

    Returns
    -------
    pd.DataFrame with columns [user_id] + SEQUENCE_FEATURE_COLUMNS.
    All values are float32; users with 0 events get 0.0.
    """
    t0 = time.time()
    all_user_ids = dataset["user_id"].astype(int).drop_duplicates()
    uid_set = set(all_user_ids.tolist())
    base = all_user_ids.to_frame("user_id")
    print(f"[sequence_features] Building {len(SEQUENCE_FEATURE_COLUMNS)} features "
          f"for {len(uid_set)} users...", flush=True)

    # ── Load tables ──────────────────────────────────────────────────────────
    twd = load_clean_table("twd_transfer").copy()
    crypto = load_clean_table("crypto_transfer").copy()
    swap = load_clean_table("usdt_swap").copy()
    trade = load_clean_table("usdt_twd_trading").copy()
    print(f"[sequence_features] Loaded: twd={len(twd)}, crypto={len(crypto)}, "
          f"swap={len(swap)}, trade={len(trade)}", flush=True)

    # Filter to known users & parse timestamps
    for df in (twd, crypto, swap, trade):
        df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
    twd = twd[twd["user_id"].isin(uid_set)].copy()
    crypto = crypto[crypto["user_id"].isin(uid_set)].copy()
    swap = swap[swap["user_id"].isin(uid_set)].copy()
    trade = trade[trade["user_id"].isin(uid_set)].copy()
    twd["user_id"] = twd["user_id"].astype(int)
    crypto["user_id"] = crypto["user_id"].astype(int)
    swap["user_id"] = swap["user_id"].astype(int)
    trade["user_id"] = trade["user_id"].astype(int)

    twd["created_at"] = pd.to_datetime(twd["created_at"], utc=True, errors="coerce")
    crypto["created_at"] = pd.to_datetime(crypto["created_at"], utc=True, errors="coerce")
    swap["created_at"] = pd.to_datetime(swap["created_at"], utc=True, errors="coerce")
    _trade_ts = "updated_at" if "updated_at" in trade.columns else "created_at"
    trade[_trade_ts] = pd.to_datetime(trade[_trade_ts], utc=True, errors="coerce")

    # ── TWD deposits ─────────────────────────────────────────────────────────
    deps = twd[twd["is_deposit"] == True].dropna(subset=["created_at"]).copy()
    if "amount_twd" in deps.columns:
        deps = deps.dropna(subset=["amount_twd"])
        deps = deps[deps["amount_twd"] > 0]
    else:
        deps["amount_twd"] = 0.0

    # ── Swap buys ────────────────────────────────────────────────────────────
    swap_buy_mask = swap["kind_label"].astype(str).str.lower().str.contains("buy", na=False)
    swap_buy = swap[swap_buy_mask].dropna(subset=["created_at"]).copy()

    # ── Crypto withdrawals ───────────────────────────────────────────────────
    crypto_wd = crypto[crypto.get("is_external_transfer", pd.Series(False, index=crypto.index)) == True].copy()
    if crypto_wd.empty:
        # Fallback: use kind column
        crypto_wd = crypto[crypto["kind"].astype(str).str.upper().str.contains("WITHDRAW", na=False)].copy()
    crypto_wd = crypto_wd.dropna(subset=["created_at"])

    # ── All events combined (for burst analysis) ──────────────────────────────
    evts_parts: list[pd.DataFrame] = []
    for df, tc in [(twd, "created_at"), (crypto, "created_at"),
                   (swap, "created_at"), (trade, _trade_ts)]:
        if not df.empty and tc in df.columns:
            sub = df[["user_id", tc]].dropna().copy()
            sub = sub.rename(columns={tc: "ts"})
            evts_parts.append(sub)
    all_evts = pd.concat(evts_parts, ignore_index=True) if evts_parts else pd.DataFrame(columns=["user_id", "ts"])
    all_evts = all_evts.dropna(subset=["ts"])
    all_evts["ts"] = pd.to_datetime(all_evts["ts"], utc=True, errors="coerce")
    all_evts = all_evts.dropna(subset=["ts"])

    # Global max timestamp for recency
    global_max_ts = all_evts["ts"].max() if not all_evts.empty else pd.Timestamp("2025-01-01", tz="UTC")

    results: dict[str, pd.Series] = {}

    # ── Group 1: Fine-grained timing ─────────────────────────────────────────
    try:
        if not deps.empty and len(deps) > 1:
            dep_sorted = deps[["user_id", "created_at"]].sort_values(["user_id", "created_at"])
            dep_sorted["prev_ts"] = dep_sorted.groupby("user_id")["created_at"].shift(1)
            dep_sorted = dep_sorted.dropna(subset=["prev_ts"])
            dep_sorted["gap_h"] = (dep_sorted["created_at"] - dep_sorted["prev_ts"]).dt.total_seconds() / 3600.0
            dep_sorted = dep_sorted[dep_sorted["gap_h"] >= 0]
            inter_agg = dep_sorted.groupby("user_id")["gap_h"].agg(
                seq_min_inter_deposit_h="min",
                seq_median_inter_deposit_h="median",
            )
            results["seq_min_inter_deposit_h"] = inter_agg["seq_min_inter_deposit_h"]
            results["seq_median_inter_deposit_h"] = inter_agg["seq_median_inter_deposit_h"]
        else:
            results["seq_min_inter_deposit_h"] = pd.Series(dtype=float)
            results["seq_median_inter_deposit_h"] = pd.Series(dtype=float)
    except Exception as e:
        print(f"[sequence_features] inter-deposit timing failed: {e}", flush=True)
        results["seq_min_inter_deposit_h"] = pd.Series(dtype=float)
        results["seq_median_inter_deposit_h"] = pd.Series(dtype=float)

    # burst_count_30min: max events in any 30-min rolling window
    try:
        _30min_ns = 30 * 60 * int(1e9)
        evts_sorted = all_evts.sort_values(["user_id", "ts"])
        burst_30 = evts_sorted.groupby("user_id")["ts"].apply(
            lambda x: _max_in_window(x.values.astype("int64"), _30min_ns),
            include_groups=False,
        ).rename("seq_burst_count_30min")
        results["seq_burst_count_30min"] = burst_30
    except Exception as e:
        print(f"[sequence_features] burst_count_30min failed: {e}", flush=True)
        results["seq_burst_count_30min"] = pd.Series(dtype=float)

    # burst_days_count: days with ≥3 transactions
    try:
        all_evts["date"] = all_evts["ts"].dt.date
        daily_counts = all_evts.groupby(["user_id", "date"]).size().reset_index(name="cnt")
        burst_days = (
            daily_counts[daily_counts["cnt"] >= 3]
            .groupby("user_id")
            .size()
            .rename("seq_burst_days_count")
        )
        results["seq_burst_days_count"] = burst_days
    except Exception as e:
        print(f"[sequence_features] burst_days_count failed: {e}", flush=True)
        results["seq_burst_days_count"] = pd.Series(dtype=float)

    # ── Group 2: Structuring patterns ────────────────────────────────────────
    try:
        if not deps.empty and "amount_twd" in deps.columns:
            dep_amt_sorted = deps[["user_id", "created_at", "amount_twd"]].sort_values(
                ["user_id", "created_at"]
            )
            # max identical amount run
            max_run = dep_amt_sorted.groupby("user_id")["amount_twd"].apply(
                lambda x: _max_identical_run(x.values),
                include_groups=False,
            ).rename("seq_max_identical_amount_run")
            results["seq_max_identical_amount_run"] = max_run

            # near-identical pairs (within 1%)
            near_pairs = dep_amt_sorted.groupby("user_id")["amount_twd"].apply(
                lambda x: _near_identical_pairs(x.values),
                include_groups=False,
            ).rename("seq_n_near_identical_pairs")
            results["seq_n_near_identical_pairs"] = near_pairs

            # coefficient of variation
            amt_stats = dep_amt_sorted.groupby("user_id")["amount_twd"].agg(["mean", "std", "count"])
            cv_mask = (amt_stats["mean"] > 0) & (amt_stats["count"] >= 2)
            amt_stats["seq_deposit_amount_cv"] = 0.0
            amt_stats.loc[cv_mask, "seq_deposit_amount_cv"] = (
                amt_stats.loc[cv_mask, "std"] / amt_stats.loc[cv_mask, "mean"]
            ).clip(0, 10)
            results["seq_deposit_amount_cv"] = amt_stats["seq_deposit_amount_cv"]
        else:
            for col in ("seq_max_identical_amount_run", "seq_n_near_identical_pairs", "seq_deposit_amount_cv"):
                results[col] = pd.Series(dtype=float)
    except Exception as e:
        print(f"[sequence_features] structuring patterns failed: {e}", flush=True)
        for col in ("seq_max_identical_amount_run", "seq_n_near_identical_pairs", "seq_deposit_amount_cv"):
            results[col] = pd.Series(dtype=float)

    # ── Group 3: Cross-channel chain timing ─────────────────────────────────
    try:
        chain_df = _chain_timing_features(deps, swap_buy, crypto_wd)
        for col in ("seq_chain_min_h", "seq_chain_median_h", "seq_n_chains_48h", "seq_swap_to_crypto_min_h"):
            if col in chain_df.columns:
                results[col] = chain_df.set_index("user_id")[col]
            else:
                results[col] = pd.Series(dtype=float)
    except Exception as e:
        print(f"[sequence_features] chain timing failed: {e}", flush=True)
        for col in ("seq_chain_min_h", "seq_chain_median_h", "seq_n_chains_48h", "seq_swap_to_crypto_min_h"):
            results[col] = pd.Series(dtype=float)

    # ── Group 4: IP entropy ──────────────────────────────────────────────────
    try:
        ip_parts: list[pd.DataFrame] = []
        for df, tc in [(twd, "source_ip_hash"), (crypto, "source_ip_hash"), (trade, "source_ip_hash")]:
            if tc in df.columns:
                sub = df[["user_id", tc]].dropna().copy()
                sub = sub.rename(columns={tc: "ip"})
                ip_parts.append(sub)
        if ip_parts:
            all_ips = pd.concat(ip_parts, ignore_index=True)
            all_ips = all_ips[all_ips["ip"].notna() & (all_ips["ip"] != "")]

            # Shannon entropy
            ip_entropy = (
                all_ips.groupby("user_id")["ip"]
                .apply(lambda x: _shannon_entropy(x.values), include_groups=False)
                .rename("seq_ip_entropy")
            )
            results["seq_ip_entropy"] = ip_entropy

            # IPs per deposit
            dep_ip = twd[twd["is_deposit"] == True][["user_id", "source_ip_hash"]].dropna()
            dep_ip_stats = dep_ip.groupby("user_id").agg(
                n_dep=("source_ip_hash", "count"),
                n_ip=("source_ip_hash", "nunique"),
            )
            dep_ip_stats["seq_ip_per_deposit"] = (
                dep_ip_stats["n_ip"] / dep_ip_stats["n_dep"].clip(1)
            ).clip(0, 5)
            results["seq_ip_per_deposit"] = dep_ip_stats["seq_ip_per_deposit"]
        else:
            results["seq_ip_entropy"] = pd.Series(dtype=float)
            results["seq_ip_per_deposit"] = pd.Series(dtype=float)
    except Exception as e:
        print(f"[sequence_features] IP entropy failed: {e}", flush=True)
        results["seq_ip_entropy"] = pd.Series(dtype=float)
        results["seq_ip_per_deposit"] = pd.Series(dtype=float)

    # ── Group 5: Wallet behavior ─────────────────────────────────────────────
    try:
        wallet_parts: list[pd.DataFrame] = []
        if "from_wallet_hash" in crypto.columns:
            wallet_parts.append(
                crypto[["user_id", "from_wallet_hash"]].dropna()
                .rename(columns={"from_wallet_hash": "wallet"})
            )
        if "to_wallet_hash" in crypto.columns:
            wallet_parts.append(
                crypto[["user_id", "to_wallet_hash"]].dropna()
                .rename(columns={"to_wallet_hash": "wallet"})
            )
        if wallet_parts:
            all_wallets = pd.concat(wallet_parts, ignore_index=True)
            all_wallets = all_wallets[all_wallets["wallet"].notna() & (all_wallets["wallet"] != "")]

            wallet_counts = (
                all_wallets.groupby(["user_id", "wallet"])
                .size()
                .reset_index(name="cnt")
            )
            wallet_user_agg = wallet_counts.groupby("user_id").agg(
                total_wallets=("wallet", "count"),
                single_use=("cnt", lambda x: int((x == 1).sum())),
            )
            wallet_user_agg["seq_single_use_wallet_ratio"] = (
                wallet_user_agg["single_use"] / wallet_user_agg["total_wallets"].clip(1)
            ).clip(0, 1)
            results["seq_single_use_wallet_ratio"] = wallet_user_agg["seq_single_use_wallet_ratio"]

            # HHI
            hhi_vals = (
                wallet_counts.groupby("user_id")["cnt"]
                .apply(lambda x: _hhi(x.values), include_groups=False)
                .rename("seq_wallet_hhi")
            )
            results["seq_wallet_hhi"] = hhi_vals

            # New wallets in first 7 days
            _7d_ns = 7 * 86400 * int(1e9)
            crypto_w = crypto.copy()
            crypto_w["created_at"] = pd.to_datetime(crypto_w["created_at"], utc=True, errors="coerce")
            crypto_w = crypto_w.dropna(subset=["created_at"])
            first_ts = crypto_w.groupby("user_id")["created_at"].transform("min")
            first_week_mask = (
                (crypto_w["created_at"].values.astype("int64") -
                 first_ts.values.astype("int64")) <= _7d_ns
            )
            fw = crypto_w[first_week_mask]
            fw_wallets = pd.concat([
                fw[["user_id", "from_wallet_hash"]].rename(columns={"from_wallet_hash": "wallet"}),
                fw[["user_id", "to_wallet_hash"]].rename(columns={"to_wallet_hash": "wallet"}),
            ], ignore_index=True).dropna()
            new_wallets_fw = (
                fw_wallets.groupby("user_id")["wallet"]
                .nunique()
                .rename("seq_new_wallet_first_week")
            )
            results["seq_new_wallet_first_week"] = new_wallets_fw
        else:
            for col in ("seq_single_use_wallet_ratio", "seq_wallet_hhi", "seq_new_wallet_first_week"):
                results[col] = pd.Series(dtype=float)
    except Exception as e:
        print(f"[sequence_features] wallet behavior failed: {e}", flush=True)
        for col in ("seq_single_use_wallet_ratio", "seq_wallet_hhi", "seq_new_wallet_first_week"):
            results[col] = pd.Series(dtype=float)

    # ── Group 6: Temporal patterns ───────────────────────────────────────────
    try:
        # Convert to Asia/Taipei for night/weekend detection
        evts_tz = all_evts.copy()
        evts_tz["ts_local"] = evts_tz["ts"].dt.tz_convert("Asia/Taipei")
        evts_tz["hour"] = evts_tz["ts_local"].dt.hour
        evts_tz["dow"] = evts_tz["ts_local"].dt.dayofweek  # 0=Mon, 5=Sat, 6=Sun

        # Night: 23:00–05:00
        evts_tz["is_night"] = ((evts_tz["hour"] >= 23) | (evts_tz["hour"] < 5)).astype(int)
        evts_tz["is_weekend"] = (evts_tz["dow"] >= 5).astype(int)

        night_agg = evts_tz.groupby("user_id").agg(
            _total=("is_night", "count"),
            _night=("is_night", "sum"),
            _weekend=("is_weekend", "sum"),
        )
        night_agg["seq_night_burst_ratio"] = (
            night_agg["_night"] / night_agg["_total"].clip(1)
        ).clip(0, 1)
        night_agg["seq_weekend_ratio"] = (
            night_agg["_weekend"] / night_agg["_total"].clip(1)
        ).clip(0, 1)
        results["seq_night_burst_ratio"] = night_agg["seq_night_burst_ratio"]
        results["seq_weekend_ratio"] = night_agg["seq_weekend_ratio"]

        # Activity span and recency
        span_agg = all_evts.groupby("user_id")["ts"].agg(["min", "max"])
        span_agg["seq_activity_span_days"] = (
            (span_agg["max"] - span_agg["min"]).dt.total_seconds() / 86400.0
        ).clip(0, 3650)
        span_agg["seq_recency_days"] = (
            (global_max_ts - span_agg["max"]).dt.total_seconds() / 86400.0
        ).clip(0, 3650)
        results["seq_activity_span_days"] = span_agg["seq_activity_span_days"]
        results["seq_recency_days"] = span_agg["seq_recency_days"]
    except Exception as e:
        print(f"[sequence_features] temporal patterns failed: {e}", flush=True)
        for col in ("seq_night_burst_ratio", "seq_weekend_ratio",
                    "seq_activity_span_days", "seq_recency_days"):
            results[col] = pd.Series(dtype=float)

    # ── Assemble output ───────────────────────────────────────────────────────
    out = base.copy()
    for col in SEQUENCE_FEATURE_COLUMNS:
        series = results.get(col, pd.Series(dtype=float))
        if isinstance(series, pd.Series) and not series.empty:
            series = series.rename(col)
            out = out.merge(series.reset_index().rename(columns={"index": "user_id"}),
                            on="user_id", how="left")
        if col not in out.columns:
            out[col] = 0.0
        out[col] = out[col].fillna(0.0).astype("float32")

    elapsed = time.time() - t0
    n_nonzero = sum((out[c] != 0).sum() for c in SEQUENCE_FEATURE_COLUMNS)
    print(f"[sequence_features] Done: {len(SEQUENCE_FEATURE_COLUMNS)} features, "
          f"{n_nonzero} non-zero values, {elapsed:.1f}s", flush=True)
    return out
