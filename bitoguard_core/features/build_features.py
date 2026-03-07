from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import load_settings
from db.store import DuckDBStore


FEATURE_VERSION = "v1"


@dataclass
class SnapshotContext:
    snapshot_date: pd.Timestamp
    snapshot_end: pd.Timestamp
    lookback_7d: pd.Timestamp
    lookback_30d: pd.Timestamp
    active_users: pd.DataFrame


def _prep_timeframe(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    copied = frame.copy()
    copied[column] = pd.to_datetime(copied[column], utc=True)
    return copied


def _sum_by_user(frame: pd.DataFrame, mask: pd.Series, value_col: str, output_name: str) -> pd.DataFrame:
    subset = frame[mask]
    if subset.empty:
        return pd.DataFrame(columns=["user_id", output_name])
    result = subset.groupby("user_id")[value_col].sum().reset_index()
    return result.rename(columns={value_col: output_name})


def _count_by_user(frame: pd.DataFrame, mask: pd.Series, output_name: str) -> pd.DataFrame:
    subset = frame[mask]
    if subset.empty:
        return pd.DataFrame(columns=["user_id", output_name])
    result = subset.groupby("user_id").size().reset_index(name=output_name)
    return result


def _avg_by_user(frame: pd.DataFrame, mask: pd.Series, value_col: str, output_name: str) -> pd.DataFrame:
    subset = frame[mask]
    if subset.empty:
        return pd.DataFrame(columns=["user_id", output_name])
    result = subset.groupby("user_id")[value_col].mean().reset_index()
    return result.rename(columns={value_col: output_name})


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, pd.NA)
    return (numerator / denominator).fillna(0.0)


def _velocity_features(
    fiat_transactions: pd.DataFrame,
    crypto_transactions: pd.DataFrame,
    snapshot_end: pd.Timestamp,
    lookback_start: pd.Timestamp,
) -> pd.DataFrame:
    deposits = fiat_transactions[
        (fiat_transactions["direction"] == "deposit")
        & (fiat_transactions["occurred_at"] >= lookback_start)
        & (fiat_transactions["occurred_at"] < snapshot_end)
    ].copy()
    withdrawals = crypto_transactions[
        (crypto_transactions["direction"] == "withdrawal")
        & (crypto_transactions["occurred_at"] >= lookback_start)
        & (crypto_transactions["occurred_at"] < snapshot_end)
    ].copy()
    if deposits.empty or withdrawals.empty:
        return pd.DataFrame(columns=[
            "user_id", "fiat_in_to_crypto_out_2h", "fiat_in_to_crypto_out_6h", "fiat_in_to_crypto_out_24h",
            "avg_dwell_time", "large_deposit_withdraw_gap",
        ])

    merged = deposits.merge(withdrawals, on="user_id", suffixes=("_fiat", "_crypto"))
    merged = merged[merged["occurred_at_crypto"] >= merged["occurred_at_fiat"]].copy()
    merged["gap_hours"] = (
        (merged["occurred_at_crypto"] - merged["occurred_at_fiat"]).dt.total_seconds() / 3600.0
    )
    if merged.empty:
        return pd.DataFrame(columns=[
            "user_id", "fiat_in_to_crypto_out_2h", "fiat_in_to_crypto_out_6h", "fiat_in_to_crypto_out_24h",
            "avg_dwell_time", "large_deposit_withdraw_gap",
        ])

    merged["within_2h"] = merged["gap_hours"] <= 2
    merged["within_6h"] = merged["gap_hours"] <= 6
    merged["within_24h"] = merged["gap_hours"] <= 24
    earliest = merged.sort_values(["user_id", "occurred_at_fiat", "gap_hours"]).drop_duplicates(["user_id", "fiat_txn_id"])
    avg_gap = earliest.groupby("user_id")["gap_hours"].mean().reset_index(name="avg_dwell_time")
    large_deposit = deposits.sort_values(["user_id", "amount_twd"], ascending=[True, False]).drop_duplicates("user_id")
    large_gap = (
        large_deposit[["user_id", "fiat_txn_id"]]
        .merge(earliest[["user_id", "fiat_txn_id", "gap_hours"]], on=["user_id", "fiat_txn_id"], how="left")
        .rename(columns={"gap_hours": "large_deposit_withdraw_gap"})
    )
    flags = earliest.groupby("user_id")[["within_2h", "within_6h", "within_24h"]].max().reset_index()
    flags = flags.rename(columns={
        "within_2h": "fiat_in_to_crypto_out_2h",
        "within_6h": "fiat_in_to_crypto_out_6h",
        "within_24h": "fiat_in_to_crypto_out_24h",
    })
    result = flags.merge(avg_gap, on="user_id", how="outer").merge(large_gap, on="user_id", how="outer")
    return result.fillna({
        "fiat_in_to_crypto_out_2h": False,
        "fiat_in_to_crypto_out_6h": False,
        "fiat_in_to_crypto_out_24h": False,
        "avg_dwell_time": 0.0,
        "large_deposit_withdraw_gap": 0.0,
    })


def _night_ratio(frame: pd.DataFrame, time_col: str) -> pd.Series:
    hours = frame[time_col].dt.hour
    return hours.isin([0, 1, 2, 3, 4, 5]).groupby(frame["user_id"]).mean()


def build_feature_snapshots() -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = load_settings()
    store = DuckDBStore(settings.db_path)

    users = _prep_timeframe(store.read_table("canonical.users"), "created_at")
    fiat = _prep_timeframe(store.read_table("canonical.fiat_transactions"), "occurred_at")
    trade = _prep_timeframe(store.read_table("canonical.trade_orders"), "occurred_at")
    crypto = _prep_timeframe(store.read_table("canonical.crypto_transactions"), "occurred_at")
    login = _prep_timeframe(store.read_table("canonical.login_events"), "occurred_at")
    graph = store.read_table("features.graph_features")
    graph["snapshot_date"] = pd.to_datetime(graph["snapshot_date"])

    all_times = pd.concat([
        fiat["occurred_at"], trade["occurred_at"], crypto["occurred_at"], login["occurred_at"]
    ], ignore_index=True)
    snapshot_dates = pd.date_range(all_times.dt.date.min(), all_times.dt.date.max(), freq="D")

    user_day_records: list[pd.DataFrame] = []
    user_30d_records: list[pd.DataFrame] = []

    for snapshot_date in snapshot_dates:
        snapshot_end = snapshot_date.tz_localize("UTC") + pd.Timedelta(days=1)
        ctx = SnapshotContext(
            snapshot_date=snapshot_date,
            snapshot_end=snapshot_end,
            lookback_7d=snapshot_end - pd.Timedelta(days=7),
            lookback_30d=snapshot_end - pd.Timedelta(days=30),
            active_users=users[users["created_at"] < snapshot_end].copy(),
        )
        if ctx.active_users.empty:
            continue
        base = ctx.active_users[[
            "user_id", "kyc_level", "occupation", "monthly_income_twd",
            "expected_monthly_volume_twd", "declared_source_of_funds", "segment",
        ]].copy()
        base["snapshot_date"] = snapshot_date
        base["feature_version"] = FEATURE_VERSION
        base["feature_snapshot_id"] = base["user_id"].map(lambda uid: f"f30_{uid}_{snapshot_date.date().isoformat()}")

        fiat_1d = _sum_by_user(
            fiat,
            (fiat["occurred_at"] >= snapshot_end - pd.Timedelta(days=1)) & (fiat["occurred_at"] < snapshot_end) & (fiat["direction"] == "deposit"),
            "amount_twd",
            "fiat_in_1d",
        ).merge(
            _sum_by_user(
                fiat,
                (fiat["occurred_at"] >= snapshot_end - pd.Timedelta(days=1)) & (fiat["occurred_at"] < snapshot_end) & (fiat["direction"] == "withdrawal"),
                "amount_twd",
                "fiat_out_1d",
            ),
            on="user_id",
            how="outer",
        )
        fiat_7d = _sum_by_user(
            fiat,
            (fiat["occurred_at"] >= ctx.lookback_7d) & (fiat["occurred_at"] < snapshot_end) & (fiat["direction"] == "deposit"),
            "amount_twd",
            "fiat_in_7d",
        ).merge(
            _sum_by_user(
                fiat,
                (fiat["occurred_at"] >= ctx.lookback_7d) & (fiat["occurred_at"] < snapshot_end) & (fiat["direction"] == "withdrawal"),
                "amount_twd",
                "fiat_out_7d",
            ),
            on="user_id",
            how="outer",
        )
        fiat_30d = _sum_by_user(
            fiat,
            (fiat["occurred_at"] >= ctx.lookback_30d) & (fiat["occurred_at"] < snapshot_end) & (fiat["direction"] == "deposit"),
            "amount_twd",
            "fiat_in_30d",
        ).merge(
            _sum_by_user(
                fiat,
                (fiat["occurred_at"] >= ctx.lookback_30d) & (fiat["occurred_at"] < snapshot_end) & (fiat["direction"] == "withdrawal"),
                "amount_twd",
                "fiat_out_30d",
            ),
            on="user_id",
            how="outer",
        )
        trade_stats = (
            _count_by_user(trade, (trade["occurred_at"] >= ctx.lookback_30d) & (trade["occurred_at"] < snapshot_end), "trade_count_30d")
            .merge(_sum_by_user(trade, (trade["occurred_at"] >= ctx.lookback_30d) & (trade["occurred_at"] < snapshot_end), "notional_twd", "trade_notional_30d"), on="user_id", how="outer")
        )
        crypto_stats = _sum_by_user(
            crypto,
            (crypto["occurred_at"] >= ctx.lookback_30d) & (crypto["occurred_at"] < snapshot_end) & (crypto["direction"] == "withdrawal"),
            "amount_twd_equiv",
            "crypto_withdraw_30d",
        )
        velocity = _velocity_features(fiat, crypto, snapshot_end, ctx.lookback_30d)

        login_window = login[(login["occurred_at"] >= ctx.lookback_30d) & (login["occurred_at"] < snapshot_end)].copy()
        login_window["night_flag"] = login_window["occurred_at"].dt.hour.isin([0, 1, 2, 3, 4, 5])
        login_features = (
            _count_by_user(login_window, login_window["is_geo_jump"] == True, "geo_jump_count")
            .merge(_avg_by_user(login_window, login_window["user_id"].notna(), "is_vpn", "vpn_ratio"), on="user_id", how="outer")
            .merge(_avg_by_user(login_window, login_window["user_id"].notna(), "is_new_device", "new_device_ratio"), on="user_id", how="outer")
        )
        if not login_window.empty:
            login_features = login_features.merge(
                login_window.groupby("user_id")["ip_country"].nunique().reset_index(name="ip_country_switch_count"),
                on="user_id",
                how="outer",
            )
            login_features = login_features.merge(
                _night_ratio(login_window, "occurred_at").reset_index(name="night_login_ratio"),
                on="user_id",
                how="outer",
            )
        else:
            login_features["ip_country_switch_count"] = 0
            login_features["night_login_ratio"] = 0.0

        withdrawal_day = crypto[(crypto["occurred_at"] >= snapshot_end - pd.Timedelta(days=1)) & (crypto["occurred_at"] < snapshot_end) & (crypto["direction"] == "withdrawal")].copy()
        if not withdrawal_day.empty:
            withdrawal_day["night_large_flag"] = (
                withdrawal_day["occurred_at"].dt.hour.isin([0, 1, 2, 3, 4, 5]) & (withdrawal_day["amount_twd_equiv"] >= 50000)
            )
            night_large = withdrawal_day.groupby("user_id")["night_large_flag"].mean().reset_index(name="night_large_withdrawal_ratio")
        else:
            night_large = pd.DataFrame(columns=["user_id", "night_large_withdrawal_ratio"])

        new_device_withdraw = pd.DataFrame(columns=["user_id", "new_device_withdrawal_24h"])
        if not login_window.empty and not crypto.empty:
            new_device_events = login_window[login_window["is_new_device"] == True][["user_id", "occurred_at"]].rename(columns={"occurred_at": "login_time"})
            withdrawals = crypto[(crypto["direction"] == "withdrawal") & (crypto["occurred_at"] < snapshot_end)][["user_id", "occurred_at"]].rename(columns={"occurred_at": "withdraw_time"})
            joined = new_device_events.merge(withdrawals, on="user_id", how="inner")
            joined = joined[(joined["withdraw_time"] >= joined["login_time"]) & (joined["withdraw_time"] <= joined["login_time"] + pd.Timedelta(hours=24))]
            if not joined.empty:
                new_device_withdraw = joined.groupby("user_id").size().reset_index(name="new_device_withdrawal_24h")
                new_device_withdraw["new_device_withdrawal_24h"] = True

        result_30 = base.merge(fiat_1d, on="user_id", how="left") \
            .merge(fiat_7d, on="user_id", how="left") \
            .merge(fiat_30d, on="user_id", how="left") \
            .merge(trade_stats, on="user_id", how="left") \
            .merge(crypto_stats, on="user_id", how="left") \
            .merge(velocity, on="user_id", how="left") \
            .merge(login_features, on="user_id", how="left") \
            .merge(night_large, on="user_id", how="left") \
            .merge(new_device_withdraw, on="user_id", how="left") \
            .merge(graph[graph["snapshot_date"] == snapshot_date][[
                "user_id", "shared_device_count", "shared_bank_count", "shared_wallet_count",
                "blacklist_1hop_count", "blacklist_2hop_count", "component_size", "fan_out_ratio"
            ]], on="user_id", how="left")
        result_30 = result_30.fillna({
            "fiat_in_1d": 0.0, "fiat_out_1d": 0.0, "fiat_in_7d": 0.0, "fiat_out_7d": 0.0, "fiat_in_30d": 0.0, "fiat_out_30d": 0.0,
            "trade_count_30d": 0, "trade_notional_30d": 0.0, "crypto_withdraw_30d": 0.0,
            "fiat_in_to_crypto_out_2h": False, "fiat_in_to_crypto_out_6h": False, "fiat_in_to_crypto_out_24h": False,
            "avg_dwell_time": 0.0, "large_deposit_withdraw_gap": 0.0,
            "geo_jump_count": 0, "vpn_ratio": 0.0, "new_device_ratio": 0.0, "ip_country_switch_count": 0, "night_login_ratio": 0.0,
            "night_large_withdrawal_ratio": 0.0, "new_device_withdrawal_24h": False,
            "shared_device_count": 0, "shared_bank_count": 0, "shared_wallet_count": 0, "blacklist_1hop_count": 0,
            "blacklist_2hop_count": 0, "component_size": 1, "fan_out_ratio": 0.0,
        })
        result_30["actual_volume_expected_ratio"] = _safe_ratio(result_30["trade_notional_30d"], result_30["expected_monthly_volume_twd"])
        result_30["actual_fiat_income_ratio"] = _safe_ratio(result_30["fiat_in_30d"] + result_30["fiat_out_30d"], result_30["monthly_income_twd"])

        result_day = result_30.copy()
        result_day["feature_snapshot_id"] = result_day["user_id"].map(lambda uid: f"fd_{uid}_{snapshot_date.date().isoformat()}")

        user_30d_records.append(result_30)
        user_day_records.append(result_day)

    feature_30d = pd.concat(user_30d_records, ignore_index=True) if user_30d_records else pd.DataFrame()
    feature_day = pd.concat(user_day_records, ignore_index=True) if user_day_records else pd.DataFrame()
    store.replace_table("features.feature_snapshots_user_30d", feature_30d)
    store.replace_table("features.feature_snapshots_user_day", feature_day)
    return feature_day, feature_30d


if __name__ == "__main__":
    build_feature_snapshots()
