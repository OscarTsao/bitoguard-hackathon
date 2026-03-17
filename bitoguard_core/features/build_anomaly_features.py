from __future__ import annotations

import pandas as pd

from config import load_settings
from db.store import DuckDBStore
from features.build_features import FEATURE_VERSION, SnapshotContext, _avg_by_user, _count_by_user, _night_ratio, _prep_timeframe, _safe_ratio, _sum_by_user, _velocity_features
from features.graph_features import build_graph_features


ANOMALY_FEATURE_VERSION = f"{FEATURE_VERSION}_anomaly_v1"
ANOMALY_BASE_COLUMNS = [
    "feature_snapshot_id",
    "user_id",
    "snapshot_date",
    "feature_version",
    "fiat_in_1d",
    "fiat_in_7d",
    "fiat_in_30d",
    "fiat_out_1d",
    "fiat_out_7d",
    "fiat_out_30d",
    "trade_count_7d",
    "trade_count_30d",
    "trade_notional_7d",
    "trade_notional_30d",
    "crypto_withdraw_1d",
    "crypto_withdraw_7d",
    "crypto_withdraw_30d",
    "login_count_1d",
    "login_count_7d",
    "login_count_30d",
    "unique_ip_count_7d",
    "unique_ip_count_30d",
    "unique_counterparty_wallet_count_30d",
    "new_counterparty_wallet_ratio_30d",
    "fiat_in_spike_ratio",
    "fiat_out_spike_ratio",
    "trade_notional_spike_ratio",
    "crypto_withdraw_spike_ratio",
    "login_spike_ratio",
    "fiat_in_to_crypto_out_2h",
    "fiat_in_to_crypto_out_6h",
    "fiat_in_to_crypto_out_24h",
    "avg_dwell_time",
    "large_deposit_withdraw_gap",
    "geo_jump_count",
    "vpn_ratio",
    "new_device_ratio",
    "ip_country_switch_count",
    "night_login_ratio",
    "night_large_withdrawal_ratio",
    "new_device_withdrawal_24h",
    "shared_device_count",
    "shared_bank_count",
    "shared_wallet_count",
    "blacklist_1hop_count",
    "blacklist_2hop_count",
    "component_size",
    "fan_out_ratio",
]


def _empty_anomaly_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=ANOMALY_BASE_COLUMNS)


def _unique_count_by_user(frame: pd.DataFrame, mask: pd.Series, value_col: str, output_name: str) -> pd.DataFrame:
    subset = frame[mask & frame[value_col].notna()].copy()
    if subset.empty:
        return pd.DataFrame(columns=["user_id", output_name])
    result = subset.groupby("user_id")[value_col].nunique(dropna=True).reset_index(name=output_name)
    return result


def _spike_ratio(short_window: pd.Series, long_window: pd.Series, window_factor: float) -> pd.Series:
    return ((short_window.astype(float) + 1.0) / ((long_window.astype(float) / window_factor) + 1.0)).fillna(1.0)


def _counterparty_wallet_features(
    crypto_transactions: pd.DataFrame,
    snapshot_end: pd.Timestamp,
    lookback_30d: pd.Timestamp,
) -> pd.DataFrame:
    withdrawals = crypto_transactions[
        (crypto_transactions["direction"] == "withdrawal")
        & (crypto_transactions["occurred_at"] < snapshot_end)
        & crypto_transactions["counterparty_wallet_id"].notna()
    ][["user_id", "counterparty_wallet_id", "occurred_at"]].copy()
    if withdrawals.empty:
        return pd.DataFrame(columns=[
            "user_id",
            "unique_counterparty_wallet_count_30d",
            "new_counterparty_wallet_ratio_30d",
        ])

    window = withdrawals[withdrawals["occurred_at"] >= lookback_30d].copy()
    if window.empty:
        return pd.DataFrame(columns=[
            "user_id",
            "unique_counterparty_wallet_count_30d",
            "new_counterparty_wallet_ratio_30d",
        ])

    unique_count = (
        window.groupby("user_id")["counterparty_wallet_id"]
        .nunique(dropna=True)
        .reset_index(name="unique_counterparty_wallet_count_30d")
    )
    first_seen = (
        withdrawals.groupby(["user_id", "counterparty_wallet_id"])["occurred_at"]
        .min()
        .reset_index(name="first_seen_at")
    )
    window_unique = window[["user_id", "counterparty_wallet_id"]].drop_duplicates()
    first_seen_window = window_unique.merge(first_seen, on=["user_id", "counterparty_wallet_id"], how="left")
    new_wallet_count = (
        first_seen_window[first_seen_window["first_seen_at"] >= lookback_30d]
        .groupby("user_id")
        .size()
        .reset_index(name="new_counterparty_wallet_count_30d")
    )
    result = unique_count.merge(new_wallet_count, on="user_id", how="left").fillna({"new_counterparty_wallet_count_30d": 0})
    result["new_counterparty_wallet_ratio_30d"] = _safe_ratio(
        result["new_counterparty_wallet_count_30d"],
        result["unique_counterparty_wallet_count_30d"],
    )
    return result[["user_id", "unique_counterparty_wallet_count_30d", "new_counterparty_wallet_ratio_30d"]]


def build_anomaly_feature_snapshots() -> pd.DataFrame:
    settings = load_settings()
    store = DuckDBStore(settings.db_path)

    users = _prep_timeframe(store.read_table("canonical.users"), "created_at")
    fiat = _prep_timeframe(store.read_table("canonical.fiat_transactions"), "occurred_at")
    trade = _prep_timeframe(store.read_table("canonical.trade_orders"), "occurred_at")
    crypto = _prep_timeframe(store.read_table("canonical.crypto_transactions"), "occurred_at")
    login = _prep_timeframe(store.read_table("canonical.login_events"), "occurred_at")
    graph = store.read_table("features.graph_features")
    if graph.empty:
        graph = build_graph_features()
    if graph.empty:
        graph = pd.DataFrame(columns=[
            "user_id",
            "snapshot_date",
            "shared_device_count",
            "shared_bank_count",
            "shared_wallet_count",
            "blacklist_1hop_count",
            "blacklist_2hop_count",
            "component_size",
            "fan_out_ratio",
        ])
    graph["snapshot_date"] = pd.to_datetime(graph["snapshot_date"])

    time_series = [
        frame["occurred_at"]
        for frame in [fiat, trade, crypto, login]
        if not frame.empty
    ]
    if not time_series:
        empty = _empty_anomaly_frame()
        store.replace_table("features.feature_snapshots_user_anomaly_30d", empty)
        return empty

    all_times = pd.concat(time_series, ignore_index=True)
    snapshot_dates = pd.date_range(all_times.dt.date.min(), all_times.dt.date.max(), freq="D")
    records: list[pd.DataFrame] = []

    for snapshot_date in snapshot_dates:
        snapshot_end = snapshot_date.tz_localize("UTC") + pd.Timedelta(days=1)
        ctx = SnapshotContext(
            snapshot_date=snapshot_date,
            snapshot_end=snapshot_end,
            lookback_7d=snapshot_end - pd.Timedelta(days=7),
            lookback_30d=snapshot_end - pd.Timedelta(days=30),
            active_users=users[users["created_at"] < snapshot_end][["user_id"]].copy(),
        )
        if ctx.active_users.empty:
            continue

        base = ctx.active_users.copy()
        base["snapshot_date"] = snapshot_date
        base["feature_version"] = ANOMALY_FEATURE_VERSION
        base["feature_snapshot_id"] = base["user_id"].map(lambda user_id: f"fa_{user_id}_{snapshot_date.date().isoformat()}")

        fiat_1d = _sum_by_user(
            fiat,
            (fiat["occurred_at"] >= snapshot_end - pd.Timedelta(days=1))
            & (fiat["occurred_at"] < snapshot_end)
            & (fiat["direction"] == "deposit"),
            "amount_twd",
            "fiat_in_1d",
        ).merge(
            _sum_by_user(
                fiat,
                (fiat["occurred_at"] >= snapshot_end - pd.Timedelta(days=1))
                & (fiat["occurred_at"] < snapshot_end)
                & (fiat["direction"] == "withdrawal"),
                "amount_twd",
                "fiat_out_1d",
            ),
            on="user_id",
            how="outer",
        )
        fiat_7d = _sum_by_user(
            fiat,
            (fiat["occurred_at"] >= ctx.lookback_7d)
            & (fiat["occurred_at"] < snapshot_end)
            & (fiat["direction"] == "deposit"),
            "amount_twd",
            "fiat_in_7d",
        ).merge(
            _sum_by_user(
                fiat,
                (fiat["occurred_at"] >= ctx.lookback_7d)
                & (fiat["occurred_at"] < snapshot_end)
                & (fiat["direction"] == "withdrawal"),
                "amount_twd",
                "fiat_out_7d",
            ),
            on="user_id",
            how="outer",
        )
        fiat_30d = _sum_by_user(
            fiat,
            (fiat["occurred_at"] >= ctx.lookback_30d)
            & (fiat["occurred_at"] < snapshot_end)
            & (fiat["direction"] == "deposit"),
            "amount_twd",
            "fiat_in_30d",
        ).merge(
            _sum_by_user(
                fiat,
                (fiat["occurred_at"] >= ctx.lookback_30d)
                & (fiat["occurred_at"] < snapshot_end)
                & (fiat["direction"] == "withdrawal"),
                "amount_twd",
                "fiat_out_30d",
            ),
            on="user_id",
            how="outer",
        )

        trade_7d = _count_by_user(
            trade,
            (trade["occurred_at"] >= ctx.lookback_7d) & (trade["occurred_at"] < snapshot_end),
            "trade_count_7d",
        ).merge(
            _sum_by_user(
                trade,
                (trade["occurred_at"] >= ctx.lookback_7d) & (trade["occurred_at"] < snapshot_end),
                "notional_twd",
                "trade_notional_7d",
            ),
            on="user_id",
            how="outer",
        )
        trade_30d = _count_by_user(
            trade,
            (trade["occurred_at"] >= ctx.lookback_30d) & (trade["occurred_at"] < snapshot_end),
            "trade_count_30d",
        ).merge(
            _sum_by_user(
                trade,
                (trade["occurred_at"] >= ctx.lookback_30d) & (trade["occurred_at"] < snapshot_end),
                "notional_twd",
                "trade_notional_30d",
            ),
            on="user_id",
            how="outer",
        )

        crypto_1d = _sum_by_user(
            crypto,
            (crypto["occurred_at"] >= snapshot_end - pd.Timedelta(days=1))
            & (crypto["occurred_at"] < snapshot_end)
            & (crypto["direction"] == "withdrawal"),
            "amount_twd_equiv",
            "crypto_withdraw_1d",
        )
        crypto_7d = _sum_by_user(
            crypto,
            (crypto["occurred_at"] >= ctx.lookback_7d)
            & (crypto["occurred_at"] < snapshot_end)
            & (crypto["direction"] == "withdrawal"),
            "amount_twd_equiv",
            "crypto_withdraw_7d",
        )
        crypto_30d = _sum_by_user(
            crypto,
            (crypto["occurred_at"] >= ctx.lookback_30d)
            & (crypto["occurred_at"] < snapshot_end)
            & (crypto["direction"] == "withdrawal"),
            "amount_twd_equiv",
            "crypto_withdraw_30d",
        )
        counterparty_wallets = _counterparty_wallet_features(crypto, snapshot_end, ctx.lookback_30d)
        velocity = _velocity_features(fiat, crypto, snapshot_end, ctx.lookback_30d)

        login_1d = _count_by_user(
            login,
            (login["occurred_at"] >= snapshot_end - pd.Timedelta(days=1)) & (login["occurred_at"] < snapshot_end),
            "login_count_1d",
        )
        login_7d = _count_by_user(
            login,
            (login["occurred_at"] >= ctx.lookback_7d) & (login["occurred_at"] < snapshot_end),
            "login_count_7d",
        )
        login_30d = _count_by_user(
            login,
            (login["occurred_at"] >= ctx.lookback_30d) & (login["occurred_at"] < snapshot_end),
            "login_count_30d",
        )
        login_window = login[(login["occurred_at"] >= ctx.lookback_30d) & (login["occurred_at"] < snapshot_end)].copy()
        login_features = (
            _count_by_user(login_window, login_window["is_geo_jump"] == True, "geo_jump_count")
            .merge(_avg_by_user(login_window, login_window["user_id"].notna(), "is_vpn", "vpn_ratio"), on="user_id", how="outer")
            .merge(_avg_by_user(login_window, login_window["user_id"].notna(), "is_new_device", "new_device_ratio"), on="user_id", how="outer")
            .merge(_unique_count_by_user(login, (login["occurred_at"] >= ctx.lookback_7d) & (login["occurred_at"] < snapshot_end), "ip_address", "unique_ip_count_7d"), on="user_id", how="outer")
            .merge(_unique_count_by_user(login_window, login_window["user_id"].notna(), "ip_address", "unique_ip_count_30d"), on="user_id", how="outer")
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

        withdrawal_day = crypto[
            (crypto["occurred_at"] >= snapshot_end - pd.Timedelta(days=1))
            & (crypto["occurred_at"] < snapshot_end)
            & (crypto["direction"] == "withdrawal")
        ].copy()
        if not withdrawal_day.empty:
            withdrawal_day["night_large_flag"] = (
                withdrawal_day["occurred_at"].dt.hour.isin([0, 1, 2, 3, 4, 5])
                & (withdrawal_day["amount_twd_equiv"] >= 50000)
            )
            night_large = withdrawal_day.groupby("user_id")["night_large_flag"].mean().reset_index(name="night_large_withdrawal_ratio")
        else:
            night_large = pd.DataFrame(columns=["user_id", "night_large_withdrawal_ratio"])

        new_device_withdraw = pd.DataFrame(columns=["user_id", "new_device_withdrawal_24h"])
        if not login_window.empty and not crypto.empty:
            new_device_events = login_window[login_window["is_new_device"] == True][["user_id", "occurred_at"]].rename(columns={"occurred_at": "login_time"})
            withdrawals = crypto[
                (crypto["direction"] == "withdrawal")
                & (crypto["occurred_at"] < snapshot_end)
            ][["user_id", "occurred_at"]].rename(columns={"occurred_at": "withdraw_time"})
            joined = new_device_events.merge(withdrawals, on="user_id", how="inner")
            joined = joined[
                (joined["withdraw_time"] >= joined["login_time"])
                & (joined["withdraw_time"] <= joined["login_time"] + pd.Timedelta(hours=24))
            ]
            if not joined.empty:
                new_device_withdraw = joined.groupby("user_id").size().reset_index(name="new_device_withdrawal_24h")
                new_device_withdraw["new_device_withdrawal_24h"] = True

        graph_slice = graph[graph["snapshot_date"] == snapshot_date][[
            "user_id",
            "shared_device_count",
            "shared_bank_count",
            "shared_wallet_count",
            "blacklist_1hop_count",
            "blacklist_2hop_count",
            "component_size",
            "fan_out_ratio",
        ]].copy()

        result = (
            base.merge(fiat_1d, on="user_id", how="left")
            .merge(fiat_7d, on="user_id", how="left")
            .merge(fiat_30d, on="user_id", how="left")
            .merge(trade_7d, on="user_id", how="left")
            .merge(trade_30d, on="user_id", how="left")
            .merge(crypto_1d, on="user_id", how="left")
            .merge(crypto_7d, on="user_id", how="left")
            .merge(crypto_30d, on="user_id", how="left")
            .merge(counterparty_wallets, on="user_id", how="left")
            .merge(velocity, on="user_id", how="left")
            .merge(login_1d, on="user_id", how="left")
            .merge(login_7d, on="user_id", how="left")
            .merge(login_30d, on="user_id", how="left")
            .merge(login_features, on="user_id", how="left")
            .merge(night_large, on="user_id", how="left")
            .merge(new_device_withdraw, on="user_id", how="left")
            .merge(graph_slice, on="user_id", how="left")
        )
        result = result.fillna({
            "fiat_in_1d": 0.0,
            "fiat_in_7d": 0.0,
            "fiat_in_30d": 0.0,
            "fiat_out_1d": 0.0,
            "fiat_out_7d": 0.0,
            "fiat_out_30d": 0.0,
            "trade_count_7d": 0,
            "trade_count_30d": 0,
            "trade_notional_7d": 0.0,
            "trade_notional_30d": 0.0,
            "crypto_withdraw_1d": 0.0,
            "crypto_withdraw_7d": 0.0,
            "crypto_withdraw_30d": 0.0,
            "login_count_1d": 0,
            "login_count_7d": 0,
            "login_count_30d": 0,
            "unique_ip_count_7d": 0,
            "unique_ip_count_30d": 0,
            "unique_counterparty_wallet_count_30d": 0,
            "new_counterparty_wallet_ratio_30d": 0.0,
            "fiat_in_to_crypto_out_2h": False,
            "fiat_in_to_crypto_out_6h": False,
            "fiat_in_to_crypto_out_24h": False,
            "avg_dwell_time": 0.0,
            "large_deposit_withdraw_gap": 0.0,
            "geo_jump_count": 0,
            "vpn_ratio": 0.0,
            "new_device_ratio": 0.0,
            "ip_country_switch_count": 0,
            "night_login_ratio": 0.0,
            "night_large_withdrawal_ratio": 0.0,
            "new_device_withdrawal_24h": False,
            "shared_device_count": 0,
            "shared_bank_count": 0,
            "shared_wallet_count": 0,
            "blacklist_1hop_count": 0,
            "blacklist_2hop_count": 0,
            "component_size": 1,
            "fan_out_ratio": 0.0,
        })
        result["fiat_in_spike_ratio"] = _spike_ratio(result["fiat_in_1d"], result["fiat_in_30d"], 30.0)
        result["fiat_out_spike_ratio"] = _spike_ratio(result["fiat_out_1d"], result["fiat_out_30d"], 30.0)
        result["trade_notional_spike_ratio"] = _spike_ratio(result["trade_notional_7d"], result["trade_notional_30d"], 30.0 / 7.0)
        result["crypto_withdraw_spike_ratio"] = _spike_ratio(result["crypto_withdraw_1d"], result["crypto_withdraw_30d"], 30.0)
        result["login_spike_ratio"] = _spike_ratio(result["login_count_1d"], result["login_count_30d"], 30.0)

        result["feature_snapshot_id"] = result["feature_snapshot_id"].astype(str)
        records.append(result[ANOMALY_BASE_COLUMNS].copy())

    anomaly_frame = pd.concat(records, ignore_index=True) if records else _empty_anomaly_frame()
    store.replace_table("features.feature_snapshots_user_anomaly_30d", anomaly_frame)
    return anomaly_frame


if __name__ == "__main__":
    build_anomaly_feature_snapshots()
