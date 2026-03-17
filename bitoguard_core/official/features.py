from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from official.cohorts import build_official_cohorts
from official.common import EVENT_TIME_COLUMNS, feature_output_path, list_event_cutoffs, load_clean_table, safe_ratio, to_utc_timestamp


@dataclass(frozen=True)
class FeatureBuildResult:
    cutoff_tag: str
    output_path: str
    row_count: int
    latest_event_at: str


ROLLING_WINDOWS = ((1, "1d"), (3, "3d"), (7, "7d"), (30, "30d"))


def _normalize_user_id(frame: pd.DataFrame) -> pd.DataFrame:
    copied = frame.copy()
    if "user_id" in copied.columns:
        copied["user_id"] = pd.to_numeric(copied["user_id"], errors="coerce").astype("Int64")
    return copied


def _prepare_table(name: str, cutoff_ts: pd.Timestamp | None) -> pd.DataFrame:
    frame = _normalize_user_id(load_clean_table(name))
    time_column = EVENT_TIME_COLUMNS.get(name)
    if time_column:
        frame[time_column] = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
        if cutoff_ts is not None:
            frame = frame[frame[time_column] < cutoff_ts].copy()
    return frame


def _series_or_default(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _add_group_aggregations(
    frame: pd.DataFrame,
    group_col: str,
    numeric_col: str,
    prefix: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[
            group_col,
            f"{prefix}_count",
            f"{prefix}_sum",
            f"{prefix}_avg",
            f"{prefix}_max",
        ])
    grouped = frame.groupby(group_col)[numeric_col].agg(["count", "sum", "mean", "max"]).reset_index()
    return grouped.rename(columns={
        "count": f"{prefix}_count",
        "sum": f"{prefix}_sum",
        "mean": f"{prefix}_avg",
        "max": f"{prefix}_max",
    })


def _nunique_or_empty(frame: pd.DataFrame, group_col: str, value_col: str, output_name: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[group_col, output_name])
    return frame.groupby(group_col)[value_col].nunique(dropna=True).reset_index(name=output_name)


def _boolean_ratio(frame: pd.DataFrame, group_col: str, mask: pd.Series, output_name: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[group_col, output_name])
    subset = frame[[group_col]].copy()
    subset[output_name] = mask.astype(float)
    return subset.groupby(group_col)[output_name].mean().reset_index()


def _days_since_last(frame: pd.DataFrame, group_col: str, time_col: str, cutoff_ts: pd.Timestamp, output_name: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[group_col, output_name])
    latest = frame.groupby(group_col)[time_col].max().reset_index()
    latest[output_name] = (cutoff_ts - latest[time_col]).dt.total_seconds() / 86400.0
    return latest[[group_col, output_name]]


def _activity_days(frame: pd.DataFrame, group_col: str, time_col: str, output_name: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[group_col, output_name])
    copied = frame[[group_col, time_col]].copy()
    copied["activity_day"] = copied[time_col].dt.date
    return copied.groupby(group_col)["activity_day"].nunique().reset_index(name=output_name)


def _daily_concentration(frame: pd.DataFrame, group_col: str, time_col: str, value_col: str, output_name: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[group_col, output_name])
    copied = frame[[group_col, time_col, value_col]].copy()
    copied["activity_day"] = copied[time_col].dt.date
    daily = copied.groupby([group_col, "activity_day"])[value_col].sum().reset_index()
    total = daily.groupby(group_col)[value_col].sum().rename("total")
    max_daily = daily.groupby(group_col)[value_col].max().rename("max_daily")
    result = pd.concat([total, max_daily], axis=1).reset_index()
    result[output_name] = safe_ratio(result["max_daily"], result["total"])
    return result[[group_col, output_name]]


def _fast_cashout_features(
    twd_transfer: pd.DataFrame,
    crypto_transfer: pd.DataFrame,
) -> pd.DataFrame:
    deposits = twd_transfer[twd_transfer["kind_label"] == "deposit"][["user_id", "created_at", "amount_twd", "id"]].copy()
    withdrawals = crypto_transfer[crypto_transfer["kind_label"] == "withdrawal"][["user_id", "created_at", "amount_twd_equiv", "id"]].copy()
    if deposits.empty or withdrawals.empty:
        return pd.DataFrame(columns=["user_id", "fast_cashout_24h_count", "fast_cashout_72h_count", "avg_cashout_gap_hours"])

    merged = deposits.merge(withdrawals, on="user_id", suffixes=("_fiat", "_crypto"))
    merged = merged[merged["created_at_crypto"] >= merged["created_at_fiat"]].copy()
    if merged.empty:
        return pd.DataFrame(columns=["user_id", "fast_cashout_24h_count", "fast_cashout_72h_count", "avg_cashout_gap_hours"])
    merged["gap_hours"] = (merged["created_at_crypto"] - merged["created_at_fiat"]).dt.total_seconds() / 3600.0
    earliest = merged.sort_values(["user_id", "id_fiat", "gap_hours"]).drop_duplicates(["user_id", "id_fiat"])
    grouped = earliest.groupby("user_id").agg(
        fast_cashout_24h_count=("gap_hours", lambda s: int((s <= 24).sum())),
        fast_cashout_72h_count=("gap_hours", lambda s: int((s <= 72).sum())),
        avg_cashout_gap_hours=("gap_hours", "mean"),
    )
    return grouped.reset_index()


def _window_frame(
    frame: pd.DataFrame,
    time_col: str,
    cutoff_ts: pd.Timestamp,
    days: int,
) -> pd.DataFrame:
    start = cutoff_ts - pd.Timedelta(days=days)
    return frame[(frame[time_col] >= start) & (frame[time_col] < cutoff_ts)].copy()


def build_official_features(
    cutoff_ts: pd.Timestamp | None = None,
    cutoff_tag: str = "full",
) -> pd.DataFrame:
    cohorts = build_official_cohorts(write_outputs=True)
    cutoff_ts = to_utc_timestamp(cutoff_ts)

    user_info = _prepare_table("user_info", cutoff_ts)
    twd_transfer = _prepare_table("twd_transfer", cutoff_ts)
    crypto_transfer = _prepare_table("crypto_transfer", cutoff_ts)
    usdt_swap = _prepare_table("usdt_swap", cutoff_ts)
    usdt_twd_trading = _prepare_table("usdt_twd_trading", cutoff_ts)

    profile_columns = [
        "user_id",
        "sex",
        "age",
        "career",
        "income_source",
        "user_source",
        "kyc_level",
        "sex_label",
        "career_label",
        "income_source_label",
        "user_source_label",
        "has_email_confirmation",
        "has_level1_kyc",
        "has_level2_kyc",
        "days_email_to_level1",
        "days_level1_to_level2",
    ]
    profile = user_info[[column for column in profile_columns if column in user_info.columns]].copy()
    base = cohorts.drop(
        columns=[
            "sex",
            "age",
            "career",
            "income_source",
            "user_source",
            "kyc_level",
            "sex_label",
            "career_label",
            "income_source_label",
            "user_source_label",
            "has_email_confirmation",
            "has_level1_kyc",
            "has_level2_kyc",
            "days_email_to_level1",
            "days_level1_to_level2",
        ],
        errors="ignore",
    ).merge(profile, on="user_id", how="left")
    base["sex"] = _series_or_default(base, "sex")
    base["age"] = _series_or_default(base, "age")
    base["career"] = _series_or_default(base, "career")
    base["income_source"] = _series_or_default(base, "income_source")
    base["user_source"] = _series_or_default(base, "user_source")
    base["kyc_level"] = _series_or_default(base, "kyc_level")
    base["days_email_to_level1"] = _series_or_default(base, "days_email_to_level1")
    base["days_level1_to_level2"] = _series_or_default(base, "days_level1_to_level2")
    base["has_profile"] = base["has_profile"].fillna(False).astype(int)
    base["has_email_confirmation"] = base["has_email_confirmation"].fillna(False).astype(int)
    base["has_level1_kyc"] = base["has_level1_kyc"].fillna(False).astype(int)
    base["has_level2_kyc"] = base["has_level2_kyc"].fillna(False).astype(int)

    twd_stats = _add_group_aggregations(twd_transfer, "user_id", "amount_twd", "twd_total")
    twd_deposit = _add_group_aggregations(twd_transfer[twd_transfer["kind_label"] == "deposit"], "user_id", "amount_twd", "twd_deposit")
    twd_withdraw = _add_group_aggregations(twd_transfer[twd_transfer["kind_label"] == "withdrawal"], "user_id", "amount_twd", "twd_withdraw")
    twd_days = _activity_days(twd_transfer, "user_id", "created_at", "twd_active_days")
    twd_last = _days_since_last(twd_transfer, "user_id", "created_at", cutoff_ts or list_event_cutoffs()[1], "days_since_last_twd_transfer")

    crypto_stats = _add_group_aggregations(crypto_transfer, "user_id", "amount_twd_equiv", "crypto_total")
    crypto_withdraw = _add_group_aggregations(crypto_transfer[crypto_transfer["kind_label"] == "withdrawal"], "user_id", "amount_twd_equiv", "crypto_withdraw")
    crypto_deposit = _add_group_aggregations(crypto_transfer[crypto_transfer["kind_label"] == "deposit"], "user_id", "amount_twd_equiv", "crypto_deposit")
    crypto_days = _activity_days(crypto_transfer, "user_id", "created_at", "crypto_active_days")
    crypto_last = _days_since_last(crypto_transfer, "user_id", "created_at", cutoff_ts or list_event_cutoffs()[1], "days_since_last_crypto_transfer")
    crypto_protocols = _nunique_or_empty(crypto_transfer, "user_id", "protocol_label", "crypto_protocol_count")
    crypto_currencies = _nunique_or_empty(crypto_transfer, "user_id", "currency", "crypto_currency_count")
    crypto_internal_ratio = _boolean_ratio(crypto_transfer, "user_id", crypto_transfer["sub_kind_label"] == "internal", "crypto_internal_ratio")
    relation_rows = crypto_transfer[
        crypto_transfer["relation_user_id"].notna() & crypto_transfer["is_internal_transfer"].eq(True)
    ].copy()
    relation_counts = relation_rows.groupby("user_id").size().reset_index(name="relation_transfer_count") if not relation_rows.empty else pd.DataFrame(columns=["user_id", "relation_transfer_count"])
    relation_users = _nunique_or_empty(relation_rows, "user_id", "relation_user_id", "relation_unique_counterparty_count")

    trade_stats = _add_group_aggregations(usdt_twd_trading, "user_id", "trade_notional_twd", "order_total")
    trade_buy = _add_group_aggregations(usdt_twd_trading[usdt_twd_trading["side_label"] == "buy_usdt_with_twd"], "user_id", "trade_notional_twd", "order_buy")
    trade_sell = _add_group_aggregations(usdt_twd_trading[usdt_twd_trading["side_label"] == "sell_usdt_for_twd"], "user_id", "trade_notional_twd", "order_sell")
    trade_days = _activity_days(usdt_twd_trading, "user_id", "updated_at", "trade_active_days")
    trade_night = _boolean_ratio(usdt_twd_trading, "user_id", usdt_twd_trading["updated_at"].dt.hour.isin([0, 1, 2, 3, 4, 5]), "trade_night_ratio")
    trade_market = _boolean_ratio(usdt_twd_trading, "user_id", usdt_twd_trading["order_type_label"] == "market", "trade_market_ratio")
    trade_source_api = _boolean_ratio(usdt_twd_trading, "user_id", usdt_twd_trading["source_label"] == "api", "trade_api_ratio")
    trade_concentration = _daily_concentration(usdt_twd_trading, "user_id", "updated_at", "trade_notional_twd", "trade_intraday_concentration")

    swap_stats = _add_group_aggregations(usdt_swap, "user_id", "twd_amount", "swap_total")
    swap_buy = _add_group_aggregations(usdt_swap[usdt_swap["kind_label"] == "buy_usdt_with_twd"], "user_id", "twd_amount", "swap_buy")
    swap_sell = _add_group_aggregations(usdt_swap[usdt_swap["kind_label"] == "sell_usdt_for_twd"], "user_id", "twd_amount", "swap_sell")
    swap_days = _activity_days(usdt_swap, "user_id", "created_at", "swap_active_days")
    swap_night = _boolean_ratio(usdt_swap, "user_id", usdt_swap["created_at"].dt.hour.isin([0, 1, 2, 3, 4, 5]), "swap_night_ratio")

    fast_cashout = _fast_cashout_features(twd_transfer, crypto_transfer)

    # ── Taiwan-time night & weekend TWD deposit/withdrawal timing features ─────
    # Taiwan = UTC+8. Night = Taiwan 22:00-05:59 = UTC 14:00-21:59.
    # Weekend adjustment: UTC hour >= 16 means Taiwan has crossed midnight.
    twd_dep = twd_transfer[twd_transfer["kind_label"] == "deposit"].copy()
    _dep_utc_hour = twd_dep["created_at"].dt.hour
    twd_deposit_night = _boolean_ratio(
        twd_dep, "user_id",
        _dep_utc_hour.isin(range(14, 22)),
        "twd_deposit_night_ratio_tw",
    )
    _dep_utc_weekday = twd_dep["created_at"].dt.dayofweek
    _dep_tw_weekday = (_dep_utc_weekday + (_dep_utc_hour >= 16).astype(int)) % 7
    twd_deposit_weekend = _boolean_ratio(
        twd_dep, "user_id", _dep_tw_weekday.isin([5, 6]), "twd_deposit_weekend_ratio",
    )
    twd_wd = twd_transfer[twd_transfer["kind_label"] == "withdrawal"].copy()
    _wd_utc_hour = twd_wd["created_at"].dt.hour
    twd_withdraw_night = _boolean_ratio(
        twd_wd, "user_id",
        _wd_utc_hour.isin(range(14, 22)),
        "twd_withdraw_night_ratio_tw",
    )

    # ── External crypto wallet diversity (structuring / layering signals) ──────
    ct_ext_dep = crypto_transfer[
        (crypto_transfer["kind_label"] == "deposit")
        & (crypto_transfer["is_external_transfer"].eq(True))
    ].copy()
    crypto_dep_wallets = _nunique_or_empty(ct_ext_dep, "user_id", "from_wallet_hash", "crypto_unique_deposit_wallets")
    ct_ext_wd = crypto_transfer[
        (crypto_transfer["kind_label"] == "withdrawal")
        & (crypto_transfer["is_external_transfer"].eq(True))
    ].copy()
    crypto_wd_wallets = _nunique_or_empty(ct_ext_wd, "user_id", "to_wallet_hash", "crypto_unique_withdraw_wallets")

    # ── Crypto external transfer IP diversity (high separation for fraud) ───────
    # Fraudsters use more unique IPs for external crypto transfers (sep=0.87 on labeled data)
    ct_ext_all = crypto_transfer[crypto_transfer["is_external_transfer"].eq(True)].copy()
    crypto_ext_ip_diversity = _nunique_or_empty(ct_ext_all, "user_id", "source_ip_hash", "crypto_ext_ip_diversity")

    # ── TWD deposit coefficient of variation (irregular deposit amounts) ────────
    twd_dep_cv = (
        twd_transfer[twd_transfer["kind_label"] == "deposit"]
        .groupby("user_id")["amount_twd"]
        .agg(lambda x: float(x.std() / (x.mean() + 1)) if len(x) > 1 else 0.0)
        .reset_index(name="twd_deposit_cv")
    )

    frames = [
        twd_stats, twd_deposit, twd_withdraw, twd_days, twd_last,
        twd_deposit_night, twd_deposit_weekend, twd_withdraw_night,
        crypto_stats, crypto_withdraw, crypto_deposit, crypto_days, crypto_last, crypto_protocols,
        crypto_currencies, crypto_internal_ratio, relation_counts, relation_users,
        crypto_dep_wallets, crypto_wd_wallets, crypto_ext_ip_diversity,
        trade_stats, trade_buy, trade_sell, trade_days, trade_night, trade_market, trade_source_api, trade_concentration,
        swap_stats, swap_buy, swap_sell, swap_days, swap_night,
        fast_cashout, twd_dep_cv,
    ]
    resolved_cutoff = cutoff_ts or list_event_cutoffs()[1]
    for window_days, window_tag in ROLLING_WINDOWS:
        twd_window = _window_frame(twd_transfer, "created_at", resolved_cutoff, window_days)
        crypto_window = _window_frame(crypto_transfer, "created_at", resolved_cutoff, window_days)
        trade_window = _window_frame(usdt_twd_trading, "updated_at", resolved_cutoff, window_days)
        swap_window = _window_frame(usdt_swap, "created_at", resolved_cutoff, window_days)
        frames.extend(
            [
                _add_group_aggregations(twd_window, "user_id", "amount_twd", f"twd_total_{window_tag}"),
                _add_group_aggregations(
                    twd_window[twd_window["kind_label"] == "withdrawal"],
                    "user_id",
                    "amount_twd",
                    f"twd_withdraw_{window_tag}",
                ),
                _add_group_aggregations(crypto_window, "user_id", "amount_twd_equiv", f"crypto_total_{window_tag}"),
                _add_group_aggregations(
                    crypto_window[crypto_window["kind_label"] == "withdrawal"],
                    "user_id",
                    "amount_twd_equiv",
                    f"crypto_withdraw_{window_tag}",
                ),
                _add_group_aggregations(trade_window, "user_id", "trade_notional_twd", f"order_total_{window_tag}"),
                _add_group_aggregations(swap_window, "user_id", "twd_amount", f"swap_total_{window_tag}"),
                _activity_days(twd_window, "user_id", "created_at", f"twd_active_days_{window_tag}"),
                _activity_days(crypto_window, "user_id", "created_at", f"crypto_active_days_{window_tag}"),
                _activity_days(trade_window, "user_id", "updated_at", f"trade_active_days_{window_tag}"),
                _activity_days(swap_window, "user_id", "created_at", f"swap_active_days_{window_tag}"),
            ]
        )
    features = base
    for frame in frames:
        features = features.merge(frame, on="user_id", how="left")

    protected_columns = {
        "user_id",
        "cohort",
        "status",
        "is_known_blacklist",
        "needs_prediction",
        "in_train_label",
        "in_predict_label",
        "is_shadow_overlap",
    }
    numeric_fill_zero = [
        column
        for column in features.columns
        if column not in protected_columns and pd.api.types.is_numeric_dtype(features[column]) and not pd.api.types.is_bool_dtype(features[column])
    ]
    bool_fill_false = [
        column
        for column in features.columns
        if column not in protected_columns and pd.api.types.is_bool_dtype(features[column])
    ]
    features[numeric_fill_zero] = features[numeric_fill_zero].fillna(0.0)
    for column in bool_fill_false:
        features[column] = features[column].fillna(False)

    features["twd_net_amount"] = features["twd_deposit_sum"] - features["twd_withdraw_sum"]
    features["twd_in_out_ratio"] = safe_ratio(features["twd_deposit_sum"], features["twd_withdraw_sum"])
    features["twd_withdraw_concentration"] = safe_ratio(features["twd_withdraw_max"], features["twd_withdraw_sum"])
    features["crypto_net_amount"] = features["crypto_deposit_sum"] - features["crypto_withdraw_sum"]
    features["crypto_in_out_ratio"] = safe_ratio(features["crypto_deposit_sum"], features["crypto_withdraw_sum"])
    features["relation_fan_out_ratio"] = safe_ratio(features["relation_unique_counterparty_count"], features["relation_transfer_count"])
    features["trade_buy_sell_ratio"] = safe_ratio(features["order_buy_sum"], features["order_sell_sum"])
    features["swap_buy_sell_ratio"] = safe_ratio(features["swap_buy_sum"], features["swap_sell_sum"])
    features["activity_days_total"] = features[["twd_active_days", "crypto_active_days", "trade_active_days", "swap_active_days"]].sum(axis=1)
    features["night_activity_ratio"] = (
        features["trade_night_ratio"] * (features["trade_active_days"] > 0).astype(float)
        + features["swap_night_ratio"] * (features["swap_active_days"] > 0).astype(float)
    ) / (
        (features["trade_active_days"] > 0).astype(float) + (features["swap_active_days"] > 0).astype(float)
    ).replace(0, np.nan)
    features["night_activity_ratio"] = features["night_activity_ratio"].fillna(0.0)
    features["fast_cashout_24h_flag"] = (features["fast_cashout_24h_count"] > 0).astype(int)
    features["fast_cashout_72h_flag"] = (features["fast_cashout_72h_count"] > 0).astype(int)

    # ── FATF AML typology proxy features ─────────────────────────────────────
    _dep_count = features["twd_deposit_count"].fillna(0)
    _dep_sum   = features["twd_deposit_sum"].fillna(0)
    _dep_avg   = features["twd_deposit_avg"].fillna(0)
    _dep_max   = features["twd_deposit_max"].fillna(0)
    _age       = features["age"].fillna(0)
    # 1. Structuring ratio: many small sub-50k TWD deposits vs total volume
    features["typology_structuring_ratio"] = (
        (_dep_count / (_dep_sum / 50_000.0 + 1.0)).clip(0, 10) / 10.0
    ).astype("float32")
    # 2. Dormancy-burst: old account with recent transfer activity
    #    Uses days_since_last_twd_transfer (low = recent); twd_total_{7d} unavailable
    _days_since = features["days_since_last_twd_transfer"].fillna(999)
    features["typology_dormancy_burst"] = (
        (_age / 365.0).clip(0, 1)
        * (1.0 / (_days_since / 7.0 + 1.0))
        * (_dep_count > 0).astype(float)
    ).clip(0, 1).astype("float32")
    # 3. Round-amount proxy: avg deposit near NT$10k multiples
    _avg_dep_val = _dep_sum / (_dep_count + 1.0)
    _remainder   = _avg_dep_val % 10_000.0
    features["typology_round_amount"] = (
        (1.0 - _remainder / 10_000.0).clip(0, 1)
    ).astype("float32")
    # 4. Multi-asset layering: swap activity × cashout intensity
    _swap_cnt     = features["swap_total_count"].fillna(0)
    _cashout_rate = features["fast_cashout_72h_count"].fillna(0) / (_dep_count + 1.0)
    features["typology_multi_asset_layering"] = (
        (_swap_cnt * _cashout_rate).clip(0, 10)
    ).astype("float32")
    # 5. Deposit spike: max/avg ratio indicates burst deposits (velocity proxy)
    #    twd_total_{7d,30d} are 100% null; use max-to-avg concentration instead.
    features["typology_velocity_accel"] = (
        (_dep_max / (_dep_avg + 1.0)).clip(0, 10)
    ).astype("float32")
    # 6. Same-day cycle: rapid fiat-in → crypto-out indicator
    features["typology_same_day_cycle"] = (
        features["fast_cashout_24h_count"].fillna(0).clip(0, 10)
    ).astype("float32")
    # 7. Cashout ratio: fraction of total TWD that is withdrawal (layering signal)
    features["typology_cashout_ratio"] = safe_ratio(
        features["twd_withdraw_sum"].fillna(0),
        features["twd_total_sum"].fillna(0),
    ).astype("float32")
    # 8. Deposit concentration: largest single deposit / total (burst indicator)
    features["typology_deposit_spike"] = safe_ratio(
        _dep_max,
        _dep_sum,
    ).astype("float32")

    features["snapshot_cutoff_at"] = resolved_cutoff
    features["snapshot_cutoff_tag"] = cutoff_tag
    features.to_parquet(feature_output_path("official_user_features", cutoff_tag), index=False)
    return features


def main() -> None:
    build_official_features()


if __name__ == "__main__":
    main()
