from __future__ import annotations

import numpy as np
import pandas as pd
from transductive_v1.common import (
    EVENT_TIME_COLUMNS,
    default_temporal_cutoff,
    feature_path,
    list_event_cutoffs,
    load_clean_table,
    safe_ratio,
    to_utc_timestamp,
)
from transductive_v1.dataset import build_user_universe


ROLLING_WINDOWS = ((1, "1d"), (3, "3d"), (7, "7d"), (30, "30d"))
ANOMALY_BASE_COLUMNS = (
    "twd_total_sum",
    "twd_withdraw_sum",
    "crypto_total_sum",
    "crypto_withdraw_sum",
    "order_total_sum",
    "swap_total_sum",
)


def _prepare_table(name: str, cutoff_ts: pd.Timestamp | None) -> pd.DataFrame:
    frame = load_clean_table(name).copy()
    if "user_id" in frame.columns:
        frame["user_id"] = pd.to_numeric(frame["user_id"], errors="coerce").astype("Int64")
    time_column = EVENT_TIME_COLUMNS.get(name)
    if time_column:
        frame[time_column] = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
        if cutoff_ts is not None:
            frame = frame[frame[time_column] < cutoff_ts].copy()
    return frame


def _add_group_aggregations(frame: pd.DataFrame, group_col: str, numeric_col: str, prefix: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[group_col, f"{prefix}_count", f"{prefix}_sum", f"{prefix}_avg", f"{prefix}_max"])
    grouped = frame.groupby(group_col)[numeric_col].agg(["count", "sum", "mean", "max"]).reset_index()
    return grouped.rename(
        columns={
            "count": f"{prefix}_count",
            "sum": f"{prefix}_sum",
            "mean": f"{prefix}_avg",
            "max": f"{prefix}_max",
        }
    )


def _nunique_or_empty(frame: pd.DataFrame, group_col: str, value_col: str, output_name: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[group_col, output_name])
    return frame.groupby(group_col)[value_col].nunique(dropna=True).reset_index(name=output_name)


def _boolean_ratio(frame: pd.DataFrame, group_col: str, mask: pd.Series, output_name: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[group_col, output_name])
    copied = frame[[group_col]].copy()
    copied[output_name] = mask.astype(float)
    return copied.groupby(group_col)[output_name].mean().reset_index()


def _activity_days(frame: pd.DataFrame, group_col: str, time_col: str, output_name: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[group_col, output_name])
    copied = frame[[group_col, time_col]].copy()
    copied["activity_day"] = copied[time_col].dt.date
    return copied.groupby(group_col)["activity_day"].nunique().reset_index(name=output_name)


def _days_since_last(frame: pd.DataFrame, group_col: str, time_col: str, cutoff_ts: pd.Timestamp, output_name: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[group_col, output_name])
    latest = frame.groupby(group_col)[time_col].max().reset_index()
    latest[output_name] = (cutoff_ts - latest[time_col]).dt.total_seconds() / 86400.0
    return latest[[group_col, output_name]]


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


def _window_frame(frame: pd.DataFrame, time_col: str, cutoff_ts: pd.Timestamp, days: int) -> pd.DataFrame:
    start = cutoff_ts - pd.Timedelta(days=days)
    return frame[(frame[time_col] >= start) & (frame[time_col] < cutoff_ts)].copy()


def _fast_cashout_features(twd_transfer: pd.DataFrame, crypto_transfer: pd.DataFrame) -> pd.DataFrame:
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


def _cross_table_sequences(
    twd_transfer: pd.DataFrame,
    crypto_transfer: pd.DataFrame,
    usdt_swap: pd.DataFrame,
) -> pd.DataFrame:
    result_frames: list[pd.DataFrame] = []
    deposits = twd_transfer[twd_transfer["kind_label"] == "deposit"][["user_id", "created_at", "id"]].copy()
    swap_buy = usdt_swap[usdt_swap["kind_label"] == "buy_usdt_with_twd"][["user_id", "created_at", "id"]].copy()
    if not deposits.empty and not swap_buy.empty:
        merged = deposits.merge(swap_buy, on="user_id", suffixes=("_deposit", "_swap"))
        merged = merged[merged["created_at_swap"] >= merged["created_at_deposit"]].copy()
        if not merged.empty:
            merged["gap_hours"] = (merged["created_at_swap"] - merged["created_at_deposit"]).dt.total_seconds() / 3600.0
            earliest = merged.sort_values(["user_id", "id_deposit", "gap_hours"]).drop_duplicates(["user_id", "id_deposit"])
            seq = earliest.groupby("user_id").agg(
                fiat_to_swap_24h_count=("gap_hours", lambda s: int((s <= 24).sum())),
                avg_fiat_to_swap_gap_hours=("gap_hours", "mean"),
            )
            result_frames.append(seq.reset_index())
    if not result_frames:
        return pd.DataFrame(columns=["user_id", "fiat_to_swap_24h_count", "avg_fiat_to_swap_gap_hours"])
    output = result_frames[0]
    for frame in result_frames[1:]:
        output = output.merge(frame, on="user_id", how="outer")
    return output


def _attach_support_scores(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    anomaly_components = []
    for column in ANOMALY_BASE_COLUMNS:
        if column not in result.columns:
            continue
        series = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
        log_series = np.log1p(series.clip(lower=0))
        median = float(log_series.median())
        mad = float((log_series - median).abs().median())
        denom = mad if mad > 1e-9 else 1.0
        robust = ((log_series - median) / denom).clip(lower=0.0)
        anomaly_components.append(robust)
    if anomaly_components:
        stacked = pd.concat(anomaly_components, axis=1)
        result["anomaly_score"] = stacked.mean(axis=1).rank(method="average", pct=True).fillna(0.0)
    else:
        result["anomaly_score"] = 0.0
    result["rule_score"] = 0.0
    result["top_reason_codes"] = "[]"
    return result


def build_label_free_user_features(
    cutoff_ts: pd.Timestamp | None = None,
    cutoff_tag: str = "full",
    write_outputs: bool = True,
) -> pd.DataFrame:
    universe = build_user_universe(cutoff_tag=cutoff_tag, write_outputs=write_outputs)
    cutoff_ts = to_utc_timestamp(cutoff_ts)
    resolved_cutoff = cutoff_ts or list_event_cutoffs()[1]

    twd_transfer = _prepare_table("twd_transfer", cutoff_ts)
    crypto_transfer = _prepare_table("crypto_transfer", cutoff_ts)
    usdt_swap = _prepare_table("usdt_swap", cutoff_ts)
    usdt_twd_trading = _prepare_table("usdt_twd_trading", cutoff_ts)

    twd_stats = _add_group_aggregations(twd_transfer, "user_id", "amount_twd", "twd_total")
    twd_deposit = _add_group_aggregations(twd_transfer[twd_transfer["kind_label"] == "deposit"], "user_id", "amount_twd", "twd_deposit")
    twd_withdraw = _add_group_aggregations(twd_transfer[twd_transfer["kind_label"] == "withdrawal"], "user_id", "amount_twd", "twd_withdraw")
    twd_days = _activity_days(twd_transfer, "user_id", "created_at", "twd_active_days")
    twd_last = _days_since_last(twd_transfer, "user_id", "created_at", resolved_cutoff, "days_since_last_twd_transfer")

    crypto_stats = _add_group_aggregations(crypto_transfer, "user_id", "amount_twd_equiv", "crypto_total")
    crypto_withdraw = _add_group_aggregations(crypto_transfer[crypto_transfer["kind_label"] == "withdrawal"], "user_id", "amount_twd_equiv", "crypto_withdraw")
    crypto_deposit = _add_group_aggregations(crypto_transfer[crypto_transfer["kind_label"] == "deposit"], "user_id", "amount_twd_equiv", "crypto_deposit")
    crypto_days = _activity_days(crypto_transfer, "user_id", "created_at", "crypto_active_days")
    crypto_last = _days_since_last(crypto_transfer, "user_id", "created_at", resolved_cutoff, "days_since_last_crypto_transfer")
    crypto_protocols = _nunique_or_empty(crypto_transfer, "user_id", "protocol_label", "crypto_protocol_count")
    crypto_currencies = _nunique_or_empty(crypto_transfer, "user_id", "currency", "crypto_currency_count")
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
    trade_api = _boolean_ratio(usdt_twd_trading, "user_id", usdt_twd_trading["source_label"] == "api", "trade_api_ratio")
    trade_concentration = _daily_concentration(usdt_twd_trading, "user_id", "updated_at", "trade_notional_twd", "trade_intraday_concentration")

    swap_stats = _add_group_aggregations(usdt_swap, "user_id", "twd_amount", "swap_total")
    swap_buy = _add_group_aggregations(usdt_swap[usdt_swap["kind_label"] == "buy_usdt_with_twd"], "user_id", "twd_amount", "swap_buy")
    swap_sell = _add_group_aggregations(usdt_swap[usdt_swap["kind_label"] == "sell_usdt_for_twd"], "user_id", "twd_amount", "swap_sell")
    swap_days = _activity_days(usdt_swap, "user_id", "created_at", "swap_active_days")
    swap_night = _boolean_ratio(usdt_swap, "user_id", usdt_swap["created_at"].dt.hour.isin([0, 1, 2, 3, 4, 5]), "swap_night_ratio")

    fast_cashout = _fast_cashout_features(twd_transfer, crypto_transfer)
    cross_table = _cross_table_sequences(twd_transfer, crypto_transfer, usdt_swap)

    frames = [
        twd_stats, twd_deposit, twd_withdraw, twd_days, twd_last,
        crypto_stats, crypto_withdraw, crypto_deposit, crypto_days, crypto_last, crypto_protocols, crypto_currencies,
        relation_counts, relation_users,
        trade_stats, trade_buy, trade_sell, trade_days, trade_night, trade_market, trade_api, trade_concentration,
        swap_stats, swap_buy, swap_sell, swap_days, swap_night,
        fast_cashout, cross_table,
    ]
    for window_days, tag in ROLLING_WINDOWS:
        twd_window = _window_frame(twd_transfer, "created_at", resolved_cutoff, window_days)
        crypto_window = _window_frame(crypto_transfer, "created_at", resolved_cutoff, window_days)
        trade_window = _window_frame(usdt_twd_trading, "updated_at", resolved_cutoff, window_days)
        swap_window = _window_frame(usdt_swap, "created_at", resolved_cutoff, window_days)
        frames.extend(
            [
                _add_group_aggregations(twd_window, "user_id", "amount_twd", f"twd_total_{tag}"),
                _add_group_aggregations(crypto_window, "user_id", "amount_twd_equiv", f"crypto_total_{tag}"),
                _add_group_aggregations(trade_window, "user_id", "trade_notional_twd", f"order_total_{tag}"),
                _add_group_aggregations(swap_window, "user_id", "twd_amount", f"swap_total_{tag}"),
                _activity_days(twd_window, "user_id", "created_at", f"twd_active_days_{tag}"),
                _activity_days(crypto_window, "user_id", "created_at", f"crypto_active_days_{tag}"),
                _activity_days(trade_window, "user_id", "updated_at", f"trade_active_days_{tag}"),
                _activity_days(swap_window, "user_id", "created_at", f"swap_active_days_{tag}"),
            ]
        )

    features = universe.copy()
    for frame in frames:
        features = features.merge(frame, on="user_id", how="left")

    protected_columns = {"user_id", "cohort", "status", "needs_prediction", "in_train_label", "in_predict_label", "is_shadow_overlap"}
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
        features[column] = features[column].fillna(False).astype(int)

    features["twd_net_amount"] = features["twd_deposit_sum"] - features["twd_withdraw_sum"]
    features["twd_in_out_ratio"] = safe_ratio(features["twd_deposit_sum"], features["twd_withdraw_sum"])
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
    features["snapshot_cutoff_at"] = resolved_cutoff
    features["snapshot_cutoff_tag"] = cutoff_tag
    features = _attach_support_scores(features)

    if write_outputs:
        features.to_parquet(feature_path("label_free_user_features", cutoff_tag), index=False)
    return features
