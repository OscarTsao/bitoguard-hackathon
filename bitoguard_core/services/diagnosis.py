from __future__ import annotations

import json
from datetime import timedelta

import pandas as pd

from config import load_settings
from db.store import DuckDBStore
from services.explain import explain_user


FEATURE_ZH = {
    # --- Fiat velocity (legacy v1 names) ---
    "fiat_in_to_crypto_out_2h": "法幣入金後 2 小時內提領虛幣",
    "fiat_in_to_crypto_out_6h": "法幣入金後 6 小時內提領虛幣",
    "fiat_in_to_crypto_out_24h": "法幣入金後 24 小時內提領虛幣",
    # --- Profile / KYC ---
    "monthly_income_twd": "月收入 (TWD)",
    "expected_monthly_volume_twd": "預期月交易量",
    "actual_volume_expected_ratio": "實際交易量 / 預期交易量",
    "actual_fiat_income_ratio": "實際法幣入金 / 月收入",
    "account_age_days": "帳戶開戶天數",
    "kyc_level_code": "KYC 等級",
    # --- Graph ---
    "component_size": "關聯群體規模",
    "shared_device_count": "共用裝置關聯帳戶數",
    "shared_bank_count": "共用銀行帳戶關聯數",
    "shared_wallet_count": "共用錢包關聯數",
    "blacklist_1hop_count": "黑名單 1-hop 鄰居數",
    "blacklist_2hop_count": "黑名單 2-hop 鄰居數",
    "fan_out_ratio": "出金擴散比",
    # --- IP / Device ---
    "night_large_withdrawal_ratio": "深夜大額提領比例",
    "ip_country_switch_count": "IP 國家切換次數",
    "ip_n_entities": "IP 關聯帳戶數",
    "ip_unique_ips": "使用 IP 數",
    # --- TWD fiat (v2 windowed velocity) ---
    "twd_dep_sum": "法幣入金總額 (TWD)",
    "twd_wdr_sum": "法幣出金總額 (TWD)",
    "twd_dep_7d_sum": "7 日法幣入金總額",
    "twd_dep_30d_sum": "30 日法幣入金總額",
    "twd_wdr_7d_sum": "7 日法幣出金總額",
    "twd_wdr_30d_sum": "30 日法幣出金總額",
    "twd_dep_burst_ratio": "法幣入金爆發比 (7d/歷史均值)",
    "twd_dep_round_10k_ratio": "整數萬元入金比 (結構化信號)",
    "twd_dep_near_500k_ratio": "接近 50 萬上限入金比",
    "twd_dep_amt_entropy": "法幣入金金額分佈熵值 (低=結構化信號)",
    "twd_dep_span_days": "法幣交易活躍天數",
    "twd_all_count": "法幣交易總筆數",
    "twd_weekend_share": "法幣交易週末佔比",
    # --- Crypto (v2 windowed velocity) ---
    "crypto_wdr_twd_sum": "虛幣出金總額 (TWD 換算)",
    "crypto_dep_twd_sum": "虛幣入金總額 (TWD 換算)",
    "crypto_wdr_7d_sum": "7 日虛幣出金總額",
    "crypto_wdr_30d_sum": "30 日虛幣出金總額",
    "crypto_dep_7d_sum": "7 日虛幣入金總額",
    "crypto_wdr_burst_ratio": "虛幣出金爆發比 (7d/歷史均值)",
    "crypto_wdr_to_dep_ratio": "虛幣出金 / 入金比",
    "crypto_n_currencies": "使用虛幣種類數",
    "crypto_n_protocols": "使用鏈路協議數",
    "crypto_trx_tx_share": "TRX/TRC20 交易佔比",
    "crypto_trx_amt_share": "TRX/TRC20 金額佔比",
    "crypto_n_from_wallets": "接收錢包地址數",
    "crypto_n_to_wallets": "發送目標錢包數",
    "crypto_from_wallet_conc": "來源錢包集中度",
    "crypto_weekend_share": "虛幣交易週末佔比",
    # --- Cross-channel layering ---
    "xch_cashout_ratio_lifetime": "跨通道提現比 (全期)",
    "xch_cashout_ratio_7d": "跨通道提現比 (7 日)",
    "xch_cashout_ratio_30d": "跨通道提現比 (30 日)",
    "xch_layering_intensity": "分層洗錢強度指標",
    # --- Sequence ---
    "fiat_dep_to_swap_buy_within_1h": "1 小時內法幣轉虛幣次數",
    "fiat_dep_to_swap_buy_within_6h": "6 小時內法幣轉虛幣次數",
    "fiat_dep_to_swap_buy_within_24h": "24 小時內法幣轉虛幣次數",
    "fiat_dep_to_swap_buy_within_72h": "72 小時內法幣轉虛幣次數",
    "crypto_dep_to_fiat_wdr_within_1h": "1 小時內虛幣入金後法幣出金次數",
    "crypto_dep_to_fiat_wdr_within_6h": "6 小時內虛幣入金後法幣出金次數",
    "crypto_dep_to_fiat_wdr_within_24h": "24 小時內虛幣入金後法幣出金次數",
    "crypto_dep_to_fiat_wdr_within_72h": "72 小時內虛幣入金後法幣出金次數",
    "fiat_dep_to_fiat_wdr_within_24h": "24 小時內法幣入金即出金次數 (疑似人頭帳戶)",
    "fiat_dep_to_fiat_wdr_within_72h": "72 小時內法幣入金即出金次數 (疑似人頭帳戶)",
    "dwell_hours": "首次入金到首次出金間隔 (小時)",
    "early_3d_volume": "開戶前 3 天交易量",
    "early_3d_count": "開戶前 3 天交易次數",
    # --- Weekend activity ---
    "twd_weekend_share": "法幣交易週末佔比",
    "crypto_weekend_share": "虛幣交易週末佔比",
    # --- Rule signals ---
    "rule_fast_cash_out_2h": "規則: 快速提現 (2h)",
    "rule_high_volume": "規則: 高額交易",
    "rule_structuring": "規則: 疑似分拆交易",
    "rule_new_device_withdrawal": "規則: 新裝置出金",
}


def _timeline_summary(login: pd.DataFrame, crypto: pd.DataFrame, trade: pd.DataFrame, user_id: str) -> list[dict]:
    timeline = []
    for frame, source, amount_column in [
        (login, "login", None),
        (crypto, "crypto", "amount_twd_equiv"),
        (trade, "trade", "notional_twd"),
    ]:
        subset = frame[frame["user_id"] == user_id].copy()
        if subset.empty:
            continue
        time_col = [col for col in subset.columns if col.endswith("_at")][0]
        subset[time_col] = pd.to_datetime(subset[time_col], utc=True)
        latest = subset.sort_values(time_col, ascending=False).head(5)
        for _, row in latest.iterrows():
            timeline.append({
                "time": row[time_col].isoformat(),
                "type": source,
                "amount": None if amount_column is None else float(row.get(amount_column, 0) or 0),
            })
    timeline.sort(key=lambda item: item["time"], reverse=True)
    return timeline[:10]


def _load_prediction(
    store: DuckDBStore,
    user_id: str,
    *,
    prediction_id: str | None = None,
    snapshot_date: object | None = None,
) -> pd.DataFrame:
    if prediction_id:
        return store.fetch_df(
            "SELECT * FROM ops.model_predictions WHERE prediction_id = ? LIMIT 1",
            (prediction_id,),
        )

    if snapshot_date is not None:
        return store.fetch_df(
            """
            SELECT *
            FROM ops.model_predictions
            WHERE user_id = ? AND snapshot_date = ?
            ORDER BY prediction_time DESC
            LIMIT 1
            """,
            (user_id, pd.Timestamp(snapshot_date).date()),
        )

    return store.fetch_df(
        """
        SELECT *
        FROM ops.model_predictions
        WHERE user_id = ?
        ORDER BY snapshot_date DESC, prediction_time DESC
        LIMIT 1
        """,
        (user_id,),
    )


def build_risk_diagnosis(
    user_id: str,
    *,
    prediction_id: str | None = None,
    snapshot_date: object | None = None,
) -> dict:
    settings = load_settings()
    store = DuckDBStore(settings.db_path)
    predictions = _load_prediction(store, user_id, prediction_id=prediction_id, snapshot_date=snapshot_date)
    if predictions.empty:
        raise ValueError(f"No prediction found for user_id={user_id}")
    prediction = predictions.iloc[0].to_dict()
    prediction_snapshot_date = pd.Timestamp(predictions.iloc[0]["snapshot_date"]).date()
    snapshot_end = pd.Timestamp(prediction_snapshot_date, tz="UTC") + timedelta(days=1)
    login = store.fetch_df(
        """
        SELECT *
        FROM canonical.login_events
        WHERE user_id = ? AND CAST(occurred_at AS TIMESTAMPTZ) < ?
        ORDER BY CAST(occurred_at AS TIMESTAMPTZ) DESC
        LIMIT 50
        """,
        (user_id, snapshot_end),
    )
    crypto = store.fetch_df(
        """
        SELECT *
        FROM canonical.crypto_transactions
        WHERE user_id = ? AND CAST(occurred_at AS TIMESTAMPTZ) < ?
        ORDER BY CAST(occurred_at AS TIMESTAMPTZ) DESC
        LIMIT 50
        """,
        (user_id, snapshot_end),
    )
    trade = store.fetch_df(
        """
        SELECT *
        FROM canonical.trade_orders
        WHERE user_id = ? AND CAST(occurred_at AS TIMESTAMPTZ) < ?
        ORDER BY CAST(occurred_at AS TIMESTAMPTZ) DESC
        LIMIT 50
        """,
        (user_id, snapshot_end),
    )
    features = store.fetch_df(
        """
        SELECT *
        FROM features.feature_snapshots_user_day
        WHERE user_id = ? AND snapshot_date = ?
        LIMIT 1
        """,
        (user_id, prediction_snapshot_date),
    )
    _feat = features.iloc[0] if not features.empty else None
    graph_evidence = {
        "shared_device_count": int(_feat["shared_device_count"]) if _feat is not None else 0,
        "shared_bank_count": int(_feat["shared_bank_count"]) if _feat is not None else 0,
        "shared_wallet_count": int(_feat["shared_wallet_count"]) if _feat is not None else 0,
        "blacklist_1hop_count": int(_feat["blacklist_1hop_count"]) if _feat is not None else 0,
        "blacklist_2hop_count": int(_feat["blacklist_2hop_count"]) if _feat is not None else 0,
        "component_size": int(_feat["component_size"]) if _feat is not None else 0,
    }
    model_version = str(prediction.get("model_version") or "")
    explain_model_version = next(
        (part.split(":", 1)[-1] for part in model_version.split("+") if "lgbm_" in part),
        None,
    )
    shap_factors = []
    if explain_model_version is not None:
        shap_factors = explain_user(
            user_id,
            snapshot_date=prediction_snapshot_date,
            model_version=explain_model_version,
        )
    shap_top = [
        {
            "feature": item["feature"],
            "feature_zh": FEATURE_ZH.get(item["feature"], item["feature"]),
            "value": item["value"],
            "impact": item["impact"],
        }
        for item in shap_factors[:5]
    ]
    rule_hits = json.loads(prediction["rule_hits"]) if prediction["rule_hits"] else []
    recommended_action = "monitor"
    if prediction["risk_level"] in {"high", "critical"}:
        recommended_action = "manual_review"
    if graph_evidence["blacklist_1hop_count"] > 0 and prediction["risk_level"] == "critical":
        recommended_action = "hold_withdrawal"
    summary_zh = (
        f"用戶 {user_id} 目前風險等級為 {prediction['risk_level']}，風險分數 {prediction['risk_score']:.1f}。"
        f"主要原因包含 {', '.join(rule_hits[:3]) if rule_hits else '模型風險分數偏高'}。"
    )
    return {
        "user_id": user_id,
        "summary_zh": summary_zh,
        "risk_summary": {
            "risk_score": float(prediction["risk_score"]),
            "risk_level": prediction["risk_level"],
            "prediction_time": str(prediction["prediction_time"]),
            "snapshot_date": str(prediction_snapshot_date),
            "prediction_id": prediction.get("prediction_id"),
        },
        "shap_top_factors": shap_top,
        "rule_hits": rule_hits,
        "graph_evidence": graph_evidence,
        "timeline_summary": _timeline_summary(login, crypto, trade, user_id),
        "recommended_action": recommended_action,
    }
