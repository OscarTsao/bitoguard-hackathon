from __future__ import annotations

import json

import pandas as pd

from config import load_settings
from db.store import DuckDBStore
from services.explain import explain_user


FEATURE_ZH = {
    "fiat_in_to_crypto_out_2h": "法幣入金後 2 小時內提領虛幣",
    "fiat_in_to_crypto_out_6h": "法幣入金後 6 小時內提領虛幣",
    "fiat_in_to_crypto_out_24h": "法幣入金後 24 小時內提領虛幣",
    "monthly_income_twd": "月收入",
    "expected_monthly_volume_twd": "預期月交易量",
    "actual_volume_expected_ratio": "實際交易量 / 預期交易量",
    "actual_fiat_income_ratio": "實際法幣入金 / 月收入",
    "component_size": "關聯群體規模",
    "shared_device_count": "共用裝置關聯帳戶數",
    "shared_bank_count": "共用銀行帳戶關聯數",
    "shared_wallet_count": "共用錢包關聯數",
    "blacklist_1hop_count": "黑名單 1-hop 鄰居數",
    "blacklist_2hop_count": "黑名單 2-hop 鄰居數",
    "night_large_withdrawal_ratio": "深夜大額提領比例",
    "ip_country_switch_count": "IP 國家切換次數",
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


def build_risk_diagnosis(user_id: str) -> dict:
    settings = load_settings()
    store = DuckDBStore(settings.db_path)
    predictions = store.fetch_df(
        "SELECT * FROM ops.model_predictions WHERE user_id = ? ORDER BY snapshot_date DESC LIMIT 1",
        (user_id,),
    )
    if predictions.empty:
        raise ValueError(f"No prediction found for user_id={user_id}")
    prediction = predictions.iloc[0].to_dict()
    login = store.read_table("canonical.login_events")
    crypto = store.read_table("canonical.crypto_transactions")
    trade = store.read_table("canonical.trade_orders")
    features = store.fetch_df(
        "SELECT * FROM features.feature_snapshots_user_day WHERE user_id = ? ORDER BY snapshot_date DESC LIMIT 1",
        (user_id,),
    )
    graph_evidence = {
        "shared_device_count": int(features.iloc[0]["shared_device_count"]),
        "shared_bank_count": int(features.iloc[0]["shared_bank_count"]),
        "shared_wallet_count": int(features.iloc[0]["shared_wallet_count"]),
        "blacklist_1hop_count": int(features.iloc[0]["blacklist_1hop_count"]),
        "blacklist_2hop_count": int(features.iloc[0]["blacklist_2hop_count"]),
        "component_size": int(features.iloc[0]["component_size"]),
    }
    shap_factors = explain_user(user_id)
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
        },
        "shap_top_factors": shap_top,
        "rule_hits": rule_hits,
        "graph_evidence": graph_evidence,
        "timeline_summary": _timeline_summary(login, crypto, trade, user_id),
        "recommended_action": recommended_action,
    }
