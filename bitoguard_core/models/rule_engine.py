from __future__ import annotations

import json

import pandas as pd


RULE_DEFINITIONS = {
    "fast_cash_out_2h": "法幣入金後 2 小時內提領虛幣",
    "new_device_new_ip_large_withdraw": "新裝置、新 IP 且出現大額提領",
    "night_new_device_withdraw": "深夜提領且伴隨新裝置跡象",
    "shared_device_ring": "共用裝置關聯帳戶達 3 人以上",
    "blacklist_2hop": "與黑名單帳戶存在 2-hop 內關聯",
}


def evaluate_rules(feature_frame: pd.DataFrame) -> pd.DataFrame:
    scored = feature_frame[["user_id", "snapshot_date"]].copy()
    scored["fast_cash_out_2h"] = feature_frame["fiat_in_to_crypto_out_2h"].astype(bool)
    scored["new_device_new_ip_large_withdraw"] = (
        feature_frame["new_device_withdrawal_24h"].astype(bool)
        & (feature_frame["ip_country_switch_count"] >= 2)
        & (feature_frame["crypto_withdraw_30d"] >= 50000)
    )
    scored["night_new_device_withdraw"] = (
        feature_frame["night_large_withdrawal_ratio"] > 0
    ) & (feature_frame["new_device_ratio"] > 0)
    scored["shared_device_ring"] = feature_frame["shared_device_count"] >= 3
    scored["blacklist_2hop"] = feature_frame["blacklist_2hop_count"] >= 1
    scored["rule_score"] = scored[list(RULE_DEFINITIONS.keys())].sum(axis=1) / len(RULE_DEFINITIONS)
    scored["rule_hits"] = scored.apply(
        lambda row: json.dumps([name for name in RULE_DEFINITIONS if bool(row[name])], ensure_ascii=False),
        axis=1,
    )
    scored["top_reason_codes"] = scored["rule_hits"]
    return scored
