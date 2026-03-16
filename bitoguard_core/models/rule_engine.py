from __future__ import annotations

import json

import pandas as pd


RULE_DEFINITIONS = {
    # ── Velocity / flow rules ──────────────────────────────────────────────────
    "fast_cash_out_2h": "法幣入金後 2 小時內提領虛幣",
    "fast_cash_out_24h": "法幣入金後 24 小時內提領虛幣",
    # ── Device / IP rules ─────────────────────────────────────────────────────
    "new_device_new_ip_large_withdraw": "新裝置、新 IP 且出現大額提領",
    "night_new_device_withdraw": "深夜提領且伴隨新裝置跡象",
    # ── Graph / relational rules ──────────────────────────────────────────────
    "shared_device_ring": "共用裝置關聯帳戶達 3 人以上",
    "blacklist_2hop": "與黑名單帳戶存在 2-hop 內關聯",
    "blacklist_1hop": "直接與黑名單帳戶相連",
    # ── Fan-out / structuring ─────────────────────────────────────────────────
    "high_fan_out": "高扇出比 (轉出至多個帳戶) 且網路規模大",
    "volume_vs_declared_mismatch": "實際交易量遠超申報月收入",
    # ── Peer-deviation rules ──────────────────────────────────────────────────
    "extreme_fiat_peer_volume": "法幣流量顯著高於同 KYC 群組",
    "extreme_withdraw_peer_volume": "虛幣提領量顯著高於同 KYC 群組",
    # ── Cross-channel layering rules (v2 features) ────────────────────────────
    "fiat_passthrough": "法幣入金即出金 (人頭帳戶/快速過水)",
    "layering_burst": "法幣入金同期虛幣出金爆發 (分層洗錢)",
}

# Severity levels: high=3, medium=2, low=1
RULE_SEVERITY: dict[str, int] = {
    "fast_cash_out_2h": 3,
    "fast_cash_out_24h": 2,
    "new_device_new_ip_large_withdraw": 3,
    "night_new_device_withdraw": 2,
    "shared_device_ring": 2,
    "blacklist_2hop": 3,
    "blacklist_1hop": 3,
    "high_fan_out": 2,
    "volume_vs_declared_mismatch": 2,
    "extreme_fiat_peer_volume": 2,
    "extreme_withdraw_peer_volume": 2,
    "fiat_passthrough": 3,
    "layering_burst": 3,
}


def _get(frame: pd.DataFrame, col: str, default=0) -> pd.Series:
    """Safely get a column from a DataFrame, returning default series if missing."""
    if col in frame.columns:
        return frame[col]
    return pd.Series(default, index=frame.index)


def evaluate_rules(feature_frame: pd.DataFrame) -> pd.DataFrame:
    scored = feature_frame[["user_id", "snapshot_date"]].copy()

    # ── Velocity / flow ────────────────────────────────────────────────────────
    scored["fast_cash_out_2h"] = _get(feature_frame, "fiat_in_to_crypto_out_2h", False).astype(bool)
    scored["fast_cash_out_24h"] = _get(feature_frame, "fiat_in_to_crypto_out_24h", False).astype(bool)

    # ── Device / IP ────────────────────────────────────────────────────────────
    scored["new_device_new_ip_large_withdraw"] = (
        _get(feature_frame, "new_device_withdrawal_24h", False).astype(bool)
        & (_get(feature_frame, "ip_country_switch_count") >= 2)
        & (_get(feature_frame, "crypto_withdraw_30d") >= 50000)
    )
    scored["night_new_device_withdraw"] = (
        (_get(feature_frame, "night_large_withdrawal_ratio") > 0)
        & (_get(feature_frame, "new_device_ratio") > 0)
    )

    # ── Graph / relational ─────────────────────────────────────────────────────
    scored["shared_device_ring"] = _get(feature_frame, "shared_device_count") >= 3
    scored["blacklist_2hop"] = _get(feature_frame, "blacklist_2hop_count") >= 1
    scored["blacklist_1hop"] = _get(feature_frame, "blacklist_1hop_count") >= 1

    # ── Fan-out / structuring ──────────────────────────────────────────────────
    scored["high_fan_out"] = (
        (_get(feature_frame, "fan_out_ratio") >= 3.0)
        & (_get(feature_frame, "component_size", 1) >= 5)
    )
    # Declared-vs-actual volume: ratio > 5 = using exchange far beyond declared income
    scored["volume_vs_declared_mismatch"] = (
        _get(feature_frame, "actual_volume_expected_ratio") >= 5.0
    )

    # ── Peer-deviation rules ───────────────────────────────────────────────────
    # Trigger when user is in top 1% of fiat volume among same-KYC peers
    fiat_peer_pct = _get(feature_frame, "fiat_in_30d_peer_pct", 0.0)
    scored["extreme_fiat_peer_volume"] = fiat_peer_pct >= 0.99

    withdraw_peer_pct = _get(feature_frame, "crypto_withdraw_30d_peer_pct", 0.0)
    scored["extreme_withdraw_peer_volume"] = withdraw_peer_pct >= 0.99

    # ── Cross-channel layering rules (v2 features) ─────────────────────────────
    # Fiat pass-through: fiat in → fiat out within 24h ≥ 2 occurrences
    # Threshold of 2+ prevents single legitimate same-day deposit+refund from firing
    scored["fiat_passthrough"] = (
        _get(feature_frame, "fiat_dep_to_fiat_wdr_within_24h") >= 2
    )
    # Layering burst: simultaneous spike in fiat inflows AND crypto outflows
    # xch_layering_intensity = twd_dep_burst_ratio * crypto_wdr_burst_ratio
    # >5 means both channels burst at >2x normal simultaneously (hallmark of layering)
    scored["layering_burst"] = (
        _get(feature_frame, "xch_layering_intensity") >= 5.0
    )

    rule_cols = list(RULE_DEFINITIONS.keys())
    rule_severities = pd.Series(RULE_SEVERITY)

    scored["rule_score"] = (
        scored[rule_cols].astype(float)
        .multiply(rule_severities.reindex(rule_cols).values, axis=1)
        .sum(axis=1)
        / rule_severities.sum()
    )
    scored["rule_hit_count"] = scored[rule_cols].astype(int).sum(axis=1)
    scored["rule_hits"] = scored.apply(
        lambda row: json.dumps([name for name in rule_cols if bool(row[name])], ensure_ascii=False),
        axis=1,
    )
    scored["top_reason_codes"] = scored["rule_hits"]
    return scored
