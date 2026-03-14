# bitoguard_core/tests/test_feature_modules.py
from __future__ import annotations
import pandas as pd
import pytest
from features.profile_features import compute_profile_features


def _users_df():
    return pd.DataFrame([{
        "user_id": "u1",
        "created_at": "2025-01-01T00:00:00+08:00",
        "kyc_level": "level2",
        "occupation": "career_1",
        "monthly_income_twd": 50000.0,
        "declared_source_of_funds": "income_source_2",
        "activity_window": "web",
    }])


def test_profile_features_columns():
    result = compute_profile_features(_users_df())
    assert "user_id" in result.columns
    assert "kyc_level_code" in result.columns
    assert "account_age_days" in result.columns
    assert "occupation_code" in result.columns
    assert len(result) == 1


def test_profile_features_kyc_level2():
    result = compute_profile_features(_users_df())
    assert result.iloc[0]["kyc_level_code"] == 2


def test_profile_features_empty():
    result = compute_profile_features(pd.DataFrame(columns=_users_df().columns))
    assert len(result) == 0


from features.twd_features import compute_twd_features, _gap_stats, _agg_stats


def _fiat_df():
    return pd.DataFrame([
        {"user_id": "u1", "occurred_at": "2025-01-01T01:00:00+00:00", "direction": "deposit",    "amount_twd": 10000.0},
        {"user_id": "u1", "occurred_at": "2025-01-01T02:00:00+00:00", "direction": "deposit",    "amount_twd": 20000.0},
        {"user_id": "u1", "occurred_at": "2025-01-02T03:00:00+00:00", "direction": "withdrawal", "amount_twd": 5000.0},
        {"user_id": "u2", "occurred_at": "2025-01-05T10:00:00+00:00", "direction": "deposit",    "amount_twd": 100.0},
    ])


def test_twd_features_columns():
    result = compute_twd_features(_fiat_df())
    for col in ["twd_all_count", "twd_dep_count", "twd_wdr_count",
                "twd_net_flow", "twd_night_share",
                "twd_dep_gap_min", "twd_dep_rapid_1h_share"]:
        assert col in result.columns, f"missing {col}"


def test_twd_features_u1_counts():
    result = compute_twd_features(_fiat_df())
    u1 = result[result["user_id"] == "u1"].iloc[0]
    assert u1["twd_all_count"] == 3
    assert u1["twd_dep_count"] == 2
    assert u1["twd_wdr_count"] == 1
    assert u1["twd_net_flow"] == pytest.approx(30000.0 - 5000.0)


def test_twd_features_gap():
    result = compute_twd_features(_fiat_df())
    u1 = result[result["user_id"] == "u1"].iloc[0]
    # Two deposits 1h apart → gap_min ≈ 60 minutes
    assert u1["twd_dep_gap_min"] == pytest.approx(60.0, abs=5.0)


from features.crypto_features import compute_crypto_features


def _crypto_df():
    return pd.DataFrame([
        {"user_id": "u1", "occurred_at": "2025-01-01T01:00:00+00:00", "direction": "deposit",
         "amount_twd_equiv": 5000.0, "asset": "TRX", "network": "TRC20",
         "wallet_id": "w1", "counterparty_wallet_id": "ext1"},
        {"user_id": "u1", "occurred_at": "2025-01-01T03:00:00+00:00", "direction": "deposit",
         "amount_twd_equiv": 3000.0, "asset": "ETH", "network": "ERC20",
         "wallet_id": "w2", "counterparty_wallet_id": "ext2"},
        {"user_id": "u1", "occurred_at": "2025-01-02T08:00:00+00:00", "direction": "withdrawal",
         "amount_twd_equiv": 7000.0, "asset": "TRX", "network": "TRC20",
         "wallet_id": "w1", "counterparty_wallet_id": "ext3"},
    ])


def test_crypto_features_columns():
    result = compute_crypto_features(_crypto_df())
    for col in ["crypto_all_count", "crypto_dep_count", "crypto_wdr_count",
                "crypto_n_currencies", "crypto_trx_tx_share", "crypto_n_from_wallets",
                "crypto_dep_gap_min"]:
        assert col in result.columns, f"missing {col}"


def test_crypto_features_u1():
    result = compute_crypto_features(_crypto_df())
    u1 = result[result["user_id"] == "u1"].iloc[0]
    assert u1["crypto_all_count"] == 3
    assert u1["crypto_n_currencies"] == 2  # TRX + ETH
    assert u1["crypto_trx_tx_share"] == pytest.approx(2/3, abs=0.01)


from features.swap_features import compute_swap_features
from features.trading_features import compute_trading_features


def _trades_df():
    return pd.DataFrame([
        {"user_id": "u1", "occurred_at": "2025-01-01T10:00:00+00:00", "side": "buy",
         "base_asset": "USDT", "quote_asset": "TWD", "notional_twd": 10000.0,
         "order_type": "instant_swap"},
        {"user_id": "u1", "occurred_at": "2025-01-02T22:00:00+00:00", "side": "sell",
         "base_asset": "USDT", "quote_asset": "TWD", "notional_twd": 5000.0,
         "order_type": "instant_swap"},
        {"user_id": "u1", "occurred_at": "2025-01-03T12:00:00+00:00", "side": "buy",
         "base_asset": "USDT", "quote_asset": "TWD", "notional_twd": 20000.0,
         "order_type": "market"},
    ])


def test_swap_features_uses_instant_swap_only():
    result = compute_swap_features(_trades_df())
    u1 = result[result["user_id"] == "u1"].iloc[0]
    assert u1["swap_count"] == 2  # NOT 3


def test_swap_features_columns():
    result = compute_swap_features(_trades_df())
    for col in ["swap_count", "swap_buy_count", "swap_sell_count", "swap_buy_ratio", "swap_net_twd"]:
        assert col in result.columns


def test_trading_features_book_only():
    result = compute_trading_features(_trades_df())
    u1 = result[result["user_id"] == "u1"].iloc[0]
    assert u1["trade_count"] == 1


def test_trading_features_columns():
    result = compute_trading_features(_trades_df())
    for col in ["trade_count", "trade_buy_count", "trade_market_ratio", "trade_night_share"]:
        assert col in result.columns
