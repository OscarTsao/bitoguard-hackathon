"""KYC/onboarding timing features derived from user_info + raw event timestamps.

Features (all label-free):
1. kyc2_to_first_large_crypto_wd_h  — KYC2完成到首次大額crypto提款（AUC=0.70）
2. kyc2_to_first_crypto_wd_h        — KYC2完成到首次任何crypto提款
3. reg_to_first_deposit_h           — 註冊到首次TWD存款
4. reg_to_first_tx_h                — 註冊到首次任何交易
5. active_fraction                  — 活躍時間佔帳戶壽命比例
6. shared_wallet_tx_count           — 使用共享錢包地址的交易次數
7. shared_wallet_flag               — 是否使用過共享錢包
8. unique_ext_wallet_count          — 獨特外部提款錢包數量
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path


def build_onboarding_features(data_dir: str | Path) -> pd.DataFrame:
    """Build KYC/onboarding timing + wallet features from raw data."""
    data_dir = Path(data_dir)

    ui = pd.read_parquet(data_dir / "user_info.parquet")
    twd = pd.read_parquet(data_dir / "twd_transfer.parquet")
    crypto = pd.read_parquet(data_dir / "crypto_transfer.parquet")
    swap = pd.read_parquet(data_dir / "usdt_swap.parquet")
    trade = pd.read_parquet(data_dir / "usdt_twd_trading.parquet")

    result = ui[["user_id"]].copy()

    # ── Timestamps ──
    all_ts = pd.concat([
        twd[["user_id", "created_at"]].rename(columns={"created_at": "ts"}),
        crypto[["user_id", "created_at"]].rename(columns={"created_at": "ts"}),
        swap[["user_id", "created_at"]].rename(columns={"created_at": "ts"}),
        trade[["user_id", "updated_at"]].rename(columns={"updated_at": "ts"}),
    ])
    all_ts["ts"] = pd.to_datetime(all_ts["ts"], utc=True)
    first_tx = all_ts.groupby("user_id")["ts"].min().rename("first_tx")
    last_tx = all_ts.groupby("user_id")["ts"].max().rename("last_tx")

    twd["created_at"] = pd.to_datetime(twd["created_at"], utc=True)
    crypto["created_at"] = pd.to_datetime(crypto["created_at"], utc=True)

    first_dep = twd[twd["is_deposit"] == True].groupby("user_id")["created_at"].min().rename("first_dep")

    # External crypto withdrawals
    ext_mask = crypto["is_internal_transfer"].fillna(False) == False
    kind_label = crypto.get("kind_label", pd.Series("withdrawal", index=crypto.index)).fillna("withdrawal").str.lower()
    wd_mask = kind_label.str.contains("withdrawal", na=False)
    crypto_wd = crypto[ext_mask & wd_mask]

    first_crypto_wd = crypto_wd.groupby("user_id")["created_at"].min().rename("first_crypto_wd")

    # Large crypto withdrawal (>50000 TWD equiv)
    amount_col = None
    for col_name in ["amount_twd_equiv", "amount_twd"]:
        if col_name in crypto_wd.columns:
            amount_col = col_name
            break
    if amount_col:
        large_wd = crypto_wd[pd.to_numeric(crypto_wd[amount_col], errors="coerce").fillna(0) > 50000]
    else:
        large_wd = crypto_wd.head(0)
    first_large_crypto_wd = large_wd.groupby("user_id")["created_at"].min().rename("first_large_crypto_wd")

    # Join timestamps
    ui["confirmed_at"] = pd.to_datetime(ui["confirmed_at"], utc=True)
    if "level2_finished_at" in ui.columns:
        ui["level2_finished_at"] = pd.to_datetime(ui["level2_finished_at"], utc=True)
    result = result.merge(ui[["user_id", "confirmed_at"] + (["level2_finished_at"] if "level2_finished_at" in ui.columns else [])], on="user_id", how="left")
    result = result.merge(first_tx.reset_index(), on="user_id", how="left")
    result = result.merge(last_tx.reset_index(), on="user_id", how="left")
    result = result.merge(first_dep.reset_index(), on="user_id", how="left")
    result = result.merge(first_crypto_wd.reset_index(), on="user_id", how="left")
    result = result.merge(first_large_crypto_wd.reset_index(), on="user_id", how="left")

    # ── Feature 1: KYC2 → first large crypto withdrawal (AUC=0.70) ──
    if "level2_finished_at" in result.columns:
        result["kyc2_to_first_large_crypto_wd_h"] = (
            (result["first_large_crypto_wd"] - result["level2_finished_at"]).dt.total_seconds() / 3600
        )
        result["kyc2_to_first_crypto_wd_h"] = (
            (result["first_crypto_wd"] - result["level2_finished_at"]).dt.total_seconds() / 3600
        )
    else:
        result["kyc2_to_first_large_crypto_wd_h"] = np.nan
        result["kyc2_to_first_crypto_wd_h"] = np.nan

    # ── Feature 3: Registration → first deposit ──
    result["reg_to_first_deposit_h"] = (
        (result["first_dep"] - result["confirmed_at"]).dt.total_seconds() / 3600
    )

    # ── Feature 4: Registration → first any transaction ──
    result["reg_to_first_tx_h"] = (
        (result["first_tx"] - result["confirmed_at"]).dt.total_seconds() / 3600
    )

    # ── Feature 5: Active fraction ──
    active_span = (result["last_tx"] - result["first_tx"]).dt.total_seconds() / 3600
    account_span = (result["last_tx"] - result["confirmed_at"]).dt.total_seconds() / 3600
    result["active_fraction"] = np.where(account_span > 0, active_span / account_span, 0.0)
    result["active_fraction"] = result["active_fraction"].clip(0, 1)

    # ── Features 6-7: Shared wallet ──
    if "to_wallet_hash" in crypto.columns:
        wallet_user_counts = crypto.groupby("to_wallet_hash")["user_id"].nunique()
        shared_wallets = set(wallet_user_counts[wallet_user_counts > 1].index)
        shared_tx = crypto[crypto["to_wallet_hash"].isin(shared_wallets)].groupby("user_id").size()
        result = result.merge(shared_tx.rename("shared_wallet_tx_count").reset_index(), on="user_id", how="left")
    else:
        result["shared_wallet_tx_count"] = 0
    result["shared_wallet_tx_count"] = result["shared_wallet_tx_count"].fillna(0)
    result["shared_wallet_flag"] = (result["shared_wallet_tx_count"] > 0).astype(float)

    # ── Feature 8: Unique external wallets ──
    if "to_wallet_hash" in crypto_wd.columns:
        ext_wallets = crypto_wd.groupby("user_id")["to_wallet_hash"].nunique()
        result = result.merge(ext_wallets.rename("unique_ext_wallet_count").reset_index(), on="user_id", how="left")
    else:
        result["unique_ext_wallet_count"] = 0
    result["unique_ext_wallet_count"] = result["unique_ext_wallet_count"].fillna(0)

    # ── Select output columns ──
    output_cols = [
        "user_id",
        "kyc2_to_first_large_crypto_wd_h",
        "kyc2_to_first_crypto_wd_h",
        "reg_to_first_deposit_h",
        "reg_to_first_tx_h",
        "active_fraction",
        "shared_wallet_tx_count",
        "shared_wallet_flag",
        "unique_ext_wallet_count",
    ]
    out = result[output_cols].copy()
    for col in output_cols[1:5]:
        out[col] = out[col].fillna(-1.0)
    for col in output_cols[5:]:
        out[col] = out[col].fillna(0.0)

    print(f"[onboarding_features] Built {len(output_cols)-1} features for {len(out)} users")
    return out
