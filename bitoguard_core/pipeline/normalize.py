from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from config import load_settings
from db.store import DuckDBStore, make_id, utc_now


PRIMARY_KEYS = {
    "users": "user_id",
    "login_events": "login_id",
    "fiat_transactions": "fiat_txn_id",
    "trade_orders": "trade_id",
    "crypto_transactions": "crypto_txn_id",
    "devices": "device_id",
    "user_device_links": "link_id",
    "bank_accounts": "bank_account_id",
    "user_bank_links": "link_id",
    "crypto_wallets": "wallet_id",
    "blacklist_feed": "blacklist_entry_id",
}

REQUIRED_COLUMNS = {
    "users": ["user_id", "created_at", "segment", "kyc_level"],
    "login_events": ["login_id", "user_id", "occurred_at", "device_id", "success"],
    "fiat_transactions": ["fiat_txn_id", "user_id", "occurred_at", "direction", "amount_twd", "status"],
    "trade_orders": ["trade_id", "user_id", "occurred_at", "side", "notional_twd", "status"],
    "crypto_transactions": ["crypto_txn_id", "user_id", "occurred_at", "direction", "amount_twd_equiv", "status"],
    "devices": ["device_id", "app_channel", "first_seen_at"],
    "user_device_links": ["link_id", "user_id", "device_id", "first_seen_at"],
    "bank_accounts": ["bank_account_id", "currency", "opened_at"],
    "user_bank_links": ["link_id", "user_id", "bank_account_id", "linked_at"],
    "crypto_wallets": ["wallet_id", "asset", "network", "created_at"],
    "known_blacklist_users": ["blacklist_entry_id", "user_id", "observed_at", "is_active"],
}

TIME_COLUMNS = {
    "users": ["created_at"],
    "login_events": ["occurred_at"],
    "fiat_transactions": ["occurred_at"],
    "trade_orders": ["occurred_at"],
    "crypto_transactions": ["occurred_at"],
    "devices": ["first_seen_at"],
    "user_device_links": ["first_seen_at", "last_seen_at"],
    "bank_accounts": ["opened_at"],
    "user_bank_links": ["linked_at"],
    "crypto_wallets": ["created_at"],
    "blacklist_feed": ["observed_at"],
}

STRING_NORMALIZATION_COLUMNS = {
    "users": ["segment", "kyc_level", "occupation", "declared_source_of_funds", "residence_country", "nationality"],
    "login_events": ["ip_country", "ip_city"],
    "fiat_transactions": ["direction", "currency", "method", "status"],
    "trade_orders": ["side", "base_asset", "quote_asset", "order_type", "status"],
    "crypto_transactions": ["direction", "asset", "network", "status"],
    "devices": ["device_type", "os_family", "app_channel"],
    "bank_accounts": ["bank_code", "bank_name", "country", "currency"],
    "crypto_wallets": ["wallet_kind", "asset", "network"],
    "known_blacklist_users": ["source", "reason_code"],
}

ENUM_COLUMNS = {
    "fiat_transactions": {
        "direction": {"deposit", "withdrawal"},
        "status": {"completed"},
        "currency": {"twd"},
        "method": {"bank_transfer"},
    },
    "trade_orders": {
        "side": {"buy", "sell"},
        "status": {"filled"},
        "order_type": {"market"},
    },
    "crypto_transactions": {
        "direction": {"deposit", "withdrawal"},
        "status": {"completed"},
    },
    "devices": {
        "app_channel": {"app", "web"},
    },
    "bank_accounts": {
        "currency": {"twd"},
    },
}


def _to_timestamp(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return ts.dt.tz_convert(timezone.utc)


def _normalize_string(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.replace({"": pd.NA, "nan": pd.NA, "<na>": pd.NA})


def _record_quality_issue(store: DuckDBStore, table_name: str, issue_type: str, issue_detail: str, row_count: int) -> None:
    store.execute(
        """
        INSERT INTO ops.data_quality_issues (issue_id, recorded_at, table_name, issue_type, issue_detail, row_count)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (make_id("dqi"), utc_now(), table_name, issue_type, issue_detail, row_count),
    )


def _dedupe_latest(frame: pd.DataFrame, primary_key: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    if frame[primary_key].isna().any():
        frame = frame[frame[primary_key].notna()].copy()
    if "_loaded_at" not in frame.columns:
        return frame.drop_duplicates(subset=[primary_key], keep="last").copy()
    frame = frame.sort_values(["_loaded_at", primary_key]).drop_duplicates(subset=[primary_key], keep="last")
    return frame.copy()


def _drop_rows_with_issue(
    store: DuckDBStore,
    frame: pd.DataFrame,
    table_name: str,
    issue_type: str,
    issue_detail: str,
    mask: pd.Series,
) -> pd.DataFrame:
    count = int(mask.sum())
    if count <= 0:
        return frame
    _record_quality_issue(store, table_name, issue_type, issue_detail, count)
    return frame.loc[~mask].copy()


def _normalize_table(
    store: DuckDBStore,
    table_name: str,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    canonical = frame.copy()
    for column in STRING_NORMALIZATION_COLUMNS.get(table_name, []):
        if column in canonical.columns:
            canonical[column] = _normalize_string(canonical[column])

    for column in REQUIRED_COLUMNS.get(table_name, []):
        if column not in canonical.columns:
            continue
        missing_mask = canonical[column].isna()
        canonical = _drop_rows_with_issue(
            store,
            canonical,
            table_name,
            "missing_required_value",
            f"Removed rows with missing required field: {column}",
            missing_mask,
        )

    for column in TIME_COLUMNS.get(table_name, []):
        parsed = _to_timestamp(canonical[column])
        invalid_mask = parsed.isna()
        canonical = _drop_rows_with_issue(
            store,
            canonical,
            table_name,
            "invalid_timestamp",
            f"Removed rows with invalid timestamp field: {column}",
            invalid_mask,
        )
        parsed = _to_timestamp(canonical[column])
        canonical[column] = parsed

    for column, allowed in ENUM_COLUMNS.get(table_name, {}).items():
        if column not in canonical.columns:
            continue
        canonical[column] = _normalize_string(canonical[column])
        invalid_mask = canonical[column].notna() & ~canonical[column].isin(allowed)
        canonical = _drop_rows_with_issue(
            store,
            canonical,
            table_name,
            "invalid_enum_value",
            f"Removed rows with invalid value in {column}; allowed={sorted(allowed)}",
            invalid_mask,
        )
    return canonical


def normalize_raw_to_canonical() -> None:
    settings = load_settings()
    store = DuckDBStore(settings.db_path)
    table_names = [
        "users",
        "login_events",
        "fiat_transactions",
        "trade_orders",
        "crypto_transactions",
        "devices",
        "user_device_links",
        "bank_accounts",
        "user_bank_links",
        "crypto_wallets",
    ]
    for table_name in table_names:
        raw = store.read_table(f"raw.{table_name}")
        if raw.empty:
            continue
        primary_key = PRIMARY_KEYS[table_name]
        before = len(raw)
        canonical = _dedupe_latest(raw, primary_key)
        dropped = before - len(canonical)
        if dropped > 0:
            _record_quality_issue(store, table_name, "duplicate_primary_key", f"Dropped {dropped} duplicate rows", dropped)
        if primary_key in canonical.columns and canonical[primary_key].isna().any():
            _record_quality_issue(store, table_name, "null_primary_key", "Removed rows with null primary key", int(canonical[primary_key].isna().sum()))
            canonical = canonical[canonical[primary_key].notna()].copy()
        canonical = _normalize_table(store, table_name, canonical)
        if "_sync_run_id" in canonical.columns:
            canonical = canonical.drop(columns=["_sync_run_id", "_loaded_at"])
        store.replace_table(f"canonical.{table_name}", canonical)

    blacklist = store.read_table("raw.known_blacklist_users")
    if not blacklist.empty:
        canonical_blacklist = _dedupe_latest(blacklist, "blacklist_entry_id")
        canonical_blacklist = _normalize_table(store, "known_blacklist_users", canonical_blacklist)
        canonical_blacklist = canonical_blacklist.drop(columns=["_sync_run_id", "_loaded_at"])
        store.replace_table("canonical.blacklist_feed", canonical_blacklist)


if __name__ == "__main__":
    normalize_raw_to_canonical()
