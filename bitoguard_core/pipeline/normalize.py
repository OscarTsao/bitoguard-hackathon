from __future__ import annotations

from datetime import timezone

import pandas as pd

from config import load_settings
from db.store import DuckDBStore, _validate_table_name, make_id, utc_now


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


def _to_timestamp(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return ts.dt.tz_convert(timezone.utc)


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


def _replace_table_in_transaction(conn, table_name: str, dataframe: pd.DataFrame) -> None:
    _validate_table_name(table_name)
    if dataframe.empty:
        conn.execute(f"DELETE FROM {table_name}")
        return
    conn.register("normalized_df", dataframe)
    conn.execute(f"DELETE FROM {table_name}")
    conn.execute(f"INSERT INTO {table_name} SELECT * FROM normalized_df")
    conn.unregister("normalized_df")


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
    canonical_frames: dict[str, pd.DataFrame] = {}
    for table_name in table_names:
        raw = store.read_table(f"raw.{table_name}")
        primary_key = PRIMARY_KEYS[table_name]
        before = len(raw)
        canonical = _dedupe_latest(raw, primary_key)
        dropped = before - len(canonical)
        if dropped > 0:
            _record_quality_issue(store, table_name, "duplicate_primary_key", f"Dropped {dropped} duplicate rows", dropped)
        if primary_key in canonical.columns and canonical[primary_key].isna().any():
            _record_quality_issue(store, table_name, "null_primary_key", "Removed rows with null primary key", int(canonical[primary_key].isna().sum()))
            canonical = canonical[canonical[primary_key].notna()].copy()
        for column in TIME_COLUMNS[table_name]:
            canonical[column] = _to_timestamp(canonical[column])
        if "_sync_run_id" in canonical.columns:
            canonical = canonical.drop(columns=["_sync_run_id", "_loaded_at"])
        canonical_frames[f"canonical.{table_name}"] = canonical

    blacklist = store.read_table("raw.known_blacklist_users")
    canonical_blacklist = _dedupe_latest(blacklist, "blacklist_entry_id")
    if "observed_at" in canonical_blacklist.columns:
        canonical_blacklist["observed_at"] = _to_timestamp(canonical_blacklist["observed_at"])
    if "_sync_run_id" in canonical_blacklist.columns:
        canonical_blacklist = canonical_blacklist.drop(columns=["_sync_run_id", "_loaded_at"])
    canonical_frames["canonical.blacklist_feed"] = canonical_blacklist

    with store.transaction() as conn:
        for table_name, dataframe in canonical_frames.items():
            _replace_table_in_transaction(conn, table_name, dataframe)


if __name__ == "__main__":
    normalize_raw_to_canonical()
