from __future__ import annotations

import pandas as pd

from config import load_settings
from db.store import DuckDBStore


def rebuild_edges() -> None:
    settings = load_settings()
    store = DuckDBStore(settings.db_path)
    users = store.read_table("canonical.users")
    login_events = store.read_table("canonical.login_events")
    user_device_links = store.read_table("canonical.user_device_links")
    user_bank_links = store.read_table("canonical.user_bank_links")
    crypto_wallets = store.read_table("canonical.crypto_wallets")
    crypto_transactions = store.read_table("canonical.crypto_transactions")

    edges: list[dict] = []
    counter = 1

    for _, row in user_device_links.iterrows():
        edges.append({
            "edge_id": f"edge_{counter:06d}",
            "snapshot_time": row["first_seen_at"],
            "src_type": "user",
            "src_id": row["user_id"],
            "relation_type": "uses_device",
            "dst_type": "device",
            "dst_id": row["device_id"],
        })
        counter += 1

    for _, row in user_bank_links.iterrows():
        edges.append({
            "edge_id": f"edge_{counter:06d}",
            "snapshot_time": row["linked_at"],
            "src_type": "user",
            "src_id": row["user_id"],
            "relation_type": "uses_bank_account",
            "dst_type": "bank_account",
            "dst_id": row["bank_account_id"],
        })
        counter += 1

    for _, row in crypto_wallets[crypto_wallets["user_id"].notna()].iterrows():
        edges.append({
            "edge_id": f"edge_{counter:06d}",
            "snapshot_time": row["created_at"],
            "src_type": "user",
            "src_id": row["user_id"],
            "relation_type": "owns_wallet",
            "dst_type": "wallet",
            "dst_id": row["wallet_id"],
        })
        counter += 1

    for _, row in crypto_transactions.iterrows():
        if pd.isna(row["counterparty_wallet_id"]):
            continue
        edges.append({
            "edge_id": f"edge_{counter:06d}",
            "snapshot_time": row["occurred_at"],
            "src_type": "user",
            "src_id": row["user_id"],
            "relation_type": "crypto_transfer_to_wallet",
            "dst_type": "wallet",
            "dst_id": row["counterparty_wallet_id"],
        })
        counter += 1

    for _, row in login_events.iterrows():
        if pd.isna(row["ip_address"]):
            continue
        edges.append({
            "edge_id": f"edge_{counter:06d}",
            "snapshot_time": row["occurred_at"],
            "src_type": "user",
            "src_id": row["user_id"],
            "relation_type": "login_from_ip",
            "dst_type": "ip",
            "dst_id": row["ip_address"],
        })
        counter += 1

    edge_df = pd.DataFrame(edges)
    if edge_df.empty:
        edge_df = pd.DataFrame(columns=["edge_id", "snapshot_time", "src_type", "src_id", "relation_type", "dst_type", "dst_id"])
    store.replace_table("canonical.entity_edges", edge_df)


if __name__ == "__main__":
    rebuild_edges()
