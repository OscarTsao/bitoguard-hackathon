from __future__ import annotations

from collections import defaultdict

import networkx as nx
import pandas as pd

from config import load_settings
from db.store import DuckDBStore


def _prefix(entity_type: str, entity_id: str) -> str:
    return f"{entity_type}:{entity_id}"


def _node_type(node_id: str) -> str:
    return node_id.split(":", 1)[0]


def _other_users_via_type(graph: nx.Graph, user_node: str, via_type: str) -> set[str]:
    related: set[str] = set()
    for neighbor in graph.neighbors(user_node):
        if _node_type(neighbor) != via_type:
            continue
        for second in graph.neighbors(neighbor):
            if second != user_node and _node_type(second) == "user":
                related.add(second.split(":", 1)[1])
    return related


def build_graph_features() -> pd.DataFrame:
    settings = load_settings()
    store = DuckDBStore(settings.db_path)
    edges = store.read_table("canonical.entity_edges")
    users = store.read_table("canonical.users")
    blacklist_feed = store.read_table("canonical.blacklist_feed")
    crypto_edges = edges[edges["relation_type"] == "crypto_transfer_to_wallet"].copy()
    if edges.empty or users.empty:
        empty = pd.DataFrame(columns=[
            "graph_feature_id", "user_id", "snapshot_date", "shared_device_count", "shared_bank_count",
            "shared_wallet_count", "blacklist_1hop_count", "blacklist_2hop_count", "component_size", "fan_out_ratio",
        ])
        store.replace_table("features.graph_features", empty)
        return empty

    edges["snapshot_time"] = pd.to_datetime(edges["snapshot_time"], utc=True)
    users["created_at"] = pd.to_datetime(users["created_at"], utc=True)
    if not blacklist_feed.empty:
        blacklist_feed["observed_at"] = pd.to_datetime(blacklist_feed["observed_at"], utc=True)

    date_start = edges["snapshot_time"].dt.date.min()
    date_end = edges["snapshot_time"].dt.date.max()
    snapshot_dates = pd.date_range(date_start, date_end, freq="D").date
    records: list[dict] = []

    for snapshot_date in snapshot_dates:
        snapshot_end = pd.Timestamp(snapshot_date).tz_localize("UTC") + pd.Timedelta(days=1)
        edge_window = edges[edges["snapshot_time"] < snapshot_end]
        active_users = users[users["created_at"] < snapshot_end]
        blacklisted_users = set()
        if not blacklist_feed.empty:
            blacklisted_users = set(
                blacklist_feed[blacklist_feed["observed_at"] < snapshot_end]["user_id"].tolist()
            )

        graph = nx.Graph()
        for _, row in edge_window.iterrows():
            src = _prefix(row["src_type"], row["src_id"])
            dst = _prefix(row["dst_type"], row["dst_id"])
            graph.add_edge(src, dst, relation_type=row["relation_type"])

        transfer_counts = defaultdict(int)
        distinct_transfer_targets = defaultdict(set)
        for _, row in crypto_edges[pd.to_datetime(crypto_edges["snapshot_time"], utc=True) < snapshot_end].iterrows():
            transfer_counts[row["src_id"]] += 1
            distinct_transfer_targets[row["src_id"]].add(row["dst_id"])

        for _, user in active_users.iterrows():
            user_id = user["user_id"]
            node = _prefix("user", user_id)
            if node not in graph:
                component_size = 1
                shared_devices = set()
                shared_banks = set()
                shared_wallets = set()
                blacklist_1hop = 0
                blacklist_2hop = 0
            else:
                shared_devices = _other_users_via_type(graph, node, "device")
                shared_banks = _other_users_via_type(graph, node, "bank_account")
                shared_wallets = _other_users_via_type(graph, node, "wallet")
                component_size = len(nx.node_connected_component(graph, node))
                blacklist_1hop = 0
                blacklist_2hop = 0
                lengths = nx.single_source_shortest_path_length(graph, node, cutoff=4)
                for target, distance in lengths.items():
                    if _node_type(target) != "user":
                        continue
                    target_user_id = target.split(":", 1)[1]
                    if target_user_id == user_id or target_user_id not in blacklisted_users:
                        continue
                    if distance <= 2:
                        blacklist_1hop += 1
                    elif distance <= 4:
                        blacklist_2hop += 1

            total_transfers = transfer_counts[user_id]
            fan_out_ratio = (
                len(distinct_transfer_targets[user_id]) / total_transfers
                if total_transfers > 0 else 0.0
            )
            records.append({
                "graph_feature_id": f"gf_{user_id}_{snapshot_date.isoformat()}",
                "user_id": user_id,
                "snapshot_date": pd.Timestamp(snapshot_date),
                "shared_device_count": len(shared_devices),
                "shared_bank_count": len(shared_banks),
                "shared_wallet_count": len(shared_wallets),
                "blacklist_1hop_count": blacklist_1hop,
                "blacklist_2hop_count": blacklist_2hop,
                "component_size": component_size,
                "fan_out_ratio": fan_out_ratio,
            })

    dataframe = pd.DataFrame(records)
    store.replace_table("features.graph_features", dataframe)
    return dataframe


if __name__ == "__main__":
    build_graph_features()
