from __future__ import annotations

from collections import defaultdict

import networkx as nx
import pandas as pd

from official.common import feature_output_path, list_event_cutoffs, load_clean_table, to_utc_timestamp


def _user_entity_graph(edges: pd.DataFrame, entity_prefix: str) -> tuple[nx.Graph, dict[int, int], dict[int, int]]:
    graph = nx.Graph()
    unique_other_users: dict[int, set[int]] = defaultdict(set)
    entity_degree_count: dict[int, int] = defaultdict(int)

    for entity_id, group in edges.groupby("entity_id"):
        users = sorted(set(group["user_id"].dropna().tolist()))
        if not users:
            continue
        entity_node = f"{entity_prefix}:{entity_id}"
        graph.add_node(entity_node)
        for user_id in users:
            user_node = f"user:{user_id}"
            graph.add_edge(user_node, entity_node)
            entity_degree_count[user_id] += 1
        if len(users) > 1:
            for user_id in users:
                unique_other_users[user_id].update(other for other in users if other != user_id)

    other_user_count = {user_id: len(others) for user_id, others in unique_other_users.items()}
    return graph, other_user_count, entity_degree_count


def _component_user_sizes(graph: nx.Graph) -> dict[int, int]:
    sizes: dict[int, int] = {}
    for component in nx.connected_components(graph):
        user_ids = [int(node.split(":", 1)[1]) for node in component if node.startswith("user:")]
        if not user_ids:
            continue
        for user_id in user_ids:
            sizes[user_id] = len(user_ids)
    return sizes


def build_official_graph_features(
    cutoff_ts: pd.Timestamp | None = None,
    cutoff_tag: str = "full",
) -> pd.DataFrame:
    user_info = load_clean_table("user_info").copy()
    user_info["user_id"] = pd.to_numeric(user_info["user_id"], errors="coerce").astype("Int64")
    cutoff_ts = to_utc_timestamp(cutoff_ts)

    twd_transfer = load_clean_table("twd_transfer").copy()
    crypto_transfer = load_clean_table("crypto_transfer").copy()
    usdt_twd_trading = load_clean_table("usdt_twd_trading").copy()
    for frame, time_column in (
        (twd_transfer, "created_at"),
        (crypto_transfer, "created_at"),
        (usdt_twd_trading, "updated_at"),
    ):
        frame["user_id"] = pd.to_numeric(frame["user_id"], errors="coerce").astype("Int64")
        frame[time_column] = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
        if cutoff_ts is not None:
            frame.drop(frame[frame[time_column] >= cutoff_ts].index, inplace=True)

    ip_edges = pd.concat(
        [
            twd_transfer[["user_id", "source_ip_hash"]].rename(columns={"source_ip_hash": "entity_id"}),
            crypto_transfer[["user_id", "source_ip_hash"]].rename(columns={"source_ip_hash": "entity_id"}),
            usdt_twd_trading[["user_id", "source_ip_hash"]].rename(columns={"source_ip_hash": "entity_id"}),
        ],
        ignore_index=True,
    )
    ip_edges = ip_edges[ip_edges["entity_id"].notna()].drop_duplicates()
    ip_graph, ip_other_users, ip_degree = _user_entity_graph(ip_edges, "ip")
    ip_components = _component_user_sizes(ip_graph)

    wallet_edges = pd.concat(
        [
            crypto_transfer[["user_id", "from_wallet_hash"]].rename(columns={"from_wallet_hash": "entity_id"}),
            crypto_transfer[["user_id", "to_wallet_hash"]].rename(columns={"to_wallet_hash": "entity_id"}),
        ],
        ignore_index=True,
    )
    wallet_edges = wallet_edges[wallet_edges["entity_id"].notna()].drop_duplicates()
    wallet_graph, wallet_other_users, wallet_degree = _user_entity_graph(wallet_edges, "wallet")
    wallet_components = _component_user_sizes(wallet_graph)

    relation = crypto_transfer[
        crypto_transfer["relation_user_id"].notna() & crypto_transfer["is_internal_transfer"].eq(True)
    ][["user_id", "relation_user_id"]].copy()
    relation["relation_user_id"] = pd.to_numeric(relation["relation_user_id"], errors="coerce").astype("Int64")
    directed = relation.dropna().drop_duplicates()
    relation_graph = nx.Graph()
    relation_graph.add_nodes_from([f"user:{uid}" for uid in user_info["user_id"].dropna().tolist()])
    for _, row in directed.iterrows():
        relation_graph.add_edge(f"user:{int(row['user_id'])}", f"user:{int(row['relation_user_id'])}")
    relation_components = _component_user_sizes(relation_graph)
    degree_centrality = nx.degree_centrality(relation_graph) if relation_graph.number_of_nodes() else {}
    out_degree = directed.groupby("user_id")["relation_user_id"].nunique().reset_index(name="relation_out_degree") if not directed.empty else pd.DataFrame(columns=["user_id", "relation_out_degree"])
    in_degree = directed.groupby("relation_user_id")["user_id"].nunique().reset_index(name="relation_in_degree").rename(columns={"relation_user_id": "user_id"}) if not directed.empty else pd.DataFrame(columns=["user_id", "relation_in_degree"])
    relation_count = directed.groupby("user_id").size().reset_index(name="relation_txn_count") if not directed.empty else pd.DataFrame(columns=["user_id", "relation_txn_count"])

    result = user_info[["user_id"]].copy()
    result["shared_ip_user_count"] = result["user_id"].map(ip_other_users).fillna(0).astype(int)
    result["ip_component_size"] = result["user_id"].map(ip_components).fillna(1).astype(int)
    result["ip_entity_degree"] = result["user_id"].map(ip_degree).fillna(0).astype(int)
    result["shared_wallet_user_count"] = result["user_id"].map(wallet_other_users).fillna(0).astype(int)
    result["wallet_component_size"] = result["user_id"].map(wallet_components).fillna(1).astype(int)
    result["wallet_entity_degree"] = result["user_id"].map(wallet_degree).fillna(0).astype(int)
    result["relation_component_size"] = result["user_id"].map(relation_components).fillna(1).astype(int)
    result["relation_degree_centrality"] = result["user_id"].map(lambda uid: degree_centrality.get(f"user:{int(uid)}", 0.0)).fillna(0.0)
    result = result.merge(out_degree, on="user_id", how="left").merge(in_degree, on="user_id", how="left").merge(relation_count, on="user_id", how="left")
    result[["relation_out_degree", "relation_in_degree", "relation_txn_count"]] = result[["relation_out_degree", "relation_in_degree", "relation_txn_count"]].fillna(0).astype(int)
    denominator = result["relation_txn_count"].replace(0, 1).astype(float)
    result["relation_fan_out_ratio_graph"] = (result["relation_out_degree"].astype(float) / denominator).fillna(0.0)
    result["snapshot_cutoff_at"] = cutoff_ts or list_event_cutoffs()[1]
    result["snapshot_cutoff_tag"] = cutoff_tag
    result.to_parquet(feature_output_path("official_graph_features", cutoff_tag), index=False)
    return result


def main() -> None:
    build_official_graph_features()


if __name__ == "__main__":
    main()
