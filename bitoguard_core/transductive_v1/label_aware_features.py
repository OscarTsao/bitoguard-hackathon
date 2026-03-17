from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from transductive_v1.graph_store import GraphStore


def _positive_seed_users(train_label_frame: pd.DataFrame) -> set[int]:
    frame = train_label_frame.copy()
    frame["user_id"] = pd.to_numeric(frame["user_id"], errors="coerce").astype("Int64")
    frame["status"] = pd.to_numeric(frame["status"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["user_id", "status"])
    return set(frame[frame["status"].astype(int) == 1]["user_id"].astype(int).tolist())


def _labeled_seed_users(train_label_frame: pd.DataFrame) -> set[int]:
    frame = train_label_frame.copy()
    frame["user_id"] = pd.to_numeric(frame["user_id"], errors="coerce").astype("Int64")
    frame["status"] = pd.to_numeric(frame["status"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["user_id", "status"])
    return set(frame["user_id"].astype(int).tolist())


def _entity_reputation_features(edge_frame: pd.DataFrame, positive_seed_users: set[int], labeled_seed_users: set[int], prefix: str) -> pd.DataFrame:
    if edge_frame.empty:
        return pd.DataFrame(columns=[
            "user_id",
            f"{prefix}_positive_entity_sum",
            f"{prefix}_positive_entity_max",
            f"{prefix}_positive_rate_mean",
            f"{prefix}_labeled_entity_sum",
        ])
    entity_stats = edge_frame.groupby("entity_id").agg(
        entity_user_count=("user_id", "nunique"),
        positive_seed_count=("user_id", lambda users: int(sum(int(user_id) in positive_seed_users for user_id in set(users)))),
        labeled_seed_count=("user_id", lambda users: int(sum(int(user_id) in labeled_seed_users for user_id in set(users)))),
    ).reset_index()
    joined = edge_frame[["user_id", "entity_id"]].drop_duplicates().merge(entity_stats, on="entity_id", how="left")
    joined["positive_rate"] = joined["positive_seed_count"] / joined["labeled_seed_count"].replace(0, np.nan)
    grouped = joined.groupby("user_id").agg(
        positive_entity_sum=("positive_seed_count", "sum"),
        positive_entity_max=("positive_seed_count", "max"),
        positive_rate_mean=("positive_rate", "mean"),
        labeled_entity_sum=("labeled_seed_count", "sum"),
    ).reset_index()
    return grouped.rename(
        columns={
            "positive_entity_sum": f"{prefix}_positive_entity_sum",
            "positive_entity_max": f"{prefix}_positive_entity_max",
            "positive_rate_mean": f"{prefix}_positive_rate_mean",
            "labeled_entity_sum": f"{prefix}_labeled_entity_sum",
        }
    )


def _relation_positive_counts(graph_store: GraphStore, positive_seed_users: set[int]) -> pd.DataFrame:
    relation = graph_store.relation_edges.copy()
    if relation.empty:
        return pd.DataFrame(columns=["user_id", "direct_positive_relation_count", "direct_positive_relation_ratio"])
    relation["positive_neighbor"] = relation["relation_user_id"].astype(int).isin(positive_seed_users).astype(int)
    grouped = relation.groupby("user_id").agg(
        direct_positive_relation_count=("positive_neighbor", "sum"),
        direct_relation_total=("relation_user_id", "size"),
    ).reset_index()
    grouped["direct_positive_relation_ratio"] = grouped["direct_positive_relation_count"] / grouped["direct_relation_total"].replace(0, np.nan)
    return grouped.drop(columns=["direct_relation_total"])


def _hop_features(graph_store: GraphStore, positive_seed_users: set[int]) -> pd.DataFrame:
    one_hop_count: dict[int, int] = {}
    two_hop_count: dict[int, int] = {}
    for user_id in graph_store.user_ids:
        first_hop = set(graph_store.neighbors.get(user_id, []))
        one_hop_count[user_id] = sum(neighbor in positive_seed_users for neighbor in first_hop)
        second_hop = set()
        for neighbor in first_hop:
            second_hop.update(graph_store.neighbors.get(neighbor, []))
        second_hop.discard(user_id)
        two_hop_count[user_id] = sum(neighbor in positive_seed_users for neighbor in second_hop)
    return pd.DataFrame(
        {
            "user_id": graph_store.user_ids,
            "positive_exposure_1hop_count": [one_hop_count[user_id] for user_id in graph_store.user_ids],
            "positive_exposure_2hop_count": [two_hop_count[user_id] for user_id in graph_store.user_ids],
        }
    )


def _distance_features(graph_store: GraphStore, positive_seed_users: set[int]) -> pd.DataFrame:
    if not positive_seed_users:
        return pd.DataFrame({"user_id": graph_store.user_ids, "nearest_positive_seed_distance": -1, "positive_seed_harmonic_distance": 0.0})
    distances = {user_id: 0 for user_id in positive_seed_users}
    queue: deque[int] = deque(sorted(positive_seed_users))
    while queue:
        current = queue.popleft()
        next_distance = distances[current] + 1
        for neighbor in graph_store.neighbors.get(current, []):
            if neighbor in distances:
                continue
            distances[neighbor] = next_distance
            queue.append(neighbor)
    return pd.DataFrame(
        {
            "user_id": graph_store.user_ids,
            "nearest_positive_seed_distance": [distances.get(user_id, -1) for user_id in graph_store.user_ids],
            "positive_seed_harmonic_distance": [
                0.0 if distances.get(user_id, -1) < 0 else 1.0 / (1.0 + float(distances[user_id]))
                for user_id in graph_store.user_ids
            ],
        }
    )


def _propagation_features(graph_store: GraphStore, positive_seed_users: set[int]) -> pd.DataFrame:
    user_count = len(graph_store.user_ids)
    if user_count == 0 or not positive_seed_users:
        return pd.DataFrame({"user_id": graph_store.user_ids, "positive_seed_propagation": 0.0, "positive_seed_walk2": 0.0})
    adjacency = np.zeros((user_count, user_count), dtype=float)
    for _, row in graph_store.projected_edges.iterrows():
        src = graph_store.user_index[int(row["src_user_id"])]
        dst = graph_store.user_index[int(row["dst_user_id"])]
        adjacency[src, dst] += float(row["weight"])
    row_sum = adjacency.sum(axis=1)
    row_sum[row_sum == 0.0] = 1.0
    normalized = adjacency / row_sum[:, None]
    seed_vector = np.zeros(user_count, dtype=float)
    for user_id in positive_seed_users:
        if user_id in graph_store.user_index:
            seed_vector[graph_store.user_index[user_id]] = 1.0
    seed_vector = seed_vector / max(seed_vector.sum(), 1.0)
    walk1 = normalized @ seed_vector
    walk2 = normalized @ walk1
    alpha = 0.20
    ppr = seed_vector.copy()
    for _ in range(6):
        ppr = alpha * seed_vector + (1.0 - alpha) * (normalized @ ppr)
    return pd.DataFrame(
        {
            "user_id": graph_store.user_ids,
            "positive_seed_propagation": ppr,
            "positive_seed_walk2": walk2,
        }
    )


def build_label_aware_features(graph_store: GraphStore, train_label_frame: pd.DataFrame) -> pd.DataFrame:
    positive_seed_users = _positive_seed_users(train_label_frame)
    labeled_seed_users = _labeled_seed_users(train_label_frame)
    relation_features = _relation_positive_counts(graph_store, positive_seed_users)
    wallet_features = _entity_reputation_features(graph_store.wallet_edges, positive_seed_users, labeled_seed_users, "wallet")
    ip_features = _entity_reputation_features(graph_store.ip_edges, positive_seed_users, labeled_seed_users, "ip")
    hop_features = _hop_features(graph_store, positive_seed_users)
    distance_features = _distance_features(graph_store, positive_seed_users)
    propagation_features = _propagation_features(graph_store, positive_seed_users)

    result = pd.DataFrame({"user_id": graph_store.user_ids})
    for frame in (relation_features, wallet_features, ip_features, hop_features, distance_features, propagation_features):
        result = result.merge(frame, on="user_id", how="left")
    numeric_columns = [column for column in result.columns if column != "user_id"]
    result[numeric_columns] = result[numeric_columns].fillna(0.0)
    return result
