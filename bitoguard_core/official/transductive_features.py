from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scipy import sparse

from official.graph_dataset import TransductiveGraph


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


def _multi_source_distance_map(graph: TransductiveGraph, seed_users: set[int]) -> dict[int, int]:
    if not seed_users:
        return {}
    distances = {user_id: 0 for user_id in seed_users}
    queue: deque[int] = deque(sorted(seed_users))
    while queue:
        current = queue.popleft()
        next_distance = distances[current] + 1
        for neighbor, _ in graph.combined_neighbors.get(current, []):
            if neighbor in distances:
                continue
            distances[neighbor] = next_distance
            queue.append(neighbor)
    return distances


def _normalized_adjacency(graph: TransductiveGraph) -> sparse.csr_matrix:
    if graph.collapsed_edges.empty:
        return sparse.csr_matrix((len(graph.user_ids), len(graph.user_ids)), dtype=float)
    src = graph.collapsed_edges["src_user_id"].astype(int).map(graph.user_index).to_numpy()
    dst = graph.collapsed_edges["dst_user_id"].astype(int).map(graph.user_index).to_numpy()
    weight = graph.collapsed_edges["weight"].astype(float).to_numpy()
    adjacency = sparse.csr_matrix((weight, (src, dst)), shape=(len(graph.user_ids), len(graph.user_ids)), dtype=float)
    row_sum = np.asarray(adjacency.sum(axis=1)).ravel()
    row_sum[row_sum == 0.0] = 1.0
    inverse = sparse.diags(1.0 / row_sum)
    return inverse @ adjacency


def _propagation_scores(graph: TransductiveGraph, seed_users: set[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    user_count = len(graph.user_ids)
    if user_count == 0 or not seed_users:
        return np.zeros(user_count), np.zeros(user_count), np.zeros(user_count)
    adjacency = _normalized_adjacency(graph)
    seed_vector = np.zeros(user_count, dtype=float)
    for user_id in seed_users:
        if user_id in graph.user_index:
            seed_vector[graph.user_index[user_id]] = 1.0
    if seed_vector.sum() > 0:
        seed_vector = seed_vector / seed_vector.sum()
    one_hop = adjacency @ seed_vector
    two_hop = adjacency @ one_hop
    alpha = 0.20
    ppr = seed_vector.copy()
    for _ in range(20):
        ppr = alpha * seed_vector + (1.0 - alpha) * (adjacency @ ppr)
    return np.asarray(one_hop).ravel(), np.asarray(two_hop).ravel(), np.asarray(ppr).ravel()


def _edge_type_counts(graph: TransductiveGraph, positive_seed_users: set[int]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    edge_types = ["relation", "wallet_small", "wallet_medium", "ip_small", "ip_medium"]
    positive_by_type = graph.neighbors_by_type
    for user_id in graph.user_ids:
        row: dict[str, float] = {"user_id": user_id}
        total_positive = 0
        total_neighbors = 0
        for edge_type in edge_types:
            neighbors = positive_by_type.get(edge_type, {}).get(user_id, [])
            total = len(neighbors)
            positive = sum(1 for neighbor in neighbors if neighbor in positive_seed_users)
            row[f"{edge_type}_neighbor_count"] = total
            row[f"{edge_type}_positive_neighbor_count"] = positive
            row[f"{edge_type}_positive_neighbor_ratio"] = float(positive / total) if total else 0.0
            total_positive += positive
            total_neighbors += total
        row["positive_any_neighbor_count"] = total_positive
        row["positive_any_neighbor_ratio"] = float(total_positive / total_neighbors) if total_neighbors else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _entity_seed_aggregates(
    edge_frame: pd.DataFrame,
    positive_seed_users: set[int],
    labeled_seed_users: set[int],
    prefix: str,
) -> pd.DataFrame:
    if edge_frame.empty:
        return pd.DataFrame(
            columns=[
                "user_id",
                f"{prefix}_seed_positive_entity_sum",
                f"{prefix}_seed_positive_entity_max",
                f"{prefix}_seed_positive_ratio_mean",
                f"{prefix}_seed_labeled_ratio_mean",
            ]
        )
    entity_stats = edge_frame.groupby("entity_id").agg(
        entity_user_count=("user_id", "nunique"),
        positive_seed_count=("user_id", lambda users: int(sum(int(user_id) in positive_seed_users for user_id in set(users)))),
        labeled_seed_count=("user_id", lambda users: int(sum(int(user_id) in labeled_seed_users for user_id in set(users)))),
    ).reset_index()
    entity_stats["positive_seed_ratio"] = entity_stats["positive_seed_count"] / entity_stats["entity_user_count"].replace(0, np.nan)
    entity_stats["labeled_seed_ratio"] = entity_stats["labeled_seed_count"] / entity_stats["entity_user_count"].replace(0, np.nan)
    joined = edge_frame[["user_id", "entity_id"]].drop_duplicates().merge(entity_stats, on="entity_id", how="left")
    for column in ("positive_seed_count", "labeled_seed_count"):
        joined[column] = joined[column] - joined.apply(
            lambda row: 1 if int(row["user_id"]) in (positive_seed_users if column == "positive_seed_count" else labeled_seed_users) else 0,
            axis=1,
        )
    joined["positive_seed_ratio"] = joined["positive_seed_count"] / joined["entity_user_count"].replace(0, np.nan)
    joined["labeled_seed_ratio"] = joined["labeled_seed_count"] / joined["entity_user_count"].replace(0, np.nan)
    grouped = joined.groupby("user_id").agg(
        seed_positive_entity_sum=("positive_seed_count", "sum"),
        seed_positive_entity_max=("positive_seed_count", "max"),
        seed_positive_ratio_mean=("positive_seed_ratio", "mean"),
        seed_labeled_ratio_mean=("labeled_seed_ratio", "mean"),
    ).reset_index()
    grouped = grouped.rename(
        columns={
            "seed_positive_entity_sum": f"{prefix}_seed_positive_entity_sum",
            "seed_positive_entity_max": f"{prefix}_seed_positive_entity_max",
            "seed_positive_ratio_mean": f"{prefix}_seed_positive_ratio_mean",
            "seed_labeled_ratio_mean": f"{prefix}_seed_labeled_ratio_mean",
        }
    )
    return grouped


def _component_seed_stats(graph: TransductiveGraph, train_label_frame: pd.DataFrame) -> pd.DataFrame:
    frame = train_label_frame.copy()
    frame["user_id"] = pd.to_numeric(frame["user_id"], errors="coerce").astype("Int64")
    frame["status"] = pd.to_numeric(frame["status"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["user_id", "status"])
    frame["user_id"] = frame["user_id"].astype(int)
    frame["component_id"] = frame["user_id"].map(graph.component_id_by_user)
    component_stats = frame.groupby("component_id").agg(
        component_train_positive_count=("status", lambda values: int((pd.Series(values).astype(int) == 1).sum())),
        component_train_labeled_count=("status", "size"),
    ).reset_index()
    component_stats["component_train_positive_rate"] = (
        component_stats["component_train_positive_count"] / component_stats["component_train_labeled_count"].replace(0, np.nan)
    ).fillna(0.0)
    user_component = pd.DataFrame(
        {
            "user_id": graph.user_ids,
            "component_id": [graph.component_id_by_user[user_id] for user_id in graph.user_ids],
        }
    )
    output = user_component.merge(component_stats, on="component_id", how="left")
    output[[
        "component_train_positive_count",
        "component_train_labeled_count",
        "component_train_positive_rate",
    ]] = output[[
        "component_train_positive_count",
        "component_train_labeled_count",
        "component_train_positive_rate",
    ]].fillna(0.0)
    output["component_has_positive_seed"] = output["component_train_positive_count"].gt(0).astype(int)
    return output.drop(columns=["component_id"])


def build_transductive_feature_frame(
    graph: TransductiveGraph,
    train_label_frame: pd.DataFrame,
) -> pd.DataFrame:
    positive_seed_users = _positive_seed_users(train_label_frame)
    labeled_seed_users = _labeled_seed_users(train_label_frame)

    relation_seed = graph.relation_edges.copy()
    relation_seed["relation_positive_seed"] = relation_seed["relation_user_id"].astype(int).isin(positive_seed_users).astype(int)
    relation_stats = relation_seed.groupby("user_id").agg(
        relation_positive_seed_count=("relation_positive_seed", "sum"),
        relation_total_seed_neighbors=("relation_user_id", "size"),
    ).reset_index()
    relation_stats["relation_positive_seed_ratio"] = (
        relation_stats["relation_positive_seed_count"] / relation_stats["relation_total_seed_neighbors"].replace(0, np.nan)
    ).fillna(0.0)

    edge_type_stats = _edge_type_counts(graph, positive_seed_users)
    wallet_seed_stats = _entity_seed_aggregates(graph.wallet_edges, positive_seed_users, labeled_seed_users, "wallet")
    ip_seed_stats = _entity_seed_aggregates(graph.ip_edges, positive_seed_users, labeled_seed_users, "ip")
    component_stats = _component_seed_stats(graph, train_label_frame)

    one_hop, two_hop, ppr = _propagation_scores(graph, positive_seed_users)
    distances = _multi_source_distance_map(graph, positive_seed_users)
    propagation = pd.DataFrame(
        {
            "user_id": graph.user_ids,
            "positive_seed_weight_1hop": one_hop,
            "positive_seed_weight_2hop": two_hop,
            "positive_seed_ppr": ppr,
            "nearest_positive_seed_distance": [distances.get(user_id, -1) for user_id in graph.user_ids],
        }
    )
    propagation["harmonic_positive_seed_distance"] = propagation["nearest_positive_seed_distance"].apply(
        lambda distance: 0.0 if distance < 0 else 1.0 / (1.0 + float(distance))
    )
    propagation["has_positive_seed_path"] = propagation["nearest_positive_seed_distance"].ge(0).astype(int)

    result = pd.DataFrame({"user_id": graph.user_ids})
    for frame in (relation_stats, edge_type_stats, wallet_seed_stats, ip_seed_stats, component_stats, propagation):
        result = result.merge(frame, on="user_id", how="left")

    numeric_columns = [column for column in result.columns if column != "user_id"]
    result[numeric_columns] = result[numeric_columns].fillna(0.0)
    return result
