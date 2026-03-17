from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from math import log1p
from typing import Any

import pandas as pd

from official.common import EVENT_TIME_COLUMNS, load_clean_table, to_utc_timestamp


MAX_WALLET_PAIRWISE_USERS = 50
MAX_IP_PAIRWISE_USERS = 20


@dataclass(frozen=True)
class TransductiveGraph:
    user_ids: list[int]
    user_feature_frame: pd.DataFrame
    user_index: dict[int, int]
    relation_edges: pd.DataFrame
    wallet_edges: pd.DataFrame
    ip_edges: pd.DataFrame
    collapsed_edges: pd.DataFrame
    component_id_by_user: dict[int, int]
    combined_neighbors: dict[int, list[tuple[int, float]]]
    neighbors_by_type: dict[str, dict[int, list[int]]]
    wallet_node_frame: pd.DataFrame
    ip_node_frame: pd.DataFrame


def _prepare_table(name: str, cutoff_ts: pd.Timestamp | None) -> pd.DataFrame:
    frame = load_clean_table(name).copy()
    if "user_id" in frame.columns:
        frame["user_id"] = pd.to_numeric(frame["user_id"], errors="coerce").astype("Int64")
    time_column = EVENT_TIME_COLUMNS.get(name)
    if time_column:
        frame[time_column] = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
        if cutoff_ts is not None:
            frame = frame[frame[time_column] < cutoff_ts].copy()
    return frame


def _pairwise_user_edges(
    edge_frame: pd.DataFrame,
    max_users: int,
    edge_type_small: str,
    edge_type_medium: str,
    small_upper_bound: int,
    weight_small: float,
    weight_medium: float,
) -> pd.DataFrame:
    if edge_frame.empty:
        return pd.DataFrame(columns=["src_user_id", "dst_user_id", "edge_type", "weight"])
    rows: list[dict[str, Any]] = []
    for _, group in edge_frame.groupby("entity_id"):
        users = sorted({int(user_id) for user_id in group["user_id"].dropna().tolist()})
        user_count = len(users)
        if user_count <= 1 or user_count > max_users:
            continue
        edge_type = edge_type_small if user_count <= small_upper_bound else edge_type_medium
        weight = weight_small if user_count <= small_upper_bound else weight_medium
        for left, right in combinations(users, 2):
            rows.append({"src_user_id": left, "dst_user_id": right, "edge_type": edge_type, "weight": weight})
            rows.append({"src_user_id": right, "dst_user_id": left, "edge_type": edge_type, "weight": weight})
    return pd.DataFrame(rows)


def _relation_user_edges(relation_edges: pd.DataFrame) -> pd.DataFrame:
    if relation_edges.empty:
        return pd.DataFrame(columns=["src_user_id", "dst_user_id", "edge_type", "weight"])
    rows: list[dict[str, Any]] = []
    for _, row in relation_edges.iterrows():
        left = int(row["user_id"])
        right = int(row["relation_user_id"])
        if left == right:
            continue
        rows.append({"src_user_id": left, "dst_user_id": right, "edge_type": "relation", "weight": 1.0})
        rows.append({"src_user_id": right, "dst_user_id": left, "edge_type": "relation", "weight": 1.0})
    return pd.DataFrame(rows).drop_duplicates()


def _entity_node_frame(edge_frame: pd.DataFrame, prefix: str, hub_cutoff: int) -> pd.DataFrame:
    if edge_frame.empty:
        return pd.DataFrame(columns=["entity_id", f"{prefix}_user_count", f"{prefix}_link_count", f"{prefix}_is_hub", f"{prefix}_log_user_count"])
    counts = edge_frame.groupby("entity_id").agg(
        user_count=("user_id", "nunique"),
        link_count=("user_id", "size"),
    ).reset_index()
    counts[f"{prefix}_user_count"] = counts["user_count"].astype(int)
    counts[f"{prefix}_link_count"] = counts["link_count"].astype(int)
    counts[f"{prefix}_is_hub"] = counts[f"{prefix}_user_count"].gt(hub_cutoff).astype(int)
    counts[f"{prefix}_log_user_count"] = counts[f"{prefix}_user_count"].map(lambda value: float(log1p(value)))
    return counts[["entity_id", f"{prefix}_user_count", f"{prefix}_link_count", f"{prefix}_is_hub", f"{prefix}_log_user_count"]]


def _component_id_map(user_ids: list[int], edges: pd.DataFrame) -> dict[int, int]:
    parent = {user_id: user_id for user_id in user_ids}

    def find(value: int) -> int:
        root = parent.setdefault(value, value)
        if root != value:
            parent[value] = find(root)
        return parent[value]

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for _, row in edges.iterrows():
        union(int(row["src_user_id"]), int(row["dst_user_id"]))
    roots = {user_id: find(user_id) for user_id in user_ids}
    root_to_component: dict[int, int] = {}
    component_map: dict[int, int] = {}
    for user_id, root in roots.items():
        if root not in root_to_component:
            root_to_component[root] = len(root_to_component) + 1
        component_map[user_id] = root_to_component[root]
    return component_map


def _neighbor_maps(collapsed_edges: pd.DataFrame) -> tuple[dict[int, list[tuple[int, float]]], dict[str, dict[int, list[int]]]]:
    combined: dict[int, list[tuple[int, float]]] = defaultdict(list)
    by_type: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    if collapsed_edges.empty:
        return {}, {}
    for _, row in collapsed_edges.iterrows():
        src = int(row["src_user_id"])
        dst = int(row["dst_user_id"])
        weight = float(row["weight"])
        edge_type = str(row["edge_type"])
        combined[src].append((dst, weight))
        by_type[edge_type][src].append(dst)
    return dict(combined), {edge_type: dict(mapping) for edge_type, mapping in by_type.items()}


def _numeric_user_feature_frame(dataset: pd.DataFrame) -> pd.DataFrame:
    excluded = {
        "user_id",
        "status",
        "cohort",
        "snapshot_cutoff_at",
        "snapshot_cutoff_tag",
        "top_reason_codes",
        "is_known_blacklist",
        "needs_prediction",
        "in_train_label",
        "in_predict_label",
        "is_shadow_overlap",
    }
    frame = dataset.copy()
    bool_columns = [column for column in frame.columns if pd.api.types.is_bool_dtype(frame[column])]
    for column in bool_columns:
        frame[column] = frame[column].fillna(False).astype(int)
    numeric_columns = [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    output = frame[["user_id", *numeric_columns]].copy()
    output[numeric_columns] = output[numeric_columns].fillna(0.0)
    return output


def build_transductive_graph(
    dataset: pd.DataFrame,
    cutoff_ts: pd.Timestamp | None = None,
) -> TransductiveGraph:
    cutoff_ts = to_utc_timestamp(cutoff_ts)
    user_feature_frame = _numeric_user_feature_frame(dataset)
    user_ids = sorted(user_feature_frame["user_id"].astype(int).tolist())
    allowed_users = set(user_ids)
    user_index = {user_id: idx for idx, user_id in enumerate(user_ids)}

    twd = _prepare_table("twd_transfer", cutoff_ts)
    crypto = _prepare_table("crypto_transfer", cutoff_ts)
    trade = _prepare_table("usdt_twd_trading", cutoff_ts)

    relation_edges = crypto[
        crypto["relation_user_id"].notna() & crypto["is_internal_transfer"].eq(True)
    ][["user_id", "relation_user_id"]].copy()
    relation_edges["relation_user_id"] = pd.to_numeric(relation_edges["relation_user_id"], errors="coerce").astype("Int64")
    relation_edges = relation_edges.dropna()
    relation_edges["user_id"] = relation_edges["user_id"].astype(int)
    relation_edges["relation_user_id"] = relation_edges["relation_user_id"].astype(int)
    relation_edges = relation_edges[
        relation_edges["user_id"].isin(allowed_users)
        & relation_edges["relation_user_id"].isin(allowed_users)
    ].drop_duplicates()

    wallet_edges = pd.concat(
        [
            crypto[["user_id", "from_wallet_hash"]].rename(columns={"from_wallet_hash": "entity_id"}),
            crypto[["user_id", "to_wallet_hash"]].rename(columns={"to_wallet_hash": "entity_id"}),
        ],
        ignore_index=True,
    )
    wallet_edges = wallet_edges[wallet_edges["entity_id"].notna()].copy()
    wallet_edges["user_id"] = pd.to_numeric(wallet_edges["user_id"], errors="coerce").astype("Int64")
    wallet_edges = wallet_edges.dropna(subset=["user_id"])
    wallet_edges["user_id"] = wallet_edges["user_id"].astype(int)
    wallet_edges = wallet_edges[wallet_edges["user_id"].isin(allowed_users)].drop_duplicates()
    wallet_counts = wallet_edges.groupby("entity_id")["user_id"].nunique().rename("entity_user_count").reset_index()
    wallet_edges = wallet_edges.merge(wallet_counts, on="entity_id", how="left")

    ip_edges = pd.concat(
        [
            twd[["user_id", "source_ip_hash"]].rename(columns={"source_ip_hash": "entity_id"}),
            crypto[["user_id", "source_ip_hash"]].rename(columns={"source_ip_hash": "entity_id"}),
            trade[["user_id", "source_ip_hash"]].rename(columns={"source_ip_hash": "entity_id"}),
        ],
        ignore_index=True,
    )
    ip_edges = ip_edges[ip_edges["entity_id"].notna()].copy()
    ip_edges["user_id"] = pd.to_numeric(ip_edges["user_id"], errors="coerce").astype("Int64")
    ip_edges = ip_edges.dropna(subset=["user_id"])
    ip_edges["user_id"] = ip_edges["user_id"].astype(int)
    ip_edges = ip_edges[ip_edges["user_id"].isin(allowed_users)].drop_duplicates()
    ip_counts = ip_edges.groupby("entity_id")["user_id"].nunique().rename("entity_user_count").reset_index()
    ip_edges = ip_edges.merge(ip_counts, on="entity_id", how="left")

    relation_user_edges = _relation_user_edges(relation_edges)
    wallet_user_edges = _pairwise_user_edges(
        wallet_edges[wallet_edges["entity_user_count"] <= MAX_WALLET_PAIRWISE_USERS],
        max_users=MAX_WALLET_PAIRWISE_USERS,
        edge_type_small="wallet_small",
        edge_type_medium="wallet_medium",
        small_upper_bound=10,
        weight_small=0.70,
        weight_medium=0.40,
    )
    ip_user_edges = _pairwise_user_edges(
        ip_edges[ip_edges["entity_user_count"] <= MAX_IP_PAIRWISE_USERS],
        max_users=MAX_IP_PAIRWISE_USERS,
        edge_type_small="ip_small",
        edge_type_medium="ip_medium",
        small_upper_bound=5,
        weight_small=0.50,
        weight_medium=0.25,
    )
    collapsed_edges = pd.concat([relation_user_edges, wallet_user_edges, ip_user_edges], ignore_index=True)
    if collapsed_edges.empty:
        collapsed_edges = pd.DataFrame(columns=["src_user_id", "dst_user_id", "edge_type", "weight"])
    else:
        collapsed_edges = (
            collapsed_edges.groupby(["src_user_id", "dst_user_id", "edge_type"], as_index=False)["weight"]
            .sum()
            .sort_values(["src_user_id", "dst_user_id", "edge_type"])
            .reset_index(drop=True)
        )

    component_id_by_user = _component_id_map(user_ids, collapsed_edges[["src_user_id", "dst_user_id"]].drop_duplicates())
    combined_neighbors, neighbors_by_type = _neighbor_maps(collapsed_edges)
    wallet_node_frame = _entity_node_frame(wallet_edges, "wallet", MAX_WALLET_PAIRWISE_USERS)
    ip_node_frame = _entity_node_frame(ip_edges, "ip", MAX_IP_PAIRWISE_USERS)

    return TransductiveGraph(
        user_ids=user_ids,
        user_feature_frame=user_feature_frame,
        user_index=user_index,
        relation_edges=relation_edges.reset_index(drop=True),
        wallet_edges=wallet_edges.reset_index(drop=True),
        ip_edges=ip_edges.reset_index(drop=True),
        collapsed_edges=collapsed_edges,
        component_id_by_user=component_id_by_user,
        combined_neighbors=combined_neighbors,
        neighbors_by_type=neighbors_by_type,
        wallet_node_frame=wallet_node_frame,
        ip_node_frame=ip_node_frame,
    )
