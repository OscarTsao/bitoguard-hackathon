from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from transductive_v1.common import EVENT_TIME_COLUMNS, feature_path, load_clean_table, to_utc_timestamp


IP_PROJECTION_MAX_DEGREE = 20
WALLET_PROJECTION_MAX_DEGREE = 50
IP_BUCKETS = ((2, 3, "ip_entities_deg_2_3"), (4, 10, "ip_entities_deg_4_10"), (11, 20, "ip_entities_deg_11_20"), (21, 999999, "ip_entities_deg_21_plus"))
WALLET_BUCKETS = ((2, 3, "wallet_entities_deg_2_3"), (4, 10, "wallet_entities_deg_4_10"), (11, 20, "wallet_entities_deg_11_20"), (21, 50, "wallet_entities_deg_21_50"), (51, 999999, "wallet_entities_deg_51_plus"))


@dataclass(frozen=True)
class GraphStore:
    user_ids: list[int]
    user_index: dict[int, int]
    relation_edges: pd.DataFrame
    wallet_edges: pd.DataFrame
    ip_edges: pd.DataFrame
    projected_edges: pd.DataFrame
    neighbors: dict[int, list[int]]
    weighted_neighbors: dict[int, list[tuple[int, float]]]
    structural_features: pd.DataFrame


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


def _project_entities(entity_frame: pd.DataFrame, max_degree: int, edge_type: str, weight: float) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    if entity_frame.empty:
        return pd.DataFrame(columns=["src_user_id", "dst_user_id", "edge_type", "weight"])
    for _, group in entity_frame.groupby("entity_id"):
        users = sorted({int(user_id) for user_id in group["user_id"].dropna().tolist()})
        if len(users) <= 1 or len(users) > max_degree:
            continue
        for left, right in combinations(users, 2):
            rows.append({"src_user_id": left, "dst_user_id": right, "edge_type": edge_type, "weight": weight})
            rows.append({"src_user_id": right, "dst_user_id": left, "edge_type": edge_type, "weight": weight})
    return pd.DataFrame(rows)


def _relation_projection(relation_edges: pd.DataFrame) -> pd.DataFrame:
    if relation_edges.empty:
        return pd.DataFrame(columns=["src_user_id", "dst_user_id", "edge_type", "weight"])
    rows = []
    for _, row in relation_edges.iterrows():
        left = int(row["user_id"])
        right = int(row["relation_user_id"])
        if left == right:
            continue
        rows.append({"src_user_id": left, "dst_user_id": right, "edge_type": "relation", "weight": 1.0})
        rows.append({"src_user_id": right, "dst_user_id": left, "edge_type": "relation", "weight": 1.0})
    return pd.DataFrame(rows).drop_duplicates()


def _neighbor_maps(projected_edges: pd.DataFrame) -> tuple[dict[int, list[int]], dict[int, list[tuple[int, float]]]]:
    neighbors: dict[int, list[int]] = defaultdict(list)
    weighted: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for _, row in projected_edges.iterrows():
        src = int(row["src_user_id"])
        dst = int(row["dst_user_id"])
        weight = float(row["weight"])
        neighbors[src].append(dst)
        weighted[src].append((dst, weight))
    return dict(neighbors), dict(weighted)


def _component_sizes(user_ids: list[int], neighbors: dict[int, list[int]]) -> dict[int, int]:
    remaining = set(user_ids)
    component_sizes: dict[int, int] = {}
    while remaining:
        start = remaining.pop()
        queue = deque([start])
        component = {start}
        while queue:
            current = queue.popleft()
            for neighbor in neighbors.get(current, []):
                if neighbor in component:
                    continue
                component.add(neighbor)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                queue.append(neighbor)
        size = len(component)
        for user_id in component:
            component_sizes[user_id] = size
    return component_sizes


def _entity_bucket_features(entity_frame: pd.DataFrame, buckets: tuple[tuple[int, int, str], ...], prefix: str) -> pd.DataFrame:
    if entity_frame.empty:
        columns = ["user_id", f"{prefix}_entity_count", f"{prefix}_max_entity_degree", f"shared_{prefix}_user_count"]
        columns.extend(name for _, _, name in buckets)
        return pd.DataFrame(columns=columns)
    counts = entity_frame.groupby("entity_id")["user_id"].nunique().rename("entity_user_count").reset_index()
    enriched = entity_frame.merge(counts, on="entity_id", how="left")[["user_id", "entity_id", "entity_user_count"]].drop_duplicates()
    rows = []
    for user_id, group in enriched.groupby("user_id"):
        row = {
            "user_id": int(user_id),
            f"{prefix}_entity_count": int(group["entity_id"].nunique()),
            f"{prefix}_max_entity_degree": int(group["entity_user_count"].max()),
            f"shared_{prefix}_user_count": int(group["entity_user_count"].sub(1).clip(lower=0).sum()),
        }
        for lower, upper, name in buckets:
            row[name] = int(group["entity_user_count"].between(lower, upper).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def build_graph_store(
    user_ids: list[int],
    cutoff_ts: pd.Timestamp | None = None,
    cutoff_tag: str = "full",
    write_outputs: bool = True,
) -> GraphStore:
    cutoff_ts = to_utc_timestamp(cutoff_ts)
    allowed_users = set(int(user_id) for user_id in user_ids)
    user_index = {int(user_id): idx for idx, user_id in enumerate(sorted(allowed_users))}

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
        relation_edges["user_id"].isin(allowed_users) & relation_edges["relation_user_id"].isin(allowed_users)
    ].drop_duplicates()

    wallet_edges = pd.concat(
        [
            crypto[["user_id", "from_wallet_hash"]].rename(columns={"from_wallet_hash": "entity_id"}),
            crypto[["user_id", "to_wallet_hash"]].rename(columns={"to_wallet_hash": "entity_id"}),
        ],
        ignore_index=True,
    )
    wallet_edges["user_id"] = pd.to_numeric(wallet_edges["user_id"], errors="coerce").astype("Int64")
    wallet_edges = wallet_edges[wallet_edges["entity_id"].notna() & wallet_edges["user_id"].notna()].copy()
    wallet_edges["user_id"] = wallet_edges["user_id"].astype(int)
    wallet_edges = wallet_edges[wallet_edges["user_id"].isin(allowed_users)].drop_duplicates()

    ip_edges = pd.concat(
        [
            twd[["user_id", "source_ip_hash"]].rename(columns={"source_ip_hash": "entity_id"}),
            crypto[["user_id", "source_ip_hash"]].rename(columns={"source_ip_hash": "entity_id"}),
            trade[["user_id", "source_ip_hash"]].rename(columns={"source_ip_hash": "entity_id"}),
        ],
        ignore_index=True,
    )
    ip_edges["user_id"] = pd.to_numeric(ip_edges["user_id"], errors="coerce").astype("Int64")
    ip_edges = ip_edges[ip_edges["entity_id"].notna() & ip_edges["user_id"].notna()].copy()
    ip_edges["user_id"] = ip_edges["user_id"].astype(int)
    ip_edges = ip_edges[ip_edges["user_id"].isin(allowed_users)].drop_duplicates()

    projected_edges = pd.concat(
        [
            _relation_projection(relation_edges),
            _project_entities(wallet_edges, WALLET_PROJECTION_MAX_DEGREE, "wallet", 0.4),
            _project_entities(ip_edges, IP_PROJECTION_MAX_DEGREE, "ip", 0.25),
        ],
        ignore_index=True,
    )
    if projected_edges.empty:
        projected_edges = pd.DataFrame(columns=["src_user_id", "dst_user_id", "edge_type", "weight"])
    else:
        projected_edges = projected_edges.groupby(["src_user_id", "dst_user_id", "edge_type"], as_index=False)["weight"].sum()
    neighbors, weighted_neighbors = _neighbor_maps(projected_edges)
    component_sizes = _component_sizes(sorted(allowed_users), neighbors)

    relation_out = relation_edges.groupby("user_id").size().reset_index(name="relation_out_degree") if not relation_edges.empty else pd.DataFrame(columns=["user_id", "relation_out_degree"])
    relation_in = relation_edges.groupby("relation_user_id").size().reset_index(name="relation_in_degree").rename(columns={"relation_user_id": "user_id"}) if not relation_edges.empty else pd.DataFrame(columns=["user_id", "relation_in_degree"])
    wallet_features = _entity_bucket_features(wallet_edges, WALLET_BUCKETS, "wallet")
    ip_features = _entity_bucket_features(ip_edges, IP_BUCKETS, "ip")

    structural = pd.DataFrame({"user_id": sorted(allowed_users)})
    structural = structural.merge(relation_out, on="user_id", how="left")
    structural = structural.merge(relation_in, on="user_id", how="left")
    structural = structural.merge(wallet_features, on="user_id", how="left")
    structural = structural.merge(ip_features, on="user_id", how="left")
    structural["projected_degree"] = structural["user_id"].map(lambda user_id: len(neighbors.get(int(user_id), []))).fillna(0).astype(int)
    structural["projected_component_size"] = structural["user_id"].map(lambda user_id: component_sizes.get(int(user_id), 1)).fillna(1).astype(int)
    structural["projected_component_log_size"] = np.log1p(structural["projected_component_size"])
    structural["connected_flag"] = structural["projected_degree"].gt(0).astype(int)
    structural = structural.fillna(0.0)

    if write_outputs:
        structural.to_parquet(feature_path("graph_structural_features", cutoff_tag), index=False)
        relation_edges.to_parquet(feature_path("graph_relation_edges", cutoff_tag), index=False)
        wallet_edges.to_parquet(feature_path("graph_wallet_edges", cutoff_tag), index=False)
        ip_edges.to_parquet(feature_path("graph_ip_edges", cutoff_tag), index=False)
        projected_edges.to_parquet(feature_path("graph_projected_edges", cutoff_tag), index=False)

    return GraphStore(
        user_ids=sorted(allowed_users),
        user_index=user_index,
        relation_edges=relation_edges.reset_index(drop=True),
        wallet_edges=wallet_edges.reset_index(drop=True),
        ip_edges=ip_edges.reset_index(drop=True),
        projected_edges=projected_edges.reset_index(drop=True),
        neighbors=neighbors,
        weighted_neighbors=weighted_neighbors,
        structural_features=structural,
    )


def load_graph_store(cutoff_tag: str = "full") -> GraphStore:
    structural = pd.read_parquet(feature_path("graph_structural_features", cutoff_tag))
    relation_edges = pd.read_parquet(feature_path("graph_relation_edges", cutoff_tag))
    wallet_edges = pd.read_parquet(feature_path("graph_wallet_edges", cutoff_tag))
    ip_edges = pd.read_parquet(feature_path("graph_ip_edges", cutoff_tag))
    projected_edges = pd.read_parquet(feature_path("graph_projected_edges", cutoff_tag))
    user_ids = sorted(structural["user_id"].astype(int).tolist())
    user_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    neighbors, weighted_neighbors = _neighbor_maps(projected_edges)
    return GraphStore(
        user_ids=user_ids,
        user_index=user_index,
        relation_edges=relation_edges,
        wallet_edges=wallet_edges,
        ip_edges=ip_edges,
        projected_edges=projected_edges,
        neighbors=neighbors,
        weighted_neighbors=weighted_neighbors,
        structural_features=structural,
    )
