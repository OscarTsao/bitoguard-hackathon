from __future__ import annotations

from collections import defaultdict

import pandas as pd

from official.common import feature_output_path, list_event_cutoffs, load_clean_table, to_utc_timestamp


MAX_IP_ENTITY_USERS = 200
MAX_WALLET_ENTITY_USERS = 200


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, node: int) -> int:
        parent = self.parent
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _prepare_entity_edges(frame: pd.DataFrame, entity_column: str) -> pd.DataFrame:
    edges = frame[["user_id", entity_column]].rename(columns={entity_column: "entity_id"}).copy()
    edges["user_id"] = pd.to_numeric(edges["user_id"], errors="coerce").astype("Int64")
    return edges[edges["user_id"].notna() & edges["entity_id"].notna()].drop_duplicates().reset_index(drop=True)


def _component_sizes_from_bipartite(
    user_ids: list[int],
    edges: pd.DataFrame,
) -> dict[int, int]:
    if edges.empty:
        return {}
    user_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    edges = edges[edges["user_id"].astype(int).isin(user_index)].copy()
    if edges.empty:
        return {}
    entity_codes, _ = pd.factorize(edges["entity_id"], sort=False)
    user_codes = edges["user_id"].astype(int).map(user_index).to_numpy()
    user_count = len(user_ids)
    union_find = UnionFind(user_count + len(entity_codes))
    for user_code, entity_code in zip(user_codes, entity_codes, strict=False):
        union_find.union(int(user_code), user_count + int(entity_code))
    root_user_counts: dict[int, int] = defaultdict(int)
    for user_code in range(user_count):
        root_user_counts[union_find.find(user_code)] += 1
    return {
        user_id: root_user_counts[union_find.find(user_index[user_id])]
        for user_id in user_ids
    }


def _component_sizes_from_user_pairs(user_ids: list[int], pairs: pd.DataFrame) -> dict[int, int]:
    if not user_ids:
        return {}
    user_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    union_find = UnionFind(len(user_ids))
    if not pairs.empty:
        pairs = pairs[
            pairs["user_id"].astype(int).isin(user_index)
            & pairs["relation_user_id"].astype(int).isin(user_index)
        ].copy()
        left_codes = pairs["user_id"].astype(int).map(user_index).to_numpy()
        right_codes = pairs["relation_user_id"].astype(int).map(user_index).to_numpy()
        for left_code, right_code in zip(left_codes, right_codes, strict=False):
            union_find.union(int(left_code), int(right_code))
    root_user_counts: dict[int, int] = defaultdict(int)
    for user_code in range(len(user_ids)):
        root_user_counts[union_find.find(user_code)] += 1
    return {
        user_id: root_user_counts[union_find.find(user_index[user_id])]
        for user_id in user_ids
    }


def _shared_other_user_count(edges: pd.DataFrame) -> dict[int, int]:
    if edges.empty:
        return {}
    shared_users: dict[int, set[int]] = defaultdict(set)
    for _, group in edges.groupby("entity_id"):
        users = sorted(set(group["user_id"].astype(int).tolist()))
        if len(users) <= 1:
            continue
        for user_id in users:
            shared_users[user_id].update(other_user for other_user in users if other_user != user_id)
    return {user_id: len(other_users) for user_id, other_users in shared_users.items()}


def _build_entity_metrics(
    edges: pd.DataFrame,
    user_ids: list[int],
    max_entity_users: int,
    prefix: str,
) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame({
            "user_id": user_ids,
            f"shared_{prefix}_user_count": 0,
            f"{prefix}_component_size": 1,
            f"{prefix}_entity_degree": 0,
            f"{prefix}_high_fanout_entity_degree": 0,
            f"{prefix}_max_entity_user_count": 0,
        })

    entity_user_count = (
        edges.groupby("entity_id")["user_id"]
        .nunique()
        .rename("entity_user_count")
        .reset_index()
    )
    edges = edges.merge(entity_user_count, on="entity_id", how="left")
    trimmed = edges[edges["entity_user_count"] <= max_entity_users].copy()

    trimmed_degree = trimmed.groupby("user_id").size().rename(f"{prefix}_entity_degree")
    high_fanout_degree = (
        edges[edges["entity_user_count"] > max_entity_users]
        .groupby("user_id")
        .size()
        .rename(f"{prefix}_high_fanout_entity_degree")
    )
    max_entity_user_count = (
        edges.groupby("user_id")["entity_user_count"]
        .max()
        .rename(f"{prefix}_max_entity_user_count")
    )
    shared_other_count = pd.Series(
        _shared_other_user_count(trimmed),
        name=f"shared_{prefix}_user_count",
        dtype="int64",
    )
    shared_other_count.index.name = "user_id"
    component_sizes = pd.Series(
        _component_sizes_from_bipartite(user_ids, trimmed[["user_id", "entity_id"]].drop_duplicates()),
        name=f"{prefix}_component_size",
        dtype="int64",
    )
    component_sizes.index.name = "user_id"

    result = pd.DataFrame({"user_id": user_ids})
    for series in (
        trimmed_degree,
        high_fanout_degree,
        max_entity_user_count,
        shared_other_count,
        component_sizes,
    ):
        if not series.empty:
            result = result.merge(series.reset_index(), on="user_id", how="left")

    fill_zero_columns = [
        f"shared_{prefix}_user_count",
        f"{prefix}_entity_degree",
        f"{prefix}_high_fanout_entity_degree",
        f"{prefix}_max_entity_user_count",
    ]
    for column in fill_zero_columns:
        if column not in result.columns:
            result[column] = 0
        result[column] = result[column].fillna(0).astype(int)
    if f"{prefix}_component_size" not in result.columns:
        result[f"{prefix}_component_size"] = 1
    result[f"{prefix}_component_size"] = result[f"{prefix}_component_size"].fillna(1).astype(int)
    return result


def build_official_graph_features(
    cutoff_ts: pd.Timestamp | None = None,
    cutoff_tag: str = "full",
) -> pd.DataFrame:
    user_info = load_clean_table("user_info").copy()
    user_info["user_id"] = pd.to_numeric(user_info["user_id"], errors="coerce").astype("Int64")
    user_ids = sorted(user_info["user_id"].dropna().astype(int).tolist())
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
            _prepare_entity_edges(twd_transfer, "source_ip_hash"),
            _prepare_entity_edges(crypto_transfer, "source_ip_hash"),
            _prepare_entity_edges(usdt_twd_trading, "source_ip_hash"),
        ],
        ignore_index=True,
    ).drop_duplicates()
    ip_metrics = _build_entity_metrics(ip_edges, user_ids, MAX_IP_ENTITY_USERS, "ip")

    wallet_edges = pd.concat(
        [
            _prepare_entity_edges(crypto_transfer, "from_wallet_hash"),
            _prepare_entity_edges(crypto_transfer, "to_wallet_hash"),
        ],
        ignore_index=True,
    ).drop_duplicates()
    wallet_metrics = _build_entity_metrics(wallet_edges, user_ids, MAX_WALLET_ENTITY_USERS, "wallet")

    relation = crypto_transfer[
        crypto_transfer["relation_user_id"].notna() & crypto_transfer["is_internal_transfer"].eq(True)
    ][["user_id", "relation_user_id"]].copy()
    relation["relation_user_id"] = pd.to_numeric(relation["relation_user_id"], errors="coerce").astype("Int64")
    directed = relation.dropna().drop_duplicates().copy()
    if not directed.empty:
        directed["user_id"] = directed["user_id"].astype(int)
        directed["relation_user_id"] = directed["relation_user_id"].astype(int)

    relation_pairs = directed.copy()
    if not relation_pairs.empty:
        relation_pairs["pair_left"] = relation_pairs[["user_id", "relation_user_id"]].min(axis=1)
        relation_pairs["pair_right"] = relation_pairs[["user_id", "relation_user_id"]].max(axis=1)
        relation_pairs = relation_pairs[relation_pairs["pair_left"] != relation_pairs["pair_right"]]
        relation_pairs = relation_pairs[["pair_left", "pair_right"]].drop_duplicates().rename(
            columns={"pair_left": "user_id", "pair_right": "relation_user_id"}
        )

    relation_component_size = pd.Series(
        _component_sizes_from_user_pairs(user_ids, relation_pairs),
        name="relation_component_size",
        dtype="int64",
    )
    relation_component_size.index.name = "user_id"
    if relation_pairs.empty:
        relation_degree = pd.Series(dtype="int64", name="relation_degree")
    else:
        relation_degree = (
            pd.concat(
                [
                    relation_pairs["user_id"],
                    relation_pairs["relation_user_id"].rename("user_id"),
                ],
                ignore_index=True,
            )
            .value_counts()
            .rename_axis("user_id")
            .rename("relation_degree")
        )
    denominator = max(1, len(user_ids) - 1)
    relation_degree_centrality = (
        relation_degree.astype(float) / float(denominator)
    ).rename("relation_degree_centrality")
    out_degree = (
        directed.groupby("user_id")["relation_user_id"]
        .nunique()
        .reset_index(name="relation_out_degree")
        if not directed.empty
        else pd.DataFrame(columns=["user_id", "relation_out_degree"])
    )
    in_degree = (
        directed.groupby("relation_user_id")["user_id"]
        .nunique()
        .reset_index(name="relation_in_degree")
        .rename(columns={"relation_user_id": "user_id"})
        if not directed.empty
        else pd.DataFrame(columns=["user_id", "relation_in_degree"])
    )
    relation_count = (
        directed.groupby("user_id")
        .size()
        .reset_index(name="relation_txn_count")
        if not directed.empty
        else pd.DataFrame(columns=["user_id", "relation_txn_count"])
    )

    result = user_info[["user_id"]].copy()
    result["user_id"] = result["user_id"].astype(int)
    result = result.merge(ip_metrics, on="user_id", how="left")
    result = result.merge(wallet_metrics, on="user_id", how="left")
    result = result.merge(relation_component_size.reset_index(), on="user_id", how="left")
    if not relation_degree_centrality.empty:
        result = result.merge(relation_degree_centrality.reset_index(), on="user_id", how="left")
    result = result.merge(out_degree, on="user_id", how="left")
    result = result.merge(in_degree, on="user_id", how="left")
    result = result.merge(relation_count, on="user_id", how="left")

    result["relation_component_size"] = result["relation_component_size"].fillna(1).astype(int)
    result["relation_degree_centrality"] = result["relation_degree_centrality"].fillna(0.0)
    result[["relation_out_degree", "relation_in_degree", "relation_txn_count"]] = (
        result[["relation_out_degree", "relation_in_degree", "relation_txn_count"]]
        .fillna(0)
        .astype(int)
    )
    relation_denominator = result["relation_txn_count"].replace(0, 1).astype(float)
    result["relation_fan_out_ratio_graph"] = (
        result["relation_out_degree"].astype(float) / relation_denominator
    ).fillna(0.0)
    result["snapshot_cutoff_at"] = cutoff_ts or list_event_cutoffs()[1]
    result["snapshot_cutoff_tag"] = cutoff_tag
    result.to_parquet(feature_output_path("official_graph_features", cutoff_tag), index=False)
    return result


def main() -> None:
    build_official_graph_features()


if __name__ == "__main__":
    main()
