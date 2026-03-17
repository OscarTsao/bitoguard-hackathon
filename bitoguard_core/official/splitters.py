from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from official.common import RANDOM_SEED, feature_output_path, load_clean_table, to_utc_timestamp


DEFAULT_GROUPING_PARAMS = {
    "hard_wallet_user_min": 2,
    "hard_wallet_user_max": 10,
    "soft_wallet_user_min": 11,
    "soft_wallet_user_max": 50,
    "hard_ip_user_min": 2,
    "hard_ip_user_max": 5,
    "min_ip_event_count": 2,
    "soft_ip_user_min": 6,
    "soft_ip_user_max": 30,
    "n_splits": 5,
    "split_seed_candidates": (42, 52, 62, 72, 82),
    "min_positive_per_fold": 250,
}


@dataclass
class _UnionFind:
    parent: dict[int, int]

    def find(self, value: int) -> int:
        parent = self.parent.setdefault(value, value)
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def build_graph_inputs(cutoff_ts: pd.Timestamp | None = None) -> dict[str, pd.DataFrame]:
    cutoff_ts = to_utc_timestamp(cutoff_ts)
    twd = load_clean_table("twd_transfer").copy()
    crypto = load_clean_table("crypto_transfer").copy()
    trade = load_clean_table("usdt_twd_trading").copy()
    for frame, time_col in ((twd, "created_at"), (crypto, "created_at"), (trade, "updated_at")):
        frame["user_id"] = pd.to_numeric(frame["user_id"], errors="coerce").astype("Int64")
        frame[time_col] = pd.to_datetime(frame[time_col], utc=True, errors="coerce")
        if cutoff_ts is not None:
            frame.drop(frame[frame[time_col] >= cutoff_ts].index, inplace=True)

    ip_edges = pd.concat(
        [
            twd[["user_id", "source_ip_hash", "id"]].rename(columns={"source_ip_hash": "entity_id", "id": "event_id"}),
            crypto[["user_id", "source_ip_hash", "id"]].rename(columns={"source_ip_hash": "entity_id", "id": "event_id"}),
            trade[["user_id", "source_ip_hash", "id"]].rename(columns={"source_ip_hash": "entity_id", "id": "event_id"}),
        ],
        ignore_index=True,
    )
    ip_edges = ip_edges[ip_edges["entity_id"].notna()].copy()

    wallet_edges = pd.concat(
        [
            crypto[["user_id", "from_wallet_hash"]].rename(columns={"from_wallet_hash": "entity_id"}),
            crypto[["user_id", "to_wallet_hash"]].rename(columns={"to_wallet_hash": "entity_id"}),
        ],
        ignore_index=True,
    )
    wallet_edges = wallet_edges[wallet_edges["entity_id"].notna()].copy()

    relation_edges = crypto[
        crypto["relation_user_id"].notna() & crypto["is_internal_transfer"].eq(True)
    ][["user_id", "relation_user_id"]].copy()
    relation_edges["relation_user_id"] = pd.to_numeric(relation_edges["relation_user_id"], errors="coerce").astype("Int64")
    relation_edges = relation_edges.dropna().drop_duplicates()

    return {
        "ip_edges": ip_edges,
        "wallet_edges": wallet_edges,
        "relation_edges": relation_edges,
    }


def build_strong_groups(
    dataset: pd.DataFrame,
    graph_inputs: dict[str, pd.DataFrame],
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    params = {**DEFAULT_GROUPING_PARAMS, **(params or {})}
    users = dataset.copy()
    users["user_id"] = pd.to_numeric(users["user_id"], errors="coerce").astype("Int64")
    users = users.dropna(subset=["user_id"]).copy()
    user_ids = users["user_id"].astype(int).tolist()
    uf = _UnionFind(parent={user_id: user_id for user_id in user_ids})
    allowed_users = set(user_ids)

    relation_edges = graph_inputs["relation_edges"].copy()
    for _, row in relation_edges.iterrows():
        left = int(row["user_id"])
        right = int(row["relation_user_id"])
        if left in allowed_users and right in allowed_users:
            uf.union(left, right)

    wallet_edges = graph_inputs["wallet_edges"].copy()
    wallet_edges["user_id"] = pd.to_numeric(wallet_edges["user_id"], errors="coerce").astype("Int64")
    wallet_edges = wallet_edges[wallet_edges["user_id"].isin(list(allowed_users))].copy()
    for _, frame in wallet_edges.groupby("entity_id"):
        edge_users = sorted({int(user_id) for user_id in frame["user_id"].dropna().tolist()})
        if params["hard_wallet_user_min"] <= len(edge_users) <= params["hard_wallet_user_max"]:
            head = edge_users[0]
            for user_id in edge_users[1:]:
                uf.union(head, user_id)

    ip_edges = graph_inputs["ip_edges"].copy()
    ip_edges["user_id"] = pd.to_numeric(ip_edges["user_id"], errors="coerce").astype("Int64")
    ip_edges = ip_edges[ip_edges["user_id"].isin(list(allowed_users))].copy()
    for _, frame in ip_edges.groupby("entity_id"):
        edge_users = sorted({int(user_id) for user_id in frame["user_id"].dropna().tolist()})
        event_count = int(frame["event_id"].nunique())
        if (
            params["hard_ip_user_min"] <= len(edge_users) <= params["hard_ip_user_max"]
            and event_count >= params["min_ip_event_count"]
        ):
            head = edge_users[0]
            for user_id in edge_users[1:]:
                uf.union(head, user_id)

    roots = {user_id: uf.find(user_id) for user_id in user_ids}
    root_to_group: dict[int, int] = {}
    for user_id, root in roots.items():
        if root not in root_to_group:
            root_to_group[root] = len(root_to_group) + 1
    users["strong_group_id"] = users["user_id"].map(lambda user_id: root_to_group[roots[int(user_id)]])

    group_stats = users.groupby("strong_group_id").agg(
        strong_group_size=("user_id", "size"),
        group_has_shadow_overlap=("is_shadow_overlap", "max"),
        group_has_predict_only=("cohort", lambda s: bool((s == "predict_only").any())),
        group_has_labeled=("status", lambda s: bool(s.notna().any())),
    ).reset_index()
    result = users.merge(group_stats, on="strong_group_id", how="left")
    return result.sort_values(["strong_group_id", "user_id"]).reset_index(drop=True)


def reserve_shadow_groups(group_index: pd.DataFrame) -> pd.DataFrame:
    result = group_index.copy()
    result["group_role"] = "core_trainable"
    result["shadow_split"] = None
    return result


def _is_valid_fold_assignment(
    frame: pd.DataFrame,
    fold_assignments: list[pd.DataFrame],
    min_positive_per_fold: int,
) -> bool:
    for summary in _fold_label_summary(frame, fold_assignments):
        positive_count = summary["positive_count"]
        negative_count = summary["negative_count"]
        if positive_count < min_positive_per_fold or negative_count == 0:
            return False
    return True


def _fold_label_summary(frame: pd.DataFrame, fold_assignments: list[pd.DataFrame]) -> list[dict[str, int]]:
    summaries: list[dict[str, int]] = []
    for assignment in fold_assignments:
        fold_frame = frame.iloc[assignment["row_index"].tolist()]
        summaries.append(
            {
                "core_fold": int(assignment["core_fold"].iloc[0]),
                "positive_count": int(fold_frame["status"].astype(int).eq(1).sum()),
                "negative_count": int(fold_frame["status"].astype(int).eq(0).sum()),
            }
        )
    return summaries


def make_core_group_folds(
    labeled_core: pd.DataFrame,
    n_splits: int = 5,
    seed_candidates: tuple[int, ...] = (RANDOM_SEED,),
    min_positive_per_fold: int = 250,
) -> pd.DataFrame:
    frame = labeled_core.copy()
    if frame.empty:
        return pd.DataFrame(columns=["user_id", "strong_group_id", "core_fold"])

    y = frame["status"].astype(int)
    positive_groups = frame[y == 1]["strong_group_id"].nunique()
    negative_groups = frame[y == 0]["strong_group_id"].nunique()
    resolved_splits = max(2, min(n_splits, int(positive_groups or 1), int(negative_groups or 1), int(frame["strong_group_id"].nunique())))
    best_assignments: list[pd.DataFrame] | None = None
    best_rank: tuple[int, int, int] | None = None
    for seed in seed_candidates:
        splitter = StratifiedGroupKFold(n_splits=resolved_splits, shuffle=True, random_state=int(seed))
        fold_assignments: list[pd.DataFrame] = []
        for fold_index, (_, valid_idx) in enumerate(splitter.split(frame, y, frame["strong_group_id"])):
            fold_frame = frame.iloc[valid_idx][["user_id", "strong_group_id"]].copy()
            fold_frame["core_fold"] = fold_index
            fold_frame["row_index"] = valid_idx
            fold_assignments.append(fold_frame)
        if _is_valid_fold_assignment(frame, fold_assignments, min_positive_per_fold=min_positive_per_fold):
            return pd.concat(
                [fold.drop(columns=["row_index"]) for fold in fold_assignments],
                ignore_index=True,
            )
        summary = _fold_label_summary(frame, fold_assignments)
        rank = (
            min(item["positive_count"] for item in summary),
            -max(item["positive_count"] for item in summary) + min(item["positive_count"] for item in summary),
            min(item["negative_count"] for item in summary),
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_assignments = fold_assignments

    if best_assignments is None:
        raise ValueError(
            "Unable to find any grouped split with the configured seeds."
        )

    return pd.concat(
        [fold.drop(columns=["row_index"]) for fold in best_assignments],
        ignore_index=True,
    )


def compute_weak_purge_map(
    group_index: pd.DataFrame,
    weak_edges: pd.DataFrame,
) -> dict[int, set[int]]:
    user_to_group = {
        int(row["user_id"]): int(row["strong_group_id"])
        for _, row in group_index[["user_id", "strong_group_id"]].dropna().iterrows()
    }
    weak = weak_edges.copy()
    weak["user_id"] = pd.to_numeric(weak["user_id"], errors="coerce").astype("Int64")
    purge_map: dict[int, set[int]] = defaultdict(set)
    for _, frame in weak.groupby("entity_id"):
        edge_users = sorted({int(user_id) for user_id in frame["user_id"].dropna().tolist() if int(user_id) in user_to_group})
        for user_id in edge_users:
            group_id = user_to_group[user_id]
            purge_map[group_id].update(other for other in edge_users if other != user_id)
    return {group_id: set(sorted(user_ids)) for group_id, user_ids in purge_map.items()}


def _build_soft_purge_edges(
    graph_inputs: dict[str, pd.DataFrame],
    allowed_users: set[int],
    params: dict[str, Any],
) -> pd.DataFrame:
    wallet_edges = graph_inputs["wallet_edges"].copy()
    wallet_edges["user_id"] = pd.to_numeric(wallet_edges["user_id"], errors="coerce").astype("Int64")
    wallet_edges = wallet_edges[wallet_edges["user_id"].isin(list(allowed_users))][["user_id", "entity_id"]].drop_duplicates()
    wallet_sizes = wallet_edges.groupby("entity_id")["user_id"].nunique()
    wallet_entities = wallet_sizes[
        (wallet_sizes >= params["soft_wallet_user_min"]) & (wallet_sizes <= params["soft_wallet_user_max"])
    ].index
    wallet_edges = wallet_edges[wallet_edges["entity_id"].isin(wallet_entities)].copy()
    wallet_edges["entity_id"] = "wallet:" + wallet_edges["entity_id"].astype(str)

    ip_edges = graph_inputs["ip_edges"].copy()
    ip_edges["user_id"] = pd.to_numeric(ip_edges["user_id"], errors="coerce").astype("Int64")
    ip_edges = ip_edges[ip_edges["user_id"].isin(list(allowed_users))][["user_id", "entity_id"]].drop_duplicates()
    ip_sizes = ip_edges.groupby("entity_id")["user_id"].nunique()
    ip_entities = ip_sizes[
        (ip_sizes >= params["soft_ip_user_min"]) & (ip_sizes <= params["soft_ip_user_max"])
    ].index
    ip_edges = ip_edges[ip_edges["entity_id"].isin(ip_entities)].copy()
    ip_edges["entity_id"] = "ip:" + ip_edges["entity_id"].astype(str)

    return pd.concat([wallet_edges, ip_edges], ignore_index=True)


def build_split_artifacts(
    dataset: pd.DataFrame,
    cutoff_ts: pd.Timestamp | None = None,
    cutoff_tag: str = "full",
    params: dict[str, Any] | None = None,
    write_outputs: bool = True,
) -> tuple[pd.DataFrame, dict[int, set[int]], dict[str, pd.DataFrame]]:
    params = {**DEFAULT_GROUPING_PARAMS, **(params or {})}
    graph_inputs = build_graph_inputs(cutoff_ts=cutoff_ts)
    labeled_dataset = dataset[dataset["status"].notna()].copy()
    group_index = build_strong_groups(labeled_dataset, graph_inputs, params)
    group_index = reserve_shadow_groups(group_index)

    allowed_users = set(group_index["user_id"].astype(int).tolist())
    weak_edges = _build_soft_purge_edges(graph_inputs, allowed_users, params)
    purge_map = compute_weak_purge_map(group_index, weak_edges)

    labeled_core = group_index[(group_index["group_role"] == "core_trainable") & (group_index["status"].notna())].copy()
    fold_index = make_core_group_folds(
        labeled_core,
        n_splits=int(params["n_splits"]),
        seed_candidates=tuple(int(seed) for seed in params["split_seed_candidates"]),
        min_positive_per_fold=int(params["min_positive_per_fold"]),
    )
    group_index = group_index.merge(fold_index[["user_id", "core_fold"]], on="user_id", how="left")

    if write_outputs:
        group_index.to_parquet(feature_output_path("official_group_index", cutoff_tag), index=False)
    return group_index, purge_map, graph_inputs
