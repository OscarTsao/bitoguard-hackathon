from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from transductive_v1.common import RANDOM_SEED, feature_path
from transductive_v1.graph_store import GraphStore


SECONDARY_N_SPLITS = 5


@dataclass(frozen=True)
class SecondarySplitSpec:
    n_splits: int = SECONDARY_N_SPLITS
    random_state: int = RANDOM_SEED


class _UnionFind:
    def __init__(self, values: list[int]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        parent = self.parent.setdefault(value, value)
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _build_groups(labeled_frame: pd.DataFrame, graph_store: GraphStore) -> pd.DataFrame:
    users = sorted(labeled_frame["user_id"].astype(int).tolist())
    allowed_users = set(users)
    uf = _UnionFind(users)
    for _, row in graph_store.relation_edges.iterrows():
        left = int(row["user_id"])
        right = int(row["relation_user_id"])
        if left in allowed_users and right in allowed_users:
            uf.union(left, right)
    for edge_frame in (graph_store.wallet_edges, graph_store.ip_edges):
        for _, group in edge_frame.groupby("entity_id"):
            group_users = sorted({int(user_id) for user_id in group["user_id"].tolist() if int(user_id) in allowed_users})
            if len(group_users) <= 1:
                continue
            head = group_users[0]
            for user_id in group_users[1:]:
                uf.union(head, user_id)
    roots = {user_id: uf.find(user_id) for user_id in users}
    root_to_group: dict[int, int] = {}
    for user_id, root in roots.items():
        if root not in root_to_group:
            root_to_group[root] = len(root_to_group) + 1
    result = labeled_frame.copy()
    result["secondary_group_id"] = result["user_id"].astype(int).map(lambda user_id: root_to_group[roots[int(user_id)]])
    return result


def build_secondary_group_split(
    labeled_frame: pd.DataFrame,
    graph_store: GraphStore,
    cutoff_tag: str = "full",
    spec: SecondarySplitSpec | None = None,
    write_outputs: bool = True,
) -> pd.DataFrame:
    spec = spec or SecondarySplitSpec()
    grouped = _build_groups(labeled_frame, graph_store)
    splitter = StratifiedGroupKFold(n_splits=spec.n_splits, shuffle=True, random_state=spec.random_state)
    folds = []
    y = grouped["status"].astype(int)
    for fold_id, (_, valid_idx) in enumerate(splitter.split(grouped[["user_id"]], y, grouped["secondary_group_id"])):
        fold_frame = grouped.iloc[valid_idx][["user_id", "secondary_group_id"]].copy()
        fold_frame["secondary_fold"] = fold_id
        folds.append(fold_frame)
    result = grouped.merge(pd.concat(folds, ignore_index=True), on=["user_id", "secondary_group_id"], how="left")
    result["secondary_fold"] = result["secondary_fold"].astype(int)
    if write_outputs:
        result.to_parquet(feature_path("secondary_group_split", cutoff_tag), index=False)
    return result


def iter_secondary_folds(split_frame: pd.DataFrame) -> list[tuple[int, set[int], set[int]]]:
    labeled_users = set(split_frame["user_id"].astype(int).tolist())
    assignments = []
    for fold_id in sorted(int(value) for value in split_frame["secondary_fold"].unique()):
        valid_users = set(split_frame[split_frame["secondary_fold"] == fold_id]["user_id"].astype(int).tolist())
        assignments.append((fold_id, labeled_users.difference(valid_users), valid_users))
    return assignments
