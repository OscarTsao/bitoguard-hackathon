from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from official.common import RANDOM_SEED, feature_output_path
from official.splitters import build_split_artifacts


PRIMARY_FOLD_COUNT = 5


@dataclass(frozen=True)
class PrimarySplitSpec:
    n_splits: int = PRIMARY_FOLD_COUNT
    random_state: int = RANDOM_SEED


def build_primary_transductive_splits(
    dataset: pd.DataFrame,
    cutoff_tag: str = "full",
    spec: PrimarySplitSpec | None = None,
    write_outputs: bool = True,
) -> pd.DataFrame:
    spec = spec or PrimarySplitSpec()
    labeled = dataset[dataset["status"].notna()][["user_id", "status"]].copy()
    labeled["user_id"] = pd.to_numeric(labeled["user_id"], errors="coerce").astype("Int64")
    labeled = labeled.dropna(subset=["user_id"]).drop_duplicates(subset=["user_id"]).sort_values("user_id").reset_index(drop=True)

    splitter = StratifiedKFold(n_splits=spec.n_splits, shuffle=True, random_state=spec.random_state)
    folds: list[pd.DataFrame] = []
    for fold_id, (_, valid_idx) in enumerate(splitter.split(labeled[["user_id"]], labeled["status"].astype(int))):
        fold_frame = labeled.iloc[valid_idx][["user_id"]].copy()
        fold_frame["primary_fold"] = fold_id
        folds.append(fold_frame)
    result = labeled.merge(pd.concat(folds, ignore_index=True), on="user_id", how="left")
    result["primary_fold"] = result["primary_fold"].astype(int)

    if write_outputs:
        result.to_parquet(feature_output_path("official_primary_transductive_split", cutoff_tag), index=False)
    return result


def build_secondary_strict_splits(
    dataset: pd.DataFrame,
    cutoff_tag: str = "full",
    params: dict[str, Any] | None = None,
    write_outputs: bool = True,
) -> pd.DataFrame:
    group_index, purge_map, graph_inputs = build_split_artifacts(
        dataset,
        cutoff_tag=cutoff_tag,
        params=params,
        write_outputs=write_outputs,
    )
    secondary = group_index[group_index["core_fold"].notna()].copy()
    secondary["secondary_fold"] = secondary["core_fold"].astype(int)
    if write_outputs:
        secondary.to_parquet(feature_output_path("official_secondary_group_split", cutoff_tag), index=False)
    secondary.attrs["purge_map"] = purge_map
    secondary.attrs["graph_inputs"] = graph_inputs
    return secondary


def iter_fold_assignments(split_frame: pd.DataFrame, fold_column: str) -> list[tuple[int, set[int], set[int]]]:
    labeled_users = set(split_frame["user_id"].astype(int).tolist())
    assignments: list[tuple[int, set[int], set[int]]] = []
    for fold_id in sorted(int(value) for value in split_frame[fold_column].dropna().unique()):
        valid_users = set(split_frame[split_frame[fold_column] == fold_id]["user_id"].astype(int).tolist())
        train_users = labeled_users.difference(valid_users)
        assignments.append((fold_id, train_users, valid_users))
    return assignments
