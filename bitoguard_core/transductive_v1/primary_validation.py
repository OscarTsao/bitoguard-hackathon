from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from transductive_v1.common import RANDOM_SEED, feature_path


PRIMARY_N_SPLITS = 10


@dataclass(frozen=True)
class PrimarySplitSpec:
    n_splits: int = PRIMARY_N_SPLITS
    random_state: int = RANDOM_SEED


def build_primary_split(
    labeled_frame: pd.DataFrame,
    cutoff_tag: str = "full",
    spec: PrimarySplitSpec | None = None,
    write_outputs: bool = True,
) -> pd.DataFrame:
    spec = spec or PrimarySplitSpec()
    frame = labeled_frame[["user_id", "status"]].copy()
    frame["user_id"] = pd.to_numeric(frame["user_id"], errors="coerce").astype("Int64")
    frame["status"] = pd.to_numeric(frame["status"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["user_id", "status"]).drop_duplicates(subset=["user_id"]).sort_values("user_id").reset_index(drop=True)

    splitter = StratifiedKFold(n_splits=spec.n_splits, shuffle=True, random_state=spec.random_state)
    folds = []
    for fold_id, (_, valid_idx) in enumerate(splitter.split(frame[["user_id"]], frame["status"].astype(int))):
        fold_frame = frame.iloc[valid_idx][["user_id"]].copy()
        fold_frame["primary_fold"] = fold_id
        folds.append(fold_frame)
    result = frame.merge(pd.concat(folds, ignore_index=True), on="user_id", how="left")
    result["primary_fold"] = result["primary_fold"].astype(int)
    if write_outputs:
        result.to_parquet(feature_path("primary_split", cutoff_tag), index=False)
    return result


def iter_primary_folds(split_frame: pd.DataFrame) -> list[tuple[int, set[int], set[int]]]:
    labeled_users = set(split_frame["user_id"].astype(int).tolist())
    assignments = []
    for fold_id in sorted(int(value) for value in split_frame["primary_fold"].unique()):
        valid_users = set(split_frame[split_frame["primary_fold"] == fold_id]["user_id"].astype(int).tolist())
        assignments.append((fold_id, labeled_users.difference(valid_users), valid_users))
    return assignments
