from __future__ import annotations

import pandas as pd

from transductive_v1.common import feature_path, load_clean_table


def build_user_universe(cutoff_tag: str = "full", write_outputs: bool = True) -> pd.DataFrame:
    user_info = load_clean_table("user_info").copy()
    train_label = load_clean_table("train_label").copy()
    predict_label = load_clean_table("predict_label").copy()

    user_info["user_id"] = pd.to_numeric(user_info["user_id"], errors="coerce").astype("Int64")
    train_label["user_id"] = pd.to_numeric(train_label["user_id"], errors="coerce").astype("Int64")
    train_label["status"] = pd.to_numeric(train_label["status"], errors="coerce").astype("Int64")
    predict_label["user_id"] = pd.to_numeric(predict_label["user_id"], errors="coerce").astype("Int64")

    user_ids = pd.Series(
        pd.concat(
            [
                user_info["user_id"],
                train_label["user_id"],
                predict_label["user_id"],
            ],
            ignore_index=True,
        ).dropna().unique(),
        name="user_id",
    )

    universe = pd.DataFrame({"user_id": user_ids}).sort_values("user_id").reset_index(drop=True)
    universe = universe.merge(user_info, on="user_id", how="left")
    universe = universe.merge(train_label[["user_id", "status"]], on="user_id", how="left")
    universe["in_train_label"] = universe["status"].notna()
    universe = universe.merge(predict_label.assign(needs_prediction=True), on="user_id", how="left")
    universe["needs_prediction"] = universe["needs_prediction"].eq(True)
    universe["cohort"] = "unlabeled_context"
    universe.loc[universe["in_train_label"], "cohort"] = "train_only"
    universe.loc[~universe["in_train_label"] & universe["needs_prediction"], "cohort"] = "predict_only"
    universe["is_shadow_overlap"] = False

    if write_outputs:
        universe.to_parquet(feature_path("user_universe", cutoff_tag), index=False)
    return universe
