from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from official.anomaly import build_official_anomaly_features
from official.bundle import save_selected_bundle
from official.common import RANDOM_SEED, encode_frame, feature_output_path, load_official_paths, save_json, save_pickle
from official.features import build_official_features
from official.graph_features import build_official_graph_features
from official.modeling import fit_lgbm
from official.rules import evaluate_official_rules
from official.splitters import DEFAULT_GROUPING_PARAMS, build_split_artifacts


META_COLUMNS = {
    "user_id",
    "cohort",
    "snapshot_cutoff_at",
    "snapshot_cutoff_tag",
    "status",
    "needs_prediction",
    "in_train_label",
    "in_predict_label",
    "is_shadow_overlap",
    "strong_group_id",
    "strong_group_size",
    "group_has_shadow_overlap",
    "group_has_predict_only",
    "group_has_labeled",
    "group_role",
    "shadow_split",
    "core_fold",
}


def _predict_lgbm(model: object, encoded_columns: list[str], frame: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
    x_frame, _ = encode_frame(frame, feature_columns, reference_columns=encoded_columns)
    return pd.Series(model.predict_proba(x_frame)[:, 1], index=frame.index)


def _load_dataset(cutoff_tag: str = "full") -> pd.DataFrame:
    feature_path = feature_output_path("official_user_features", cutoff_tag)
    graph_path = feature_output_path("official_graph_features", cutoff_tag)
    anomaly_path = feature_output_path("official_anomaly_features", cutoff_tag)
    if not feature_path.exists():
        build_official_features(cutoff_tag=cutoff_tag)
    if not graph_path.exists():
        build_official_graph_features(cutoff_tag=cutoff_tag)
    if not anomaly_path.exists():
        build_official_anomaly_features(cutoff_tag=cutoff_tag)
    dataset = pd.read_parquet(feature_path)
    dataset = dataset.merge(pd.read_parquet(graph_path), on=["user_id", "snapshot_cutoff_at", "snapshot_cutoff_tag"], how="left")
    dataset = dataset.merge(pd.read_parquet(anomaly_path), on=["user_id", "snapshot_cutoff_at", "snapshot_cutoff_tag"], how="left")
    dataset = dataset.merge(evaluate_official_rules(dataset), on="user_id", how="left")
    return dataset


def _split_artifact_paths(paths: Any) -> dict[str, Path]:
    return {
        "oof": paths.feature_dir / "official_oof_predictions.parquet",
        "shadow": paths.feature_dir / "official_shadow_predictions.parquet",
        "split": paths.feature_dir / "official_primary_split.parquet",
        "group_index": paths.feature_dir / "official_group_index_full.parquet",
    }


def train_official_model() -> dict[str, Any]:
    dataset = _load_dataset("full")
    grouping_params = DEFAULT_GROUPING_PARAMS.copy()
    group_index, purge_map, _ = build_split_artifacts(dataset, cutoff_tag="full", params=grouping_params, write_outputs=True)

    split_columns = [
        "user_id",
        "strong_group_id",
        "strong_group_size",
        "group_has_shadow_overlap",
        "group_has_predict_only",
        "group_has_labeled",
        "group_role",
        "shadow_split",
        "core_fold",
    ]
    dataset = dataset.drop(columns=[column for column in split_columns if column != "user_id" and column in dataset.columns], errors="ignore")
    dataset = dataset.merge(group_index[split_columns], on="user_id", how="left")

    core_labeled = dataset[(dataset["group_role"] == "core_trainable") & (dataset["status"].notna()) & (dataset["core_fold"].notna())].copy()
    feature_columns = [column for column in dataset.columns if column not in META_COLUMNS]
    fold_ids = sorted(int(fold_id) for fold_id in core_labeled["core_fold"].dropna().unique())

    oof_rows: list[pd.DataFrame] = []
    for fold_id in fold_ids:
        valid_frame = core_labeled[core_labeled["core_fold"] == fold_id].copy()
        train_frame = core_labeled[core_labeled["core_fold"] != fold_id].copy()
        valid_group_ids = set(valid_frame["strong_group_id"].astype(int).tolist())
        purge_users: set[int] = set()
        for group_id in valid_group_ids:
            purge_users.update(purge_map.get(group_id, set()))
        if purge_users:
            train_frame = train_frame[~train_frame["user_id"].astype(int).isin(purge_users)].copy()
        fit_result = fit_lgbm(train_frame, valid_frame, feature_columns)
        fold_predictions = valid_frame[["user_id", "status", "strong_group_id", "core_fold"]].copy()
        fold_predictions["raw_probability"] = fit_result.validation_probabilities
        fold_predictions["selected_model"] = fit_result.model_name
        fold_predictions["purged_train_users"] = len(purge_users)
        oof_rows.append(fold_predictions)

    oof_predictions = pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame(columns=["user_id", "status", "strong_group_id", "core_fold", "raw_probability", "selected_model", "purged_train_users"])

    final_fit = fit_lgbm(core_labeled, None, feature_columns)
    paths = load_official_paths()
    version = f"official_lgbm_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    model_path = paths.model_dir / f"{version}.pkl"
    meta_path = paths.model_dir / f"{version}.json"
    save_pickle(final_fit.model, model_path)
    save_json(
        {
            "model_version": version,
            "feature_columns": feature_columns,
            "encoded_columns": final_fit.encoded_columns,
            "training_cohort": "core_trainable",
            "grouping_params": grouping_params,
            "fold_count": len(fold_ids),
            "train_rows": int(len(core_labeled)),
            "random_seed": RANDOM_SEED,
        },
        meta_path,
    )

    shadow_frame = dataset[(dataset["group_role"] == "shadow_reserved") & (dataset["status"].notna()) & (dataset["shadow_split"].notna())].copy()
    shadow_frame["raw_probability"] = _predict_lgbm(final_fit.model, final_fit.encoded_columns or [], shadow_frame, feature_columns)
    shadow_predictions = shadow_frame[["user_id", "status", "strong_group_id", "shadow_split", "raw_probability"]].copy()
    shadow_predictions["selected_model"] = "lgbm"

    split_frame = dataset[["user_id", "status", "cohort", "group_role", "strong_group_id", "core_fold", "shadow_split", "is_shadow_overlap"]].copy()
    artifacts = _split_artifact_paths(paths)
    oof_predictions.to_parquet(artifacts["oof"], index=False)
    shadow_predictions.to_parquet(artifacts["shadow"], index=False)
    split_frame.to_parquet(artifacts["split"], index=False)
    group_index.to_parquet(artifacts["group_index"], index=False)

    bundle = {
        "bundle_version": f"official_bundle_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "selected_model": "lgbm",
        "model_paths": {"lgbm": str(model_path)},
        "feature_columns_lgbm": feature_columns,
        "encoded_columns_lgbm": final_fit.encoded_columns,
        "catboost_feature_columns": [],
        "catboost_cat_features": [],
        "calibrator": None,
        "selected_threshold": None,
        "shadow_protocol": {"dev_ratio": grouping_params["shadow_dev_ratio"], "holdout_ratio": 1.0 - grouping_params["shadow_dev_ratio"]},
        "grouping_params": grouping_params,
        "split_path": str(artifacts["split"]),
        "group_index_path": str(artifacts["group_index"]),
        "oof_predictions_path": str(artifacts["oof"]),
        "shadow_predictions_path": str(artifacts["shadow"]),
    }
    bundle_path = save_selected_bundle(bundle)
    return {
        "model_version": version,
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "bundle_path": str(bundle_path),
        "oof_predictions_path": str(artifacts["oof"]),
    }


def main() -> None:
    print(train_official_model())


if __name__ == "__main__":
    main()
