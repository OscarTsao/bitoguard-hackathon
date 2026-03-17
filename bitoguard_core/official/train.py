from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from official.anomaly import build_official_anomaly_features
from official.bundle import save_selected_bundle
from official.common import RANDOM_SEED, feature_output_path, load_official_paths, save_json, save_pickle
from official.features import build_official_features
from official.graph_dataset import TransductiveGraph, build_transductive_graph
from official.graph_features import build_official_graph_features
from official.graph_model import save_graph_model, train_graphsage_model
from official.modeling import fit_catboost
from official.rules import evaluate_official_rules
from official.stacking import STACKER_FEATURE_COLUMNS, build_stacker_oof, save_stacker_model
from official.transductive_features import build_transductive_feature_frame
from official.transductive_validation import (
    PrimarySplitSpec,
    build_primary_transductive_splits,
    iter_fold_assignments,
)


LABEL_FREE_EXCLUDED_COLUMNS = {
    "user_id",
    "cohort",
    "snapshot_cutoff_at",
    "snapshot_cutoff_tag",
    "status",
    "is_known_blacklist",
    "needs_prediction",
    "in_train_label",
    "in_predict_label",
    "is_shadow_overlap",
    "top_reason_codes",
}

PRIMARY_GRAPH_MAX_EPOCHS = 40
FINAL_GRAPH_MIN_EPOCHS = 10


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


def _artifact_paths(paths: Any) -> dict[str, Path]:
    return {
        "oof": paths.feature_dir / "official_oof_predictions.parquet",
        "primary_split": paths.feature_dir / "official_primary_split.parquet",
        "primary_labeled_split": paths.feature_dir / "official_primary_transductive_split_full.parquet",
    }


def _label_frame(dataset: pd.DataFrame) -> pd.DataFrame:
    frame = dataset[dataset["status"].notna()][["user_id", "status"]].copy()
    frame["user_id"] = pd.to_numeric(frame["user_id"], errors="coerce").astype("Int64")
    frame["status"] = pd.to_numeric(frame["status"], errors="coerce").astype("Int64")
    return frame.dropna(subset=["user_id", "status"]).copy()


def _label_free_feature_columns(dataset: pd.DataFrame) -> list[str]:
    return [column for column in dataset.columns if column not in LABEL_FREE_EXCLUDED_COLUMNS]


def _transductive_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column != "user_id"]


def _prepare_base_frames(
    dataset: pd.DataFrame,
    transductive_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_free = dataset.copy()
    with_transductive = dataset.merge(transductive_features, on="user_id", how="left")
    trans_columns = _transductive_feature_columns(transductive_features)
    with_transductive[trans_columns] = with_transductive[trans_columns].fillna(0.0)
    return label_free, with_transductive


def run_transductive_oof_pipeline(
    dataset: pd.DataFrame,
    graph: TransductiveGraph,
    split_frame: pd.DataFrame,
    fold_column: str,
    base_a_feature_columns: list[str],
    base_b_feature_columns: list[str] | None = None,
    graph_max_epochs: int = 40,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    label_frame = _label_frame(dataset)
    assignments = iter_fold_assignments(split_frame, fold_column)
    rows: list[pd.DataFrame] = []
    fold_training_meta: list[dict[str, Any]] = []
    for fold_id, train_users, valid_users in assignments:
        fold_train_labels = label_frame[label_frame["user_id"].astype(int).isin(train_users)].copy()
        transductive_features = build_transductive_feature_frame(graph, fold_train_labels)
        label_free_frame, with_transductive_frame = _prepare_base_frames(dataset, transductive_features)

        train_label_free = label_free_frame[label_free_frame["user_id"].astype(int).isin(train_users)].copy()
        valid_label_free = label_free_frame[label_free_frame["user_id"].astype(int).isin(valid_users)].copy()
        train_transductive = with_transductive_frame[with_transductive_frame["user_id"].astype(int).isin(train_users)].copy()
        valid_transductive = with_transductive_frame[with_transductive_frame["user_id"].astype(int).isin(valid_users)].copy()

        base_a_fit = fit_catboost(train_label_free, valid_label_free, base_a_feature_columns)
        resolved_base_b_columns = base_b_feature_columns or [column for column in train_transductive.columns if column != "user_id"]
        base_b_fit = fit_catboost(train_transductive, valid_transductive, resolved_base_b_columns)
        graph_fit = train_graphsage_model(
            graph,
            label_frame=label_frame,
            train_user_ids=train_users,
            valid_user_ids=valid_users,
            max_epochs=graph_max_epochs,
        )

        fold_frame = valid_label_free[["user_id", "status"]].copy()
        fold_frame[fold_column] = fold_id
        fold_frame["base_a_probability"] = np.asarray(base_a_fit.validation_probabilities, dtype=float)
        fold_frame["base_b_probability"] = np.asarray(base_b_fit.validation_probabilities, dtype=float)
        fold_frame["base_c_probability"] = np.asarray(graph_fit.validation_probabilities, dtype=float)
        fold_frame["rule_score"] = pd.to_numeric(valid_label_free["rule_score"], errors="coerce").fillna(0.0).to_numpy()
        fold_frame["anomaly_score"] = pd.to_numeric(valid_label_free["anomaly_score"], errors="coerce").fillna(0.0).to_numpy()
        rows.append(fold_frame)
        fold_training_meta.append(
            {
                "fold": fold_id,
                "train_users": int(len(train_users)),
                "valid_users": int(len(valid_users)),
                "positive_count": int(valid_label_free["status"].astype(int).eq(1).sum()),
                "negative_count": int(valid_label_free["status"].astype(int).eq(0).sum()),
                "graph_best_epoch": int(graph_fit.model_meta["best_epoch"]),
            }
        )
    return pd.concat(rows, ignore_index=True).sort_values("user_id").reset_index(drop=True), fold_training_meta


def train_official_model() -> dict[str, Any]:
    dataset = _load_dataset("full")
    paths = load_official_paths()
    artifacts = _artifact_paths(paths)

    primary_split = build_primary_transductive_splits(
        dataset,
        cutoff_tag="full",
        spec=PrimarySplitSpec(),
        write_outputs=True,
    )
    split_frame = dataset[["user_id", "status", "cohort", "is_shadow_overlap"]].copy()
    split_frame = split_frame.merge(primary_split[["user_id", "primary_fold"]], on="user_id", how="left")
    split_frame.to_parquet(artifacts["primary_split"], index=False)

    graph = build_transductive_graph(dataset)
    base_a_feature_columns = _label_free_feature_columns(dataset)
    sample_transductive = build_transductive_feature_frame(graph, _label_frame(dataset))
    base_b_feature_columns = base_a_feature_columns + _transductive_feature_columns(sample_transductive)

    primary_oof, fold_training_meta = run_transductive_oof_pipeline(
        dataset,
        graph,
        primary_split,
        fold_column="primary_fold",
        base_a_feature_columns=base_a_feature_columns,
        base_b_feature_columns=base_b_feature_columns,
        graph_max_epochs=PRIMARY_GRAPH_MAX_EPOCHS,
    )
    primary_oof, stacker_model = build_stacker_oof(primary_oof, primary_split, fold_column="primary_fold")
    primary_oof.to_parquet(artifacts["oof"], index=False)

    label_frame = _label_frame(dataset)
    full_transductive = build_transductive_feature_frame(graph, label_frame)
    label_free_frame, with_transductive_frame = _prepare_base_frames(dataset, full_transductive)
    labeled_user_ids = set(label_frame["user_id"].astype(int).tolist())
    train_label_free = label_free_frame[label_free_frame["user_id"].astype(int).isin(labeled_user_ids)].copy()
    train_transductive = with_transductive_frame[with_transductive_frame["user_id"].astype(int).isin(labeled_user_ids)].copy()

    base_a_final = fit_catboost(train_label_free, None, base_a_feature_columns)
    base_b_final = fit_catboost(train_transductive, None, base_b_feature_columns)
    graph_epochs = int(np.median([item["graph_best_epoch"] + 1 for item in fold_training_meta])) if fold_training_meta else 40
    graph_final = train_graphsage_model(
        graph,
        label_frame=label_frame,
        train_user_ids=labeled_user_ids,
        valid_user_ids=None,
        max_epochs=max(FINAL_GRAPH_MIN_EPOCHS, graph_epochs),
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_a_path = paths.model_dir / f"official_catboost_base_a_{timestamp}.pkl"
    base_b_path = paths.model_dir / f"official_catboost_base_b_{timestamp}.pkl"
    graph_path = paths.model_dir / f"official_graphsage_{timestamp}.pt"
    stacker_path = paths.model_dir / f"official_stacker_{timestamp}.pkl"
    meta_path = paths.model_dir / f"official_transductive_{timestamp}.json"

    save_pickle(base_a_final.model, base_a_path)
    save_pickle(base_b_final.model, base_b_path)
    save_graph_model(graph_final.model_state, graph_path)
    save_stacker_model(stacker_model, stacker_path)

    meta = {
        "model_version": f"official_transductive_{timestamp}",
        "primary_validation_protocol": {
            "mode": "label_mask_transductive_cv",
            "n_splits": 5,
            "random_state": RANDOM_SEED,
        },
        "base_a_feature_columns": base_a_feature_columns,
        "base_b_feature_columns": base_b_feature_columns,
        "stacker_feature_columns": STACKER_FEATURE_COLUMNS,
        "fold_training_meta": fold_training_meta,
        "train_rows": int(len(train_label_free)),
        "predict_rows": int(dataset["needs_prediction"].eq(True).sum()),
    }
    save_json(meta, meta_path)

    bundle = {
        "bundle_version": f"official_bundle_{timestamp}",
        "selected_model": "stacked_transductive",
        "primary_validation_protocol": meta["primary_validation_protocol"],
        "base_model_paths": {
            "base_a_catboost": str(base_a_path),
            "base_b_catboost": str(base_b_path),
        },
        "graph_model_path": str(graph_path),
        "stacker_path": str(stacker_path),
        "stacker_feature_columns": STACKER_FEATURE_COLUMNS,
        "feature_columns_base_a": base_a_feature_columns,
        "feature_columns_base_b": base_b_feature_columns,
        "calibration_selection_basis": None,
        "secondary_stress_summary": None,
        "calibrator": None,
        "selected_threshold": None,
        "grouping_params": {},
        "shadow_protocol": {
            "mode": "secondary_only",
            "reason": "strict_group_split_used_only_for_robustness",
        },
        "oof_predictions_path": str(artifacts["oof"]),
        "primary_split_path": str(artifacts["primary_split"]),
        "primary_labeled_split_path": str(artifacts["primary_labeled_split"]),
        "model_meta_path": str(meta_path),
    }
    bundle_path = save_selected_bundle(bundle)
    return {
        "model_version": meta["model_version"],
        "base_a_model_path": str(base_a_path),
        "base_b_model_path": str(base_b_path),
        "graph_model_path": str(graph_path),
        "stacker_path": str(stacker_path),
        "meta_path": str(meta_path),
        "bundle_path": str(bundle_path),
        "oof_predictions_path": str(artifacts["oof"]),
    }


def main() -> None:
    print(train_official_model())


if __name__ == "__main__":
    main()
