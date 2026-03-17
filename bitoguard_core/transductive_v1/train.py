from __future__ import annotations

import gc
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd

from official.rules import evaluate_official_rules
from transductive_v1.branch_tabular import fit_catboost
from transductive_v1.common import RANDOM_SEED, bundle_path, feature_path, model_path, report_path, save_json, save_pickle
from transductive_v1.dataset import build_user_universe
from transductive_v1.graph_risk import build_graph_risk_features
from transductive_v1.graph_store import GraphStore, build_graph_store
from transductive_v1.label_aware_features import build_label_aware_features
from transductive_v1.label_free_features import build_label_free_user_features
from transductive_v1.primary_validation import PRIMARY_N_SPLITS, build_primary_split, iter_primary_folds
from transductive_v1.stacking import build_stacker_oof, resolve_stacker_feature_columns


def _labeled_frame(frame: pd.DataFrame) -> pd.DataFrame:
    labeled = frame[frame["status"].notna()].copy()
    labeled["user_id"] = pd.to_numeric(labeled["user_id"], errors="coerce").astype("Int64")
    labeled["status"] = pd.to_numeric(labeled["status"], errors="coerce").astype("Int64")
    return labeled.dropna(subset=["user_id", "status"])


def _merge_training_frame(label_free_frame: pd.DataFrame, graph_store: GraphStore, label_aware_frame: pd.DataFrame) -> pd.DataFrame:
    graph_risk = build_graph_risk_features(label_aware_frame)
    frame = label_free_frame.merge(graph_store.structural_features, on="user_id", how="left")
    frame = frame.merge(label_aware_frame, on="user_id", how="left")
    frame = frame.merge(graph_risk[["user_id", "graph_risk_score"]], on="user_id", how="left")
    fill_columns = [column for column in frame.columns if column not in {"user_id", "status", "cohort", "top_reason_codes", "snapshot_cutoff_at", "snapshot_cutoff_tag"}]
    for column in fill_columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
            if pd.api.types.is_float_dtype(frame[column]):
                frame[column] = frame[column].astype("float32")
            elif pd.api.types.is_integer_dtype(frame[column]):
                frame[column] = frame[column].astype("int32")
    rules = evaluate_official_rules(frame)
    frame = frame.drop(columns=["rule_score", "top_reason_codes"], errors="ignore").merge(rules, on="user_id", how="left")
    frame["rule_score"] = pd.to_numeric(frame["rule_score"], errors="coerce").fillna(0.0)
    frame["top_reason_codes"] = frame["top_reason_codes"].fillna("[]")
    return frame


def _feature_columns(frame: pd.DataFrame, include_label_aware: bool) -> list[str]:
    excluded = {
        "user_id",
        "status",
        "cohort",
        "needs_prediction",
        "in_train_label",
        "in_predict_label",
        "is_shadow_overlap",
        "top_reason_codes",
        "snapshot_cutoff_at",
        "snapshot_cutoff_tag",
    }
    columns = [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not include_label_aware:
        columns = [
            column
            for column in columns
            if not (
                column.startswith("direct_positive_")
                or column.startswith("wallet_positive_")
                or column.startswith("ip_positive_")
                or column.startswith("positive_exposure_")
                or column.startswith("positive_seed_")
                or column == "graph_risk_score"
            )
        ]
    return columns


def _resolved_feature_columns(
    label_free_frame: pd.DataFrame,
    graph_store: GraphStore,
    label_aware_frame: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    excluded = {
        "user_id",
        "status",
        "cohort",
        "needs_prediction",
        "in_train_label",
        "in_predict_label",
        "is_shadow_overlap",
        "top_reason_codes",
        "snapshot_cutoff_at",
        "snapshot_cutoff_tag",
    }
    label_free_numeric = [
        column
        for column in label_free_frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(label_free_frame[column])
    ]
    graph_structural_numeric = [
        column
        for column in graph_store.structural_features.columns
        if column != "user_id" and pd.api.types.is_numeric_dtype(graph_store.structural_features[column])
    ]
    label_aware_numeric = [
        column
        for column in label_aware_frame.columns
        if column != "user_id" and pd.api.types.is_numeric_dtype(label_aware_frame[column])
    ]
    base_a = list(dict.fromkeys(label_free_numeric + graph_structural_numeric))
    base_b = list(dict.fromkeys(base_a + label_aware_numeric + ["graph_risk_score"]))
    return base_a, base_b


def run_primary_base_oof(
    label_free_frame: pd.DataFrame,
    graph_store: GraphStore,
    primary_split: pd.DataFrame,
    cutoff_tag: str = "full",
    split_artifact_name: str = "primary_split",
    fold_column: str = "primary_fold",
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str], list[str]]:
    rows: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, Any]] = []
    labeled_all = _labeled_frame(label_free_frame)
    label_frame = labeled_all[["user_id", "status"]].copy()
    labeled_user_ids = set(labeled_all["user_id"].astype(int).tolist())
    graph_structural_labeled = graph_store.structural_features[
        graph_store.structural_features["user_id"].astype(int).isin(labeled_user_ids)
    ].copy()
    merged_store = GraphStore(
        user_ids=graph_store.user_ids,
        user_index=graph_store.user_index,
        relation_edges=graph_store.relation_edges,
        wallet_edges=graph_store.wallet_edges,
        ip_edges=graph_store.ip_edges,
        projected_edges=graph_store.projected_edges,
        neighbors=graph_store.neighbors,
        weighted_neighbors=graph_store.weighted_neighbors,
        structural_features=graph_structural_labeled,
    )
    sample_label_aware = build_label_aware_features(graph_store, label_frame)
    sample_label_aware_labeled = sample_label_aware[sample_label_aware["user_id"].astype(int).isin(labeled_user_ids)].copy()
    base_a_feature_columns, base_b_feature_columns = _resolved_feature_columns(
        labeled_all,
        merged_store,
        sample_label_aware_labeled,
    )
    del sample_label_aware, sample_label_aware_labeled
    gc.collect()
    split_path = feature_path(split_artifact_name, cutoff_tag)
    primary_split.to_parquet(split_path, index=False)
    for fold_id in sorted(int(value) for value in primary_split[fold_column].dropna().unique()):
        output_path = feature_path(f"{split_artifact_name}_fold_{fold_id}_oof", cutoff_tag)
        job_path = feature_path(f"{split_artifact_name}_fold_{fold_id}_job", cutoff_tag).with_suffix(".json")
        job_config = {
            "cutoff_tag": cutoff_tag,
            "split_path": str(split_path),
            "fold_column": fold_column,
            "fold_id": fold_id,
            "output_path": str(output_path),
            "base_a_feature_columns": base_a_feature_columns,
            "base_b_feature_columns": base_b_feature_columns,
        }
        job_path.write_text(json.dumps(job_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        subprocess.run(
            [sys.executable, "-m", "transductive_v1.fold_worker", "--job-config", str(job_path)],
            cwd=str(Path(__file__).resolve().parents[1]),
            check=True,
        )
        fold_frame = pd.read_parquet(output_path)
        rows.append(fold_frame)
        fold_summaries.append(
            {
                "primary_fold": int(fold_id),
                "train_rows": int(primary_split[primary_split[fold_column] != fold_id]["user_id"].nunique()),
                "valid_rows": int(primary_split[primary_split[fold_column] == fold_id]["user_id"].nunique()),
                "positive_count": int(fold_frame["status"].astype(int).eq(1).sum()),
                "negative_count": int(fold_frame["status"].astype(int).eq(0).sum()),
            }
        )
        if output_path.exists():
            output_path.unlink()
        if job_path.exists():
            job_path.unlink()
        gc.collect()
    oof = pd.concat(rows, ignore_index=True).sort_values("user_id").reset_index(drop=True)
    return oof, fold_summaries, base_a_feature_columns, base_b_feature_columns


def _score_full_users(
    label_free_frame: pd.DataFrame,
    graph_store: GraphStore,
    label_frame: pd.DataFrame,
    base_a_feature_columns: list[str],
    base_b_feature_columns: list[str],
) -> tuple[pd.DataFrame, Any, Any]:
    label_aware = build_label_aware_features(graph_store, label_frame)
    training_frame = _merge_training_frame(label_free_frame, graph_store, label_aware)
    labeled_train = training_frame[training_frame["status"].notna()].copy()
    base_a_fit = fit_catboost(labeled_train, None, base_a_feature_columns)
    base_b_fit = fit_catboost(labeled_train, None, base_b_feature_columns)
    scoring = training_frame[[
        "user_id",
        "status",
        "needs_prediction",
        "rule_score",
        "anomaly_score",
        "graph_risk_score",
        "projected_component_log_size",
        "connected_flag",
        "top_reason_codes",
        "is_shadow_overlap",
    ]].copy()
    base_a_x = training_frame[base_a_feature_columns].astype("float32")
    base_b_x = training_frame[base_b_feature_columns].astype("float32")
    scoring["base_a_probability"] = base_a_fit.model.predict_proba(base_a_x)[:, 1]
    scoring["base_b_probability"] = base_b_fit.model.predict_proba(base_b_x)[:, 1]
    return scoring, base_a_fit.model, base_b_fit.model


def train_transductive_v1(cutoff_tag: str = "full") -> dict[str, Any]:
    label_free_frame = build_label_free_user_features(cutoff_tag=cutoff_tag, write_outputs=True)
    graph_store = build_graph_store(label_free_frame["user_id"].astype(int).tolist(), cutoff_tag=cutoff_tag, write_outputs=True)
    primary_split = build_primary_split(_labeled_frame(label_free_frame), cutoff_tag=cutoff_tag, write_outputs=True)
    base_oof, fold_summaries, base_a_feature_columns, base_b_feature_columns = run_primary_base_oof(
        label_free_frame,
        graph_store,
        primary_split,
        cutoff_tag=cutoff_tag,
        split_artifact_name="primary_split",
        fold_column="primary_fold",
    )
    base_oof.to_parquet(feature_path("primary_base_oof", cutoff_tag), index=False)
    stack_oof, stacker_model, stacker_feature_columns = build_stacker_oof(base_oof, fold_column="primary_fold")
    stack_oof.to_parquet(feature_path("primary_stack_oof", cutoff_tag), index=False)

    label_frame = _labeled_frame(label_free_frame)[["user_id", "status"]].copy()
    full_scoring_frame, base_a_model, base_b_model = _score_full_users(
        label_free_frame,
        graph_store,
        label_frame,
        base_a_feature_columns,
        base_b_feature_columns,
    )
    full_scoring_frame.to_parquet(feature_path("full_scoring_frame", cutoff_tag), index=False)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_a_path = model_path(f"base_a_catboost_{timestamp}.pkl")
    base_b_path = model_path(f"base_b_catboost_{timestamp}.pkl")
    stacker_path = model_path(f"stacker_{timestamp}.pkl")
    save_pickle(base_a_model, base_a_path)
    save_pickle(base_b_model, base_b_path)
    save_pickle(stacker_model, stacker_path)

    bundle = {
        "bundle_version": f"transductive_v1_{timestamp}",
        "selected_model": "stacked_transductive_v1",
        "primary_validation_protocol": {
            "mode": "label_mask_transductive_cv",
            "n_splits": PRIMARY_N_SPLITS,
            "random_state": RANDOM_SEED,
        },
        "base_model_paths": {
            "base_a_catboost": str(base_a_path),
            "base_b_catboost": str(base_b_path),
        },
        "stacker_path": str(stacker_path),
        "feature_columns_base_a": base_a_feature_columns,
        "feature_columns_base_b": base_b_feature_columns,
        "stacker_feature_columns": stacker_feature_columns,
        "oof_predictions_path": str(feature_path("primary_stack_oof", cutoff_tag)),
        "primary_split_path": str(feature_path("primary_split", cutoff_tag)),
        "calibrator": None,
        "decision_rule": None,
        "selected_threshold": None,
        "secondary_stress_summary": None,
    }
    save_json(bundle, bundle_path())
    meta = {
        "bundle_version": bundle["bundle_version"],
        "fold_summaries": fold_summaries,
        "primary_base_oof_path": str(feature_path("primary_base_oof", cutoff_tag)),
        "primary_stack_oof_path": str(feature_path("primary_stack_oof", cutoff_tag)),
        "full_scoring_frame_path": str(feature_path("full_scoring_frame", cutoff_tag)),
    }
    save_json(meta, report_path("train_meta.json"))
    return {
        "bundle_path": str(bundle_path()),
        "base_a_model_path": str(base_a_path),
        "base_b_model_path": str(base_b_path),
        "stacker_path": str(stacker_path),
        "primary_stack_oof_path": str(feature_path("primary_stack_oof", cutoff_tag)),
    }
