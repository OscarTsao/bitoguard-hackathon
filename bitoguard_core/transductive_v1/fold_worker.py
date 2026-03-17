from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Cap each worker's BLAS/OpenMP threads to avoid core oversubscription when
# multiple fold workers run in parallel.
_worker_threads = os.getenv("BITOGUARD_CPU_THREADS")
if _worker_threads:
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_var, _worker_threads)

import pandas as pd

from transductive_v1.common import feature_path
from transductive_v1.graph_store import load_graph_store
from transductive_v1.label_aware_features import build_label_aware_features
from transductive_v1.train import _feature_columns, _labeled_frame, _merge_training_frame
from transductive_v1.branch_tabular import fit_catboost


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one transductive_v1 CV fold in an isolated subprocess.")
    parser.add_argument("--job-config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    job = json.loads(Path(args.job_config).read_text(encoding="utf-8"))
    cutoff_tag = job["cutoff_tag"]
    split_path = Path(job["split_path"])
    fold_column = job["fold_column"]
    fold_id = int(job["fold_id"])
    output_path = Path(job["output_path"])
    base_a_feature_columns = job.get("base_a_feature_columns")
    base_b_feature_columns = job.get("base_b_feature_columns")

    label_free_frame = pd.read_parquet(feature_path("label_free_user_features", cutoff_tag))
    graph_store = load_graph_store(cutoff_tag)
    split_frame = pd.read_parquet(split_path)
    labeled_all = _labeled_frame(label_free_frame)
    labeled_user_ids = set(labeled_all["user_id"].astype(int).tolist())

    valid_users = set(split_frame[split_frame[fold_column] == fold_id]["user_id"].astype(int).tolist())
    train_users = labeled_user_ids.difference(valid_users)
    label_frame = labeled_all[["user_id", "status"]].copy()
    fold_train_labels = label_frame[label_frame["user_id"].astype(int).isin(train_users)].copy()
    label_aware = build_label_aware_features(graph_store, fold_train_labels)
    label_aware_labeled = label_aware[label_aware["user_id"].astype(int).isin(labeled_user_ids)].copy()
    graph_structural_labeled = graph_store.structural_features[graph_store.structural_features["user_id"].astype(int).isin(labeled_user_ids)].copy()
    merged_store = graph_store.__class__(
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
    training_frame = _merge_training_frame(labeled_all, merged_store, label_aware_labeled)
    train_frame = training_frame[training_frame["user_id"].astype(int).isin(train_users)].copy()
    valid_frame = training_frame[training_frame["user_id"].astype(int).isin(valid_users)].copy()
    base_a_feature_columns = base_a_feature_columns or _feature_columns(training_frame, include_label_aware=False)
    base_b_feature_columns = base_b_feature_columns or _feature_columns(training_frame, include_label_aware=True)
    base_a_fit = fit_catboost(train_frame, valid_frame, base_a_feature_columns)
    base_b_fit = fit_catboost(train_frame, valid_frame, base_b_feature_columns)

    fold_frame = valid_frame[[
        "user_id",
        "status",
        "rule_score",
        "anomaly_score",
        "graph_risk_score",
        "projected_component_log_size",
        "connected_flag",
    ]].copy()
    fold_frame[fold_column] = fold_id
    fold_frame["base_a_probability"] = pd.Series(base_a_fit.validation_probabilities, index=fold_frame.index, dtype=float)
    fold_frame["base_b_probability"] = pd.Series(base_b_fit.validation_probabilities, index=fold_frame.index, dtype=float)
    fold_frame.to_parquet(output_path, index=False)


if __name__ == "__main__":
    main()
