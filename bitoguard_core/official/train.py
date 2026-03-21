from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from official.anomaly import build_official_anomaly_features
from official.bundle import save_selected_bundle
from official.common import RANDOM_SEED, feature_output_path, load_official_paths, save_json, save_pickle
from official.correct_and_smooth import correct_and_smooth
from official.features import build_official_features
from official.graph_dataset import TransductiveGraph, build_transductive_graph
from official.graph_features import build_official_graph_features
from official.graph_model import save_graph_model, train_graphsage_model
from official.modeling import fit_catboost, fit_lgbm
from official.modeling_xgb import fit_xgboost
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

import os as _os
# Multi-seed ensembles — averaging reduces prediction variance by 1/sqrt(n).
# Set ABLATION_FAST=1 to use reduced seeds for quick screening (~300s/exp saved).
_FAST = _os.environ.get("ABLATION_FAST", "0") == "1"
_BASE_A_SEEDS = [42, 52] if _FAST else [42, 52, 62, 72]    # CatBoost: 2 (fast) / 4 (full)
_BASE_D_SEEDS = [42] if _FAST else [42, 123, 456]           # LightGBM: 1 (fast) / 3 (full)
_BASE_E_SEEDS = [42] if _FAST else [42, 123]                # XGBoost:  1 (fast) / 2 (full)

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
    non_null = {col for col in dataset.columns if not dataset[col].isna().all()}
    return [column for column in dataset.columns if column not in LABEL_FREE_EXCLUDED_COLUMNS and column in non_null]


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
    catboost_params: dict | None = None,
    use_negative_propagation: bool = False,
    cs_restore_top_pct: float = 0.0,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    label_frame = _label_frame(dataset)
    assignments = iter_fold_assignments(split_frame, fold_column)
    rows: list[pd.DataFrame] = []
    fold_training_meta: list[dict[str, Any]] = []
    for fold_id, train_users, valid_users in assignments:
        fold_train_labels = label_frame[label_frame["user_id"].astype(int).isin(train_users)].copy()
        transductive_features = build_transductive_feature_frame(
            graph, fold_train_labels,
            use_negative_propagation=use_negative_propagation,
        )
        label_free_frame, with_transductive_frame = _prepare_base_frames(dataset, transductive_features)

        train_label_free = label_free_frame[label_free_frame["user_id"].astype(int).isin(train_users)].copy()
        valid_label_free = label_free_frame[label_free_frame["user_id"].astype(int).isin(valid_users)].copy()
        train_transductive = with_transductive_frame[with_transductive_frame["user_id"].astype(int).isin(train_users)].copy()
        valid_transductive = with_transductive_frame[with_transductive_frame["user_id"].astype(int).isin(valid_users)].copy()

        # Multi-seed CatBoost ensemble for Base A (4 seeds, reduces variance ~50%).
        _base_a_val_probs = []
        _base_a_models = []
        for _seed in _BASE_A_SEEDS:
            _fit = fit_catboost(train_label_free, valid_label_free, base_a_feature_columns, focal_gamma=2.0, random_seed=_seed, catboost_params=catboost_params)
            _base_a_val_probs.append(_fit.validation_probabilities)
            _base_a_models.append(_fit.model)
        base_a_fit = type(_fit)(
            model_name=_fit.model_name,
            model=_base_a_models[0],
            feature_columns=_fit.feature_columns,
            encoded_columns=_fit.encoded_columns,
            cat_features=_fit.cat_features,
            validation_probabilities=np.mean(_base_a_val_probs, axis=0).tolist(),
        )

        resolved_base_b_columns = base_b_feature_columns or [column for column in train_transductive.columns if column != "user_id"]
        _base_b_params = {"task_type": "CPU", "l2_leaf_reg": 5.0}
        base_b_fit = fit_catboost(train_transductive, valid_transductive, resolved_base_b_columns, focal_gamma=2.0, catboost_params=_base_b_params)

        # Multi-seed LightGBM ensemble for Base D (3 seeds).
        _base_d_val_probs = []
        _base_d_fit = None
        for _seed_d in _BASE_D_SEEDS:
            _base_d_fit = fit_lgbm(train_label_free, valid_label_free, base_a_feature_columns, random_seed=_seed_d)
            _base_d_val_probs.append(_base_d_fit.validation_probabilities)
        base_d_fit = type(_base_d_fit)(
            model_name=_base_d_fit.model_name,
            model=_base_d_fit.model,
            feature_columns=_base_d_fit.feature_columns,
            encoded_columns=_base_d_fit.encoded_columns,
            cat_features=_base_d_fit.cat_features,
            validation_probabilities=np.mean(_base_d_val_probs, axis=0).tolist(),
        )

        # Multi-seed XGBoost ensemble for Base E (2 seeds).
        _base_e_val_probs = []
        _base_e_fit = None
        for _seed_e in _BASE_E_SEEDS:
            _base_e_fit = fit_xgboost(train_label_free, valid_label_free, base_a_feature_columns, random_seed=_seed_e)
            _base_e_val_probs.append(_base_e_fit.validation_probabilities)
        base_e_fit = type(_base_e_fit)(
            model_name=_base_e_fit.model_name,
            model=_base_e_fit.model,
            feature_columns=_base_e_fit.feature_columns,
            encoded_columns=_base_e_fit.encoded_columns,
            cat_features=_base_e_fit.cat_features,
            validation_probabilities=np.mean(_base_e_val_probs, axis=0).tolist(),
        )

        # Sanity checks
        if base_a_fit.validation_probabilities:
            va = np.array(base_a_fit.validation_probabilities)
            if va.mean() < 1e-4 or va.max() < 0.01:
                import warnings
                warnings.warn(f"Base A fold {fold_id}: probabilities collapsed (mean={va.mean():.6f}, max={va.max():.6f})")

        if base_b_fit.validation_probabilities:
            vb = np.array(base_b_fit.validation_probabilities)
            if vb.mean() < 1e-4 or vb.max() < 0.01:
                import warnings
                warnings.warn(f"Base B fold {fold_id}: probabilities collapsed (mean={vb.mean():.6f}, max={vb.max():.6f})")

        import os as _gnn_os
        if _gnn_os.environ.get("SKIP_GNN", "0") == "1":
            from official.graph_model import GraphModelFitResult as _GFR
            graph_fit = _GFR(
                model_state={"metadata": {"user_ids": graph.user_ids, "max_epochs": 0, "hidden_dim": 128, "user_feature_columns": [], "best_epoch": 0}},
                model_meta={"user_ids": graph.user_ids, "max_epochs": 0, "hidden_dim": 128, "user_feature_columns": [], "best_epoch": 0},
                full_probabilities=np.zeros(len(graph.user_ids), dtype=np.float32),
                validation_probabilities=np.zeros(len(valid_users)),
            )
        else:
            graph_fit = train_graphsage_model(
                graph,
                label_frame=label_frame,
                train_user_ids=train_users,
                valid_user_ids=valid_users,
                max_epochs=graph_max_epochs,
                hidden_dim=128,
            )
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass

        # Correct-and-Smooth (C&S): graph post-processing on Base A OOF probs.
        _train_a_probs = np.mean(
            [m.predict_proba(train_label_free[base_a_feature_columns])[:, 1] for m in _base_a_models],
            axis=0,
        )
        _val_a_probs = np.asarray(base_a_fit.validation_probabilities, dtype=float)
        _cs_base_probs: dict[int, float] = {}
        for _uid, _prob in zip(train_label_free["user_id"].astype(int), _train_a_probs):
            _cs_base_probs[int(_uid)] = float(_prob)
        for _uid, _prob in zip(valid_label_free["user_id"].astype(int), _val_a_probs):
            _cs_base_probs[int(_uid)] = float(_prob)
        # Include unlabeled users in C&S base_probs so their fraud signal propagates.
        _all_labeled_ids = set(train_users) | set(valid_users)
        _unlabeled_frame = label_free_frame[~label_free_frame["user_id"].astype(int).isin(_all_labeled_ids)]
        if len(_unlabeled_frame) > 0:
            import os as _uf_os
            if _uf_os.environ.get("ABLATION_FAST", "0") == "1":
                # Fast mode: single model for C&S init (smoothing averages out noise).
                _unlabeled_a_probs = _base_a_models[0].predict_proba(_unlabeled_frame[base_a_feature_columns])[:, 1]
            else:
                _unlabeled_a_probs = np.mean(
                    [m.predict_proba(_unlabeled_frame[base_a_feature_columns])[:, 1] for m in _base_a_models],
                    axis=0,
                )
            for _uid, _prob in zip(_unlabeled_frame["user_id"].astype(int), _unlabeled_a_probs):
                _cs_base_probs[int(_uid)] = float(_prob)
        _cs_train_labels: dict[int, float] = dict(zip(
            fold_train_labels["user_id"].astype(int),
            fold_train_labels["status"].astype(float),
        ))
        import os as _cs_os
        _cs_fast = _cs_os.environ.get("ABLATION_FAST", "0") == "1"
        _cs_result = correct_and_smooth(
            graph, _cs_train_labels, _cs_base_probs,
            alpha_correct=0.5, alpha_smooth=0.5,
            n_correct_iter=30 if _cs_fast else 50,
            n_smooth_iter=30 if _cs_fast else 50,
            restore_isolated_top_pct=cs_restore_top_pct,
        )
        _val_ids = valid_label_free["user_id"].astype(int).tolist()
        _cs_val_probs = np.array(
            [_cs_result.get(int(_uid), float(_p)) for _uid, _p in zip(_val_ids, _val_a_probs)],
            dtype=float,
        )

        fold_frame = valid_label_free[["user_id", "status"]].copy()
        fold_frame[fold_column] = fold_id
        fold_frame["base_a_probability"] = _val_a_probs
        fold_frame["base_c_s_probability"] = _cs_val_probs
        fold_frame["base_b_probability"] = np.asarray(base_b_fit.validation_probabilities, dtype=float)
        fold_frame["base_c_probability"] = np.asarray(graph_fit.validation_probabilities, dtype=float)
        fold_frame["base_d_probability"] = np.asarray(base_d_fit.validation_probabilities, dtype=float)
        fold_frame["base_e_probability"] = np.asarray(base_e_fit.validation_probabilities, dtype=float)
        fold_frame["rule_score"] = pd.to_numeric(valid_label_free["rule_score"], errors="coerce").fillna(0.0).to_numpy() if "rule_score" in valid_label_free.columns else np.zeros(len(valid_label_free))
        fold_frame["anomaly_score"] = pd.to_numeric(valid_label_free["anomaly_score"], errors="coerce").fillna(0.0).to_numpy() if "anomaly_score" in valid_label_free.columns else np.zeros(len(valid_label_free))
        fold_frame["crypto_anomaly_score"] = pd.to_numeric(valid_label_free["crypto_anomaly_score"], errors="coerce").fillna(0.0).to_numpy() if "crypto_anomaly_score" in valid_label_free.columns else np.zeros(len(valid_label_free))
        fold_frame["anomaly_score_segmented"] = pd.to_numeric(valid_label_free["anomaly_score_segmented"], errors="coerce").fillna(0.0).to_numpy() if "anomaly_score_segmented" in valid_label_free.columns else np.zeros(len(valid_label_free))
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
    primary_oof, stacker_model = build_stacker_oof(primary_oof, primary_split, fold_column="primary_fold", use_blend=True)
    primary_oof.to_parquet(artifacts["oof"], index=False)

    label_frame = _label_frame(dataset)
    full_transductive = build_transductive_feature_frame(graph, label_frame)
    label_free_frame, with_transductive_frame = _prepare_base_frames(dataset, full_transductive)
    labeled_user_ids = set(label_frame["user_id"].astype(int).tolist())
    train_label_free = label_free_frame[label_free_frame["user_id"].astype(int).isin(labeled_user_ids)].copy()
    train_transductive = with_transductive_frame[with_transductive_frame["user_id"].astype(int).isin(labeled_user_ids)].copy()

    # Multi-seed final models
    _base_a_final_models = [
        fit_catboost(train_label_free, None, base_a_feature_columns, focal_gamma=2.0, random_seed=_seed)
        for _seed in _BASE_A_SEEDS
    ]
    base_a_final = _base_a_final_models[0]
    _base_b_final_params = {"task_type": "CPU", "l2_leaf_reg": 5.0}
    base_b_final = fit_catboost(train_transductive, None, base_b_feature_columns, focal_gamma=2.0, catboost_params=_base_b_final_params)
    _base_d_finals = [fit_lgbm(train_label_free, None, base_a_feature_columns, random_seed=s) for s in _BASE_D_SEEDS]
    base_d_final = _base_d_finals[0]
    _base_e_finals = [fit_xgboost(train_label_free, None, base_a_feature_columns, random_seed=s) for s in _BASE_E_SEEDS]
    base_e_final = _base_e_finals[0]

    graph_epochs = int(np.median([item["graph_best_epoch"] + 1 for item in fold_training_meta])) if fold_training_meta else 40
    import os as _gnn_os2
    if _gnn_os2.environ.get("SKIP_GNN", "0") == "1":
        from official.graph_model import GraphModelFitResult as _GFR2
        graph_final = _GFR2(
            model_state={"metadata": {"user_ids": graph.user_ids, "max_epochs": 0, "hidden_dim": 128, "user_feature_columns": [], "best_epoch": 0}},
            model_meta={"user_ids": graph.user_ids, "max_epochs": 0, "hidden_dim": 128, "user_feature_columns": [], "best_epoch": 0},
            full_probabilities=np.zeros(len(graph.user_ids), dtype=np.float32),
            validation_probabilities=None,
        )
    else:
        graph_final = train_graphsage_model(
            graph,
            label_frame=label_frame,
            train_user_ids=labeled_user_ids,
            valid_user_ids=None,
            max_epochs=max(FINAL_GRAPH_MIN_EPOCHS, graph_epochs),
            hidden_dim=128,
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_a_paths = [paths.model_dir / f"official_catboost_base_a_seed{seed}_{timestamp}.pkl" for seed in _BASE_A_SEEDS]
    base_a_path = base_a_paths[0]
    base_b_path = paths.model_dir / f"official_catboost_base_b_{timestamp}.pkl"
    base_d_paths = [paths.model_dir / f"official_lgbm_base_d_seed{seed}_{timestamp}.pkl" for seed in _BASE_D_SEEDS]
    base_d_path = base_d_paths[0]
    base_e_paths = [paths.model_dir / f"official_xgboost_base_e_seed{seed}_{timestamp}.pkl" for seed in _BASE_E_SEEDS]
    base_e_path = base_e_paths[0]
    graph_path = paths.model_dir / f"official_graphsage_{timestamp}.pt"
    stacker_path = paths.model_dir / f"official_stacker_{timestamp}.pkl"
    meta_path = paths.model_dir / f"official_transductive_{timestamp}.json"

    for _fit, _path in zip(_base_a_final_models, base_a_paths):
        save_pickle(_fit.model, _path)
    save_pickle(base_b_final.model, base_b_path)
    for _fit, _path in zip(_base_d_finals, base_d_paths):
        save_pickle(_fit.model, _path)
    for _fit, _path in zip(_base_e_finals, base_e_paths):
        save_pickle(_fit.model, _path)
    save_graph_model(graph_final.model_state, graph_path)
    save_stacker_model(stacker_model, stacker_path)

    from official.stacking import BlendEnsemble as _BlendEnsemble
    blend_weights = stacker_model.weights if isinstance(stacker_model, _BlendEnsemble) else None

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
        "blend_weights": blend_weights,
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
            "base_a_catboost": str(base_a_path.relative_to(paths.artifact_dir)),
            "base_a_catboost_seeds": [str(p.relative_to(paths.artifact_dir)) for p in base_a_paths],
            "base_b_catboost": str(base_b_path.relative_to(paths.artifact_dir)),
            "base_d_lgbm": str(base_d_path.relative_to(paths.artifact_dir)),
            "base_d_lgbm_seeds": [str(p.relative_to(paths.artifact_dir)) for p in base_d_paths],
            "base_e_xgboost": str(base_e_path.relative_to(paths.artifact_dir)),
            "base_e_xgboost_seeds": [str(p.relative_to(paths.artifact_dir)) for p in base_e_paths],
        },
        "graph_model_path": str(graph_path.relative_to(paths.artifact_dir)),
        "stacker_path": str(stacker_path.relative_to(paths.artifact_dir)),
        "stacker_feature_columns": STACKER_FEATURE_COLUMNS,
        "blend_weights": blend_weights,
        "feature_columns_base_a": base_a_feature_columns,
        "feature_columns_base_b": base_b_feature_columns,
        "feature_columns_base_d": base_a_feature_columns,
        "feature_columns_base_e": base_a_feature_columns,
        "encoded_columns_base_d": base_d_final.encoded_columns or [],
        "encoded_columns_base_e": base_e_final.encoded_columns or [],
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
        "base_d_model_path": str(base_d_path),
        "base_e_model_path": str(base_e_path),
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
