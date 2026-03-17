from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from official.bundle import load_selected_bundle, save_selected_bundle
from official.cohorts import build_official_cohorts, build_official_data_contract_report
from official.features import build_official_features
from official.graph_dataset import build_transductive_graph
from official.graph_features import build_official_graph_features
from official.score import score_official_predict
from official.splitters import build_split_artifacts
from official.stacking import choose_best_calibration_and_threshold
from official.thresholding import search_threshold
from official.train import _load_dataset, train_official_model
from official.transductive_features import build_transductive_feature_frame
from official.transductive_validation import build_primary_transductive_splits, build_secondary_strict_splits
from official.validate import validate_official_model


RAW_TABLES = (
    "user_info",
    "train_label",
    "predict_label",
    "twd_transfer",
    "crypto_transfer",
    "usdt_swap",
    "usdt_twd_trading",
)


def _spread_sample(series: pd.Series, count: int) -> pd.Series:
    if len(series) <= count:
        return series
    positions = np.linspace(0, len(series) - 1, count, dtype=int)
    return series.iloc[positions.tolist()]


def _sample_users() -> dict[str, set[int]]:
    user_index = pd.read_parquet(Path(__file__).resolve().parents[2] / "data" / "aws_event" / "clean" / "user_index.parquet")
    train_only = user_index[user_index["status"].notna() & ~user_index["needs_prediction"]]
    train_positive = _spread_sample(train_only[train_only["status"] == 1]["user_id"].astype(int), 24)
    train_negative = _spread_sample(train_only[train_only["status"] == 0]["user_id"].astype(int), 96)
    predict_only = _spread_sample(user_index[user_index["status"].isna() & user_index["needs_prediction"]]["user_id"].astype(int), 12)
    return {
        "train_only": set(pd.concat([train_positive, train_negative]).tolist()),
        "predict_only": set(predict_only.tolist()),
    }


def _prepare_official_subset(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    source_root = Path(__file__).resolve().parents[2]
    raw_source = source_root / "data" / "aws_event" / "raw"
    clean_source = source_root / "data" / "aws_event" / "clean"
    raw_target = tmp_path / "raw"
    clean_target = tmp_path / "clean"
    raw_target.mkdir()
    clean_target.mkdir()

    cohorts = _sample_users()
    selected_users = set().union(*cohorts.values())
    for table_name in RAW_TABLES:
        raw_frame = pd.read_parquet(raw_source / f"{table_name}.parquet")
        clean_frame = pd.read_parquet(clean_source / f"{table_name}.parquet")
        if "user_id" in raw_frame.columns:
            raw_frame = raw_frame[raw_frame["user_id"].astype(int).isin(selected_users)].copy()
        if "user_id" in clean_frame.columns:
            clean_frame = clean_frame[clean_frame["user_id"].astype(int).isin(selected_users)].copy()
        raw_frame.to_parquet(raw_target / f"{table_name}.parquet", index=False)
        clean_frame.to_parquet(clean_target / f"{table_name}.parquet", index=False)

    clean_user_index = pd.read_parquet(clean_source / "user_index.parquet")
    clean_user_index = clean_user_index[clean_user_index["user_id"].astype(int).isin(selected_users)].copy()
    clean_user_index.to_parquet(clean_target / "user_index.parquet", index=False)

    artifact_dir = tmp_path / "artifacts"
    monkeypatch.setenv("BITOGUARD_AWS_EVENT_RAW_DIR", str(raw_target))
    monkeypatch.setenv("BITOGUARD_AWS_EVENT_CLEAN_DIR", str(clean_target))
    monkeypatch.setenv("BITOGUARD_ARTIFACT_DIR", str(artifact_dir))
    return clean_target, artifact_dir


def _inject_graph_edges(clean_target: Path) -> dict[str, list[int]]:
    train_users = sorted(pd.read_parquet(clean_target / "train_label.parquet")["user_id"].astype(int).tolist())
    predict_users = sorted(pd.read_parquet(clean_target / "predict_label.parquet")["user_id"].astype(int).tolist())
    relation_users = train_users[0:2]
    wallet_users = train_users[2:6]
    ip_users = train_users[6:12]
    predict_bridge = predict_users[:2]

    crypto = pd.read_parquet(clean_target / "crypto_transfer.parquet")
    twd = pd.read_parquet(clean_target / "twd_transfer.parquet")
    crypto_template = crypto.iloc[0].copy()
    twd_template = twd.iloc[0].copy()
    next_crypto_id = int(pd.to_numeric(crypto["id"], errors="coerce").max()) + 1000
    next_twd_id = int(pd.to_numeric(twd["id"], errors="coerce").max()) + 1000

    crypto_rows = []
    relation_row = crypto_template.copy()
    relation_row["id"] = next_crypto_id
    relation_row["user_id"] = relation_users[0]
    relation_row["relation_user_id"] = relation_users[1]
    relation_row["sub_kind"] = 1
    relation_row["sub_kind_label"] = "internal"
    relation_row["is_internal_transfer"] = True
    relation_row["is_external_transfer"] = False
    relation_row["from_wallet_hash"] = "relation_wallet"
    relation_row["to_wallet_hash"] = "relation_wallet"
    relation_row["source_ip_hash"] = "relation_ip"
    crypto_rows.append(relation_row)
    next_crypto_id += 1

    for user_id in wallet_users + predict_bridge:
        row = crypto_template.copy()
        row["id"] = next_crypto_id
        row["user_id"] = user_id
        row["relation_user_id"] = pd.NA
        row["sub_kind"] = 0
        row["sub_kind_label"] = "external"
        row["is_internal_transfer"] = False
        row["is_external_transfer"] = True
        row["from_wallet_hash"] = "shared_wallet_entity"
        row["to_wallet_hash"] = "shared_wallet_entity"
        row["source_ip_hash"] = f"wallet_ip_{user_id}"
        crypto_rows.append(row)
        next_crypto_id += 1
    crypto = pd.concat([crypto, pd.DataFrame(crypto_rows)], ignore_index=True)
    crypto.to_parquet(clean_target / "crypto_transfer.parquet", index=False)

    twd_rows = []
    for user_id in ip_users:
        row = twd_template.copy()
        row["id"] = next_twd_id
        row["user_id"] = user_id
        row["source_ip_hash"] = "shared_ip_entity"
        twd_rows.append(row)
        next_twd_id += 1
    twd = pd.concat([twd, pd.DataFrame(twd_rows)], ignore_index=True)
    twd.to_parquet(clean_target / "twd_transfer.parquet", index=False)
    return {
        "relation": relation_users,
        "wallet": wallet_users,
        "ip": ip_users,
        "predict_bridge": predict_bridge,
    }


def _set_fast_graph_epochs(monkeypatch, epochs: int = 2) -> None:
    import official.train as train_module
    import official.validate as validate_module

    monkeypatch.setattr(train_module, "PRIMARY_GRAPH_MAX_EPOCHS", epochs)
    monkeypatch.setattr(train_module, "FINAL_GRAPH_MIN_EPOCHS", epochs)
    monkeypatch.setattr(validate_module, "PRIMARY_GRAPH_MAX_EPOCHS", epochs)


def test_data_contract_and_primary_split_are_labeled_only(tmp_path: Path, monkeypatch) -> None:
    clean_target, artifact_dir = _prepare_official_subset(tmp_path, monkeypatch)
    build_official_cohorts()
    report = build_official_data_contract_report()
    assert report["cohort_counts"]["shadow_overlap"] == 0
    assert (artifact_dir / "reports" / "official_data_contract_report.json").exists()

    features = build_official_features()
    split = build_primary_transductive_splits(features, write_outputs=False)
    train_rows = len(pd.read_parquet(clean_target / "train_label.parquet"))
    predict_rows = len(pd.read_parquet(clean_target / "predict_label.parquet"))
    assert len(split) == train_rows
    assert split["primary_fold"].nunique() == 5
    assert train_rows > predict_rows


def test_transductive_features_mask_validation_labels(tmp_path: Path, monkeypatch) -> None:
    clean_target, _ = _prepare_official_subset(tmp_path, monkeypatch)
    injected = _inject_graph_edges(clean_target)
    dataset = _load_dataset("full")
    graph = build_transductive_graph(dataset)

    full_labels = dataset[dataset["status"].notna()][["user_id", "status"]].copy()
    masked_labels = full_labels[full_labels["user_id"].astype(int) != injected["relation"][0]].copy()

    full_features = build_transductive_feature_frame(graph, full_labels)
    masked_features = build_transductive_feature_frame(graph, masked_labels)
    target_user = injected["relation"][1]
    full_row = full_features[full_features["user_id"].astype(int) == target_user].iloc[0]
    masked_row = masked_features[masked_features["user_id"].astype(int) == target_user].iloc[0]
    assert full_row["relation_positive_seed_count"] >= masked_row["relation_positive_seed_count"]
    assert full_row["positive_any_neighbor_count"] >= masked_row["positive_any_neighbor_count"]


def test_secondary_split_keeps_hard_components_together(tmp_path: Path, monkeypatch) -> None:
    clean_target, _ = _prepare_official_subset(tmp_path, monkeypatch)
    injected = _inject_graph_edges(clean_target)
    dataset = _load_dataset("full")
    secondary = build_secondary_strict_splits(
        dataset,
        write_outputs=False,
        params={"n_splits": 3, "split_seed_candidates": (42,), "min_positive_per_fold": 1},
    )
    relation_folds = secondary[secondary["user_id"].astype(int).isin(injected["relation"])]["secondary_fold"].nunique()
    wallet_folds = secondary[secondary["user_id"].astype(int).isin(injected["wallet"])]["secondary_fold"].nunique()
    assert relation_folds == 1
    assert wallet_folds == 1


def test_bundle_roundtrip_and_threshold_selection(tmp_path: Path, monkeypatch) -> None:
    _, artifact_dir = _prepare_official_subset(tmp_path, monkeypatch)
    bundle = {
        "bundle_version": "bundle_test",
        "selected_model": "stacked_transductive",
        "primary_validation_protocol": {"mode": "label_mask_transductive_cv"},
        "base_model_paths": {"base_a_catboost": "a.pkl", "base_b_catboost": "b.pkl"},
        "graph_model_path": "graph.pt",
        "stacker_path": "stacker.pkl",
        "shadow_protocol": {"mode": "secondary_only"},
        "grouping_params": {},
        "calibrator": None,
        "selected_threshold": None,
    }
    bundle_path = save_selected_bundle(bundle)
    loaded = load_selected_bundle(bundle_path, require_ready=False)
    assert loaded["bundle_version"] == "bundle_test"
    with pytest.raises(ValueError):
        load_selected_bundle(bundle_path, require_ready=True)
    assert bundle_path == artifact_dir / "official_bundle.json"

    probabilities = np.array([0.05, 0.10, 0.22, 0.35, 0.40, 0.68, 0.82, 0.91], dtype=float)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    report, calibrator, calibrated = choose_best_calibration_and_threshold(probabilities, labels, np.array([1, 1, 2, 2, 3, 3, 4, 4]))
    assert report["method"] in {"raw", "sigmoid", "beta", "isotonic"}
    assert 0.0 < report["selected_threshold"] < 1.0
    assert calibrated.shape == probabilities.shape


def test_official_train_validate_and_score_subset(tmp_path: Path, monkeypatch) -> None:
    clean_target, artifact_dir = _prepare_official_subset(tmp_path, monkeypatch)
    _inject_graph_edges(clean_target)
    _set_fast_graph_epochs(monkeypatch, epochs=2)

    train_meta = train_official_model()
    bundle = load_selected_bundle(require_ready=False)
    assert train_meta["bundle_path"] == str(artifact_dir / "official_bundle.json")
    oof = pd.read_parquet(bundle["oof_predictions_path"])
    assert len(oof) == len(pd.read_parquet(clean_target / "train_label.parquet"))
    assert {"base_a_probability", "base_b_probability", "base_c_probability", "stacker_raw_probability"} <= set(oof.columns)

    import official.validate as validate_module

    secondary_template = oof[["user_id", "status", "primary_fold", *validate_module.STACKER_FEATURE_COLUMNS, "stacker_raw_probability"]].copy()
    secondary_template = secondary_template.rename(columns={"primary_fold": "secondary_fold"})

    def _fake_secondary_splits(dataset, cutoff_tag="full", write_outputs=True, params=None):
        return secondary_template[["user_id", "secondary_fold"]].copy()

    def _fake_secondary_oof(dataset, graph, split_frame, fold_column, base_a_feature_columns, base_b_feature_columns=None, graph_max_epochs=2):
        return secondary_template.copy(), []

    monkeypatch.setattr(validate_module, "build_secondary_strict_splits", _fake_secondary_splits)
    monkeypatch.setattr(validate_module, "run_transductive_oof_pipeline", _fake_secondary_oof)

    with pytest.raises(ValueError):
        score_official_predict()

    validation = validate_official_model()
    ready_bundle = load_selected_bundle(require_ready=True)
    assert ready_bundle["calibrator"]["method"] in {"raw", "sigmoid", "beta", "isotonic"}
    assert validation["primary_validation_protocol"]["mode"] == "label_mask_transductive_cv"
    assert "primary_transductive_oof_metrics" in validation
    assert "secondary_group_stress_metrics" in validation

    predictions = score_official_predict()
    prediction_path = artifact_dir / "predictions" / "official_predict_scores.parquet"
    assert prediction_path.exists()
    assert len(predictions) == len(pd.read_parquet(clean_target / "predict_label.parquet"))
    assert {
        "user_id",
        "submission_probability",
        "submission_pred",
        "stacker_raw_probability",
        "base_a_probability",
        "base_b_probability",
        "base_c_probability",
        "anomaly_score",
        "rule_score",
        "analyst_risk_score",
        "risk_rank",
        "risk_level",
        "top_reason_codes",
        "is_shadow_overlap",
    } <= set(predictions.columns)
