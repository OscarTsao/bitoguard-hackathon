from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from official.bundle import load_selected_bundle, save_selected_bundle
from official.calibration import choose_calibrator
from official.cohorts import build_official_cohorts, build_official_data_contract_report
from official.features import build_official_features
from official.score import score_official_predict
from official.splitters import build_split_artifacts
from official.thresholding import search_threshold
from official.train import train_official_model
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


def _sample_users() -> dict[str, set[int]]:
    user_index = pd.read_parquet(Path(__file__).resolve().parents[2] / "data" / "aws_event" / "clean" / "user_index.parquet")
    train_only = user_index[user_index["status"].notna() & ~user_index["needs_prediction"]]
    train_positive = train_only[train_only["status"] == 1]["user_id"].astype(int).head(24)
    train_negative = train_only[train_only["status"] == 0]["user_id"].astype(int).head(96)
    shadow_overlap = user_index[user_index["status"].notna() & user_index["needs_prediction"]]
    shadow_positive = shadow_overlap[shadow_overlap["status"] == 1]["user_id"].astype(int).head(16)
    shadow_negative = shadow_overlap[shadow_overlap["status"] == 0]["user_id"].astype(int).head(16)
    predict_only = user_index[user_index["status"].isna() & user_index["needs_prediction"]]["user_id"].astype(int).head(16)
    unlabeled_only = user_index[user_index["status"].isna() & ~user_index["needs_prediction"]]["user_id"].astype(int).head(16)
    return {
        "train_only": set(pd.concat([train_positive, train_negative]).tolist()),
        "shadow_overlap": set(pd.concat([shadow_positive, shadow_negative]).tolist()),
        "predict_only": set(predict_only.tolist()),
        "unlabeled_only": set(unlabeled_only.tolist()),
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


def test_official_data_contract_and_cohorts(tmp_path: Path, monkeypatch) -> None:
    clean_target, artifact_dir = _prepare_official_subset(tmp_path, monkeypatch)
    cohorts = build_official_cohorts()
    counts = cohorts["cohort"].value_counts().to_dict()
    assert counts["train_only"] == 120
    assert counts["shadow_overlap"] == 32
    assert counts["predict_only"] == 16
    assert counts["unlabeled_only"] == 16

    report = build_official_data_contract_report()
    assert report["cohort_counts"]["all_users"] == 184
    assert report["primary_key_checks"]["user_info"]["duplicate_primary_keys"] == 0
    assert report["user_integrity_checks"]["crypto_transfer"]["orphan_user_rows"] == 0
    assert (artifact_dir / "reports" / "official_data_contract_report.json").exists()
    assert len(pd.read_parquet(clean_target / "predict_label.parquet")) == 48


def test_official_features_preserve_profile_and_internal_relation_only(tmp_path: Path, monkeypatch) -> None:
    clean_target, _ = _prepare_official_subset(tmp_path, monkeypatch)
    crypto = pd.read_parquet(clean_target / "crypto_transfer.parquet")
    user_info = pd.read_parquet(clean_target / "user_info.parquet")
    sample_user = int(user_info[user_info["age"].notna()].iloc[0]["user_id"])
    expected_age = float(user_info[user_info["user_id"] == sample_user]["age"].iloc[0])
    internal_expected = int(
        crypto[
            (crypto["user_id"].astype(int) == sample_user)
            & crypto["relation_user_id"].notna()
            & crypto["is_internal_transfer"].eq(True)
        ].shape[0]
    )
    synthetic = crypto.iloc[0].copy()
    synthetic["id"] = int(crypto["id"].max()) + 1000
    synthetic["user_id"] = sample_user
    synthetic["relation_user_id"] = 999999
    synthetic["is_internal_transfer"] = False
    synthetic["sub_kind_label"] = "external"
    crypto = pd.concat([crypto, pd.DataFrame([synthetic])], ignore_index=True)
    crypto.to_parquet(clean_target / "crypto_transfer.parquet", index=False)

    features = build_official_features()
    row = features[features["user_id"].astype(int) == sample_user].iloc[0]
    assert row["age"] == expected_age
    assert int(row["relation_transfer_count"]) == internal_expected


def test_splitters_reserve_shadow_groups_and_keep_groups_within_single_fold(tmp_path: Path, monkeypatch) -> None:
    _prepare_official_subset(tmp_path, monkeypatch)
    dataset = build_official_features()
    group_index, purge_map, _ = build_split_artifacts(dataset, write_outputs=False)
    shadow_groups = group_index[group_index["is_shadow_overlap"].astype(bool)]["strong_group_id"].unique().tolist()
    assert shadow_groups
    assert all(
        group_index[group_index["strong_group_id"] == group_id]["group_role"].eq("shadow_reserved").all()
        for group_id in shadow_groups
    )
    core_labeled = group_index[(group_index["group_role"] == "core_trainable") & (group_index["status"].notna())]
    group_folds = core_labeled.groupby("strong_group_id")["core_fold"].nunique()
    assert (group_folds <= 1).all()
    assert isinstance(purge_map, dict)


def test_bundle_roundtrip_and_score_requires_ready_bundle(tmp_path: Path, monkeypatch) -> None:
    _, artifact_dir = _prepare_official_subset(tmp_path, monkeypatch)
    bundle = {
        "bundle_version": "bundle_test",
        "selected_model": "lgbm",
        "model_paths": {"lgbm": "missing.pkl"},
        "shadow_protocol": {"dev_ratio": 0.7, "holdout_ratio": 0.3},
        "grouping_params": {"max_strong_ip_users": 5},
        "calibrator": None,
        "selected_threshold": None,
    }
    bundle_path = save_selected_bundle(bundle)
    loaded = load_selected_bundle(bundle_path, require_ready=False)
    assert loaded["bundle_version"] == "bundle_test"
    with pytest.raises(ValueError):
        load_selected_bundle(bundle_path, require_ready=True)
    with pytest.raises(ValueError):
        score_official_predict()
    assert bundle_path == artifact_dir / "official_bundle.json"


def test_calibration_and_threshold_modules(tmp_path: Path, monkeypatch) -> None:
    _prepare_official_subset(tmp_path, monkeypatch)
    probabilities = pd.Series([0.05, 0.10, 0.22, 0.35, 0.40, 0.68, 0.82, 0.91]).to_numpy()
    labels = pd.Series([0, 0, 0, 0, 1, 1, 1, 1]).to_numpy()
    calibrator_report, calibrator = choose_calibrator(probabilities, labels)
    assert calibrator_report["method"] in {"sigmoid", "beta", "isotonic"}
    assert Path(calibrator_report["calibrator_path"]).exists()
    calibrated = calibrator.predict(probabilities)
    threshold_report = search_threshold(labels, calibrated, group_ids=pd.Series([1, 1, 2, 2, 3, 3, 4, 4]).to_numpy())
    assert 0.0 < threshold_report["selected_threshold"] < 1.0
    assert threshold_report["rows"]


def test_official_train_validate_and_score_subset(tmp_path: Path, monkeypatch) -> None:
    clean_target, artifact_dir = _prepare_official_subset(tmp_path, monkeypatch)
    train_meta = train_official_model()
    bundle = load_selected_bundle(require_ready=False)
    assert train_meta["bundle_path"] == str(artifact_dir / "official_bundle.json")
    assert Path(bundle["oof_predictions_path"]).exists()
    assert Path(bundle["shadow_predictions_path"]).exists()

    with pytest.raises(ValueError):
        score_official_predict()

    validation = validate_official_model()
    ready_bundle = load_selected_bundle(require_ready=True)
    assert ready_bundle["calibrator"]["method"] in {"sigmoid", "beta", "isotonic"}
    assert ready_bundle["selected_threshold"] == validation["selected_threshold"]
    assert "shadow_holdout_metrics" in validation
    assert "temporal_stress_test" in validation

    predictions = score_official_predict()
    prediction_path = artifact_dir / "predictions" / "official_predict_scores.parquet"
    assert prediction_path.exists()
    assert len(predictions) == len(pd.read_parquet(clean_target / "predict_label.parquet"))
    assert {
        "user_id",
        "submission_probability",
        "submission_pred",
        "model_probability_raw",
        "anomaly_score",
        "rule_score",
        "analyst_risk_score",
        "risk_rank",
        "risk_level",
        "top_reason_codes",
        "is_shadow_overlap",
    } <= set(predictions.columns)
