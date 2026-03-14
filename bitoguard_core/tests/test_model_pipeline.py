from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from db.store import DuckDBStore
from features.build_features import build_feature_snapshots
from features.graph_features import build_graph_features
from models.anomaly import train_anomaly_model
from models.common import forward_date_splits, training_dataset
from models.train import train_model
from models.validate import validate_model
from services.drift import detect_drift, FeatureDriftResult


def _configure_model_store(tmp_path: Path, monkeypatch) -> DuckDBStore:
    db_path = tmp_path / "bitoguard.duckdb"
    artifact_dir = tmp_path / "artifacts"
    monkeypatch.setenv("BITOGUARD_DB_PATH", str(db_path))
    monkeypatch.setenv("BITOGUARD_ARTIFACT_DIR", str(artifact_dir))
    return DuckDBStore(db_path)


def _seed_model_tables(store: DuckDBStore) -> None:
    snapshot_dates = pd.date_range("2026-01-01", periods=6, freq="D")
    feature_rows: list[dict[str, object]] = []
    for index, snapshot_date in enumerate(snapshot_dates, start=1):
        feature_rows.append({
            "feature_snapshot_id": f"neg_{index}",
            "user_id": "u_neg",
            "snapshot_date": snapshot_date,
            "feature_version": "test_v1",
            "txn_count_30d": float(index),
            "volume_twd_30d": float(index * 10),
            "segment_code": "retail",
        })
        feature_rows.append({
            "feature_snapshot_id": f"pos_{index}",
            "user_id": "u_pos",
            "snapshot_date": snapshot_date,
            "feature_version": "test_v1",
            "txn_count_30d": float(index * 3),
            "volume_twd_30d": float(index * 30),
            "segment_code": "linked",
        })

    store.replace_table("features.feature_snapshots_user_30d", pd.DataFrame(feature_rows))
    store.replace_table(
        "ops.oracle_user_labels",
        pd.DataFrame([
            {
                "user_id": "u_pos",
                "hidden_suspicious_label": 1,
                "observed_blacklist_label": 1,
                "scenario_types": "structured_ring",
                "evidence_tags": "",
            },
            {
                "user_id": "u_neg",
                "hidden_suspicious_label": 0,
                "observed_blacklist_label": 0,
                "scenario_types": None,
                "evidence_tags": "",
            },
        ]),
    )
    store.replace_table(
        "canonical.blacklist_feed",
        pd.DataFrame([
            {
                "blacklist_entry_id": "bl_1",
                "user_id": "u_pos",
                "observed_at": pd.Timestamp("2026-01-03T12:00:00Z"),
                "source": "oracle",
                "reason_code": "test_positive",
                "is_active": True,
            }
        ]),
    )


def _seed_snapshot_population_tables(store: DuckDBStore) -> None:
    store.replace_table(
        "canonical.users",
        pd.DataFrame([
            {
                "user_id": "u_stale",
                "created_at": pd.Timestamp("2025-01-01T00:00:00Z"),
                "kyc_level": "advanced",
                "occupation": "engineer",
                "monthly_income_twd": 120000.0,
                "expected_monthly_volume_twd": 80000.0,
                "declared_source_of_funds": "salary",
                "segment": "retail",
            },
            {
                "user_id": "u_recent",
                "created_at": pd.Timestamp("2025-01-01T00:00:00Z"),
                "kyc_level": "advanced",
                "occupation": "trader",
                "monthly_income_twd": 200000.0,
                "expected_monthly_volume_twd": 150000.0,
                "declared_source_of_funds": "trading",
                "segment": "retail",
            },
            {
                "user_id": "u_labeled",
                "created_at": pd.Timestamp("2025-01-01T00:00:00Z"),
                "kyc_level": "advanced",
                "occupation": "consultant",
                "monthly_income_twd": 90000.0,
                "expected_monthly_volume_twd": 60000.0,
                "declared_source_of_funds": "salary",
                "segment": "retail",
            },
        ]),
    )
    store.replace_table(
        "canonical.login_events",
        pd.DataFrame([
            {
                "login_event_id": "login_stale",
                "user_id": "u_stale",
                "occurred_at": pd.Timestamp("2026-01-05T10:00:00Z"),
                "ip_country": "TW",
                "is_geo_jump": False,
                "is_vpn": False,
                "is_new_device": False,
            },
            {
                "login_event_id": "login_recent",
                "user_id": "u_recent",
                "occurred_at": pd.Timestamp("2026-02-10T09:00:00Z"),
                "ip_country": "TW",
                "is_geo_jump": False,
                "is_vpn": False,
                "is_new_device": True,
            },
        ]),
    )
    store.replace_table(
        "canonical.blacklist_feed",
        pd.DataFrame([
            {
                "blacklist_entry_id": "bl_labeled",
                "user_id": "u_labeled",
                "observed_at": pd.Timestamp("2026-01-10T08:00:00Z"),
                "source": "oracle",
                "reason_code": "test_blacklist",
                "is_active": True,
            }
        ]),
    )
    store.replace_table(
        "canonical.entity_edges",
        pd.DataFrame([
            {
                "edge_id": "edge_stale",
                "snapshot_time": pd.Timestamp("2026-01-05T10:00:00Z"),
                "src_type": "user",
                "src_id": "u_stale",
                "relation_type": "uses_device",
                "dst_type": "device",
                "dst_id": "device_stale",
            },
            {
                "edge_id": "edge_recent",
                "snapshot_time": pd.Timestamp("2026-02-10T09:00:00Z"),
                "src_type": "user",
                "src_id": "u_recent",
                "relation_type": "uses_device",
                "dst_type": "device",
                "dst_id": "device_recent",
            },
        ]),
    )


def _seed_refresh_incremental_source_tables(store: DuckDBStore) -> None:
    store.replace_table(
        "canonical.users",
        pd.DataFrame([
            {
                "user_id": "u_direct",
                "created_at": pd.Timestamp("2025-01-01T00:00:00Z"),
                "kyc_level": "advanced",
                "occupation": "engineer",
                "monthly_income_twd": 120000.0,
                "expected_monthly_volume_twd": 100000.0,
                "declared_source_of_funds": "salary",
                "segment": "retail",
            },
            {
                "user_id": "u_neighbor",
                "created_at": pd.Timestamp("2025-01-01T00:00:00Z"),
                "kyc_level": "advanced",
                "occupation": "designer",
                "monthly_income_twd": 110000.0,
                "expected_monthly_volume_twd": 70000.0,
                "declared_source_of_funds": "salary",
                "segment": "retail",
            },
            {
                "user_id": "u_untouched",
                "created_at": pd.Timestamp("2025-01-01T00:00:00Z"),
                "kyc_level": "basic",
                "occupation": "teacher",
                "monthly_income_twd": 80000.0,
                "expected_monthly_volume_twd": 40000.0,
                "declared_source_of_funds": "salary",
                "segment": "retail",
            },
        ]),
    )
    store.replace_table(
        "canonical.login_events",
        pd.DataFrame([
            {
                "login_id": "login_direct",
                "user_id": "u_direct",
                "occurred_at": pd.Timestamp("2026-02-10T09:00:00Z"),
                "device_id": "device_shared",
                "ip_address": "1.1.1.1",
                "ip_country": "TW",
                "ip_city": "Taipei",
                "is_vpn": False,
                "is_new_device": True,
                "is_geo_jump": False,
                "success": True,
            },
            {
                "login_id": "login_untouched",
                "user_id": "u_untouched",
                "occurred_at": pd.Timestamp("2026-02-01T12:00:00Z"),
                "device_id": "device_other",
                "ip_address": "2.2.2.2",
                "ip_country": "TW",
                "ip_city": "Taichung",
                "is_vpn": False,
                "is_new_device": False,
                "is_geo_jump": False,
                "success": True,
            },
        ]),
    )
    store.replace_table(
        "canonical.fiat_transactions",
        pd.DataFrame([
            {
                "fiat_txn_id": "fiat_direct",
                "user_id": "u_direct",
                "occurred_at": pd.Timestamp("2026-02-10T08:00:00Z"),
                "direction": "deposit",
                "amount_twd": 100000.0,
                "currency": "TWD",
                "bank_account_id": "bank_direct",
                "method": "bank",
                "status": "completed",
            },
        ]),
    )
    store.replace_table(
        "canonical.trade_orders",
        pd.DataFrame([
            {
                "trade_id": "trade_historical",
                "user_id": "u_untouched",
                "occurred_at": pd.Timestamp("2026-01-01T00:00:00Z"),
                "side": "buy",
                "base_asset": "BTC",
                "quote_asset": "TWD",
                "price_twd": 1000.0,
                "quantity": 1.0,
                "notional_twd": 1000.0,
                "fee_twd": 0.0,
                "order_type": "market",
                "status": "completed",
            }
        ]),
    )
    store.replace_table(
        "canonical.crypto_transactions",
        pd.DataFrame([
            {
                "crypto_txn_id": "crypto_direct",
                "user_id": "u_direct",
                "occurred_at": pd.Timestamp("2026-02-10T08:30:00Z"),
                "direction": "withdrawal",
                "asset": "BTC",
                "network": "BTC",
                "wallet_id": "wallet_direct",
                "counterparty_wallet_id": "wallet_counterparty",
                "amount_asset": 0.1,
                "amount_twd_equiv": 50000.0,
                "tx_hash": "hash_direct",
                "status": "completed",
            },
        ]),
    )
    store.replace_table(
        "canonical.blacklist_feed",
        pd.DataFrame([
            {
                "blacklist_entry_id": "blacklist_historical",
                "user_id": "u_untouched",
                "observed_at": pd.Timestamp("2026-01-01T00:00:00Z"),
                "source": "oracle",
                "reason_code": "seed",
                "is_active": True,
            }
        ]),
    )
    store.replace_table(
        "canonical.entity_edges",
        pd.DataFrame([
            {
                "edge_id": "edge_direct_device",
                "snapshot_time": pd.Timestamp("2026-02-10T09:00:00Z"),
                "src_type": "user",
                "src_id": "u_direct",
                "relation_type": "uses_device",
                "dst_type": "device",
                "dst_id": "device_shared",
            },
            {
                "edge_id": "edge_neighbor_device",
                "snapshot_time": pd.Timestamp("2026-02-10T09:00:00Z"),
                "src_type": "user",
                "src_id": "u_neighbor",
                "relation_type": "uses_device",
                "dst_type": "device",
                "dst_id": "device_shared",
            },
            {
                "edge_id": "edge_untouched_device",
                "snapshot_time": pd.Timestamp("2026-02-01T12:00:00Z"),
                "src_type": "user",
                "src_id": "u_untouched",
                "relation_type": "uses_device",
                "dst_type": "device",
                "dst_id": "device_other",
            },
            {
                "edge_id": "edge_direct_wallet",
                "snapshot_time": pd.Timestamp("2026-02-10T08:30:00Z"),
                "src_type": "user",
                "src_id": "u_direct",
                "relation_type": "crypto_transfer_to_wallet",
                "dst_type": "wallet",
                "dst_id": "wallet_counterparty",
            },
        ]),
    )


def _seed_refresh_incremental_feature_tables(store: DuckDBStore) -> None:
    latest_snapshot = pd.Timestamp("2026-02-10")
    previous_snapshot = pd.Timestamp("2026-02-09")

    store.replace_table(
        "features.graph_features",
        pd.DataFrame([
            {
                "graph_feature_id": "gf_u_direct_2026-02-10",
                "user_id": "u_direct",
                "snapshot_date": latest_snapshot,
                "shared_device_count": 99,
                "shared_bank_count": 0,
                "shared_wallet_count": 0,
                "blacklist_1hop_count": 0,
                "blacklist_2hop_count": 0,
                "component_size": 1,
                "fan_out_ratio": 0.0,
            },
            {
                "graph_feature_id": "gf_u_untouched_2026-02-10",
                "user_id": "u_untouched",
                "snapshot_date": latest_snapshot,
                "shared_device_count": 7,
                "shared_bank_count": 0,
                "shared_wallet_count": 0,
                "blacklist_1hop_count": 0,
                "blacklist_2hop_count": 0,
                "component_size": 1,
                "fan_out_ratio": 0.0,
            },
            {
                "graph_feature_id": "gf_u_direct_2026-02-09",
                "user_id": "u_direct",
                "snapshot_date": previous_snapshot,
                "shared_device_count": 5,
                "shared_bank_count": 0,
                "shared_wallet_count": 0,
                "blacklist_1hop_count": 0,
                "blacklist_2hop_count": 0,
                "component_size": 1,
                "fan_out_ratio": 0.0,
            },
        ]),
    )

    day_rows = pd.DataFrame([
        {
            "feature_snapshot_id": "fd_u_direct_2026-02-10",
            "user_id": "u_direct",
            "snapshot_date": latest_snapshot,
            "feature_version": "stale",
            "legacy_marker": 999.0,
        },
        {
            "feature_snapshot_id": "fd_u_untouched_2026-02-10",
            "user_id": "u_untouched",
            "snapshot_date": latest_snapshot,
            "feature_version": "stale",
            "legacy_marker": 42.0,
        },
        {
            "feature_snapshot_id": "fd_u_direct_2026-02-09",
            "user_id": "u_direct",
            "snapshot_date": previous_snapshot,
            "feature_version": "older",
            "legacy_marker": 55.0,
        },
    ])
    rolling_rows = day_rows.copy()
    rolling_rows["feature_snapshot_id"] = [
        "f30_u_direct_2026-02-10",
        "f30_u_untouched_2026-02-10",
        "f30_u_direct_2026-02-09",
    ]
    store.replace_table("features.feature_snapshots_user_day", day_rows)
    store.replace_table("features.feature_snapshots_user_30d", rolling_rows)


def _seed_refresh_state(store: DuckDBStore, last_source_event_at: str | pd.Timestamp) -> None:
    store.execute("DELETE FROM ops.refresh_state")
    store.append_dataframe(
        "ops.refresh_state",
        pd.DataFrame([
            {
                "pipeline_name": "refresh_live",
                "status": "success",
                "last_success_at": pd.Timestamp("2026-02-09T00:00:00Z"),
                "last_source_event_at": pd.Timestamp(last_source_event_at),
                "last_run_started_at": pd.Timestamp("2026-02-09T00:00:00Z"),
                "last_run_finished_at": pd.Timestamp("2026-02-09T00:05:00Z"),
                "last_error": None,
                "details_json": json.dumps({"seed": True}),
            }
        ]),
    )


def _seed_existing_feature_tables(store: DuckDBStore) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    graph_df = pd.DataFrame([
        {
            "graph_feature_id": "existing_graph",
            "user_id": "u_existing",
            "snapshot_date": pd.Timestamp("2025-12-31"),
            "shared_device_count": 9,
            "shared_bank_count": 8,
            "shared_wallet_count": 7,
            "blacklist_1hop_count": 6,
            "blacklist_2hop_count": 5,
            "component_size": 4,
            "fan_out_ratio": 0.75,
        }
    ])
    day_df = pd.DataFrame([
        {
            "feature_snapshot_id": "existing_day",
            "user_id": "u_existing",
            "snapshot_date": pd.Timestamp("2025-12-31"),
            "feature_version": "existing",
            "component_size": 4,
        }
    ])
    rolling_df = pd.DataFrame([
        {
            "feature_snapshot_id": "existing_30d",
            "user_id": "u_existing",
            "snapshot_date": pd.Timestamp("2025-12-31"),
            "feature_version": "existing",
            "component_size": 4,
        }
    ])

    store.replace_table("features.graph_features", graph_df)
    store.replace_table("features.feature_snapshots_user_day", day_df)
    store.replace_table("features.feature_snapshots_user_30d", rolling_df)
    return graph_df, day_df, rolling_df


def test_training_dataset_excludes_pre_observation_positive_snapshots(tmp_path: Path, monkeypatch) -> None:
    store = _configure_model_store(tmp_path, monkeypatch)
    _seed_model_tables(store)

    dataset = training_dataset().sort_values(["user_id", "snapshot_date"]).reset_index(drop=True)

    assert len(dataset) == 10
    assert dataset.loc[dataset["user_id"] == "u_pos", "snapshot_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-01-03",
        "2026-01-04",
        "2026-01-05",
        "2026-01-06",
    ]
    assert dataset.loc[dataset["user_id"] == "u_neg", "snapshot_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-01-01",
        "2026-01-02",
        "2026-01-03",
        "2026-01-04",
        "2026-01-05",
        "2026-01-06",
    ]
    assert set(dataset.loc[dataset["user_id"] == "u_neg", "scenario_types"]) == {""}
    assert set(dataset.loc[dataset["user_id"] == "u_pos", "scenario_types"]) == {"structured_ring"}


def test_forward_date_splits_keep_future_validation_and_holdout_segments() -> None:
    split_dates = forward_date_splits(pd.Series(pd.date_range("2026-01-01", periods=6, freq="D")))
    assert [str(value) for value in split_dates["train"]] == [
        "2026-01-01",
        "2026-01-02",
        "2026-01-03",
        "2026-01-04",
    ]
    assert [str(value) for value in split_dates["valid"]] == ["2026-01-05"]
    assert [str(value) for value in split_dates["holdout"]] == ["2026-01-06"]

    minimum_dates = forward_date_splits(pd.Series(pd.date_range("2026-02-01", periods=3, freq="D")))
    assert [len(minimum_dates[name]) for name in ["train", "valid", "holdout"]] == [1, 1, 1]


def test_model_training_and_validation_use_dynamic_forward_splits(tmp_path: Path, monkeypatch) -> None:
    store = _configure_model_store(tmp_path, monkeypatch)
    _seed_model_tables(store)

    train_info = train_model()
    anomaly_info = train_anomaly_model()
    report = validate_model()

    train_meta = json.loads(Path(train_info["meta_path"]).read_text(encoding="utf-8"))
    anomaly_meta = json.loads(Path(anomaly_info["meta_path"]).read_text(encoding="utf-8"))

    assert train_meta["train_dates"] == ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
    assert train_meta["valid_dates"] == ["2026-01-05"]
    assert train_meta["holdout_dates"] == ["2026-01-06"]
    assert train_meta["holdout_rows"] == 2
    assert anomaly_meta["train_dates"] == train_meta["train_dates"]
    assert report["model_version"] == train_info["model_version"]
    assert sum(report["confusion_matrix"].values()) == 2
    assert {item["scenario"] for item in report["scenario_breakdown"]} == {"clean", "structured_ring"}

    validation_reports = store.read_table("ops.validation_reports")
    assert len(validation_reports) == 1


def test_snapshot_builders_limit_population_to_recent_activity_or_prior_blacklist(tmp_path: Path, monkeypatch) -> None:
    store = _configure_model_store(tmp_path, monkeypatch)
    _seed_snapshot_population_tables(store)

    graph_df = build_graph_features()
    feature_day, feature_30d = build_feature_snapshots()

    graph_df["snapshot_date"] = pd.to_datetime(graph_df["snapshot_date"])
    feature_day["snapshot_date"] = pd.to_datetime(feature_day["snapshot_date"])
    feature_30d["snapshot_date"] = pd.to_datetime(feature_30d["snapshot_date"])

    pre_blacklist_snapshot = pd.Timestamp("2026-01-09")
    latest_snapshot = pd.Timestamp("2026-02-10")

    assert set(graph_df.loc[graph_df["snapshot_date"] == pre_blacklist_snapshot, "user_id"]) == {"u_stale"}
    assert set(feature_day.loc[feature_day["snapshot_date"] == pre_blacklist_snapshot, "user_id"]) == {"u_stale"}
    assert set(feature_30d.loc[feature_30d["snapshot_date"] == pre_blacklist_snapshot, "user_id"]) == {"u_stale"}

    expected_latest_users = {"u_recent", "u_labeled"}
    assert set(graph_df.loc[graph_df["snapshot_date"] == latest_snapshot, "user_id"]) == expected_latest_users
    assert set(feature_day.loc[feature_day["snapshot_date"] == latest_snapshot, "user_id"]) == expected_latest_users
    assert set(feature_30d.loc[feature_30d["snapshot_date"] == latest_snapshot, "user_id"]) == expected_latest_users
    assert "u_stale" not in set(feature_day.loc[feature_day["snapshot_date"] == latest_snapshot, "user_id"])

    labeled_feature = feature_30d[
        (feature_30d["snapshot_date"] == latest_snapshot)
        & (feature_30d["user_id"] == "u_labeled")
    ].iloc[0]
    assert labeled_feature["fiat_in_30d"] == 0.0
    assert labeled_feature["trade_count_30d"] == 0


def test_targeted_graph_build_returns_only_requested_users_for_snapshot_date(tmp_path: Path, monkeypatch) -> None:
    _configure_model_store(tmp_path, monkeypatch)
    store = DuckDBStore(tmp_path / "bitoguard.duckdb")
    _seed_snapshot_population_tables(store)

    latest_snapshot = pd.Timestamp("2026-02-10")
    graph_df = build_graph_features(
        snapshot_dates=pd.DatetimeIndex([latest_snapshot]),
        target_user_ids={"u_recent", "u_stale"},
        persist=False,
    )
    graph_df["snapshot_date"] = pd.to_datetime(graph_df["snapshot_date"])

    assert set(graph_df["snapshot_date"]) == {latest_snapshot}
    assert set(graph_df["user_id"]) == {"u_recent", "u_stale"}


def test_targeted_feature_build_returns_only_requested_users_for_snapshot_date(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BITOGUARD_GRAPH_FEATURES_TRUSTED_ONLY", "false")
    _configure_model_store(tmp_path, monkeypatch)
    store = DuckDBStore(tmp_path / "bitoguard.duckdb")
    _seed_snapshot_population_tables(store)

    latest_snapshot = pd.Timestamp("2026-02-10")
    feature_day, feature_30d = build_feature_snapshots(
        snapshot_dates=pd.DatetimeIndex([latest_snapshot]),
        target_user_ids={"u_recent", "u_stale"},
        persist=False,
    )
    feature_day["snapshot_date"] = pd.to_datetime(feature_day["snapshot_date"])
    feature_30d["snapshot_date"] = pd.to_datetime(feature_30d["snapshot_date"])

    assert set(feature_day["snapshot_date"]) == {latest_snapshot}
    assert set(feature_30d["snapshot_date"]) == {latest_snapshot}
    assert set(feature_day["user_id"]) == {"u_recent", "u_stale"}
    assert set(feature_30d["user_id"]) == {"u_recent", "u_stale"}
    assert feature_30d.loc[feature_30d["user_id"] == "u_recent", "component_size"].iloc[0] == 2


def test_targeted_builders_with_persist_false_do_not_overwrite_feature_tables(tmp_path: Path, monkeypatch) -> None:
    _configure_model_store(tmp_path, monkeypatch)
    store = DuckDBStore(tmp_path / "bitoguard.duckdb")
    _seed_snapshot_population_tables(store)
    expected_graph, expected_day, expected_30d = _seed_existing_feature_tables(store)

    latest_snapshot = pd.Timestamp("2026-02-10")
    build_graph_features(
        snapshot_dates=pd.DatetimeIndex([latest_snapshot]),
        target_user_ids={"u_recent", "u_stale"},
        persist=False,
    )
    build_feature_snapshots(
        snapshot_dates=pd.DatetimeIndex([latest_snapshot]),
        target_user_ids={"u_recent", "u_stale"},
        persist=False,
    )

    pd.testing.assert_frame_equal(store.read_table("features.graph_features"), expected_graph)
    pd.testing.assert_frame_equal(store.read_table("features.feature_snapshots_user_day"), expected_day)
    pd.testing.assert_frame_equal(store.read_table("features.feature_snapshots_user_30d"), expected_30d)


def test_refresh_live_no_ops_when_watermark_covers_current_source_data(tmp_path: Path, monkeypatch) -> None:
    store = _configure_model_store(tmp_path, monkeypatch)
    _seed_refresh_incremental_source_tables(store)
    _seed_refresh_state(store, "2026-02-10T09:00:00Z")

    from pipeline import refresh_live as refresh_live_module

    def unexpected_call(*args, **kwargs):
        raise AssertionError("incremental no-op should not rebuild tables")

    monkeypatch.setattr(refresh_live_module, "build_graph_features", unexpected_call)
    monkeypatch.setattr(refresh_live_module, "build_feature_snapshots", unexpected_call)
    monkeypatch.setattr(refresh_live_module, "score_latest_snapshot", unexpected_call)

    summary = refresh_live_module.refresh_live()
    state = store.fetch_df("SELECT status, last_source_event_at FROM ops.refresh_state WHERE pipeline_name = 'refresh_live'")

    assert summary["status"] == "success"
    assert summary["no_op"] is True
    assert summary["affected_user_count"] == 0
    assert summary["updated_row_counts"] == {
        "features.graph_features": 0,
        "features.feature_snapshots_user_day": 0,
        "features.feature_snapshots_user_30d": 0,
    }
    assert pd.Timestamp(state.iloc[0]["last_source_event_at"]).tz_convert("UTC").isoformat() == "2026-02-10T09:00:00+00:00"
    assert state.iloc[0]["status"] == "success"


def test_refresh_live_watermark_advances_only_on_success(tmp_path: Path, monkeypatch) -> None:
    store = _configure_model_store(tmp_path, monkeypatch)
    _seed_refresh_incremental_source_tables(store)
    _seed_refresh_state(store, "2026-02-09T00:00:00Z")

    from pipeline import refresh_live as refresh_live_module

    def fake_build_graph_features(snapshot_dates=None, target_user_ids=None, persist=True) -> pd.DataFrame:
        snapshot_date = pd.Timestamp(snapshot_dates[0])
        return pd.DataFrame([
            {
                "graph_feature_id": f"gf_{user_id}_{snapshot_date.date().isoformat()}",
                "user_id": user_id,
                "snapshot_date": snapshot_date,
                "shared_device_count": 1,
                "shared_bank_count": 0,
                "shared_wallet_count": 0,
                "blacklist_1hop_count": 0,
                "blacklist_2hop_count": 0,
                "component_size": 2,
                "fan_out_ratio": 0.0,
            }
            for user_id in sorted(target_user_ids)
        ])

    def fake_build_feature_snapshots(snapshot_dates=None, target_user_ids=None, persist=True) -> tuple[pd.DataFrame, pd.DataFrame]:
        snapshot_date = pd.Timestamp(snapshot_dates[0])
        day = pd.DataFrame([
            {
                "feature_snapshot_id": f"fd_{user_id}_{snapshot_date.date().isoformat()}",
                "user_id": user_id,
                "snapshot_date": snapshot_date,
                "feature_version": "test_v1",
            }
            for user_id in sorted(target_user_ids)
        ])
        rolling = day.copy()
        rolling["feature_snapshot_id"] = [
            f"f30_{user_id}_{snapshot_date.date().isoformat()}"
            for user_id in sorted(target_user_ids)
        ]
        return day, rolling

    def fake_score_latest_snapshot() -> pd.DataFrame:
        return pd.DataFrame([
            {"user_id": "u_direct", "risk_level": "high"},
        ])

    monkeypatch.setattr(refresh_live_module, "build_graph_features", fake_build_graph_features)
    monkeypatch.setattr(refresh_live_module, "build_feature_snapshots", fake_build_feature_snapshots)
    monkeypatch.setattr(refresh_live_module, "score_latest_snapshot", fake_score_latest_snapshot)

    summary = refresh_live_module.refresh_live()
    state = store.fetch_df("SELECT status, last_source_event_at FROM ops.refresh_state WHERE pipeline_name = 'refresh_live'")

    assert summary["status"] == "success"
    assert summary["affected_user_count"] == 1
    assert pd.Timestamp(state.iloc[0]["last_source_event_at"]).tz_convert("UTC").isoformat() == "2026-02-10T09:00:00+00:00"

    login_events = store.read_table("canonical.login_events")
    login_events = pd.concat([
        login_events,
        pd.DataFrame([
            {
                "login_id": "login_direct_new",
                "user_id": "u_direct",
                "occurred_at": pd.Timestamp("2026-02-11T01:00:00Z"),
                "device_id": "device_shared",
                "ip_address": "1.1.1.1",
                "ip_country": "TW",
                "ip_city": "Taipei",
                "is_vpn": False,
                "is_new_device": False,
                "is_geo_jump": False,
                "success": True,
            }
        ]),
    ], ignore_index=True)
    store.replace_table("canonical.login_events", login_events)

    def failing_build_feature_snapshots(snapshot_dates=None, target_user_ids=None, persist=True):
        raise RuntimeError("synthetic refresh failure")

    monkeypatch.setattr(refresh_live_module, "build_feature_snapshots", failing_build_feature_snapshots)

    with pytest.raises(RuntimeError, match="synthetic refresh failure"):
        refresh_live_module.refresh_live()

    failed_state = store.fetch_df(
        "SELECT status, last_source_event_at, last_error FROM ops.refresh_state WHERE pipeline_name = 'refresh_live'"
    )
    assert failed_state.iloc[0]["status"] == "failed"
    assert pd.Timestamp(failed_state.iloc[0]["last_source_event_at"]).tz_convert("UTC").isoformat() == "2026-02-10T09:00:00+00:00"
    assert "synthetic refresh failure" in failed_state.iloc[0]["last_error"]


def test_refresh_live_derives_affected_users_from_new_events_only(tmp_path: Path, monkeypatch) -> None:
    store = _configure_model_store(tmp_path, monkeypatch)
    _seed_refresh_incremental_source_tables(store)

    from pipeline import refresh_live as refresh_live_module

    affected_user_ids = refresh_live_module._derive_affected_user_ids(
        store,
        pd.Timestamp("2026-02-09T00:00:00Z"),
        pd.Timestamp("2026-02-10T09:00:00Z"),
    )

    assert affected_user_ids == ["u_direct"]


def test_refresh_live_upsert_only_touches_target_users_for_latest_snapshot_rows(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BITOGUARD_GRAPH_FEATURES_TRUSTED_ONLY", "false")
    store = _configure_model_store(tmp_path, monkeypatch)
    _seed_refresh_incremental_source_tables(store)
    _seed_refresh_incremental_feature_tables(store)
    _seed_refresh_state(store, "2026-02-09T00:00:00Z")

    from pipeline import refresh_live as refresh_live_module

    monkeypatch.setattr(
        refresh_live_module,
        "score_latest_snapshot",
        lambda: pd.DataFrame([
            {"user_id": "u_direct", "risk_level": "high"},
        ]),
    )

    summary = refresh_live_module.refresh_live()

    graph_df = store.read_table("features.graph_features")
    feature_day = store.read_table("features.feature_snapshots_user_day")
    feature_30d = store.read_table("features.feature_snapshots_user_30d")
    graph_df["snapshot_date"] = pd.to_datetime(graph_df["snapshot_date"])
    feature_day["snapshot_date"] = pd.to_datetime(feature_day["snapshot_date"])
    feature_30d["snapshot_date"] = pd.to_datetime(feature_30d["snapshot_date"])

    latest_snapshot = pd.Timestamp("2026-02-10")
    previous_snapshot = pd.Timestamp("2026-02-09")

    assert summary["updated_row_counts"] == {
        "features.graph_features": 1,
        "features.feature_snapshots_user_day": 1,
        "features.feature_snapshots_user_30d": 1,
    }

    assert graph_df.loc[
        (graph_df["user_id"] == "u_untouched") & (graph_df["snapshot_date"] == latest_snapshot),
        "shared_device_count",
    ].iloc[0] == 7
    assert graph_df.loc[
        (graph_df["user_id"] == "u_direct") & (graph_df["snapshot_date"] == previous_snapshot),
        "shared_device_count",
    ].iloc[0] == 5
    assert graph_df.loc[
        (graph_df["user_id"] == "u_direct") & (graph_df["snapshot_date"] == latest_snapshot),
        "shared_device_count",
    ].iloc[0] == 1
    direct_latest_day = feature_day[
        (feature_day["user_id"] == "u_direct") & (feature_day["snapshot_date"] == latest_snapshot)
    ].iloc[0]
    untouched_latest_day = feature_day[
        (feature_day["user_id"] == "u_untouched") & (feature_day["snapshot_date"] == latest_snapshot)
    ].iloc[0]
    direct_previous_day = feature_day[
        (feature_day["user_id"] == "u_direct") & (feature_day["snapshot_date"] == previous_snapshot)
    ].iloc[0]
    assert pd.isna(direct_latest_day["legacy_marker"])
    assert untouched_latest_day["legacy_marker"] == 42.0
    assert direct_previous_day["legacy_marker"] == 55.0

    direct_latest_30d = feature_30d[
        (feature_30d["user_id"] == "u_direct") & (feature_30d["snapshot_date"] == latest_snapshot)
    ].iloc[0]
    untouched_latest_30d = feature_30d[
        (feature_30d["user_id"] == "u_untouched") & (feature_30d["snapshot_date"] == latest_snapshot)
    ].iloc[0]
    direct_previous_30d = feature_30d[
        (feature_30d["user_id"] == "u_direct") & (feature_30d["snapshot_date"] == previous_snapshot)
    ].iloc[0]
    assert pd.isna(direct_latest_30d["legacy_marker"])
    assert untouched_latest_30d["legacy_marker"] == 42.0
    assert direct_previous_30d["legacy_marker"] == 55.0


def test_refresh_live_main_uses_incremental_path_without_training_chain(tmp_path: Path, monkeypatch, capsys) -> None:
    store = _configure_model_store(tmp_path, monkeypatch)
    _seed_refresh_incremental_source_tables(store)
    _seed_refresh_state(store, "2026-02-09T00:00:00Z")

    from pipeline import refresh_live as refresh_live_module

    calls: list[tuple[str, object, object, object]] = []

    def fail_if_called():
        raise AssertionError("historical training steps should not run in incremental refresh")

    def fake_build_graph_features(snapshot_dates=None, target_user_ids=None, persist=True) -> pd.DataFrame:
        calls.append(("build_graph_features", snapshot_dates, set(target_user_ids), persist))
        snapshot_date = pd.Timestamp(snapshot_dates[0])
        return pd.DataFrame([
            {
                "graph_feature_id": f"gf_{user_id}_{snapshot_date.date().isoformat()}",
                "user_id": user_id,
                "snapshot_date": snapshot_date,
                "shared_device_count": 1,
                "shared_bank_count": 0,
                "shared_wallet_count": 0,
                "blacklist_1hop_count": 0,
                "blacklist_2hop_count": 0,
                "component_size": 2,
                "fan_out_ratio": 0.0,
            }
            for user_id in sorted(target_user_ids)
        ])

    def fake_build_feature_snapshots(snapshot_dates=None, target_user_ids=None, persist=True) -> tuple[pd.DataFrame, pd.DataFrame]:
        calls.append(("build_feature_snapshots", snapshot_dates, set(target_user_ids), persist))
        snapshot_date = pd.Timestamp(snapshot_dates[0])
        day = pd.DataFrame([
            {
                "feature_snapshot_id": f"fd_{user_id}_{snapshot_date.date().isoformat()}",
                "user_id": user_id,
                "snapshot_date": snapshot_date,
                "feature_version": "test_v1",
            }
            for user_id in sorted(target_user_ids)
        ])
        rolling = day.copy()
        rolling["feature_snapshot_id"] = [
            f"f30_{user_id}_{snapshot_date.date().isoformat()}"
            for user_id in sorted(target_user_ids)
        ]
        return day, rolling

    def fake_score_latest_snapshot() -> pd.DataFrame:
        calls.append(("score_latest_snapshot", None, None, None))
        return pd.DataFrame([
            {"user_id": "u_direct", "risk_level": "high"},
        ])

    monkeypatch.setattr(refresh_live_module, "build_graph_features", fake_build_graph_features)
    monkeypatch.setattr(refresh_live_module, "build_feature_snapshots", fake_build_feature_snapshots)
    monkeypatch.setattr(refresh_live_module, "score_latest_snapshot", fake_score_latest_snapshot)
    monkeypatch.setattr(refresh_live_module, "train_model", fail_if_called, raising=False)
    monkeypatch.setattr(refresh_live_module, "train_anomaly_model", fail_if_called, raising=False)
    monkeypatch.setattr(refresh_live_module, "validate_model", fail_if_called, raising=False)

    summary = refresh_live_module.main()
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    assert [call[0] for call in calls] == [
        "build_graph_features",
        "build_feature_snapshots",
        "score_latest_snapshot",
    ]
    assert calls[0][1] == [pd.Timestamp("2026-02-10")]
    assert calls[0][2] == {"u_direct"}
    assert calls[0][3] is False
    assert calls[1][1] == [pd.Timestamp("2026-02-10")]
    assert calls[1][2] == {"u_direct"}
    assert calls[1][3] is False
    assert summary == printed
    assert summary["mode"] == "latest_snapshot_incremental"
    assert summary["status"] == "success"
    assert summary["no_op"] is False
    assert summary["prediction_rows"] == 1
    assert summary["high_risk_count"] == 1


# ── Drift detection tests ──────────────────────────────────────────────────────


def test_drift_detect_no_drift():
    """Identical snapshots produce no drifted features and health_ok=True."""
    df = pd.DataFrame({
        "user_id": ["u1", "u2", "u3"],
        "snapshot_date": pd.Timestamp("2026-01-01"),
        "fiat_in_30d": [100.0, 200.0, 150.0],
        "crypto_withdraw_30d": [50.0, 80.0, 60.0],
        "trade_count_30d": [5.0, 10.0, 7.0],
    })
    result = detect_drift(df, df.copy(), "2026-01-01", "2026-01-02")
    assert isinstance(result, FeatureDriftResult)
    assert result.health_ok is True
    assert result.total_drifted == 0


def test_drift_detect_mean_shift():
    """Large mean shift on a column triggers drift detection."""
    df_from = pd.DataFrame({
        "user_id": ["u1", "u2", "u3"],
        "fiat_in_30d": [100.0, 100.0, 100.0],
        "stable_col": [10.0, 10.0, 10.0],
    })
    df_to = pd.DataFrame({
        "user_id": ["u1", "u2", "u3"],
        "fiat_in_30d": [0.0, 0.0, 0.0],   # complete collapse → 100% relative change
        "stable_col": [10.0, 10.0, 10.0],
    })
    result = detect_drift(df_from, df_to, "2026-01-01", "2026-01-02")
    drifted_cols = {f["feature"] for f in result.drifted_features}
    assert "fiat_in_30d" in drifted_cols
    assert "stable_col" not in drifted_cols
    assert result.health_ok is False


def test_drift_detect_zero_rate_spike():
    """Large increase in zero-rate flags a feature even without mean shift."""
    n = 50
    df_from = pd.DataFrame({
        "user_id": [f"u{i}" for i in range(n)],
        "vol": [float(i + 1) for i in range(n)],   # no zeros
    })
    df_to = pd.DataFrame({
        "user_id": [f"u{i}" for i in range(n)],
        "vol": [0.0] * 40 + [float(i + 1) for i in range(10)],  # 80% zeros
    })
    result = detect_drift(df_from, df_to, "2026-01-01", "2026-01-02")
    drifted_cols = {f["feature"] for f in result.drifted_features}
    assert "vol" in drifted_cols


def test_iforest_contamination_is_fixed() -> None:
    """IsolationForest must use a fixed contamination, not derived from labels."""
    import inspect
    from models.anomaly import train_anomaly_model
    src = inspect.getsource(train_anomaly_model)
    assert "contamination=0.05" in src, "contamination should be fixed at 0.05"
    assert "hidden_suspicious_label" not in src.split("contamination")[1].split("IsolationForest")[0], \
        "IsolationForest contamination must not depend on hidden_suspicious_label"
