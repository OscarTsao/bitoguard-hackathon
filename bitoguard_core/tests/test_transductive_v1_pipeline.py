from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from transductive_v1.branch_tabular import fit_catboost
from transductive_v1.common import bundle_path, feature_path, prediction_path
from transductive_v1.dataset import build_user_universe
from transductive_v1.graph_store import build_graph_store
from transductive_v1.label_aware_features import build_label_aware_features
from transductive_v1.label_free_features import build_label_free_user_features
from transductive_v1.primary_validation import PrimarySplitSpec, build_primary_split
from transductive_v1.secondary_validation import SecondarySplitSpec, build_secondary_group_split
from transductive_v1.train import train_transductive_v1
from transductive_v1.validate import validate_transductive_v1
from transductive_v1.score import score_transductive_v1


TABLES = (
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


def _prepare_subset(tmp_path: Path, monkeypatch) -> Path:
    source_root = Path(__file__).resolve().parents[2]
    clean_source = source_root / "data" / "aws_event" / "clean"
    clean_target = tmp_path / "clean"
    clean_target.mkdir(parents=True, exist_ok=True)

    user_info = pd.read_parquet(clean_source / "user_info.parquet")
    train = pd.read_parquet(clean_source / "train_label.parquet")
    predict_df = pd.read_parquet(clean_source / "predict_label.parquet")
    train_positive = _spread_sample(train[train["status"] == 1]["user_id"].astype(int), 12)
    train_negative = _spread_sample(train[train["status"] == 0]["user_id"].astype(int), 48)
    predict_users = _spread_sample(predict_df["user_id"].astype(int), 8)
    selected_users = set(pd.concat([train_positive, train_negative, predict_users]).tolist())

    for table in TABLES:
        frame = pd.read_parquet(clean_source / f"{table}.parquet")
        if "user_id" in frame.columns:
            frame = frame[frame["user_id"].astype(int).isin(selected_users)].copy()
        frame.to_parquet(clean_target / f"{table}.parquet", index=False)

    monkeypatch.setenv("BITOGUARD_AWS_EVENT_CLEAN_DIR", str(clean_target))
    monkeypatch.setenv("BITOGUARD_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    return clean_target


def _inject_graph_edges(clean_target: Path) -> dict[str, list[int]]:
    train_users = sorted(pd.read_parquet(clean_target / "train_label.parquet")["user_id"].astype(int).tolist())
    predict_users = sorted(pd.read_parquet(clean_target / "predict_label.parquet")["user_id"].astype(int).tolist())
    relation_users = train_users[0:2]
    wallet_users = train_users[2:5]
    ip_users = train_users[5:8]
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
    }


def _speed_up(monkeypatch) -> None:
    import transductive_v1.branch_tabular as branch_tabular
    import transductive_v1.primary_validation as primary_validation
    import transductive_v1.secondary_validation as secondary_validation
    import transductive_v1.train as train_module
    import transductive_v1.validate as validate_module

    monkeypatch.setattr(branch_tabular, "CATBOOST_ITERATIONS", 25)
    monkeypatch.setenv("BITOGUARD_TV1_CATBOOST_ITERATIONS", "25")
    monkeypatch.setattr(
        train_module,
        "build_primary_split",
        lambda labeled_frame, cutoff_tag="full", write_outputs=True: primary_validation.build_primary_split(
            labeled_frame,
            cutoff_tag=cutoff_tag,
            spec=primary_validation.PrimarySplitSpec(n_splits=3),
            write_outputs=write_outputs,
        ),
    )
    monkeypatch.setattr(
        validate_module,
        "build_secondary_group_split",
        lambda labeled_frame, graph_store, cutoff_tag="full", write_outputs=True: secondary_validation.build_secondary_group_split(
            labeled_frame,
            graph_store,
            cutoff_tag=cutoff_tag,
            spec=secondary_validation.SecondarySplitSpec(n_splits=2),
            write_outputs=write_outputs,
        ),
    )


def test_user_universe_and_primary_split_rows(tmp_path: Path, monkeypatch) -> None:
    clean_target = _prepare_subset(tmp_path, monkeypatch)
    universe = build_user_universe(write_outputs=True)
    train = pd.read_parquet(clean_target / "train_label.parquet")
    predict_df = pd.read_parquet(clean_target / "predict_label.parquet")
    assert set(train["user_id"].astype(int)).isdisjoint(set(predict_df["user_id"].astype(int)))
    split = build_primary_split(universe[universe["status"].notna()], spec=PrimarySplitSpec(n_splits=3), write_outputs=False)
    assert len(split) == len(train)
    assert split["primary_fold"].nunique() == 3


def test_label_masking_changes_label_aware_features(tmp_path: Path, monkeypatch) -> None:
    clean_target = _prepare_subset(tmp_path, monkeypatch)
    injected = _inject_graph_edges(clean_target)
    label_free = build_label_free_user_features(write_outputs=True)
    graph_store = build_graph_store(label_free["user_id"].astype(int).tolist(), write_outputs=True)
    full_labels = label_free[label_free["status"].notna()][["user_id", "status"]].copy()
    masked_labels = full_labels[full_labels["user_id"].astype(int) != injected["relation"][0]].copy()
    full_features = build_label_aware_features(graph_store, full_labels)
    masked_features = build_label_aware_features(graph_store, masked_labels)
    target_user = injected["relation"][1]
    full_row = full_features[full_features["user_id"].astype(int) == target_user].iloc[0]
    masked_row = masked_features[masked_features["user_id"].astype(int) == target_user].iloc[0]
    assert full_row["direct_positive_relation_count"] >= masked_row["direct_positive_relation_count"]
    assert full_row["positive_seed_propagation"] >= masked_row["positive_seed_propagation"]


def test_secondary_split_keeps_hard_components_together(tmp_path: Path, monkeypatch) -> None:
    clean_target = _prepare_subset(tmp_path, monkeypatch)
    injected = _inject_graph_edges(clean_target)
    label_free = build_label_free_user_features(write_outputs=True)
    graph_store = build_graph_store(label_free["user_id"].astype(int).tolist(), write_outputs=True)
    split = build_secondary_group_split(
        label_free[label_free["status"].notna()],
        graph_store,
        spec=SecondarySplitSpec(n_splits=2),
        write_outputs=False,
    )
    relation_folds = split[split["user_id"].astype(int).isin(injected["relation"])]["secondary_fold"].nunique()
    wallet_folds = split[split["user_id"].astype(int).isin(injected["wallet"])]["secondary_fold"].nunique()
    assert relation_folds == 1
    assert wallet_folds == 1


def test_train_validate_score_subset(tmp_path: Path, monkeypatch) -> None:
    _prepare_subset(tmp_path, monkeypatch)
    _speed_up(monkeypatch)
    train_meta = train_transductive_v1()
    assert Path(train_meta["bundle_path"]).exists()
    primary_oof = pd.read_parquet(feature_path("primary_stack_oof"))
    labeled = build_user_universe(write_outputs=False)
    assert len(primary_oof) == int(labeled["status"].notna().sum())
    report = validate_transductive_v1()
    assert "primary_transductive_oof_metrics" in report
    assert "secondary_group_stress_metrics" in report
    predictions = score_transductive_v1()
    predict_count = int(build_user_universe(write_outputs=False)["needs_prediction"].eq(True).sum())
    assert len(predictions) == predict_count
    assert prediction_path("predict_scores.parquet").exists()
    assert prediction_path("predict_scores.csv").exists()
    bundle = json.loads(bundle_path().read_text(encoding="utf-8"))
    assert bundle["decision_rule"] is not None
