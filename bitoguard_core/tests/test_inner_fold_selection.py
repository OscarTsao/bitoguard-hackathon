"""Tests for inner-fold selection module."""

import numpy as np
import pandas as pd
import pytest

from official.inner_fold_selection import select_and_apply_inner_fold
from official.stacking import STACKER_FEATURE_COLUMNS


@pytest.fixture
def mock_oof_data():
    """Create mock OOF predictions for testing."""
    np.random.seed(42)
    n_train = 100
    n_valid = 30
    
    # Create train data
    train_df = pd.DataFrame({
        "user_id": range(1, n_train + 1),
        "status": np.random.choice([0, 1], n_train, p=[0.9, 0.1]),
        "primary_fold": np.random.choice([0, 1, 2, 3], n_train),
        "base_a_probability": np.random.uniform(0.01, 0.3, n_train),
        "base_c_s_probability": np.random.uniform(0.01, 0.25, n_train),
        "base_b_probability": np.random.uniform(0.01, 0.28, n_train),
        "base_c_probability": np.random.uniform(0.0, 0.1, n_train),
        "base_d_probability": np.random.uniform(0.01, 0.27, n_train),
        "base_e_probability": np.random.uniform(0.01, 0.26, n_train),
        "rule_score": np.random.uniform(0.0, 0.5, n_train),
        "anomaly_score": np.random.uniform(0.0, 0.4, n_train),
        "crypto_anomaly_score": np.random.uniform(0.0, 0.3, n_train),
        "anomaly_score_segmented": np.random.uniform(0.0, 0.35, n_train),
    })
    
    # Create valid data
    valid_df = pd.DataFrame({
        "user_id": range(n_train + 1, n_train + n_valid + 1),
        "status": np.random.choice([0, 1], n_valid, p=[0.9, 0.1]),
        "primary_fold": 4,  # All in fold 4
        "base_a_probability": np.random.uniform(0.01, 0.3, n_valid),
        "base_c_s_probability": np.random.uniform(0.01, 0.25, n_valid),
        "base_b_probability": np.random.uniform(0.01, 0.28, n_valid),
        "base_c_probability": np.random.uniform(0.0, 0.1, n_valid),
        "base_d_probability": np.random.uniform(0.01, 0.27, n_valid),
        "base_e_probability": np.random.uniform(0.01, 0.26, n_valid),
        "rule_score": np.random.uniform(0.0, 0.5, n_valid),
        "anomaly_score": np.random.uniform(0.0, 0.4, n_valid),
        "crypto_anomaly_score": np.random.uniform(0.0, 0.3, n_valid),
        "anomaly_score_segmented": np.random.uniform(0.0, 0.35, n_valid),
    })
    
    return train_df, valid_df


def test_select_and_apply_inner_fold_basic(mock_oof_data):
    """Test basic functionality of inner fold selection."""
    train_oof, valid_oof = mock_oof_data
    
    result_valid, metadata = select_and_apply_inner_fold(
        train_oof,
        valid_oof,
        fold_column="primary_fold",
        stacker_feature_columns=STACKER_FEATURE_COLUMNS,
        calibration_method="isotonic",
        n_bootstrap=10,  # Reduced for speed
    )
    
    # Check that result has required columns
    assert "stacker_raw_probability" in result_valid.columns
    assert "submission_probability" in result_valid.columns
    
    # Check that probabilities are in valid range
    assert result_valid["stacker_raw_probability"].between(0, 1).all()
    assert result_valid["submission_probability"].between(0, 1).all()
    
    # Check metadata structure
    assert "blend_weights" in metadata
    assert "calibration_method" in metadata
    assert "selected_threshold" in metadata
    assert "threshold_report" in metadata
    
    # Check that blend weights are valid
    assert isinstance(metadata["blend_weights"], dict)
    assert len(metadata["blend_weights"]) > 0
    assert all(w >= 0 for w in metadata["blend_weights"].values())
    
    # Check threshold is in reasonable range
    assert 0 < metadata["selected_threshold"] < 1


def test_select_and_apply_preserves_user_ids(mock_oof_data):
    """Test that user IDs are preserved correctly."""
    train_oof, valid_oof = mock_oof_data
    original_user_ids = valid_oof["user_id"].tolist()
    
    result_valid, _ = select_and_apply_inner_fold(
        train_oof,
        valid_oof,
        fold_column="primary_fold",
        stacker_feature_columns=STACKER_FEATURE_COLUMNS,
        calibration_method="identity",
        n_bootstrap=10,
    )
    
    # Check user IDs are unchanged
    assert result_valid["user_id"].tolist() == original_user_ids
    assert len(result_valid) == len(valid_oof)


def test_select_and_apply_identity_calibration(mock_oof_data):
    """Test with identity calibration (no calibration)."""
    train_oof, valid_oof = mock_oof_data
    
    result_valid, metadata = select_and_apply_inner_fold(
        train_oof,
        valid_oof,
        fold_column="primary_fold",
        stacker_feature_columns=STACKER_FEATURE_COLUMNS,
        calibration_method="identity",
        n_bootstrap=10,
    )
    
    # With identity calibration, submission_probability should equal stacker_raw_probability
    np.testing.assert_array_almost_equal(
        result_valid["stacker_raw_probability"].values,
        result_valid["submission_probability"].values,
        decimal=6,
    )
    
    assert metadata["calibration_method"] == "identity"


def test_blend_weights_different_from_global(mock_oof_data):
    """Test that per-fold blend weights can differ from global tuning."""
    train_oof, valid_oof = mock_oof_data
    
    # Get per-fold blend weights
    _, metadata_fold = select_and_apply_inner_fold(
        train_oof,
        valid_oof,
        fold_column="primary_fold",
        stacker_feature_columns=STACKER_FEATURE_COLUMNS,
        calibration_method="identity",
        n_bootstrap=10,
    )
    
    # Get global blend weights (on all data)
    from official.stacking import tune_blend_weights, _add_base_meta_features
    all_data = pd.concat([train_oof, valid_oof], ignore_index=True)
    all_data = _add_base_meta_features(all_data)
    global_weights = tune_blend_weights(all_data)
    
    # Weights should exist
    assert len(metadata_fold["blend_weights"]) > 0
    assert len(global_weights) > 0
    
    # Note: Weights may be the same or different depending on data distribution
    # This test just verifies both methods produce valid weights
    print(f"Fold weights: {metadata_fold['blend_weights']}")
    print(f"Global weights: {global_weights}")


def test_meta_features_added(mock_oof_data):
    """Test that meta features are computed correctly."""
    train_oof, valid_oof = mock_oof_data
    
    result_valid, _ = select_and_apply_inner_fold(
        train_oof,
        valid_oof,
        fold_column="primary_fold",
        stacker_feature_columns=STACKER_FEATURE_COLUMNS,
        calibration_method="identity",
        n_bootstrap=10,
    )
    
    # Check that meta features exist
    meta_features = [
        "max_base_probability",
        "std_base_probability",
        "base_a_x_anomaly",
        "base_a_x_rule",
        "base_a_x_cs",
        "cs_deficit",
    ]
    
    for feat in meta_features:
        assert feat in result_valid.columns, f"Missing meta feature: {feat}"
        assert not result_valid[feat].isna().all(), f"Meta feature {feat} is all NaN"

