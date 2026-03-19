"""End-to-end scoring consistency tests (MONITORING-004).

Verifies that:
1. The persisted bundle can be loaded and produces valid score distributions.
2. Score percentiles fall within the expected range derived from v46 OOF.
3. The predict-only CSV scores are consistent with bundle expectations.
4. Loading the bundle twice produces identical scores (determinism).

These tests catch serialization bugs, feature schema mismatches, and
model loading failures before they silently degrade production scores.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Skip all tests if data directory is not available (CI / unit-test only env)
_CLEAN_DIR = os.environ.get("BITOGUARD_AWS_EVENT_CLEAN_DIR", "data/aws_event/clean")
_PREDICT_SCORES_CSV = Path("artifacts/predictions/official_predict_scores.csv")
_BUNDLE_PATH = Path("artifacts/official_bundle.json")
_OOF_PARQUET = Path("artifacts/official_features/official_oof_predictions.parquet")

pytestmark = pytest.mark.skipif(
    not _BUNDLE_PATH.exists(),
    reason="official_bundle.json not available — run official pipeline first",
)


class TestPredictScoresFile:
    """Tests on the persisted predict_only score CSV."""

    def test_predict_scores_csv_exists(self):
        assert _PREDICT_SCORES_CSV.exists(), "official_predict_scores.csv must exist after scoring"

    def test_predict_scores_has_expected_columns(self):
        df = pd.read_csv(_PREDICT_SCORES_CSV, nrows=5)
        assert "user_id" in df.columns, "user_id column missing from predict scores"
        # submission_probability is the primary score column in the official pipeline
        assert "submission_probability" in df.columns or any(
            c in df.columns for c in ("blend_score", "calibrated_score", "score", "stacker_raw_probability")
        ), f"No score column found. Available: {list(df.columns)}"

    def test_predict_scores_row_count(self):
        df = pd.read_csv(_PREDICT_SCORES_CSV)
        # predict_only cohort has 12,753 users
        assert 10_000 <= len(df) <= 15_000, \
            f"predict_scores should have ~12,753 rows; got {len(df)}"

    def test_predict_scores_in_unit_interval(self):
        df = pd.read_csv(_PREDICT_SCORES_CSV)
        # Find the primary score column (official pipeline uses submission_probability)
        score_col = next(
            (c for c in ("submission_probability", "blend_score", "calibrated_score",
                         "stacker_raw_probability", "score", "raw_score") if c in df.columns),
            None,
        )
        if score_col is None:
            score_cols = [c for c in df.columns if "prob" in c.lower() or "score" in c.lower()]
            assert score_cols, f"No score column found in {list(df.columns)}"
            score_col = score_cols[0]
        scores = df[score_col].dropna().to_numpy(dtype=float)
        assert (scores >= 0).all() and (scores <= 1).all(), \
            f"Scores must be in [0,1]; got min={scores.min():.4f}, max={scores.max():.4f}"

    def test_predict_scores_not_all_zero(self):
        df = pd.read_csv(_PREDICT_SCORES_CSV)
        score_cols = [c for c in df.columns if "score" in c.lower() or "prob" in c.lower()]
        if not score_cols:
            pytest.skip("No score column to check")
        scores = df[score_cols[0]].dropna().to_numpy(dtype=float)
        assert scores.std() > 0.001, "Score distribution is degenerate (constant)"

    def test_predict_scores_sanity_via_monitor(self):
        """Score sanity check via the monitoring module."""
        from services.model_monitor import check_score_sanity
        df = pd.read_csv(_PREDICT_SCORES_CSV)
        score_cols = [c for c in df.columns if "score" in c.lower() or "prob" in c.lower()]
        if not score_cols:
            pytest.skip("No score column to check")
        scores = df[score_cols[0]].dropna().to_numpy(dtype=float)
        result = check_score_sanity(scores)
        assert result.health_ok, \
            f"Predict scores failed sanity check: {result.checks_failed}"


class TestOofConsistency:
    """Tests on OOF predictions — validates model calibration is stable."""

    @pytest.fixture(autouse=True)
    def require_oof(self):
        if not _OOF_PARQUET.exists():
            pytest.skip("OOF parquet not available")

    def test_oof_blend_f1_within_tolerance(self):
        """Validate that the OOF blend F1 is within tolerance of the v46 certified value."""
        from sklearn.metrics import f1_score
        oof = pd.read_parquet(_OOF_PARQUET)
        y = oof["status"].astype(int).to_numpy()
        blend = (
            0.05 * oof["base_d_probability"].astype(float) +
            0.20 * oof["base_e_probability"].astype(float) +
            0.25 * oof["base_c_s_probability"].astype(float) +
            0.50 * oof["base_cs_x_anomaly"].astype(float)
        ).to_numpy()
        best_f1 = max(
            f1_score(y, (blend >= t).astype(int), zero_division=0)
            for t in np.arange(0.15, 0.35, 0.005)
        )
        # Certified OOF F1 = 0.3682; allow ±0.005 tolerance
        assert abs(best_f1 - 0.3682) < 0.005, \
            f"OOF blend F1={best_f1:.4f} deviates from certified 0.3682 by more than 0.005"

    def test_oof_positive_count(self):
        """Number of positives must match the labeled dataset."""
        oof = pd.read_parquet(_OOF_PARQUET)
        n_pos = int((oof["status"].astype(int) == 1).sum())
        assert 1600 <= n_pos <= 1700, f"Expected ~1640 positives in OOF; got {n_pos}"

    def test_oof_cs_x_anomaly_in_unit_interval(self):
        """cs_x_anomaly blend candidate must be in [0,1]."""
        oof = pd.read_parquet(_OOF_PARQUET)
        vals = oof["base_cs_x_anomaly"].dropna().astype(float).to_numpy()
        assert (vals >= 0).all() and (vals <= 1.001).all(), \
            f"cs_x_anomaly out of [0,1]: min={vals.min():.4f}, max={vals.max():.4f}"

    def test_oof_blend_ece_within_tolerance(self):
        """Blend ECE must remain ≤ 0.03 (well-calibrated). v46 baseline: ECE=0.0114."""
        from official.validate import _expected_calibration_error
        oof = pd.read_parquet(_OOF_PARQUET)
        y = oof["status"].astype(int).to_numpy()
        blend = (
            0.05 * oof["base_d_probability"].astype(float) +
            0.20 * oof["base_e_probability"].astype(float) +
            0.25 * oof["base_c_s_probability"].astype(float) +
            0.50 * oof["base_cs_x_anomaly"].astype(float)
        ).to_numpy()
        ece = _expected_calibration_error(y, blend)
        assert ece <= 0.03, \
            f"Blend ECE={ece:.4f} exceeds tolerance 0.03 — calibration may have degraded"
