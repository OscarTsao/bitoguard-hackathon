"""Tests for new feature modules: dormancy, n-grams, statistical features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.dormancy import compute_dormancy_score, is_dormant, split_dormant_active
from features.event_ngram_features import (
    compute_event_ngram_features,
    _transition_entropy,
    _longest_same_streak,
)
from features.statistical_features import (
    _benford_chi_squared,
    _amount_entropy,
    _round_number_ratio,
    _burst_score,
    compute_statistical_features,
)


class TestDormancy:
    def _dormant_df(self) -> pd.DataFrame:
        return pd.DataFrame({col: [0] for col in [
            "twd_dep_count", "twd_wdr_count", "twd_all_count",
            "crypto_dep_count", "crypto_wdr_count", "crypto_all_count",
            "swap_count", "trading_count",
            "ip_n_unique", "ip_n_sessions",
            "early_3d_count",
        ]})

    def _active_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "twd_dep_count": [5], "twd_wdr_count": [3], "twd_all_count": [8],
            "crypto_dep_count": [0], "crypto_wdr_count": [0], "crypto_all_count": [0],
            "swap_count": [2], "trading_count": [0],
            "ip_n_unique": [3], "ip_n_sessions": [10], "early_3d_count": [4],
        })

    def test_fully_dormant_scores_one(self):
        score = compute_dormancy_score(self._dormant_df()).iloc[0]
        assert score == pytest.approx(1.0), f"Expected 1.0, got {score}"

    def test_fully_dormant_is_dormant(self):
        assert is_dormant(self._dormant_df()).iloc[0]

    def test_active_scores_less_than_one(self):
        score = compute_dormancy_score(self._active_df()).iloc[0]
        assert score < 1.0, f"Active user should score <1.0, got {score}"

    def test_active_is_not_dormant(self):
        assert not is_dormant(self._active_df()).iloc[0]

    def test_split_correct_partition(self):
        df = pd.concat([self._dormant_df(), self._active_df()], ignore_index=True)
        dormant, active = split_dormant_active(df)
        assert len(dormant) == 1
        assert len(active) == 1

    def test_missing_columns_returns_zero(self):
        empty_df = pd.DataFrame({"user_id": ["u1"], "some_other_col": [42]})
        scores = compute_dormancy_score(empty_df)
        assert scores.iloc[0] == pytest.approx(0.0)

    def test_official_pipeline_columns(self):
        """Test with official pipeline column names (twd_total_count etc.)."""
        df = pd.DataFrame({
            "twd_total_count": [0], "twd_deposit_count": [0], "twd_withdraw_count": [0],
            "crypto_total_count": [0], "crypto_deposit_count": [0], "crypto_withdraw_count": [0],
            "swap_total_count": [0],
        })
        score = compute_dormancy_score(df).iloc[0]
        assert score == pytest.approx(1.0)

    def test_threshold_partial(self):
        """Near-dormant user with 0.8 threshold."""
        df = pd.DataFrame({
            "twd_dep_count": [0], "twd_wdr_count": [0], "twd_all_count": [0],
            "crypto_dep_count": [0], "crypto_wdr_count": [0], "crypto_all_count": [0],
            "swap_count": [0], "trading_count": [0],
            "ip_n_unique": [1], "ip_n_sessions": [1],  # non-zero
            "early_3d_count": [0],
        })
        score = compute_dormancy_score(df).iloc[0]
        assert 0.0 < score < 1.0
        assert not is_dormant(df, threshold=1.0).iloc[0]
        assert is_dormant(df, threshold=score).iloc[0]


class TestEventNgrams:
    def _make_fiat(self) -> pd.DataFrame:
        return pd.DataFrame({
            "user_id": ["u1", "u1", "u1", "u2"],
            "occurred_at": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-01"],
            "direction": ["deposit", "withdrawal", "deposit", "deposit"],
        })

    def _make_crypto(self) -> pd.DataFrame:
        return pd.DataFrame({
            "user_id": ["u1", "u2"],
            "occurred_at": ["2025-01-01T12:00", "2025-01-02"],
            "direction": ["withdrawal", "deposit"],
        })

    def _make_trades(self) -> pd.DataFrame:
        return pd.DataFrame({
            "user_id": ["u1"],
            "occurred_at": ["2025-01-01T06:00"],
            "side": ["buy"],
            "order_type": ["instant_swap"],
        })

    def test_basic_two_users(self):
        result = compute_event_ngram_features(
            self._make_fiat(), self._make_crypto(), self._make_trades()
        )
        assert len(result) == 2

    def test_expected_columns_present(self):
        result = compute_event_ngram_features(
            self._make_fiat(), self._make_crypto(), self._make_trades()
        )
        assert "bg_FD_SB" in result.columns
        assert "bg_FD_CW" in result.columns
        assert "tg_FD_SB_CW" in result.columns
        assert "seq_transition_entropy" in result.columns
        assert "seq_longest_streak" in result.columns
        assert "seq_outflow_fraction" in result.columns
        assert "seq_inflow_outflow_ratio" in result.columns

    def test_kind_label_column_variant(self):
        """Test with kind_label instead of direction (official pipeline format)."""
        fiat = pd.DataFrame({
            "user_id": ["u1", "u1"],
            "created_at": ["2025-01-01", "2025-01-02"],
            "kind_label": ["deposit", "withdrawal"],
        })
        result = compute_event_ngram_features(fiat, pd.DataFrame(), pd.DataFrame())
        assert len(result) == 1
        assert result["bg_FD_FW"].iloc[0] == 1

    def test_swap_order_type(self):
        """Swap events (order_type=instant_swap) should map to SB/SS tokens."""
        trades = pd.DataFrame({
            "user_id": ["u1"],
            "occurred_at": ["2025-01-01"],
            "side": ["buy"],
            "order_type": ["instant_swap"],
        })
        result = compute_event_ngram_features(pd.DataFrame(), pd.DataFrame(), trades)
        assert result["seq_length"].iloc[0] == 1
        assert result["seq_n_unique_types"].iloc[0] == 1

    def test_transition_entropy_deterministic_is_zero(self):
        """Perfectly alternating sequence A→B→A→B has entropy 0 (deterministic)."""
        assert _transition_entropy(["A", "B", "A", "B"]) == pytest.approx(0.0)

    def test_transition_entropy_branching(self):
        """When A→B and A→C both occur, entropy > 0."""
        assert _transition_entropy(["A", "B", "A", "C"]) > 0.0

    def test_transition_entropy_constant(self):
        assert _transition_entropy(["A", "A", "A"]) == pytest.approx(0.0)

    def test_transition_entropy_short(self):
        assert _transition_entropy([]) == pytest.approx(0.0)
        assert _transition_entropy(["A"]) == pytest.approx(0.0)

    def test_longest_streak(self):
        assert _longest_same_streak(["A", "A", "B", "B", "B", "A"]) == 3
        assert _longest_same_streak(["A"]) == 1
        assert _longest_same_streak([]) == 0

    def test_empty_tables_returns_empty(self):
        result = compute_event_ngram_features(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )
        assert result.empty

    def test_outflow_fraction_layering_pattern(self):
        """Classic layering: FD → SB → CW should have high outflow fraction."""
        fiat = pd.DataFrame({
            "user_id": ["u1"],
            "occurred_at": ["2025-01-01 08:00"],
            "direction": ["deposit"],
        })
        trades = pd.DataFrame({
            "user_id": ["u1"],
            "occurred_at": ["2025-01-01 09:00"],
            "side": ["buy"],
            "order_type": ["instant_swap"],
        })
        crypto = pd.DataFrame({
            "user_id": ["u1"],
            "occurred_at": ["2025-01-01 10:00"],
            "direction": ["withdrawal"],
        })
        result = compute_event_ngram_features(fiat, crypto, trades)
        assert result["tg_FD_SB_CW"].iloc[0] == 1
        assert result["bg_FD_SB"].iloc[0] == 1
        assert result["bg_SB_CW"].iloc[0] == 1


class TestStatisticalFeatures:
    def test_benford_natural_low_chi2(self):
        np.random.seed(42)
        natural = pd.Series(np.random.lognormal(10, 2, 1000))
        chi2 = _benford_chi_squared(natural)
        assert chi2 < 1.0, f"Natural lognormal should have low Benford chi2, got {chi2:.4f}"

    def test_benford_structured_higher_chi2(self):
        structured = pd.Series([1000.0, 2000.0, 3000.0, 5000.0, 10000.0] * 200)
        natural = pd.Series(np.random.lognormal(10, 2, 1000))
        assert _benford_chi_squared(structured) > _benford_chi_squared(natural)

    def test_benford_few_samples_returns_zero(self):
        assert _benford_chi_squared(pd.Series([1000.0, 2000.0])) == 0.0

    def test_round_ratio_exact(self):
        amounts = pd.Series([1000.0, 2000.0, 500.0, 3000.0, 750.0])
        ratio = _round_number_ratio(amounts)
        assert ratio == pytest.approx(0.6), f"Expected 0.6, got {ratio}"

    def test_round_ratio_empty(self):
        assert _round_number_ratio(pd.Series([])) == 0.0

    def test_entropy_uniform_is_zero(self):
        uniform = pd.Series([1000.0] * 50)
        assert _amount_entropy(uniform) == pytest.approx(0.0, abs=1e-6)

    def test_entropy_diverse_positive(self):
        np.random.seed(42)
        diverse = pd.Series(np.random.uniform(1, 100000, 100))
        assert _amount_entropy(diverse) > 0.0

    def test_entropy_ordered(self):
        np.random.seed(42)
        low = _amount_entropy(pd.Series([1000.0] * 100))
        high = _amount_entropy(pd.Series(np.random.uniform(1, 100000, 100)))
        assert high > low

    def test_burst_score_single_day_high(self):
        ts = pd.Series(pd.date_range("2025-01-01", periods=10, freq="h"))
        score = _burst_score(ts)
        assert score >= 1.0

    def test_burst_score_few_events_ok(self):
        ts = pd.Series(pd.date_range("2025-01-01", periods=2, freq="D"))
        score = _burst_score(ts)
        assert score >= 0.0

    def test_compute_all_returns_one_row(self):
        fiat = pd.DataFrame({
            "user_id": ["u1"] * 5,
            "occurred_at": pd.date_range("2025-01-01", periods=5, freq="D"),
            "amount_twd": [1000.0, 2000.0, 500.0, 3000.0, 750.0],
        })
        result = compute_statistical_features(fiat, pd.DataFrame(), pd.DataFrame())
        assert len(result) == 1
        assert "fiat_benford_chi2" in result.columns
        assert "fiat_amount_entropy" in result.columns
        assert "fiat_burst_score" in result.columns
        assert "fiat_round_ratio" in result.columns
        assert "fiat_inter_event_cv" in result.columns

    def test_compute_all_empty_returns_empty(self):
        result = compute_statistical_features(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )
        assert result.empty

    def test_official_column_names_created_at(self):
        """Official pipeline uses created_at instead of occurred_at."""
        fiat = pd.DataFrame({
            "user_id": ["u1"] * 5,
            "created_at": pd.date_range("2025-01-01", periods=5, freq="D"),
            "amount_twd": [1000.0, 2000.0, 500.0, 3000.0, 750.0],
        })
        result = compute_statistical_features(fiat, pd.DataFrame(), pd.DataFrame())
        assert len(result) == 1
        assert result["fiat_burst_score"].iloc[0] >= 0.0
