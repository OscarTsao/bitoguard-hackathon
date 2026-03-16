"""Unit tests for models/rule_engine.py.

Each rule has at least one TRIGGER case (row satisfying the condition)
and one NO-TRIGGER case (row that should NOT satisfy the condition),
ensuring threshold regressions are caught immediately.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from models.rule_engine import evaluate_rules, RULE_DEFINITIONS, RULE_SEVERITY


# ── Helpers ──────────────────────────────────────────────────────────────────

def _base_row(**kwargs) -> pd.DataFrame:
    """Return a single-row feature DataFrame with sensible safe defaults."""
    defaults = {
        "user_id": "u001",
        "snapshot_date": "2026-01-30",
        # velocity
        "fiat_in_to_crypto_out_2h": False,
        "fiat_in_to_crypto_out_24h": False,
        # device / IP
        "new_device_withdrawal_24h": False,
        "ip_country_switch_count": 0,
        "crypto_withdraw_30d": 0.0,
        "night_large_withdrawal_ratio": 0.0,
        "new_device_ratio": 0.0,
        # graph
        "shared_device_count": 0,
        "blacklist_1hop_count": 0,
        "blacklist_2hop_count": 0,
        "fan_out_ratio": 0.0,
        "component_size": 1,
        # volume declared vs actual
        "actual_volume_expected_ratio": 0.0,
        # peer deviation
        "fiat_in_30d_peer_pct": 0.5,
        "crypto_withdraw_30d_peer_pct": 0.5,
    }
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


def _score(row_kwargs: dict) -> pd.Series:
    frame = _base_row(**row_kwargs)
    result = evaluate_rules(frame)
    return result.iloc[0]


# ── Structural tests ──────────────────────────────────────────────────────────

def test_evaluate_rules_returns_all_rule_columns():
    row = _score({})
    for rule in RULE_DEFINITIONS:
        assert rule in row.index, f"Missing rule column: {rule}"


def test_evaluate_rules_has_derived_columns():
    row = _score({})
    assert "rule_score" in row.index
    assert "rule_hit_count" in row.index
    assert "rule_hits" in row.index
    assert "top_reason_codes" in row.index


def test_clean_user_has_zero_hits():
    row = _score({})
    assert row["rule_hit_count"] == 0
    assert json.loads(row["rule_hits"]) == []
    assert row["rule_score"] == pytest.approx(0.0)


def test_rule_score_is_normalised_to_max_one():
    # Trigger all rules simultaneously → score must be ≤ 1.0
    row = _score({
        "fiat_in_to_crypto_out_2h": True,
        "fiat_in_to_crypto_out_24h": True,
        "new_device_withdrawal_24h": True,
        "ip_country_switch_count": 5,
        "crypto_withdraw_30d": 200_000,
        "night_large_withdrawal_ratio": 1.0,
        "new_device_ratio": 0.5,
        "shared_device_count": 5,
        "blacklist_1hop_count": 2,
        "blacklist_2hop_count": 1,
        "fan_out_ratio": 5.0,
        "component_size": 10,
        "actual_volume_expected_ratio": 10.0,
        "fiat_in_30d_peer_pct": 1.0,
        "crypto_withdraw_30d_peer_pct": 1.0,
    })
    assert 0.0 <= float(row["rule_score"]) <= 1.0


# ── Velocity rules ────────────────────────────────────────────────────────────

def test_fast_cash_out_2h_triggers():
    row = _score({"fiat_in_to_crypto_out_2h": True})
    assert bool(row["fast_cash_out_2h"]) is True


def test_fast_cash_out_2h_no_trigger():
    row = _score({"fiat_in_to_crypto_out_2h": False})
    assert bool(row["fast_cash_out_2h"]) is False


def test_fast_cash_out_24h_triggers():
    row = _score({"fiat_in_to_crypto_out_24h": True})
    assert bool(row["fast_cash_out_24h"]) is True


def test_fast_cash_out_24h_no_trigger():
    row = _score({"fiat_in_to_crypto_out_24h": False})
    assert bool(row["fast_cash_out_24h"]) is False


# ── Device / IP rules ─────────────────────────────────────────────────────────

def test_new_device_new_ip_large_withdraw_triggers():
    row = _score({
        "new_device_withdrawal_24h": True,
        "ip_country_switch_count": 2,
        "crypto_withdraw_30d": 100_000,
    })
    assert bool(row["new_device_new_ip_large_withdraw"]) is True


def test_new_device_new_ip_large_withdraw_no_trigger_small_amount():
    row = _score({
        "new_device_withdrawal_24h": True,
        "ip_country_switch_count": 3,
        "crypto_withdraw_30d": 10_000,  # below 50k threshold
    })
    assert bool(row["new_device_new_ip_large_withdraw"]) is False


def test_new_device_new_ip_large_withdraw_no_trigger_low_ip_switch():
    row = _score({
        "new_device_withdrawal_24h": True,
        "ip_country_switch_count": 1,  # below 2 threshold
        "crypto_withdraw_30d": 100_000,
    })
    assert bool(row["new_device_new_ip_large_withdraw"]) is False


def test_night_new_device_withdraw_triggers():
    row = _score({
        "night_large_withdrawal_ratio": 0.5,
        "new_device_ratio": 0.3,
    })
    assert bool(row["night_new_device_withdraw"]) is True


def test_night_new_device_withdraw_no_trigger_no_night():
    row = _score({
        "night_large_withdrawal_ratio": 0.0,
        "new_device_ratio": 1.0,
    })
    assert bool(row["night_new_device_withdraw"]) is False


def test_night_new_device_withdraw_no_trigger_no_new_device():
    row = _score({
        "night_large_withdrawal_ratio": 1.0,
        "new_device_ratio": 0.0,
    })
    assert bool(row["night_new_device_withdraw"]) is False


# ── Graph / relational rules ──────────────────────────────────────────────────

def test_shared_device_ring_triggers_at_threshold():
    row = _score({"shared_device_count": 3})
    assert bool(row["shared_device_ring"]) is True


def test_shared_device_ring_no_trigger_below_threshold():
    row = _score({"shared_device_count": 2})
    assert bool(row["shared_device_ring"]) is False


def test_blacklist_1hop_triggers():
    row = _score({"blacklist_1hop_count": 1})
    assert bool(row["blacklist_1hop"]) is True


def test_blacklist_1hop_no_trigger():
    row = _score({"blacklist_1hop_count": 0})
    assert bool(row["blacklist_1hop"]) is False


def test_blacklist_2hop_triggers():
    row = _score({"blacklist_2hop_count": 1})
    assert bool(row["blacklist_2hop"]) is True


def test_blacklist_2hop_no_trigger():
    row = _score({"blacklist_2hop_count": 0})
    assert bool(row["blacklist_2hop"]) is False


# ── Fan-out / structuring rules ───────────────────────────────────────────────

def test_high_fan_out_triggers():
    row = _score({"fan_out_ratio": 4.0, "component_size": 10})
    assert bool(row["high_fan_out"]) is True


def test_high_fan_out_no_trigger_low_ratio():
    row = _score({"fan_out_ratio": 2.0, "component_size": 10})
    assert bool(row["high_fan_out"]) is False


def test_high_fan_out_no_trigger_small_component():
    row = _score({"fan_out_ratio": 5.0, "component_size": 4})  # below 5 threshold
    assert bool(row["high_fan_out"]) is False


def test_volume_vs_declared_mismatch_triggers():
    row = _score({"actual_volume_expected_ratio": 5.0})
    assert bool(row["volume_vs_declared_mismatch"]) is True


def test_volume_vs_declared_mismatch_no_trigger():
    row = _score({"actual_volume_expected_ratio": 4.9})
    assert bool(row["volume_vs_declared_mismatch"]) is False


# ── Peer-deviation rules ──────────────────────────────────────────────────────

def test_extreme_fiat_peer_volume_triggers_at_p99():
    row = _score({"fiat_in_30d_peer_pct": 0.99})
    assert bool(row["extreme_fiat_peer_volume"]) is True


def test_extreme_fiat_peer_volume_no_trigger_below_p99():
    row = _score({"fiat_in_30d_peer_pct": 0.98})
    assert bool(row["extreme_fiat_peer_volume"]) is False


def test_extreme_withdraw_peer_volume_triggers_at_p99():
    row = _score({"crypto_withdraw_30d_peer_pct": 0.99})
    assert bool(row["extreme_withdraw_peer_volume"]) is True


def test_extreme_withdraw_peer_volume_no_trigger_below_p99():
    row = _score({"crypto_withdraw_30d_peer_pct": 0.985})
    assert bool(row["extreme_withdraw_peer_volume"]) is False


# ── Rule hit list correctness ─────────────────────────────────────────────────

def test_rule_hits_json_contains_triggered_rule_names():
    row = _score({
        "fiat_in_to_crypto_out_2h": True,
        "blacklist_1hop_count": 1,
    })
    hits = json.loads(str(row["rule_hits"]))
    assert "fast_cash_out_2h" in hits
    assert "blacklist_1hop" in hits
    assert "night_new_device_withdraw" not in hits


def test_rule_hit_count_matches_hit_list_length():
    row = _score({
        "fiat_in_to_crypto_out_2h": True,
        "fiat_in_to_crypto_out_24h": True,
        "shared_device_count": 3,
    })
    hits = json.loads(str(row["rule_hits"]))
    assert row["rule_hit_count"] == len(hits)


# ── Severity weight sanity ────────────────────────────────────────────────────

def test_severity_3_rules_score_higher_than_severity_2_rules():
    # fast_cash_out_2h (sev=3) alone should outscore fast_cash_out_24h (sev=2) alone
    row_high = _score({"fiat_in_to_crypto_out_2h": True})
    row_low = _score({"fiat_in_to_crypto_out_24h": True})
    assert float(row_high["rule_score"]) > float(row_low["rule_score"])


def test_all_rule_names_have_severity_defined():
    for rule in RULE_DEFINITIONS:
        assert rule in RULE_SEVERITY, f"Rule '{rule}' missing from RULE_SEVERITY"


# ── Cross-channel v2 rules ─────────────────────────────────────────────────────

def test_fiat_passthrough_triggers_at_threshold():
    row = _score({"fiat_dep_to_fiat_wdr_within_24h": 2})
    assert bool(row["fiat_passthrough"]) is True


def test_fiat_passthrough_no_trigger_below_threshold():
    row = _score({"fiat_dep_to_fiat_wdr_within_24h": 1})
    assert bool(row["fiat_passthrough"]) is False


def test_fiat_passthrough_no_trigger_for_clean_user():
    row = _score({})
    assert bool(row["fiat_passthrough"]) is False


def test_layering_burst_triggers_at_threshold():
    row = _score({"xch_layering_intensity": 5.1})
    assert bool(row["layering_burst"]) is True


def test_layering_burst_no_trigger_below_threshold():
    row = _score({"xch_layering_intensity": 4.9})
    assert bool(row["layering_burst"]) is False


def test_layering_burst_no_trigger_for_clean_user():
    row = _score({})
    assert bool(row["layering_burst"]) is False
