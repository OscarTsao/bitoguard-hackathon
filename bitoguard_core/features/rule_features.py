"""Rule signal features: evaluate AML rules on v2 features, add outputs as ML inputs.

Why rule signals as features (not filters):
  - Filtering training data by rules introduces selection bias and loses the
    majority class, making the stacker train on a skewed distribution.
  - Adding rule outputs as features lets the stacker learn interactions:
    e.g., a high fan-out_ratio + rule_fast_cash_out_2h fires → very high risk,
    but either alone might be benign.
  - Rules remain active in scoring (M1 contributes 20% of risk_score) — these
    features give the stacker awareness of what M1 is seeing.

Output columns (all prefixed with `rule_`):
  11 binary flags (0/1 int), rule_score (float 0-1), rule_hit_count (int ≥ 0)
"""
from __future__ import annotations

import pandas as pd

from models.rule_engine import evaluate_rules


# Columns returned by this module (excluding user_id).
# Keep in sync with evaluate_rules() output schema.
RULE_FEATURE_COLUMNS = [
    "rule_fast_cash_out_2h",
    "rule_fast_cash_out_24h",
    "rule_new_device_new_ip_large_withdraw",
    "rule_night_new_device_withdraw",
    "rule_shared_device_ring",
    "rule_blacklist_2hop",
    "rule_blacklist_1hop",
    "rule_high_fan_out",
    "rule_volume_vs_declared_mismatch",
    "rule_extreme_fiat_peer_volume",
    "rule_extreme_withdraw_peer_volume",
    "rule_score",
    "rule_hit_count",
]

# Maps v2 column names → v1 names expected by evaluate_rules().
# Mirrors _build_rule_compat_frame() in models/score.py.
_V2_TO_V1_SCALARS = {
    "fiat_dep_to_swap_buy_within_1h":  None,  # handled specially (bool cast)
    "fiat_dep_to_swap_buy_within_24h": None,  # handled specially (bool cast)
    "crypto_wdr_twd_sum":              "crypto_withdraw_30d",
    "twd_dep_sum":                     "fiat_in_30d",
}

_RULE_FLAG_COLS = [
    "fast_cash_out_2h",
    "fast_cash_out_24h",
    "new_device_new_ip_large_withdraw",
    "night_new_device_withdraw",
    "shared_device_ring",
    "blacklist_2hop",
    "blacklist_1hop",
    "high_fan_out",
    "volume_vs_declared_mismatch",
    "extreme_fiat_peer_volume",
    "extreme_withdraw_peer_volume",
]


def compute_rule_features(
    v2_frame: pd.DataFrame,
    snapshot_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Evaluate AML rules on v2 features and return rule outputs as ML features.

    Args:
        v2_frame: DataFrame with v2 feature columns and a `user_id` column.
        snapshot_date: Snapshot date; forwarded to evaluate_rules() for the
            rule result DataFrame index. If None, uses today's date.

    Returns:
        DataFrame with columns: user_id + RULE_FEATURE_COLUMNS.
        All users in v2_frame are represented; missing signal → 0.
    """
    if v2_frame.empty:
        return pd.DataFrame(columns=["user_id"] + RULE_FEATURE_COLUMNS)

    if snapshot_date is None:
        snapshot_date = pd.Timestamp.now(tz="UTC").normalize()

    # Build v1-compatible frame (same logic as models/score.py::_build_rule_compat_frame)
    f = v2_frame[["user_id"]].copy()

    # Velocity rules: v2 stores integer counts; rules expect bool flags
    f["fiat_in_to_crypto_out_2h"] = (
        v2_frame.get("fiat_dep_to_swap_buy_within_1h",  pd.Series(0, index=v2_frame.index)) > 0
    ).values
    f["fiat_in_to_crypto_out_24h"] = (
        v2_frame.get("fiat_dep_to_swap_buy_within_24h", pd.Series(0, index=v2_frame.index)) > 0
    ).values

    # Volume proxies (v2 uses lifetime sums; acceptable approximation for rule thresholds)
    f["crypto_withdraw_30d"] = v2_frame.get("crypto_wdr_twd_sum",  pd.Series(0.0, index=v2_frame.index)).values
    f["fiat_in_30d"]         = v2_frame.get("twd_dep_sum",         pd.Series(0.0, index=v2_frame.index)).values

    # Device/IP rules: not reconstructible from v2 aggregate features → 0 (rules won't fire)
    for col in ("new_device_withdrawal_24h", "ip_country_switch_count",
                "night_large_withdrawal_ratio", "new_device_ratio"):
        f[col] = 0

    # Graph rules: disabled features remain 0
    for col in ("shared_device_count", "blacklist_1hop_count",
                "blacklist_2hop_count", "component_size"):
        f[col] = 0

    # Fan-out: ip_n_entities (number of distinct users sharing an IP) is a proxy
    f["fan_out_ratio"] = v2_frame.get("ip_n_entities", pd.Series(0, index=v2_frame.index)).astype(float).values

    # Volume vs declared income (monthly_income_twd is NULL in current data → ratio ≈ 0)
    vol_total = (
        v2_frame.get("twd_all_twd_sum",    pd.Series(0.0, index=v2_frame.index)) +
        v2_frame.get("crypto_all_twd_sum", pd.Series(0.0, index=v2_frame.index))
    )
    income = v2_frame.get("monthly_income_twd", pd.Series(1.0, index=v2_frame.index)).clip(lower=1.0)
    f["actual_volume_expected_ratio"] = (vol_total / income).values

    # Peer percentiles: not computed in v2 → 0 (peer-volume rules won't fire)
    f["fiat_in_30d_peer_pct"]         = 0.0
    f["crypto_withdraw_30d_peer_pct"] = 0.0

    # Inject snapshot_date so evaluate_rules() can group correctly
    f["snapshot_date"] = snapshot_date.date()

    rule_results = evaluate_rules(f)

    # Build output: rename rule flag columns, keep rule_score and rule_hit_count
    out = rule_results[["user_id"]].copy()
    for flag in _RULE_FLAG_COLS:
        src = flag
        dst = f"rule_{flag}"
        out[dst] = rule_results[src].astype(int) if src in rule_results.columns else 0

    out["rule_score"]     = rule_results["rule_score"].values     if "rule_score"     in rule_results.columns else 0.0
    out["rule_hit_count"] = rule_results["rule_hit_count"].values if "rule_hit_count" in rule_results.columns else 0

    return out
