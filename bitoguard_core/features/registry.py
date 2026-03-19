# bitoguard_core/features/registry.py
"""Feature registry: assembles all v2 feature modules into one master table.

build_v2_features() -> one row per user_id, ~168 columns.
  Modules: profile, twd, crypto, swap, trading, ip, sequence, bipartite, rule_signals.
build_and_store_v2_features() -> writes to features.feature_snapshots_v2.

Note: graph_propagation (label-aware, 7 features) is NOT included here.
It is added separately during model training (models/stacker.py) per-fold.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

from features.profile_features  import compute_profile_features, build_profile_category_maps
from features.twd_features      import compute_twd_features
from features.crypto_features   import compute_crypto_features
from features.swap_features     import compute_swap_features
from features.trading_features  import compute_trading_features
from features.ip_features       import compute_ip_features
from features.sequence_features import compute_sequence_features
from features.graph_bipartite   import compute_bipartite_features
from features.rule_features     import compute_rule_features
from features.typology_features import compute_typology_features
from features.event_ngram_features import compute_event_ngram_features
from features.statistical_features import compute_statistical_features
from features.dormancy import compute_dormancy_score

FEATURE_VERSION_V2 = "v2"

# Sentinel user_id used only for schema probing — never appears in real data.
_PROBE_UID = "__registry_probe__"


def _probe_schema(fn, *probe_args) -> list[str]:
    """Return the non-user_id column names that `fn(*probe_args)` would produce."""
    try:
        result = fn(*probe_args)
        if result is not None and not result.empty and "user_id" in result.columns:
            return [c for c in result.columns if c != "user_id"]
    except Exception:
        pass
    return []


def _make_probe_fiat():
    return pd.DataFrame([{
        "user_id": _PROBE_UID, "occurred_at": "2025-01-01T00:00:00+00:00",
        "direction": "deposit", "amount_twd": 1.0,
    }])


def _make_probe_crypto():
    return pd.DataFrame([{
        "user_id": _PROBE_UID, "occurred_at": "2025-01-01T00:00:00+00:00",
        "direction": "deposit", "amount_twd_equiv": 1.0,
        "asset": "BTC", "network": "BTC",
        "wallet_id": "w0", "counterparty_wallet_id": "ext0",
    }])


def _make_probe_trades():
    return pd.DataFrame([{
        "user_id": _PROBE_UID, "occurred_at": "2025-01-01T00:00:00+00:00",
        "side": "buy", "base_asset": "USDT", "quote_asset": "TWD",
        "notional_twd": 1.0, "order_type": "instant_swap",
    }])


def _make_probe_logins():
    return pd.DataFrame([{
        "user_id": _PROBE_UID, "occurred_at": "2025-01-01T00:00:00+00:00",
        "ip_address": "0.0.0.0",
    }])


def build_v2_features(
    users:          pd.DataFrame,
    fiat:           pd.DataFrame,
    crypto:         pd.DataFrame,
    trades:         pd.DataFrame,
    logins:         pd.DataFrame,
    edges:          pd.DataFrame,
    snapshot_date:  pd.Timestamp | None = None,
    category_maps:  dict | None = None,
) -> pd.DataFrame:
    """Assemble all label-free feature modules. Returns one row per user_id.

    When a module's input table is empty its columns are still included in the
    output (filled with 0) so that the feature schema is stable regardless of
    which tables happen to be populated.
    """
    user_ids = users["user_id"].dropna().unique().tolist()
    base = pd.DataFrame({"user_id": user_ids})

    module_entries: list[pd.DataFrame | None] = [
        compute_profile_features(users, snapshot_date=snapshot_date,
                                  category_maps=category_maps),
        compute_twd_features(fiat, snapshot_date=snapshot_date),
        compute_crypto_features(crypto, snapshot_date=snapshot_date),
        compute_swap_features(trades),
        compute_trading_features(trades),
        compute_ip_features(logins),
        compute_sequence_features(fiat, trades, crypto),
        compute_bipartite_features(edges, user_ids, snapshot_date=snapshot_date),
        compute_event_ngram_features(fiat, crypto, trades),
        compute_statistical_features(fiat, crypto, trades),
    ]
    # Paired probe functions for modules that need special probing
    probe_fns = [
        None,
        lambda: compute_twd_features(_make_probe_fiat()),
        lambda: compute_crypto_features(_make_probe_crypto(), snapshot_date=snapshot_date),
        lambda: compute_swap_features(_make_probe_trades()),
        lambda: compute_trading_features(_make_probe_trades()),
        lambda: compute_ip_features(_make_probe_logins()),
        lambda: compute_sequence_features(_make_probe_fiat(), _make_probe_trades(), _make_probe_crypto()),
        None,
        None,  # event_ngram_features — no probe needed (handles empty inputs gracefully)
        None,  # statistical_features — no probe needed (handles empty inputs gracefully)
    ]

    for module_df, probe_fn in zip(module_entries, probe_fns):
        if module_df is None:
            continue
        cols = getattr(module_df, "columns", [])
        if "user_id" not in cols or (module_df.empty and len(cols) <= 1):
            # Result has no useful columns — probe for the schema
            if probe_fn is not None:
                try:
                    schema_df = probe_fn()
                    if schema_df is not None and not schema_df.empty and "user_id" in schema_df.columns:
                        zero_cols = [c for c in schema_df.columns if c != "user_id" and c not in base.columns]
                        for col in zero_cols:
                            base[col] = 0
                except Exception:
                    pass
            continue
        if module_df.empty:
            # Has column schema but no rows — add zero-filled columns
            new_cols = [c for c in module_df.columns if c != "user_id" and c not in base.columns]
            for col in new_cols:
                base[col] = 0
            continue
        new_cols = [c for c in module_df.columns if c not in base.columns or c == "user_id"]
        base = base.merge(module_df[new_cols], on="user_id", how="left")

    # fillna(0) covers users absent from a partial module result
    base = base.fillna(0)

    # --- Cross-channel derived features ---
    # Computed BEFORE rule signals so that xch_layering_intensity is available
    # to the layering_burst rule in compute_rule_features().
    if "twd_dep_sum" in base.columns and "crypto_wdr_twd_sum" in base.columns:
        base["xch_cashout_ratio_lifetime"] = (
            base["crypto_wdr_twd_sum"] / (base["twd_dep_sum"] + 1.0)
        ).clip(upper=10.0)

    if "twd_dep_7d_sum" in base.columns and "crypto_wdr_7d_sum" in base.columns:
        base["xch_cashout_ratio_7d"] = (
            base["crypto_wdr_7d_sum"] / (base["twd_dep_7d_sum"] + 1.0)
        ).clip(upper=10.0)

    if "twd_dep_30d_sum" in base.columns and "crypto_wdr_30d_sum" in base.columns:
        base["xch_cashout_ratio_30d"] = (
            base["crypto_wdr_30d_sum"] / (base["twd_dep_30d_sum"] + 1.0)
        ).clip(upper=10.0)

    # Layering intensity: cross-channel velocity burst (crypto out burst + fiat in burst)
    # Must be before rule signals — layering_burst rule fires when this >= 5.0.
    if "twd_dep_burst_ratio" in base.columns and "crypto_wdr_burst_ratio" in base.columns:
        base["xch_layering_intensity"] = (
            base["twd_dep_burst_ratio"] * base["crypto_wdr_burst_ratio"]
        ).clip(upper=100.0)

    # --- High-signal interaction features (derived from top-importance features) ---
    # These combine the top LightGBM gain features to pre-compute interactions
    # that tree models may miss when features span many decision boundaries.

    # TRX/TRC20 exposure × withdrawal volume: high-value USDT-Tron transfers are
    # the dominant AML vector in Asian crypto (Elliptic 2023, CipherTrace 2024).
    if "crypto_trx_tx_share" in base.columns and "crypto_wdr_twd_sum" in base.columns:
        base["trx_volume_signal"] = (
            base["crypto_trx_tx_share"] * (base["crypto_wdr_twd_sum"].clip(upper=1e7) / 1e6)
        ).clip(upper=50.0)

    # Early activity density: high volume in first 3 days relative to total age
    # captures mule accounts that transact immediately and then slow down.
    if "early_3d_volume" in base.columns and "account_age_days" in base.columns:
        base["early_activity_ratio"] = (
            base["early_3d_volume"] / (base["account_age_days"].clip(lower=1.0) * 1000.0)
        ).clip(upper=10.0)

    # Cross-channel cashout × TRX share: high cashout ratio with TRX preference
    # captures the USDT-Tron layering pattern (fiat in → swap to USDT-TRC20 → out).
    if "xch_cashout_ratio_lifetime" in base.columns and "crypto_trx_tx_share" in base.columns:
        base["trx_cashout_signal"] = (
            base["xch_cashout_ratio_lifetime"] * base["crypto_trx_tx_share"]
        ).clip(upper=10.0)

    # Early-burst × cashout: mule accounts transact heavily in first 3 days AND cash out
    # immediately. This multiplicative interaction is the strongest known mule signal.
    if "xch_cashout_ratio_7d" in base.columns and "early_3d_volume" in base.columns:
        base["cashout_early_burst"] = (
            base["xch_cashout_ratio_7d"] * (base["early_3d_volume"].clip(upper=1e6) / 1e5)
        ).clip(upper=10.0)

    # Night-hour cashout: nocturnal cash-out pattern (money mules often operate off-hours)
    if "trade_night_share" in base.columns and "xch_cashout_ratio_lifetime" in base.columns:
        base["night_cashout_signal"] = (
            base["trade_night_share"] * base["xch_cashout_ratio_lifetime"]
        ).clip(upper=5.0)

    # Rule trigger × volume: rule hits weighted by withdrawal volume amplify signal for
    # high-volume suspicious patterns while suppressing low-volume false positives.
    if "rule_hit_count" in base.columns and "crypto_wdr_twd_sum" in base.columns:
        base["rule_volume_signal"] = (
            base["rule_hit_count"] * (base["crypto_wdr_twd_sum"].clip(upper=1e7) / 1e6)
        ).clip(upper=20.0)

    # --- Module: FATF AML typology signals ---
    # Inserted after cross-channel derived features so xch_cashout_ratio_7d/lifetime
    # are already present. Best-effort: failures result in missing columns filled to 0.
    try:
        typology_feats = compute_typology_features(base)
        if typology_feats is not None and not typology_feats.empty:
            for col in typology_feats.columns:
                if col == "user_id":
                    continue
                if col not in base.columns:
                    base[col] = typology_feats[col]
    except Exception:
        pass  # Typology features are best-effort; missing -> 0 via fillna below

    # Rule signals: evaluate M1 rules on the assembled v2 frame (including cross-channel)
    # and add outputs as features. This lets the stacker learn interactions between
    # rule firings and behavioral features (e.g., fast_cash_out_2h AND high crypto volume).
    try:
        rule_df = compute_rule_features(base, snapshot_date=snapshot_date)
        if rule_df is not None and not rule_df.empty:
            new_rule_cols = [c for c in rule_df.columns if c not in base.columns or c == "user_id"]
            base = base.merge(rule_df[new_rule_cols], on="user_id", how="left")
    except Exception:
        pass  # Rule features are best-effort; missing → 0 via fillna below

    # fillna(0) for rule columns and any remaining NaN
    base = base.fillna(0)

    # Task 1: Add explicit dormancy score (fraction of behavioral columns that are zero)
    base["dormancy_score"] = compute_dormancy_score(base)

    return base.reset_index(drop=True)


def build_and_store_v2_features(
    users:         pd.DataFrame,
    fiat:          pd.DataFrame,
    crypto:        pd.DataFrame,
    trades:        pd.DataFrame,
    logins:        pd.DataFrame,
    edges:         pd.DataFrame,
    snapshot_date: pd.Timestamp | None = None,
    store=None,
    export_to_s3: bool = False,
) -> pd.DataFrame:
    """Compute v2 features and persist to features.feature_snapshots_v2.
    
    Args:
        export_to_s3: If True, export feature snapshot to S3 (requires AWS credentials)
    """
    from db.store import DuckDBStore, make_id
    from config import load_settings
    import os
    import logging

    logger = logging.getLogger(__name__)

    if snapshot_date is None:
        snapshot_date = pd.Timestamp.now(tz="UTC").normalize()

    settings = load_settings()
    artifact_dir = settings.artifact_dir
    if store is None:
        store = DuckDBStore(settings.db_path)

    # Build and save deterministic category maps from current users population
    category_maps = build_profile_category_maps(users)
    maps_path = artifact_dir / "profile_category_maps.json"
    maps_path.parent.mkdir(parents=True, exist_ok=True)
    maps_path.write_text(
        json.dumps(category_maps, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    master = build_v2_features(users, fiat, crypto, trades, logins, edges,
                               snapshot_date=snapshot_date,
                               category_maps=category_maps)
    if master.empty:
        return master

    master.insert(0, "feature_snapshot_id",
                  [make_id(f"v2_{uid[-4:]}") for uid in master["user_id"]])
    master.insert(2, "snapshot_date", snapshot_date.date())
    master.insert(3, "feature_version", FEATURE_VERSION_V2)

    store.replace_table("features.feature_snapshots_v2", master)
    
    # Export to S3 if enabled
    if export_to_s3:
        try:
            from ml_pipeline.feature_store import FeatureStore
            
            # Get S3 configuration from environment
            bucket_name = os.environ.get("BITOGUARD_ML_ARTIFACTS_BUCKET")
            region = os.environ.get("AWS_REGION", "us-east-1")
            
            if not bucket_name:
                logger.warning("S3 export requested but BITOGUARD_ML_ARTIFACTS_BUCKET not set")
            else:
                feature_store = FeatureStore(
                    bucket_name=bucket_name,
                    prefix="features",
                    local_cache_dir=str(artifact_dir / "feature_cache"),
                    region_name=region
                )
                
                # Generate snapshot ID from timestamp
                snapshot_id = snapshot_date.strftime("%Y%m%dT%H%M%SZ")
                
                # Prepare metadata
                metadata = {
                    "feature_version": FEATURE_VERSION_V2,
                    "snapshot_date": str(snapshot_date.date()),
                    "user_count": len(master),
                    "feature_count": len(master.columns),
                    "category_maps": category_maps
                }
                
                # Save to S3
                snapshot = feature_store.save_snapshot(
                    df=master,
                    snapshot_id=snapshot_id,
                    metadata=metadata,
                    partition_date=snapshot_date.to_pydatetime()
                )
                
                logger.info(
                    f"Exported feature snapshot to S3: {snapshot.s3_path} "
                    f"({snapshot.row_count} rows, {snapshot.feature_count} features)"
                )
        
        except Exception as e:
            logger.error(f"Failed to export features to S3: {e}")
            # Don't fail the entire pipeline if S3 export fails
    
    return master
