# bitoguard_core/features/registry.py
"""Feature registry: assembles all v2 label-free modules into one master table.

build_v2_features() -> one row per user_id, ~155 columns (label-free).
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
        compute_crypto_features(crypto),
        compute_swap_features(trades),
        compute_trading_features(trades),
        compute_ip_features(logins),
        compute_sequence_features(fiat, trades, crypto),
        compute_bipartite_features(edges, user_ids),
    ]
    # Paired probe functions for modules that need special probing
    probe_fns = [
        None,
        lambda: compute_twd_features(_make_probe_fiat()),
        lambda: compute_crypto_features(_make_probe_crypto()),
        lambda: compute_swap_features(_make_probe_trades()),
        lambda: compute_trading_features(_make_probe_trades()),
        lambda: compute_ip_features(_make_probe_logins()),
        lambda: compute_sequence_features(_make_probe_fiat(), _make_probe_trades(), _make_probe_crypto()),
        None,
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
    return base.fillna(0).reset_index(drop=True)


def build_and_store_v2_features(
    users:         pd.DataFrame,
    fiat:          pd.DataFrame,
    crypto:        pd.DataFrame,
    trades:        pd.DataFrame,
    logins:        pd.DataFrame,
    edges:         pd.DataFrame,
    snapshot_date: pd.Timestamp | None = None,
    store=None,
) -> pd.DataFrame:
    """Compute v2 features and persist to features.feature_snapshots_v2."""
    from db.store import DuckDBStore, make_id
    from config import load_settings

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
    return master
