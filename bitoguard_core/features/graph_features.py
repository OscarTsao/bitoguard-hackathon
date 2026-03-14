"""Graph feature computation.

SAFETY NOTE — graph_trusted_only mode (default: enabled)
  When settings.graph_trusted_only is True (the default), the following
  unsafe features are returned as zeros / defaults and NOT computed:
    - shared_device_count   (A7: placeholder super-node artifact)
    - component_size        (A7: inflated by same super-node)
    - blacklist_1hop_count  (A5: label leakage via graph proximity)
    - blacklist_2hop_count  (A5: label leakage via graph proximity)

  Only trusted features (fan_out_ratio, shared_wallet_count, shared_bank_count)
  are computed in trusted mode.

  To re-enable unsafe features, set:
    BITOGUARD_GRAPH_FEATURES_TRUSTED_ONLY=false

  Do NOT do this until the graph recovery plan (docs/GRAPH_RECOVERY_PLAN.md)
  has been fully executed and all 4 M5 audit checks pass.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import networkx as nx
import pandas as pd

from config import UNSAFE_GRAPH_FEATURES, load_settings
from db.store import DuckDBStore
from features.build_features import iter_eligible_users_by_snapshot

logger = logging.getLogger(__name__)


def _prefix(entity_type: str, entity_id: str) -> str:
    return f"{entity_type}:{entity_id}"


def _node_type(node_id: str) -> str:
    return node_id.split(":", 1)[0]


def _other_users_via_type(graph: nx.Graph, user_node: str, via_type: str) -> set[str]:
    related: set[str] = set()
    for neighbor in graph.neighbors(user_node):
        if _node_type(neighbor) != via_type:
            continue
        for second in graph.neighbors(neighbor):
            if second != user_node and _node_type(second) == "user":
                related.add(second.split(":", 1)[1])
    return related


def _empty_graph_features() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "graph_feature_id", "user_id", "snapshot_date", "shared_device_count", "shared_bank_count",
        "shared_wallet_count", "blacklist_1hop_count", "blacklist_2hop_count", "component_size", "fan_out_ratio",
    ])


def _normalize_snapshot_dates(snapshot_dates: object) -> pd.DatetimeIndex:
    if isinstance(snapshot_dates, pd.Timestamp) or not isinstance(snapshot_dates, (pd.Index, pd.Series, list, tuple, set)):
        values = [snapshot_dates]
    else:
        values = list(snapshot_dates)
    normalized = pd.DatetimeIndex(pd.to_datetime(values, utc=True)).tz_localize(None).normalize()
    return pd.DatetimeIndex(sorted(normalized.unique()))


def _build_graph_features_fast(
    store: "DuckDBStore",
    target_user_ids: set[str],
    snapshot_dates: pd.DatetimeIndex,
    trusted_only: bool,
) -> pd.DataFrame:
    """Fast path: build graph ONCE at latest snapshot, replicate across all dates.

    This avoids rebuilding an nx.Graph for every one of the 32 snapshot_dates
    (which was O(32 × |edges|)) and instead does a single build at the latest
    snapshot then replicates the static features across all dates.
    The fan_out_ratio is computed from the snapshot‑bounded crypto edges but uses
    the same single‑pass graph; temporal precision is acceptable for the hackathon.
    """
    import time as _time
    t0 = _time.time()
    edges_df = store.read_table("canonical.entity_edges")
    crypto_all = store.read_table("canonical.crypto_transactions")
    blacklist_feed = store.read_table("canonical.blacklist_feed")
    blacklist_feed["observed_at"] = pd.to_datetime(blacklist_feed["observed_at"], utc=True)

    edges_df["snapshot_time"] = pd.to_datetime(edges_df["snapshot_time"], utc=True)

    # Use full edge set (latest snapshot = max snapshot_time)
    edge_window = edges_df.copy()

    # Build 1-hop neighborhood of target users to keep the graph tractable
    user_edges = edge_window[edge_window["src_type"] == "user"]
    direct_edges = user_edges[user_edges["src_id"].isin(target_user_ids)]
    if direct_edges.empty:
        records = []
        for uid in target_user_ids:
            for sd in snapshot_dates:
                records.append({
                    "graph_feature_id": f"gf_{uid}_{sd.date().isoformat()}",
                    "user_id": uid, "snapshot_date": pd.Timestamp(sd),
                    "shared_device_count": 0, "shared_bank_count": 0,
                    "shared_wallet_count": 0, "blacklist_1hop_count": 0,
                    "blacklist_2hop_count": 0, "component_size": 1, "fan_out_ratio": 0.0,
                })
        return pd.DataFrame(records)

    touched_targets = direct_edges[["dst_type", "dst_id"]].drop_duplicates()
    neighborhood_edges = user_edges.merge(touched_targets, on=["dst_type", "dst_id"], how="inner")
    graph_source = (
        pd.concat([direct_edges, neighborhood_edges], ignore_index=True)
        .drop_duplicates(subset=["edge_id"])
    )

    # Build graph once
    graph = nx.Graph()
    for _, row in graph_source.iterrows():
        src = _prefix(row["src_type"], row["src_id"])
        dst = _prefix(row["dst_type"], row["dst_id"])
        graph.add_edge(src, dst)

    # Per-target-user: compute static graph features once.
    # Blacklist-dependent features (blacklist_1hop_count, blacklist_2hop_count) are
    # intentionally excluded here and computed per snapshot date in the loop below
    # to prevent label leakage from future blacklist entries.
    per_user: dict[str, dict] = {}
    # Store per-user neighbor distances for use in per-snapshot blacklist computation.
    per_user_neighbor_distances: dict[str, dict[str, int]] = {}
    for uid in target_user_ids:
        node = _prefix("user", uid)
        if node not in graph:
            per_user[uid] = {
                "shared_device_count": 0, "shared_bank_count": 0,
                "shared_wallet_count": 0, "blacklist_1hop_count": 0,
                "blacklist_2hop_count": 0, "component_size": 1,
            }
            per_user_neighbor_distances[uid] = {}
            continue

        shared_banks = _other_users_via_type(graph, node, "bank_account")
        shared_wallets = _other_users_via_type(graph, node, "wallet")

        if trusted_only:
            shared_device_count = 0
            component_size = 1
            # Blacklist hop counts will be zeroed per snapshot date too
            neighbor_user_distances: dict[str, int] = {}
        else:
            shared_devices = _other_users_via_type(graph, node, "device")
            shared_device_count = len(shared_devices)
            component_size = len(nx.node_connected_component(graph, node))
            lengths = nx.single_source_shortest_path_length(graph, node, cutoff=4)
            neighbor_user_distances = {
                target_node.split(":", 1)[1]: distance
                for target_node, distance in lengths.items()
                if _node_type(target_node) == "user" and target_node.split(":", 1)[1] != uid
            }

        per_user[uid] = {
            "shared_device_count": shared_device_count,
            "shared_bank_count": len(shared_banks),
            "shared_wallet_count": len(shared_wallets),
            # blacklist hop counts are placeholders; overwritten per snapshot below
            "blacklist_1hop_count": 0,
            "blacklist_2hop_count": 0,
            "component_size": component_size,
        }
        per_user_neighbor_distances[uid] = neighbor_user_distances

    # Compute fan_out_ratio per user per snapshot_date from crypto transactions
    crypto_all["occurred_at"] = pd.to_datetime(crypto_all["occurred_at"], utc=True)
    crypto_user = crypto_all[crypto_all["user_id"].isin(target_user_ids)].copy()

    logger.info(
        "build_graph_features fast path: graph built in %.1fs, %d target users, %d snapshot_dates",
        _time.time() - t0, len(per_user), len(snapshot_dates),
    )

    # Replicate static features across all snapshot dates, with per-date fan_out
    # and per-date blacklist to prevent label leakage.
    records = []
    for sd in snapshot_dates:
        snapshot_end = pd.Timestamp(sd, tz="UTC") + pd.Timedelta(days=1)
        sd_end = snapshot_end
        crypto_window = crypto_user[crypto_user["occurred_at"] < sd_end]
        transfer_counts = crypto_window.groupby("user_id").size().to_dict()
        distinct_targets = (
            crypto_window.groupby("user_id")["to_wallet"].nunique().to_dict()
            if "to_wallet" in crypto_window.columns else {}
        )

        # Compute blacklisted_set bounded to this snapshot date to prevent leakage
        blacklisted_set = set(
            blacklist_feed[blacklist_feed["observed_at"] < snapshot_end]["user_id"].astype(str)
        )

        for uid, feats in per_user.items():
            total = transfer_counts.get(uid, 0)
            distinct = distinct_targets.get(uid, 0)
            fan_out_ratio = distinct / total if total > 0 else 0.0

            # Compute blacklist hop counts using snapshot-bounded blacklisted_set
            if trusted_only:
                blacklist_1hop = 0
                blacklist_2hop = 0
            else:
                blacklist_1hop = 0
                blacklist_2hop = 0
                for neighbor_uid, distance in per_user_neighbor_distances.get(uid, {}).items():
                    if neighbor_uid not in blacklisted_set:
                        continue
                    if distance <= 2:
                        blacklist_1hop += 1
                    elif distance <= 4:
                        blacklist_2hop += 1

            records.append({
                "graph_feature_id": f"gf_{uid}_{sd.date().isoformat()}",
                "user_id": uid,
                "snapshot_date": pd.Timestamp(sd),
                **{k: v for k, v in feats.items() if k not in ("blacklist_1hop_count", "blacklist_2hop_count")},
                "blacklist_1hop_count": blacklist_1hop,
                "blacklist_2hop_count": blacklist_2hop,
                "fan_out_ratio": fan_out_ratio,
            })

    return pd.DataFrame(records)


def build_graph_features(
    snapshot_dates: pd.DatetimeIndex | None = None,
    target_user_ids: set[str] | None = None,
    persist: bool = True,
) -> pd.DataFrame:
    settings = load_settings()
    trusted_only = settings.graph_trusted_only
    if trusted_only:
        logger.info(
            "build_graph_features: graph_trusted_only=True — "
            "unsafe features (%s) will be zeroed. "
            "Set BITOGUARD_GRAPH_FEATURES_TRUSTED_ONLY=false only after "
            "completing docs/GRAPH_RECOVERY_PLAN.md.",
            ", ".join(sorted(UNSAFE_GRAPH_FEATURES)),
        )
    store = DuckDBStore(settings.db_path)
    edges = store.read_table("canonical.entity_edges")
    users = store.read_table("canonical.users")
    blacklist_feed = store.read_table("canonical.blacklist_feed")
    fiat = store.read_table("canonical.fiat_transactions")
    trade = store.read_table("canonical.trade_orders")
    crypto = store.read_table("canonical.crypto_transactions")
    login = store.read_table("canonical.login_events")
    target_user_filter = set(target_user_ids) if target_user_ids is not None else None
    if users.empty:
        empty = _empty_graph_features()
        if persist:
            store.replace_table("features.graph_features", empty)
        return empty

    edges["snapshot_time"] = pd.to_datetime(edges["snapshot_time"], utc=True)
    users["created_at"] = pd.to_datetime(users["created_at"], utc=True)
    blacklist_feed["observed_at"] = pd.to_datetime(blacklist_feed["observed_at"], utc=True)
    fiat["occurred_at"] = pd.to_datetime(fiat["occurred_at"], utc=True)
    trade["occurred_at"] = pd.to_datetime(trade["occurred_at"], utc=True)
    crypto["occurred_at"] = pd.to_datetime(crypto["occurred_at"], utc=True)
    login["occurred_at"] = pd.to_datetime(login["occurred_at"], utc=True)
    crypto_edges = edges[edges["relation_type"] == "crypto_transfer_to_wallet"].copy()
    crypto_edges["snapshot_time"] = pd.to_datetime(crypto_edges["snapshot_time"], utc=True)

    if target_user_filter is not None:
        users = users[users["user_id"].isin(target_user_filter)].copy()
        fiat = fiat[fiat["user_id"].isin(target_user_filter)].copy()
        trade = trade[trade["user_id"].isin(target_user_filter)].copy()
        crypto = crypto[crypto["user_id"].isin(target_user_filter)].copy()
        login = login[login["user_id"].isin(target_user_filter)].copy()
        crypto_edges = crypto_edges[crypto_edges["src_id"].isin(target_user_filter)].copy()

        # Resolve snapshot_dates before fast path so we can pass them in
        if snapshot_dates is not None:
            _snap_dates = _normalize_snapshot_dates(snapshot_dates)
        elif not edges.empty:
            date_start = edges["snapshot_time"].dt.date.min()
            date_end = edges["snapshot_time"].dt.date.max()
            _snap_dates = pd.date_range(date_start, date_end, freq="D")
        else:
            _snap_dates = pd.DatetimeIndex([])

        if len(_snap_dates) > 0:
            logger.info(
                "build_graph_features: using fast single-pass path for %d target users, %d dates",
                len(target_user_filter), len(_snap_dates),
            )
            dataframe = _build_graph_features_fast(store, target_user_filter, _snap_dates, trusted_only)
            if persist:
                store.replace_table("features.graph_features", dataframe)
            return dataframe

    if snapshot_dates is None:
        if edges.empty:
            empty = _empty_graph_features()
            if persist:
                store.replace_table("features.graph_features", empty)
            return empty
        date_start = edges["snapshot_time"].dt.date.min()
        date_end = edges["snapshot_time"].dt.date.max()
        snapshot_dates = pd.date_range(date_start, date_end, freq="D")
    else:
        snapshot_dates = _normalize_snapshot_dates(snapshot_dates)

    if len(snapshot_dates) == 0:
        empty = _empty_graph_features()
        if persist:
            store.replace_table("features.graph_features", empty)
        return empty

    records: list[dict] = []

    for snapshot_date, snapshot_end, eligible_users, blacklisted_users in iter_eligible_users_by_snapshot(
        users,
        snapshot_dates,
        blacklist_feed,
        (fiat, "occurred_at"),
        (trade, "occurred_at"),
        (crypto, "occurred_at"),
        (login, "occurred_at"),
    ):
        edge_window = edges[edges["snapshot_time"] < snapshot_end]
        active_users = eligible_users
        if target_user_filter is not None:
            active_users = users[users["created_at"] < snapshot_end].copy()
        if active_users.empty:
            continue

        graph_source = edge_window
        if target_user_filter is not None:
            user_edges = edge_window[edge_window["src_type"] == "user"].copy()
            direct_edges = user_edges[user_edges["src_id"].isin(active_users["user_id"])].copy()
            if direct_edges.empty:
                graph_source = direct_edges
            else:
                touched_targets = direct_edges[["dst_type", "dst_id"]].drop_duplicates()
                neighborhood_edges = user_edges.merge(
                    touched_targets,
                    on=["dst_type", "dst_id"],
                    how="inner",
                )
                graph_source = (
                    pd.concat([direct_edges, neighborhood_edges], ignore_index=True)
                    .drop_duplicates(subset=["edge_id"])
                )
                # Keep only TARGET users for feature output (neighborhood is scaffolding only)
                active_users = active_users[active_users["user_id"].isin(target_user_filter)].copy()
            if active_users.empty:
                continue

        graph = nx.Graph()
        for _, row in graph_source.iterrows():
            src = _prefix(row["src_type"], row["src_id"])
            dst = _prefix(row["dst_type"], row["dst_id"])
            graph.add_edge(src, dst, relation_type=row["relation_type"])

        transfer_counts = defaultdict(int)
        distinct_transfer_targets = defaultdict(set)
        for _, row in crypto_edges[crypto_edges["snapshot_time"] < snapshot_end].iterrows():
            transfer_counts[row["src_id"]] += 1
            distinct_transfer_targets[row["src_id"]].add(row["dst_id"])

        for _, user in active_users.iterrows():
            user_id = user["user_id"]
            node = _prefix("user", user_id)

            # Trusted features: always computed (wallet sharing, fan_out)
            if node not in graph:
                shared_banks = set()
                shared_wallets = set()
            else:
                shared_banks = _other_users_via_type(graph, node, "bank_account")
                shared_wallets = _other_users_via_type(graph, node, "wallet")

            total_transfers = transfer_counts[user_id]
            fan_out_ratio = (
                len(distinct_transfer_targets[user_id]) / total_transfers
                if total_transfers > 0 else 0.0
            )

            # Unsafe features: only computed when graph_trusted_only=False.
            # Default (trusted_only=True): return safe defaults.
            if trusted_only:
                shared_device_count = 0
                component_size = 1
                blacklist_1hop = 0
                blacklist_2hop = 0
            else:
                if node not in graph:
                    shared_devices: set[str] = set()
                    component_size = 1
                    blacklist_1hop = 0
                    blacklist_2hop = 0
                else:
                    shared_devices = _other_users_via_type(graph, node, "device")
                    component_size = len(nx.node_connected_component(graph, node))
                    blacklist_1hop = 0
                    blacklist_2hop = 0
                    lengths = nx.single_source_shortest_path_length(graph, node, cutoff=4)
                    for target, distance in lengths.items():
                        if _node_type(target) != "user":
                            continue
                        target_user_id = target.split(":", 1)[1]
                        if target_user_id == user_id or target_user_id not in blacklisted_users:
                            continue
                        if distance <= 2:
                            blacklist_1hop += 1
                        elif distance <= 4:
                            blacklist_2hop += 1
                shared_device_count = len(shared_devices)

            records.append({
                "graph_feature_id": f"gf_{user_id}_{snapshot_date.date().isoformat()}",
                "user_id": user_id,
                "snapshot_date": pd.Timestamp(snapshot_date),
                "shared_device_count": shared_device_count,
                "shared_bank_count": len(shared_banks),
                "shared_wallet_count": len(shared_wallets),
                "blacklist_1hop_count": blacklist_1hop,
                "blacklist_2hop_count": blacklist_2hop,
                "component_size": component_size,
                "fan_out_ratio": fan_out_ratio,
            })

    dataframe = pd.DataFrame(records)
    if persist:
        store.replace_table("features.graph_features", dataframe)
    return dataframe


if __name__ == "__main__":
    build_graph_features()
