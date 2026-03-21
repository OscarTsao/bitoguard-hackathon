from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from math import log1p
from typing import Any

import numpy as np
import pandas as pd

from official.common import EVENT_TIME_COLUMNS, load_clean_table, to_utc_timestamp


MAX_WALLET_PAIRWISE_USERS = 50
MAX_IP_PAIRWISE_USERS = 20
# v47: Temporal co-occurrence cap — tighter than wallet (50) since time-bucket edges are
# weaker signal (many legitimate users transact at the same minute vs shared wallets which
# are almost always coordinated). Cap at 30 ensures only tight co-activity clusters get edges.
_MAX_TEMPORAL_COOCCURRENCE_USERS = 30

# P0-2: Known null/placeholder entity hash values to filter from IP and wallet edges.
# These are hashes of degenerate inputs (empty string, "0", "null", null byte) that
# would create artificial super-nodes linking thousands of unrelated users.
_SENTINEL_ENTITY_HASHES: frozenset[str] = frozenset({
    "cfcd208495d565ef66e7dff9f98764da",  # MD5("0")
    "d41d8cd98f00b204e9800998ecf8427e",  # MD5("")
    "37a6259cc0c1dae299a7866489dff0bd",  # MD5("null")
    "93b885adfe0da089cdf634904fd59f71",  # MD5(b"\x00")
    "4ae71336e44bf9bf79d2752e234818a5",  # MD5("NULL")
    "37a6259cc0c1dae299a7866489dff0bd",  # duplicate of MD5("null"), kept for clarity
})
_SENTINEL_DEGREE_GATE = 500  # Any entity connecting >500 unique users is treated as sentinel


def _filter_sentinel_entities(
    edges: pd.DataFrame,
    degree_gate: int = _SENTINEL_DEGREE_GATE,
) -> pd.DataFrame:
    """Remove edges involving sentinel/placeholder entity hash values.

    Two filters applied in order:
    1. Known sentinel hashes (MD5 of null-like inputs).
    2. Degree gate: any entity connecting > degree_gate unique users is a super-node.

    Called on ip_edges and wallet_edges BEFORE computing entity_user_count,
    so the count computation operates on clean data only.
    """
    if edges.empty or "entity_id" not in edges.columns:
        return edges

    # Filter 1: known sentinel hashes
    n_before = len(edges)
    mask_sentinel = edges["entity_id"].astype(str).str.lower().isin(_SENTINEL_ENTITY_HASHES)
    edges = edges[~mask_sentinel].copy()
    n_sentinel_removed = n_before - len(edges)

    # Filter 2: degree gate — compute per-entity user counts and drop mega-entities
    if len(edges) > 0:
        entity_user_counts = edges.groupby("entity_id")["user_id"].nunique()
        mega_entities = set(entity_user_counts[entity_user_counts > degree_gate].index)
        if mega_entities:
            mask_mega = edges["entity_id"].isin(mega_entities)
            edges = edges[~mask_mega].copy()
            if n_sentinel_removed > 0 or len(mega_entities) > 0:
                print(
                    f"  [P0-2] entity filter: removed {n_sentinel_removed} sentinel edges, "
                    f"{len(mega_entities)} mega-entities (>{degree_gate} users)"
                )
        elif n_sentinel_removed > 0:
            print(f"  [P0-2] entity filter: removed {n_sentinel_removed} sentinel edges")

    return edges

# v48: Default edge weights (these can be overridden by HPO via hpo_edge_weights.py).
# Weights represent edge reliability: relation (direct internal transfer) = strongest signal (1.0),
# wallet_small (2-10 shared users) = high (0.70), wallet_medium (11-50) = medium (0.40),
# ip_small (2-5 shared users) = high (0.50), ip_medium (6-20) = medium (0.25),
# temporal_small/medium = weak co-timing signal (0.30/0.15).
DEFAULT_EDGE_WEIGHTS: dict[str, float] = {
    "relation": 1.0,
    "wallet_small": 0.70,
    "wallet_medium": 0.40,
    "ip_small": 0.50,
    "ip_medium": 0.25,
    "temporal_small": 0.30,
    "temporal_medium": 0.15,
}


@dataclass(frozen=True)
class TransductiveGraph:
    user_ids: list[int]
    user_feature_frame: pd.DataFrame
    user_index: dict[int, int]
    relation_edges: pd.DataFrame
    wallet_edges: pd.DataFrame
    ip_edges: pd.DataFrame
    temporal_edges: pd.DataFrame
    collapsed_edges: pd.DataFrame
    component_id_by_user: dict[int, int]
    combined_neighbors: dict[int, list[tuple[int, float]]]
    neighbors_by_type: dict[str, dict[int, list[int]]]
    wallet_node_frame: pd.DataFrame
    ip_node_frame: pd.DataFrame


def _prepare_table(name: str, cutoff_ts: pd.Timestamp | None) -> pd.DataFrame:
    frame = load_clean_table(name).copy()
    if "user_id" in frame.columns:
        frame["user_id"] = pd.to_numeric(frame["user_id"], errors="coerce").astype("Int64")
    time_column = EVENT_TIME_COLUMNS.get(name)
    if time_column:
        frame[time_column] = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
        if cutoff_ts is not None:
            frame = frame[frame[time_column] < cutoff_ts].copy()
    return frame


def _pairwise_user_edges(
    edge_frame: pd.DataFrame,
    max_users: int,
    edge_type_small: str,
    edge_type_medium: str,
    small_upper_bound: int,
    weight_small: float,
    weight_medium: float,
) -> pd.DataFrame:
    if edge_frame.empty:
        return pd.DataFrame(columns=["src_user_id", "dst_user_id", "edge_type", "weight"])
    rows: list[dict[str, Any]] = []
    for _, group in edge_frame.groupby("entity_id"):
        users = sorted({int(user_id) for user_id in group["user_id"].dropna().tolist()})
        user_count = len(users)
        if user_count <= 1 or user_count > max_users:
            continue
        edge_type = edge_type_small if user_count <= small_upper_bound else edge_type_medium
        weight = weight_small if user_count <= small_upper_bound else weight_medium
        for left, right in combinations(users, 2):
            rows.append({"src_user_id": left, "dst_user_id": right, "edge_type": edge_type, "weight": weight})
            rows.append({"src_user_id": right, "dst_user_id": left, "edge_type": edge_type, "weight": weight})
    return pd.DataFrame(rows)


def _relation_user_edges(relation_edges: pd.DataFrame) -> pd.DataFrame:
    if relation_edges.empty:
        return pd.DataFrame(columns=["src_user_id", "dst_user_id", "edge_type", "weight"])
    rows: list[dict[str, Any]] = []
    for _, row in relation_edges.iterrows():
        left = int(row["user_id"])
        right = int(row["relation_user_id"])
        if left == right:
            continue
        rows.append({"src_user_id": left, "dst_user_id": right, "edge_type": "relation", "weight": 1.0})
        rows.append({"src_user_id": right, "dst_user_id": left, "edge_type": "relation", "weight": 1.0})
    return pd.DataFrame(rows).drop_duplicates()


def _entity_node_frame(edge_frame: pd.DataFrame, prefix: str, hub_cutoff: int) -> pd.DataFrame:
    if edge_frame.empty:
        return pd.DataFrame(columns=["entity_id", f"{prefix}_user_count", f"{prefix}_link_count", f"{prefix}_is_hub", f"{prefix}_log_user_count"])
    counts = edge_frame.groupby("entity_id").agg(
        user_count=("user_id", "nunique"),
        link_count=("user_id", "size"),
    ).reset_index()
    counts[f"{prefix}_user_count"] = counts["user_count"].astype(int)
    counts[f"{prefix}_link_count"] = counts["link_count"].astype(int)
    counts[f"{prefix}_is_hub"] = counts[f"{prefix}_user_count"].gt(hub_cutoff).astype(int)
    counts[f"{prefix}_log_user_count"] = counts[f"{prefix}_user_count"].map(lambda value: float(log1p(value)))
    return counts[["entity_id", f"{prefix}_user_count", f"{prefix}_link_count", f"{prefix}_is_hub", f"{prefix}_log_user_count"]]


def _temporal_cooccurrence_edges(
    twd: pd.DataFrame,
    crypto: pd.DataFrame,
    allowed_users: set[int],
    bucket_minutes: int = 15,
    weight_small_override: float = 0.30,
    weight_medium_override: float = 0.15,
) -> pd.DataFrame:
    """Create user-user edges for users who transact within the same time window.

    v47: Captures coordinated fraud ring activity where money mules deposit/withdraw
    at similar times. Distinct from wallet/IP sharing: ring members can use different
    infrastructure but coordinate timing (e.g., coordinated withdrawals after a deposit
    is confirmed). Tight 15-minute buckets exclude natural coincidence (many users
    trade at market open/close) while still catching coordinated short-window activity.

    Weight 0.30 (small cluster) / 0.15 (medium cluster): weaker than direct wallet
    sharing (0.70) but additive signal for users in the same fraud ring.
    """
    frames: list[pd.DataFrame] = []
    for table in (twd, crypto):
        if table.empty or "user_id" not in table.columns or "created_at" not in table.columns:
            continue
        sub = table[["user_id", "created_at"]].copy()
        sub["user_id"] = pd.to_numeric(sub["user_id"], errors="coerce")
        sub = sub.dropna(subset=["user_id", "created_at"])
        sub["user_id"] = sub["user_id"].astype(int)
        sub = sub[sub["user_id"].isin(allowed_users)]
        if not sub.empty:
            frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=["src_user_id", "dst_user_id", "edge_type", "weight"])
    combined = pd.concat(frames, ignore_index=True)
    # Floor timestamps to bucket_minutes-minute intervals — integer bucket key
    ts_series = pd.to_datetime(combined["created_at"], utc=True, errors="coerce")
    bucket_ns = int(bucket_minutes * 60 * 1_000_000_000)
    combined["entity_id"] = ts_series.astype("int64") // bucket_ns
    combined = combined.dropna(subset=["entity_id"])
    # One entry per (user, bucket) — if a user transacts 3 times in the same window
    # they still form only one edge with each co-occurring user (avoids weight inflation).
    combined = combined[["user_id", "entity_id"]].drop_duplicates()
    if combined.empty:
        return pd.DataFrame(columns=["src_user_id", "dst_user_id", "edge_type", "weight"])
    return _pairwise_user_edges(
        combined,
        max_users=_MAX_TEMPORAL_COOCCURRENCE_USERS,
        edge_type_small="temporal_small",
        edge_type_medium="temporal_medium",
        small_upper_bound=5,
        weight_small=weight_small_override,
        weight_medium=weight_medium_override,
    )


def _flow_user_edges(crypto: pd.DataFrame, allowed_users: set[int]) -> pd.DataFrame:
    """Build amount-weighted direct flow edges from crypto_transfer.relation_user_id.

    Captures ALL user-to-user crypto transfers (both internal and external) weighted
    by log(1 + total_amount_transferred). Unlike the existing 'relation' edges (binary,
    internal only), these provide the *amount dimension* of monetary flow between users.

    Weight is normalized to [0, 0.8] range (below 'relation'=1.0 and 'wallet_small'=0.70).
    """
    if "relation_user_id" not in crypto.columns:
        return pd.DataFrame(columns=["src_user_id", "dst_user_id", "edge_type", "weight"])

    flow = crypto[crypto["relation_user_id"].notna()].copy()
    if flow.empty:
        return pd.DataFrame(columns=["src_user_id", "dst_user_id", "edge_type", "weight"])

    flow["relation_user_id"] = pd.to_numeric(flow["relation_user_id"], errors="coerce")
    flow = flow.dropna(subset=["relation_user_id"])
    flow["user_id"] = pd.to_numeric(flow["user_id"], errors="coerce")
    flow = flow.dropna(subset=["user_id"])
    flow["user_id"] = flow["user_id"].astype(int)
    flow["relation_user_id"] = flow["relation_user_id"].astype(int)

    # Filter: both endpoints must be in the cohort
    flow = flow[
        flow["user_id"].isin(allowed_users) &
        flow["relation_user_id"].isin(allowed_users) &
        (flow["user_id"] != flow["relation_user_id"])
    ]
    if flow.empty:
        return pd.DataFrame(columns=["src_user_id", "dst_user_id", "edge_type", "weight"])

    # Aggregate total amount per directed pair, then symmetrize
    amt_col = "amount" if "amount" in flow.columns else None
    if amt_col:
        flow[amt_col] = pd.to_numeric(flow[amt_col], errors="coerce").fillna(0.0).clip(0)
        pair_agg = (
            flow.groupby(["user_id", "relation_user_id"])[amt_col]
            .sum()
            .reset_index()
        )
    else:
        pair_agg = (
            flow.groupby(["user_id", "relation_user_id"])
            .size()
            .reset_index(name=amt_col or "count")
        )
        pair_agg.rename(columns={pair_agg.columns[-1]: "amount"}, inplace=True)
        amt_col = "amount"

    # Log-scale weight, normalize to [0, 0.8]
    pair_agg["raw_w"] = np.log1p(pair_agg[amt_col].values).astype(float)
    max_w = pair_agg["raw_w"].max()
    if max_w > 0:
        pair_agg["weight"] = (pair_agg["raw_w"] / max_w * 0.8).clip(0.05, 0.8)
    else:
        pair_agg["weight"] = 0.4

    rows: list[dict[str, Any]] = []
    for _, row in pair_agg.iterrows():
        src, dst, wt = int(row["user_id"]), int(row["relation_user_id"]), float(row["weight"])
        rows.append({"src_user_id": src, "dst_user_id": dst, "edge_type": "flow", "weight": wt})
        rows.append({"src_user_id": dst, "dst_user_id": src, "edge_type": "flow", "weight": wt})

    result = pd.DataFrame(rows)
    result = result.drop_duplicates(subset=["src_user_id", "dst_user_id"])
    print(f"  [flow_graph_edges] {len(result)//2} user pairs, "
          f"{result['weight'].mean():.3f} mean weight")
    return result


def _component_id_map(user_ids: list[int], edges: pd.DataFrame) -> dict[int, int]:
    parent = {user_id: user_id for user_id in user_ids}

    def find(value: int) -> int:
        root = parent.setdefault(value, value)
        if root != value:
            parent[value] = find(root)
        return parent[value]

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for _, row in edges.iterrows():
        union(int(row["src_user_id"]), int(row["dst_user_id"]))
    roots = {user_id: find(user_id) for user_id in user_ids}
    root_to_component: dict[int, int] = {}
    component_map: dict[int, int] = {}
    for user_id, root in roots.items():
        if root not in root_to_component:
            root_to_component[root] = len(root_to_component) + 1
        component_map[user_id] = root_to_component[root]
    return component_map


def _neighbor_maps(collapsed_edges: pd.DataFrame) -> tuple[dict[int, list[tuple[int, float]]], dict[str, dict[int, list[int]]]]:
    combined: dict[int, list[tuple[int, float]]] = defaultdict(list)
    by_type: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    if collapsed_edges.empty:
        return {}, {}
    for _, row in collapsed_edges.iterrows():
        src = int(row["src_user_id"])
        dst = int(row["dst_user_id"])
        weight = float(row["weight"])
        edge_type = str(row["edge_type"])
        combined[src].append((dst, weight))
        by_type[edge_type][src].append(dst)
    return dict(combined), {edge_type: dict(mapping) for edge_type, mapping in by_type.items()}


def _numeric_user_feature_frame(dataset: pd.DataFrame) -> pd.DataFrame:
    excluded = {
        "user_id",
        "status",
        "cohort",
        "snapshot_cutoff_at",
        "snapshot_cutoff_tag",
        "top_reason_codes",
        "is_known_blacklist",
        "needs_prediction",
        "in_train_label",
        "in_predict_label",
        "is_shadow_overlap",
    }
    frame = dataset.copy()
    bool_columns = [column for column in frame.columns if pd.api.types.is_bool_dtype(frame[column])]
    for column in bool_columns:
        frame[column] = frame[column].fillna(False).astype(int)
    numeric_columns = [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    output = frame[["user_id", *numeric_columns]].copy()
    output[numeric_columns] = output[numeric_columns].fillna(0.0)
    return output


def build_transductive_graph(
    dataset: pd.DataFrame,
    cutoff_ts: pd.Timestamp | None = None,
    edge_weights: dict[str, float] | None = None,
    hub_ip_prune_above: int | None = None,
    use_time_decay: bool = False,
    time_decay_half_life_days: float = 90.0,
    use_flow_edges: bool = False,
) -> TransductiveGraph:
    """Build the full transductive user-user graph.

    Args:
        dataset: Full dataset with all users and features.
        cutoff_ts: Optional cutoff timestamp (only events before this are used).
        edge_weights: Optional dict overriding DEFAULT_EDGE_WEIGHTS for HPO experiments.
            Keys: 'relation', 'wallet_small', 'wallet_medium', 'ip_small', 'ip_medium',
                  'temporal_small', 'temporal_medium'.
            Missing keys fall back to DEFAULT_EDGE_WEIGHTS.
        hub_ip_prune_above: Optional int. When set, IP entities with more than this
            many unique users are pruned (dropped entirely) before building pairwise
            edges. This removes hub IPs (e.g. VPN exit nodes, corporate proxies) that
            create spurious connections. Default None = use MAX_IP_PAIRWISE_USERS (20).
        use_time_decay: If True, downweight edges involving users who have been
            inactive for longer. Uses per-user last-activity timestamp with
            exponential decay (half-life = time_decay_half_life_days). Stale
            connections (e.g. shared IP from 6 months ago) get lower weight than
            recent ones, reducing noise from dormant-account pairings.
        time_decay_half_life_days: Half-life for the exponential decay in days.
            Default 90: weight halves for every 3 months of inactivity.
    """
    # v48: Merge supplied weights with defaults (supplied weights take precedence).
    w = {**DEFAULT_EDGE_WEIGHTS, **(edge_weights or {})}

    cutoff_ts = to_utc_timestamp(cutoff_ts)
    user_feature_frame = _numeric_user_feature_frame(dataset)
    user_ids = sorted(user_feature_frame["user_id"].astype(int).tolist())
    allowed_users = set(user_ids)
    user_index = {user_id: idx for idx, user_id in enumerate(user_ids)}

    twd = _prepare_table("twd_transfer", cutoff_ts)
    crypto = _prepare_table("crypto_transfer", cutoff_ts)
    trade = _prepare_table("usdt_twd_trading", cutoff_ts)

    relation_edges = crypto[
        crypto["relation_user_id"].notna() & crypto["is_internal_transfer"].eq(True)
    ][["user_id", "relation_user_id"]].copy()
    relation_edges["relation_user_id"] = pd.to_numeric(relation_edges["relation_user_id"], errors="coerce").astype("Int64")
    relation_edges = relation_edges.dropna()
    relation_edges["user_id"] = relation_edges["user_id"].astype(int)
    relation_edges["relation_user_id"] = relation_edges["relation_user_id"].astype(int)
    relation_edges = relation_edges[
        relation_edges["user_id"].isin(allowed_users)
        & relation_edges["relation_user_id"].isin(allowed_users)
    ].drop_duplicates()

    wallet_edges = pd.concat(
        [
            crypto[["user_id", "from_wallet_hash"]].rename(columns={"from_wallet_hash": "entity_id"}),
            crypto[["user_id", "to_wallet_hash"]].rename(columns={"to_wallet_hash": "entity_id"}),
        ],
        ignore_index=True,
    )
    wallet_edges = wallet_edges[wallet_edges["entity_id"].notna()].copy()
    wallet_edges["user_id"] = pd.to_numeric(wallet_edges["user_id"], errors="coerce").astype("Int64")
    wallet_edges = wallet_edges.dropna(subset=["user_id"])
    wallet_edges["user_id"] = wallet_edges["user_id"].astype(int)
    wallet_edges = wallet_edges[wallet_edges["user_id"].isin(allowed_users)].drop_duplicates()
    # P0-2: Remove sentinel/placeholder wallet hashes before computing entity counts.
    wallet_edges = _filter_sentinel_entities(wallet_edges)
    wallet_counts = wallet_edges.groupby("entity_id")["user_id"].nunique().rename("entity_user_count").reset_index()
    wallet_edges = wallet_edges.merge(wallet_counts, on="entity_id", how="left")

    ip_edges = pd.concat(
        [
            twd[["user_id", "source_ip_hash"]].rename(columns={"source_ip_hash": "entity_id"}),
            crypto[["user_id", "source_ip_hash"]].rename(columns={"source_ip_hash": "entity_id"}),
            trade[["user_id", "source_ip_hash"]].rename(columns={"source_ip_hash": "entity_id"}),
        ],
        ignore_index=True,
    )
    ip_edges = ip_edges[ip_edges["entity_id"].notna()].copy()
    ip_edges["user_id"] = pd.to_numeric(ip_edges["user_id"], errors="coerce").astype("Int64")
    ip_edges = ip_edges.dropna(subset=["user_id"])
    ip_edges["user_id"] = ip_edges["user_id"].astype(int)
    ip_edges = ip_edges[ip_edges["user_id"].isin(allowed_users)].drop_duplicates()
    # P0-2: Remove sentinel/placeholder IP hashes before computing entity counts.
    ip_edges = _filter_sentinel_entities(ip_edges)
    ip_counts = ip_edges.groupby("entity_id")["user_id"].nunique().rename("entity_user_count").reset_index()
    ip_edges = ip_edges.merge(ip_counts, on="entity_id", how="left")

    # Build pairwise user-user edges using (potentially HPO-tuned) weights.
    relation_user_edges = _relation_user_edges(relation_edges)
    # Override relation weight if specified.
    if relation_user_edges is not None and not relation_user_edges.empty and w["relation"] != 1.0:
        relation_user_edges = relation_user_edges.copy()
        relation_user_edges["weight"] = w["relation"]

    wallet_user_edges = _pairwise_user_edges(
        wallet_edges[wallet_edges["entity_user_count"] <= MAX_WALLET_PAIRWISE_USERS],
        max_users=MAX_WALLET_PAIRWISE_USERS,
        edge_type_small="wallet_small",
        edge_type_medium="wallet_medium",
        small_upper_bound=10,
        weight_small=w["wallet_small"],
        weight_medium=w["wallet_medium"],
    )
    # v4/configurable: hub_ip_prune_above — prune IPs shared by too many users.
    _ip_max = hub_ip_prune_above if hub_ip_prune_above is not None else MAX_IP_PAIRWISE_USERS
    ip_user_edges = _pairwise_user_edges(
        ip_edges[ip_edges["entity_user_count"] <= _ip_max],
        max_users=_ip_max,
        edge_type_small="ip_small",
        edge_type_medium="ip_medium",
        small_upper_bound=5,
        weight_small=w["ip_small"],
        weight_medium=w["ip_medium"],
    )
    # v47: Temporal co-occurrence edges — users transacting in the same 15-minute window.
    temporal_user_edges = _temporal_cooccurrence_edges(
        twd, crypto, allowed_users,
        weight_small_override=w["temporal_small"],
        weight_medium_override=w["temporal_medium"],
    )
    # Phase 3: amount-weighted direct transfer flow edges
    flow_user_edges = pd.DataFrame(columns=["src_user_id", "dst_user_id", "edge_type", "weight"])
    if use_flow_edges:
        flow_user_edges = _flow_user_edges(crypto, allowed_users)

    _edge_frames = [relation_user_edges, wallet_user_edges, ip_user_edges, temporal_user_edges]
    if use_flow_edges and not flow_user_edges.empty:
        _edge_frames.append(flow_user_edges)
    collapsed_edges = pd.concat(_edge_frames, ignore_index=True)
    if collapsed_edges.empty:
        collapsed_edges = pd.DataFrame(columns=["src_user_id", "dst_user_id", "edge_type", "weight"])
    else:
        collapsed_edges = (
            collapsed_edges.groupby(["src_user_id", "dst_user_id", "edge_type"], as_index=False)["weight"]
            .sum()
            .sort_values(["src_user_id", "dst_user_id", "edge_type"])
            .reset_index(drop=True)
        )

    # ── Edge time decay: downweight connections involving recently-inactive users ──
    if use_time_decay and not collapsed_edges.empty:
        _ts_frames: list[pd.DataFrame] = []
        for _tbl in (twd, crypto, trade):
            if "created_at" in _tbl.columns and "user_id" in _tbl.columns:
                _sub = _tbl[["user_id", "created_at"]].copy()
                _sub["user_id"] = pd.to_numeric(_sub["user_id"], errors="coerce")
                _sub = _sub.dropna(subset=["user_id", "created_at"])
                _sub["user_id"] = _sub["user_id"].astype(int)
                _sub["created_at"] = pd.to_datetime(_sub["created_at"], utc=True, errors="coerce")
                _sub = _sub.dropna(subset=["created_at"])
                _ts_frames.append(_sub.rename(columns={"created_at": "ts"}))
        if _ts_frames:
            _all_ts = pd.concat(_ts_frames, ignore_index=True)
            _last_active = _all_ts.groupby("user_id")["ts"].max()
            _global_max = _last_active.max()
            # days since each user's last event (0 = maximally recent, larger = more stale)
            _days_ago = (_global_max - _last_active).dt.total_seconds() / 86400.0
            _days_ago = _days_ago.clip(lower=0.0)
            # exp decay: weight multiplier ∈ (0, 1], = 0.5 at half_life_days
            _decay = np.exp(-np.log(2.0) * _days_ago / time_decay_half_life_days)
            _decay_map: dict[int, float] = _decay.to_dict()
            _src_d = collapsed_edges["src_user_id"].map(_decay_map).fillna(1.0).to_numpy()
            _dst_d = collapsed_edges["dst_user_id"].map(_decay_map).fillna(1.0).to_numpy()
            # Edge weight = base_weight × min(decay_src, decay_dst)
            # min() ensures the edge is as stale as the stalest participant
            collapsed_edges = collapsed_edges.copy()
            collapsed_edges["weight"] = (
                collapsed_edges["weight"].to_numpy() * np.minimum(_src_d, _dst_d)
            ).astype("float32")
            print(f"  [edge_time_decay] half_life={time_decay_half_life_days}d; "
                  f"mean_decay={np.minimum(_src_d, _dst_d).mean():.3f} "
                  f"(edges={len(collapsed_edges)})")

    component_id_by_user = _component_id_map(user_ids, collapsed_edges[["src_user_id", "dst_user_id"]].drop_duplicates())
    combined_neighbors, neighbors_by_type = _neighbor_maps(collapsed_edges)
    wallet_node_frame = _entity_node_frame(wallet_edges, "wallet", MAX_WALLET_PAIRWISE_USERS)
    ip_node_frame = _entity_node_frame(ip_edges, "ip", MAX_IP_PAIRWISE_USERS)

    return TransductiveGraph(
        user_ids=user_ids,
        user_feature_frame=user_feature_frame,
        user_index=user_index,
        relation_edges=relation_edges.reset_index(drop=True),
        wallet_edges=wallet_edges.reset_index(drop=True),
        ip_edges=ip_edges.reset_index(drop=True),
        temporal_edges=temporal_user_edges.reset_index(drop=True),
        collapsed_edges=collapsed_edges,
        component_id_by_user=component_id_by_user,
        combined_neighbors=combined_neighbors,
        neighbors_by_type=neighbors_by_type,
        wallet_node_frame=wallet_node_frame,
        ip_node_frame=ip_node_frame,
    )
