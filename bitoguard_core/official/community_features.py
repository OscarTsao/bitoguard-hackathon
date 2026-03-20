from __future__ import annotations

"""Community detection features for the official fraud detection pipeline.

Louvain/greedy community detection identifies clusters of closely connected
users in the transaction graph. These features complement PPR propagation by
providing cluster-level risk signals: a fraud ring sitting in a high-risk
community is distinguishable from an isolated suspicious user.

Features produced (one row per user):
    community_id            - integer community label
    community_size          - number of users in the community
    community_pos_count     - labeled positives in the community
    community_pos_ratio     - fraction of labeled positives in the community
    community_ppr_sum       - sum of PPR scores within the community (if provided)
    community_max_degree    - max weighted degree centrality in the community
    is_high_risk_community  - bool: community_pos_ratio > 0.1
    cross_community_edges   - count of edges leaving the user's community
"""

import logging
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from official.graph_dataset import TransductiveGraph

logger = logging.getLogger(__name__)

# Community detection: try python-louvain, fall back to greedy modularity.
_LOUVAIN_AVAILABLE = False
try:
    import community as community_louvain  # python-louvain package

    _LOUVAIN_AVAILABLE = True
except ImportError:
    pass

# Community risk threshold: communities with positive ratio above this are
# flagged as high-risk. 0.10 (10%) mirrors the overall dataset positive rate
# and is a natural decision boundary.
_HIGH_RISK_COMMUNITY_THRESHOLD = 0.10

# Empty feature columns used as a zero-filled fallback on failure.
_COMMUNITY_FEATURE_COLUMNS = [
    "community_id",
    "community_size",
    "community_pos_count",
    "community_pos_ratio",
    "community_ppr_sum",
    "community_max_degree",
    "is_high_risk_community",
    "cross_community_edges",
]


def _zero_community_frame(user_ids: list[int]) -> pd.DataFrame:
    """Return a DataFrame with all community features set to zero."""
    frame = pd.DataFrame({"user_id": user_ids})
    for col in _COMMUNITY_FEATURE_COLUMNS:
        frame[col] = 0
    # Each isolated user is its own singleton community (id = row index + 1).
    frame["community_id"] = (frame.index + 1).astype(int)
    frame["community_size"] = 1
    return frame


def _build_nx_graph(graph: "TransductiveGraph") -> nx.Graph:
    """Build an undirected NetworkX graph from the collapsed user-user edges.

    Each node is a user_id (int). Edge weights come directly from the
    collapsed_edges DataFrame (which has already summed duplicate edges by
    (src, dst, edge_type) triple).  Where multiple edge types connect the same
    pair of users the weights are summed again here (since collapsed_edges may
    have one row per edge-type pair).
    """
    g: nx.Graph = nx.Graph()
    g.add_nodes_from(graph.user_ids)

    if graph.collapsed_edges.empty:
        return g

    edges = graph.collapsed_edges.copy()
    # Keep only user→user edges (both endpoints must be in the user set).
    user_set = set(graph.user_ids)
    mask = (
        edges["src_user_id"].astype(int).isin(user_set)
        & edges["dst_user_id"].astype(int).isin(user_set)
    )
    edges = edges[mask]

    if edges.empty:
        return g

    # Sum weights for the same (src, dst) pair across edge types.
    aggregated = (
        edges.groupby(["src_user_id", "dst_user_id"], as_index=False)["weight"]
        .sum()
    )
    for _, row in aggregated.iterrows():
        src = int(row["src_user_id"])
        dst = int(row["dst_user_id"])
        w = float(row["weight"])
        if g.has_edge(src, dst):
            g[src][dst]["weight"] += w
        else:
            g.add_edge(src, dst, weight=w)
    return g


def _detect_communities(g: nx.Graph) -> dict[int, int]:
    """Run community detection and return {user_id: community_id}.

    Strategy:
    1. Try python-louvain (best quality, randomised).
    2. Fall back to NetworkX greedy_modularity_communities (deterministic).
    3. Fall back to connected components (each component = one community).

    Disconnected / isolated nodes are guaranteed a community of their own.
    """
    # Always assign isolated (degree-0) nodes first so they get a stable id
    # regardless of which algorithm runs.
    assignment: dict[int, int] = {}
    next_id = 1

    if _LOUVAIN_AVAILABLE:
        try:
            # python-louvain returns {node: community_int} directly.
            raw = community_louvain.best_partition(g, weight="weight", random_state=42)
            # Remap to 1-based stable ids.
            seen: dict[int, int] = {}
            for node, cid in raw.items():
                if cid not in seen:
                    seen[cid] = next_id
                    next_id += 1
                assignment[int(node)] = seen[cid]
            # Fill any nodes not covered (isolated nodes not returned by louvain).
            for node in g.nodes():
                if int(node) not in assignment:
                    assignment[int(node)] = next_id
                    next_id += 1
            return assignment
        except Exception as exc:
            logger.warning("python-louvain failed (%s), falling back to greedy modularity.", exc)

    # NetworkX greedy modularity communities (returns frozensets).
    try:
        communities = list(nx.algorithms.community.greedy_modularity_communities(g, weight="weight"))
        for community_set in communities:
            cid = next_id
            next_id += 1
            for node in community_set:
                assignment[int(node)] = cid
        # Fill missing isolated nodes (greedy may omit degree-0 nodes in some NX versions).
        for node in g.nodes():
            if int(node) not in assignment:
                assignment[int(node)] = next_id
                next_id += 1
        return assignment
    except Exception as exc:
        logger.warning("greedy_modularity_communities failed (%s), falling back to components.", exc)

    # Last resort: each connected component is a community.
    for component in nx.connected_components(g):
        cid = next_id
        next_id += 1
        for node in component:
            assignment[int(node)] = cid
    return assignment


def build_community_features(
    graph: "TransductiveGraph",
    label_frame: pd.DataFrame,
    ppr_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute Louvain/greedy community detection features for all users.

    Parameters
    ----------
    graph:
        TransductiveGraph instance produced by build_transductive_graph().
    label_frame:
        DataFrame with columns [user_id, status] where status=1 means labeled
        positive.  Used only to compute community-level positive statistics.
        Should contain only *training* labels (no test-set labels) to avoid
        leakage.
    ppr_scores:
        Optional DataFrame with columns [user_id, <ppr_col>].  The first
        non-user_id numeric column is used as the PPR score for
        community_ppr_sum aggregation.  If None, community_ppr_sum is 0.

    Returns
    -------
    pd.DataFrame
        One row per user in graph.user_ids, columns:
        user_id, community_id, community_size, community_pos_count,
        community_pos_ratio, community_ppr_sum, community_max_degree,
        is_high_risk_community, cross_community_edges.
        All numeric columns default to 0 for isolated users or on error.
    """
    user_ids: list[int] = graph.user_ids

    if not user_ids:
        return pd.DataFrame(columns=["user_id", *_COMMUNITY_FEATURE_COLUMNS])

    try:
        # ── 1. Build NetworkX graph ──────────────────────────────────────────
        g = _build_nx_graph(graph)

        # ── 2. Community detection ───────────────────────────────────────────
        community_assignment = _detect_communities(g)

        # ── 3. Prepare label lookup ──────────────────────────────────────────
        lf = label_frame.copy()
        lf["user_id"] = pd.to_numeric(lf["user_id"], errors="coerce").astype("Int64")
        lf["status"] = pd.to_numeric(lf["status"], errors="coerce").fillna(0).astype(int)
        lf = lf.dropna(subset=["user_id"])
        lf["user_id"] = lf["user_id"].astype(int)
        positive_users: set[int] = set(lf[lf["status"] == 1]["user_id"].tolist())

        # ── 4. Prepare PPR lookup ────────────────────────────────────────────
        ppr_by_user: dict[int, float] = {}
        if ppr_scores is not None and not ppr_scores.empty:
            ppr_frame = ppr_scores.copy()
            ppr_frame["user_id"] = pd.to_numeric(ppr_frame["user_id"], errors="coerce").astype(int)
            # Use first non-user_id numeric column as the PPR score.
            numeric_cols = [
                c for c in ppr_frame.columns
                if c != "user_id" and pd.api.types.is_numeric_dtype(ppr_frame[c])
            ]
            if numeric_cols:
                ppr_by_user = dict(zip(ppr_frame["user_id"].tolist(), ppr_frame[numeric_cols[0]].tolist()))

        # ── 5. Degree centrality (weighted) ──────────────────────────────────
        # Weighted degree = sum of adjacent edge weights.
        weighted_degree: dict[int, float] = {
            int(node): float(sum(d["weight"] for _, d in g[node].items()))
            for node in g.nodes()
        }

        # ── 6. Build community-level aggregates ──────────────────────────────
        # community_id → list of member user_ids
        community_members: dict[int, list[int]] = {}
        for uid, cid in community_assignment.items():
            community_members.setdefault(cid, []).append(uid)

        community_size: dict[int, int] = {cid: len(members) for cid, members in community_members.items()}
        community_pos_count: dict[int, int] = {
            cid: sum(1 for uid in members if uid in positive_users)
            for cid, members in community_members.items()
        }
        community_pos_ratio: dict[int, float] = {
            cid: community_pos_count[cid] / community_size[cid]
            if community_size[cid] > 0 else 0.0
            for cid in community_members
        }
        community_ppr_sum: dict[int, float] = {
            cid: float(sum(ppr_by_user.get(uid, 0.0) for uid in members))
            for cid, members in community_members.items()
        }
        community_max_degree: dict[int, float] = {
            cid: float(max((weighted_degree.get(uid, 0.0) for uid in members), default=0.0))
            for cid, members in community_members.items()
        }

        # ── 7. Cross-community edges: edges from user to a different community ─
        # Pre-build user → community map for fast lookup.
        user_community: dict[int, int] = community_assignment
        cross_edges_count: dict[int, int] = {uid: 0 for uid in user_ids}
        if not graph.collapsed_edges.empty:
            edges = graph.collapsed_edges.copy()
            user_set = set(user_ids)
            mask = (
                edges["src_user_id"].astype(int).isin(user_set)
                & edges["dst_user_id"].astype(int).isin(user_set)
            )
            edges = edges[mask]
            for _, row in edges.iterrows():
                src = int(row["src_user_id"])
                dst = int(row["dst_user_id"])
                if user_community.get(src, -1) != user_community.get(dst, -2):
                    cross_edges_count[src] = cross_edges_count.get(src, 0) + 1

        # ── 8. Assemble output DataFrame ─────────────────────────────────────
        rows: list[dict] = []
        for uid in user_ids:
            cid = community_assignment.get(uid, 0)
            rows.append({
                "user_id": uid,
                "community_id": cid,
                "community_size": community_size.get(cid, 1),
                "community_pos_count": community_pos_count.get(cid, 0),
                "community_pos_ratio": community_pos_ratio.get(cid, 0.0),
                "community_ppr_sum": community_ppr_sum.get(cid, 0.0),
                "community_max_degree": community_max_degree.get(cid, 0.0),
                "is_high_risk_community": int(
                    community_pos_ratio.get(cid, 0.0) > _HIGH_RISK_COMMUNITY_THRESHOLD
                ),
                "cross_community_edges": cross_edges_count.get(uid, 0),
            })

        output = pd.DataFrame(rows)
        output["user_id"] = output["user_id"].astype("Int64")
        return output

    except Exception as exc:
        logger.error(
            "build_community_features failed (%s); returning zero-filled fallback.", exc, exc_info=True
        )
        return _zero_community_frame(user_ids)
