# bitoguard_core/features/graph_propagation.py
"""Per-fold label-aware propagation features (17 features).

LEAKAGE CONTRACT:
  `labels` must contain ONLY training-fold labels. Never pass test/validation
  user labels here. The stacker in models/stacker.py enforces this by passing
  only fold training indices.

Propagation is 1-hop: a user's prop_ip score = fraction of their IP entities
that are connected to at least one positive training user. This avoids the
multi-hop leakage risk of deeper propagation.
"""
from __future__ import annotations
from collections import defaultdict, deque
import pandas as pd
import numpy as np
from scipy import sparse

_IP_REL_TYPES      = frozenset({"login_from_ip"})
_WALLET_REL_TYPES  = frozenset({"owns_wallet", "crypto_transfer_to_wallet"})
_RELATION_DST_TYPE = "user"


def compute_label_propagation(
    edges:    pd.DataFrame,
    labels:   pd.Series,         # index=user_id, value=0/1, TRAINING FOLD ONLY
    user_ids: list[str],
) -> pd.DataFrame:
    """Compute 17 label-aware graph propagation features.

    Args:
        edges:    canonical.entity_edges
        labels:   training-fold label Series (index=user_id)
        user_ids: users to score (typically includes both train and val users)
    """
    pos_users: set[str] = set(labels[labels == 1].index)
    all_needed = set(user_ids) | set(labels.index)

    ip_pos:     defaultdict[str, set[str]] = defaultdict(set)
    ip_all:     defaultdict[str, set[str]] = defaultdict(set)
    wal_pos:    defaultdict[str, set[str]] = defaultdict(set)
    wal_all:    defaultdict[str, set[str]] = defaultdict(set)
    user_ip:    defaultdict[str, set[str]] = defaultdict(set)
    user_wal:   defaultdict[str, set[str]] = defaultdict(set)

    rel_pos: defaultdict[str, set[str]] = defaultdict(set)
    rel_all: defaultdict[str, set[str]] = defaultdict(set)
    adj: defaultdict[str, set[str]] = defaultdict(set)

    if not edges.empty:
        for _, row in edges.iterrows():
            uid    = row.get("src_id")
            src_t  = row.get("src_type")
            dst_t  = row.get("dst_type")
            dst_id = row.get("dst_id")
            rel    = row.get("relation_type", "")
            if src_t != "user" or uid not in all_needed:
                continue
            if dst_t == "ip" and rel in _IP_REL_TYPES:
                user_ip[uid].add(dst_id)
                ip_all[dst_id].add(uid)
                if uid in pos_users:
                    ip_pos[dst_id].add(uid)
            elif dst_t == "wallet" and rel in _WALLET_REL_TYPES:
                user_wal[uid].add(dst_id)
                wal_all[dst_id].add(uid)
                if uid in pos_users:
                    wal_pos[dst_id].add(uid)
            elif dst_t == _RELATION_DST_TYPE and dst_id in all_needed:
                rel_all[dst_id].add(uid)
                if uid in pos_users:
                    rel_pos[dst_id].add(uid)
                adj[uid].add(dst_id)
                adj[dst_id].add(uid)

    # Build projected adjacency from shared IP entities
    for entity_users in ip_all.values():
        users_list = list(entity_users)
        if len(users_list) < 2 or len(users_list) > 200:
            continue
        for u in users_list:
            if u in all_needed:
                for v in users_list:
                    if v != u:
                        adj[u].add(v)

    # Build projected adjacency from shared wallet entities
    for entity_users in wal_all.values():
        users_list = list(entity_users)
        if len(users_list) < 2 or len(users_list) > 200:
            continue
        for u in users_list:
            if u in all_needed:
                for v in users_list:
                    if v != u:
                        adj[u].add(v)

    # Multi-source BFS capped at depth=2
    bfs_dist: dict[str, int] = {u: 0 for u in pos_users}
    queue: deque[str] = deque(sorted(pos_users))
    while queue:
        curr = queue.popleft()
        nd = bfs_dist[curr] + 1
        if nd > 2:
            continue
        for nb in adj.get(curr, set()):
            if nb not in bfs_dist:
                bfs_dist[nb] = nd
                queue.append(nb)

    # PPR computation
    all_uid_list = list(all_needed)
    uid_idx: dict[str, int] = {u: i for i, u in enumerate(all_uid_list)}
    n_nodes = len(all_uid_list)
    ppr_scores: np.ndarray = np.zeros(n_nodes, dtype=np.float32)

    if n_nodes > 0 and pos_users:
        src_list, dst_list, w_list = [], [], []
        for u, nbrs in adj.items():
            if u not in uid_idx:
                continue
            ui = uid_idx[u]
            for v in nbrs:
                if v in uid_idx:
                    src_list.append(ui)
                    dst_list.append(uid_idx[v])
                    w_list.append(1.0)
        if src_list:
            A = sparse.csr_matrix(
                (np.array(w_list, dtype=np.float32),
                 (np.array(src_list, dtype=np.int32),
                  np.array(dst_list, dtype=np.int32))),
                shape=(n_nodes, n_nodes),
            )
            row_sum = np.asarray(A.sum(axis=1)).ravel()
            row_sum[row_sum == 0.0] = 1.0
            D_inv = sparse.diags(1.0 / row_sum)
            A_norm = D_inv @ A
            seed_vec = np.zeros(n_nodes, dtype=np.float32)
            for u in pos_users:
                if u in uid_idx:
                    seed_vec[uid_idx[u]] = 1.0
            if seed_vec.sum() > 0:
                seed_vec /= seed_vec.sum()
            alpha = 0.35
            ppr_scores = seed_vec.copy()
            for _ in range(20):
                ppr_scores = alpha * seed_vec + (1.0 - alpha) * (A_norm @ ppr_scores)

    # Union-Find for components
    parent: dict[str, str] = {u: u for u in all_needed}

    def _find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent.get(x, x), x)
            x = parent[x]
        return x

    def _union(a: str, b: str) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for u, nbrs in adj.items():
        if u in all_needed:
            for v in nbrs:
                if v in all_needed:
                    _union(u, v)

    comp_members: defaultdict[str, set[str]] = defaultdict(set)
    for u in all_needed:
        comp_members[_find(u)].add(u)
    comp_pos_count: dict[str, int] = {
        root: sum(1 for u in members if u in pos_users)
        for root, members in comp_members.items()
    }
    comp_size: dict[str, int] = {root: len(members) for root, members in comp_members.items()}

    rows = []
    for uid in user_ids:
        ip_ents  = list(user_ip.get(uid, set()))
        wal_ents = list(user_wal.get(uid, set()))
        row: dict = {"user_id": uid}

        row["prop_ip"] = float(
            sum(1 for e in ip_ents if ip_pos.get(e)) / max(1, len(ip_ents))
        ) if ip_ents else 0.0

        row["prop_wallet"] = float(
            sum(1 for e in wal_ents if wal_pos.get(e)) / max(1, len(wal_ents))
        ) if wal_ents else 0.0

        row["prop_combined"] = float(max(row["prop_ip"], row["prop_wallet"]))

        row["ip_rep_max_rate"] = float(max(
            (len(ip_pos.get(e, set())) / max(1, len(ip_all.get(e, set()))) for e in ip_ents),
            default=0.0,
        ))
        row["wallet_rep_max_rate"] = float(max(
            (len(wal_pos.get(e, set())) / max(1, len(wal_all.get(e, set()))) for e in wal_ents),
            default=0.0,
        ))

        has_pos_ip  = any(ip_pos.get(e)  for e in ip_ents)
        has_pos_wal = any(wal_pos.get(e) for e in wal_ents)
        row["rel_has_pos_neighbor"]  = int(has_pos_ip or has_pos_wal)
        row["rel_direct_pos_count"]  = int(
            sum(1 for e in ip_ents  if ip_pos.get(e)) +
            sum(1 for e in wal_ents if wal_pos.get(e))
        )

        # Feature 8: bfs_dist_1
        row["bfs_dist_1"] = int(bfs_dist.get(uid, -1) == 1)
        # Feature 9: bfs_dist_2
        d = bfs_dist.get(uid, -1)
        row["bfs_dist_2"] = int(0 < d <= 2)
        # Feature 10: ppr_score
        row["ppr_score"] = float(ppr_scores[uid_idx[uid]] if uid in uid_idx else 0.0)
        # Features 11-13: per-edge-type positive-neighbor counts
        row["pos_neighbor_count_relation"] = int(
            len(rel_pos.get(uid, set())) +
            sum(len(rel_pos.get(v, set())) for v in adj.get(uid, set()) if v in pos_users)
        )
        row["pos_neighbor_count_wallet"] = int(sum(1 for e in wal_ents if wal_pos.get(e)))
        row["pos_neighbor_count_ip"] = int(sum(1 for e in ip_ents if ip_pos.get(e)))
        # Features 14-15: entity-level max seed rate
        def _max_seed_rate(ents, pos_map, all_map, self_id):
            rates = []
            for e in ents:
                p = len(pos_map.get(e, set()) - {self_id})
                a = max(1, len(all_map.get(e, set())))
                rates.append(p / a)
            return float(max(rates)) if rates else 0.0
        row["entity_wallet_max_seed_rate"] = _max_seed_rate(wal_ents, wal_pos, wal_all, uid)
        row["entity_ip_max_seed_rate"] = _max_seed_rate(ip_ents, ip_pos, ip_all, uid)
        # Features 16-17: component seed statistics
        root = _find(uid)
        sz = comp_size.get(root, 1)
        pc = comp_pos_count.get(root, 0)
        row["component_seed_fraction"] = float(pc / max(1, sz))
        row["component_seed_count"] = int(pc)

        rows.append(row)

    return pd.DataFrame(rows).fillna(0).reset_index(drop=True)
