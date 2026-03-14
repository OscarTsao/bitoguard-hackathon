# bitoguard_core/features/graph_propagation.py
"""Per-fold label-aware propagation features (7 features).

LEAKAGE CONTRACT:
  `labels` must contain ONLY training-fold labels. Never pass test/validation
  user labels here. The stacker in models/stacker.py enforces this by passing
  only fold training indices.

Propagation is 1-hop: a user's prop_ip score = fraction of their IP entities
that are connected to at least one positive training user. This avoids the
multi-hop leakage risk of deeper propagation.
"""
from __future__ import annotations
from collections import defaultdict
import pandas as pd

_IP_EDGE_TYPES     = frozenset({"login_from_ip"})
_WALLET_EDGE_TYPES = frozenset({"owns_wallet", "crypto_transfer_to_wallet"})


def compute_label_propagation(
    edges:    pd.DataFrame,
    labels:   pd.Series,         # index=user_id, value=0/1, TRAINING FOLD ONLY
    user_ids: list[str],
) -> pd.DataFrame:
    """Compute 7 label-aware graph propagation features.

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

    if not edges.empty:
        for _, row in edges.iterrows():
            uid    = row.get("src_id")
            src_t  = row.get("src_type")
            dst_t  = row.get("dst_type")
            dst_id = row.get("dst_id")
            rel    = row.get("relation_type", "")
            if src_t != "user" or uid not in all_needed:
                continue
            if dst_t == "ip" and rel in _IP_EDGE_TYPES:
                user_ip[uid].add(dst_id)
                ip_all[dst_id].add(uid)
                if uid in pos_users:
                    ip_pos[dst_id].add(uid)
            elif dst_t == "wallet" and rel in _WALLET_EDGE_TYPES:
                user_wal[uid].add(dst_id)
                wal_all[dst_id].add(uid)
                if uid in pos_users:
                    wal_pos[dst_id].add(uid)

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
        rows.append(row)

    return pd.DataFrame(rows).fillna(0).reset_index(drop=True)
