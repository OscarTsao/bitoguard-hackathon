# bitoguard_core/tests/test_graph_bipartite.py
from __future__ import annotations
import pandas as pd
import pytest
from features.graph_bipartite import compute_bipartite_features


def _edges_df():
    return pd.DataFrame([
        {"src_type": "user", "src_id": "u1", "relation_type": "login_from_ip",             "dst_type": "ip",     "dst_id": "ip1"},
        {"src_type": "user", "src_id": "u2", "relation_type": "login_from_ip",             "dst_type": "ip",     "dst_id": "ip1"},
        {"src_type": "user", "src_id": "u1", "relation_type": "owns_wallet",               "dst_type": "wallet", "dst_id": "w1"},
        {"src_type": "user", "src_id": "u1", "relation_type": "crypto_transfer_to_wallet", "dst_type": "wallet", "dst_id": "ext1"},
    ])


def test_bipartite_features_columns():
    result = compute_bipartite_features(_edges_df(), ["u1", "u2", "u3"])
    for col in ["ip_n_entities", "ip_total_event_count", "wallet_n_entities",
                "rel_out_degree", "graph_is_isolated"]:
        assert col in result.columns


def test_bipartite_u1_ip():
    result = compute_bipartite_features(_edges_df(), ["u1", "u2"])
    u1 = result[result["user_id"] == "u1"].iloc[0]
    assert u1["ip_n_entities"] == 1     # connected to ip1
    assert u1["wallet_n_entities"] >= 1


def test_bipartite_isolated_user():
    result = compute_bipartite_features(_edges_df(), ["u3"])
    u3 = result[result["user_id"] == "u3"].iloc[0]
    assert u3["graph_is_isolated"] == 1
    assert u3["ip_n_entities"] == 0


from features.graph_propagation import compute_label_propagation


def test_propagation_reaches_neighbor():
    edges = _edges_df()
    # u2 is positive; u1 shares ip1 with u2 → u1 should get IP propagation signal
    labels = pd.Series({"u2": 1, "u1": 0})
    result = compute_label_propagation(edges, labels, user_ids=["u1"])
    u1 = result[result["user_id"] == "u1"].iloc[0]
    assert u1["prop_ip"] > 0.0


def test_propagation_columns():
    labels = pd.Series({"u1": 1, "u2": 0})
    result = compute_label_propagation(_edges_df(), labels, user_ids=["u1", "u2"])
    for col in ["prop_ip", "prop_wallet", "prop_combined",
                "ip_rep_max_rate", "wallet_rep_max_rate",
                "rel_has_pos_neighbor", "rel_direct_pos_count"]:
        assert col in result.columns


def test_propagation_no_leakage():
    """Test user absent from labels still gets correct propagation."""
    edges = _edges_df()
    labels = pd.Series({"u2": 1})   # u1 not in labels (it's the test user)
    result = compute_label_propagation(edges, labels, user_ids=["u1"])
    u1 = result[result["user_id"] == "u1"].iloc[0]
    # u1 receives signal from u2 (training) via shared ip1 — this is correct, not leakage
    assert u1["prop_ip"] > 0.0
