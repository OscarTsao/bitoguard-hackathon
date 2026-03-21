"""Configurable pipeline for V4 ablation experiments.

Wraps the existing ``run_transductive_oof_pipeline`` with component-level flags
that enable/disable individual model improvements. Each component can be toggled
independently via a ``config: dict[str, bool]`` dictionary.

Usage:
    cd bitoguard_core
    PYTHONPATH=. python -c "
    from official.configurable_pipeline import run_configurable_pipeline, DEFAULT_CONFIG
    result = run_configurable_pipeline(DEFAULT_CONFIG, experiment_id='baseline')
    print(result)
    "
"""
from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

# ---------------------------------------------------------------------------
# Component registry
# ---------------------------------------------------------------------------
COMPONENTS: dict[str, dict[str, Any]] = {
    "self_training":            {"tier": "S", "default": False},
    "multi_scale_ppr":          {"tier": "S", "default": True},     # already in v45
    "negative_propagation":     {"tier": "S", "default": False},
    "edge_weight_hpo":          {"tier": "A", "default": False},
    "temporal_edges":           {"tier": "A", "default": True},     # already in v47
    "threshold_hpo":            {"tier": "A", "default": False},
    "pu_learning_loss":         {"tier": "A", "default": False},
    "node2vec_embeddings":      {"tier": "A", "default": False},
    "graphsage_3layer":         {"tier": "B", "default": False},    # no-GNN wins (+0.012 F1, GNN AP=0.061)
    "gbm_stacker":              {"tier": "B", "default": False},
    "profile_similarity_edges": {"tier": "B", "default": False},
    "hub_ip_pruning":           {"tier": "B", "default": False},
    "directed_flow_edges":      {"tier": "B", "default": False},
    "edge_time_decay":          {"tier": "B", "default": False},
    "feature_eng_node_attrs":   {"tier": "B", "default": False},
    "label_spreading":          {"tier": "B", "default": False},
    # Tier D: new methods
    "community_features":       {"tier": "D", "default": True},     # +0.012 F1, confirmed 10+ seeds
    "lag_features":             {"tier": "D", "default": False},
    "gru_sequence_branch":      {"tier": "D", "default": False},
    "dgi_rf_features":          {"tier": "D", "default": False},
    "nnpu_weights":             {"tier": "D", "default": False},
    "hard_negative_mining":     {"tier": "D", "default": False},
    "seed_perturbation_avg":    {"tier": "D", "default": False},
    # v50: Selective C&S restore — only restore top-5% isolated users by base_a score.
    # Avoids the -0.015 regression from blanket restore_isolated=True (which restores
    # all ~42K isolated users including ~41K negatives). Top-5% targets ~2,100 users
    # with highest base_a score among isolated, capturing most isolated positives.
    "conditional_cs_restore":   {"tier": "D", "default": False},
    # v50: Base B transductive-only — train Base B on graph/PPR features only (not tabular+PPR).
    # Reduces dilution from 239 tabular features in Base B, allowing it to specialize in
    # graph propagation signal. Base A already covers tabular; stacker combines both.
    "base_b_transductive_only": {"tier": "D", "default": False},
    # P0-5: Uncertainty entropy filter for self_training pseudo-label selection.
    # Rejects pseudo-labels where H(p) > max_entropy (0.5 nats ≈ p>=0.77).
    # Prevents noisy borderline pseudo-labels from degrading transductive features.
    "uncertainty_entropy_filter": {"tier": "P0", "default": False},
    # Phase 2A: Temporal sequence features from 707K raw events.
    # Captures AML patterns invisible to 239-column aggregates:
    # structuring (round deposits), layering (fiat→swap→crypto timing),
    # burst patterns, IP diversity, cross-channel cycle speed.
    # Expected AP contribution: +0.02-0.05. Label-free, pandas-only.
    "temporal_sequence_features": {"tier": "D", "default": False},
    # Phase 2A-2: Raw event sequence features — 20 novel features from 707K events.
    # Distinct from temporal_features.py (window aggregates): these capture event-level
    # patterns — min inter-deposit gap, 30min burst windows, identical-amount runs,
    # precise chain latency, IP Shannon entropy, wallet HHI, night/weekend concentration.
    # Estimated AP +0.02-0.05; no DL required (pure pandas).
    "sequence_features":            {"tier": "D", "default": False},
    # Phase 3: Transaction flow graph — direct user-to-user crypto transfer edges.
    # Uses crypto_transfer.relation_user_id to build amount-weighted flow edges.
    # Complements entity graph (IP/wallet sharing) with direct monetary flow signal.
    # Expected to reconnect 44.8% isolated positives via their crypto counterparties.
    "flow_graph_edges":             {"tier": "D", "default": False},
    # v53: LR stacker — forces use_blend=False, using fold-by-fold LogisticRegression.
    # Tests v13 hypothesis: LR can assign nonzero weight to weak graph signals (AP<0.08)
    # that BlendEnsemble's hard threshold drops. V13 used LR and got F1=0.390 vs
    # BlendEnsemble's ~0.382. Especially valuable when combined with graphsage_3layer=True.
    "lr_stacker":               {"tier": "A", "default": False},
}

DEFAULT_CONFIG: dict[str, bool] = {name: comp["default"] for name, comp in COMPONENTS.items()}

# ---------------------------------------------------------------------------
# Module-level feature cache — avoids rebuilding identical features between
# sequential experiments in the same process (~30s saved per experiment).
# ---------------------------------------------------------------------------
_FEATURE_CACHE: dict[str, Any] = {}


def _cached_base_dataset() -> pd.DataFrame:
    """Load and cache the base dataset + rules (identical across all experiments)."""
    if "base_dataset" not in _FEATURE_CACHE:
        from official.train import _load_dataset
        from official.rules import evaluate_official_rules
        ds = _load_dataset("full")
        rule_df = evaluate_official_rules(ds)
        _rule_cols_to_drop = [c for c in rule_df.columns if c != "user_id" and c in ds.columns]
        if _rule_cols_to_drop:
            ds = ds.drop(columns=_rule_cols_to_drop)
        ds = ds.merge(rule_df, on="user_id", how="left")
        _FEATURE_CACHE["base_dataset"] = ds
        print("[cache] Base dataset loaded and cached.")
    return _FEATURE_CACHE["base_dataset"].copy()


def _cached_sequence_features(dataset: pd.DataFrame) -> pd.DataFrame | None:
    """Build and cache sequence features (only depends on raw events, not config)."""
    if "sequence_features" not in _FEATURE_CACHE:
        try:
            from official.sequence_features import build_sequence_features
            _FEATURE_CACHE["sequence_features"] = build_sequence_features(dataset)
            print("[cache] Sequence features built and cached.")
        except Exception as e:
            print(f"[cache] sequence_features failed: {e}")
            _FEATURE_CACHE["sequence_features"] = None
    return _FEATURE_CACHE["sequence_features"]


def _cached_temporal_features(dataset: pd.DataFrame) -> pd.DataFrame | None:
    """Build and cache temporal sequence features (skip_layering=True, default fast mode)."""
    if "temporal_features" not in _FEATURE_CACHE:
        try:
            from official.temporal_features import build_temporal_features
            _FEATURE_CACHE["temporal_features"] = build_temporal_features(dataset, skip_layering=True)
            print("[cache] Temporal features built and cached.")
        except Exception as e:
            print(f"[cache] temporal_features failed: {e}")
            _FEATURE_CACHE["temporal_features"] = None
    return _FEATURE_CACHE["temporal_features"]


def _cached_lag_features(dataset: pd.DataFrame) -> pd.DataFrame | None:
    """Build and cache lag/cross-channel correlation features."""
    if "lag_features" not in _FEATURE_CACHE:
        try:
            from official.lag_features import build_lag_features
            _FEATURE_CACHE["lag_features"] = build_lag_features(dataset)
            print("[cache] Lag features built and cached.")
        except Exception as e:
            print(f"[cache] lag_features failed: {e}")
            _FEATURE_CACHE["lag_features"] = None
    return _FEATURE_CACHE["lag_features"]




# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _quick_oof_f1(
    oof_frame: pd.DataFrame,
    original_user_ids: set[int],
) -> tuple[float, float, float, float, float, int]:
    """Compute best OOF F1 + associated metrics against original labels.

    Returns (f1, precision, recall, pr_auc, threshold, n_flagged).
    """
    eval_frame = oof_frame[oof_frame["user_id"].astype(int).isin(original_user_ids)].copy()
    if eval_frame.empty or "stacker_raw_probability" not in eval_frame.columns:
        return 0.0, 0.0, 0.0, 0.0, 0.10, 0

    labels = eval_frame["status"].astype(int).values
    probs = eval_frame["stacker_raw_probability"].values

    pr_auc = float(average_precision_score(labels, probs))

    best_f1, best_thr = 0.0, 0.10
    for thr in np.arange(0.04, 0.65, 0.005):
        f1 = float(f1_score(labels, (probs >= thr).astype(int), zero_division=0))
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)

    preds = (probs >= best_thr).astype(int)
    prec = float(precision_score(labels, preds, zero_division=0))
    rec = float(recall_score(labels, preds, zero_division=0))
    n_flagged = int(preds.sum())

    return best_f1, prec, rec, pr_auc, best_thr, n_flagged


def _fit_lgbm_stacker(oof_frame: pd.DataFrame, stacker_cols: list[str]) -> tuple[pd.DataFrame, float]:
    """Fit a shallow LightGBM stacker on OOF meta-features and return (frame, f1).

    Uses 5-fold OOF evaluation on the ``primary_fold`` column.
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        return oof_frame, -1.0

    fold_col = "primary_fold"
    if fold_col not in oof_frame.columns:
        return oof_frame, -1.0

    available = [c for c in stacker_cols if c in oof_frame.columns]
    if not available:
        return oof_frame, -1.0

    result_frame = oof_frame.copy()
    result_frame["gbm_stacker_prob"] = np.nan

    for fold_id in sorted(oof_frame[fold_col].dropna().unique()):
        train_mask = oof_frame[fold_col] != fold_id
        valid_mask = oof_frame[fold_col] == fold_id
        train_df = oof_frame[train_mask]
        valid_df = oof_frame[valid_mask]

        y_train = train_df["status"].astype(int)
        positives = max(1, int(y_train.sum()))
        negatives = max(1, len(y_train) - positives)
        cw_ratio = min(float(negatives) / positives, 5.0)

        model = LGBMClassifier(
            n_estimators=300,
            num_leaves=15,
            learning_rate=0.05,
            min_child_samples=30,
            reg_lambda=10.0,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=cw_ratio,
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        )
        model.fit(train_df[available], y_train)
        result_frame.loc[valid_mask, "gbm_stacker_prob"] = model.predict_proba(valid_df[available])[:, 1]

    labeled = result_frame.dropna(subset=["status", "gbm_stacker_prob"])
    if labeled.empty:
        return oof_frame, -1.0

    y = labeled["status"].astype(int).values
    scores = labeled["gbm_stacker_prob"].values
    best_f1 = 0.0
    for thr in np.arange(0.05, 0.60, 0.01):
        f = float(f1_score(y, (scores >= thr).astype(int), zero_division=0))
        if f > best_f1:
            best_f1 = f

    return result_frame, best_f1


def _add_node2vec_features(
    dataset: pd.DataFrame,
    graph: "Any",
) -> pd.DataFrame:
    """Add Node2Vec embedding features to the dataset.

    Uses torch_geometric.nn.Node2Vec if available, otherwise skips silently.
    """
    try:
        import torch
        from torch_geometric.nn import Node2Vec as _N2V
    except ImportError:
        warnings.warn("[configurable_pipeline] torch_geometric not available, skipping node2vec.")
        return dataset

    user_ids = graph.user_ids
    user_index = graph.user_index
    edges = graph.collapsed_edges

    if edges.empty:
        return dataset

    src = edges["src_user_id"].astype(int).map(user_index).dropna().astype(int).values
    dst = edges["dst_user_id"].astype(int).map(user_index).dropna().astype(int).values
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

    n2v_model = _N2V(
        edge_index,
        embedding_dim=16,
        walk_length=10,
        context_size=5,
        walks_per_node=5,
        num_negative_samples=1,
        sparse=True,
    )
    loader = n2v_model.loader(batch_size=256, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(n2v_model.parameters(), lr=0.01)

    n2v_model.train()
    for _epoch in range(30):
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = n2v_model.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()

    n2v_model.eval()
    with torch.no_grad():
        embeddings = n2v_model.embedding.weight.detach().cpu().numpy()

    emb_df = pd.DataFrame(
        embeddings,
        columns=[f"n2v_{i}" for i in range(embeddings.shape[1])],
    )
    emb_df["user_id"] = user_ids

    dataset = dataset.merge(emb_df, on="user_id", how="left")
    for col in [f"n2v_{i}" for i in range(embeddings.shape[1])]:
        dataset[col] = dataset[col].fillna(0.0).astype("float32")
    return dataset


def _add_label_spreading_features(
    dataset: pd.DataFrame,
    graph: "Any",
    label_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Add label-spreading propagation features via sklearn LabelSpreading.

    Uses a precomputed affinity matrix from the graph adjacency.
    """
    try:
        from sklearn.semi_supervised import LabelSpreading
    except ImportError:
        warnings.warn("[configurable_pipeline] sklearn.semi_supervised unavailable, skipping.")
        return dataset

    user_ids = graph.user_ids
    n = len(user_ids)
    if n == 0 or graph.collapsed_edges.empty:
        return dataset

    # Memory guard: dense n×n affinity matrix costs n² × 4 bytes.
    # At n=63,770 that's ~16 GB — skip if too large.
    MAX_USERS_FOR_DENSE_LS = 15_000
    if n > MAX_USERS_FOR_DENSE_LS:
        # Fall back to sparse label propagation via power iteration.
        # F(t+1) = alpha * W * F(t) + (1 - alpha) * Y, iterated to convergence.
        from official.transductive_features import _normalized_adjacency
        import scipy.sparse as sp
        adj = _normalized_adjacency(graph)  # sparse (n × n)

        label_map: dict[int, int] = {}
        for _, row in label_frame.iterrows():
            label_map[int(row["user_id"])] = int(row["status"])

        y_vec = np.zeros(n, dtype=np.float32)
        labeled_mask = np.zeros(n, dtype=bool)
        for i, uid in enumerate(user_ids):
            if uid in label_map:
                y_vec[i] = float(label_map[uid])
                labeled_mask[i] = True

        if labeled_mask.sum() < 2 or len(set(y_vec[labeled_mask])) < 2:
            return dataset

        alpha = 0.2
        f = y_vec.copy()
        for _ in range(30):
            f_new = alpha * adj.dot(f) + (1.0 - alpha) * y_vec
            # Re-clamp labeled nodes to their true labels
            f_new[labeled_mask] = y_vec[labeled_mask]
            if np.max(np.abs(f_new - f)) < 1e-4:
                break
            f = f_new

        proba = np.clip(f, 0.0, 1.0)
        ls_df = pd.DataFrame({"user_id": user_ids, "label_spread_prob": proba.astype("float32")})
        dataset = dataset.merge(ls_df, on="user_id", how="left")
        dataset["label_spread_prob"] = dataset["label_spread_prob"].fillna(0.0).astype("float32")
        return dataset

    # Dense path for small graphs (n <= 15,000)
    from official.transductive_features import _normalized_adjacency
    adj = _normalized_adjacency(graph)

    # Build label vector: +1 = positive, 0 = negative, -1 = unlabeled
    label_map_dense: dict[int, int] = {}
    for _, row in label_frame.iterrows():
        uid = int(row["user_id"])
        status = int(row["status"])
        label_map_dense[uid] = status

    y = np.full(n, -1, dtype=int)
    for i, uid in enumerate(user_ids):
        if uid in label_map_dense:
            y[i] = label_map_dense[uid]

    # LabelSpreading requires at least 2 classes in labeled data
    if len(set(y[y >= 0])) < 2:
        return dataset

    try:
        # Convert sparse adjacency to dense affinity
        affinity = adj.toarray() + np.eye(n) * 0.01  # self-loops for stability
        affinity = (affinity + affinity.T) / 2.0  # symmetrize

        ls = LabelSpreading(kernel="precomputed", max_iter=30, alpha=0.2)
        ls.fit(affinity, y)
        proba = ls.label_distributions_[:, 1] if ls.label_distributions_.shape[1] > 1 else np.zeros(n)
    except Exception:
        return dataset

    ls_df = pd.DataFrame({
        "user_id": user_ids,
        "label_spread_prob": proba.astype("float32"),
    })
    dataset = dataset.merge(ls_df, on="user_id", how="left")
    dataset["label_spread_prob"] = dataset["label_spread_prob"].fillna(0.0).astype("float32")
    return dataset


def _add_feature_eng_node_attrs(dataset: pd.DataFrame) -> pd.DataFrame:
    """Add statistical n-gram and interaction features as additional node attributes."""
    df = dataset.copy()
    _age = df["account_age_days"].fillna(0).clip(0).astype("float32")
    _twd_cnt = df["twd_total_count"].fillna(0).astype("float32")
    _crypto_cnt = df["crypto_total_count"].fillna(0).astype("float32")
    _swap_cnt = df["swap_total_count"].fillna(0).astype("float32")
    _twd_sum = df["twd_total_sum"].fillna(0).astype("float32")
    _crypto_sum = df["crypto_total_sum"].fillna(0).astype("float32")

    # Transaction entropy: how dispersed are transactions across channels
    total_cnt = (_twd_cnt + _crypto_cnt + _swap_cnt).clip(1)
    p_twd = _twd_cnt / total_cnt
    p_crypto = _crypto_cnt / total_cnt
    p_swap = _swap_cnt / total_cnt
    eps = 1e-8
    df["txn_channel_entropy"] = -(
        p_twd * np.log(p_twd + eps) + p_crypto * np.log(p_crypto + eps) + p_swap * np.log(p_swap + eps)
    ).clip(0, 3).astype("float32")

    # Volume concentration: how much of total volume is in crypto vs TWD
    total_vol = (_twd_sum + _crypto_sum).clip(1)
    df["crypto_volume_share"] = (_crypto_sum / total_vol).clip(0, 1).astype("float32")

    # Account velocity: total transactions per day of account age
    df["account_txn_velocity"] = (total_cnt / _age.clip(1)).clip(0, 10).astype("float32")

    return df


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def run_configurable_pipeline(
    config: dict[str, bool],
    experiment_id: str = "unnamed",
    catboost_params: dict | None = None,
    graph_max_epochs: int | None = None,
) -> dict[str, float]:
    """Run the official OOF pipeline with configurable component toggles.

    Parameters
    ----------
    config : dict[str, bool]
        Component toggle flags. See ``COMPONENTS`` for valid keys.
    experiment_id : str
        Human-readable name for logging.
    catboost_params : dict, optional
        CatBoost hyperparameters (from HPO). If None, attempts to load saved HPO params.

    Returns
    -------
    dict with keys: f1, precision, recall, pr_auc, threshold, n_flagged,
                    n_positives_in_seed, elapsed_seconds, experiment_id
    """
    from official.graph_dataset import build_transductive_graph
    from official.stacking import (
        STACKER_FEATURE_COLUMNS,
        BlendEnsemble,
        _add_base_meta_features,
        build_stacker_oof,
        tune_blend_weights,
    )
    from official.train import (
        PRIMARY_GRAPH_MAX_EPOCHS,
        _label_frame,
        _label_free_feature_columns,
        _transductive_feature_columns,
        run_transductive_oof_pipeline,
    )
    from official.transductive_features import build_transductive_feature_frame
    from official.transductive_validation import (
        PrimarySplitSpec,
        build_primary_transductive_splits,
    )

    t0 = time.time()
    cfg = {**DEFAULT_CONFIG, **config}
    print(f"\n[configurable_pipeline] Experiment: {experiment_id}")
    enabled = [k for k, v in cfg.items() if v]
    print(f"[configurable_pipeline] Enabled components: {enabled}")

    # ------------------------------------------------------------------
    # Self-training: delegates to its own full pipeline
    # ------------------------------------------------------------------
    if cfg.get("self_training"):
        from official.self_training import run_fast_self_training
        print("[configurable_pipeline] Delegating to self_training pipeline...")
        # P0-5: entropy filter — max_entropy=0.5 nats (≈ p>=0.77) to reject borderline pseudo-labels
        _st_max_entropy = 0.5 if cfg.get("uncertainty_entropy_filter") else None
        st_result = run_fast_self_training(
            n_rounds=2, confidence_threshold=0.70, max_new_per_round=200,
            max_entropy=_st_max_entropy,
        )
        final = st_result.get("final_round", {})
        f1 = final.get("oof_f1_original_labels", 0.0)
        thr = final.get("best_threshold", 0.10)
        elapsed = time.time() - t0
        return {
            "f1": f1,
            "precision": 0.0,
            "recall": 0.0,
            "pr_auc": 0.0,
            "threshold": thr,
            "n_flagged": 0,
            "n_positives_in_seed": final.get("n_positives_in_seed", 0),
            "elapsed_seconds": round(elapsed, 1),
            "experiment_id": experiment_id,
        }

    # ------------------------------------------------------------------
    # 1. Load dataset + rules (cached across sequential experiments)
    # ------------------------------------------------------------------
    dataset = _cached_base_dataset()

    # Optionally add feature engineering node attributes
    if cfg.get("feature_eng_node_attrs"):
        dataset = _add_feature_eng_node_attrs(dataset)

    label_frame = _label_frame(dataset)
    original_user_ids = set(label_frame["user_id"].astype(int).tolist())
    n_positives_in_seed = int(label_frame["status"].astype(int).eq(1).sum())

    # ------------------------------------------------------------------
    # 2. Load CatBoost HPO params
    # ------------------------------------------------------------------
    if catboost_params is None:
        try:
            from official.hpo import load_hpo_best_params
            catboost_params = load_hpo_best_params()
        except Exception:
            catboost_params = None

    # PU learning loss: lower negative class weight in CatBoost
    if cfg.get("pu_learning_loss"):
        catboost_params = dict(catboost_params or {})
        # pu_negative_weight=0.7: reduce negative sample influence since some
        # "negatives" are actually unlabeled positives (PU learning setting).
        catboost_params["pu_negative_weight"] = float(cfg.get("pu_custom_weight", 0.7))

    # nnPU-corrected weights: non-negative PU risk estimator (stronger than simple weight reduction)
    if cfg.get("nnpu_weights"):
        try:
            from official.nnpu_loss import estimate_pu_prior
            pi = estimate_pu_prior(label_frame)
            # nnPU correction: weight negatives by (1 - pi) to account for unlabeled positives
            # This is more principled than the simple 0.7 heuristic
            pu_neg_weight = max(0.3, 1.0 - pi * 2.0)  # pi~0.35 → weight~0.30
            catboost_params = dict(catboost_params or {})
            catboost_params["pu_negative_weight"] = pu_neg_weight
            print(f"[configurable_pipeline] nnPU prior={pi:.3f} → neg_weight={pu_neg_weight:.3f}")
        except Exception as e:
            print(f"[configurable_pipeline] nnpu_weights failed: {e}")

    # ------------------------------------------------------------------
    # 3. Build graph with optional edge weight HPO
    # ------------------------------------------------------------------
    edge_weights: dict[str, float] | None = None
    if cfg.get("edge_weight_hpo"):
        try:
            from official.hpo_edge_weights import load_best_edge_weights
            edge_weights = load_best_edge_weights()
            if edge_weights:
                print(f"[configurable_pipeline] Using HPO edge weights: {edge_weights}")
        except Exception:
            pass

    # Hub IP pruning: pass to graph builder
    graph_kwargs: dict[str, Any] = {}
    if cfg.get("hub_ip_pruning"):
        graph_kwargs["hub_ip_prune_above"] = 15
    if cfg.get("edge_time_decay"):
        graph_kwargs["use_time_decay"] = True
        graph_kwargs["time_decay_half_life_days"] = float(cfg.get("time_decay_half_life_days", 90.0))
    if cfg.get("flow_graph_edges"):
        graph_kwargs["use_flow_edges"] = True

    graph = build_transductive_graph(dataset, edge_weights=edge_weights, **graph_kwargs)

    # Optionally add Node2Vec embeddings to dataset
    if cfg.get("node2vec_embeddings"):
        dataset = _add_node2vec_features(dataset, graph)

    # Optionally add label-spreading features
    if cfg.get("label_spreading"):
        dataset = _add_label_spreading_features(dataset, graph, label_frame)

    # ── Tier D: community features (Louvain cluster membership) ──────────────
    # Only label-FREE community features go into dataset/Base A to prevent OOF leakage.
    # Label-aware features (community_pos_count/ratio) use ALL labels if merged here,
    # so they're excluded from Base A. Only graph-structural features (size, degree) are safe.
    if cfg.get("community_features"):
        try:
            from official.community_features import build_community_features
            comm_df = build_community_features(graph, label_frame)
            if not comm_df.empty and len(comm_df.columns) > 1:
                # Only keep label-free structural columns (no pos_count/ratio/ppr_sum)
                label_free_comm_cols = [
                    c for c in comm_df.columns
                    if c != "user_id" and not any(
                        kw in c for kw in ("pos_count", "pos_ratio", "high_risk", "ppr_sum")
                    )
                ]
                if label_free_comm_cols:
                    comm_safe = comm_df[["user_id"] + label_free_comm_cols]
                    dataset = dataset.merge(comm_safe, on="user_id", how="left")
                    for col in label_free_comm_cols:
                        if col in dataset.columns:
                            dataset[col] = dataset[col].fillna(0.0)
                    print(f"[configurable_pipeline] Community features (label-free): {label_free_comm_cols}")
        except Exception as e:
            print(f"[configurable_pipeline] community_features failed: {e}")

    # ── Phase 2A-2: Raw event sequence features ───────────────────────────────
    if cfg.get("sequence_features"):
        seq_df = _cached_sequence_features(dataset)
        if seq_df is not None and not seq_df.empty and len(seq_df.columns) > 1:
            new_cols = [c for c in seq_df.columns if c != "user_id"]
            dataset = dataset.merge(seq_df, on="user_id", how="left")
            for col in new_cols:
                if col in dataset.columns:
                    dataset[col] = dataset[col].fillna(0.0)
            print(f"[configurable_pipeline] Sequence features: {len(new_cols)} cols added")

    # ── Phase 2A: Temporal sequence features ──────────────────────────────────
    if cfg.get("temporal_sequence_features"):
        # Use cache when skip_layering=True (default); bypass cache when temporal_layering=True
        _skip_layering = not cfg.get("temporal_layering", False)
        if _skip_layering:
            temp_df = _cached_temporal_features(dataset)
        else:
            try:
                from official.temporal_features import build_temporal_features
                temp_df = build_temporal_features(dataset, skip_layering=False)
            except Exception as e:
                print(f"[configurable_pipeline] temporal_sequence_features failed: {e}")
                temp_df = None
        if temp_df is not None and not temp_df.empty and len(temp_df.columns) > 1:
            new_cols = [c for c in temp_df.columns if c != "user_id"]
            dataset = dataset.merge(temp_df, on="user_id", how="left")
            for col in new_cols:
                if col in dataset.columns:
                    dataset[col] = dataset[col].fillna(0.0)
            print(f"[configurable_pipeline] Temporal features: {len(new_cols)} cols added")

    # ── Tier D: lag/cross-channel correlation features ────────────────────────
    if cfg.get("lag_features"):
        lag_df = _cached_lag_features(dataset)
        if lag_df is not None and not lag_df.empty and len(lag_df.columns) > 1:
            dataset = dataset.merge(lag_df, on="user_id", how="left")
            new_cols = [c for c in lag_df.columns if c != "user_id"]
            for col in new_cols:
                if col in dataset.columns:
                    dataset[col] = dataset[col].fillna(0.0)
            print(f"[configurable_pipeline] Lag features: {len(new_cols)} cols added")

    # ── Tier D: GRU sequence branch ───────────────────────────────────────────
    if cfg.get("gru_sequence_branch"):
        try:
            from official.sequence_model import build_sequence_features
            seq_df = build_sequence_features(dataset, max_seq_len=30, embed_dim=8, n_epochs=10)
            if not seq_df.empty and len(seq_df.columns) > 1:
                dataset = dataset.merge(seq_df, on="user_id", how="left")
                new_cols = [c for c in seq_df.columns if c != "user_id"]
                for col in new_cols:
                    if col in dataset.columns:
                        dataset[col] = dataset[col].fillna(0.0)
                print(f"[configurable_pipeline] GRU sequence features: {len(new_cols)} cols added")
        except Exception as e:
            print(f"[configurable_pipeline] gru_sequence_branch failed: {e}")

    # ── Tier D: DGI self-supervised embeddings + RF score ─────────────────────
    if cfg.get("dgi_rf_features"):
        try:
            from official.dgi_embeddings import build_dgi_features
            dgi_df = build_dgi_features(graph, label_frame, embed_dim=16, n_epochs=50)
            if not dgi_df.empty and len(dgi_df.columns) > 1:
                dataset = dataset.merge(dgi_df, on="user_id", how="left")
                new_cols = [c for c in dgi_df.columns if c != "user_id"]
                for col in new_cols:
                    if col in dataset.columns:
                        dataset[col] = dataset[col].fillna(0.0)
                print(f"[configurable_pipeline] DGI features: {len(new_cols)} cols added")
        except Exception as e:
            print(f"[configurable_pipeline] dgi_rf_features failed: {e}")

    # ------------------------------------------------------------------
    # 4. Build splits
    # ------------------------------------------------------------------
    primary_split = build_primary_transductive_splits(
        dataset,
        cutoff_tag="full",
        spec=PrimarySplitSpec(),
        write_outputs=False,
    )

    # ------------------------------------------------------------------
    # 5. Feature columns
    # ------------------------------------------------------------------
    base_a_feature_columns = _label_free_feature_columns(dataset)
    sample_trans = build_transductive_feature_frame(
        graph,
        label_frame,
        use_negative_propagation=cfg.get("negative_propagation", False),
    )
    _trans_only_cols = _transductive_feature_columns(sample_trans)
    if cfg.get("base_b_transductive_only"):
        # v50: Train Base B on transductive-only features (not base_a + transductive).
        # Forces Base B to specialize in graph-propagation signal rather than recapitulating
        # Base A. The stacker combines both: Base A handles tabular, Base B handles graph.
        # Expected: AP improvement from ~0.084 to 0.12+ (less noise from tabular dilution).
        base_b_feature_columns = _trans_only_cols
        print(f"[configurable_pipeline] Base B: transductive-only ({len(_trans_only_cols)} features)")
    else:
        base_b_feature_columns = base_a_feature_columns + _trans_only_cols

    # ------------------------------------------------------------------
    # 6. Extract PU negative weight for fit_catboost if set
    # ------------------------------------------------------------------
    cb_params_for_pipeline = dict(catboost_params or {})
    pu_neg_weight = cb_params_for_pipeline.pop("pu_negative_weight", None)
    if pu_neg_weight is not None:
        # Pass to fit_catboost via the catboost_params dict under a custom key.
        # modeling.py now reads this key.
        cb_params_for_pipeline["pu_negative_weight"] = pu_neg_weight

    # ------------------------------------------------------------------
    # 7. Run OOF pipeline
    # ------------------------------------------------------------------
    _gnn_epochs = graph_max_epochs if graph_max_epochs is not None else PRIMARY_GRAPH_MAX_EPOCHS
    # When graphsage_3layer is disabled, skip GNN training entirely by setting epochs=0.
    # graph_model.py handles max_epochs=0 correctly: the training loop is skipped,
    # producing near-constant base_c_probability (~prior rate), which the blend/stacker
    # will zero-weight automatically.
    if not cfg.get("graphsage_3layer"):
        _gnn_epochs = 0
    # v51: Tunable C&S restore threshold. Default 0.05 (top-5% isolated) was tested at tier D.
    # Higher pct (0.10-0.25) captures more isolated positives at the cost of more FP restoration.
    # Controlled by "cs_restore_pct" in cfg or falls back to 0.05 when conditional_cs_restore=True.
    if cfg.get("conditional_cs_restore"):
        _cs_restore_pct = float(cfg.get("cs_restore_pct", 0.05))
    else:
        _cs_restore_pct = 0.0
    oof_frame, fold_meta = run_transductive_oof_pipeline(
        dataset,
        graph,
        primary_split,
        fold_column="primary_fold",
        base_a_feature_columns=base_a_feature_columns,
        base_b_feature_columns=base_b_feature_columns,
        graph_max_epochs=_gnn_epochs,
        catboost_params=cb_params_for_pipeline,
        use_negative_propagation=cfg.get("negative_propagation", False),
        cs_restore_top_pct=_cs_restore_pct,
    )

    # ------------------------------------------------------------------
    # 8. Stacking
    # ------------------------------------------------------------------
    # v53: lr_stacker flag forces use_blend=False — tests v13 hypothesis that
    # LR stacker outperforms BlendEnsemble when GNN/C&S output is present.
    # BlendEnsemble drops signals with AP<0.08; LR can weight weak signals positively.
    _use_blend = not cfg.get("lr_stacker", False)
    oof_frame, stacker_model = build_stacker_oof(
        oof_frame, primary_split, fold_column="primary_fold", use_blend=_use_blend,
    )

    # GBM stacker comparison
    if cfg.get("gbm_stacker"):
        stacker_cols = [c for c in STACKER_FEATURE_COLUMNS if c in oof_frame.columns]
        gbm_frame, gbm_f1 = _fit_lgbm_stacker(oof_frame, stacker_cols)
        # Compare with blend F1
        blend_f1, _, _, _, _, _ = _quick_oof_f1(oof_frame, original_user_ids)
        print(f"[configurable_pipeline] BlendEnsemble F1={blend_f1:.4f} vs GBM stacker F1={gbm_f1:.4f}")
        if gbm_f1 > blend_f1 + 0.002:
            print("[configurable_pipeline] GBM stacker wins — using GBM probabilities.")
            oof_frame = gbm_frame
            oof_frame["stacker_raw_probability"] = oof_frame["gbm_stacker_prob"]
        else:
            print("[configurable_pipeline] BlendEnsemble wins — keeping blend probabilities.")

    # ── Tier D: hard negative mining — identify and up-weight borderline negatives ──
    if cfg.get("hard_negative_mining"):
        try:
            # Hard negatives = labeled negatives with high model score (likely unlabeled positives)
            # Strategy: re-weight these samples as uncertain (weight=0.5) rather than confident negatives
            if "stacker_raw_probability" in oof_frame.columns:
                neg_mask = oof_frame["status"].astype(int) == 0
                high_score_neg = oof_frame.loc[neg_mask, "stacker_raw_probability"] > 0.3
                hard_neg_frac = high_score_neg.sum() / max(1, neg_mask.sum())
                print(f"[configurable_pipeline] Hard negatives: {high_score_neg.sum()} ({hard_neg_frac:.1%} of negatives)")
                # Mark hard negatives in oof_frame for potential downstream use
                oof_frame["is_hard_negative"] = (neg_mask & (oof_frame["stacker_raw_probability"] > 0.3)).astype(int)
        except Exception as e:
            print(f"[configurable_pipeline] hard_negative_mining failed: {e}")

    # ── Tier D: seed perturbation averaging — ensemble over ±5% perturbed seeds ──
    if cfg.get("seed_perturbation_avg"):
        try:
            from official.transductive_features import build_transductive_feature_frame
            from official.stacking import tune_blend_weights, BlendEnsemble, _add_base_meta_features
            import random
            # Run 3 perturbations: drop 5% of positive seeds, add results to oof_frame
            rng = random.Random(42)
            pos_ids = list(label_frame[label_frame["status"].astype(int) == 1]["user_id"].astype(int).tolist())
            perturb_probs_list: list[np.ndarray] = []
            for _pi in range(3):
                keep_n = max(int(len(pos_ids) * 0.95), len(pos_ids) - 10)
                kept_ids = rng.sample(pos_ids, keep_n)
                perturbed_lf = label_frame[label_frame["user_id"].astype(int).isin(kept_ids + list(
                    label_frame[label_frame["status"].astype(int) == 0]["user_id"].astype(int).tolist()
                ))].copy()
                ptrans = build_transductive_feature_frame(graph, perturbed_lf)
                # Merge with oof_frame to get per-user PPR in this perturbation
                ppr_col = "ppr_score" if "ppr_score" in ptrans.columns else None
                if ppr_col:
                    _merged = oof_frame[["user_id"]].merge(ptrans[["user_id", ppr_col]], on="user_id", how="left")
                    perturb_probs_list.append(_merged[ppr_col].fillna(0.0).values)
            if perturb_probs_list and "stacker_raw_probability" in oof_frame.columns:
                avg_perturb = np.mean(perturb_probs_list, axis=0)
                # Blend 80% original + 20% perturbation average for stability
                oof_frame["stacker_raw_probability"] = (
                    0.80 * oof_frame["stacker_raw_probability"].values +
                    0.20 * avg_perturb
                )
                print(f"[configurable_pipeline] Seed perturbation: blended {len(perturb_probs_list)} perturbations")
        except Exception as e:
            print(f"[configurable_pipeline] seed_perturbation_avg failed: {e}")

    # ------------------------------------------------------------------
    # 9. Threshold HPO (optional)
    # ------------------------------------------------------------------
    if cfg.get("threshold_hpo"):
        import os as _os
        _hpo_skip = _os.getenv("BITOGUARD_HPO_SKIP", "0") == "1"
        if _hpo_skip:
            # Async mode: save OOF to .npz for external processing, skip inline HPO.
            try:
                from pathlib import Path as _Path
                eval_frame = oof_frame[oof_frame["user_id"].astype(int).isin(original_user_ids)].copy()
                if "stacker_raw_probability" in eval_frame.columns and not eval_frame.empty:
                    raw_probs = eval_frame["stacker_raw_probability"].values
                    labels_arr = eval_frame["status"].astype(int).values
                    _oof_save = _Path("artifacts/reports") / f"oof_for_hpo_{experiment_id}.npz"
                    _oof_save.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(str(_oof_save), raw_probs=raw_probs, labels=labels_arr)
                    print(f"[configurable_pipeline] Threshold HPO: OOF saved async → {_oof_save}")
            except Exception as exc:
                print(f"[configurable_pipeline] Threshold HPO async save failed: {exc}")
        else:
            try:
                from official.hpo_threshold import run_threshold_hpo

                eval_frame = oof_frame[oof_frame["user_id"].astype(int).isin(original_user_ids)].copy()
                if "stacker_raw_probability" in eval_frame.columns and not eval_frame.empty:
                    raw_probs = eval_frame["stacker_raw_probability"].values
                    labels_arr = eval_frame["status"].astype(int).values
                    thr_result = run_threshold_hpo(
                        raw_probs, labels_arr, group_ids=None, n_trials=100, timeout=300.0,
                    )
                    hpo_thr = thr_result.get("threshold", 0.10)
                    hpo_f1 = thr_result.get("bootstrap_f1", 0.0)
                    print(f"[configurable_pipeline] Threshold HPO: F1={hpo_f1:.4f} @ thr={hpo_thr:.4f}")
            except Exception as exc:
                print(f"[configurable_pipeline] Threshold HPO failed: {exc}")

    # ------------------------------------------------------------------
    # 10. Evaluate and return metrics
    # ------------------------------------------------------------------
    f1, prec, rec, pr_auc, threshold, n_flagged = _quick_oof_f1(oof_frame, original_user_ids)
    elapsed = time.time() - t0

    # P0-3: Compute cohort breakdown (dormant vs active).
    try:
        from official.experiment_tracker import compute_cohort_metrics
        cohort_metrics = compute_cohort_metrics(
            oof_frame[oof_frame["user_id"].astype(int).isin(original_user_ids)].copy(),
            threshold=threshold,
        )
    except Exception as _exc:
        cohort_metrics = {}

    metrics: dict[str, float] = {
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "pr_auc": round(pr_auc, 4),
        "threshold": round(threshold, 4),
        "n_flagged": n_flagged,
        "n_positives_in_seed": n_positives_in_seed,
        "elapsed_seconds": round(elapsed, 1),
        "experiment_id": experiment_id,
    }
    if cohort_metrics:
        metrics["cohort_metrics"] = cohort_metrics  # type: ignore[assignment]
        d_f1 = cohort_metrics.get("dormant", {}).get("f1", float("nan"))
        a_f1 = cohort_metrics.get("active", {}).get("f1", float("nan"))
        print(f"[configurable_pipeline] Result: F1={f1:.4f} P={prec:.4f} R={rec:.4f} AP={pr_auc:.4f} "
              f"thr={threshold:.4f} flagged={n_flagged} elapsed={elapsed:.0f}s "
              f"[dormant={d_f1:.4f}, active={a_f1:.4f}]")
    else:
        print(f"[configurable_pipeline] Result: F1={f1:.4f} P={prec:.4f} R={rec:.4f} AP={pr_auc:.4f} "
              f"thr={threshold:.4f} flagged={n_flagged} elapsed={elapsed:.0f}s")

    # Save OOF probabilities for seed ensemble analysis.
    try:
        from pathlib import Path as _Path
        _eval = oof_frame[oof_frame["user_id"].astype(int).isin(original_user_ids)].copy()
        if "stacker_raw_probability" in _eval.columns and not _eval.empty:
            _oof_path = _Path("artifacts/reports") / f"oof_{experiment_id}.npz"
            _oof_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(str(_oof_path),
                     user_ids=_eval["user_id"].values,
                     raw_probs=_eval["stacker_raw_probability"].values,
                     labels=_eval["status"].astype(int).values)
    except Exception:
        pass

    return metrics
