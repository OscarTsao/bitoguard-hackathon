"""Correct-and-Smooth (C&S) graph post-processing for base model probabilities.

Implements the two-step algorithm from Huang et al. (2021), "Combining Label
Propagation and Simple Models Outperforms Graph Neural Networks", ICLR 2021.

Step 1 -- Correct: propagate training-label residuals over the graph to fix
systematic errors in the base model's predictions.

Step 2 -- Smooth: propagate the corrected predictions over the graph so that
connected users receive similar scores.

This module depends only on numpy and scipy.sparse (both already in the venv)
and reuses the existing ``_normalized_adjacency`` builder from
``official.transductive_features``.
"""

from __future__ import annotations

import numpy as np

from official.graph_dataset import TransductiveGraph
from official.transductive_features import _normalized_adjacency


def correct_and_smooth(
    graph: TransductiveGraph,
    train_labels: dict[int, float],
    base_probs: dict[int, float],
    alpha_correct: float = 0.5,
    alpha_smooth: float = 0.5,
    n_correct_iter: int = 50,
    n_smooth_iter: int = 50,
    restore_isolated: bool = False,
    restore_isolated_top_pct: float = 0.0,
) -> dict[int, float]:
    """Apply Correct-and-Smooth graph post-processing to base model probabilities.

    Parameters
    ----------
    graph:
        ``TransductiveGraph`` containing user IDs, their matrix indices
        (``user_index``), and the collapsed weighted edge list used to
        build the row-normalised adjacency matrix.
    train_labels:
        Mapping of ``user_id -> {0, 1}`` for the *training* split only.
        These are the only labels used to compute residuals -- test/predict
        labels are never touched, so there is no label leakage.
    base_probs:
        Mapping of ``user_id -> probability`` in [0, 1] produced by the
        base model (e.g. CatBoost Base A out-of-fold predictions).
    alpha_correct:
        Diffusion rate for the **correct** step.
    alpha_smooth:
        Diffusion rate for the **smooth** step.
    n_correct_iter:
        Number of power-iteration steps in the correct phase.
    n_smooth_iter:
        Number of power-iteration steps in the smooth phase.
    restore_isolated:
        If True, isolated nodes (degree=0) are restored to their corrected
        (pre-smooth) probability after the smooth step.
    restore_isolated_top_pct:
        If > 0, only restore the top-P% of isolated users by base probability.

    Returns
    -------
    dict[int, float]
        Mapping of ``user_id -> corrected_smooth_probability``.
    """

    n_users = len(graph.user_ids)

    if n_users == 0 or graph.collapsed_edges.empty:
        return dict(base_probs)

    adj = _normalized_adjacency(graph)  # sparse CSR, shape (n_users, n_users)

    degree = np.asarray(adj.sum(axis=1)).ravel()
    isolated_mask = (degree == 0)

    base_vec = np.zeros(n_users, dtype=np.float64)
    for user_id, prob in base_probs.items():
        idx = graph.user_index.get(user_id)
        if idx is not None:
            base_vec[idx] = prob

    residual_vec = np.zeros(n_users, dtype=np.float64)
    for user_id, label in train_labels.items():
        idx = graph.user_index.get(user_id)
        if idx is not None:
            residual_vec[idx] = float(label) - base_vec[idx]

    # Step 1: CORRECT -- propagate residuals
    r0 = residual_vec.copy()
    r = residual_vec.copy()
    for _ in range(n_correct_iter):
        r = alpha_correct * r0 + (1.0 - alpha_correct) * (adj @ r)

    corrected = base_vec + r
    np.clip(corrected, 0.0, 1.0, out=corrected)

    # Step 2: SMOOTH -- propagate corrected predictions
    f0 = corrected.copy()
    f = corrected.copy()
    for _ in range(n_smooth_iter):
        f = alpha_smooth * f0 + (1.0 - alpha_smooth) * (adj @ f)

    np.clip(f, 0.0, 1.0, out=f)

    if restore_isolated and isolated_mask.any():
        f[isolated_mask] = corrected[isolated_mask]

    if restore_isolated_top_pct > 0.0 and isolated_mask.any():
        iso_probs = base_vec[isolated_mask]
        pct_threshold = np.percentile(iso_probs, 100.0 * (1.0 - restore_isolated_top_pct))
        restore_mask = isolated_mask & (base_vec >= pct_threshold)
        f[restore_mask] = corrected[restore_mask]

    result: dict[int, float] = {}
    for user_id, prob in base_probs.items():
        idx = graph.user_index.get(user_id)
        if idx is not None:
            result[user_id] = float(f[idx])
        else:
            result[user_id] = prob

    return result
