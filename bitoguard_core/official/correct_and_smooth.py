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
        Diffusion rate for the **correct** step.  At each iteration the
        residual vector is updated as
        ``r^{t+1} = alpha * r^{0} + (1 - alpha) * A @ r^{t}``.
        Higher alpha retains more of the original residual signal; lower
        alpha lets the graph smooth it further.
    alpha_smooth:
        Diffusion rate for the **smooth** step.  Same formula applied to
        the corrected probability vector.
    n_correct_iter:
        Number of power-iteration steps in the correct phase.
    n_smooth_iter:
        Number of power-iteration steps in the smooth phase.

    Returns
    -------
    dict[int, float]
        Mapping of ``user_id -> corrected_smooth_probability`` for every
        user present in *base_probs*.  Users that do not appear in the
        graph are returned with their original base probability unchanged.
    """

    n_users = len(graph.user_ids)

    # -- Early exit: nothing to propagate --------------------------------
    if n_users == 0 or graph.collapsed_edges.empty:
        return dict(base_probs)

    # -- Build the row-normalised adjacency matrix -----------------------
    adj = _normalized_adjacency(graph)  # sparse CSR, shape (n_users, n_users)

    # -- Construct dense vectors aligned with graph.user_ids ordering ----
    base_vec = np.zeros(n_users, dtype=np.float64)
    for user_id, prob in base_probs.items():
        idx = graph.user_index.get(user_id)
        if idx is not None:
            base_vec[idx] = prob

    # Residual vector: (y_true - base_prob) for training users, 0 elsewhere.
    # This is the key to avoiding label leakage: only training-fold labels
    # contribute to the correction signal.
    residual_vec = np.zeros(n_users, dtype=np.float64)
    for user_id, label in train_labels.items():
        idx = graph.user_index.get(user_id)
        if idx is not None:
            residual_vec[idx] = float(label) - base_vec[idx]

    # ====================================================================
    # Step 1: CORRECT -- propagate residuals
    # r^{t+1} = alpha * r^{0} + (1 - alpha) * A @ r^{t}
    # ====================================================================
    r0 = residual_vec.copy()
    r = residual_vec.copy()
    for _ in range(n_correct_iter):
        r = alpha_correct * r0 + (1.0 - alpha_correct) * (adj @ r)

    # Apply correction: base_prob + propagated_residual
    corrected = base_vec + r
    # Clip to valid probability range for numerical stability
    np.clip(corrected, 0.0, 1.0, out=corrected)

    # ====================================================================
    # Step 2: SMOOTH -- propagate corrected predictions
    # f^{t+1} = alpha * f^{0} + (1 - alpha) * A @ f^{t}
    # ====================================================================
    f0 = corrected.copy()
    f = corrected.copy()
    for _ in range(n_smooth_iter):
        f = alpha_smooth * f0 + (1.0 - alpha_smooth) * (adj @ f)

    # Final clip for numerical stability
    np.clip(f, 0.0, 1.0, out=f)

    # -- Map back to user_id -> probability ------------------------------
    result: dict[int, float] = {}
    for user_id, prob in base_probs.items():
        idx = graph.user_index.get(user_id)
        if idx is not None:
            result[user_id] = float(f[idx])
        else:
            # User is not in the graph; return base probability unchanged
            result[user_id] = prob

    return result
