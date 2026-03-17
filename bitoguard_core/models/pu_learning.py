"""PU Learning calibration using Elkan-Noto 2008.

The SCAR assumption: labeling is Selected Completely At Random among true positives.
Given a trained scorer g(x), the true posterior is: p(y=1|x) = g(x) / c
where c = E[g(x) | labeled=1] estimated on a held-out positive set.

Reference: Elkan & Noto, KDD 2008, "Learning Classifiers from Only Positive
and Unlabeled Data". https://cseweb.ucsd.edu/~elkan/posonly.pdf
"""
from __future__ import annotations

import numpy as np


def estimate_c(
    probabilities: np.ndarray,
    labels: np.ndarray,
    hold_out_ratio: float = 0.10,
    random_state: int = 42,
) -> float:
    """Estimate c = P(label=1 | y=1) using held-out positive mean.

    Args:
        probabilities: Predicted probabilities from a trained classifier (shape: N,)
        labels: Binary labels (1=labeled positive, 0=unlabeled) (shape: N,)
        hold_out_ratio: Fraction of labeled positives reserved for c estimation
        random_state: For reproducible positive holdout split

    Returns:
        c in [0.3, 0.9] (clipped for numerical stability)
    """
    rng = np.random.default_rng(random_state)
    pos_indices = np.where(labels == 1)[0]
    if len(pos_indices) == 0:
        return 0.5
    n_holdout = max(1, int(len(pos_indices) * hold_out_ratio))
    holdout_idx = rng.choice(pos_indices, size=n_holdout, replace=False)
    c_raw = float(probabilities[holdout_idx].mean())
    return float(np.clip(c_raw, 0.30, 0.90))


def pu_adjust(
    probabilities: np.ndarray,
    c: float,
) -> np.ndarray:
    """Apply PU posterior correction: p_true(y=1|x) = g(x) / c, clipped to [0, 1].

    Args:
        probabilities: Raw predicted probabilities (shape: N,)
        c: Estimated label frequency from estimate_c()

    Returns:
        PU-adjusted probabilities, clipped to [0, 1]
    """
    return np.clip(probabilities / max(c, 1e-6), 0.0, 1.0)
