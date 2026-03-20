from __future__ import annotations

"""Non-negative Positive-Unlabeled (nnPU) learning utilities.

In our setting, the "negative" class in the training data is actually *unlabeled*:
it contains a mixture of true negatives and undetected positives.  Standard binary
cross-entropy penalises the model whenever it scores an unlabeled-but-true-positive
user highly, biasing the model towards low recall.

The nnPU estimator (Kiryo et al., 2017) corrects for this by:
1. Estimating the positive prior π = P(y=1) from the fraction of labeled positives.
2. Computing the unbiased PU risk: R_pu = π·R_p − (π·R̃_n − R_n^−)
   where R_p is empirical positive risk, R̃_n is the negative-labelled risk evaluated
   using the positive loss, and R_n^− is the negative-labelled risk.
3. Clamping any negative risk contribution to 0 (non-negative constraint) to prevent
   the gradient from going in the wrong direction on samples that look positive.

This module provides per-sample weight arrays compatible with CatBoost/sklearn's
`sample_weight` parameter.  The implementation re-weights labeled negatives so that
their effective contribution to the loss matches the nnPU correction.

References
----------
Kiryo et al. (2017). "Positive-Unlabeled Learning with Non-Negative Risk Estimator."
NeurIPS 2017. https://arxiv.org/abs/1703.00593

Elkan & Noto (2008). "Learning classifiers from only positive and unlabeled data."
KDD 2008. https://dl.acm.org/doi/10.1145/1401890.1401920
"""

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Sentinel for minimum prior to avoid division-by-zero.
_MIN_PRIOR = 1e-6
# Default beta for non-negative clamping.  beta=0.0 means clamp at zero (standard
# nnPU); positive beta allows a small negative risk contribution before clamping
# (useful to smooth gradients when the dataset is noisy).
_DEFAULT_BETA = 0.0


def estimate_pu_prior(label_frame: pd.DataFrame) -> float:
    """Estimate the positive class prior π from labeled training data.

    The prior is simply the fraction of labeled-positive users among ALL users
    in label_frame (both positives and labeled negatives).  Unlabeled users
    (those not in label_frame at all) are assumed to contain some unknwon
    fraction of true positives, which is what the nnPU correction accounts for.

    Parameters
    ----------
    label_frame:
        DataFrame with columns [user_id, status].  status=1 → labeled positive,
        status=0 → labeled negative.  Rows with NaN status are dropped.

    Returns
    -------
    float
        Estimated prior π in (0, 1).  Clamped to [_MIN_PRIOR, 1-_MIN_PRIOR].
    """
    lf = label_frame.copy()
    lf["status"] = pd.to_numeric(lf["status"], errors="coerce")
    lf = lf.dropna(subset=["status"])
    if lf.empty:
        logger.warning("estimate_pu_prior: empty label_frame, returning default prior=0.5")
        return 0.5
    n_total = len(lf)
    n_positive = int((lf["status"].astype(int) == 1).sum())
    if n_total == 0:
        return 0.5
    prior = float(n_positive) / float(n_total)
    prior = float(np.clip(prior, _MIN_PRIOR, 1.0 - _MIN_PRIOR))
    logger.debug("estimate_pu_prior: n_positive=%d / n_total=%d → π=%.4f", n_positive, n_total, prior)
    return prior


def nnpu_sample_weights(
    y: np.ndarray,
    pi: float | None = None,
    beta: float = _DEFAULT_BETA,
) -> np.ndarray:
    """Compute per-sample nnPU risk correction weights.

    The nnPU risk can be written as a weighted sum over samples:
        R_pu = (1/n_p) Σ_{i∈P} w_p · l(f(x_i)) +
               (1/n_u) Σ_{i∈U} w_u · l(-f(x_i))
    where l is the loss function.  For the sample-weight approximation used
    here we absorb π into the negative-class weight:

        w_positive = 1.0  (always)
        w_negative = max(β, 1 − π)   # non-negative clamp

    This is a simplification of the full nnPU gradient correction that can be
    applied directly as a `sample_weight` argument to sklearn/CatBoost.

    Parameters
    ----------
    y:
        1-D array of binary labels.  1=positive (labeled), 0=negative/unlabeled.
    pi:
        Positive prior π.  If None, estimated from `y` directly.
    beta:
        Non-negative clamp threshold.  Negative risk contributions smaller than
        beta are clamped to beta.  Typically 0.0.

    Returns
    -------
    np.ndarray of float32, shape (n_samples,)
        Per-sample weights.  Positives get weight=1.0.  Negatives get
        weight=max(beta, 1−π).
    """
    y_arr = np.asarray(y, dtype=int)
    if pi is None:
        n_p = int(y_arr.sum())
        n_total = len(y_arr)
        pi = float(np.clip(float(n_p) / max(n_total, 1), _MIN_PRIOR, 1.0 - _MIN_PRIOR))

    # nnPU weight for negative-class samples: correction factor (1 - π).
    # The non-negative clamp ensures this is never below beta.
    neg_weight = float(np.clip(1.0 - pi, beta, 1.0))

    weights = np.where(y_arr == 1, 1.0, neg_weight).astype(np.float32)
    return weights


def pu_adjusted_catboost_weights(
    label_frame: pd.DataFrame,
    all_user_ids: pd.Series,
    pi: float | None = None,
) -> np.ndarray:
    """Generate per-sample CatBoost training weights for PU learning.

    Assigns weights based on each user's label status:
        Labeled positives  → weight = 1.0
        Labeled negatives  → weight = max(0, 1 − π)   (nnPU correction)
        Unlabeled users    → weight = 0.0              (excluded from training)

    This function is intended to be called once per training fold.  Pass only
    the *training-fold* labeled users in `label_frame` to avoid leakage.

    Parameters
    ----------
    label_frame:
        DataFrame with columns [user_id, status].  Only labeled users.
    all_user_ids:
        pd.Series of all user IDs in the order expected by the CatBoost training
        DataFrame (i.e. matching the row order of X_train).
    pi:
        Positive prior π.  If None, estimated from label_frame.

    Returns
    -------
    np.ndarray of float32, shape (len(all_user_ids),)
        Per-sample weight for each user.
    """
    lf = label_frame.copy()
    lf["user_id"] = pd.to_numeric(lf["user_id"], errors="coerce").astype("Int64")
    lf["status"] = pd.to_numeric(lf["status"], errors="coerce").fillna(-1).astype(int)
    lf = lf.dropna(subset=["user_id"])
    lf["user_id"] = lf["user_id"].astype(int)

    if pi is None:
        valid_labels = lf[lf["status"].isin([0, 1])]
        pi = estimate_pu_prior(valid_labels)

    # Labeled negatives receive the nnPU-corrected weight.
    neg_weight = float(np.clip(1.0 - pi, 0.0, 1.0))

    # Build lookup: user_id → weight.
    weight_lookup: dict[int, float] = {}
    for _, row in lf.iterrows():
        uid = int(row["user_id"])
        status = int(row["status"])
        if status == 1:
            weight_lookup[uid] = 1.0
        elif status == 0:
            weight_lookup[uid] = neg_weight
        # Other status values (e.g. -1) are treated as unlabeled → weight=0.

    user_ids_arr = pd.to_numeric(all_user_ids, errors="coerce").fillna(-1).astype(int).to_numpy()
    weights = np.array(
        [weight_lookup.get(int(uid), 0.0) for uid in user_ids_arr],
        dtype=np.float32,
    )

    n_pos = int((weights == 1.0).sum())
    n_neg = int(((weights > 0.0) & (weights < 1.0)).sum())
    n_unl = int((weights == 0.0).sum())
    logger.debug(
        "pu_adjusted_catboost_weights: π=%.4f, positives=%d, labeled_neg=%d (w=%.4f), unlabeled=%d (w=0)",
        pi, n_pos, n_neg, neg_weight, n_unl,
    )
    return weights


def calibrate_pu_scores(
    raw_probs: np.ndarray,
    pi: float,
) -> np.ndarray:
    """Calibrate PU model output probabilities to approximate P(y=1|x).

    Uses the Elkan & Noto (2008) identity:
        P(y=1|x) ≈ f(x) / c
    where f(x) is the model's output probability (probability of being in the
    positive *selected* set S) and c = P(s=1|y=1) is the labelling probability
    — i.e. the probability that a true positive is actually labeled.

    Under the SCAR (Selected Completely At Random) assumption,
        c = P(s=1) = π_label / π_true
    But since we estimated π from the labeled fraction directly, we approximate
        c ≈ mean(f(x_i)) for labeled positives / π
    In the absence of a labeled set at inference time, a simpler approximation
    is used: we estimate c from the raw score distribution under the assumption
    that the top-π fraction of users are the true positives:
        c ≈ mean(f(x)) over the top-π users / 1.0 ≈ quantile correction

    Practical approximation (safe fallback when labeled positives aren't available):
        P(y=1|x) ≈ f(x) / (mean(f(x)) / π)  = f(x) * π / mean(f(x))
    This re-scales the scores so that the marginal P(y=1) ≈ π.

    Parameters
    ----------
    raw_probs:
        1-D array of model output probabilities in [0, 1].
    pi:
        Positive prior π (estimated fraction of true positives in the full dataset).

    Returns
    -------
    np.ndarray of float64, shape (n_samples,)
        Calibrated probabilities clipped to [0, 1].
    """
    probs = np.asarray(raw_probs, dtype=float)
    if probs.size == 0:
        return probs

    mean_score = float(probs.mean())
    if mean_score < 1e-9:
        warnings.warn(
            "calibrate_pu_scores: mean raw score is near-zero (%.2e). "
            "Returning raw probabilities without calibration." % mean_score,
            stacklevel=2,
        )
        return probs.clip(0.0, 1.0)

    # Elkan & Noto calibration: scale so that mean(P(y=1|x)) ≈ π.
    # c = mean(f(x)) / π is the estimated P(s=1) under SCAR.
    c = mean_score / float(np.clip(pi, _MIN_PRIOR, 1.0 - _MIN_PRIOR))
    calibrated = probs / float(np.clip(c, 1e-6, 1.0))
    return calibrated.clip(0.0, 1.0)
