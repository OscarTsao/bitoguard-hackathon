from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score

from official.calibration import BetaCalibrator, IsotonicCalibrator, SigmoidCalibrator
from official.common import RANDOM_SEED, load_official_paths, save_pickle
from official.thresholding import search_threshold
from models.pu_learning import estimate_c, pu_adjust


class IdentityCalibrator:
    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "IdentityCalibrator":
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return np.asarray(probabilities, dtype=float)


# Base model probabilities + anomaly/rule meta-features fed to the stacker.
# v30: Simplified to core 9 features — removed lof/ocsvm (weak, AP<0.09) and
# individual rule flags (rule_score already captures combined effect). Keeping
# max/std meta-features for model-consensus signal.
STACKER_FEATURE_COLUMNS = [
    "base_a_probability",
    "base_c_s_probability",  # v35: Correct-and-Smooth post-processing of Base A (graph-corrected)
    "base_b_probability",
    "base_c_probability",
    "base_d_probability",
    "base_e_probability",
    "rule_score",
    "anomaly_score",
    # Meta-features computed from base probabilities.
    # max_base: at least one model strongly suspects fraud.
    # std_base: model disagreement — high std suggests uncertain/novel case.
    "max_base_probability",
    "std_base_probability",
    # v32: Interaction features for nonlinear stacker.
    # base_a × anomaly: both models flag same user → very high confidence.
    # base_a × rule: model + domain rule agreement → precision boost.
    # These help depth-3 CatBoost find tight positive clusters.
    "base_a_x_anomaly",
    "base_a_x_rule",
    # v37: Graph-confirmed fraud interaction — AP=0.310 > either individual model.
    # base_a × C&S: when label-free CatBoost AND graph-propagated signal both agree,
    # the product is very high (high-confidence fraud) vs negatives where one is near 0.
    "base_a_x_cs",
    # v41: Additional cross-model confirmation features — AP=0.303-0.309 on OOF.
    # base_d × C&S: LightGBM (leaf-wise, different regularization) × graph correction.
    #   AP=0.303 > base_d alone (0.271) — filtering to LightGBM-AND-graph-confirmed fraud.
    # base_e × C&S: XGBoost × graph correction.
    #   AP=0.309 > base_e alone (0.287) — similar cross-model amplification.
    # These are orthogonal to base_a_x_cs (CatBoost-based) because LightGBM/XGBoost
    # use different tree structures and regularization, capturing different FN subgroups.
    # Not in blend candidates (would require n=6 at step=0.05 → 53K combos, OOM).
    # Reserved for future nonlinear stacker experiments.
    "base_d_x_cs",
    "base_e_x_cs",
    # v43: Multi-source confirmation features.
    # base_a_x_e: CatBoost × XGBoost label-free cross-model agreement.
    #   Both trained on same features but with fundamentally different regularization.
    #   Product amplifies users flagged by two independent algorithms simultaneously.
    # base_cs_x_anomaly: C&S probability × anomaly score.
    #   Graph-propagated fraud signal × statistical outlier detection.
    #   Targets users who are BOTH graph-suspicious AND statistically anomalous.
    #   Added as blend candidate: product AP expected > individual AP of anomaly alone.
    # base_b_x_cs: Transductive CatBoost × C&S — two independent graph-informed signals.
    #   base_b uses graph structure as features; C&S propagates scores on the graph.
    #   Agreement between these two orthogonal graph methods → very high confidence.
    "base_a_x_e",
    "base_cs_x_anomaly",
    "base_b_x_cs",
    # v44: Graph isolation signal — measures how much the graph REDUCED the CatBoost prediction.
    # cs_deficit = base_a - base_c_s_probability.
    # Positive value: C&S smoothed the user DOWN (they're graph-isolated or near negatives).
    # Negative value: C&S boosted the user UP (they're connected to fraud clusters).
    # Analysis: cs_deficit AP=0.197, pos_mean=0.152, neg_mean=0.047.
    # Isolated positives (735, 45% of all positives) have cs_deficit≈0.11 (graph hurt them).
    # Valuable for nonlinear stackers that can learn: "high base_a + large cs_deficit → isolated fraudster".
    "cs_deficit",
    # v49: Crypto-anomaly interaction features.
    # base_cs_x_crypto_anomaly: C&S × crypto_anomaly_score — graph-suspicious AND crypto-anomalous.
    # base_a_x_crypto_anomaly: label-free CatBoost × crypto_anomaly — targets FN fraud pattern.
    "base_cs_x_crypto_anomaly",
    "base_a_x_crypto_anomaly",
]

# Columns eligible for the AP-weighted blend (non-rule, non-meta columns).
# Only probability-scale columns are used for blend weighting.
_BLEND_CANDIDATE_COLUMNS = [
    "base_a_probability",
    "base_c_s_probability",  # v35: C&S — AP > 0.08 expected (offline test: AP=0.2915)
    "base_b_probability",
    "base_c_probability",
    "base_d_probability",
    "base_e_probability",
    "anomaly_score",
    # NOTE v37: base_a_x_cs (AP=0.310) was tested as a blend candidate but
    # caused -0.002 F1 regression: adding it as a 6th eligible col forces step
    # 0.05→0.10, losing fine-grained 5% allocations for base_a and base_d.
    # It remains in STACKER_FEATURE_COLUMNS for future non-linear stacker use
    # and is computed in _add_base_meta_features.
    # v43: base_cs_x_anomaly as blend candidate — graph + anomaly dual confirmation.
    # Expected AP > anomaly alone (0.08-0.15) and > C&S alone (0.295) is unlikely,
    # but product AP > 0.08 threshold should be met; top-5 selection filters winners.
    # If AP of this product > anomaly_score AP, it replaces anomaly with a more
    # precise signal: only users who are BOTH graph-suspicious AND statistically anomalous.
    "base_cs_x_anomaly",
    # v49: Crypto-anomaly blend candidates — IsoForest trained on crypto features only.
    # Targets FN fraud pattern: high crypto volume, non-structuring, older accounts.
    # Top-5 AP selection will include these only if they outperform current blend members.
    "base_cs_x_crypto_anomaly",
    "anomaly_score_segmented",
]

# Minimum AP threshold to include a model in the blend.
# Models with AP < this are excluded to avoid noise injection.
_MIN_AP_FOR_BLEND = 0.08


class BlendEnsemble:
    """AP-weighted linear blend ensemble — drop-in replacement for the LR stacker.

    Empirically outperforms LogisticRegression on OOF data when base models have
    varying quality (AP range 0.05-0.29): the unconstrained LR is dominated by
    Base A (coef=3.5x others) but still dragged down by near-random Base C.
    The blend approach excludes low-AP models and normalizes weights, achieving
    F1=0.3550 vs LR F1=0.3435 on pre-v30 OOF.

    v50: Optional segment-aware blending — if `isolated_weights` is provided,
    uses a different weight set for isolated users (cs_deficit > 0.05). This
    avoids over-weighting base_c_s and base_cs_x_anomaly for isolated users
    (whose C&S scores are artificially compressed), boosting recall for Type B
    (isolated, older-account, high-crypto) fraudsters.

    Interface: compatible with sklearn's predict_proba(X) -> array shape (n, 2).
    """

    def __init__(
        self,
        weights: dict[str, float],
        isolated_weights: dict[str, float] | None = None,
    ) -> None:
        self.weights = {k: float(v) for k, v in weights.items() if float(v) > 0}
        self.isolated_weights = (
            {k: float(v) for k, v in isolated_weights.items() if float(v) > 0}
            if isolated_weights else None
        )

    def _apply_weights(self, X: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
        blend = np.zeros(len(X), dtype=float)
        total_weight = 0.0
        for col, w in weights.items():
            if col in X.columns:
                vals = pd.to_numeric(X[col], errors="coerce").fillna(0.0).to_numpy()
                blend += w * vals
                total_weight += w
        if total_weight > 0:
            blend = blend / total_weight
        return blend

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.isolated_weights is not None and "cs_deficit" in X.columns:
            # Segment-aware blending: use isolated_weights for users where
            # C&S significantly compressed the score (cs_deficit > 0.05).
            is_isolated = pd.to_numeric(X["cs_deficit"], errors="coerce").fillna(0.0).to_numpy() > 0.05
            blend_connected = self._apply_weights(X, self.weights)
            blend_isolated = self._apply_weights(X, self.isolated_weights)
            blend = np.where(is_isolated, blend_isolated, blend_connected)
        else:
            blend = self._apply_weights(X, self.weights)
        blend = np.clip(blend, 0.0, 1.0)
        return np.column_stack([1.0 - blend, blend])

_BASE_PROB_COLUMNS = [
    "base_a_probability", "base_c_s_probability", "base_b_probability",
    "base_c_probability", "base_d_probability", "base_e_probability",
]


def _add_base_meta_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute max/std across base model probabilities and interaction features for stacker enrichment."""
    frame = frame.copy()
    available = [c for c in _BASE_PROB_COLUMNS if c in frame.columns]
    if available:
        frame["max_base_probability"] = frame[available].max(axis=1)
        frame["std_base_probability"] = frame[available].std(axis=1).fillna(0.0)
    else:
        frame["max_base_probability"] = 0.0
        frame["std_base_probability"] = 0.0
    # v32: Interaction features — model × anomaly/rule agreement signals.
    # These help the nonlinear CatBoost stacker (depth=3) find tight positive
    # clusters where both model and anomaly/rule sources flag the same user.
    a = pd.to_numeric(frame.get("base_a_probability", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    anomaly = pd.to_numeric(frame.get("anomaly_score", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    rule = pd.to_numeric(frame.get("rule_score", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    frame["base_a_x_anomaly"] = (a * anomaly).astype(np.float32)
    frame["base_a_x_rule"] = (a * rule).astype(np.float32)
    # v37: Graph-confirmed fraud interactions — AP=0.310 > base_a(0.298) or C&S(0.295).
    # base_a_x_cs amplifies cases where BOTH label-free model AND graph-corrected model
    # agree: product squeezes negative-score overlap for users where only one model fires.
    # Eligible for blend (AP > 0.08 threshold) and improves blend F1 when step adaptive.
    cs = pd.to_numeric(frame.get("base_c_s_probability", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    frame["base_a_x_cs"] = (a * cs).astype(np.float32)
    # v41: Cross-model confirmation products for future nonlinear stacker experiments.
    # base_d_x_cs (AP=0.303): LightGBM × C&S — orthogonal to CatBoost-based base_a_x_cs.
    # base_e_x_cs (AP=0.309): XGBoost × C&S — further diversity.
    d = pd.to_numeric(frame.get("base_d_probability", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    e = pd.to_numeric(frame.get("base_e_probability", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    frame["base_d_x_cs"] = (d * cs).astype(np.float32)
    frame["base_e_x_cs"] = (e * cs).astype(np.float32)
    # v43: Multi-source confirmation features.
    # base_a_x_e: CatBoost × XGBoost — two label-free models with fundamentally
    #   different regularization (L2-leaf vs column-block). Product amplifies users
    #   flagged simultaneously by both, reducing single-model FP noise.
    b = pd.to_numeric(frame.get("base_b_probability", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    frame["base_a_x_e"] = (a * e).astype(np.float32)
    # base_cs_x_anomaly: C&S × anomaly — graph-propagated fraud signal × statistical
    #   outlier detection. Users who are BOTH graph-suspicious AND statistically anomalous
    #   are very likely fraudsters. Expected AP > anomaly alone; eligible for blend.
    frame["base_cs_x_anomaly"] = (cs * anomaly).astype(np.float32)
    # base_b_x_cs: Transductive CatBoost × C&S — two independent graph-informed signals.
    #   base_b uses graph structure as features; C&S propagates scores on the graph.
    #   Their product provides strong confirmation for graph-connected fraud.
    frame["base_b_x_cs"] = (b * cs).astype(np.float32)
    # v44: Graph isolation residual — base_a minus C&S-corrected probability.
    # Positive = C&S reduced the prediction (user is graph-isolated or near negatives).
    # Negative = C&S boosted the prediction (user is connected to fraud clusters).
    # Provides explicit isolation signal for nonlinear stackers.
    frame["cs_deficit"] = (a - cs).astype(np.float32)
    # v49: Crypto-anomaly interaction — C&S × crypto_anomaly_score.
    # crypto_anomaly_score is an IsoForest trained on crypto features only; it targets
    # the FN fraud pattern (high crypto volume, non-structuring). The interaction with
    # C&S filters to users who are BOTH graph-suspicious AND crypto-anomalous.
    crypto_anomaly = pd.to_numeric(
        frame.get("crypto_anomaly_score", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)
    frame["base_cs_x_crypto_anomaly"] = (cs * crypto_anomaly).astype(np.float32)
    frame["base_a_x_crypto_anomaly"] = (a * crypto_anomaly).astype(np.float32)
    return frame


CALIBRATION_CANDIDATES = {
    "raw": IdentityCalibrator,
    "sigmoid": SigmoidCalibrator,
    "beta": BetaCalibrator,
    "isotonic": IsotonicCalibrator,
}


def fit_logistic_stacker(frame: pd.DataFrame, feature_columns: list[str]) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(frame[feature_columns], frame["status"].astype(int))
    return model


def tune_blend_weights_segmented(
    frame: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, float]]:
    """Tune separate blend weights for connected vs isolated users.

    Isolated users (cs_deficit > 0.05) have their C&S scores compressed by ~50%,
    so the optimal blend weights for them should rely less on base_c_s_probability
    and base_cs_x_anomaly, and more on base_a_probability directly.

    Returns (connected_weights, isolated_weights). Falls back to a single weight
    set if either segment is too small (<100 labeled users).
    """
    labeled = frame.dropna(subset=["status"])
    cs_def = pd.to_numeric(
        labeled.get("cs_deficit", pd.Series(0.0, index=labeled.index)),
        errors="coerce",
    ).fillna(0.0)
    connected_mask = cs_def <= 0.05
    isolated_mask = cs_def > 0.05

    connected_frame = labeled[connected_mask]
    isolated_frame = labeled[isolated_mask]

    min_size = 100
    if connected_mask.sum() >= min_size:
        connected_weights = tune_blend_weights(connected_frame)
    else:
        connected_weights = tune_blend_weights(labeled)

    if isolated_mask.sum() >= min_size:
        isolated_weights = tune_blend_weights(isolated_frame)
    else:
        isolated_weights = connected_weights

    print(
        f"[stacker] Segmented blend: {connected_mask.sum()} connected, "
        f"{isolated_mask.sum()} isolated labeled users"
    )
    print(f"[stacker]   connected weights: {connected_weights}")
    print(f"[stacker]   isolated weights:  {isolated_weights}")
    return connected_weights, isolated_weights


def tune_blend_weights(frame: pd.DataFrame) -> dict[str, float]:
    """Grid-search blend weights on OOF predictions to maximize bootstrap-mean F1.

    Strategy:
    1. Identify eligible columns (AP >= _MIN_AP_FOR_BLEND).
    2. Grid-search over weight combinations using coarse-to-fine resolution.
    3. Return the weight dict that maximizes OOF F1.

    The grid search is fast (~1s) because it operates on precomputed OOF arrays.
    """
    labeled = frame.dropna(subset=["status"])
    y = labeled["status"].astype(int).to_numpy()

    # Identify eligible columns.
    eligible: dict[str, np.ndarray] = {}
    for col in _BLEND_CANDIDATE_COLUMNS:
        if col not in labeled.columns:
            continue
        vals = pd.to_numeric(labeled[col], errors="coerce").fillna(0.0).to_numpy()
        if vals.std() < 1e-6:
            continue
        ap = float(average_precision_score(y, vals))
        if ap >= _MIN_AP_FOR_BLEND:
            eligible[col] = vals

    if not eligible:
        # Fallback: uniform weight on all available blend columns.
        eligible = {
            col: pd.to_numeric(labeled[col], errors="coerce").fillna(0.0).to_numpy()
            for col in _BLEND_CANDIDATE_COLUMNS
            if col in labeled.columns
        }

    if len(eligible) == 1:
        return {list(eligible)[0]: 1.0}

    # v40: When n > 5, keep only top-5 components by AP to preserve step=0.05 resolution.
    # With n=6 (e.g. Base B recovery) the adaptive step would coarsen to 0.10,
    # losing the 5% precision needed for the optimal CS/E/Anom weights.
    # Top-5 by AP ensures the most predictive models dominate the blend while
    # maintaining fine-grained weight search. The 6th model's signal is still
    # indirectly captured via its correlation with top-5 models.
    if len(eligible) > 5:
        top5 = sorted(eligible, key=lambda c: float(average_precision_score(y, eligible[c])), reverse=True)[:5]
        eligible = {c: eligible[c] for c in top5}

    cols = list(eligible)
    arrays = np.stack([eligible[c] for c in cols], axis=0)  # (n_cols, n_samples)
    n = len(cols)

    # Build all weight combinations summing to 1 at adaptive step size.
    # Grid scales as C(parts+n-1, n-1) — exponential in n — target ≤ 10K combos:
    #   n=5, step=0.05 → C(24,4)=10626 combos ✓  (always reached via top-5 selection)
    # v37: Adaptive step so adding interaction columns (base_a_x_cs etc.) doesn't OOM.
    # v40: Top-5 pre-selection guarantees n≤5, so step=0.05 always applies.
    if n <= 5:
        step = 0.05
    elif n == 6:
        step = 0.10
    else:
        step = 0.15
    parts = round(1.0 / step)

    def _integer_compositions(total: int, k: int) -> list[list[int]]:
        if k == 1:
            return [[total]]
        result: list[list[int]] = []
        for first in range(total + 1):
            for rest in _integer_compositions(total - first, k - 1):
                result.append([first] + rest)
        return result

    combos: list[list[float]] = [[v * step for v in comp] for comp in _integer_compositions(parts, n)]
    if not combos:
        return {col: 1.0 / n for col in cols}

    combo_mat = np.array(combos, dtype=np.float32)  # (n_combos, n_cols)
    # Blend scores: (n_combos, n_samples) = combo_mat @ arrays
    blend_scores = (combo_mat @ arrays).astype(np.float32)  # (n_combos, n_samples)

    # Fully vectorized F1 grid over thresholds.
    # For each threshold: compute TP/FP/FN for ALL combos simultaneously.
    # Memory per iteration: (n_combos, n_samples) bool ≈ n_combos * n_samples bytes.
    thresholds = np.arange(0.05, 0.90, 0.01)
    y_bool = (y == 1)
    pos_total = int(y_bool.sum())
    best_f1_per_combo = np.zeros(len(combo_mat), dtype=np.float32)

    for t in thresholds:
        pred = blend_scores >= t  # (n_combos, n_samples), bool
        tp = pred[:, y_bool].sum(axis=1).astype(np.float32)   # (n_combos,)
        pp = pred.sum(axis=1).astype(np.float32)               # predicted positives
        fp = pp - tp
        fn = pos_total - tp
        denom_p = tp + fp
        denom_r = tp + fn
        prec = np.where(denom_p > 0, tp / denom_p, 0.0)
        rec = np.where(denom_r > 0, tp / denom_r, 0.0)
        denom_f = prec + rec
        f1 = np.where(denom_f > 0, 2.0 * prec * rec / denom_f, 0.0)
        np.maximum(best_f1_per_combo, f1, out=best_f1_per_combo)

    best_idx = int(best_f1_per_combo.argmax())
    best_w = combo_mat[best_idx].tolist()
    return {col: float(w) for col, w in zip(cols, best_w) if w > 1e-6}


def _best_f1(y: np.ndarray, scores: np.ndarray) -> float:
    """Find the peak F1 over a dense threshold grid."""
    best = 0.0
    for t in np.arange(0.05, 0.90, 0.02):
        f = float(f1_score(y, (scores >= t).astype(int), zero_division=0))
        if f > best:
            best = f
    return best


def _fit_catboost_stacker(frame: pd.DataFrame, feature_columns: list[str]) -> Any:
    """Fit a shallow CatBoost stacker for non-linear meta-learning.

    Depth=3 is intentionally shallow to avoid overfitting on the ~50k OOF
    meta-features. Heavy L2 regularization + min_data_in_leaf=30 ensure
    the model only splits on genuinely useful non-linear interactions.
    """
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        return fit_logistic_stacker(frame, feature_columns)

    y = frame["status"].astype(int)
    positives = max(1, int(y.sum()))
    negatives = max(1, len(y) - positives)
    # Cap positive class weight at 5x for the stacker (meta-features are already
    # calibrated probabilities, so extreme imbalance handling is less needed).
    class_weight_ratio = min(float(negatives) / positives, 5.0)
    cat_features = [c for c in feature_columns if frame[c].dtype == bool or str(frame[c].dtype) == "bool"]

    model = CatBoostClassifier(
        depth=3,
        iterations=400,
        learning_rate=0.05,
        l2_leaf_reg=15.0,
        min_data_in_leaf=30,
        random_strength=0.5,
        class_weights=[1.0, class_weight_ratio],
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=RANDOM_SEED,
        verbose=False,
    )
    x = frame[feature_columns].copy()
    for c in feature_columns:
        if x[c].dtype == bool or str(x[c].dtype) == "bool":
            x[c] = x[c].astype(int)
    model.fit(x, y)
    return model


def _predict_stacker(model: Any, frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    """Unified predict that works for both LR and CatBoost stackers."""
    x = frame[feature_columns].copy()
    for c in feature_columns:
        if x[c].dtype == bool or str(x[c].dtype) == "bool":
            x[c] = x[c].astype(int)
    return model.predict_proba(x)[:, 1]


def build_stacker_oof(
    base_oof_frame: pd.DataFrame,
    split_frame: pd.DataFrame,
    fold_column: str = "primary_fold",
    use_nonlinear: bool = False,
    use_blend: bool = True,
    auto_select_stacker: bool = True,
) -> tuple[pd.DataFrame, Any]:
    """Build stacker OOF predictions and return (oof_frame, final_model).

    auto_select_stacker=True (default when use_blend=True): evaluates both
      BlendEnsemble and CatBoost nonlinear stacker via OOF F1, picks the winner.
      BlendEnsemble is fast (no fold loop); CatBoost stacker adds ~60s overhead.
      When auto_select_stacker=True, use_nonlinear is ignored.

    use_blend=True (legacy): AP-weighted blend ensemble (skip CatBoost comparison).
      Outperforms LR stacker when base models have varying quality.
      F1=0.3550 vs LR F1=0.3435 on pre-v30 OOF (+0.012).

    use_blend=False: Original fold-by-fold LR (or CatBoost) meta-learner.
    """
    if fold_column in base_oof_frame.columns:
        frame = base_oof_frame.copy()
    else:
        frame = base_oof_frame.merge(split_frame[["user_id", fold_column]], on="user_id", how="left")
    if frame[fold_column].isna().any():
        raise ValueError(f"Missing fold assignments in {fold_column}")

    # Enrich with base-probability meta-features (max, std across models).
    frame = _add_base_meta_features(frame)

    if use_blend and auto_select_stacker:
        # v48: Auto-select: compare BlendEnsemble vs CatBoost stacker via OOF F1.
        # Both are evaluated on the same OOF frame to ensure fair comparison.
        # Winner is returned as the final model.
        return _auto_select_best_stacker(frame, fold_column)

    if use_blend:
        # Tune AP-proportional blend weights from OOF data.
        blend_weights = tune_blend_weights(frame)
        final_model = BlendEnsemble(blend_weights)
        stacker_cols = [c for c in STACKER_FEATURE_COLUMNS if c in frame.columns]
        frame["stacker_raw_probability"] = final_model.predict_proba(frame[stacker_cols])[:, 1]
        return frame, final_model

    # Original fold-by-fold LR / CatBoost stacker path.
    available_cols = [c for c in STACKER_FEATURE_COLUMNS if c in frame.columns]

    oof_rows: list[pd.DataFrame] = []
    for fold_id in sorted(int(value) for value in frame[fold_column].dropna().unique()):
        valid_frame = frame[frame[fold_column] == fold_id].copy()
        train_frame = frame[frame[fold_column] != fold_id].copy()
        if use_nonlinear:
            model = _fit_catboost_stacker(train_frame, available_cols)
        else:
            model = fit_logistic_stacker(train_frame, available_cols)
        valid_frame["stacker_raw_probability"] = _predict_stacker(model, valid_frame, available_cols)
        oof_rows.append(valid_frame)
    oof_frame = pd.concat(oof_rows, ignore_index=True).sort_values("user_id").reset_index(drop=True)
    if use_nonlinear:
        final_model = _fit_catboost_stacker(frame, available_cols)
    else:
        final_model = fit_logistic_stacker(frame, available_cols)
    return oof_frame, final_model


def _auto_select_best_stacker(
    frame: pd.DataFrame,
    fold_column: str,
) -> tuple[pd.DataFrame, Any]:
    """Auto-select between BlendEnsemble and CatBoost nonlinear stacker via OOF F1.

    Both models are evaluated on the same OOF meta-feature frame (already enriched
    with _add_base_meta_features). The model with higher peak OOF F1 is selected.

    CatBoost path: fold-by-fold training on meta-features, OOF predictions.
    Blend path: direct application of AP-tuned weights to OOF probabilities.
    Both paths produce stacker_raw_probability and the best is returned.
    """
    stacker_cols = [c for c in STACKER_FEATURE_COLUMNS if c in frame.columns]

    # --- Blend path (global weights) ---
    blend_weights = tune_blend_weights(frame)
    blend_model = BlendEnsemble(blend_weights)
    blend_frame = frame.copy()
    blend_frame["stacker_raw_probability"] = blend_model.predict_proba(blend_frame[stacker_cols])[:, 1]
    blend_f1 = _best_f1(
        blend_frame.dropna(subset=["status"])["status"].astype(int).to_numpy(),
        blend_frame.dropna(subset=["status"])["stacker_raw_probability"].to_numpy(),
    )

    # --- Segment-aware blend path (v51: separate weights for isolated vs connected users) ---
    # Isolated users (cs_deficit > 0.05) have C&S scores compressed by ~50%, so tuning
    # separate weights on each segment recovers the information lost for isolated positives.
    try:
        connected_weights, isolated_weights_seg = tune_blend_weights_segmented(frame)
        seg_blend_model = BlendEnsemble(connected_weights, isolated_weights=isolated_weights_seg)
        seg_blend_frame = frame.copy()
        seg_blend_frame["stacker_raw_probability"] = seg_blend_model.predict_proba(seg_blend_frame[stacker_cols])[:, 1]
        seg_blend_f1 = _best_f1(
            seg_blend_frame.dropna(subset=["status"])["status"].astype(int).to_numpy(),
            seg_blend_frame.dropna(subset=["status"])["stacker_raw_probability"].to_numpy(),
        )
    except Exception as exc:
        print(f"[stacker] Segment-aware blend failed: {exc}")
        seg_blend_f1 = -1.0
        seg_blend_model = None
        seg_blend_frame = None

    # --- CatBoost nonlinear stacker path ---
    try:
        oof_rows: list[pd.DataFrame] = []
        for fold_id in sorted(int(v) for v in frame[fold_column].dropna().unique()):
            valid_f = frame[frame[fold_column] == fold_id].copy()
            train_f = frame[frame[fold_column] != fold_id].copy()
            cb_model = _fit_catboost_stacker(train_f, stacker_cols)
            valid_f["stacker_raw_probability"] = _predict_stacker(cb_model, valid_f, stacker_cols)
            oof_rows.append(valid_f)
        cb_oof = pd.concat(oof_rows, ignore_index=True).sort_values("user_id").reset_index(drop=True)
        cb_f1 = _best_f1(
            cb_oof.dropna(subset=["status"])["status"].astype(int).to_numpy(),
            cb_oof.dropna(subset=["status"])["stacker_raw_probability"].to_numpy(),
        )
        cb_final = _fit_catboost_stacker(frame, stacker_cols)
    except Exception:
        cb_f1 = -1.0
        cb_oof = None
        cb_final = None

    print(
        f"[stacker] Blend F1: {blend_f1:.4f} | Seg-blend F1: {seg_blend_f1:.4f} | "
        f"CatBoost stacker F1: {cb_f1:.4f}"
    )
    best_f1 = max(blend_f1, seg_blend_f1, cb_f1)
    if cb_oof is not None and cb_f1 == best_f1 and cb_f1 > blend_f1 + 0.002:
        print(f"[stacker] Selected: CatBoost nonlinear stacker (ΔF1=+{cb_f1 - blend_f1:.4f})")
        return cb_oof, cb_final
    elif seg_blend_frame is not None and seg_blend_f1 == best_f1 and seg_blend_f1 > blend_f1 + 0.001:
        print(f"[stacker] Selected: Segment-aware BlendEnsemble (ΔF1=+{seg_blend_f1 - blend_f1:.4f})")
        return seg_blend_frame, seg_blend_model
    else:
        print(f"[stacker] Selected: BlendEnsemble (global weights)")
        return blend_frame, blend_model


def choose_best_calibration_and_threshold(
    raw_probabilities: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray | None,
    use_pu_adjustment: bool = True,
) -> tuple[dict[str, Any], Any, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    raw_probabilities = np.asarray(raw_probabilities, dtype=float)
    paths = load_official_paths()
    candidate_rows: list[dict[str, Any]] = []
    best_rank: tuple[float, float, float] | None = None
    best_payload: tuple[dict[str, Any], Any, np.ndarray] | None = None

    for method, builder in CALIBRATION_CANDIDATES.items():
        calibrator = builder().fit(raw_probabilities, labels)
        calibrated = calibrator.predict(raw_probabilities)

        # PU Learning adjustment (Elkan-Noto 2008): rescale calibrated
        # probabilities to account for unlabeled true positives.
        if use_pu_adjustment:
            c_estimate = estimate_c(calibrated, labels)
            pu_calibrated = pu_adjust(calibrated, c_estimate)
        else:
            c_estimate = None
            pu_calibrated = calibrated

        threshold_report = search_threshold(labels, pu_calibrated, group_ids, beta=1.0)
        selected_row = threshold_report["selected_row"]
        ap = float(average_precision_score(labels, pu_calibrated))
        candidate_report = {
            "method": method,
            "average_precision": ap,
            "selected_threshold": float(threshold_report["selected_threshold"]),
            "selected_row": dict(selected_row),
            "threshold_report": threshold_report,
            "pu_c_estimate": float(c_estimate) if c_estimate is not None else None,
        }
        candidate_rows.append(candidate_report)
        rank = (
            float(selected_row["bootstrap_mean_f1"]),
            ap,
            -float(selected_row["fpr"]),
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_payload = (candidate_report, calibrator, pu_calibrated, c_estimate)

    assert best_payload is not None
    selected_report, calibrator, pu_calibrated, c_estimate = best_payload
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    calibrator_path = paths.model_dir / f"official_stacker_calibrator_{selected_report['method']}_{timestamp}.pkl"
    save_pickle(calibrator, calibrator_path)
    report = {
        "method": selected_report["method"],
        "average_precision": selected_report["average_precision"],
        "selected_threshold": selected_report["selected_threshold"],
        "selected_row": selected_report["selected_row"],
        "threshold_report": selected_report["threshold_report"],
        "calibrator_path": str(calibrator_path),
        "candidates": candidate_rows,
        "selection_basis": {
            "priority": ["best_bootstrap_mean_f1", "best_average_precision", "lowest_fpr"],
        },
        "pu_c_estimate": float(c_estimate) if c_estimate is not None else None,
        "pu_adjustment_enabled": use_pu_adjustment,
    }
    return report, calibrator, pu_calibrated


def save_stacker_model(model: Any, path: Path) -> None:
    save_pickle(model, path)
