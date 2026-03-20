"""Self-training: iteratively expand positive seeds with high-confidence predictions.

Strategy (fast mode — keeps Base A/D/E fixed across rounds):
  Round 0: Load existing pipeline OOF predictions → score predict_label users
  Round 1: Add pseudo-positives to seed set → rebuild transductive features
           → retrain Base B + GNN only → re-score → select more pseudo-positives
  ...repeat for N rounds

Leakage guard:
  - Pseudo-labels are NEVER used in OOF evaluation.
  - OOF F1 is ALWAYS measured against ORIGINAL labels only.
  - Pseudo-positives only expand the SEED SET used for transductive feature computation.
  - predict_label users are NEVER in the train/valid folds (cohort separation maintained).

Why this works:
  PPR propagation only reaches users within graph distance of the 1,608 known positives.
  Users in disconnected components get zero propagation signal (model can only use tabular).
  Pseudo-positive seeds "light up" previously dark graph regions, propagating signal to
  their graph neighbors that were previously unreachable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from official.common import RANDOM_SEED, feature_output_path, load_official_paths, save_json
from official.graph_dataset import build_transductive_graph
from official.graph_model import train_graphsage_model
from official.modeling import fit_catboost
from official.stacking import STACKER_FEATURE_COLUMNS, BlendEnsemble, tune_blend_weights, _add_base_meta_features
from official.train import (
    _label_frame,
    _label_free_feature_columns,
    _load_dataset,
    _prepare_base_frames,
    _transductive_feature_columns,
    run_transductive_oof_pipeline,
    PRIMARY_GRAPH_MAX_EPOCHS,
    _BASE_A_SEEDS,
    LABEL_FREE_EXCLUDED_COLUMNS,
)
from official.transductive_features import build_transductive_feature_frame
from official.transductive_validation import (
    PrimarySplitSpec,
    build_primary_transductive_splits,
    iter_fold_assignments,
)


def _prediction_entropy(probs: np.ndarray) -> np.ndarray:
    """Compute Shannon entropy (nats) of binary predictions.

    H(p) = -p*ln(p) - (1-p)*ln(1-p), clipped to [0, ln(2)].
    Lower entropy = higher certainty. H=0 at p=0 or p=1; H=ln(2)≈0.693 at p=0.5.

    P0-5: Used to filter out uncertain pseudo-labels in self-training.
    """
    p = np.clip(probs, 1e-9, 1.0 - 1e-9)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def expand_with_pseudo_labels(
    original_label_frame: pd.DataFrame,
    all_user_ids: pd.Series,
    stacker_scores: np.ndarray,
    cohort_series: pd.Series,
    confidence_threshold: float = 0.70,
    max_new: int = 200,
    max_entropy: float | None = None,
) -> tuple[pd.DataFrame, int]:
    """Select high-confidence predict_label users as pseudo-positives.

    Only adds users that:
    - Are in the predict_label cohort (cohort == "predict_label")
    - Have stacker probability >= confidence_threshold
    - Are NOT already in original_label_frame
    - (P0-5) Have prediction entropy <= max_entropy (if specified)

    Args:
        original_label_frame: DataFrame with columns [user_id, status]
        all_user_ids: Series of all user IDs in scoring order
        stacker_scores: probabilities aligned with all_user_ids
        cohort_series: cohort values aligned with all_user_ids
        confidence_threshold: minimum probability to qualify as pseudo-positive
        max_new: maximum new pseudo-positives to add (top-K by probability)
        max_entropy: P0-5 — maximum Shannon entropy (nats) allowed for pseudo-labels.
            Rejects predictions near the decision boundary. E.g. max_entropy=0.3
            accepts only p>=0.95 predictions. None disables entropy filtering.

    Returns:
        (expanded_label_frame, n_new_pseudo_positives)
    """
    original_ids = set(original_label_frame["user_id"].astype(int).tolist())

    # Build scoring DataFrame
    scores = pd.DataFrame({
        "user_id": all_user_ids.values,
        "probability": stacker_scores,
        "cohort": cohort_series.values,
    })

    # P0-5: Entropy filter — reject uncertain predictions
    if max_entropy is not None:
        entropies = _prediction_entropy(stacker_scores)
        scores["_entropy"] = entropies
        n_before = len(scores)
        scores = scores[scores["_entropy"] <= max_entropy]
        n_filtered = n_before - len(scores)
        if n_filtered > 0:
            print(f"  [P0-5] entropy filter: removed {n_filtered} uncertain pseudo-labels (max_H={max_entropy:.3f} nats)")

    # Only consider predict_label users above threshold
    candidates = scores[
        (scores["cohort"] == "predict_label")
        & (~scores["user_id"].astype(int).isin(original_ids))
        & (scores["probability"] >= confidence_threshold)
    ].copy()

    candidates = candidates.nlargest(max_new, "probability")

    if candidates.empty:
        return original_label_frame.copy(), 0

    pseudo = pd.DataFrame({
        "user_id": pd.array(candidates["user_id"].astype(int).tolist(), dtype="Int64"),
        "status": pd.array([1] * len(candidates), dtype="Int64"),
    })
    expanded = pd.concat([original_label_frame, pseudo], ignore_index=True)
    expanded = expanded.drop_duplicates(subset=["user_id"], keep="first")
    return expanded, len(candidates)


def _quick_oof_f1(
    oof_frame: pd.DataFrame,
    stacker_cols: list[str],
    original_user_ids: set[int],
) -> tuple[float, float]:
    """Compute best OOF F1 against original labels only."""
    eval_frame = oof_frame[oof_frame["user_id"].astype(int).isin(original_user_ids)].copy()
    if eval_frame.empty or "stacker_raw_probability" not in eval_frame.columns:
        return 0.0, 0.10
    labels = eval_frame["status"].astype(int).values
    probs = eval_frame["stacker_raw_probability"].values
    best_f1, best_thr = 0.0, 0.10
    for thr in np.arange(0.05, 0.60, 0.01):
        f1 = float(f1_score(labels, (probs >= thr).astype(int), zero_division=0))
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_f1, best_thr


def run_fast_self_training(
    n_rounds: int = 2,
    confidence_threshold: float = 0.70,
    max_new_per_round: int = 200,
    graph_max_epochs: int = None,
    max_entropy: float | None = None,
) -> dict[str, Any]:
    """Run efficient self-training: retrain only transductive components per round.

    Fast mode: Base A / Base D / Base E are trained ONCE (label-free, don't change).
    Each round rebuilds: transductive features + Base B + GNN + stacker.

    Args:
        n_rounds: Number of self-training rounds.
        confidence_threshold: Minimum stacker probability to qualify as pseudo-positive.
        max_new_per_round: Maximum pseudo-positives added per round.
        graph_max_epochs: GNN training epochs (None = use DEFAULT).
        max_entropy: P0-5 — entropy ceiling (nats) for pseudo-label acceptance.
            Filters out borderline predictions near the decision boundary.
            None disables entropy filtering. Typical value: 0.3 (≈ p>=0.95).

    Returns a report with per-round OOF F1 metrics.
    """
    if graph_max_epochs is None:
        graph_max_epochs = PRIMARY_GRAPH_MAX_EPOCHS

    dataset = _load_dataset("full")
    paths = load_official_paths()
    original_label_frame = _label_frame(dataset)
    original_user_ids = set(original_label_frame["user_id"].astype(int).tolist())
    current_label_frame = original_label_frame.copy()

    # Load HPO params if available
    catboost_params: dict | None = None
    try:
        from official.hpo import load_hpo_best_params
        catboost_params = load_hpo_best_params()
    except Exception:
        pass
    # Force CPU for self-training to avoid GPU OOM when pipeline also uses GPU.
    catboost_params = dict(catboost_params or {})
    catboost_params["task_type"] = "CPU"

    # Build primary split (shared across rounds — split is based on original labels)
    primary_split = build_primary_transductive_splits(
        dataset, cutoff_tag="full",
        spec=PrimarySplitSpec(), write_outputs=False,
    )

    # Train Base A / D / E ONCE (label-free — not affected by pseudo-labels)
    print("\n[self_training] Training label-free base models (once)...")
    from official.modeling_xgb import fit_xgboost
    from official.train import _BASE_D_SEEDS, _BASE_E_SEEDS
    from official.rules import evaluate_official_rules

    dataset_with_rules = dataset.copy()
    rule_df = evaluate_official_rules(dataset_with_rules)
    dataset_with_rules = dataset_with_rules.merge(rule_df, on="user_id", how="left")

    # Compute base_a_feature_columns AFTER merging rules so rule_score is available.
    base_a_feature_columns = _label_free_feature_columns(dataset_with_rules)

    label_free_frame_full = dataset_with_rules.copy()
    all_labeled_ids = original_user_ids

    train_lf_full = label_free_frame_full[
        label_free_frame_full["user_id"].astype(int).isin(all_labeled_ids)
    ].copy()

    # Pre-fit Base A (4 seeds) for full-data scoring
    _base_a_finals = []
    for seed in _BASE_A_SEEDS:
        fit = fit_catboost(train_lf_full, None, base_a_feature_columns,
                           focal_gamma=2.0, random_seed=seed, catboost_params=catboost_params)
        _base_a_finals.append(fit.model)

    round_reports: list[dict[str, Any]] = []

    for round_i in range(n_rounds + 1):
        n_pos = int(current_label_frame["status"].astype(int).eq(1).sum())
        n_original = len(original_user_ids)
        n_pseudo = n_pos - int(original_label_frame["status"].astype(int).eq(1).sum())
        print(f"\n[self_training] {'='*50}")
        print(f"[self_training] Round {round_i}: {n_pos} positives ({n_pseudo} pseudo), {len(current_label_frame) - n_pos} negatives")

        # Rebuild graph and transductive features with current label frame (expanded seeds)
        graph = build_transductive_graph(dataset_with_rules)
        sample_trans = build_transductive_feature_frame(graph, current_label_frame)
        base_b_feature_columns = base_a_feature_columns + _transductive_feature_columns(sample_trans)

        # OOF pipeline (Base A + B + GNN) with current expanded label frame
        oof_frame, fold_meta = run_transductive_oof_pipeline(
            dataset_with_rules,
            graph,
            primary_split,
            fold_column="primary_fold",
            base_a_feature_columns=base_a_feature_columns,
            base_b_feature_columns=base_b_feature_columns,
            graph_max_epochs=graph_max_epochs,
            catboost_params=catboost_params,
        )

        # Apply meta-features and blend stacker
        oof_frame = _add_base_meta_features(oof_frame)
        stacker_cols = [c for c in STACKER_FEATURE_COLUMNS if c in oof_frame.columns]
        blend_weights = tune_blend_weights(oof_frame)
        blend_model = BlendEnsemble(blend_weights)
        oof_frame["stacker_raw_probability"] = blend_model.predict_proba(
            oof_frame[stacker_cols]
        )[:, 1]

        # Evaluate against ORIGINAL labels only (leakage guard)
        round_f1, round_thr = _quick_oof_f1(oof_frame, stacker_cols, original_user_ids)
        print(f"[self_training] Round {round_i} OOF F1 (original labels): {round_f1:.4f} @ thr={round_thr:.4f}")
        print(f"[self_training] Blend weights: {blend_weights}")

        round_report: dict[str, Any] = {
            "round": round_i,
            "n_positives_in_seed": n_pos,
            "n_pseudo_positives": n_pseudo,
            "oof_f1_original_labels": round(round_f1, 4),
            "best_threshold": round(round_thr, 4),
            "blend_weights": blend_weights,
        }
        round_reports.append(round_report)

        # Stop after last round (don't expand labels after last round)
        if round_i >= n_rounds:
            break

        # Score ALL users with full-data models for pseudo-label selection
        print(f"[self_training] Scoring all users for pseudo-label selection...")
        transductive_full = build_transductive_feature_frame(graph, current_label_frame)
        _lf_frame, _td_frame = _prepare_base_frames(dataset_with_rules, transductive_full)
        all_labeled_now = set(current_label_frame["user_id"].astype(int).tolist())
        train_td_full = _td_frame[_td_frame["user_id"].astype(int).isin(all_labeled_now)].copy()
        resolved_b_cols = [c for c in train_td_full.columns if c != "user_id"]

        _base_b_params = dict(catboost_params or {})
        _base_b_params["task_type"] = "CPU"
        _base_b_params["l2_leaf_reg"] = 5.0
        base_b_scoring = fit_catboost(train_td_full, None, resolved_b_cols,
                                       focal_gamma=2.0, catboost_params=_base_b_params)
        graph_full = train_graphsage_model(
            graph, label_frame=current_label_frame,
            train_user_ids=all_labeled_now, valid_user_ids=None,
            max_epochs=max(20, graph_max_epochs),
        )

        all_base_a = np.mean(
            [m.predict_proba(_lf_frame[base_a_feature_columns])[:, 1] for m in _base_a_finals],
            axis=0,
        )
        all_base_b = base_b_scoring.model.predict_proba(_td_frame[resolved_b_cols])[:, 1]
        all_base_c = graph_full.full_probabilities

        meta_input = pd.DataFrame({
            "base_a_probability": all_base_a,
            "base_b_probability": all_base_b,
            "base_c_probability": all_base_c,
            "rule_score": pd.to_numeric(_lf_frame.get("rule_score", pd.Series(0.0, index=_lf_frame.index)), errors="coerce").fillna(0.0).values,
            "anomaly_score": pd.to_numeric(_lf_frame.get("anomaly_score", pd.Series(0.0, index=_lf_frame.index)), errors="coerce").fillna(0.0).values,
        })
        meta_input = _add_base_meta_features(meta_input.assign(user_id=_lf_frame["user_id"].values))
        stacker_cols_full = [c for c in stacker_cols if c in meta_input.columns]
        all_stacker_probs = blend_model.predict_proba(meta_input[stacker_cols_full])[:, 1]

        # Expand label frame with pseudo-positives
        current_label_frame, n_new = expand_with_pseudo_labels(
            current_label_frame,
            _lf_frame["user_id"],
            all_stacker_probs,
            dataset_with_rules.loc[_lf_frame.index, "cohort"] if "cohort" in dataset_with_rules.columns else pd.Series("predict_label", index=_lf_frame.index),
            confidence_threshold=confidence_threshold,
            max_new=max_new_per_round,
            max_entropy=max_entropy,
        )
        print(f"[self_training] Added {n_new} pseudo-positives (threshold={confidence_threshold})")
        round_report["n_pseudo_added"] = n_new

    # Save results
    output_path = paths.report_dir / "self_training_report.json"
    result = {
        "rounds": round_reports,
        "final_round": round_reports[-1],
        "total_pseudo_positives": sum(r.get("n_pseudo_added", 0) for r in round_reports),
        "confidence_threshold": confidence_threshold,
        "max_new_per_round": max_new_per_round,
        "max_entropy": max_entropy,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(result, output_path)
    print(f"\n[self_training] Results saved to {output_path}")
    print(f"[self_training] Final OOF F1: {round_reports[-1]['oof_f1_original_labels']:.4f}")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Self-training pipeline")
    parser.add_argument("--rounds", type=int, default=2, help="Number of rounds (default: 2)")
    parser.add_argument("--threshold", type=float, default=0.70, help="Confidence threshold (default: 0.70)")
    parser.add_argument("--max-new", type=int, default=200, help="Max pseudo-positives per round (default: 200)")
    args = parser.parse_args()

    result = run_fast_self_training(
        n_rounds=args.rounds,
        confidence_threshold=args.threshold,
        max_new_per_round=args.max_new,
    )
    print(json.dumps(result["final_round"], indent=2))
