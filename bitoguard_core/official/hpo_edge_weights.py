"""Edge weight HPO via Optuna — optimize graph edge weights to maximize OOF F1.

Strategy:
  Search over 7 edge weights that control propagation strength in the transductive graph.
  Each trial rebuilds transductive features with the candidate weights, runs Base B OOF
  (the model that directly uses these features), and evaluates OOF F1 against original labels.

  Fast mode: Only Base B is retrained per trial (Base A and GNN are expensive; edge weights
  primarily affect transductive feature propagation used by Base B).

Usage:
  cd bitoguard_core
  BITOGUARD_AWS_EVENT_CLEAN_DIR=data/aws_event/clean \\
    PYTHONPATH=. python -m official.hpo_edge_weights \\
    --n-trials 50 --timeout 3600

Results saved to: artifacts/reports/hpo_edge_weights_report.json
Best weights saved to: artifacts/reports/hpo_edge_weights_best.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# Default weights (current hand-tuned values, used as starting point for HPO).
DEFAULT_WEIGHTS: dict[str, float] = {
    "relation": 1.0,
    "wallet_small": 0.70,
    "wallet_medium": 0.40,
    "ip_small": 0.50,
    "ip_medium": 0.25,
    "temporal_small": 0.30,
    "temporal_medium": 0.15,
}

# Path where best weights are persisted between runs.
_BEST_WEIGHTS_PATH_KEY = "hpo_edge_weights_best.json"


def load_best_edge_weights() -> dict[str, float] | None:
    """Load best edge weights from a previous HPO run, or None if not available."""
    from official.common import load_official_paths
    paths = load_official_paths()
    p = paths.report_dir / _BEST_WEIGHTS_PATH_KEY
    if p.exists():
        try:
            data = json.loads(p.read_text())
            weights = data.get("best_weights")
            if isinstance(weights, dict):
                return {str(k): float(v) for k, v in weights.items()}
        except Exception:
            pass
    return None


def _objective(trial: Any, dataset_with_rules: pd.DataFrame, primary_split: Any,
               base_a_feature_columns: list[str], catboost_params: dict | None,
               original_user_ids: set[int]) -> float:
    """Optuna objective: build graph with candidate weights → OOF Base B AP."""
    from official.graph_dataset import build_transductive_graph
    from official.transductive_features import build_transductive_feature_frame
    from official.modeling import fit_catboost
    from official.transductive_validation import iter_fold_assignments
    from sklearn.metrics import average_precision_score

    candidate_weights = {
        "relation":        trial.suggest_float("relation",        0.3, 2.0),
        "wallet_small":    trial.suggest_float("wallet_small",    0.1, 1.5),
        "wallet_medium":   trial.suggest_float("wallet_medium",   0.05, 1.0),
        "ip_small":        trial.suggest_float("ip_small",        0.1, 1.5),
        "ip_medium":       trial.suggest_float("ip_medium",       0.05, 0.8),
        "temporal_small":  trial.suggest_float("temporal_small",  0.05, 0.8),
        "temporal_medium": trial.suggest_float("temporal_medium", 0.02, 0.5),
    }

    # Build graph with candidate weights.
    graph = build_transductive_graph(dataset_with_rules, edge_weights=candidate_weights)

    # Build label frame from primary split.
    label_col = "status"
    user_id_col = "user_id"
    label_frame = dataset_with_rules[
        dataset_with_rules[user_id_col].astype(int).isin(original_user_ids)
    ][[user_id_col, label_col]].copy()
    label_frame[user_id_col] = label_frame[user_id_col].astype(int)
    label_frame[label_col] = pd.to_numeric(label_frame[label_col], errors="coerce").astype("Int64")
    label_frame = label_frame.dropna(subset=[label_col])

    trans_frame = build_transductive_feature_frame(graph, label_frame)
    b_cols = [c for c in trans_frame.columns if c != user_id_col]

    # Quick 3-fold OOF evaluation of Base B only (fast proxy for full pipeline AP).
    # primary_split is a DataFrame with user_id, status, primary_fold columns.
    split_df = primary_split[["user_id", "primary_fold"]].copy()
    merged = trans_frame.merge(split_df, on=user_id_col, how="inner")
    merged = merged.merge(label_frame[[user_id_col, label_col]], on=user_id_col, how="inner")
    merged[label_col] = merged[label_col].astype(int)

    folds = sorted(merged["primary_fold"].dropna().unique().tolist())
    oof_probs: list[float] = []
    oof_labels: list[int] = []

    _cb_params = dict(catboost_params or {})
    _cb_params["task_type"] = "CPU"

    for fold_id in folds:
        train_mask = merged["primary_fold"] != fold_id
        valid_mask = merged["primary_fold"] == fold_id
        train_df = merged[train_mask].copy()
        valid_df = merged[valid_mask].copy()
        if train_df[label_col].sum() < 10 or valid_df[label_col].sum() < 5:
            continue
        fit_result = fit_catboost(
            train_df[[user_id_col, *b_cols, label_col]].rename(columns={label_col: "status"}),
            None,
            b_cols,
            focal_gamma=2.0,
            catboost_params=_cb_params,
        )
        probs = fit_result.model.predict_proba(valid_df[b_cols])[:, 1]
        oof_probs.extend(probs.tolist())
        oof_labels.extend(valid_df[label_col].tolist())

    if len(oof_probs) < 50 or sum(oof_labels) < 10:
        return 0.0

    return float(average_precision_score(oof_labels, oof_probs))


def run_edge_weight_hpo(
    n_trials: int = 50,
    timeout: float | None = 3600.0,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Run Optuna HPO for edge weights.

    Returns:
        dict with 'best_weights', 'best_value', 'n_trials' keys.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as exc:
        raise ImportError("optuna is required for edge weight HPO: pip install optuna") from exc

    from official.common import load_official_paths, save_json
    from official.train import (
        _label_frame,
        _label_free_feature_columns,
        _load_dataset,
    )
    from official.transductive_validation import PrimarySplitSpec, build_primary_transductive_splits
    from official.rules import evaluate_official_rules

    try:
        from official.hpo import load_hpo_best_params
        catboost_params = load_hpo_best_params()
    except Exception:
        catboost_params = None

    dataset = _load_dataset("full")
    dataset_with_rules = dataset.copy()
    rule_df = evaluate_official_rules(dataset_with_rules)
    dataset_with_rules = dataset_with_rules.merge(rule_df, on="user_id", how="left")

    original_label_frame = _label_frame(dataset)
    original_user_ids = set(original_label_frame["user_id"].astype(int).tolist())
    base_a_feature_columns = _label_free_feature_columns(dataset)

    primary_split = build_primary_transductive_splits(
        dataset, cutoff_tag="full",
        spec=PrimarySplitSpec(), write_outputs=False,
    )

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=min(15, n_trials // 3))
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Seed with default weights as first trial.
    study.enqueue_trial(DEFAULT_WEIGHTS)

    print(f"[hpo_edge_weights] Starting {n_trials} trials (timeout={timeout}s)...")
    study.optimize(
        lambda trial: _objective(
            trial, dataset_with_rules, primary_split,
            base_a_feature_columns, catboost_params, original_user_ids,
        ),
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=False,
    )

    best_trial = study.best_trial
    best_weights = best_trial.params
    best_value = best_trial.value

    print(f"[hpo_edge_weights] Best Base B AP: {best_value:.4f}")
    print(f"[hpo_edge_weights] Best weights: {best_weights}")

    paths = load_official_paths()
    paths.report_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "best_weights": best_weights,
        "best_value": best_value,
        "n_trials": len(study.trials),
        "default_weights": DEFAULT_WEIGHTS,
    }
    save_json(result, paths.report_dir / "hpo_edge_weights_report.json")
    save_json(result, paths.report_dir / _BEST_WEIGHTS_PATH_KEY)
    print(f"[hpo_edge_weights] Saved to {paths.report_dir / _BEST_WEIGHTS_PATH_KEY}")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Edge weight HPO via Optuna")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials (default: 50)")
    parser.add_argument("--timeout", type=float, default=3600.0, help="Time budget in seconds (default: 3600)")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs (default: 1)")
    args = parser.parse_args()
    result = run_edge_weight_hpo(n_trials=args.n_trials, timeout=args.timeout, n_jobs=args.n_jobs)
    print(json.dumps({k: v for k, v in result.items() if k != "default_weights"}, indent=2))
