"""Ablation runner for V4 progressive addition and leave-one-out experiments.

Drives ``run_configurable_pipeline`` through a systematic series of steps:
  1. Progressive addition: start from baseline (defaults only) and incrementally
     enable components in priority order, measuring the marginal lift of each.
  2. Leave-one-out: from the best progressive config, disable each component
     one at a time to measure its contribution.

All results are logged via ``experiment_tracker.log_experiment``.

Usage:
    cd bitoguard_core
    PYTHONPATH=. python -m official.ablation_runner
    PYTHONPATH=. python -m official.ablation_runner --mode progressive
    PYTHONPATH=. python -m official.ablation_runner --mode leave_one_out
    PYTHONPATH=. python -m official.ablation_runner --mode both
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from typing import Any

from official.configurable_pipeline import COMPONENTS, DEFAULT_CONFIG, run_configurable_pipeline
from official.experiment_tracker import log_experiment, print_leaderboard


# ---------------------------------------------------------------------------
# Progressive addition order (steps 0-17)
# ---------------------------------------------------------------------------
# Each step adds ONE new component on top of all prior enabled components.
# Order reflects the V4 plan: high-impact / low-risk items first.
PROGRESSIVE_STEPS: list[dict[str, Any]] = [
    # Step 0: Pure baseline (only defaults enabled: multi_scale_ppr, temporal_edges, graphsage_3layer)
    {"step": 0, "name": "baseline", "add": None},
    # Step 1: Edge weight HPO (use Optuna-tuned edge weights if available)
    {"step": 1, "name": "+edge_weight_hpo", "add": "edge_weight_hpo"},
    # Step 2: Threshold HPO
    {"step": 2, "name": "+threshold_hpo", "add": "threshold_hpo"},
    # Step 3: PU learning loss (reduce negative weight)
    {"step": 3, "name": "+pu_learning_loss", "add": "pu_learning_loss"},
    # Step 4: GBM stacker
    {"step": 4, "name": "+gbm_stacker", "add": "gbm_stacker"},
    # Step 5: Hub IP pruning
    {"step": 5, "name": "+hub_ip_pruning", "add": "hub_ip_pruning"},
    # Step 6: Negative propagation
    {"step": 6, "name": "+negative_propagation", "add": "negative_propagation"},
    # Step 7: Feature engineering node attributes
    {"step": 7, "name": "+feature_eng_node_attrs", "add": "feature_eng_node_attrs"},
    # Step 8: Profile similarity edges
    {"step": 8, "name": "+profile_similarity_edges", "add": "profile_similarity_edges"},
    # Step 9: Directed flow edges
    {"step": 9, "name": "+directed_flow_edges", "add": "directed_flow_edges"},
    # Step 10: Edge time decay
    {"step": 10, "name": "+edge_time_decay", "add": "edge_time_decay"},
    # Step 11: Label spreading
    {"step": 11, "name": "+label_spreading", "add": "label_spreading"},
    # Step 12: Node2Vec embeddings
    {"step": 12, "name": "+node2vec_embeddings", "add": "node2vec_embeddings"},
    # Step 13: Self-training (most expensive, run last in Tier A-B)
    {"step": 13, "name": "+self_training", "add": "self_training"},
    # ── Tier D: New methods ──────────────────────────────────────────────────
    # Step 14: Community detection features (Louvain) — graph cluster membership
    {"step": 14, "name": "+community_features", "add": "community_features"},
    # Step 15: Lag/cross-channel correlation features — temporal AML patterns
    {"step": 15, "name": "+lag_features", "add": "lag_features"},
    # Step 16: GRU sequence branch — temporal transaction encoder
    {"step": 16, "name": "+gru_sequence_branch", "add": "gru_sequence_branch"},
    # Step 17: DGI RF features — self-supervised graph embeddings + RF score
    {"step": 17, "name": "+dgi_rf_features", "add": "dgi_rf_features"},
    # Step 18: nnPU-corrected weights — non-negative PU risk estimator
    {"step": 18, "name": "+nnpu_weights", "add": "nnpu_weights"},
    # Step 19: Hard negative mining — up-weight borderline negatives
    {"step": 19, "name": "+hard_negative_mining", "add": "hard_negative_mining"},
    # Step 20: Seed perturbation averaging — ensemble over perturbed label sets
    {"step": 20, "name": "+seed_perturbation_avg", "add": "seed_perturbation_avg"},
    # Step 21: Conditional C&S restore — restore top-5% isolated users by base_a score
    {"step": 21, "name": "+conditional_cs_restore", "add": "conditional_cs_restore"},
    # Step 22: Base B transductive-only — specialize Base B on graph features only
    {"step": 22, "name": "+base_b_transductive_only", "add": "base_b_transductive_only"},
    # Step 23: Uncertainty entropy filter (P0-5) — for self_training, reject borderline pseudo-labels
    {"step": 23, "name": "+uncertainty_entropy_filter", "add": "uncertainty_entropy_filter"},
]


def _build_progressive_config(up_to_step: int) -> dict[str, bool]:
    """Build config with all components enabled up to (and including) the given step."""
    config = dict(DEFAULT_CONFIG)
    for step_info in PROGRESSIVE_STEPS:
        if step_info["step"] > up_to_step:
            break
        component = step_info["add"]
        if component is not None:
            config[component] = True
    return config


def run_progressive_ablation(
    start_step: int = 0,
    end_step: int | None = None,
    catboost_params: dict | None = None,
) -> list[dict[str, Any]]:
    """Run progressive addition ablation from start_step to end_step.

    Returns list of result dicts (one per step).
    """
    max_step = len(PROGRESSIVE_STEPS) - 1
    if end_step is None:
        end_step = max_step
    end_step = min(end_step, max_step)

    results: list[dict[str, Any]] = []
    print(f"\n{'='*80}")
    print(f"PROGRESSIVE ADDITION ABLATION: steps {start_step} -> {end_step}")
    print(f"{'='*80}")

    for step_info in PROGRESSIVE_STEPS:
        step = step_info["step"]
        if step < start_step or step > end_step:
            continue

        name = step_info["name"]
        config = _build_progressive_config(step)
        experiment_id = f"v4_progressive_step{step:02d}_{name}"

        print(f"\n{'='*70}")
        print(f"Step {step}/{end_step}: {name}")
        enabled = [k for k, v in config.items() if v]
        print(f"Enabled: {enabled}")
        print(f"{'='*70}")

        try:
            metrics = run_configurable_pipeline(
                config,
                experiment_id=experiment_id,
                catboost_params=catboost_params,
            )
        except Exception as exc:
            traceback.print_exc()
            print(f"[ablation] Step {step} FAILED: {exc}")
            metrics = {"f1": 0.0, "error": str(exc), "experiment_id": experiment_id}

        # Log to experiment tracker
        log_experiment(
            experiment_id=experiment_id,
            config=config,
            metrics=metrics,
            notes=f"Progressive step {step}: {name}",
        )
        results.append({"step": step, "name": name, "config": config, "metrics": metrics})

    # Print summary
    _print_progressive_summary(results)
    return results


def run_leave_one_out(
    base_config: dict[str, bool] | None = None,
    catboost_params: dict | None = None,
) -> list[dict[str, Any]]:
    """Run leave-one-out ablation from the given base config.

    For each enabled non-default component, disable it and measure the delta.

    Returns list of result dicts.
    """
    if base_config is None:
        # Use all components enabled (full config)
        base_config = _build_progressive_config(len(PROGRESSIVE_STEPS) - 1)

    # First: run baseline with all components
    print(f"\n{'='*80}")
    print(f"LEAVE-ONE-OUT ABLATION")
    print(f"{'='*80}")

    base_id = "v4_loo_baseline_all"
    print(f"\nRunning baseline (all enabled)...")
    try:
        base_metrics = run_configurable_pipeline(
            base_config, experiment_id=base_id, catboost_params=catboost_params,
        )
    except Exception as exc:
        traceback.print_exc()
        print(f"[ablation] Baseline FAILED: {exc}")
        base_metrics = {"f1": 0.0, "error": str(exc), "experiment_id": base_id}

    log_experiment(
        experiment_id=base_id, config=base_config, metrics=base_metrics,
        notes="LOO baseline: all components enabled",
    )

    base_f1 = base_metrics.get("f1", 0.0)
    results: list[dict[str, Any]] = [{"name": "baseline_all", "config": base_config, "metrics": base_metrics, "delta_f1": 0.0}]

    # Find non-default components that are enabled
    toggleable = [
        name for name, enabled in base_config.items()
        if enabled and not COMPONENTS.get(name, {}).get("default", False)
    ]

    print(f"\nToggleable components: {toggleable}")

    for component in toggleable:
        config = dict(base_config)
        config[component] = False
        experiment_id = f"v4_loo_minus_{component}"

        print(f"\n{'='*60}")
        print(f"LOO: disabling {component}")
        print(f"{'='*60}")

        try:
            metrics = run_configurable_pipeline(
                config, experiment_id=experiment_id, catboost_params=catboost_params,
            )
        except Exception as exc:
            print(f"[ablation] LOO {component} FAILED: {exc}")
            metrics = {"f1": 0.0, "error": str(exc), "experiment_id": experiment_id}

        delta = base_f1 - metrics.get("f1", 0.0)
        log_experiment(
            experiment_id=experiment_id, config=config, metrics=metrics,
            notes=f"LOO: disabled {component}, delta_f1={delta:+.4f}",
        )
        results.append({"name": f"-{component}", "config": config, "metrics": metrics, "delta_f1": delta})

    # Print summary
    _print_loo_summary(results, base_f1)
    return results


def _print_progressive_summary(results: list[dict[str, Any]]) -> None:
    """Print a formatted summary table for progressive ablation results."""
    print(f"\n{'='*90}")
    print(f"PROGRESSIVE ADDITION SUMMARY")
    print(f"{'='*90}")
    print(f"{'Step':<6} {'Name':<35} {'F1':>8} {'P':>8} {'R':>8} {'AP':>8} {'Thr':>8} {'Time':>8}")
    print(f"{'-'*90}")

    prev_f1 = 0.0
    for r in results:
        m = r["metrics"]
        f1 = m.get("f1", 0.0)
        delta = f1 - prev_f1
        delta_str = f"({delta:+.4f})" if r["step"] > 0 else ""
        print(
            f"{r['step']:<6} {r['name']:<35} "
            f"{f1:>8.4f} {m.get('precision', 0):>8.4f} "
            f"{m.get('recall', 0):>8.4f} {m.get('pr_auc', 0):>8.4f} "
            f"{m.get('threshold', 0):>8.4f} {m.get('elapsed_seconds', 0):>7.0f}s"
        )
        if delta_str:
            print(f"{'':>41} {delta_str}")
        prev_f1 = f1
    print(f"{'='*90}")

    best = max(results, key=lambda r: r["metrics"].get("f1", 0.0))
    print(f"\nBest step: {best['step']} ({best['name']}) — F1={best['metrics'].get('f1', 0):.4f}")


def _print_loo_summary(results: list[dict[str, Any]], base_f1: float) -> None:
    """Print a formatted summary table for leave-one-out results."""
    print(f"\n{'='*80}")
    print(f"LEAVE-ONE-OUT SUMMARY (base F1={base_f1:.4f})")
    print(f"{'='*80}")
    print(f"{'Component':<35} {'F1':>8} {'Delta':>10} {'Impact':>10}")
    print(f"{'-'*80}")

    for r in sorted(results, key=lambda x: x.get("delta_f1", 0), reverse=True):
        m = r["metrics"]
        f1 = m.get("f1", 0.0)
        delta = r.get("delta_f1", 0.0)
        impact = "CRITICAL" if delta > 0.01 else "helpful" if delta > 0.002 else "neutral" if delta > -0.002 else "HARMFUL"
        print(f"{r['name']:<35} {f1:>8.4f} {delta:>+10.4f} {impact:>10}")
    print(f"{'='*80}")


def run_single_experiment(
    config: dict[str, bool],
    experiment_id: str,
    catboost_params: dict | None = None,
) -> dict[str, float]:
    """Run a single experiment and log it."""
    metrics = run_configurable_pipeline(config, experiment_id=experiment_id, catboost_params=catboost_params)
    log_experiment(experiment_id=experiment_id, config=config, metrics=metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="V4 ablation runner")
    parser.add_argument(
        "--mode",
        choices=["progressive", "leave_one_out", "both", "leaderboard"],
        default="progressive",
        help="Ablation mode (default: progressive)",
    )
    parser.add_argument("--start-step", type=int, default=0, help="Start step for progressive (default: 0)")
    parser.add_argument("--end-step", type=int, default=None, help="End step for progressive (default: all)")
    args = parser.parse_args()

    if args.mode == "leaderboard":
        print_leaderboard(top_n=30)
        return

    if args.mode in ("progressive", "both"):
        progressive_results = run_progressive_ablation(
            start_step=args.start_step,
            end_step=args.end_step,
        )

    if args.mode in ("leave_one_out", "both"):
        # For LOO: use the full config (all components enabled)
        run_leave_one_out()

    print_leaderboard(top_n=20)


if __name__ == "__main__":
    main()
