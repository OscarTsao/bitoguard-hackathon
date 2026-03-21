"""
Phase 3A: CatBoost Base A HPO via Optuna (30 trials, ABLATION_FAST).
Phase 3B: Untested flags on best config (seq+temporal+flow) after HPO.
Phase 4:  Stacker architecture experiments (LR, GBM, CatBoost stacker).
Phase 6:  Flow graph variants.

Auto-chains all phases sequentially.
"""
import os, sys, time, traceback

BASE_DIR = "/home/user/YuNing/bitoguard-hackathon"
CORE_DIR = f"{BASE_DIR}/bitoguard_core"
PYTHON   = f"{BASE_DIR}/.venv/bin/python"

os.environ["BITOGUARD_AWS_EVENT_CLEAN_DIR"] = "data/aws_event/clean"
sys.path.insert(0, CORE_DIR)
os.chdir(CORE_DIR)

from official.configurable_pipeline import run_configurable_pipeline
from official.experiment_tracker import log_experiment

BEST_CONFIG = {
    "sequence_features": True,
    "temporal_sequence_features": True,
    "flow_graph_edges": True,
}

results_all = []

def run_exp(exp_id, config, notes, fast=True):
    if fast:
        os.environ["ABLATION_FAST"] = "1"
    else:
        os.environ.pop("ABLATION_FAST", None)
    print(f"\n{'='*60}\n=== {exp_id} ===")
    print(f"Config extras: {[k for k,v in config.items() if v and k not in BEST_CONFIG]}")
    print(f"Mode: {'ABLATION_FAST' if fast else 'FULL params'}")
    try:
        result = run_configurable_pipeline(config=config, experiment_id=exp_id)
        f1 = result["f1"]
        ap = result.get("ap", result.get("average_precision", 0.0))
        elapsed = result.get("elapsed", 0)
        log_experiment(exp_id, config, result, notes=notes)
        print(f"Result: F1={f1:.4f} AP={ap:.4f} ({elapsed:.0f}s)")
        results_all.append((exp_id, f1, ap, "ok"))
        return f1
    except Exception as e:
        print(f"FAILED: {e}\n{traceback.format_exc()[:600]}")
        results_all.append((exp_id, None, None, f"FAILED: {e}"))
        return None

# ─────────────────────────────────────────────────────────────────
# PHASE 3A: CatBoost HPO via Optuna (30 trials, ABLATION_FAST)
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PHASE 3A: CatBoost Base A Optuna HPO")
print("="*70)
os.environ["ABLATION_FAST"] = "1"

hpo_f1 = None
try:
    from official.hpo import run_hpo_study
    t0 = time.time()
    print("[HPO] Starting 30-trial Optuna study (ABLATION_FAST)...")
    result = run_hpo_study(n_trials=30, seed=42)
    elapsed = time.time() - t0
    hpo_f1 = result.get("best_value", 0)
    print(f"[HPO] Done in {elapsed:.0f}s. Best F1={hpo_f1:.4f}")
    print(f"[HPO] Best params: {result.get('best_params', {})}")
except Exception as e:
    print(f"[HPO] FAILED: {e}\n{traceback.format_exc()[:600]}")

# Validate HPO params with best config (ABLATION_FAST first)
run_exp("p3a_hpo_validate_fast", BEST_CONFIG.copy(),
        "Best config + HPO params (ABLATION_FAST validate)", fast=True)

# ─────────────────────────────────────────────────────────────────
# PHASE 3B: Untested flags on best config
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PHASE 3B: Untested flags ablation")
print("="*70)

P3B_EXPERIMENTS = [
    ("p3b_lag",         {**BEST_CONFIG, "seq_lag_features": True},
     "+seq_lag_features on best config"),
    ("p3b_threshold_hpo", {**BEST_CONFIG, "threshold_hpo": True},
     "+threshold_hpo on best config"),
    ("p3b_self_train",  {**BEST_CONFIG, "self_training": True},
     "+self_training on best config"),
    ("p3b_neg_prop",    {**BEST_CONFIG, "negative_propagation": True},
     "+negative_propagation on best config"),
    ("p3b_lag_temp",    {**BEST_CONFIG, "seq_lag_features": True, "temporal_sequence_features": True},
     "+seq_lag + all temporal on best config"),
]

p3b_base_f1 = None
for exp_id, config, notes in P3B_EXPERIMENTS:
    f1 = run_exp(exp_id, config, notes, fast=True)
    if exp_id == "p3b_lag" and f1 is not None:
        p3b_base_f1 = f1

# ─────────────────────────────────────────────────────────────────
# PHASE 4: Stacker architecture (FULL params, NaN fix applied)
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PHASE 4: Stacker architecture experiments (FULL params)")
print("="*70)

P4_EXPERIMENTS = [
    ("p4_lr_stacker",   {**BEST_CONFIG, "lr_stacker": True},
     "Best config + lr_stacker (NaN fix) FULL params"),
    ("p4_gbm_stacker",  {**BEST_CONFIG, "gbm_stacker": True},
     "Best config + gbm_stacker (LightGBM) FULL params"),
    ("p4_best_full",    BEST_CONFIG.copy(),
     "Best config FULL params, CatBoost stacker auto-select"),
]

for exp_id, config, notes in P4_EXPERIMENTS:
    run_exp(exp_id, config, notes, fast=False)

# ─────────────────────────────────────────────────────────────────
# PHASE 6: Flow graph variants
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PHASE 6: Flow graph variants")
print("="*70)

P6_EXPERIMENTS = [
    ("p6_directed_flow",   {**BEST_CONFIG, "directed_flow_edges": True},
     "+directed_flow_edges on best config"),
    ("p6_profile_sim",     {**BEST_CONFIG, "profile_similarity_edges": True},
     "+profile_similarity_edges on best config"),
    ("p6_all_flow",        {**BEST_CONFIG, "directed_flow_edges": True, "profile_similarity_edges": True},
     "+all flow variants on best config"),
]

for exp_id, config, notes in P6_EXPERIMENTS:
    f1 = run_exp(exp_id, config, notes, fast=True)

# ─────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("=== FULL PIPELINE SUMMARY ===")
print(f"{'Experiment':<30} {'F1':>7}  Status")
print("-"*70)
for exp_id, f1, ap, status in results_all:
    if f1 is not None:
        print(f"{exp_id:<30} {f1:>7.4f}  ok")
    else:
        print(f"{exp_id:<30} {'--':>7}  {status[:40]}")
print("="*70)

valid = [(e, f1) for e, f1, ap, s in results_all if f1 is not None]
if valid:
    best_exp, best_f1 = max(valid, key=lambda x: x[1])
    print(f"\nOverall best: {best_exp} F1={best_f1:.4f}")
