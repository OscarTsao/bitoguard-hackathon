"""
Final experiments — runs on Machine A (.5) after phase3_hpo.py completes.
Covers: B2 (HPO validate FULL), B3 (lag+lr), B4 (lag FULL),
        B5 (flow extras FAST), B6 (final FULL + secondary validation).
"""
import os, sys, time, traceback, json
from pathlib import Path

BASE_DIR = "/home/user/YuNing/bitoguard-hackathon"
CORE_DIR = f"{BASE_DIR}/bitoguard_core"
os.environ["BITOGUARD_AWS_EVENT_CLEAN_DIR"] = "data/aws_event/clean"
sys.path.insert(0, CORE_DIR)
os.chdir(CORE_DIR)

BEST_CONFIG = {
    "sequence_features": True,
    "temporal_sequence_features": True,
    "flow_graph_edges": True,
}
BEST_PLUS_LAG = {**BEST_CONFIG, "lag_features": True}

results = []

def set_fast(fast):
    if fast:
        os.environ["ABLATION_FAST"] = "1"
    else:
        os.environ.pop("ABLATION_FAST", None)

def run_exp(exp_id, config, notes, fast=False):
    set_fast(fast)
    mode = "FAST" if fast else "FULL"
    print(f"\n{'='*60}\n=== {exp_id} [{mode}] ===\nConfig extras: {[k for k,v in config.items() if v and k not in BEST_CONFIG]}")
    sys.stdout.flush()
    try:
        from official.configurable_pipeline import run_configurable_pipeline
        result = run_configurable_pipeline(config=config, experiment_id=exp_id)
        f1 = result["f1"]
        elapsed = result.get("elapsed_seconds", 0)
        print(f"[EXP] {exp_id}: F1={f1:.4f} | {notes}")
        sys.stdout.flush()
        results.append((exp_id, f1, elapsed, "ok"))
        return f1
    except Exception as e:
        print(f"FAILED: {e}\n{traceback.format_exc()[:800]}")
        sys.stdout.flush()
        results.append((exp_id, None, 0, f"FAILED: {e}"))
        return None

print("\n" + "="*70)
print("B2: Validate best config + HPO params (FULL params)")
print("="*70)
b2_f1 = run_exp("b2_hpo_validate_full", BEST_CONFIG.copy(),
                "Best config + HPO params FULL", fast=False)

print("\n" + "="*70)
print("B3: Best + lag + lr_stacker (FULL params)")
print("="*70)
b3_config = {**BEST_PLUS_LAG, "lr_stacker": True}
b3_f1 = run_exp("b3_lag_lr_full", b3_config,
                "Best+lag+lr_stacker FULL (tests NaN fix)", fast=False)

print("\n" + "="*70)
print("B4: Best + lag (FULL params)")
print("="*70)
b4_f1 = run_exp("b4_lag_full", BEST_PLUS_LAG.copy(),
                "Best+lag FULL params", fast=False)

print("\n" + "="*70)
print("B5: Flow extras ablation (FAST)")
print("="*70)
b5_config = {**BEST_PLUS_LAG, "directed_flow_edges": True, "profile_similarity_edges": True}
b5_f1 = run_exp("b5_lag_flow_all", b5_config,
                "Best+lag+all flow extras FAST", fast=True)

best_configs_seen = [
    (b4_f1 or 0.0, "b4", BEST_PLUS_LAG.copy()),
    (b2_f1 or 0.0, "b2", BEST_CONFIG.copy()),
]
best_score, best_name, best_cfg = max(best_configs_seen, key=lambda x: x[0])
print(f"\n[DECISION] Best config for final run: {best_name} (F1={best_score:.4f})")

print("\n" + "="*70)
print("B6: FINAL full validation (FULL params)")
print("="*70)
b6_f1 = run_exp("b6_final_full", best_cfg,
                f"FINAL best combined config ({best_name})", fast=False)

print("\n" + "="*70)
print("B6b: Secondary (group-aware) validation")
print("="*70)
try:
    from official.validate import run_secondary_validation
    sec_result = run_secondary_validation()
    print(f"[SECONDARY] F1={sec_result.get('f1', 0):.4f} AP={sec_result.get('ap', 0):.4f}")
    results.append(("b6b_secondary", sec_result.get("f1", 0), 0, "ok"))
except Exception as e:
    print(f"[SECONDARY] FAILED: {e}")
    results.append(("b6b_secondary", None, 0, f"FAILED: {e}"))

print("\n" + "="*70)
print("FINAL EXPERIMENT SUMMARY")
print("="*70)
for exp_id, f1, elapsed, status in results:
    f1_str = f"F1={f1:.4f}" if f1 is not None else "FAILED"
    print(f"  {exp_id:35s} {f1_str}  ({elapsed:.0f}s)  [{status[:20]}]")

best_f1 = max((f1 for _, f1, _, _ in results if f1 is not None), default=0.0)
print(f"\nBest F1 achieved: {best_f1:.4f}")
print("phase_final.py COMPLETE")
