"""Phase 1: 8-grid fast ablation on DEFAULT_CONFIG baseline.

Run with ABLATION_FAST=1 for ~8 min/exp (screening).
Run without for ~20 min/exp (FULL params validation).
"""
import os, sys, time
os.environ.setdefault("BITOGUARD_AWS_EVENT_CLEAN_DIR", "data/aws_event/clean")
sys.path.insert(0, "/home/user/YuNing/bitoguard-hackathon/bitoguard_core")
os.chdir("/home/user/YuNing/bitoguard-hackathon/bitoguard_core")

from official.configurable_pipeline import run_configurable_pipeline, DEFAULT_CONFIG
from official.experiment_tracker import log_experiment

fast_mode = os.environ.get("ABLATION_FAST", "0") == "1"
print(f"Mode: {'ABLATION_FAST (screening)' if fast_mode else 'FULL PARAMS (validation)'}")

experiments = [
    ("p1_baseline",      {},                                                           "DEFAULT_CONFIG baseline"),
    ("p1_seq",           {"sequence_features": True},                                 "+sequence_features"),
    ("p1_temp",          {"temporal_sequence_features": True},                        "+temporal_sequence_features"),
    ("p1_seq_temp",      {"sequence_features": True, "temporal_sequence_features": True}, "+seq+temporal"),
    ("p1_flow",          {"flow_graph_edges": True},                                  "+flow_graph_edges"),
    ("p1_all_new",       {"sequence_features": True, "temporal_sequence_features": True, "flow_graph_edges": True}, "+seq+temporal+flow"),
    ("p1_lr",            {"lr_stacker": True},                                        "+lr_stacker"),
    ("p1_ewhpo",         {"edge_weight_hpo": True},                                   "+edge_weight_hpo"),
]

results = []
for exp_id, overrides, note in experiments:
    cfg = {**DEFAULT_CONFIG, **overrides}
    print(f"\n{'='*60}\n=== {exp_id}: {note} ===")
    t0 = time.time()
    try:
        r = run_configurable_pipeline(config=cfg, experiment_id=exp_id)
        elapsed = time.time() - t0
        log_experiment(exp_id, cfg, r, notes=f"{note} [ABLATION_FAST={fast_mode}]")
        f1 = r.get("f1", 0.0); ap = r.get("pr_auc", 0.0)
        print(f"Result: F1={f1:.4f} AP={ap:.4f} ({elapsed:.0f}s)")
        results.append((exp_id, f1, ap, elapsed))
    except Exception as e:
        elapsed = time.time() - t0
        print(f"FAILED ({elapsed:.0f}s): {e}")
        import traceback; traceback.print_exc()
        results.append((exp_id, 0.0, 0.0, elapsed))

print(f"\n{'='*60}\n=== PHASE 1 SUMMARY ===")
baseline_f1 = next((f1 for eid, f1, _, _ in results if eid == "p1_baseline"), 0.0)
for exp_id, f1, ap, elapsed in results:
    delta = f1 - baseline_f1
    marker = " ✓ POSITIVE" if delta >= 0.005 else (" ~ neutral" if delta >= -0.002 else " ✗ harmful")
    print(f"  {exp_id:<22} F1={f1:.4f} AP={ap:.4f} Δ={delta:+.4f}{marker} ({elapsed:.0f}s)")
