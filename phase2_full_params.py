"""
Phase 2: Validate winning Phase 1 components with FULL params (no ABLATION_FAST).
Goal: confirm ΔF1 ≥ +0.005 holds at full iterations/depth/seeds.

Experiments:
  p2_baseline        — DEFAULT_CONFIG, no extras (FULL params baseline)
  p2_seq             — +sequence_features
  p2_seq_temporal    — +sequence_features +temporal_sequence_features  (best in V7)
  p2_all_new         — +seq +temporal +flow_graph_edges
  p2_lr_stacker      — +lr_stacker (NaN fix applied)
  p2_seq_lr          — +seq +lr_stacker

Phase 1 winners (ABLATION_FAST baseline=0.3736):
  p1_seq        F1=0.3797 (+0.0061) POSITIVE
  p1_seq_temp   F1=0.3801 (+0.0065) POSITIVE
  p1_all_new    F1=0.3805 (+0.0069) POSITIVE
  V7 seq_temporal F1=0.3852 (+0.0122) STRONG (different baseline 0.3730)
"""
import os
import sys
import traceback

# FULL params — do NOT set ABLATION_FAST
os.environ.pop("ABLATION_FAST", None)
os.environ["BITOGUARD_AWS_EVENT_CLEAN_DIR"] = "data/aws_event/clean"

sys.path.insert(0, "/home/user/YuNing/bitoguard-hackathon/bitoguard_core")
os.chdir("/home/user/YuNing/bitoguard-hackathon/bitoguard_core")

from official.configurable_pipeline import run_configurable_pipeline
from official.experiment_tracker import log_experiment

EXPERIMENTS = [
    ("p2_baseline",     {"sequence_features": False, "temporal_sequence_features": False,
                         "flow_graph_edges": False, "lr_stacker": False},
     "FULL params DEFAULT_CONFIG baseline"),
    ("p2_seq",          {"sequence_features": True},
     "+sequence_features FULL params"),
    ("p2_seq_temporal", {"sequence_features": True, "temporal_sequence_features": True},
     "+seq+temporal FULL params (best V7: 0.3852)"),
    ("p2_all_new",      {"sequence_features": True, "temporal_sequence_features": True,
                         "flow_graph_edges": True},
     "+seq+temporal+flow FULL params"),
    ("p2_lr_stacker",   {"lr_stacker": True},
     "+lr_stacker FULL params (NaN fix)"),
    ("p2_seq_lr",       {"sequence_features": True, "lr_stacker": True},
     "+seq+lr_stacker FULL params"),
]

results = []
baseline_f1 = None

print("=" * 70)
print("PHASE 2: FULL PARAMS VALIDATION")
print("=" * 70)

for exp_id, config, notes in EXPERIMENTS:
    print("\n" + "=" * 60)
    print(f"=== {exp_id} ===")
    print(f"Config: {[k for k, v in config.items() if v]}")
    try:
        result = run_configurable_pipeline(config=config, experiment_id=exp_id)
        f1 = result["f1"]
        ap = result.get("ap", result.get("average_precision", 0.0))
        elapsed = result.get("elapsed", 0)
        log_experiment(exp_id, config, result, notes=notes)
        if exp_id == "p2_baseline":
            baseline_f1 = f1
        delta = f1 - baseline_f1 if baseline_f1 is not None else 0.0
        verdict = "POSITIVE" if delta >= 0.005 else ("HARMFUL" if delta < -0.002 else "neutral")
        print(f"Result: F1={f1:.4f} AP={ap:.4f} ({elapsed:.0f}s) delta={delta:+.4f} {verdict}")
        results.append((exp_id, f1, ap, delta, "ok"))
    except Exception as e:
        tb = traceback.format_exc()
        print(f"FAILED ({type(e).__name__}): {e}")
        print(tb[:800])
        results.append((exp_id, None, None, None, f"FAILED: {e}"))

print("\n" + "=" * 70)
print("=== PHASE 2 SUMMARY ===")
print(f"{'Experiment':<25} {'F1':>7} {'AP':>7} {'delta':>8}  Verdict")
print("-" * 70)
for exp_id, f1, ap, delta, status in results:
    if f1 is not None:
        verdict = "STRONG" if delta >= 0.010 else ("POSITIVE" if delta >= 0.005 else ("HARMFUL" if delta < -0.002 else "neutral"))
        print(f"{exp_id:<25} {f1:>7.4f} {ap:>7.4f} {delta:>+8.4f}  {verdict}")
    else:
        print(f"{exp_id:<25} {'--':>7} {'--':>7} {'--':>8}  {status}")
print("=" * 70)

positive = [(e, f1, delta) for e, f1, ap, delta, s in results if delta is not None and delta >= 0.005]
if positive:
    best = max(positive, key=lambda x: x[1])
    print(f"\nBest Phase 2 config: {best[0]} (F1={best[1]:.4f}, delta={best[2]:+.4f})")
    print("=> Proceed to Phase 3 HPO with this config as base")
else:
    print("\nNo config cleared delta>=+0.005 with FULL params.")
    print("=> Re-evaluate Phase 1 results; consider wider search")
