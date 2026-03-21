"""Re-test lr_stacker with predict-time NaN fix (commit 9b18faf)."""
import os, sys, traceback
BASE_DIR = "/home/user/YuNing/bitoguard-hackathon"
CORE_DIR = f"{BASE_DIR}/bitoguard_core"
os.environ["BITOGUARD_AWS_EVENT_CLEAN_DIR"] = "data/aws_event/clean"
sys.path.insert(0, CORE_DIR)
os.chdir(CORE_DIR)

from official.configurable_pipeline import run_configurable_pipeline

BEST_CONFIG = {
    "sequence_features": True,
    "temporal_sequence_features": True,
    "flow_graph_edges": True,
}

print("=== lr_stacker retest (predict-time NaN fix) ===")
cfg = {**BEST_CONFIG, "lr_stacker": True}
try:
    r = run_configurable_pipeline(cfg, experiment_id="lr_stacker_retest")
    print(f"[RESULT] lr_stacker F1={r['f1']:.4f}")
except Exception as e:
    print(f"FAILED: {e}\n{traceback.format_exc()[:800]}")
