import os, sys, traceback
BASE_DIR = "/home/user/YuNing/bitoguard-hackathon"
CORE_DIR = f"{BASE_DIR}/bitoguard_core"
os.environ["BITOGUARD_AWS_EVENT_CLEAN_DIR"] = "data/aws_event/clean"
sys.path.insert(0, CORE_DIR)
os.chdir(CORE_DIR)

from official.configurable_pipeline import run_configurable_pipeline, DEFAULT_CONFIG

print("=== diag_no_community: DEFAULT_CONFIG minus community_features ===")
cfg = {**DEFAULT_CONFIG,
       "community_features": False,
       "sequence_features": False,
       "temporal_sequence_features": False,
       "flow_graph_edges": False}
try:
    r = run_configurable_pipeline(cfg, experiment_id="diag_no_community")
    print(f"[RESULT] no_community baseline F1={r['f1']:.4f}")
except Exception as e:
    print(f"FAILED: {e}\n{traceback.format_exc()[:600]}")
