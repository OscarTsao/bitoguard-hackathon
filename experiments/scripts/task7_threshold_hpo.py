import os, sys, traceback
BASE_DIR = "/home/user/YuNing/bitoguard-hackathon"
CORE_DIR = f"{BASE_DIR}/bitoguard_core"
os.environ["BITOGUARD_AWS_EVENT_CLEAN_DIR"] = "data/aws_event/clean"
sys.path.insert(0, CORE_DIR)
os.chdir(CORE_DIR)

from official.configurable_pipeline import run_configurable_pipeline, DEFAULT_CONFIG

BASE = {**DEFAULT_CONFIG, "community_features": False,
        "sequence_features": True, "temporal_sequence_features": True,
        "flow_graph_edges": False}

results = []

def run(exp_id, config, notes):
    print(f"\n{'='*60}\n=== {exp_id} ===")
    sys.stdout.flush()
    try:
        r = run_configurable_pipeline(config, experiment_id=exp_id)
        f1 = r["f1"]
        print(f"[RESULT] {exp_id}: F1={f1:.4f} | {notes}")
        sys.stdout.flush()
        results.append((exp_id, f1))
        return f1
    except Exception as e:
        print(f"FAILED: {e}\n{traceback.format_exc()[:800]}")
        results.append((exp_id, None))
        return None

run("t7_threshold_hpo", {**BASE, "threshold_hpo": True}, "clean_seq_temporal + threshold HPO")
run("t7_threshold_hpo_flow", {**BASE, "threshold_hpo": True, "flow_graph_edges": True}, "clean_seq_temporal+flow + threshold HPO")

print("\n" + "="*60 + "\nSUMMARY")
for exp_id, f1 in results:
    print(f"  {exp_id:35s} F1={f1:.4f}" if f1 else f"  {exp_id:35s} FAILED")
