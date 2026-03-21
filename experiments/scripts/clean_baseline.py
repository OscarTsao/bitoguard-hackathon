import os, sys, traceback
BASE_DIR = "/home/user/YuNing/bitoguard-hackathon"
CORE_DIR = f"{BASE_DIR}/bitoguard_core"
os.environ["BITOGUARD_AWS_EVENT_CLEAN_DIR"] = "data/aws_event/clean"
sys.path.insert(0, CORE_DIR)
os.chdir(CORE_DIR)

from official.configurable_pipeline import run_configurable_pipeline, DEFAULT_CONFIG

results = []

def run(exp_id, config, notes):
    print(f"\n{"="*60}\n=== {exp_id} ===")
    sys.stdout.flush()
    try:
        r = run_configurable_pipeline(config, experiment_id=exp_id)
        f1 = r["f1"]
        print(f"[RESULT] {exp_id}: F1={f1:.4f} | {notes}")
        sys.stdout.flush()
        results.append((exp_id, f1))
        return f1
    except Exception as e:
        print(f"FAILED: {e}\n{traceback.format_exc()[:600]}")
        results.append((exp_id, None))
        return None

# 1. Clean baseline (no community, no extras, no hpo poison)
run("clean_baseline", {**DEFAULT_CONFIG, "community_features": False,
    "sequence_features": False, "temporal_sequence_features": False,
    "flow_graph_edges": False}, "Clean baseline, no hpo_best_params")

# 2. + sequence_features
run("clean_seq", {**DEFAULT_CONFIG, "community_features": False,
    "sequence_features": True, "temporal_sequence_features": False,
    "flow_graph_edges": False}, "Clean + seq")

# 3. + temporal_sequence_features
run("clean_seq_temporal", {**DEFAULT_CONFIG, "community_features": False,
    "sequence_features": True, "temporal_sequence_features": True,
    "flow_graph_edges": False}, "Clean + seq + temporal")

# 4. + flow_graph_edges (best config on clean baseline)
run("clean_best", {**DEFAULT_CONFIG, "community_features": False,
    "sequence_features": True, "temporal_sequence_features": True,
    "flow_graph_edges": True}, "Clean + seq + temporal + flow")

print("\n" + "="*60)
print("SUMMARY")
for exp_id, f1 in results:
    print(f"  {exp_id:30s} F1={f1:.4f}" if f1 else f"  {exp_id:30s} FAILED")
