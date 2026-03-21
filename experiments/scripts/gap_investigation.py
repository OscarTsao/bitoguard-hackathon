"""Investigate the 0.043 gap between configurable_pipeline and pipeline.py (0.4218).
Tests: run_official_pipeline() directly vs configurable_pipeline DEFAULT_CONFIG.
"""
import os, sys, traceback
BASE_DIR = "/home/user/YuNing/bitoguard-hackathon"
CORE_DIR = f"{BASE_DIR}/bitoguard_core"
os.environ["BITOGUARD_AWS_EVENT_CLEAN_DIR"] = "data/aws_event/clean"
sys.path.insert(0, CORE_DIR)
os.chdir(CORE_DIR)

# Test 1: configurable_pipeline with community_features=False (clean baseline)
print("=== Test 1: configurable_pipeline bare (no community, no extras) ===")
from official.configurable_pipeline import run_configurable_pipeline, DEFAULT_CONFIG
cfg_bare = {**DEFAULT_CONFIG,
            "community_features": False,
            "sequence_features": False,
            "temporal_sequence_features": False,
            "flow_graph_edges": False}
try:
    r1 = run_configurable_pipeline(cfg_bare, experiment_id="gap_bare")
    print(f"[RESULT] gap_bare F1={r1['f1']:.4f}")
except Exception as e:
    print(f"FAILED: {e}\n{traceback.format_exc()[:400]}")

# Test 2: check label_free_feature_columns count
print("\n=== Test 2: Feature column audit ===")
try:
    from official.train import _load_dataset, _label_free_feature_columns
    from official.rules import evaluate_official_rules
    ds = _load_dataset("full")
    cols = _label_free_feature_columns(ds)
    print(f"[AUDIT] _load_dataset cols: {len(ds.columns)}, label_free_cols: {len(cols)}")
    print(f"[AUDIT] Sample excluded cols: {[c for c in ds.columns if c not in cols][:10]}")
except Exception as e:
    print(f"FAILED: {e}")
