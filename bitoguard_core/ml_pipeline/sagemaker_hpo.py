"""SageMaker HPO 入口 — 在 SageMaker 上跑 CatBoost Base A HPO。"""
from __future__ import annotations
import os, sys, shutil
from pathlib import Path

sys.path.insert(0, "/opt/ml/code")

os.environ["BITOGUARD_AWS_EVENT_RAW_DIR"] = "/opt/ml/input/data/raw"
os.environ["BITOGUARD_AWS_EVENT_CLEAN_DIR"] = "/opt/ml/work/clean"
os.environ["BITOGUARD_ARTIFACT_DIR"] = "/opt/ml/work/artifacts"
os.environ["SKIP_GNN"] = "1"
os.environ["BITOGUARD_USE_GPU"] = "0"

for d in ("/opt/ml/work/clean", "/opt/ml/work/artifacts", "/opt/ml/model"):
    Path(d).mkdir(parents=True, exist_ok=True)

# 資料前處理
import ml_pipeline.sagemaker_e15_train as m
m.SM_INPUT_RAW = Path("/opt/ml/input/data/raw")
m.SM_CLEAN_DIR = Path("/opt/ml/work/clean")
m._prepare_clean_tables()

# 跑 HPO
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--n-trials", type=int, default=50)
parser.add_argument("--l2-min", type=float, default=1.0)
parser.add_argument("--l2-max", type=float, default=60.0)
parser.add_argument("--output", type=str, default="hpo_catboost_sm.json")
args = parser.parse_args()

from official.hpo import run_hpo_study
result = run_hpo_study(
    n_trials=args.n_trials,
    l2_min=args.l2_min,
    l2_max=args.l2_max,
    output_filename=args.output,
)
print(f"Best F1={result['best_f1']:.4f}")

# 複製結果到 /opt/ml/model/
shutil.copytree("/opt/ml/work/artifacts", "/opt/ml/model/artifacts", dirs_exist_ok=True)
print("DONE")
