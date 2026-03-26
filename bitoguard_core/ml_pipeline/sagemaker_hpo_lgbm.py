"""SageMaker HPO 入口 — LightGBM Base D HPO（CPU only）。"""
from __future__ import annotations
import os, sys, json, time, logging
from pathlib import Path
import numpy as np

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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--n-trials", type=int, default=50)
args = parser.parse_args()

import optuna
import pandas as pd
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from official.train import _load_dataset, _label_free_feature_columns
from official.transductive_validation import PrimarySplitSpec, build_primary_transductive_splits, iter_fold_assignments
from official.common import RANDOM_SEED, encode_frame, save_json, load_official_paths
from hardware import lightgbm_runtime_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

logger.info("Loading dataset...")
dataset = _load_dataset("full")
primary_split = build_primary_transductive_splits(dataset, cutoff_tag="full", spec=PrimarySplitSpec(), write_outputs=False)
feature_cols = _label_free_feature_columns(dataset)
assignments = list(iter_fold_assignments(primary_split, "primary_fold"))

# Pre-encode folds
logger.info("Pre-encoding folds...")
fold_data = []
for fold_id, train_users, valid_users in assignments:
    tr = dataset[dataset["user_id"].astype(int).isin(train_users)].copy()
    va = dataset[dataset["user_id"].astype(int).isin(valid_users)].copy()
    tx, ec = encode_frame(tr, feature_cols)
    vx, _ = encode_frame(va, feature_cols, reference_columns=ec)
    yt = tr["status"].astype(int).values
    yv = va["status"].astype(int).values
    pos = max(1, int(yt.sum())); neg = max(1, len(yt) - pos)
    fold_data.append({"tx": tx, "vx": vx, "yt": yt, "yv": yv, "spw": float(neg)/pos})
logger.info(f"Ready: {len(fold_data)} folds, {tx.shape[1]} encoded features")

results = []

def objective(trial):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }
    runtime = lightgbm_runtime_params()
    t0 = time.time()
    op, ol = [], []
    for fd in fold_data:
        model = LGBMClassifier(
            **p, scale_pos_weight=fd["spw"], random_state=RANDOM_SEED, verbosity=-1, **runtime,
        )
        model.fit(fd["tx"], fd["yt"], eval_set=[(fd["vx"], fd["yv"])],
                  callbacks=[lgb_early_stop(100)])
        op.extend(model.predict_proba(fd["vx"])[:, 1].tolist())
        ol.extend(fd["yv"].tolist())
    op = np.array(op); ol = np.array(ol)
    bf, bt = 0, 0.1
    for t in np.arange(0.05, 0.50, 0.01):
        f = float(f1_score(ol, (op >= t).astype(int), zero_division=0))
        if f > bf: bf = f; bt = t
    el = time.time() - t0
    results.append({"trial": trial.number, "f1": bf, "threshold": bt, "elapsed_s": round(el, 1), **p})
    br = max(results, key=lambda r: r["f1"])
    save_json({
        "best_params": {k: v for k, v in br.items() if k not in ("trial", "f1", "threshold", "elapsed_s")},
        "best_f1": br["f1"], "n_trials": len(results),
        "trial_results": sorted(results, key=lambda x: -x["f1"]),
    }, load_official_paths().feature_dir / "hpo_lgbm_e15.json")
    logger.info(f"Trial {trial.number}: F1={bf:.4f} thr={bt:.2f} leaves={p['num_leaves']} lr={p['learning_rate']:.4f} ({el:.0f}s)")
    return bf

# LightGBM early stopping callback
from lightgbm import early_stopping as lgb_early_stop

sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED, n_startup_trials=10)
study = optuna.create_study(direction="maximize", sampler=sampler)
# Enqueue current defaults as baseline
study.enqueue_trial({
    "n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31,
    "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 1.0,
    "reg_alpha": 0.001, "reg_lambda": 0.001, "min_child_samples": 20,
})
optuna.logging.set_verbosity(optuna.logging.WARNING)
study.optimize(objective, n_trials=args.n_trials)
logger.info(f"Best F1={study.best_value:.4f}, params={study.best_params}")

import shutil
shutil.copytree("/opt/ml/work/artifacts", "/opt/ml/model/artifacts", dirs_exist_ok=True)
print("DONE")
