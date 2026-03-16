#!/bin/bash
# BitoGuard — SageMaker Stacker Training (Python 3.11, real data)
#
# Exports the v2 feature snapshot from local DuckDB to Parquet,
# uploads to S3, then launches a SageMaker training job using the
# PyTorch py311 container (supports Script Mode + auto-install via requirements.txt).
#
# Usage: ./scripts/launch-sagemaker-stacker.sh [n_folds]
#   n_folds defaults to 5
#
# Prerequisites:
#   - Valid AWS credentials in environment
#   - bitoguard_core/.venv activated or PYTHONPATH set
#   - DuckDB with populated features.feature_snapshots_v2

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Config ─────────────────────────────────────────────────────────────────────
N_FOLDS="${1:-5}"
AWS_REGION="${AWS_REGION:-us-west-2}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
BUCKET_NAME="bitoguard-ml-${ACCOUNT_ID}"
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/sagemaker-immersion-day-SageMakerExecutionRole-xSqhC3Ls9p0E"
# PyTorch 2.3 / Python 3.11 DLC — available in all commercial regions
TRAINING_IMAGE="763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com/pytorch-training:2.3.0-cpu-py311-ubuntu20.04-sagemaker"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DB_PATH="${REPO_ROOT}/bitoguard_core/artifacts/bitoguard.duckdb"

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}  BitoGuard — SageMaker Stacker Training${NC}"
echo -e "${BLUE}  Account: ${ACCOUNT_ID}  Region: ${AWS_REGION}${NC}"
echo -e "${BLUE}  Folds: ${N_FOLDS}  Bucket: ${BUCKET_NAME}${NC}"
echo -e "${BLUE}======================================================${NC}"

# ── Step 1: Export training data from DuckDB ───────────────────────────────────
log_info "[1/5] Exporting feature snapshot from DuckDB..."

PARQUET_PATH="/tmp/bitoguard_training_$(date +%Y%m%d).parquet"

python3 - <<EXPORT_EOF
import sys, duckdb, pandas as pd
db = "${DB_PATH}"
try:
    con = duckdb.connect(db, read_only=True)
except Exception as e:
    print(f"Cannot open DuckDB: {e}", file=sys.stderr)
    sys.exit(1)

df = con.execute("""
    WITH ped AS (
        SELECT user_id, CAST(MIN(observed_at) AS DATE) AS ped
        FROM canonical.blacklist_feed
        WHERE observed_at IS NOT NULL
        GROUP BY user_id
    )
    SELECT f.*,
           COALESCE(l.hidden_suspicious_label, 0) AS hidden_suspicious_label
    FROM features.feature_snapshots_v2 f
    LEFT JOIN ops.oracle_user_labels l ON f.user_id = l.user_id
    LEFT JOIN ped ON f.user_id = ped.user_id
    WHERE COALESCE(l.hidden_suspicious_label, 0) = 0
       OR (ped.ped IS NOT NULL AND f.snapshot_date >= ped.ped)
""").df()
df["hidden_suspicious_label"] = df["hidden_suspicious_label"].fillna(0).astype(int)
con.close()

df.to_parquet("${PARQUET_PATH}", index=False, engine="pyarrow")
pos = int(df["hidden_suspicious_label"].sum())
print(f"Exported {len(df):,} rows, {len(df.columns)} columns, {pos} positives ({pos/len(df):.2%})")
EXPORT_EOF

log_ok "Exported to ${PARQUET_PATH}"

# ── Step 2: Upload training data ───────────────────────────────────────────────
log_info "[2/5] Uploading training data to S3..."
S3_DATA_PREFIX="s3://${BUCKET_NAME}/training-data"
aws s3 cp "${PARQUET_PATH}" "${S3_DATA_PREFIX}/bitoguard_training.parquet"
log_ok "Uploaded to ${S3_DATA_PREFIX}"

# ── Step 3: Write training entrypoint ─────────────────────────────────────────
log_info "[3/5] Preparing training code..."
CODE_DIR="/tmp/bitoguard_sm_code"
rm -rf "${CODE_DIR}" && mkdir -p "${CODE_DIR}"

cat > "${CODE_DIR}/train.py" <<'TRAIN_EOF'
"""BitoGuard Stacker — SageMaker Script Mode entrypoint.

Reads Parquet from SM_CHANNEL_TRAINING, trains 5-fold OOF stacker
(CatBoost + LightGBM -> LogisticRegression), saves models to SM_MODEL_DIR.
"""
import os, json, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

NON_FEATURE_COLUMNS = frozenset({
    "feature_snapshot_id", "user_id", "snapshot_date",
    "feature_version", "hidden_suspicious_label",
})
CAT_FEATURE_NAMES = frozenset({
    "kyc_level_code", "occupation_code", "income_source_code", "user_source_code",
})

if __name__ == "__main__":
    data_dir  = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    n_folds   = int(os.environ.get("SM_HP_N_FOLDS", "5"))

    # Load all Parquet files in training channel
    files = list(Path(data_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {data_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"Loaded {len(df):,} rows from {[f.name for f in files]}")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    cat_indices  = [i for i, c in enumerate(feature_cols) if c in CAT_FEATURE_NAMES]

    X      = df[feature_cols].fillna(0)
    y      = df["hidden_suspicious_label"].astype(int).values
    groups = df["user_id"].values

    pos = int(y.sum())
    print(f"Features: {len(feature_cols)}, Positives: {pos:,} ({y.mean():.2%}), Folds: {n_folds}")

    oof_cb   = np.zeros(len(X))
    oof_lgbm = np.zeros(len(X))
    fold_metrics: list[dict] = []
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold_i, (tr_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups), 1):
        tr_pos = max(1, int(y[tr_idx].sum()))
        tr_neg = max(1, len(tr_idx) - tr_pos)
        spw    = tr_neg / tr_pos
        print(f"\n[Fold {fold_i}/{n_folds}] train={len(tr_idx):,} val={len(val_idx):,} spw={spw:.1f}")

        cb = CatBoostClassifier(
            iterations=300, learning_rate=0.05, depth=6,
            scale_pos_weight=spw, cat_features=cat_indices,
            random_seed=42, verbose=0,
        )
        cb.fit(X.iloc[tr_idx], y[tr_idx])
        oof_cb[val_idx] = cb.predict_proba(X.iloc[val_idx])[:, 1]
        cb_auc = roc_auc_score(y[val_idx], oof_cb[val_idx])
        cb_ap  = average_precision_score(y[val_idx], oof_cb[val_idx])
        print(f"  CatBoost  AUC={cb_auc:.4f}  PR-AUC={cb_ap:.4f}")

        lgbm = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=spw, random_state=42, n_jobs=-1,
        )
        lgbm.fit(X.iloc[tr_idx], y[tr_idx])
        oof_lgbm[val_idx] = lgbm.predict_proba(X.iloc[val_idx])[:, 1]
        lgbm_auc = roc_auc_score(y[val_idx], oof_lgbm[val_idx])
        lgbm_ap  = average_precision_score(y[val_idx], oof_lgbm[val_idx])
        print(f"  LightGBM  AUC={lgbm_auc:.4f}  PR-AUC={lgbm_ap:.4f}")

        fold_metrics.append({
            "fold": fold_i,
            "n_train": int(len(tr_idx)),
            "n_val": int(len(val_idx)),
            "catboost": {"auc": round(cb_auc, 4), "pr_auc": round(cb_ap, 4)},
            "lgbm":     {"auc": round(lgbm_auc, 4), "pr_auc": round(lgbm_ap, 4)},
        })

    oof_cb_auc   = roc_auc_score(y, oof_cb)
    oof_lgbm_auc = roc_auc_score(y, oof_lgbm)
    oof_cb_ap    = average_precision_score(y, oof_cb)
    oof_lgbm_ap  = average_precision_score(y, oof_lgbm)

    oof_mat = np.column_stack([oof_cb, oof_lgbm])
    meta    = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    meta.fit(oof_mat, y)
    stacker_preds = meta.predict_proba(oof_mat)[:, 1]
    stacker_auc   = roc_auc_score(y, stacker_preds)
    stacker_ap    = average_precision_score(y, stacker_preds)

    print(f"\n{'='*60}")
    print(f"OOF CatBoost  AUC={oof_cb_auc:.4f}  PR-AUC={oof_cb_ap:.4f}")
    print(f"OOF LightGBM  AUC={oof_lgbm_auc:.4f}  PR-AUC={oof_lgbm_ap:.4f}")
    print(f"OOF Stacker   AUC={stacker_auc:.4f}  PR-AUC={stacker_ap:.4f}")
    print(f"Meta coefs: {meta.coef_.tolist()}")
    print(f"{'='*60}")

    # Retrain on all data for the final deployed models
    pos_all = max(1, int(y.sum()))
    neg_all = max(1, len(y) - pos_all)
    spw_all = neg_all / pos_all

    final_cb = CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=6,
        scale_pos_weight=spw_all, cat_features=cat_indices,
        random_seed=42, verbose=0,
    )
    final_cb.fit(X, y)

    final_lgbm = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=spw_all, random_state=42, n_jobs=-1,
    )
    final_lgbm.fit(X, y)

    # Save artifacts
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(final_cb,   os.path.join(model_dir, "catboost.joblib"))
    joblib.dump(final_lgbm, os.path.join(model_dir, "lgbm.joblib"))
    joblib.dump(meta,       os.path.join(model_dir, "stacker_meta.joblib"))

    results = {
        "feature_columns": feature_cols,
        "n_folds": n_folds,
        "oof": {
            "catboost": {"auc": round(oof_cb_auc, 4), "pr_auc": round(oof_cb_ap, 4)},
            "lgbm":     {"auc": round(oof_lgbm_auc, 4), "pr_auc": round(oof_lgbm_ap, 4)},
            "stacker":  {"auc": round(stacker_auc, 4),  "pr_auc": round(stacker_ap, 4)},
        },
        "meta_coefs": meta.coef_.tolist(),
        "folds": fold_metrics,
    }
    with open(os.path.join(model_dir, "cv_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved models and cv_results.json to {model_dir}")
TRAIN_EOF

# requirements.txt: Python 3.11-compatible, SageMaker auto-installs this
cat > "${CODE_DIR}/requirements.txt" <<'REQ_EOF'
catboost>=1.2.7
lightgbm>=4.3.0
scikit-learn>=1.4.0
pandas>=2.1.0
numpy>=1.26.0
pyarrow>=14.0.0
joblib>=1.3.0
REQ_EOF

# Package code
cd /tmp
tar -czf bitoguard_code.tar.gz -C "${CODE_DIR}" .
CODE_S3="s3://${BUCKET_NAME}/code/bitoguard_code_$(date +%Y%m%d_%H%M%S).tar.gz"
aws s3 cp bitoguard_code.tar.gz "${CODE_S3}"
log_ok "Code uploaded to ${CODE_S3}"
cd "${REPO_ROOT}"

# ── Step 4: Launch SageMaker training job ──────────────────────────────────────
log_info "[4/5] Launching SageMaker training job..."
TRAINING_JOB_NAME="bitoguard-stacker-v2-$(date +%Y%m%d-%H%M%S)"

aws sagemaker create-training-job \
  --training-job-name "${TRAINING_JOB_NAME}" \
  --role-arn "${ROLE_ARN}" \
  --algorithm-specification \
    "TrainingImage=${TRAINING_IMAGE},TrainingInputMode=File" \
  --input-data-config "[
    {
      \"ChannelName\": \"training\",
      \"DataSource\": {
        \"S3DataSource\": {
          \"S3DataType\": \"S3Prefix\",
          \"S3Uri\": \"${S3_DATA_PREFIX}/\",
          \"S3DataDistributionType\": \"FullyReplicated\"
        }
      },
      \"ContentType\": \"application/x-parquet\"
    }
  ]" \
  --output-data-config "S3OutputPath=s3://${BUCKET_NAME}/models/" \
  --resource-config \
    "InstanceType=ml.c5.9xlarge,InstanceCount=1,VolumeSizeInGB=50" \
  --stopping-condition "MaxRuntimeInSeconds=7200" \
  --hyper-parameters \
    "sagemaker_program=train.py,sagemaker_submit_directory=${CODE_S3},n_folds=${N_FOLDS}" \
  --region "${AWS_REGION}"

# ── Step 5: Monitor ────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}======================================================${NC}"
echo -e "${GREEN}  ✓ Training job launched: ${TRAINING_JOB_NAME}${NC}"
echo -e "${GREEN}  Container: PyTorch 2.3 / Python 3.11${NC}"
echo -e "${GREEN}  Instance:  ml.c5.9xlarge (36 vCPUs)${NC}"
echo -e "${GREEN}  Folds:     ${N_FOLDS}${NC}"
echo -e "${GREEN}======================================================${NC}"
echo ""
echo "Monitor:"
echo "  aws sagemaker describe-training-job --training-job-name ${TRAINING_JOB_NAME} --region ${AWS_REGION} | jq .TrainingJobStatus"
echo "  aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix ${TRAINING_JOB_NAME} --follow"
echo ""
echo "Results (after completion):"
echo "  aws s3 ls s3://${BUCKET_NAME}/models/${TRAINING_JOB_NAME}/output/"
