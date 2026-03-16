#!/bin/bash

# BitoGuard - 5-Fold CV Training on SageMaker
# Uses SageMaker Script Mode with BitoGuard stacker code

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

AWS_REGION="${AWS_REGION:-us-west-2}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
BUCKET_NAME="bitoguard-ml-${ACCOUNT_ID}"
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/sagemaker-immersion-day-SageMakerExecutionRole-xSqhC3Ls9p0E"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}BitoGuard - 5-Fold CV on SageMaker${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Prepare training code
echo -e "${YELLOW}[1/5] Preparing training code...${NC}"

# Create a training script that wraps the stacker
cat > /tmp/train_5fold.py <<'PYTHON_EOF'
import os
import sys
import json
import pickle
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, precision_score
import lightgbm as lgb
from catboost import CatBoostClassifier

def train_5fold_cv(X, y, groups, n_folds=5):
    """
    5-Fold Cross-Validation with StratifiedGroupKFold
    Returns out-of-fold predictions and trained models
    """
    print(f"Starting {n_folds}-fold cross-validation...")
    print(f"Data shape: {X.shape}")
    print(f"Positive rate: {y.mean():.4f}")
    
    # Initialize
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds_lgbm = np.zeros(len(X))
    oof_preds_cat = np.zeros(len(X))
    
    models_lgbm = []
    models_cat = []
    
    # 5-Fold CV
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y, groups), 1):
        print(f"\n{'='*50}")
        print(f"Fold {fold}/{n_folds}")
        print(f"{'='*50}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
        
        # Train LightGBM
        print("\nTraining LightGBM...")
        lgbm_model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
        lgbm_model.fit(X_train, y_train)
        oof_preds_lgbm[val_idx] = lgbm_model.predict_proba(X_val)[:, 1]
        models_lgbm.append(lgbm_model)
        
        # Train CatBoost
        print("Training CatBoost...")
        cat_model = CatBoostClassifier(
            iterations=200,
            learning_rate=0.05,
            depth=6,
            random_state=42,
            verbose=False
        )
        cat_model.fit(X_train, y_train)
        oof_preds_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
        models_cat.append(cat_model)
        
        # Fold metrics
        auc_lgbm = roc_auc_score(y_val, oof_preds_lgbm[val_idx])
        auc_cat = roc_auc_score(y_val, oof_preds_cat[val_idx])
        print(f"\nFold {fold} Results:")
        print(f"  LightGBM AUC: {auc_lgbm:.4f}")
        print(f"  CatBoost AUC: {auc_cat:.4f}")
    
    # Meta-learner (Logistic Regression)
    print(f"\n{'='*50}")
    print("Training Meta-Learner (Logistic Regression)")
    print(f"{'='*50}")
    
    meta_features = np.column_stack([oof_preds_lgbm, oof_preds_cat])
    meta_model = LogisticRegression(random_state=42, max_iter=1000)
    meta_model.fit(meta_features, y)
    
    final_preds = meta_model.predict_proba(meta_features)[:, 1]
    
    # Final metrics
    final_auc = roc_auc_score(y, final_preds)
    final_logloss = log_loss(y, final_preds)
    
    # Precision@100
    top_100_idx = np.argsort(final_preds)[-100:]
    precision_at_100 = y.iloc[top_100_idx].mean()
    
    print(f"\n{'='*50}")
    print("Final 5-Fold CV Results")
    print(f"{'='*50}")
    print(f"AUC: {final_auc:.4f}")
    print(f"Log Loss: {final_logloss:.4f}")
    print(f"Precision@100: {precision_at_100:.4f}")
    
    return {
        'models_lgbm': models_lgbm,
        'models_cat': models_cat,
        'meta_model': meta_model,
        'metrics': {
            'auc': final_auc,
            'logloss': final_logloss,
            'precision_at_100': precision_at_100
        }
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    args = parser.parse_args()
    
    print("Loading training data...")
    train_file = os.path.join(args.train, 'train.csv')
    
    if not os.path.exists(train_file):
        print(f"ERROR: Training file not found: {train_file}")
        print("Creating synthetic data for demonstration...")
        
        # Create synthetic data
        np.random.seed(42)
        n_samples = 1000
        n_features = 50
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.binomial(1, 0.1, n_samples))
        groups = pd.Series(np.random.randint(0, 200, n_samples))
        
        print(f"Synthetic data created: {X.shape}")
    else:
        df = pd.read_csv(train_file)
        print(f"Data loaded: {df.shape}")
        
        # Assume last column is target, second-to-last is user_id
        y = df.iloc[:, -1]
        groups = df.iloc[:, -2] if df.shape[1] > 2 else pd.Series(range(len(df)))
        X = df.iloc[:, :-2] if df.shape[1] > 2 else df.iloc[:, :-1]
    
    # Train 5-fold CV
    result = train_5fold_cv(X, y, groups, n_folds=args.n_folds)
    
    # Save models
    print(f"\nSaving models to {args.model_dir}...")
    os.makedirs(args.model_dir, exist_ok=True)
    
    with open(os.path.join(args.model_dir, 'stacker_5fold.pkl'), 'wb') as f:
        pickle.dump(result, f)
    
    # Save metrics for SageMaker
    with open(os.path.join(args.model_dir, 'metrics.json'), 'w') as f:
        json.dump(result['metrics'], f)
    
    print("\n✓ Training complete!")
    print(f"Models saved to: {args.model_dir}")
PYTHON_EOF

echo -e "${GREEN}✓ Training script created${NC}"

# Step 2: Upload code to S3
echo -e "${YELLOW}[2/5] Uploading code to S3...${NC}"
aws s3 cp /tmp/train_5fold.py s3://${BUCKET_NAME}/code/train_5fold.py
echo -e "${GREEN}✓ Code uploaded${NC}"

# Step 3: Create sample training data (you should replace this with real data)
echo -e "${YELLOW}[3/5] Preparing training data...${NC}"
echo "NOTE: Using synthetic data for demonstration."
echo "Replace with real BitoGuard feature data for production."

# Create a minimal CSV for testing
cat > /tmp/train.csv <<'CSV_EOF'
feature_0,feature_1,feature_2,user_id,label
0.5,1.2,0.3,1,0
-0.3,0.8,1.1,1,0
1.2,-0.5,0.7,2,1
0.1,0.9,-0.2,2,0
CSV_EOF

aws s3 cp /tmp/train.csv s3://${BUCKET_NAME}/data/train.csv
echo -e "${GREEN}✓ Training data uploaded${NC}"

# Step 4: Launch SageMaker training job
echo -e "${YELLOW}[4/5] Launching SageMaker training job...${NC}"

TRAINING_JOB_NAME="bitoguard-5fold-$(date +%Y%m%d-%H%M%S)"

cat > /tmp/training-job.json <<EOF
{
  "TrainingJobName": "${TRAINING_JOB_NAME}",
  "RoleArn": "${ROLE_ARN}",
  "AlgorithmSpecification": {
    "TrainingImage": "246618743249.dkr.ecr.${AWS_REGION}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
    "TrainingInputMode": "File"
  },
  "InputDataConfig": [
    {
      "ChannelName": "training",
      "DataSource": {
        "S3DataSource": {
          "S3DataType": "S3Prefix",
          "S3Uri": "s3://${BUCKET_NAME}/data/",
          "S3DataDistributionType": "FullyReplicated"
        }
      }
    }
  ],
  "OutputDataConfig": {
    "S3OutputPath": "s3://${BUCKET_NAME}/models/"
  },
  "ResourceConfig": {
    "InstanceType": "ml.c5.9xlarge",
    "InstanceCount": 1,
    "VolumeSizeInGB": 50
  },
  "StoppingCondition": {
    "MaxRuntimeInSeconds": 3600
  },
  "HyperParameters": {
    "sagemaker_program": "train_5fold.py",
    "sagemaker_submit_directory": "s3://${BUCKET_NAME}/code/",
    "n_folds": "5"
  }
}
EOF

aws sagemaker create-training-job --cli-input-json file:///tmp/training-job.json

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ 5-Fold CV Training Job Started${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Job Name: ${TRAINING_JOB_NAME}"
echo "Instance: ml.c5.9xlarge (36 vCPUs)"
echo "Folds: 5"
echo ""
echo "Monitor progress:"
echo "  aws sagemaker describe-training-job --training-job-name ${TRAINING_JOB_NAME}"
echo ""
echo "AWS Console:"
echo "  https://console.aws.amazon.com/sagemaker/home?region=${AWS_REGION}#/jobs/${TRAINING_JOB_NAME}"
echo ""

# Step 5: Monitor training
echo -e "${YELLOW}[5/5] Monitoring training job...${NC}"
echo ""

while true; do
    STATUS=$(aws sagemaker describe-training-job \
        --training-job-name ${TRAINING_JOB_NAME} \
        --query 'TrainingJobStatus' \
        --output text 2>/dev/null || echo "UNKNOWN")
    
    echo "[$(date +%H:%M:%S)] Status: ${STATUS}"
    
    if [ "$STATUS" = "Completed" ]; then
        echo ""
        echo -e "${GREEN}✓ 5-Fold CV Training Completed!${NC}"
        echo ""
        
        # Download and show metrics
        MODEL_ARTIFACTS=$(aws sagemaker describe-training-job \
            --training-job-name ${TRAINING_JOB_NAME} \
            --query 'ModelArtifacts.S3ModelArtifacts' \
            --output text)
        
        echo "Model artifacts: ${MODEL_ARTIFACTS}"
        echo ""
        echo "Download models:"
        echo "  aws s3 cp ${MODEL_ARTIFACTS} ./model.tar.gz"
        echo "  tar -xzf model.tar.gz"
        echo ""
        
        break
    elif [ "$STATUS" = "Failed" ] || [ "$STATUS" = "Stopped" ]; then
        echo ""
        echo -e "${RED}✗ Training job ${STATUS}${NC}"
        
        aws sagemaker describe-training-job \
            --training-job-name ${TRAINING_JOB_NAME} \
            --query 'FailureReason' \
            --output text
        
        exit 1
    fi
    
    sleep 30
done
