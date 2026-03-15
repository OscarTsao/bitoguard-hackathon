# AWS SageMaker + Frontend Deployment Design
**Date:** 2026-03-15
**Project:** BitoGuard AML Detection System
**Goal:** Deploy full automated ML pipeline on SageMaker + frontend on AWS Amplify for hackathon demo

---

## 1. Overview

Deploy BitoGuard to AWS with:
- **AWS Amplify** serving the Next.js frontend (SSR + API routes supported natively)
- **ECS Fargate** running the FastAPI backend (:8001) with DuckDB on EFS
- **Step Functions** orchestrating the full pipeline: sync → features → SageMaker training → model registry → scoring
- **SageMaker Hyperparameter Tuning** for pre-demo model optimization; best params stored in SSM for demo-day fast runs
- **Pre-seeded data** (local DuckDB → S3 → EFS bootstrap) with live BitoPro API as primary source

This is a hackathon demo deployment. Correctness and impressiveness take priority over cost optimization.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────┐
│                       AWS                            │
│                                                      │
│  Amplify ──────► Next.js frontend (:3000)           │
│                       │                              │
│                 Next.js API routes                   │
│                       │                              │
│  ALB ──────► ECS Fargate: FastAPI backend (:8001)   │
│                    │         │                       │
│                   EFS      S3 (artifacts + features) │
│               (DuckDB)                               │
│                                                      │
│  POST /pipeline/run  ──►  Step Functions             │
│                              │                       │
│            ┌─────────────────┼──────────────┐        │
│            │                 │              │         │
│         DataSync     FeatureEngineering  Preprocessing│
│         (ECS task)   (ECS task → S3)    (SM Processing)│
│                                              │        │
│                              ┌───────────────┤        │
│                    tuning=true│              │tuning=false
│                              │              │        │
│                     HPO (SM Tuning)   [skip tuning]  │
│                     20 jobs parallel               │  │
│                              │              │        │
│                         TrainStacker (SM Training Job)│
│                              │                       │
│                     RegisterModel (Lambda)           │
│                              │                       │
│                          Scoring (ECS task)          │
│                              │                       │
│                       DriftDetection → Notify        │
│                                                      │
│  SageMaker Model Registry:                          │
│    lgbm-models, catboost-models, stacker-models     │
└─────────────────────────────────────────────────────┘
```

**Key architectural decisions:**
- DuckDB lives on EFS, shared via a single EFS access point (`/artifacts`) by all ECS tasks (backend + pipeline)
- SageMaker jobs cannot mount EFS directly; a SageMaker Processing Job (`preprocessing_entrypoint.py`) bridges DuckDB → S3 Parquet before training
- Step Functions `CheckTuningEnabled` state reads SSM param `/bitoguard/ml-pipeline/tuning_enabled` to branch between tuning and fast-path modes
- All AWS documentation referenced at implementation time per project requirement

---

## 3. Pipeline Modes

### Tuning Mode (pre-demo, run once)
Set SSM: `/bitoguard/ml-pipeline/tuning_enabled = true`

```
DataSync → FeatureEngineering → Preprocessing →
HyperparameterTuning (LightGBM + CatBoost, 20 jobs each, parallel) →
TrainStacker (best params) → RegisterModel → Scoring → DriftDetection → Notify
```

- SageMaker HPO uses Bayesian optimization
- Best hyperparameters written back to SSM by the `tuning_analyzer` Lambda
- Stacker trained on top of best LightGBM + CatBoost configs
- Estimated runtime: 2–3 hours, ~$50–80

### Demo Mode (demo day, fast)
Set SSM: `/bitoguard/ml-pipeline/tuning_enabled = false`

```
DataSync → FeatureEngineering → Preprocessing →
[CheckTuningEnabled → skip HPO] →
TrainStacker (params from SSM) → RegisterModel → Scoring → DriftDetection → Notify
```

- Reads best hyperparameters from SSM (written during tuning run)
- Estimated runtime: 15–25 minutes
- Triggered via `POST /pipeline/run` from the frontend UI

---

## 4. Infrastructure Fixes Required

Five targeted fixes to existing untracked code before deployment:

### F1 — Broken Import (BLOCKER)
**File:** `bitoguard_core/ml_pipeline/train_entrypoint.py` line 20
**Change:** `from models.train_catboost import train_catboost` → `from models.train_catboost import train_catboost_model as train_catboost`
**Why:** `train_catboost.py` exports `train_catboost_model`, not `train_catboost`. Causes immediate `ImportError` for any catboost or stacker job.

### F2 — EFS Mount Path Alignment (HIGH)
**File:** `infra/aws/terraform/ecs_ml_tasks.tf`
**Change:** All ML pipeline ECS tasks must use the same EFS access point as the backend (`aws_efs_access_point.artifacts`, root `/artifacts`, mounted at `/mnt/efs`). Add env vars: `BITOGUARD_DB_PATH=/mnt/efs/artifacts/bitoguard.duckdb`, `BITOGUARD_ARTIFACT_DIR=/mnt/efs/artifacts`.
**Why:** Backend mounts at `/mnt/efs/artifacts`, pipeline tasks were mounting at `/opt/ml/artifacts` on a different access point — they could never share DuckDB.
**Ref:** https://docs.aws.amazon.com/AmazonECS/latest/developerguide/efs-volumes.html

### F3 — S3 Feature Export Flag (MEDIUM)
**File:** `bitoguard_core/features/build_features_v2.py`
**Change:** Read `EXPORT_TO_S3` environment variable; pass `export_to_s3=True` to `build_and_store_v2_features()` when set.
**Why:** ECS task definition sets `EXPORT_TO_S3=true` but `build_v2()` never reads it — features were written to DuckDB only, breaking the SageMaker data chain.

### F4 — SageMaker Training Data Bridge (HIGH)
**File:** `bitoguard_core/ml_pipeline/train_entrypoint.py`
**Change:** Add `--use_s3_data` CLI flag. When set, load training DataFrame from Parquet files at `/opt/ml/input/data/training` instead of calling internal `training_dataset()` which requires DuckDB.
**Why:** SageMaker training jobs run on isolated EC2 instances with no EFS access. Training data must come from S3 input channel.
**Ref:** https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html

### F5 — Pipeline Trigger Endpoint (MEDIUM)
**File:** `bitoguard_core/api/main.py`
**Change:** Add `POST /pipeline/run` endpoint that calls `boto3` `stepfunctions.start_execution()` with the state machine ARN (read from env var `BITOGUARD_STEP_FUNCTIONS_ARN`). Returns execution ARN and console URL.
**Why:** No endpoint exists to trigger Step Functions from the frontend UI.
**Ref:** https://docs.aws.amazon.com/step-functions/latest/dg/tutorial-api-gateway.html

### A1 — Amplify Terraform Resource (NEW)
**File:** `infra/aws/terraform/amplify.tf` (new file)
**Change:** Add `aws_amplify_app`, `aws_amplify_branch` (main), and `aws_amplify_environment_variable` resources. Set `BITOGUARD_INTERNAL_API_BASE` to the ALB URL.
**Ref:** https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/amplify_app

### A2 — EFS Bootstrap (NEW)
**File:** `scripts/bootstrap-efs.sh` (new file)
**Change:** Script that uploads local `bitoguard.duckdb` to S3, then runs a one-shot ECS task that copies it from S3 to EFS if EFS is empty. Called once after `terraform apply`.
**Ref:** https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task_definitions.html

---

## 5. Deployment Sequence

```bash
# Step 1 — Bootstrap ECR (must exist before images can be pushed)
cd infra/aws/terraform
terraform apply -target=aws_ecr_repository.backend -target=aws_ecr_repository.training

# Step 2 — Build and push Docker images
cd ../../..
# Backend image
docker build -f bitoguard_core/Dockerfile -t bitoguard-backend:latest bitoguard_core/
docker tag bitoguard-backend:latest <account>.dkr.ecr.<region>.amazonaws.com/bitoguard-backend:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/bitoguard-backend:latest

# Training image
docker build -f bitoguard_core/Dockerfile.training -t bitoguard-training:latest bitoguard_core/
docker push <account>.dkr.ecr.<region>.amazonaws.com/bitoguard-backend:training

# Step 3 — Full Terraform apply
cd infra/aws/terraform
terraform apply

# Step 4 — Bootstrap EFS with pre-seeded data
aws s3 cp bitoguard_core/artifacts/bitoguard.duckdb s3://bitoguard-<env>-artifacts/seed/bitoguard.duckdb
./scripts/bootstrap-efs.sh

# Step 5 — Deploy frontend to Amplify
# Amplify picks up from Terraform output (app ID + branch)
# Set BITOGUARD_INTERNAL_API_BASE env var to ALB URL in Amplify console

# Step 6 — Pre-demo tuning run (~2-3 hours, run the night before)
aws ssm put-parameter \
  --name /bitoguard/ml-pipeline/tuning_enabled \
  --value true --type String --overwrite
aws stepfunctions start-execution \
  --state-machine-arn $(terraform output -raw step_functions_arn) \
  --name "pre-demo-tuning-$(date +%Y%m%d)"

# Step 7 — Demo day (fast mode)
aws ssm put-parameter \
  --name /bitoguard/ml-pipeline/tuning_enabled \
  --value false --type String --overwrite
# Trigger via UI: POST /pipeline/run
```

---

## 6. Acceptance Criteria

- [ ] `terraform apply` completes without errors
- [ ] Backend health check: `GET https://<alb-url>/healthz` returns 200
- [ ] Frontend loads at Amplify URL, connects to backend
- [ ] `POST /pipeline/run` returns execution ARN and triggers Step Functions
- [ ] Tuning run completes: best params visible in SSM and SageMaker Model Registry
- [ ] Demo-mode pipeline completes in < 30 minutes
- [ ] Trained stacker model appears as `Approved` in SageMaker Model Registry
- [ ] Alert list in frontend shows scored users after pipeline run
- [ ] `GET /metrics/drift` shows green drift health

---

## 7. AWS Documentation References

All implementation must reference official AWS docs. Key references:

| Component | Reference |
|-----------|-----------|
| ECS Fargate task definitions | https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task_definitions.html |
| EFS volumes in ECS | https://docs.aws.amazon.com/AmazonECS/latest/developerguide/efs-volumes.html |
| SageMaker Training containers | https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html |
| SageMaker Hyperparameter Tuning | https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html |
| SageMaker Model Registry | https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html |
| Step Functions state machines | https://docs.aws.amazon.com/step-functions/latest/dg/concepts-amazon-states-language.html |
| Step Functions + API Gateway | https://docs.aws.amazon.com/step-functions/latest/dg/tutorial-api-gateway.html |
| AWS Amplify Next.js SSR | https://docs.aws.amazon.com/amplify/latest/userguide/server-side-rendering-amplify.html |
| Amplify Terraform resource | https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/amplify_app |
| SageMaker Processing Jobs | https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html |
