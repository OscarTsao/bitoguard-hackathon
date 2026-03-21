# AWS Deployment Preparedness Assessment

**Audit date:** 2026-03-14
**Auditor:** Claude Opus 4.6

---

## Executive Summary

AWS deployment artifacts are **prepared but not executed**. All infrastructure scaffolding (task definitions, IAM policies, deployment scripts, CI pipeline) exists and is syntactically valid. Placeholder values remain in template files. No actual AWS deployment has been attempted.

**Verdict: DEPLOYMENT_PREPARED (not DEPLOYED)**

---

## Artifact Inventory

### Docker

| Artifact | Path | Status |
|----------|------|--------|
| Backend Dockerfile | `bitoguard_core/Dockerfile` | EXISTS - Python 3.12-slim, libgomp1, uvicorn CMD |
| Frontend Dockerfile | `bitoguard_frontend/Dockerfile` | EXISTS - referenced in compose.yaml |
| Docker Compose | `compose.yaml` | EXISTS - backend + frontend, health check, port binding |
| .dockerignore | `.dockerignore` | EXISTS |

### AWS Infrastructure

| Artifact | Path | Status |
|----------|------|--------|
| Backend task definition | `infra/aws/task-def-backend.json` | EXISTS - Fargate, 1 vCPU, 2GB, EFS mount |
| Frontend task definition | `infra/aws/task-def-frontend.json` | EXISTS - Fargate, 0.5 vCPU, 1GB |
| Refresh task definition | `infra/aws/task-def-refresh.json` | EXISTS - Fargate, 1 vCPU, 2GB, EFS mount |
| Task execution trust policy | `infra/aws/task-execution-trust.json` | EXISTS |
| Task trust policy | `infra/aws/task-trust.json` | EXISTS |
| Task IAM policy | `infra/aws/task-policy.json` | EXISTS |

### Deployment Scripts

| Script | Path | Status |
|--------|------|--------|
| Build and push to ECR | `scripts/build_and_push.sh` | EXISTS - parameterized (ACCOUNT_ID, REGION) |
| Full AWS deploy | `scripts/deploy_aws.sh` | EXISTS - build+push, register task defs, update services, wait for stability, health check |

### CI/CD

| Artifact | Path | Status |
|----------|------|--------|
| GitHub Actions workflow | `.github/workflows/ci.yml` | EXISTS - test + lint + docker build + manual deploy |

---

## Placeholder Analysis

The following placeholders remain in infrastructure files and must be replaced before deployment:

| File | Placeholder | Required Value |
|------|-------------|----------------|
| `task-def-backend.json` | `<ACCOUNT_ID>` | AWS account ID |
| `task-def-backend.json` | `<REGION>` | e.g., ap-northeast-1 |
| `task-def-backend.json` | `<EFS_FILE_SYSTEM_ID>` | EFS file system ID |
| `task-def-frontend.json` | `<ACCOUNT_ID>` | AWS account ID |
| `task-def-frontend.json` | `<REGION>` | e.g., ap-northeast-1 |
| `task-def-refresh.json` | `<ACCOUNT_ID>` | AWS account ID |
| `task-def-refresh.json` | `<REGION>` | e.g., ap-northeast-1 |
| `task-def-refresh.json` | `<EFS_FILE_SYSTEM_ID>` | EFS file system ID |

---

## Architecture Assessment

### Compute: ECS Fargate

- **Backend:** 1 vCPU, 2GB RAM - sufficient for DuckDB + LightGBM inference
- **Frontend:** 0.5 vCPU, 1GB RAM - sufficient for Next.js SSR
- **Refresh:** 1 vCPU, 2GB RAM - runs `pipeline/refresh_live.py` as a one-shot task

### Storage: EFS

- Backend and refresh tasks share an EFS volume at `/mnt/efs/`
- DuckDB file at `/mnt/efs/bitoguard.duckdb`
- Model artifacts at `/mnt/efs/artifacts/`
- **Concern:** DuckDB single-writer limitation means concurrent writes from backend API and refresh task could conflict. Acceptable at current scale since refresh runs every 15 minutes and completes in seconds.

### Networking

- Backend exposed on port 8001 (behind ALB)
- Frontend exposed on port 3000 (behind ALB)
- Both services use `awsvpc` networking (each gets its own ENI)
- Frontend connects to backend via ECS service discovery (`bitoguard-backend.bitoguard.local:8001`)

### Monitoring

- CloudWatch Logs configured for all three task definitions
- Log groups: `/ecs/bitoguard-backend`, `/ecs/bitoguard-frontend`, `/ecs/bitoguard-refresh`
- Health checks defined in task definitions (backend: urllib healthz, frontend: curl)

### Security

- IAM roles follow least-privilege pattern (separate execution and task roles)
- Transit encryption enabled for EFS
- No secrets in environment variables (all configuration is non-sensitive)
- **Gap:** No authentication on API endpoints (see PROD_GAP_LIST.md GAP-M3)
- **Gap:** AWS Secrets Manager not used (recommended in RUNBOOK_AWS.md but not enforced)

---

## Deployment Script Quality

### `scripts/build_and_push.sh`

- Uses `set -euo pipefail` (safe bash defaults)
- Parameterized with ACCOUNT_ID and REGION
- Authenticates to ECR, builds both images, tags and pushes
- Frontend build includes `BITOGUARD_INTERNAL_API_BASE` build arg
- **Quality: Good**

### `scripts/deploy_aws.sh`

- Uses `set -euo pipefail`
- Calls `build_and_push.sh` as first step
- Registers both task definitions
- Updates both ECS services with `--force-new-deployment`
- Waits for service stability with `aws ecs wait services-stable`
- Attempts health check via private IP
- **Quality: Good**

---

## CI/CD Pipeline Quality

### `.github/workflows/ci.yml`

- Triggers on push to main/develop and PRs to main
- 3 parallel jobs: test-core, lint-frontend, docker-build
- docker-build depends on test-core and lint-frontend (gates deployment on test success)
- deploy-aws requires manual trigger (`workflow_dispatch`) on main branch only
- Uses GitHub environment `production` for deploy (enables environment protection rules)
- AWS credentials stored in GitHub secrets (not in code)
- **Quality: Good**

---

## EventBridge Scheduler

The RUNBOOK_AWS.md documents a `rate(15 minutes)` EventBridge scheduler for `refresh_live`. The `task-def-refresh.json` defines the corresponding task. The scheduler itself would need to be created via AWS CLI (commands provided in RUNBOOK_AWS.md).

---

## Cost Estimate Verification

| Component | Claimed Cost | Assessment |
|-----------|-------------|------------|
| ECS Fargate backend (1 vCPU, 2GB, 24/7) | ~$36/mo | Reasonable for ap-northeast-1 |
| ECS Fargate frontend (0.5 vCPU, 1GB, 24/7) | ~$9/mo | Reasonable |
| ECS Fargate refresh (1 vCPU, 4GB, 2h/day) | ~$12/mo | Reasonable |
| EFS Standard (200MB) | ~$0.10/mo | Reasonable |
| ALB | ~$18/mo | Reasonable |
| ECR (2 images, ~500MB) | ~$0.10/mo | Reasonable |
| CloudWatch Logs (1GB/day) | ~$3/mo | Reasonable |
| **Total** | **~$78/mo** | **Plausible** |

---

## Pre-Deployment Checklist

Before executing the deployment:

- [ ] Replace all `<ACCOUNT_ID>`, `<REGION>`, `<EFS_FILE_SYSTEM_ID>` placeholders
- [ ] Create ECR repositories (`bitoguard-backend`, `bitoguard-frontend`)
- [ ] Create EFS file system with mount targets in desired subnets
- [ ] Create ECS cluster (`bitoguard`)
- [ ] Create IAM roles with provided trust/policy documents
- [ ] Create ALB with target groups for backend (8001) and frontend (3000)
- [ ] Create security groups for backend, frontend, and EFS
- [ ] Set `BITOGUARD_GRAPH_FEATURES_TRUSTED_ONLY=true` in task environment
- [ ] Set `BITOGUARD_CORS_ORIGINS` to actual frontend domain
- [ ] Consider adding API authentication before production exposure
- [ ] Run `scripts/deploy_aws.sh <ACCOUNT_ID> <REGION>`
- [ ] Verify health check: `GET /healthz` returns `{"status": "ok"}`
- [ ] Create EventBridge scheduler for refresh_live
- [ ] Run initial sync: `POST /pipeline/sync`
- [ ] Run initial feature build: `POST /features/rebuild`
- [ ] Run initial training: `POST /model/train`
- [ ] Verify scoring: `POST /model/score`
