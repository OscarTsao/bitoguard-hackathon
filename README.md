# BitoGuard — Exchange-Centric AML Risk Detection System

BitoGuard is a production-minded Anti-Money Laundering (AML) / fraud-risk detection system built over the BitoPro AWS-event data model. The repository now includes the main backend/frontend stack, competition-aligned official pipelines, a mock upstream API, and sample or simulated datasets for offline development.

## Repo Layout

- `bitoguard_core/`: main Python backend, training and scoring code, official experiment pipeline in `official/`, transductive research pipeline in `transductive_v1/`, and tracked outputs in `artifacts/`
- `bitoguard_frontend/`: Next.js App Router frontend for alerts, graph, and model operations
- `bitoguard_mock_api/`: read-only FastAPI mock of the upstream source API backed by `bitoguard_sim_output/`
- `bitoguard_sample_output/`, `bitoguard_sim_output/`, `bitoguard_simulator/`, `data/aws_event/`: sample, simulated, and organizer-supplied offline data assets
- `infra/aws/`, `deploy/`, `docs/`: Terraform, deployment scripts, and runbooks

## Architecture Overview

| Module | Description | Key Files |
|--------|-------------|-----------|
| M1: Rules | 11 deterministic AML rules, severity-weighted scoring | `bitoguard_core/models/rule_engine.py` |
| M2: Statistical | Peer-deviation features, cohort percentile ranks, rolling windows | `bitoguard_core/features/build_features.py` |
| M3: Supervised | CatBoost + LightGBM stacker, 5-fold OOF, AUC 0.9495 | `bitoguard_core/models/stacker.py`, `bitoguard_core/models/score.py` |
| M4: Anomaly | IsolationForest novelty detection, anomaly score + type | `bitoguard_core/models/anomaly.py` |
| M5: Graph | NetworkX heterogeneous graph (IP/wallet/user), blacklist proximity | `bitoguard_core/features/graph_features.py` |
| M6: Ops | SHAP case reports, incremental refresh, drift detection, AWS prep | `bitoguard_core/services/`, `pipeline/refresh_live.py` |

## Quick Start

```bash
# Backend setup and validation
make setup
make test

# Start local services in separate terminals
make serve
make frontend

# Core data/model pipeline
make sync && make features && make train && make score && make drift

# Or start the full stack with Docker
cp deploy/.env.compose.example .env
docker compose up --build
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| `bitoguard_core` | 8001 | FastAPI — pipeline, model, alerts, graph, metrics |
| `bitoguard_frontend` | 3000 | Next.js — alerts dashboard, model ops, graph explorer |
| `bitoguard_mock_api` | 8000 | Optional FastAPI mock of the upstream source API for offline testing |

### Frontend

```bash
cd bitoguard_frontend && npm ci && npm run dev
# Open http://localhost:3000
```

## Makefile Targets (run from the repo root)

```bash
make setup       # Create bitoguard_core/.venv and install backend deps
make test        # Run the backend pytest suite
make sync        # Sync live BitoPro data
make features    # Build feature snapshots + graph features
make features-v2 # Build v2 feature snapshots (~155 columns per user)
make train       # Train CatBoost + LightGBM stacker (v2 features)
make refresh     # Incremental refresh (watermark-bounded)
make score       # Score latest snapshot → alerts
make drift       # Feature distribution drift check
make serve       # Start backend API on :8001
make frontend    # Start Next.js app on :3000
make docker-build
make docker-up
```

## API Endpoints (bitoguard_core, port 8001)

| Endpoint | Description |
|----------|-------------|
| `GET /healthz` | Health check |
| `POST /pipeline/sync` | Trigger data sync |
| `POST /features/rebuild` | Rebuild feature snapshots |
| `POST /model/train` | Train + evaluate model |
| `POST /model/score` | Score latest snapshot |
| `GET /alerts` | List alerts (paginated) |
| `GET /alerts/{id}/report` | Risk diagnosis with SHAP + graph |
| `POST /alerts/{id}/decision` | Case decision |
| `GET /users/{id}/360` | User 360 view |
| `GET /users/{id}/graph` | Graph neighborhood (1-2 hops) |
| `GET /metrics/model` | Full validation report (P@K, calibration, FI) |
| `GET /metrics/threshold` | Threshold sensitivity table |
| `GET /metrics/drift` | Feature drift health (auto-refreshes 60s in UI) |

## Documentation

| Document | Location |
|----------|----------|
| Local runbook | `docs/RUNBOOK_LOCAL.md` |
| Rule book | `docs/RULEBOOK.md` |
| Model card | `docs/MODEL_CARD.md` |
| Data contract | `docs/DATA_CONTRACT.md` |
| Graph trust boundary | `docs/GRAPH_TRUST_BOUNDARY.md` |
| Graph recovery plan | `docs/GRAPH_RECOVERY_PLAN.md` |
| ML pipeline summary | `docs/ML_PIPELINE_SUMMARY.md` |
| SageMaker deployment guide | `docs/SAGEMAKER_DEPLOYMENT_GUIDE.md` |
| Latest official experiment summary | `OFFICIAL_EXPERIMENT_SUMMARY_20260317.md` |

## Validation

```
make test-quick
cd bitoguard_frontend && npm run lint && npm run build
cd bitoguard_mock_api && python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && pytest
```

## AWS Deployment

Complete production-ready deployment using AWS managed services (ECS Fargate, ALB, EFS, ECR).

### Quick Deploy

```bash
# 1. Deploy infrastructure (one-time)
cd infra/aws/terraform
terraform init
cp terraform.tfvars.example terraform.tfvars
terraform apply

# 2. Deploy application
cd ../../..
./scripts/deploy-aws.sh

# 3. Get URL
terraform output alb_url
```

### Documentation

- [Deployment Guide](docs/SAGEMAKER_DEPLOYMENT_GUIDE.md) - Complete AWS/SageMaker documentation
- [Architecture](infra/aws/ARCHITECTURE.md) - AWS architecture deep dive
- [Terraform Guide](infra/aws/README.md) - Infrastructure setup and module breakdown

### What's Included

- **Infrastructure-as-Code**: Terraform for all AWS resources
- **High Availability**: Multi-AZ deployment with auto-scaling
- **Monitoring**: CloudWatch logs, metrics, and alarms
- **CI/CD**: GitHub Actions workflows
- **Security**: Private subnets, security groups, encrypted storage
- **Cost**: ~$190/month (optimizable to $50-140)

See [infra/aws/README.md](infra/aws/README.md) and [docs/SAGEMAKER_DEPLOYMENT_GUIDE.md](docs/SAGEMAKER_DEPLOYMENT_GUIDE.md) for the current deployment documentation.
