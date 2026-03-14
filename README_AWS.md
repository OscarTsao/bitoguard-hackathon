# BitoGuard AWS Deployment

Complete AWS deployment using managed services (ECS Fargate, ALB, EFS, ECR).

## Quick Start

```bash
# 1. Deploy infrastructure
cd infra/aws/terraform
terraform init
cp terraform.tfvars.example terraform.tfvars
terraform apply

# 2. Deploy application
cd ../../..
./scripts/deploy-aws.sh

# 3. Get URL
cd infra/aws/terraform
terraform output alb_url
```

## Documentation

- [Quick Start Guide](docs/QUICK_START_AWS.md) - Get running in 30 minutes
- [Deployment Guide](docs/AWS_DEPLOYMENT_GUIDE.md) - Complete deployment documentation
- [Architecture](infra/aws/ARCHITECTURE.md) - AWS architecture overview
- [Cost Optimization](docs/COST_OPTIMIZATION.md) - Reduce costs by 30-75%

## Architecture

- **ECS Fargate**: Serverless containers (backend + frontend)
- **ALB**: Load balancing with health checks
- **EFS**: Persistent storage for DuckDB
- **ECR**: Container registry
- **CloudWatch**: Logging and monitoring
- **Auto Scaling**: CPU/memory based scaling

## Estimated Costs

- **Base**: ~$198/month
- **Optimized**: ~$140/month (30% savings)
- **Development**: ~$50/month (75% savings)

See [Cost Optimization Guide](docs/COST_OPTIMIZATION.md) for details.

## Operations

```bash
# Deploy updates
./scripts/deploy-aws.sh

# Check health
./scripts/check-deployment.sh

# View logs
aws logs tail /ecs/bitoguard-prod-backend --follow

# Rollback
./scripts/rollback-deployment.sh backend 5

# Scale
# Edit terraform.tfvars, then:
terraform apply

# Cleanup
terraform destroy
```

## CI/CD

GitHub Actions workflow included at `.github/workflows/deploy-aws.yml`

Configure secrets:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

## Support

- Infrastructure: `infra/aws/terraform/`
- Scripts: `scripts/`
- Docs: `docs/`
