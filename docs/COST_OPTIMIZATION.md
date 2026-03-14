# BitoGuard AWS Cost Optimization Guide

## Current Cost Breakdown (us-west-2)

### Base Configuration
- Backend: 2 tasks × 1 vCPU, 2GB RAM
- Frontend: 2 tasks × 0.5 vCPU, 1GB RAM
- Running 24/7

| Service | Monthly Cost | Annual Cost |
|---------|--------------|-------------|
| ECS Fargate (Backend) | $60 | $720 |
| ECS Fargate (Frontend) | $30 | $360 |
| Application Load Balancer | $20 | $240 |
| NAT Gateway (2×) | $70 | $840 |
| EFS (10GB) | $3 | $36 |
| CloudWatch Logs (10GB) | $5 | $60 |
| Data Transfer | $10 | $120 |
| **Total** | **$198** | **$2,376** |

## Optimization Strategies

### 1. Use Single NAT Gateway
**Savings: $35/month ($420/year)**

Trade-off: Reduced high availability

```hcl
# In vpc.tf, change count from 2 to 1
resource "aws_nat_gateway" "main" {
  count = 1  # Changed from 2
  # ...
}
```

### 2. Use Fargate Spot
**Savings: ~30% on compute ($27/month, $324/year)**

Trade-off: Tasks may be interrupted

```hcl
# In ecs.tf
capacity_provider_strategy {
  capacity_provider = "FARGATE_SPOT"
  weight            = 70
  base              = 0
}
capacity_provider_strategy {
  capacity_provider = "FARGATE"
  weight            = 30
  base              = 1
}
```

### 3. Reduce Task Count
**Savings: $45/month ($540/year)**

For non-production or low-traffic environments:

```hcl
backend_desired_count  = 1
frontend_desired_count = 1
```

### 4. Right-Size Resources
**Savings: $15-30/month**

Monitor actual usage and reduce if over-provisioned:

```hcl
backend_cpu    = 512   # Down from 1024
backend_memory = 1024  # Down from 2048
```

### 5. Schedule Scaling
**Savings: ~50% during off-hours**

Scale down during nights/weekends using Lambda:

```python
# Lambda function to scale down at night
import boto3
ecs = boto3.client('ecs')

def lambda_handler(event, context):
    ecs.update_service(
        cluster='bitoguard-prod-cluster',
        service='bitoguard-prod-backend',
        desiredCount=1  # Scale to 1 at night
    )
```

### 6. Use S3 for Large Artifacts
**Savings: $2-3/month on EFS**

Store model files in S3 instead of EFS:

```python
# Use boto3 to load models from S3
import boto3
s3 = boto3.client('s3')
s3.download_file('bitoguard-artifacts', 'models/lgbm.pkl', '/tmp/model.pkl')
```

### 7. Optimize Logs Retention
**Savings: $2-3/month**

```hcl
resource "aws_cloudwatch_log_group" "backend" {
  retention_in_days = 3  # Down from 7
}
```

### 8. Use VPC Endpoints
**Savings: $5-10/month on data transfer**

Add VPC endpoints for S3, ECR, CloudWatch:

```hcl
resource "aws_vpc_endpoint" "s3" {
  vpc_id       = aws_vpc.main.id
  service_name = "com.amazonaws.${var.aws_region}.s3"
}
```

## Cost Optimization Tiers

### Tier 1: Development ($50/month)
- 1 NAT Gateway
- 1 Backend task (512 CPU, 1GB)
- 1 Frontend task (256 CPU, 512MB)
- Fargate Spot
- 1-day log retention

**Savings: 75% ($148/month)**

### Tier 2: Staging ($100/month)
- 1 NAT Gateway
- 1 Backend task (1024 CPU, 2GB)
- 1 Frontend task (512 CPU, 1GB)
- 70% Fargate Spot
- 3-day log retention

**Savings: 50% ($98/month)**

### Tier 3: Production Optimized ($140/month)
- 2 NAT Gateways (HA)
- 2 Backend tasks (1024 CPU, 2GB)
- 2 Frontend tasks (512 CPU, 1GB)
- 30% Fargate Spot
- 7-day log retention
- VPC Endpoints

**Savings: 30% ($58/month)**

## Monitoring Costs

### Set Up Cost Alerts

```bash
aws budgets create-budget \
  --account-id YOUR_ACCOUNT_ID \
  --budget file://budget.json \
  --notifications-with-subscribers file://notifications.json
```

budget.json:
```json
{
  "BudgetName": "BitoGuard-Monthly",
  "BudgetLimit": {
    "Amount": "200",
    "Unit": "USD"
  },
  "TimeUnit": "MONTHLY",
  "BudgetType": "COST"
}
```

### Use AWS Cost Explorer

```bash
# Get last month's costs by service
aws ce get-cost-and-usage \
  --time-period Start=2024-01-01,End=2024-01-31 \
  --granularity MONTHLY \
  --metrics BlendedCost \
  --group-by Type=DIMENSION,Key=SERVICE
```

## Implementation Plan

### Phase 1: Quick Wins (Week 1)
1. Reduce log retention to 3 days
2. Right-size tasks based on metrics
3. Set up cost alerts

**Expected savings: $10-15/month**

### Phase 2: Infrastructure (Week 2-3)
1. Implement VPC endpoints
2. Move artifacts to S3
3. Consider single NAT gateway for non-prod

**Expected savings: $40-50/month**

### Phase 3: Advanced (Month 2)
1. Implement Fargate Spot
2. Set up scheduled scaling
3. Optimize data transfer

**Expected savings: $50-70/month**

## Cost Monitoring Dashboard

Create a custom dashboard to track costs:

```bash
./scripts/setup-cost-monitoring.sh
```

## Best Practices

1. **Tag all resources** for cost allocation
2. **Review costs weekly** using Cost Explorer
3. **Set up billing alerts** at 50%, 80%, 100%
4. **Use Reserved Capacity** for predictable workloads (not applicable to Fargate)
5. **Delete unused resources** (old ECR images, snapshots)
6. **Monitor data transfer** costs
7. **Use CloudWatch Insights** to reduce log volume

## Conclusion

With optimizations, you can reduce costs from $198/month to:
- **Development**: $50/month (75% savings)
- **Production**: $140/month (30% savings)

Choose optimizations based on your availability and performance requirements.
