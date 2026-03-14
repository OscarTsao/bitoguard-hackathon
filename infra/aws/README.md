# BitoGuard AWS Deployment

This directory contains infrastructure-as-code for deploying BitoGuard on AWS using Terraform and managed services.

## Architecture

The deployment uses the following AWS services:

- **ECS Fargate**: Serverless container orchestration for backend and frontend
- **ECR**: Container registry for Docker images
- **ALB**: Application Load Balancer for traffic distribution
- **EFS**: Elastic File System for persistent DuckDB storage and artifacts
- **VPC**: Isolated network with public/private subnets across 2 AZs
- **CloudWatch**: Logging and monitoring
- **Secrets Manager**: Secure API key storage
- **Auto Scaling**: Automatic scaling based on CPU/memory metrics

## Prerequisites

1. AWS CLI configured with appropriate credentials
2. Terraform >= 1.0
3. Docker
4. Sufficient AWS permissions to create resources

## Quick Start

### 1. Initialize Infrastructure

```bash
# Initialize Terraform
cd infra/aws/terraform
terraform init

# Copy and edit configuration
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your settings

# Plan infrastructure
terraform plan

# Apply infrastructure
terraform apply
```

### 2. Build and Deploy Application

```bash
# From project root
./scripts/deploy-aws.sh
```

### 3. Access Application

After deployment, get the ALB DNS name:

```bash
terraform output alb_url
```

Visit the URL to access the BitoGuard frontend.

## Configuration

### terraform.tfvars

Key variables to configure:

```hcl
aws_region              = "us-west-2"
environment             = "prod"
backend_cpu             = 1024      # CPU units (1024 = 1 vCPU)
backend_memory          = 2048      # Memory in MB
frontend_cpu            = 512
frontend_memory         = 1024
backend_desired_count   = 2         # Number of backend tasks
frontend_desired_count  = 2         # Number of frontend tasks
bitoguard_source_url    = "https://aws-event-api.bitopro.com"
```

### Environment Variables

The deployment automatically configures:

**Backend:**
- `BITOGUARD_SOURCE_URL`: BitoPro API endpoint
- `BITOGUARD_DB_PATH`: DuckDB path on EFS
- `BITOGUARD_ARTIFACT_DIR`: Artifacts directory on EFS
- `BITOGUARD_API_KEY`: Auto-generated API key from Secrets Manager

**Frontend:**
- `BITOGUARD_INTERNAL_API_BASE`: Backend API URL
- `BITOGUARD_INTERNAL_API_KEY`: Same API key as backend

## Deployment Process

### Initial Deployment

1. **Provision Infrastructure** (one-time):
   ```bash
   cd infra/aws/terraform
   terraform apply
   ```

2. **Build and Push Images**:
   ```bash
   ./scripts/deploy-aws.sh
   ```

### Updates

For application updates, just run:
```bash
./scripts/deploy-aws.sh
```

This will:
- Build new Docker images
- Push to ECR
- Force ECS service redeployment with zero downtime

## Monitoring

### CloudWatch Logs

View logs:
```bash
# Backend logs
aws logs tail /ecs/bitoguard-prod-backend --follow

# Frontend logs
aws logs tail /ecs/bitoguard-prod-frontend --follow
```

### Metrics

CloudWatch alarms are configured for:
- Backend CPU > 80%
- Backend Memory > 80%

Auto-scaling triggers at 70% CPU/memory utilization.

### ECS Service Status

```bash
aws ecs describe-services \
  --cluster bitoguard-prod-cluster \
  --services bitoguard-prod-backend bitoguard-prod-frontend
```

## Scaling

### Manual Scaling

Update desired count in terraform.tfvars and apply:
```hcl
backend_desired_count  = 4
frontend_desired_count = 4
```

```bash
terraform apply
```

### Auto Scaling

Auto-scaling is configured automatically:
- Min: desired_count
- Max: 10 tasks
- Target: 70% CPU/Memory utilization

## Costs

Estimated monthly costs (us-west-2):

| Service | Configuration | Monthly Cost |
|---------|--------------|--------------|
| ECS Fargate (Backend) | 2 tasks × 1 vCPU, 2GB | ~$60 |
| ECS Fargate (Frontend) | 2 tasks × 0.5 vCPU, 1GB | ~$30 |
| ALB | 1 ALB | ~$20 |
| EFS | 10GB storage | ~$3 |
| NAT Gateway | 2 NAT gateways | ~$70 |
| CloudWatch Logs | 10GB/month | ~$5 |
| **Total** | | **~$188/month** |

To reduce costs:
- Use 1 NAT gateway instead of 2 (reduces HA)
- Reduce task counts
- Use FARGATE_SPOT for non-critical workloads

## Security

- All traffic between services uses private subnets
- EFS encryption at rest enabled
- Secrets stored in AWS Secrets Manager
- Security groups restrict traffic to necessary ports
- Container images scanned on push to ECR

## Troubleshooting

### Tasks not starting

Check ECS events:
```bash
aws ecs describe-services \
  --cluster bitoguard-prod-cluster \
  --services bitoguard-prod-backend \
  --query 'services[0].events[0:5]'
```

### Health check failures

View task logs:
```bash
aws logs tail /ecs/bitoguard-prod-backend --follow
```

### EFS mount issues

Verify EFS mount targets are in the same subnets as ECS tasks:
```bash
aws efs describe-mount-targets \
  --file-system-id $(terraform output -raw efs_file_system_id)
```

## Cleanup

To destroy all resources:

```bash
cd infra/aws/terraform
terraform destroy
```

**Warning**: This will delete all data including the DuckDB database on EFS.

## Advanced Configuration

### Custom Domain

To use a custom domain:

1. Add domain to terraform.tfvars:
   ```hcl
   domain_name = "bitoguard.example.com"
   ```

2. Create ACM certificate for the domain

3. Update ALB listener to use HTTPS (modify alb.tf)

### VPN Access

To restrict access to VPN only, modify security group rules in `security_groups.tf`:

```hcl
ingress {
  from_port   = 443
  to_port     = 443
  protocol    = "tcp"
  cidr_blocks = ["YOUR_VPN_CIDR"]
}
```

### Multi-Region

For multi-region deployment:
1. Copy terraform directory
2. Update region in terraform.tfvars
3. Deploy separately
4. Use Route53 for DNS failover
