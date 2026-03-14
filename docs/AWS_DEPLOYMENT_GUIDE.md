# BitoGuard AWS Deployment Guide

Complete guide for deploying BitoGuard on AWS using managed services and Terraform.

## Architecture Overview

```
Internet
    ↓
Application Load Balancer (ALB)
    ↓
┌─────────────────────────────────────────┐
│  VPC (10.0.0.0/16)                      │
│                                         │
│  ┌──────────────┐  ┌──────────────┐   │
│  │ Public Subnet│  │ Public Subnet│   │
│  │   AZ-1       │  │   AZ-2       │   │
│  │ NAT Gateway  │  │ NAT Gateway  │   │
│  └──────────────┘  └──────────────┘   │
│                                         │
│  ┌──────────────┐  ┌──────────────┐   │
│  │Private Subnet│  │Private Subnet│   │
│  │   AZ-1       │  │   AZ-2       │   │
│  │              │  │              │   │
│  │ ECS Fargate  │  │ ECS Fargate  │   │
│  │ - Backend    │  │ - Backend    │   │
│  │ - Frontend   │  │ - Frontend   │   │
│  └──────────────┘  └──────────────┘   │
│         ↓                  ↓            │
│  ┌─────────────────────────────────┐  │
│  │  EFS (Shared Storage)           │  │
│  │  - DuckDB Database              │  │
│  │  - Model Artifacts              │  │
│  └─────────────────────────────────┘  │
└─────────────────────────────────────────┘
         ↓
   CloudWatch Logs & Metrics
   Secrets Manager
```

## AWS Services Used

| Service | Purpose | Cost Impact |
|---------|---------|-------------|
| ECS Fargate | Serverless container orchestration | High |
| ECR | Container image registry | Low |
| ALB | Load balancing & routing | Medium |
| EFS | Persistent storage for DuckDB | Low |
| VPC | Network isolation | Free |
| NAT Gateway | Outbound internet for private subnets | High |
| CloudWatch | Logging & monitoring | Low |
| Secrets Manager | API key storage | Low |
| Auto Scaling | Automatic capacity management | Free |

## Prerequisites

### 1. Install Required Tools

```bash
# AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Terraform
wget https://releases.hashicorp.com/terraform/1.7.0/terraform_1.7.0_linux_amd64.zip
unzip terraform_1.7.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/

# Docker (if not installed)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

### 2. Configure AWS Credentials

```bash
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Default region: us-west-2
# Default output format: json
```

### 3. Verify Permissions

Your AWS user/role needs permissions for:
- VPC, Subnets, Route Tables, Internet Gateway, NAT Gateway
- ECS, ECR
- EFS
- ALB, Target Groups
- IAM Roles and Policies
- CloudWatch Logs and Alarms
- Secrets Manager
- Auto Scaling

## Deployment Steps

### Step 1: Initialize Terraform

```bash
cd infra/aws/terraform

# Initialize Terraform
terraform init

# Copy example configuration
cp terraform.tfvars.example terraform.tfvars

# Edit configuration
nano terraform.tfvars
```

**Key configurations in terraform.tfvars:**

```hcl
aws_region              = "us-west-2"
environment             = "prod"
vpc_cidr                = "10.0.0.0/16"

# Backend configuration
backend_cpu             = 1024    # 1 vCPU
backend_memory          = 2048    # 2 GB
backend_desired_count   = 2       # Number of tasks

# Frontend configuration
frontend_cpu            = 512     # 0.5 vCPU
frontend_memory         = 1024    # 1 GB
frontend_desired_count  = 2       # Number of tasks

# Application settings
bitoguard_source_url    = "https://aws-event-api.bitopro.com"
```

### Step 2: Plan Infrastructure

```bash
# Review what will be created
terraform plan

# Should show ~50 resources to be created
```

### Step 3: Deploy Infrastructure

```bash
# Deploy all resources
terraform apply

# Type 'yes' when prompted
# This takes ~5-10 minutes
```

**What gets created:**
- VPC with 2 public and 2 private subnets across 2 AZs
- Internet Gateway and 2 NAT Gateways
- Application Load Balancer
- ECS Cluster
- 2 ECR repositories (backend, frontend)
- EFS file system with mount targets
- Security groups
- IAM roles and policies
- CloudWatch log groups
- Secrets Manager secret with auto-generated API key
- Auto-scaling policies

### Step 4: Build and Deploy Application

```bash
# From project root
chmod +x scripts/deploy-aws.sh
./scripts/deploy-aws.sh
```

This script:
1. Authenticates with ECR
2. Builds backend Docker image
3. Builds frontend Docker image
4. Pushes images to ECR
5. Forces ECS service redeployment

**Build time:** ~5-10 minutes

### Step 5: Verify Deployment

```bash
# Get ALB URL
terraform output alb_url

# Check ECS service status
aws ecs describe-services \
  --cluster bitoguard-prod-cluster \
  --services bitoguard-prod-backend bitoguard-prod-frontend \
  --query 'services[*].[serviceName,status,runningCount,desiredCount]' \
  --output table

# View backend logs
aws logs tail /ecs/bitoguard-prod-backend --follow

# View frontend logs
aws logs tail /ecs/bitoguard-prod-frontend --follow
```

### Step 6: Access Application

```bash
# Get the URL
ALB_URL=$(terraform output -raw alb_url)
echo "Application URL: $ALB_URL"

# Test backend health
curl $ALB_URL/healthz

# Open frontend in browser
open $ALB_URL
```

## Post-Deployment Configuration

### Initialize Database

```bash
# Get backend task ID
TASK_ID=$(aws ecs list-tasks \
  --cluster bitoguard-prod-cluster \
  --service-name bitoguard-prod-backend \
  --query 'taskArns[0]' \
  --output text | cut -d'/' -f3)

# Execute sync command
aws ecs execute-command \
  --cluster bitoguard-prod-cluster \
  --task $TASK_ID \
  --container backend \
  --interactive \
  --command "python -m pipeline.sync --full"
```

### Run Initial Training

```bash
# SSH into backend container
aws ecs execute-command \
  --cluster bitoguard-prod-cluster \
  --task $TASK_ID \
  --container backend \
  --interactive \
  --command "/bin/bash"

# Inside container:
cd /app/bitoguard_core
python -m features.build_features
python -m models.train
python -m models.anomaly
python -m models.score
```

## Monitoring & Operations

### CloudWatch Dashboards

Create a custom dashboard:

```bash
aws cloudwatch put-dashboard \
  --dashboard-name BitoGuard-Prod \
  --dashboard-body file://cloudwatch-dashboard.json
```

### View Logs

```bash
# Backend logs (last 1 hour)
aws logs tail /ecs/bitoguard-prod-backend --since 1h

# Frontend logs (follow mode)
aws logs tail /ecs/bitoguard-prod-frontend --follow

# Filter for errors
aws logs tail /ecs/bitoguard-prod-backend --filter-pattern "ERROR"
```

### Metrics

Key metrics to monitor:
- ECS CPU/Memory utilization
- ALB request count and latency
- ECS task count
- EFS throughput
- CloudWatch alarm states

```bash
# Get CPU utilization
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name CPUUtilization \
  --dimensions Name=ClusterName,Value=bitoguard-prod-cluster Name=ServiceName,Value=bitoguard-prod-backend \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average
```

### Alarms

Pre-configured alarms:
- Backend CPU > 80%
- Backend Memory > 80%

Add custom alarms in `cloudwatch.tf`.

## Scaling

### Manual Scaling

Update `terraform.tfvars`:

```hcl
backend_desired_count  = 4
frontend_desired_count = 4
```

Apply changes:

```bash
terraform apply
```

### Auto Scaling

Auto-scaling is pre-configured:
- **Trigger:** 70% CPU or Memory
- **Min tasks:** desired_count
- **Max tasks:** 10
- **Scale-out cooldown:** 60 seconds
- **Scale-in cooldown:** 300 seconds

### Vertical Scaling

Increase task resources in `terraform.tfvars`:

```hcl
backend_cpu     = 2048  # 2 vCPU
backend_memory  = 4096  # 4 GB
```

Apply and redeploy:

```bash
terraform apply
./scripts/deploy-aws.sh
```

## Updates & Maintenance

### Application Updates

```bash
# Build and deploy new version
./scripts/deploy-aws.sh

# Or with custom tag
IMAGE_TAG=v1.2.3 ./scripts/deploy-aws.sh
```

### Infrastructure Updates

```bash
# Edit terraform files
nano infra/aws/terraform/ecs.tf

# Plan changes
terraform plan

# Apply changes
terraform apply
```

### Database Backup

```bash
# Create EFS backup
aws backup start-backup-job \
  --backup-vault-name Default \
  --resource-arn $(terraform output -raw efs_file_system_id) \
  --iam-role-arn arn:aws:iam::ACCOUNT_ID:role/service-role/AWSBackupDefaultServiceRole
```

### Rollback

```bash
# List previous task definitions
aws ecs list-task-definitions \
  --family-prefix bitoguard-prod-backend \
  --sort DESC

# Update service to previous version
aws ecs update-service \
  --cluster bitoguard-prod-cluster \
  --service bitoguard-prod-backend \
  --task-definition bitoguard-prod-backend:PREVIOUS_REVISION
```

## Security Best Practices

### 1. Enable HTTPS

Add ACM certificate and update ALB listener:

```hcl
# In alb.tf
resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.main.arn
  port              = "443"
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS-1-2-2017-01"
  certificate_arn   = "arn:aws:acm:REGION:ACCOUNT:certificate/CERT_ID"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.frontend.arn
  }
}
```

### 2. Restrict Access

Update security group to allow only specific IPs:

```hcl
# In security_groups.tf
ingress {
  from_port   = 443
  to_port     = 443
  protocol    = "tcp"
  cidr_blocks = ["YOUR_OFFICE_IP/32"]
}
```

### 3. Enable VPC Flow Logs

```hcl
resource "aws_flow_log" "main" {
  vpc_id          = aws_vpc.main.id
  traffic_type    = "ALL"
  iam_role_arn    = aws_iam_role.flow_logs.arn
  log_destination = aws_cloudwatch_log_group.flow_logs.arn
}
```

### 4. Rotate API Keys

```bash
# Generate new key
aws secretsmanager update-secret \
  --secret-id bitoguard-prod-api-key \
  --secret-string "NEW_RANDOM_KEY"

# Restart services
aws ecs update-service \
  --cluster bitoguard-prod-cluster \
  --service bitoguard-prod-backend \
  --force-new-deployment
```

## Cost Optimization

### Current Estimated Costs (us-west-2)

| Resource | Monthly Cost |
|----------|--------------|
| ECS Fargate (Backend: 2×1vCPU,2GB) | $60 |
| ECS Fargate (Frontend: 2×0.5vCPU,1GB) | $30 |
| ALB | $20 |
| NAT Gateway (2) | $70 |
| EFS (10GB) | $3 |
| CloudWatch Logs (10GB) | $5 |
| **Total** | **~$188/month** |

### Optimization Strategies

**1. Use Single NAT Gateway** (saves $35/month)
- Reduces high availability
- Edit `vpc.tf` to create only 1 NAT gateway

**2. Use FARGATE_SPOT** (saves ~30%)
```hcl
capacity_provider_strategy {
  capacity_provider = "FARGATE_SPOT"
  weight            = 100
}
```

**3. Reduce Task Count**
```hcl
backend_desired_count  = 1
frontend_desired_count = 1
```

**4. Schedule Scaling**
- Scale down during off-hours
- Use AWS Lambda + EventBridge

**5. Use S3 for Artifacts**
- Store model artifacts in S3 instead of EFS
- Reduces EFS costs for large files

## Troubleshooting

### Tasks Failing to Start

**Check task stopped reason:**
```bash
aws ecs describe-tasks \
  --cluster bitoguard-prod-cluster \
  --tasks TASK_ID \
  --query 'tasks[0].stoppedReason'
```

**Common issues:**
- Image pull errors → Check ECR permissions
- Health check failures → Check application logs
- Resource limits → Increase CPU/memory

### Health Check Failures

```bash
# Check target health
aws elbv2 describe-target-health \
  --target-group-arn $(terraform output -raw backend_target_group_arn)

# Test health endpoint from within VPC
aws ecs execute-command \
  --cluster bitoguard-prod-cluster \
  --task TASK_ID \
  --container backend \
  --command "curl localhost:8001/healthz"
```

### EFS Mount Issues

```bash
# Verify mount targets
aws efs describe-mount-targets \
  --file-system-id $(terraform output -raw efs_file_system_id)

# Check security group rules
aws ec2 describe-security-groups \
  --group-ids $(terraform output -raw efs_security_group_id)
```

### High Costs

```bash
# Check NAT Gateway data transfer
aws cloudwatch get-metric-statistics \
  --namespace AWS/NATGateway \
  --metric-name BytesOutToDestination \
  --start-time $(date -u -d '7 days ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 86400 \
  --statistics Sum

# Use AWS Cost Explorer for detailed breakdown
```

## Cleanup

### Destroy All Resources

```bash
cd infra/aws/terraform
terraform destroy
```

**Warning:** This permanently deletes:
- All ECS tasks and services
- EFS file system (including DuckDB database)
- All logs
- Load balancer
- VPC and networking

### Backup Before Destroy

```bash
# Backup EFS data
aws datasync create-task \
  --source-location-arn EFS_LOCATION \
  --destination-location-arn S3_LOCATION

# Export logs
aws logs create-export-task \
  --log-group-name /ecs/bitoguard-prod-backend \
  --from $(date -d '30 days ago' +%s)000 \
  --to $(date +%s)000 \
  --destination s3-bucket-name
```

## Advanced Topics

### Multi-Region Deployment

1. Copy terraform directory for each region
2. Deploy separately
3. Use Route53 for DNS failover

### CI/CD Integration

Add to GitHub Actions:

```yaml
name: Deploy to AWS
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-west-2
      - name: Deploy
        run: ./scripts/deploy-aws.sh
```

### Custom Domain Setup

1. Register domain in Route53
2. Create ACM certificate
3. Add HTTPS listener to ALB
4. Create Route53 alias record

## Support

For issues:
1. Check CloudWatch logs
2. Review ECS service events
3. Verify security group rules
4. Check IAM permissions
5. Review Terraform state

## Next Steps

After successful deployment:
1. Set up monitoring dashboards
2. Configure backup schedule
3. Enable HTTPS with custom domain
4. Set up CI/CD pipeline
5. Configure alerting (SNS, PagerDuty, etc.)
6. Review and optimize costs
7. Implement disaster recovery plan
