# BitoGuard Terraform Infrastructure

This directory contains Terraform configuration for deploying BitoGuard on AWS.

## Quick Start

```bash
# Initialize
terraform init

# Configure
cp terraform.tfvars.example terraform.tfvars
nano terraform.tfvars

# Plan
terraform plan

# Deploy
terraform apply
```

## Files

- `main.tf` - Provider configuration
- `variables.tf` - Input variables
- `outputs.tf` - Output values
- `vpc.tf` - VPC, subnets, NAT gateways
- `security_groups.tf` - Security group rules
- `alb.tf` - Application Load Balancer
- `ecs.tf` - ECS cluster and services
- `ecr.tf` - Container registries
- `efs.tf` - Elastic File System
- `iam.tf` - IAM roles and policies
- `cloudwatch.tf` - Logging and monitoring
- `autoscaling.tf` - Auto-scaling policies
- `secrets.tf` - Secrets Manager
- `backend.tf` - Remote state (optional)

## Configuration

Edit `terraform.tfvars`:

```hcl
aws_region              = "us-west-2"
environment             = "prod"
backend_cpu             = 1024
backend_memory          = 2048
backend_desired_count   = 2
frontend_cpu            = 512
frontend_memory         = 1024
frontend_desired_count  = 2
```

## Outputs

After deployment:

```bash
# Get ALB URL
terraform output alb_url

# Get ECR repository URLs
terraform output backend_ecr_repository_url
terraform output frontend_ecr_repository_url

# Get all outputs
terraform output
```

## Remote State (Optional)

To use S3 backend for state:

1. Create S3 bucket and DynamoDB table (see backend.tf)
2. Uncomment backend block in backend.tf
3. Run: `terraform init -migrate-state`

## Maintenance

```bash
# Update infrastructure
terraform plan
terraform apply

# Destroy everything
terraform destroy
```
