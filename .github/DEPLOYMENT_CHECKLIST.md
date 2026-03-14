# BitoGuard AWS Deployment Checklist

Use this checklist to ensure a successful AWS deployment.

## Pre-Deployment

### Prerequisites
- [ ] AWS account with appropriate permissions
- [ ] AWS CLI installed and configured (`aws configure`)
- [ ] Terraform >= 1.0 installed
- [ ] Docker installed
- [ ] Git repository access

### AWS Permissions Required
- [ ] VPC, Subnets, Route Tables, Internet Gateway, NAT Gateway
- [ ] ECS, ECR
- [ ] EFS
- [ ] ALB, Target Groups
- [ ] IAM Roles and Policies
- [ ] CloudWatch Logs and Alarms
- [ ] Secrets Manager
- [ ] Auto Scaling

## Infrastructure Deployment

### Step 1: Initialize Terraform
- [ ] Navigate to `infra/aws/terraform/`
- [ ] Run `terraform init`
- [ ] Copy `terraform.tfvars.example` to `terraform.tfvars`
- [ ] Edit `terraform.tfvars` with your configuration
- [ ] Review variables (region, environment, resource sizes)

### Step 2: Plan Infrastructure
- [ ] Run `terraform plan`
- [ ] Review planned resources (~50 resources)
- [ ] Verify no unexpected changes
- [ ] Check estimated costs

### Step 3: Deploy Infrastructure
- [ ] Run `terraform apply`
- [ ] Type 'yes' to confirm
- [ ] Wait for completion (~10 minutes)
- [ ] Verify all resources created successfully
- [ ] Save outputs (ALB URL, ECR URLs, etc.)

## Application Deployment

### Step 4: Build and Push Images
- [ ] Return to project root
- [ ] Make scripts executable: `chmod +x scripts/*.sh`
- [ ] Run `./scripts/deploy-aws.sh`
- [ ] Verify backend image pushed to ECR
- [ ] Verify frontend image pushed to ECR
- [ ] Wait for ECS services to stabilize (~5 minutes)

### Step 5: Verify Deployment
- [ ] Run `./scripts/check-deployment.sh`
- [ ] Check ECS service status (running count = desired count)
- [ ] Check target health (all healthy)
- [ ] Test backend health: `curl http://<ALB_DNS>/healthz`
- [ ] Test frontend: Open `http://<ALB_DNS>` in browser

## Post-Deployment

### Step 6: Initialize Data
- [ ] Get backend task ID
- [ ] Run data sync: `python -m pipeline.sync --full`
- [ ] Build features: `python -m features.build_features`
- [ ] Train models: `python -m models.train`
- [ ] Generate scores: `python -m models.score`

### Step 7: Set Up Monitoring
- [ ] Run `./scripts/setup-monitoring.sh`
- [ ] Open CloudWatch dashboard
- [ ] Verify metrics are flowing
- [ ] Test alarms (optional)
- [ ] Set up SNS notifications (optional)

### Step 8: Configure CI/CD (Optional)
- [ ] Add GitHub secrets:
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
- [ ] Test workflow by pushing to main branch
- [ ] Verify automatic deployment works

### Step 9: Security Hardening
- [ ] Review security group rules
- [ ] Rotate API keys if needed
- [ ] Enable VPC Flow Logs (optional)
- [ ] Set up WAF rules (optional)
- [ ] Configure backup schedule
- [ ] Document access procedures

### Step 10: Cost Optimization
- [ ] Review actual resource usage
- [ ] Implement cost optimizations from COST_OPTIMIZATION.md
- [ ] Set up AWS Budget alerts
- [ ] Schedule scaling for off-hours (optional)

## Validation

### Functional Tests
- [ ] Backend API responds to health checks
- [ ] Frontend loads successfully
- [ ] Backend can access EFS
- [ ] Database queries work
- [ ] Model scoring works
- [ ] Alerts are generated

### Performance Tests
- [ ] Load test ALB
- [ ] Verify auto-scaling triggers
- [ ] Check response times
- [ ] Monitor resource utilization

### Security Tests
- [ ] Verify private subnets have no public IPs
- [ ] Test security group rules
- [ ] Verify EFS encryption
- [ ] Check IAM permissions
- [ ] Review CloudWatch logs

## Documentation

- [ ] Document ALB URL
- [ ] Document ECR repository URLs
- [ ] Document EFS file system ID
- [ ] Document API key location (Secrets Manager)
- [ ] Update team wiki/docs
- [ ] Share access procedures

## Rollback Plan

- [ ] Document current task definition revisions
- [ ] Test rollback procedure: `./scripts/rollback-deployment.sh`
- [ ] Document EFS backup procedure
- [ ] Create disaster recovery plan

## Ongoing Operations

### Daily
- [ ] Check CloudWatch alarms
- [ ] Review error logs
- [ ] Monitor costs

### Weekly
- [ ] Review CloudWatch metrics
- [ ] Check auto-scaling events
- [ ] Review and optimize costs
- [ ] Update documentation

### Monthly
- [ ] Review and rotate credentials
- [ ] Update dependencies
- [ ] Review and optimize resources
- [ ] Test backup/restore procedures

## Troubleshooting

If issues occur, check:
- [ ] CloudWatch logs: `aws logs tail /ecs/bitoguard-prod-backend --follow`
- [ ] ECS service events
- [ ] Target health status
- [ ] Security group rules
- [ ] IAM permissions
- [ ] EFS mount targets

## Cleanup (When Needed)

- [ ] Backup EFS data: `./scripts/backup-efs.sh`
- [ ] Export CloudWatch logs
- [ ] Document lessons learned
- [ ] Run `terraform destroy`
- [ ] Verify all resources deleted
- [ ] Check for orphaned resources

## Sign-Off

- [ ] Infrastructure deployed successfully
- [ ] Application running and accessible
- [ ] Monitoring configured
- [ ] Documentation updated
- [ ] Team trained on operations
- [ ] Rollback plan tested

**Deployed by:** _______________
**Date:** _______________
**Environment:** _______________
**ALB URL:** _______________

## Resources

- Quick Start: [docs/QUICK_START_AWS.md](../docs/QUICK_START_AWS.md)
- Full Guide: [docs/AWS_DEPLOYMENT_GUIDE.md](../docs/AWS_DEPLOYMENT_GUIDE.md)
- Architecture: [infra/aws/ARCHITECTURE.md](../infra/aws/ARCHITECTURE.md)
- Cost Tips: [docs/COST_OPTIMIZATION.md](../docs/COST_OPTIMIZATION.md)
