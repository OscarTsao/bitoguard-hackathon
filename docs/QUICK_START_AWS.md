# BitoGuard AWS Quick Start

Get BitoGuard running on AWS in under 30 minutes.

## Prerequisites (5 minutes)

```bash
# 1. Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip && sudo ./aws/install

# 2. Install Terraform
wget https://releases.hashicorp.com/terraform/1.7.0/terraform_1.7.0_linux_amd64.zip
unzip terraform_1.7.0_linux_amd64.zip && sudo mv terraform /usr/local/bin/

# 3. Configure AWS credentials
aws configure
# Enter: Access Key, Secret Key, Region (us-west-2), Output (json)

# 4. Verify
aws sts get-caller-identity
terraform version
```

## Deploy Infrastructure (10 minutes)

```bash
# 1. Navigate to terraform directory
cd infra/aws/terraform

# 2. Initialize
terraform init

# 3. Create config
cp terraform.tfvars.example terraform.tfvars

# 4. Deploy (type 'yes' when prompted)
terraform apply
```

## Deploy Application (10 minutes)

```bash
# 1. Return to project root
cd ../../..

# 2. Run deployment script
./scripts/deploy-aws.sh
```

## Access Application (2 minutes)

```bash
# Get URL
cd infra/aws/terraform
terraform output alb_url

# Open in browser or test
curl $(terraform output -raw alb_url)/healthz
```

## Initialize Data (5 minutes)

```bash
# Get backend task
TASK_ID=$(aws ecs list-tasks \
  --cluster bitoguard-prod-cluster \
  --service-name bitoguard-prod-backend \
  --query 'taskArns[0]' --output text | cut -d'/' -f3)

# Run sync
aws ecs execute-command \
  --cluster bitoguard-prod-cluster \
  --task $TASK_ID \
  --container backend \
  --interactive \
  --command "python -m pipeline.sync --full"
```

## Done!

Your BitoGuard instance is now running on AWS.

**Next steps:**
- View logs: `aws logs tail /ecs/bitoguard-prod-backend --follow`
- Monitor: Check CloudWatch dashboard
- Scale: Edit `terraform.tfvars` and run `terraform apply`
- Update: Run `./scripts/deploy-aws.sh`

**Costs:** ~$198/month (see docs/COST_OPTIMIZATION.md to reduce)

**Cleanup:** `terraform destroy` (from infra/aws/terraform/)
