#!/usr/bin/env bash
# bootstrap-efs.sh — Seed EFS with local DuckDB on first deploy.
# Run once after `terraform apply` completes.
# Ref: https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task_definitions.html
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TF_DIR="$PROJECT_ROOT/infra/aws/terraform"
DUCKDB_PATH="$PROJECT_ROOT/bitoguard_core/artifacts/bitoguard.duckdb"

if [ ! -f "$DUCKDB_PATH" ]; then
  echo "ERROR: $DUCKDB_PATH not found. Run 'make sync && make features' locally first."
  exit 1
fi

# Get values from Terraform outputs
BUCKET=$(terraform -chdir="$TF_DIR" output -raw artifacts_bucket_name)
CLUSTER=$(terraform -chdir="$TF_DIR" output -raw ecs_cluster_name)
TASK_DEF=$(terraform -chdir="$TF_DIR" output -raw copy_seed_task_definition_arn)
REGION=$(terraform -chdir="$TF_DIR" output -raw aws_region 2>/dev/null || echo "${AWS_REGION:-us-east-1}")
# Convert JSON array ["subnet-xxx","subnet-yyy"] to comma-separated string for awsvpc config
SUBNETS=$(terraform -chdir="$TF_DIR" output -json private_subnet_ids | tr -d '[]" ' | tr ',' ',')
SG=$(terraform -chdir="$TF_DIR" output -raw ecs_security_group_id)

echo "=== BitoGuard EFS Bootstrap ==="
echo "Uploading DuckDB to s3://$BUCKET/seed/bitoguard.duckdb ..."
aws s3 cp "$DUCKDB_PATH" "s3://$BUCKET/seed/bitoguard.duckdb" --region "$REGION"
echo "Upload complete."

echo "Running copy-seed ECS task on cluster: $CLUSTER ..."
TASK_ARN=$(aws ecs run-task \
  --cluster "$CLUSTER" \
  --task-definition "$TASK_DEF" \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SG],assignPublicIp=DISABLED}" \
  --region "$REGION" \
  --query 'tasks[0].taskArn' \
  --output text)

echo "Task started: $TASK_ARN"
echo "Waiting for task to complete (this may take 1-2 minutes)..."
aws ecs wait tasks-stopped --cluster "$CLUSTER" --tasks "$TASK_ARN" --region "$REGION"

EXIT_CODE=$(aws ecs describe-tasks \
  --cluster "$CLUSTER" \
  --tasks "$TASK_ARN" \
  --region "$REGION" \
  --query 'tasks[0].containers[0].exitCode' \
  --output text)

if [ "$EXIT_CODE" = "0" ]; then
  echo "EFS bootstrap complete. DuckDB is ready on EFS."
else
  echo "ERROR: copy-seed task failed with exit code $EXIT_CODE"
  echo "Check CloudWatch logs: /ecs/$(terraform -chdir="$TF_DIR" output -raw ecs_cluster_name | sed 's/-cluster$//')-prod/copy-seed"
  echo "Or browse: https://console.aws.amazon.com/cloudwatch/home?region=$REGION#logsV2:log-groups"
  exit 1
fi
