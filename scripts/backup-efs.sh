#!/bin/bash
set -e

# BitoGuard EFS Backup Script

AWS_REGION="${AWS_REGION:-us-west-2}"
ENVIRONMENT="${ENVIRONMENT:-prod}"
BACKUP_BUCKET="${BACKUP_BUCKET:-bitoguard-backups-${ENVIRONMENT}}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== BitoGuard EFS Backup ===${NC}"

# Get EFS ID
cd ../infra/aws/terraform
EFS_ID=$(terraform output -raw efs_file_system_id)
cd -

echo "EFS ID: $EFS_ID"
echo "Backup Bucket: $BACKUP_BUCKET"

# Create backup bucket if not exists
if ! aws s3 ls "s3://${BACKUP_BUCKET}" 2>/dev/null; then
  echo -e "${YELLOW}Creating backup bucket...${NC}"
  aws s3 mb "s3://${BACKUP_BUCKET}" --region "$AWS_REGION"
  aws s3api put-bucket-versioning \
    --bucket "$BACKUP_BUCKET" \
    --versioning-configuration Status=Enabled
fi

# Create backup using AWS Backup
BACKUP_VAULT="bitoguard-${ENVIRONMENT}-vault"

# Create vault if not exists
if ! aws backup describe-backup-vault --backup-vault-name "$BACKUP_VAULT" 2>/dev/null; then
  echo -e "${YELLOW}Creating backup vault...${NC}"
  aws backup create-backup-vault \
    --backup-vault-name "$BACKUP_VAULT" \
    --region "$AWS_REGION"
fi

# Start backup job
echo -e "${YELLOW}Starting backup job...${NC}"
BACKUP_JOB_ID=$(aws backup start-backup-job \
  --backup-vault-name "$BACKUP_VAULT" \
  --resource-arn "arn:aws:elasticfilesystem:${AWS_REGION}:$(aws sts get-caller-identity --query Account --output text):file-system/${EFS_ID}" \
  --iam-role-arn "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/service-role/AWSBackupDefaultServiceRole" \
  --region "$AWS_REGION" \
  --query 'BackupJobId' \
  --output text)

echo -e "${GREEN}Backup job started: ${BACKUP_JOB_ID}${NC}"
echo "Monitor progress:"
echo "  aws backup describe-backup-job --backup-job-id $BACKUP_JOB_ID --region $AWS_REGION"
