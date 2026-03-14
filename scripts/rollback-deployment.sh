#!/bin/bash
set -e

# BitoGuard Rollback Script

AWS_REGION="${AWS_REGION:-us-west-2}"
ENVIRONMENT="${ENVIRONMENT:-prod}"
CLUSTER_NAME="bitoguard-${ENVIRONMENT}-cluster"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SERVICE="${1}"
REVISION="${2}"

if [ -z "$SERVICE" ] || [ -z "$REVISION" ]; then
  echo "Usage: $0 <backend|frontend> <revision_number>"
  echo ""
  echo "Example: $0 backend 5"
  echo ""
  echo "Available backend revisions:"
  aws ecs list-task-definitions \
    --family-prefix "bitoguard-${ENVIRONMENT}-backend" \
    --sort DESC \
    --max-items 10 \
    --region "$AWS_REGION" \
    --query 'taskDefinitionArns' \
    --output table
  echo ""
  echo "Available frontend revisions:"
  aws ecs list-task-definitions \
    --family-prefix "bitoguard-${ENVIRONMENT}-frontend" \
    --sort DESC \
    --max-items 10 \
    --region "$AWS_REGION" \
    --query 'taskDefinitionArns' \
    --output table
  exit 1
fi

SERVICE_NAME="bitoguard-${ENVIRONMENT}-${SERVICE}"
TASK_DEFINITION="bitoguard-${ENVIRONMENT}-${SERVICE}:${REVISION}"

echo -e "${YELLOW}Rolling back ${SERVICE} to revision ${REVISION}...${NC}"

aws ecs update-service \
  --cluster "$CLUSTER_NAME" \
  --service "$SERVICE_NAME" \
  --task-definition "$TASK_DEFINITION" \
  --region "$AWS_REGION" \
  > /dev/null

echo -e "${GREEN}Rollback initiated${NC}"
echo "Monitor deployment status:"
echo "  aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $AWS_REGION"
