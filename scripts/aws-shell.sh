#!/bin/bash
set -e

# BitoGuard AWS Shell Access Script
# Provides interactive shell access to running ECS tasks

AWS_REGION="${AWS_REGION:-us-west-2}"
ENVIRONMENT="${ENVIRONMENT:-prod}"
SERVICE="${1:-backend}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [ "$SERVICE" != "backend" ] && [ "$SERVICE" != "frontend" ]; then
  echo -e "${RED}Error: Service must be 'backend' or 'frontend'${NC}"
  echo "Usage: $0 <backend|frontend>"
  exit 1
fi

CLUSTER_NAME="bitoguard-${ENVIRONMENT}-cluster"
SERVICE_NAME="bitoguard-${ENVIRONMENT}-${SERVICE}"

echo -e "${YELLOW}Getting task ID for ${SERVICE}...${NC}"

TASK_ARN=$(aws ecs list-tasks \
  --cluster "$CLUSTER_NAME" \
  --service-name "$SERVICE_NAME" \
  --desired-status RUNNING \
  --region "$AWS_REGION" \
  --query 'taskArns[0]' \
  --output text)

if [ -z "$TASK_ARN" ] || [ "$TASK_ARN" = "None" ]; then
  echo -e "${RED}Error: No running tasks found for ${SERVICE}${NC}"
  exit 1
fi

TASK_ID=$(echo "$TASK_ARN" | cut -d'/' -f3)

echo -e "${GREEN}Connecting to task: ${TASK_ID}${NC}"
echo -e "${YELLOW}Note: ECS Exec must be enabled on the service${NC}"
echo ""

aws ecs execute-command \
  --cluster "$CLUSTER_NAME" \
  --task "$TASK_ID" \
  --container "$SERVICE" \
  --interactive \
  --command "/bin/bash" \
  --region "$AWS_REGION"
