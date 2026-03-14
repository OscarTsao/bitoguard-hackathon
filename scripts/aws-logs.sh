#!/bin/bash

# BitoGuard AWS Logs Viewer

AWS_REGION="${AWS_REGION:-us-west-2}"
ENVIRONMENT="${ENVIRONMENT:-prod}"

SERVICE="${1:-backend}"
FOLLOW="${2:-false}"

LOG_GROUP="/ecs/bitoguard-${ENVIRONMENT}-${SERVICE}"

if [ "$FOLLOW" = "follow" ] || [ "$FOLLOW" = "-f" ]; then
  echo "Following logs for ${SERVICE}..."
  aws logs tail "$LOG_GROUP" --follow --region "$AWS_REGION"
else
  echo "Showing last 100 lines for ${SERVICE}..."
  aws logs tail "$LOG_GROUP" --region "$AWS_REGION"
fi
