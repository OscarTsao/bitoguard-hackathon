#!/bin/bash
set -e

AWS_REGION="${AWS_REGION:-us-west-2}"
ENVIRONMENT="${ENVIRONMENT:-prod}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== Setting up CloudWatch Dashboard ===${NC}"

DASHBOARD_BODY=$(cat <<EOF
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/ECS", "CPUUtilization", {"stat": "Average"}],
          [".", "MemoryUtilization", {"stat": "Average"}]
        ],
        "period": 300,
        "stat": "Average",
        "region": "${AWS_REGION}",
        "title": "ECS Resource Utilization"
      }
    },
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/ApplicationELB", "TargetResponseTime", {"stat": "Average"}],
          [".", "RequestCount", {"stat": "Sum"}]
        ],
        "period": 300,
        "stat": "Average",
        "region": "${AWS_REGION}",
        "title": "ALB Metrics"
      }
    }
  ]
}
EOF
)

aws cloudwatch put-dashboard \
  --dashboard-name "BitoGuard-${ENVIRONMENT}" \
  --dashboard-body "$DASHBOARD_BODY" \
  --region "$AWS_REGION"

echo -e "${GREEN}Dashboard created: BitoGuard-${ENVIRONMENT}${NC}"
echo "View at: https://console.aws.amazon.com/cloudwatch/home?region=${AWS_REGION}#dashboards:name=BitoGuard-${ENVIRONMENT}"
