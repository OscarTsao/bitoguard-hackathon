#!/bin/bash
set -e

# BitoGuard Deployment Health Check Script

AWS_REGION="${AWS_REGION:-us-west-2}"
ENVIRONMENT="${ENVIRONMENT:-prod}"
CLUSTER_NAME="bitoguard-${ENVIRONMENT}-cluster"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== BitoGuard Deployment Health Check ===${NC}\n"

# Check ECS Services
echo -e "${YELLOW}Checking ECS Services...${NC}"
aws ecs describe-services \
  --cluster "$CLUSTER_NAME" \
  --services "bitoguard-${ENVIRONMENT}-backend" "bitoguard-${ENVIRONMENT}-frontend" \
  --region "$AWS_REGION" \
  --query 'services[*].[serviceName,status,runningCount,desiredCount,deployments[0].rolloutState]' \
  --output table

# Check Target Health
echo -e "\n${YELLOW}Checking Target Group Health...${NC}"
BACKEND_TG_ARN=$(aws elbv2 describe-target-groups \
  --names "bitoguard-${ENVIRONMENT}-backend-tg" \
  --region "$AWS_REGION" \
  --query 'TargetGroups[0].TargetGroupArn' \
  --output text)

FRONTEND_TG_ARN=$(aws elbv2 describe-target-groups \
  --names "bitoguard-${ENVIRONMENT}-frontend-tg" \
  --region "$AWS_REGION" \
  --query 'TargetGroups[0].TargetGroupArn' \
  --output text)

echo "Backend targets:"
aws elbv2 describe-target-health \
  --target-group-arn "$BACKEND_TG_ARN" \
  --region "$AWS_REGION" \
  --query 'TargetHealthDescriptions[*].[Target.Id,TargetHealth.State,TargetHealth.Reason]' \
  --output table

echo -e "\nFrontend targets:"
aws elbv2 describe-target-health \
  --target-group-arn "$FRONTEND_TG_ARN" \
  --region "$AWS_REGION" \
  --query 'TargetHealthDescriptions[*].[Target.Id,TargetHealth.State,TargetHealth.Reason]' \
  --output table

# Check ALB
echo -e "\n${YELLOW}Checking Load Balancer...${NC}"
ALB_DNS=$(aws elbv2 describe-load-balancers \
  --names "bitoguard-${ENVIRONMENT}-alb" \
  --region "$AWS_REGION" \
  --query 'LoadBalancers[0].DNSName' \
  --output text)

echo "ALB DNS: $ALB_DNS"
echo "Testing backend health endpoint..."
if curl -sf "http://${ALB_DNS}/healthz" > /dev/null; then
  echo -e "${GREEN}✓ Backend health check passed${NC}"
else
  echo -e "${RED}✗ Backend health check failed${NC}"
fi

echo -e "\nTesting frontend..."
if curl -sf "http://${ALB_DNS}/" > /dev/null; then
  echo -e "${GREEN}✓ Frontend accessible${NC}"
else
  echo -e "${RED}✗ Frontend not accessible${NC}"
fi

# Check CloudWatch Alarms
echo -e "\n${YELLOW}Checking CloudWatch Alarms...${NC}"
aws cloudwatch describe-alarms \
  --alarm-name-prefix "bitoguard-${ENVIRONMENT}" \
  --region "$AWS_REGION" \
  --query 'MetricAlarms[*].[AlarmName,StateValue]' \
  --output table

# Recent Logs
echo -e "\n${YELLOW}Recent Backend Logs (last 5 minutes)...${NC}"
aws logs tail "/ecs/bitoguard-${ENVIRONMENT}-backend" \
  --since 5m \
  --region "$AWS_REGION" \
  --format short | tail -20

echo -e "\n${GREEN}=== Health Check Complete ===${NC}"
echo -e "\nApplication URL: http://${ALB_DNS}"
