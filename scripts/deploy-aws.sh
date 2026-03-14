#!/bin/bash
set -e

# BitoGuard AWS Deployment Script
# This script builds and pushes Docker images to ECR, then deploys to ECS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
AWS_REGION="${AWS_REGION:-us-west-2}"
ENVIRONMENT="${ENVIRONMENT:-prod}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo -e "${GREEN}=== BitoGuard AWS Deployment ===${NC}"
echo "Region: $AWS_REGION"
echo "Environment: $ENVIRONMENT"
echo "Image Tag: $IMAGE_TAG"
echo ""

# Check prerequisites
command -v aws >/dev/null 2>&1 || { echo -e "${RED}Error: AWS CLI is required but not installed.${NC}" >&2; exit 1; }
command -v docker >/dev/null 2>&1 || { echo -e "${RED}Error: Docker is required but not installed.${NC}" >&2; exit 1; }

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "AWS Account ID: $AWS_ACCOUNT_ID"

# ECR repository URLs
BACKEND_ECR_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/bitoguard-${ENVIRONMENT}-backend"
FRONTEND_ECR_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/bitoguard-${ENVIRONMENT}-frontend"

echo -e "\n${YELLOW}Step 1: Authenticating with ECR...${NC}"
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo -e "\n${YELLOW}Step 2: Building backend image...${NC}"
cd "$PROJECT_ROOT"
docker build -f bitoguard_core/Dockerfile -t bitoguard-backend:${IMAGE_TAG} .

echo -e "\n${YELLOW}Step 3: Building frontend image...${NC}"
docker build -f bitoguard_frontend/Dockerfile -t bitoguard-frontend:${IMAGE_TAG} .

echo -e "\n${YELLOW}Step 4: Tagging images...${NC}"
docker tag bitoguard-backend:${IMAGE_TAG} ${BACKEND_ECR_REPO}:${IMAGE_TAG}
docker tag bitoguard-frontend:${IMAGE_TAG} ${FRONTEND_ECR_REPO}:${IMAGE_TAG}

echo -e "\n${YELLOW}Step 5: Pushing backend image to ECR...${NC}"
docker push ${BACKEND_ECR_REPO}:${IMAGE_TAG}

echo -e "\n${YELLOW}Step 6: Pushing frontend image to ECR...${NC}"
docker push ${FRONTEND_ECR_REPO}:${IMAGE_TAG}

echo -e "\n${YELLOW}Step 7: Updating ECS services...${NC}"
CLUSTER_NAME="bitoguard-${ENVIRONMENT}-cluster"
BACKEND_SERVICE="bitoguard-${ENVIRONMENT}-backend"
FRONTEND_SERVICE="bitoguard-${ENVIRONMENT}-frontend"

# Force new deployment
aws ecs update-service \
  --cluster "$CLUSTER_NAME" \
  --service "$BACKEND_SERVICE" \
  --force-new-deployment \
  --region "$AWS_REGION" \
  > /dev/null

aws ecs update-service \
  --cluster "$CLUSTER_NAME" \
  --service "$FRONTEND_SERVICE" \
  --force-new-deployment \
  --region "$AWS_REGION" \
  > /dev/null

echo -e "\n${GREEN}=== Deployment Complete ===${NC}"
echo -e "Backend image: ${BACKEND_ECR_REPO}:${IMAGE_TAG}"
echo -e "Frontend image: ${FRONTEND_ECR_REPO}:${IMAGE_TAG}"
echo -e "\nTo check deployment status:"
echo -e "  aws ecs describe-services --cluster $CLUSTER_NAME --services $BACKEND_SERVICE $FRONTEND_SERVICE --region $AWS_REGION"
echo -e "\nTo view logs:"
echo -e "  aws logs tail /ecs/bitoguard-${ENVIRONMENT}-backend --follow --region $AWS_REGION"
echo -e "  aws logs tail /ecs/bitoguard-${ENVIRONMENT}-frontend --follow --region $AWS_REGION"
