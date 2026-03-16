#!/bin/bash
# BitoGuard — Deploy Next.js frontend to AWS ECS via ECR
#
# Builds a production Docker image using Next.js standalone output,
# pushes to ECR, then forces a new ECS deployment.
#
# Prerequisites:
#   - Valid AWS credentials (WSParticipantRole or equivalent)
#   - Docker running locally
#   - ECS cluster and service already provisioned (Terraform applied)
#
# Usage: ./scripts/deploy-frontend.sh [--api-base <url>]

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[ OK ]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERR ]${NC} $*"; exit 1; }

# ── Parse args ─────────────────────────────────────────────────────────────────
AWS_REGION="${AWS_REGION:-us-west-2}"
API_BASE="${BITOGUARD_INTERNAL_API_BASE:-http://backend:8001}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --api-base) API_BASE="$2"; shift 2 ;;
    --region)   AWS_REGION="$2"; shift 2 ;;
    *) log_error "Unknown argument: $1" ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── Discover AWS resources ─────────────────────────────────────────────────────
log_info "Discovering AWS resources..."
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text) \
  || log_error "Cannot get caller identity — check AWS credentials"

ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
CLUSTER_NAME=$(aws ecs list-clusters --region "${AWS_REGION}" --query 'clusterArns[0]' --output text \
  | sed 's|.*/||') || log_error "Cannot list ECS clusters"
SERVICE_NAME=$(aws ecs list-services --cluster "${CLUSTER_NAME}" --region "${AWS_REGION}" \
  --query 'serviceArns[?contains(@, `frontend`)]|[0]' --output text | sed 's|.*/||') \
  || log_error "Cannot find frontend ECS service"

log_info "Account:  ${ACCOUNT_ID}"
log_info "Region:   ${AWS_REGION}"
log_info "Cluster:  ${CLUSTER_NAME}"
log_info "Service:  ${SERVICE_NAME}"

# ── Get ECR repo name ──────────────────────────────────────────────────────────
ECR_FRONTEND_REPO=$(aws ecr describe-repositories --region "${AWS_REGION}" \
  --query 'repositories[?contains(repositoryName, `frontend`)].repositoryUri' \
  --output text | head -1) || log_error "Cannot find frontend ECR repository"
log_info "ECR repo: ${ECR_FRONTEND_REPO}"

# ── ECR login ──────────────────────────────────────────────────────────────────
log_info "Authenticating with ECR..."
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${ECR_REPO}"
log_ok "ECR authentication successful"

# ── Build image ────────────────────────────────────────────────────────────────
IMAGE_TAG="$(date +%Y%m%d%H%M%S)"
IMAGE_URI="${ECR_FRONTEND_REPO}:${IMAGE_TAG}"
LATEST_URI="${ECR_FRONTEND_REPO}:latest"

log_info "Building frontend image (standalone, production)..."
log_info "  Tag:     ${IMAGE_TAG}"
log_info "  API URL: ${API_BASE}"

docker build \
  --platform linux/amd64 \
  --build-arg "BITOGUARD_INTERNAL_API_BASE=${API_BASE}" \
  --tag "${IMAGE_URI}" \
  --tag "${LATEST_URI}" \
  --file "${REPO_ROOT}/bitoguard_frontend/Dockerfile" \
  "${REPO_ROOT}"

log_ok "Image built: $(docker image inspect "${IMAGE_URI}" --format '{{.Size}}' | numfmt --to=iec)"

# ── Push to ECR ────────────────────────────────────────────────────────────────
log_info "Pushing to ECR..."
docker push "${IMAGE_URI}"
docker push "${LATEST_URI}"
log_ok "Pushed: ${IMAGE_URI}"

# ── Force ECS deployment ───────────────────────────────────────────────────────
log_info "Forcing new ECS deployment..."
aws ecs update-service \
  --cluster "${CLUSTER_NAME}" \
  --service "${SERVICE_NAME}" \
  --force-new-deployment \
  --region "${AWS_REGION}" \
  --query 'service.deployments[0].{status:status,desiredCount:desiredCount,runningCount:runningCount}' \
  --output table

# ── Wait for stability ─────────────────────────────────────────────────────────
log_info "Waiting for service to stabilize (up to 5 min)..."
aws ecs wait services-stable \
  --cluster "${CLUSTER_NAME}" \
  --services "${SERVICE_NAME}" \
  --region "${AWS_REGION}" && log_ok "Service stable" \
  || log_warn "Timed out waiting — check ECS console for details"

# ── Get ALB URL ────────────────────────────────────────────────────────────────
ALB_DNS=$(aws elbv2 describe-load-balancers --region "${AWS_REGION}" \
  --query 'LoadBalancers[?contains(LoadBalancerName, `bitoguard`)].DNSName | [0]' \
  --output text 2>/dev/null || echo "")

echo ""
echo -e "${GREEN}======================================================${NC}"
echo -e "${GREEN}  Frontend deployed successfully!${NC}"
echo -e "${GREEN}======================================================${NC}"
echo ""
if [[ -n "${ALB_DNS}" && "${ALB_DNS}" != "None" ]]; then
  echo -e "  URL: ${BLUE}http://${ALB_DNS}${NC}"
else
  echo "  Check ECS console for the ALB URL"
fi
echo "  Image: ${IMAGE_URI}"
echo ""
