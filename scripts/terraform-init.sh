#!/bin/bash
set -e

# BitoGuard Terraform Initialization Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERRAFORM_DIR="$SCRIPT_DIR/../infra/aws/terraform"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== BitoGuard Terraform Initialization ===${NC}"

# Check prerequisites
command -v terraform >/dev/null 2>&1 || { echo "Error: Terraform is required but not installed." >&2; exit 1; }
command -v aws >/dev/null 2>&1 || { echo "Error: AWS CLI is required but not installed." >&2; exit 1; }

cd "$TERRAFORM_DIR"

# Copy example tfvars if not exists
if [ ! -f terraform.tfvars ]; then
  echo -e "${YELLOW}Creating terraform.tfvars from example...${NC}"
  cp terraform.tfvars.example terraform.tfvars
  echo "Please edit terraform.tfvars with your configuration before running terraform apply"
fi

echo -e "\n${YELLOW}Initializing Terraform...${NC}"
terraform init

echo -e "\n${YELLOW}Validating Terraform configuration...${NC}"
terraform validate

echo -e "\n${YELLOW}Formatting Terraform files...${NC}"
terraform fmt -recursive

echo -e "\n${GREEN}=== Initialization Complete ===${NC}"
echo -e "\nNext steps:"
echo -e "  1. Edit terraform.tfvars with your configuration"
echo -e "  2. Run: terraform plan"
echo -e "  3. Run: terraform apply"
