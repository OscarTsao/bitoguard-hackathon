terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

locals {
  name_prefix = "bitoguard-${var.environment}"
  common_tags = {
    Project     = "BitoGuard"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}
