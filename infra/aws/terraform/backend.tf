# Terraform Backend Configuration
# Uncomment and configure for remote state storage

# terraform {
#   backend "s3" {
#     bucket         = "bitoguard-terraform-state"
#     key            = "prod/terraform.tfstate"
#     region         = "us-west-2"
#     encrypt        = true
#     dynamodb_table = "bitoguard-terraform-locks"
#   }
# }

# To set up remote state:
# 1. Create S3 bucket:
#    aws s3 mb s3://bitoguard-terraform-state --region us-west-2
#    aws s3api put-bucket-versioning --bucket bitoguard-terraform-state --versioning-configuration Status=Enabled
#    aws s3api put-bucket-encryption --bucket bitoguard-terraform-state --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
#
# 2. Create DynamoDB table:
#    aws dynamodb create-table \
#      --table-name bitoguard-terraform-locks \
#      --attribute-definitions AttributeName=LockID,AttributeType=S \
#      --key-schema AttributeName=LockID,KeyType=HASH \
#      --billing-mode PAY_PER_REQUEST \
#      --region us-west-2
#
# 3. Uncomment the backend block above
# 4. Run: terraform init -migrate-state
