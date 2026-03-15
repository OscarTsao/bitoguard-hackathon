# AWS Amplify Frontend (Next.js SSR)
# Ref: https://docs.aws.amazon.com/amplify/latest/userguide/server-side-rendering-amplify.html
# Ref: https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/amplify_app

resource "aws_amplify_app" "frontend" {
  name         = "${local.name_prefix}-frontend"
  repository   = var.github_repo_url
  access_token = var.github_access_token

  # WEB_COMPUTE required for Next.js SSR and API routes
  platform = "WEB_COMPUTE"

  build_spec = <<-EOT
    version: 1
    frontend:
      phases:
        preBuild:
          commands:
            - cd bitoguard_frontend
            - npm ci
        build:
          commands:
            - cd bitoguard_frontend
            - npm run build
      artifacts:
        baseDirectory: bitoguard_frontend/.next
        files:
          - '**/*'
      cache:
        paths:
          - bitoguard_frontend/node_modules/**/*
  EOT

  # Redirect all routes to Next.js for SPA/SSR handling
  custom_rule {
    source = "/<*>"
    status = "404-200"
    target = "/index.html"
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-frontend"
  })
}

resource "aws_amplify_branch" "main" {
  app_id      = aws_amplify_app.frontend.id
  branch_name = "main"
  framework   = "Next.js - SSR"
  stage       = "PRODUCTION"

  environment_variables = {
    # Backend API base — proxied through Next.js API routes
    # Set to the internal ALB DNS (not public) since Amplify SSR runs server-side
    BITOGUARD_INTERNAL_API_BASE = "http://${aws_lb.main.dns_name}"
    NODE_ENV                    = "production"
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-frontend-main"
  })
}

output "amplify_app_url" {
  description = "Amplify app default domain"
  value       = "https://main.${aws_amplify_app.frontend.default_domain}"
}

output "amplify_app_id" {
  description = "Amplify app ID (for manual deploys)"
  value       = aws_amplify_app.frontend.id
}
