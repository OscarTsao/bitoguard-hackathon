resource "random_password" "api_key" {
  length  = 32
  special = true
}

resource "aws_secretsmanager_secret" "bitoguard_api_key" {
  name                    = "${local.name_prefix}-api-key"
  description             = "BitoGuard API key for internal authentication"
  recovery_window_in_days = 7

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-api-key"
  })
}

resource "aws_secretsmanager_secret_version" "bitoguard_api_key" {
  secret_id     = aws_secretsmanager_secret.bitoguard_api_key.id
  secret_string = random_password.api_key.result
}
