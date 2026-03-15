# ECS Task Definitions for ML Pipeline

# Data Sync Task Definition
resource "aws_ecs_task_definition" "ml_sync" {
  family                   = "${local.name_prefix}-sync-task"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "1024"  # 1 vCPU
  memory                   = "2048"  # 2GB
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ml_pipeline_task.arn

  container_definitions = jsonencode([
    {
      name      = "sync"
      image     = "${aws_ecr_repository.backend.repository_url}:latest"
      essential = true

      command = [
        "python", "-m", "pipeline.sync"
      ]

      environment = [
        {
          name  = "PYTHONPATH"
          value = "."
        },
        {
          name  = "BITOGUARD_SOURCE_URL"
          value = var.bitopro_api_url
        },
        {
          name  = "AWS_REGION"
          value = var.aws_region
        },
        {
          name  = "BITOGUARD_ML_ARTIFACTS_BUCKET"
          value = aws_s3_bucket.artifacts.id
        },
        {
          name  = "BITOGUARD_DB_PATH"
          value = "/mnt/efs/artifacts/bitoguard.duckdb"
        },
        {
          name  = "BITOGUARD_ARTIFACT_DIR"
          value = "/mnt/efs/artifacts"
        }
      ]

      secrets = [
        {
          name      = "BITOGUARD_API_KEY"
          valueFrom = aws_secretsmanager_secret.bitoguard_api_key.arn
        }
      ]

      mountPoints = [
        {
          sourceVolume  = "efs-storage"
          containerPath = "/mnt/efs/artifacts"
          readOnly      = false
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ml_pipeline.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "sync"
        }
      }
    }
  ])

  volume {
    name = "efs-storage"

    efs_volume_configuration {
      file_system_id     = aws_efs_file_system.bitoguard.id
      transit_encryption = "ENABLED"
      authorization_config {
        access_point_id = aws_efs_access_point.artifacts.id
      }
    }
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-sync-task"
  })
}

# Feature Engineering Task Definition
resource "aws_ecs_task_definition" "ml_features" {
  family                   = "${local.name_prefix}-features-task"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "2048"  # 2 vCPU
  memory                   = "4096"  # 4GB
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ml_pipeline_task.arn

  container_definitions = jsonencode([
    {
      name      = "features"
      image     = "${aws_ecr_repository.backend.repository_url}:latest"
      essential = true

      command = [
        "python", "-m", "features.build_features_v2"
      ]

      environment = [
        {
          name  = "PYTHONPATH"
          value = "."
        },
        {
          name  = "AWS_REGION"
          value = var.aws_region
        },
        {
          name  = "BITOGUARD_ML_ARTIFACTS_BUCKET"
          value = aws_s3_bucket.artifacts.id
        },
        {
          name  = "EXPORT_TO_S3"
          value = "true"
        },
        {
          name  = "BITOGUARD_DB_PATH"
          value = "/mnt/efs/artifacts/bitoguard.duckdb"
        },
        {
          name  = "BITOGUARD_ARTIFACT_DIR"
          value = "/mnt/efs/artifacts"
        }
      ]

      mountPoints = [
        {
          sourceVolume  = "efs-storage"
          containerPath = "/mnt/efs/artifacts"
          readOnly      = false
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ml_pipeline.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "features"
        }
      }
    }
  ])

  volume {
    name = "efs-storage"

    efs_volume_configuration {
      file_system_id     = aws_efs_file_system.bitoguard.id
      transit_encryption = "ENABLED"
      authorization_config {
        access_point_id = aws_efs_access_point.artifacts.id
      }
    }
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-features-task"
  })
}

# Scoring Task Definition
resource "aws_ecs_task_definition" "ml_scoring" {
  family                   = "${local.name_prefix}-scoring-task"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "2048"  # 2 vCPU
  memory                   = "4096"  # 4GB
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ml_pipeline_task.arn

  container_definitions = jsonencode([
    {
      name      = "scoring"
      image     = "${aws_ecr_repository.backend.repository_url}:latest"
      essential = true

      command = [
        "python", "-m", "models.score"
      ]

      environment = [
        {
          name  = "PYTHONPATH"
          value = "."
        },
        {
          name  = "AWS_REGION"
          value = var.aws_region
        },
        {
          name  = "BITOGUARD_ML_ARTIFACTS_BUCKET"
          value = aws_s3_bucket.artifacts.id
        },
        {
          name  = "BITOGUARD_DB_PATH"
          value = "/mnt/efs/artifacts/bitoguard.duckdb"
        },
        {
          name  = "BITOGUARD_ARTIFACT_DIR"
          value = "/mnt/efs/artifacts"
        }
      ]

      mountPoints = [
        {
          sourceVolume  = "efs-storage"
          containerPath = "/mnt/efs/artifacts"
          readOnly      = false
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ml_pipeline.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "scoring"
        }
      }
    }
  ])

  volume {
    name = "efs-storage"

    efs_volume_configuration {
      file_system_id     = aws_efs_file_system.bitoguard.id
      transit_encryption = "ENABLED"
      authorization_config {
        access_point_id = aws_efs_access_point.artifacts.id
      }
    }
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-scoring-task"
  })
}

# Outputs
output "ml_sync_task_definition_arn" {
  description = "ARN of the ML sync task definition"
  value       = aws_ecs_task_definition.ml_sync.arn
}

output "ml_features_task_definition_arn" {
  description = "ARN of the ML features task definition"
  value       = aws_ecs_task_definition.ml_features.arn
}

output "ml_scoring_task_definition_arn" {
  description = "ARN of the ML scoring task definition"
  value       = aws_ecs_task_definition.ml_scoring.arn
}

# EFS Bootstrap: Copy-Seed Task Definition
# Copies bitoguard.duckdb from S3 to EFS on first deploy; skips if already present.
resource "aws_ecs_task_definition" "copy_seed" {
  family                   = "${local.name_prefix}-copy-seed"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 256
  memory                   = 512
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.backend_task.arn

  container_definitions = jsonencode([{
    name      = "copy-seed"
    image     = "${aws_ecr_repository.backend.repository_url}:latest"
    essential = true
    command = [
      "sh", "-c",
      "if [ ! -f /mnt/efs/bitoguard.duckdb ]; then aws s3 cp s3://${aws_s3_bucket.artifacts.bucket}/seed/bitoguard.duckdb /mnt/efs/bitoguard.duckdb && echo 'Seed complete'; else echo 'EFS already seeded, skipping'; fi"
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.copy_seed.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }
    mountPoints = [{
      sourceVolume  = "efs-storage"
      containerPath = "/mnt/efs"
      readOnly      = false
    }]
  }])

  volume {
    name = "efs-storage"

    efs_volume_configuration {
      file_system_id     = aws_efs_file_system.bitoguard.id
      transit_encryption = "ENABLED"
      authorization_config {
        access_point_id = aws_efs_access_point.artifacts.id
      }
    }
  }

  tags = merge(local.common_tags, { Name = "${local.name_prefix}-copy-seed" })
}

resource "aws_cloudwatch_log_group" "copy_seed" {
  name              = "/ecs/${local.name_prefix}/copy-seed"
  retention_in_days = 7
  tags              = local.common_tags
}

output "copy_seed_task_definition_arn" {
  description = "ARN of the copy-seed EFS bootstrap task definition"
  value       = aws_ecs_task_definition.copy_seed.arn
}
