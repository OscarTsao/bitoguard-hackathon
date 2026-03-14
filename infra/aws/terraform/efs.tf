resource "aws_efs_file_system" "bitoguard" {
  creation_token = "${local.name_prefix}-efs"
  encrypted      = true

  lifecycle_policy {
    transition_to_ia = "AFTER_30_DAYS"
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-efs"
  })
}

resource "aws_efs_mount_target" "bitoguard" {
  count           = 2
  file_system_id  = aws_efs_file_system.bitoguard.id
  subnet_id       = aws_subnet.private[count.index].id
  security_groups = [aws_security_group.efs.id]
}

resource "aws_efs_access_point" "artifacts" {
  file_system_id = aws_efs_file_system.bitoguard.id

  posix_user {
    gid = 1000
    uid = 1000
  }

  root_directory {
    path = "/artifacts"
    creation_info {
      owner_gid   = 1000
      owner_uid   = 1000
      permissions = "755"
    }
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-artifacts-ap"
  })
}
